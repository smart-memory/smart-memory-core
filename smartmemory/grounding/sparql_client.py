"""Wikidata SPARQL (WDQS) client with circuit breaker and negative caching.

Guardrails:
- 2s timeout per request
- 1 req/s rate limit
- Circuit breaker: CLOSED → OPEN (5 failures) → HALF_OPEN (60s) → CLOSED
- Negative cache: in-memory dict with TTL (or Redis if available)
"""

from __future__ import annotations

import enum
import hashlib
import logging
import time
from smartmemory.grounding.models import PublicEntity
from smartmemory.grounding.type_map import WIKIDATA_TYPE_MAP

logger = logging.getLogger(__name__)


class CircuitState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Three-state circuit breaker for external service calls."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._failure_count = 0
        self._state = CircuitState.CLOSED
        self._last_failure_time = 0.0

    @property
    def state(self) -> CircuitState:
        return self._state

    def allow_request(self) -> bool:
        if self._state == CircuitState.CLOSED:
            return True
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self._recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                return True
            return False
        # HALF_OPEN — allow one probe request
        return True

    def record_success(self) -> None:
        self._failure_count = 0
        self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
        elif self._failure_count >= self._failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                "Circuit breaker OPEN after %d failures", self._failure_count
            )


class WDQSClient:
    """Wikidata Query Service (SPARQL) client with safety guardrails."""

    ENDPOINT = "https://query.wikidata.org/sparql"
    TIMEOUT = 2.0
    RATE_LIMIT = 1.0  # minimum seconds between requests

    def __init__(self, redis_client=None):
        self._circuit = CircuitBreaker()
        self._redis = redis_client
        self._last_request_time = 0.0
        # In-memory negative cache: surface_form_hash → expiry_time
        self._negative_cache: dict[str, float] = {}
        self._negative_ttl = 86400.0  # 24 hours

    def query_entity(
        self, surface_form: str, type_filter_qids: list[str] | None = None
    ) -> list[PublicEntity]:
        """Single-entity SPARQL lookup with circuit breaker and negative cache.

        Returns matching PublicEntity instances, or empty list on miss/error.
        """
        if self._is_negative_cached(surface_form):
            return []

        if not self._circuit.allow_request():
            logger.debug("Circuit breaker OPEN, skipping SPARQL for '%s'", surface_form)
            return []

        self._rate_limit()

        try:
            import requests

            sparql = self._build_entity_query(surface_form, type_filter_qids)
            response = requests.get(
                self.ENDPOINT,
                params={"query": sparql, "format": "json"},
                timeout=self.TIMEOUT,
                headers={"User-Agent": "SmartMemory/1.0 (grounding)"},
            )
            response.raise_for_status()
            data = response.json()
            results = self._parse_results(data)

            if results:
                self._circuit.record_success()
            else:
                self._set_negative_cache(surface_form)
                self._circuit.record_success()

            return results

        except Exception as e:
            logger.warning("SPARQL query failed for '%s': %s", surface_form, e)
            self._circuit.record_failure()
            return []

    def query_domain(
        self, domain_qid: str, limit: int = 5000
    ) -> list[PublicEntity]:
        """Batch domain query for snapshot pipeline (no circuit breaker).

        This is a build-time operation, not runtime. No circuit breaker
        or negative cache — we want all results or a clear failure.
        """
        try:
            import requests

            sparql = self._build_domain_query(domain_qid, limit)
            response = requests.get(
                self.ENDPOINT,
                params={"query": sparql, "format": "json"},
                timeout=30.0,  # longer timeout for batch
                headers={"User-Agent": "SmartMemory/1.0 (snapshot-build)"},
            )
            response.raise_for_status()
            data = response.json()
            return self._parse_results(data)
        except Exception as e:
            logger.error("SPARQL domain query failed for '%s': %s", domain_qid, e)
            raise

    def _is_negative_cached(self, surface_form: str) -> bool:
        """Check if a surface form is in the negative cache."""
        key = self._cache_key(surface_form)
        if self._redis:
            return bool(self._redis.exists(f"smartmemory:sparql:miss:{key}"))
        expiry = self._negative_cache.get(key)
        if expiry is None:
            return False
        if time.monotonic() > expiry:
            del self._negative_cache[key]
            return False
        return True

    def _set_negative_cache(self, surface_form: str) -> None:
        """Mark a surface form as a SPARQL miss."""
        key = self._cache_key(surface_form)
        if self._redis:
            self._redis.setex(f"smartmemory:sparql:miss:{key}", 86400, "1")
        else:
            self._negative_cache[key] = time.monotonic() + self._negative_ttl

    def _cache_key(self, surface_form: str) -> str:
        return hashlib.sha256(surface_form.lower().encode()).hexdigest()[:16]

    def _rate_limit(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.RATE_LIMIT:
            time.sleep(self.RATE_LIMIT - elapsed)
        self._last_request_time = time.monotonic()

    def _build_entity_query(
        self, surface_form: str, type_filter_qids: list[str] | None = None
    ) -> str:
        type_filter = ""
        if type_filter_qids:
            values = " ".join(f"wd:{qid}" for qid in type_filter_qids)
            type_filter = f"VALUES ?type {{ {values} }} ?item wdt:P31 ?type ."

        return f"""
        SELECT ?item ?itemLabel ?itemDescription ?instanceOf WHERE {{
            ?item rdfs:label "{surface_form}"@en .
            {type_filter}
            OPTIONAL {{ ?item wdt:P31 ?instanceOf . }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
        }}
        LIMIT 10
        """

    def _build_domain_query(self, domain_qid: str, limit: int) -> str:
        return f"""
        SELECT ?item ?itemLabel ?itemDescription ?instanceOf ?altLabel WHERE {{
            ?item wdt:P31/wdt:P279* wd:{domain_qid} .
            OPTIONAL {{ ?item wdt:P31 ?instanceOf . }}
            OPTIONAL {{ ?item skos:altLabel ?altLabel . FILTER(LANG(?altLabel) = "en") }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
        }}
        LIMIT {limit}
        """

    def _parse_results(self, data: dict) -> list[PublicEntity]:
        """Parse SPARQL JSON results into PublicEntity instances."""
        entities: dict[str, PublicEntity] = {}
        bindings = data.get("results", {}).get("bindings", [])

        for binding in bindings:
            item_uri = binding.get("item", {}).get("value", "")
            qid = item_uri.rsplit("/", 1)[-1] if "/" in item_uri else ""
            if not qid.startswith("Q"):
                continue

            if qid not in entities:
                label = binding.get("itemLabel", {}).get("value", "")
                description = binding.get("itemDescription", {}).get("value", "")
                instance_of_uri = binding.get("instanceOf", {}).get("value", "")
                instance_of_qid = (
                    instance_of_uri.rsplit("/", 1)[-1]
                    if "/" in instance_of_uri
                    else ""
                )
                entity_type = WIKIDATA_TYPE_MAP.get(instance_of_qid, "")

                entities[qid] = PublicEntity(
                    qid=qid,
                    label=label,
                    description=description,
                    entity_type=entity_type,
                    instance_of=[instance_of_qid] if instance_of_qid else [],
                )
            else:
                # Add additional instance_of
                instance_of_uri = binding.get("instanceOf", {}).get("value", "")
                instance_of_qid = (
                    instance_of_uri.rsplit("/", 1)[-1]
                    if "/" in instance_of_uri
                    else ""
                )
                if instance_of_qid and instance_of_qid not in entities[qid].instance_of:
                    entities[qid].instance_of.append(instance_of_qid)
                    if not entities[qid].entity_type:
                        entities[qid].entity_type = WIKIDATA_TYPE_MAP.get(
                            instance_of_qid, ""
                        )

            # Collect alt labels
            alt_label = binding.get("altLabel", {}).get("value", "")
            if alt_label and alt_label not in entities[qid].aliases:
                entities[qid].aliases.append(alt_label)

        return list(entities.values())
