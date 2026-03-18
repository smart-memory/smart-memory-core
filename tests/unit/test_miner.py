"""Unit tests for Wikidata miner with mocked SPARQL."""

import json
from unittest.mock import MagicMock

from smartmemory.grounding.miner import (
    EXPANDED_DOMAINS,
    MiningCheckpoint,
    WikidataMiner,
    load_domain_config,
)
from smartmemory.grounding.models import PublicEntity


class TestMiningCheckpoint:
    def test_save_and_load(self, tmp_path):
        cp = MiningCheckpoint(
            completed_domains=["programming_languages"],
            total_queries=5,
            total_entities=100,
        )
        cp_path = tmp_path / "checkpoint.json"
        cp.save(cp_path)

        loaded = MiningCheckpoint.load(cp_path)
        assert loaded.completed_domains == ["programming_languages"]
        assert loaded.total_queries == 5
        assert loaded.total_entities == 100

    def test_load_missing(self, tmp_path):
        cp = MiningCheckpoint.load(tmp_path / "nope.json")
        assert cp.completed_domains == []
        assert cp.total_queries == 0


class TestWikidataMiner:
    def _mock_client(self, entities: list[PublicEntity] | None = None):
        client = MagicMock()
        client.query_domain.return_value = entities or [
            PublicEntity(qid="Q123", label="Python", entity_type="Language", instance_of=["Q9143"]),
            PublicEntity(qid="Q456", label="Java", entity_type="Language", instance_of=["Q9143"]),
        ]
        client._rate_limit = MagicMock()
        client._parse_results = MagicMock(return_value=[])
        client.ENDPOINT = "https://query.wikidata.org/sparql"
        return client

    def test_mine_single_domain(self, tmp_path):
        client = self._mock_client()
        miner = WikidataMiner(client=client, limit_per_domain=100)
        result = miner.mine_domains(
            domains={"Q9143": "programming_languages"},
            output_dir=tmp_path / "output",
            output_format="snapshot",
        )
        assert len(result.entities) == 2
        assert result.domain_counts["programming_languages"] == 2
        assert (tmp_path / "output" / "mined_entities.jsonl").exists()

    def test_mine_corpus_output(self, tmp_path):
        client = self._mock_client()
        miner = WikidataMiner(client=client)
        result = miner.mine_domains(
            domains={"Q9143": "programming_languages"},
            output_dir=tmp_path / "output",
            output_format="corpus",
        )
        corpus_path = tmp_path / "output" / "mined_corpus.jsonl"
        assert corpus_path.exists()
        lines = corpus_path.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 entities

    def test_quota_limit(self, tmp_path):
        client = self._mock_client()
        miner = WikidataMiner(client=client, quota_limit=1)
        result = miner.mine_domains(
            domains={"Q9143": "lang", "Q271680": "frameworks"},
            output_dir=tmp_path / "output",
            output_format="snapshot",
        )
        # Should stop after first domain due to quota
        assert result.total_queries == 1
        assert result.quota_exhausted

    def test_deduplicates_by_qid(self, tmp_path):
        entities = [
            PublicEntity(qid="Q123", label="Python", entity_type="Language"),
            PublicEntity(qid="Q123", label="Python", entity_type="Language"),
        ]
        client = self._mock_client(entities)
        miner = WikidataMiner(client=client)
        result = miner.mine_domains(
            domains={"Q9143": "lang"},
            output_dir=tmp_path / "output",
            output_format="snapshot",
        )
        assert len(result.entities) == 1

    def test_checkpoint_resume(self, tmp_path):
        cp_path = tmp_path / "checkpoint.json"
        # Pre-save a checkpoint marking first domain as done
        cp = MiningCheckpoint(completed_domains=["lang"], total_queries=1)
        cp.save(cp_path)

        client = self._mock_client()
        miner = WikidataMiner(client=client)
        result = miner.mine_domains(
            domains={"Q9143": "lang", "Q271680": "frameworks"},
            output_dir=tmp_path / "output",
            output_format="snapshot",
            checkpoint_path=cp_path,
        )
        # Should skip "lang" and only mine "frameworks"
        assert "lang" not in result.domain_counts
        assert "frameworks" in result.domain_counts

    def test_handles_query_failure(self, tmp_path):
        client = self._mock_client()
        client.query_domain.side_effect = Exception("SPARQL timeout")
        miner = WikidataMiner(client=client)
        result = miner.mine_domains(
            domains={"Q9143": "lang"},
            output_dir=tmp_path / "output",
            output_format="snapshot",
        )
        assert result.domain_counts["lang"] == 0
        assert len(result.entities) == 0


class TestExpandedDomains:
    def test_expanded_domains_superset_of_original(self):
        original = {"Q9143", "Q271680", "Q188860", "Q9135", "Q4830453", "Q3918"}
        assert original.issubset(set(EXPANDED_DOMAINS.keys()))

    def test_expanded_domains_has_new_entries(self):
        assert len(EXPANDED_DOMAINS) > 6


class TestLoadDomainConfig:
    def test_load_config(self, tmp_path):
        config = {
            "aws_services": {"qid": "Q456", "limit": 2000, "type": "Technology"},
            "k8s_concepts": {"qid": "Q789", "limit": 1000, "type": "Technology"},
        }
        config_path = tmp_path / "domains.json"
        config_path.write_text(json.dumps(config))

        domains = load_domain_config(config_path)
        assert domains == {"Q456": "aws_services", "Q789": "k8s_concepts"}
