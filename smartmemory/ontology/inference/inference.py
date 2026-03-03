"""
LLM Inference Service for Ontology Construction

LLM-native inference system that extracts concepts, relations, and taxonomy
from raw text chunks using structured prompts and lightweight NLP.
"""
import json
import logging
import re
from datetime import datetime, UTC
from typing import Dict, List, Tuple, Any

from smartmemory.configuration import MemoryConfig
from smartmemory.ontology.ir_models import (
    OntologyIR, Concept, Relation, TaxonomyRelation, Evidence, Status, Origin, Meta, InferenceRequest, InferenceResponse
)

logger = logging.getLogger(__name__)


class LLMInferenceService:
    """LLM-native ontology inference service"""

    def __init__(self, config: MemoryConfig = None, llm_client=None, embedding_client=None):
        if config is None:
            config = MemoryConfig()

        self.config = config
        self.llm_client = llm_client
        self.embedding_client = embedding_client

        # Default thresholds from plan
        self.tau_low = 0.55
        self.tau_review = 0.78
        self.tau_high = 0.78

        # Hearst patterns for is-a extraction
        self.hearst_patterns = [
            r"(.+?) such as (.+)",
            r"(.+?) including (.+)",
            r"(.+?) especially (.+)",
            r"(.+?) like (.+)",
            r"(.+?) and other (.+)",
            r"(.+?) is a (?:type of |kind of )?(.+)",
            r"(.+?) are (?:types of |kinds of )?(.+)"
        ]

    def run_inference(self, request: InferenceRequest) -> InferenceResponse:
        """Run complete ontology inference pipeline"""
        logger.info(f"Starting inference for {len(request.raw_chunks)} chunks")

        # Extract concepts and relations from chunks
        concepts = []
        relations = []
        taxonomy = []

        for chunk in request.raw_chunks:
            chunk_concepts, chunk_relations, chunk_taxonomy = self._process_chunk(
                chunk, request.params
            )
            concepts.extend(chunk_concepts)
            relations.extend(chunk_relations)
            taxonomy.extend(chunk_taxonomy)

        # Canonicalize and merge similar concepts/relations
        concepts = self._canonicalize_concepts(concepts)
        relations = self._canonicalize_relations(relations)
        taxonomy = self._deduplicate_taxonomy(taxonomy)

        # Score and filter by confidence thresholds
        concepts = self._score_and_filter_concepts(concepts, request.params)
        relations = self._score_and_filter_relations(relations, request.params)
        taxonomy = self._score_and_filter_taxonomy(taxonomy, request.params)

        # Build changeset IR
        changeset = OntologyIR(
            registry_id=request.registry_id,
            concepts=concepts,
            relations=relations,
            taxonomy=taxonomy,
            built_at=datetime.now(UTC)
        )

        metrics = {
            "concepts_count": len(concepts),
            "relations_count": len(relations),
            "taxonomy_count": len(taxonomy)
        }

        return InferenceResponse(
            changeset=changeset,
            metrics=metrics,
            status="completed"
        )

    def _process_chunk(self, chunk: Dict[str, str], params: Dict[str, Any]) -> Tuple[List[Concept], List[Relation], List[TaxonomyRelation]]:
        """Process single text chunk for concepts and relations"""
        doc_id = chunk["doc_id"]
        text = chunk["text"]

        # P1: Extract concepts and synonyms
        concepts = self._extract_concepts(doc_id, text, params)

        # P3: Extract relations  
        relations = self._extract_relations(doc_id, text, params)

        # Hearst patterns for taxonomy
        taxonomy = []
        if params.get("use_hearst_patterns", True):
            taxonomy = self._extract_hearst_taxonomy(doc_id, text)

        return concepts, relations, taxonomy

    def _extract_concepts(self, doc_id: str, text: str, params: Dict[str, Any]) -> List[Concept]:
        """Extract concepts using LLM P1 prompt"""
        if not self.llm_client:
            return self._fallback_concept_extraction(doc_id, text)

        prompt = self._build_concept_prompt(text)

        try:
            # Use synchronous LLM call
            response = self.llm_client.complete(prompt, temperature=0.1)
            concepts_data = json.loads(response)

            concepts = []
            for item in concepts_data.get("concepts", []):
                concept = Concept(
                    id=self._generate_concept_id(item["label"]),
                    label=item["label"].lower().strip(),
                    synonyms=item.get("synonyms", []),
                    status=Status.PROPOSED,
                    origin=Origin.AI,
                    confidence=item.get("confidence", 0.5),
                    evidence=[Evidence(
                        doc=doc_id,
                        quote=item.get("quote", ""),
                        span=item.get("span")
                    )],
                    meta=Meta(
                        created_by="ai",
                        created_at=datetime.now(UTC)
                    )
                )
                concepts.append(concept)

            return concepts

        except Exception as e:
            logger.error(f"LLM concept extraction failed: {e}")
            return self._fallback_concept_extraction(doc_id, text)

    def _extract_relations(self, doc_id: str, text: str, params: Dict[str, Any]) -> List[Relation]:
        """Extract relations using LLM P3 prompt"""
        if not self.llm_client:
            return self._fallback_relation_extraction(doc_id, text)

        prompt = self._build_relation_prompt(text)

        try:
            # Use synchronous LLM call
            response = self.llm_client.complete(prompt, temperature=0.1)
            relations_data = json.loads(response)

            relations = []
            for item in relations_data.get("relations", []):
                relation = Relation(
                    id=self._generate_relation_id(item["label"]),
                    label=item["label"].lower().strip(),
                    aliases=item.get("aliases", []),
                    domain=self._generate_concept_id(item.get("domain", "")),
                    range=self._generate_concept_id(item.get("range", "")),
                    status=Status.PROPOSED,
                    confidence=item.get("confidence", 0.5),
                    evidence=[Evidence(
                        doc=doc_id,
                        quote=item.get("quote", ""),
                        span=item.get("span")
                    )]
                )
                relations.append(relation)

            return relations

        except Exception as e:
            logger.error(f"LLM relation extraction failed: {e}")
            return self._fallback_relation_extraction(doc_id, text)

    def _extract_hearst_taxonomy(self, doc_id: str, text: str) -> List[TaxonomyRelation]:
        """Extract is-a relations using Hearst patterns"""
        taxonomy = []

        for pattern in self.hearst_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                child_term = match.group(1).strip()
                parent_term = match.group(2).strip()

                # Clean up terms
                child_term = self._clean_term(child_term)
                parent_term = self._clean_term(parent_term)

                if child_term and parent_term:
                    taxonomy.append(TaxonomyRelation(
                        parent=self._generate_concept_id(parent_term),
                        child=self._generate_concept_id(child_term),
                        status=Status.PROPOSED,
                        origin=Origin.AI,
                        confidence=0.7,
                        evidence=[Evidence(
                            doc=doc_id,
                            quote=match.group(0),
                            span=[match.start(), match.end()]
                        )]
                    ))

        return taxonomy

    def _canonicalize_concepts(self, concepts: List[Concept]) -> List[Concept]:
        """Canonicalize and merge similar concepts"""
        if not concepts:
            return []

        # Group by normalized label
        concept_groups = {}
        for concept in concepts:
            normalized = self._normalize_label(concept.label)
            if normalized not in concept_groups:
                concept_groups[normalized] = []
            concept_groups[normalized].append(concept)

        # Merge concepts in each group
        merged_concepts = []
        for group in concept_groups.values():
            if len(group) == 1:
                merged_concepts.append(group[0])
            else:
                merged = self._merge_concepts(group)
                merged_concepts.append(merged)

        return merged_concepts

    def _canonicalize_relations(self, relations: List[Relation]) -> List[Relation]:
        """Canonicalize and merge similar relations"""
        if not relations:
            return []

        # Group by normalized label
        relation_groups = {}
        for relation in relations:
            normalized = self._normalize_label(relation.label)
            if normalized not in relation_groups:
                relation_groups[normalized] = []
            relation_groups[normalized].append(relation)

        # Merge relations in each group
        merged_relations = []
        for group in relation_groups.values():
            if len(group) == 1:
                merged_relations.append(group[0])
            else:
                merged = self._merge_relations(group)
                merged_relations.append(merged)

        return merged_relations

    def _score_and_filter_concepts(self, concepts: List[Concept], params: Dict[str, Any]) -> List[Concept]:
        """Score concepts and filter by confidence threshold"""
        threshold = params.get("confidence_threshold", self.tau_low)

        scored_concepts = []
        for concept in concepts:
            # Scoring formula: 0.4*LLM_cert + 0.3*freq + 0.3*cluster_cohesion
            llm_cert = concept.confidence
            freq = len(concept.evidence)  # Simple frequency proxy
            cluster_cohesion = len(concept.synonyms) * 0.1  # Simple cohesion proxy

            final_score = 0.4 * llm_cert + 0.3 * min(freq / 10, 1.0) + 0.3 * min(cluster_cohesion, 1.0)
            concept.confidence = final_score

            if final_score >= threshold:
                scored_concepts.append(concept)

        return scored_concepts

    def _score_and_filter_relations(self, relations: List[Relation], params: Dict[str, Any]) -> List[Relation]:
        """Score relations and filter by confidence threshold"""
        threshold = params.get("confidence_threshold", self.tau_low)

        scored_relations = []
        for relation in relations:
            # Scoring formula: 0.5*tuple_support + 0.3*dep_conf + 0.2*type_consistency
            tuple_support = len(relation.signatures) if relation.signatures else 1
            dep_conf = relation.confidence  # From LLM
            type_consistency = 1.0 if relation.domain and relation.range else 0.5

            final_score = 0.5 * min(tuple_support / 10, 1.0) + 0.3 * dep_conf + 0.2 * type_consistency
            relation.confidence = final_score

            if final_score >= threshold:
                scored_relations.append(relation)

        return scored_relations

    def _score_and_filter_taxonomy(self, taxonomy: List[TaxonomyRelation], params: Dict[str, Any]) -> List[TaxonomyRelation]:
        """Score taxonomy relations and filter by confidence threshold"""
        threshold = params.get("confidence_threshold", self.tau_low)
        return [t for t in taxonomy if t.confidence >= threshold]

    def _deduplicate_taxonomy(self, taxonomy: List[TaxonomyRelation]) -> List[TaxonomyRelation]:
        """Remove duplicate taxonomy relations"""
        seen = set()
        unique_taxonomy = []

        for tax in taxonomy:
            key = (tax.parent, tax.child)
            if key not in seen:
                seen.add(key)
                unique_taxonomy.append(tax)

        return unique_taxonomy

    def _merge_concepts(self, concepts: List[Concept]) -> Concept:
        """Merge multiple concepts into one canonical concept"""
        if not concepts:
            return None

        # Use the concept with highest confidence as base
        base = max(concepts, key=lambda c: c.confidence)

        # Merge synonyms and evidence
        all_synonyms = set(base.synonyms)
        all_evidence = list(base.evidence)

        for concept in concepts:
            if concept != base:
                all_synonyms.update(concept.synonyms)
                all_synonyms.add(concept.label)  # Add other labels as synonyms
                all_evidence.extend(concept.evidence)

        base.synonyms = list(all_synonyms)
        base.evidence = all_evidence
        base.confidence = sum(c.confidence for c in concepts) / len(concepts)

        return base

    def _merge_relations(self, relations: List[Relation]) -> Relation:
        """Merge multiple relations into one canonical relation"""
        if not relations:
            return None

        # Use the relation with highest confidence as base
        base = max(relations, key=lambda r: r.confidence)

        # Merge aliases and evidence
        all_aliases = set(base.aliases)
        all_evidence = list(base.evidence)

        for relation in relations:
            if relation != base:
                all_aliases.update(relation.aliases)
                all_aliases.add(relation.label)  # Add other labels as aliases
                all_evidence.extend(relation.evidence)

        base.aliases = list(all_aliases)
        base.evidence = all_evidence
        base.confidence = sum(r.confidence for r in relations) / len(relations)

        return base

    def _normalize_label(self, label: str) -> str:
        """Normalize label for comparison"""
        return re.sub(r'[^\w\s]', '', label.lower().strip())

    def _clean_term(self, term: str) -> str:
        """Clean extracted term"""
        # Remove articles, punctuation, extra whitespace
        term = re.sub(r'^(the|a|an)\s+', '', term.lower())
        term = re.sub(r'[^\w\s]', '', term)
        return term.strip()

    def _generate_concept_id(self, label: str) -> str:
        """Generate stable concept ID"""
        if not label:
            return ""
        normalized = self._normalize_label(label)
        slug = re.sub(r'\s+', '_', normalized)[:50]  # Max 50 chars
        return f"EX:{slug}"

    def _generate_relation_id(self, label: str) -> str:
        """Generate stable relation ID"""
        if not label:
            return ""
        normalized = self._normalize_label(label)
        slug = re.sub(r'\s+', '_', normalized)[:50]  # Max 50 chars
        return f"EX:{slug}"

    def _build_concept_prompt(self, text: str) -> str:
        """Build P1 concept extraction prompt"""
        return f"""Extract concepts and their synonyms from the following text. Return JSON with exact quotes and character spans.

Text: {text[:2000]}

Return JSON format:
{{
  "concepts": [
    {{
      "label": "disease",
      "synonyms": ["illness", "disorder"],
      "confidence": 0.9,
      "quote": "exact text span",
      "span": [120, 168]
    }}
  ]
}}"""

    def _build_relation_prompt(self, text: str) -> str:
        """Build P3 relation extraction prompt"""
        return f"""Extract semantic relations from the following text. Identify domain and range concepts. Return JSON with exact quotes.

Text: {text[:2000]}

Return JSON format:
{{
  "relations": [
    {{
      "label": "treats",
      "aliases": ["is used to treat", "helps"],
      "domain": "drug",
      "range": "disease", 
      "confidence": 0.8,
      "quote": "exact text span"
    }}
  ]
}}"""

    def _fallback_concept_extraction(self, doc_id: str, text: str) -> List[Concept]:
        """Fallback concept extraction using simple NLP"""
        # Simple noun phrase extraction as fallback
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        concepts = []

        for word in set(words):
            if len(word) > 2:  # Filter short words
                concept = Concept(
                    id=self._generate_concept_id(word),
                    label=word.lower(),
                    status=Status.PROPOSED,
                    origin=Origin.AI,
                    confidence=0.3,  # Low confidence for fallback
                    evidence=[Evidence(doc=doc_id, quote=word)],
                    meta=Meta(created_by="fallback", created_at=datetime.now(UTC))
                )
                concepts.append(concept)

        return concepts[:20]  # Limit fallback results

    def _fallback_relation_extraction(self, doc_id: str, text: str) -> List[Relation]:
        """Fallback relation extraction using simple patterns"""
        # Simple verb-based relation extraction
        patterns = [
            r'(\w+)\s+(is|are|was|were)\s+(\w+)',
            r'(\w+)\s+(has|have|had)\s+(\w+)',
            r'(\w+)\s+(uses|use|used)\s+(\w+)'
        ]

        relations = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                subj = match.group(1)
                verb = match.group(2)
                obj = match.group(3)

                relation = Relation(
                    id=self._generate_relation_id(verb),
                    label=verb.lower(),
                    domain=self._generate_concept_id(subj),
                    range=self._generate_concept_id(obj),
                    status=Status.PROPOSED,
                    confidence=0.2,  # Low confidence for fallback
                    evidence=[Evidence(doc=doc_id, quote=match.group(0))]
                )
                relations.append(relation)

        return relations[:10]  # Limit fallback results
