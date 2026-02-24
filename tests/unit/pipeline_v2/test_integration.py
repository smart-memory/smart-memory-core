"""Integration test -- run the full pipeline with mocked dependencies."""

import pytest

pytestmark = pytest.mark.unit


from unittest.mock import MagicMock, patch
from dataclasses import replace

from smartmemory.pipeline.config import PipelineConfig
from smartmemory.pipeline.runner import PipelineRunner
from smartmemory.pipeline.state import PipelineState
from smartmemory.pipeline.stages.classify import ClassifyStage
from smartmemory.pipeline.stages.coreference import CoreferenceStageCommand
from smartmemory.pipeline.stages.simplify import SimplifyStage
from smartmemory.pipeline.stages.entity_ruler import EntityRulerStage
from smartmemory.pipeline.stages.llm_extract import LLMExtractStage
from smartmemory.pipeline.stages.ontology_constrain import OntologyConstrainStage
from smartmemory.pipeline.stages.store import StoreStage
from smartmemory.pipeline.stages.link import LinkStage
from smartmemory.pipeline.stages.enrich import EnrichStage
from smartmemory.pipeline.stages.ground import GroundStage
from smartmemory.pipeline.stages.evolve import EvolveStage


# ------------------------------------------------------------------ #
# Mock helpers
# ------------------------------------------------------------------ #


def _mock_ingestion_flow():
    flow = MagicMock()
    flow.classify_item.return_value = ["semantic"]
    return flow


def _mock_ontology_graph():
    og = MagicMock()
    og.get_type_status.return_value = "seed"
    og.add_provisional.return_value = True
    og.promote.return_value = True
    return og


def _mock_memory():
    memory = MagicMock()
    memory._crud.add.return_value = {
        "memory_node_id": "test_item_123",
        "entity_node_ids": ["e1"],
    }
    memory._evolution = MagicMock()
    memory._clustering = MagicMock()
    memory._graph = MagicMock()
    memory._grounding = MagicMock()
    return memory


def _mock_linking():
    return MagicMock()


def _mock_enrichment_pipeline():
    pipeline = MagicMock()
    pipeline.run_enrichment.return_value = {"basic": "done"}
    return pipeline


def _build_all_stages():
    """Create the full 11-stage pipeline with mocked dependencies."""
    flow = _mock_ingestion_flow()
    ontology = _mock_ontology_graph()
    memory = _mock_memory()
    linking = _mock_linking()
    enrichment = _mock_enrichment_pipeline()

    return [
        ClassifyStage(flow),
        CoreferenceStageCommand(),
        SimplifyStage(),
        EntityRulerStage(),
        LLMExtractStage(),
        OntologyConstrainStage(ontology),
        StoreStage(memory),
        LinkStage(linking),
        EnrichStage(enrichment),
        GroundStage(memory),
        EvolveStage(memory),
    ]


STAGE_NAMES = [
    "classify",
    "coreference",
    "simplify",
    "entity_ruler",
    "llm_extract",
    "ontology_constrain",
    "store",
    "link",
    "enrich",
    "ground",
    "evolve",
]


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #


def _patch_coref_and_extraction():
    """Return a list of patch context managers for coreference, simplify, entity_ruler, llm_extract."""
    return [
        patch(
            "smartmemory.pipeline.stages.coreference.CoreferenceStageCommand.execute",
            side_effect=lambda self_or_state, config_or_none=None, **kw: replace(
                self_or_state if isinstance(self_or_state, PipelineState) else config_or_none,
                resolved_text=(
                    self_or_state if isinstance(self_or_state, PipelineState) else config_or_none
                ).text,
            ),
        ),
        patch(
            "smartmemory.pipeline.stages.simplify.SimplifyStage.execute",
            side_effect=lambda self_or_state, config_or_none=None, **kw: replace(
                self_or_state if isinstance(self_or_state, PipelineState) else config_or_none,
                simplified_sentences=[
                    (self_or_state if isinstance(self_or_state, PipelineState) else config_or_none).text
                ],
            ),
        ),
        patch(
            "smartmemory.pipeline.stages.entity_ruler.EntityRulerStage.execute",
            side_effect=lambda self_or_state, config_or_none=None, **kw: replace(
                self_or_state if isinstance(self_or_state, PipelineState) else config_or_none,
                ruler_entities=[{"name": "Claude", "entity_type": "technology", "confidence": 0.9}],
            ),
        ),
        patch(
            "smartmemory.pipeline.stages.llm_extract.LLMExtractStage.execute",
            side_effect=lambda self_or_state, config_or_none=None, **kw: replace(
                self_or_state if isinstance(self_or_state, PipelineState) else config_or_none,
                llm_entities=[],
                llm_relations=[],
            ),
        ),
    ]


@patch(
    "smartmemory.pipeline.stages.coreference.CoreferenceStageCommand.execute",
    side_effect=lambda self_or_state, config_or_none=None, **kw: replace(
        self_or_state if isinstance(self_or_state, PipelineState) else config_or_none,
        resolved_text=(self_or_state if isinstance(self_or_state, PipelineState) else config_or_none).text,
    ),
)
@patch(
    "smartmemory.pipeline.stages.simplify.SimplifyStage.execute",
    side_effect=lambda self_or_state, config_or_none=None, **kw: replace(
        self_or_state if isinstance(self_or_state, PipelineState) else config_or_none,
        simplified_sentences=[
            (self_or_state if isinstance(self_or_state, PipelineState) else config_or_none).text
        ],
    ),
)
@patch(
    "smartmemory.pipeline.stages.entity_ruler.EntityRulerStage.execute",
    side_effect=lambda self_or_state, config_or_none=None, **kw: replace(
        self_or_state if isinstance(self_or_state, PipelineState) else config_or_none,
        ruler_entities=[{"name": "Claude", "entity_type": "technology", "confidence": 0.9}],
    ),
)
@patch(
    "smartmemory.pipeline.stages.llm_extract.LLMExtractStage.execute",
    side_effect=lambda self_or_state, config_or_none=None, **kw: replace(
        self_or_state if isinstance(self_or_state, PipelineState) else config_or_none,
        llm_entities=[],
        llm_relations=[],
    ),
)
def test_full_pipeline_runs_all_stages(_mock_llm, _mock_ruler, _mock_simplify, _mock_coref):
    """All 11 stages execute and are recorded in stage_history."""
    stages = _build_all_stages()
    runner = PipelineRunner(stages)
    config = PipelineConfig.default(workspace_id="test")

    state = runner.run("Claude is an AI assistant.", config)

    assert len(state.stage_history) == 11
    assert state.stage_history == STAGE_NAMES
    assert state.completed_at is not None
    assert state.item_id == "test_item_123"


@patch(
    "smartmemory.pipeline.stages.coreference.CoreferenceStageCommand.execute",
    side_effect=lambda self_or_state, config_or_none=None, **kw: replace(
        self_or_state if isinstance(self_or_state, PipelineState) else config_or_none,
        resolved_text=(self_or_state if isinstance(self_or_state, PipelineState) else config_or_none).text,
    ),
)
@patch(
    "smartmemory.pipeline.stages.simplify.SimplifyStage.execute",
    side_effect=lambda self_or_state, config_or_none=None, **kw: replace(
        self_or_state if isinstance(self_or_state, PipelineState) else config_or_none,
        simplified_sentences=[
            (self_or_state if isinstance(self_or_state, PipelineState) else config_or_none).text
        ],
    ),
)
@patch(
    "smartmemory.pipeline.stages.entity_ruler.EntityRulerStage.execute",
    side_effect=lambda self_or_state, config_or_none=None, **kw: replace(
        self_or_state if isinstance(self_or_state, PipelineState) else config_or_none,
        ruler_entities=[],
    ),
)
@patch(
    "smartmemory.pipeline.stages.llm_extract.LLMExtractStage.execute",
    side_effect=lambda self_or_state, config_or_none=None, **kw: replace(
        self_or_state if isinstance(self_or_state, PipelineState) else config_or_none,
        llm_entities=[],
        llm_relations=[],
    ),
)
def test_pipeline_preview_mode(_mock_llm, _mock_ruler, _mock_simplify, _mock_coref):
    """Preview mode returns preview_item and skips evolution/clustering."""
    stages = _build_all_stages()
    runner = PipelineRunner(stages)
    config = PipelineConfig.preview(workspace_id="test")

    state = runner.run("Preview test content.", config)

    assert state.item_id == "preview_item"
    # Evolution should not have been called in preview mode
    memory = stages[6]._memory  # StoreStage holds the memory mock (index 6 in 11-stage)
    memory._evolution.run_evolution_cycle.assert_not_called()
    memory._clustering.run.assert_not_called()


@patch(
    "smartmemory.pipeline.stages.coreference.CoreferenceStageCommand.execute",
    side_effect=lambda self_or_state, config_or_none=None, **kw: replace(
        self_or_state if isinstance(self_or_state, PipelineState) else config_or_none,
        resolved_text=(self_or_state if isinstance(self_or_state, PipelineState) else config_or_none).text,
    ),
)
@patch(
    "smartmemory.pipeline.stages.simplify.SimplifyStage.execute",
    side_effect=lambda self_or_state, config_or_none=None, **kw: replace(
        self_or_state if isinstance(self_or_state, PipelineState) else config_or_none,
        simplified_sentences=[
            (self_or_state if isinstance(self_or_state, PipelineState) else config_or_none).text
        ],
    ),
)
@patch(
    "smartmemory.pipeline.stages.entity_ruler.EntityRulerStage.execute",
    side_effect=lambda self_or_state, config_or_none=None, **kw: replace(
        self_or_state if isinstance(self_or_state, PipelineState) else config_or_none,
        ruler_entities=[],
    ),
)
@patch(
    "smartmemory.pipeline.stages.llm_extract.LLMExtractStage.execute",
    side_effect=lambda self_or_state, config_or_none=None, **kw: replace(
        self_or_state if isinstance(self_or_state, PipelineState) else config_or_none,
        llm_entities=[],
        llm_relations=[],
    ),
)
def test_pipeline_run_to_ontology_constrain(_mock_llm, _mock_ruler, _mock_simplify, _mock_coref):
    """run_to('ontology_constrain') executes through the extraction stages."""
    stages = _build_all_stages()
    runner = PipelineRunner(stages)
    config = PipelineConfig.default(workspace_id="test")

    state = runner.run_to("Claude is an AI.", config, stop_after="ontology_constrain")

    assert state.stage_history == [
        "classify",
        "coreference",
        "simplify",
        "entity_ruler",
        "llm_extract",
        "ontology_constrain",
    ]
    assert "store" not in state.stage_history
    assert "evolve" not in state.stage_history


@patch(
    "smartmemory.pipeline.stages.coreference.CoreferenceStageCommand.execute",
    side_effect=lambda self_or_state, config_or_none=None, **kw: replace(
        self_or_state if isinstance(self_or_state, PipelineState) else config_or_none,
        resolved_text=(self_or_state if isinstance(self_or_state, PipelineState) else config_or_none).text,
    ),
)
@patch(
    "smartmemory.pipeline.stages.simplify.SimplifyStage.execute",
    side_effect=lambda self_or_state, config_or_none=None, **kw: replace(
        self_or_state if isinstance(self_or_state, PipelineState) else config_or_none,
        simplified_sentences=[
            (self_or_state if isinstance(self_or_state, PipelineState) else config_or_none).text
        ],
    ),
)
@patch(
    "smartmemory.pipeline.stages.entity_ruler.EntityRulerStage.execute",
    side_effect=lambda self_or_state, config_or_none=None, **kw: replace(
        self_or_state if isinstance(self_or_state, PipelineState) else config_or_none,
        ruler_entities=[],
    ),
)
@patch(
    "smartmemory.pipeline.stages.llm_extract.LLMExtractStage.execute",
    side_effect=lambda self_or_state, config_or_none=None, **kw: replace(
        self_or_state if isinstance(self_or_state, PipelineState) else config_or_none,
        llm_entities=[],
        llm_relations=[],
    ),
)
def test_pipeline_run_from_checkpoint(_mock_llm, _mock_ruler, _mock_simplify, _mock_coref):
    """run_from() resumes from a checkpoint and completes remaining stages."""
    stages = _build_all_stages()
    runner = PipelineRunner(stages)
    config = PipelineConfig.default(workspace_id="test")

    # Phase 1: run up to ontology_constrain
    partial_state = runner.run_to("Claude is an AI.", config, stop_after="ontology_constrain")
    assert partial_state.stage_history == [
        "classify",
        "coreference",
        "simplify",
        "entity_ruler",
        "llm_extract",
        "ontology_constrain",
    ]

    # Phase 2: resume from checkpoint
    final_state = runner.run_from(partial_state, config)

    assert final_state.stage_history == STAGE_NAMES
    assert final_state.completed_at is not None
    assert final_state.item_id == "test_item_123"


