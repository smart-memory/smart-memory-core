#!/usr/bin/env python3
"""
End-to-end ontology inference demo that exercises both fallback (basic)
and optional OntoGPT paths via configuration/env flags.

Usage examples:
  # Fallback/basic inference (no external calls)
  python examples/ontology_inference_e2e.py --mode basic

  # Attempt OntoGPT (requires ontogpt CLI and template); falls back on failure
  python examples/ontology_inference_e2e.py --mode ontogpt \
      --template named_entity --template-dir ~/ontogpt-templates

  # Auto-detect: use OntoGPT if available and properly configured, else basic
  python examples/ontology_inference_e2e.py --mode auto

Environment overrides (optional):
  ONTOLOGY_INFERENCE_ENGINE=ontogpt|basic
  ONTOLOGY_INFERENCE_ONTOGPT_ENABLED=true|false
  ONTOGPT_TEMPLATE, ONTOGPT_TEMPLATE_DIR, ONTOGPT_PROVIDER, ONTOGPT_MODEL, ONTOGPT_API_BASE
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

# Ensure project root (smart-memory-core) is on sys.path for local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from smartmemory.ontology.manager import OntologyManager  # noqa: E402


def build_sample_extraction_history():
    """Return a small, diverse extraction history for demo/testing."""
    return [
        {
            "entities": [
                {"name": "John Smith", "type": "person", "properties": {"role": "Engineer", "skills": "Python"}},
                {"name": "Acme Corp", "type": "organization", "properties": {"location": "San Francisco"}},
                {"name": "WidgetPro", "type": "product", "properties": {"category": "Tool"}},
            ],
            "relations": [
                {"source": "John Smith", "target": "Acme Corp", "relation_type": "WORKS_AT"},
                {"source": "Acme Corp", "target": "WidgetPro", "relation_type": "PRODUCES"},
            ],
        },
        {
            "entities": [
                {"name": "Sarah Johnson", "type": "person", "properties": {"role": "Engineer", "skills": "React"}},
                {"name": "Acme Corp", "type": "organization", "properties": {"location": "San Francisco"}},
                {"name": "ML Suite", "type": "product", "properties": {"category": "Software"}},
            ],
            "relations": [
                {"source": "Sarah Johnson", "target": "John Smith", "relation_type": "REPORTS_TO"},
                {"source": "Acme Corp", "target": "ML Suite", "relation_type": "USES"},
            ],
        },
        {
            "entities": [
                {"name": "Mike Chen", "type": "person", "properties": {"role": "Scientist", "skills": "Machine Learning"}},
                {"name": "DeepAI Labs", "type": "organization", "properties": {"location": "Mountain View"}},
                {"name": "WidgetPro", "type": "product", "properties": {"category": "Tool"}},
            ],
            "relations": [
                {"source": "Mike Chen", "target": "DeepAI Labs", "relation_type": "WORKS_AT"},
                {"source": "Mike Chen", "target": "John Smith", "relation_type": "COLLABORATES_WITH"},
            ],
        },
    ]


def resolve_config_path() -> str:
    """Resolve project config.json and export SMARTMEMORY_CONFIG for consumers."""
    default_cfg = PROJECT_ROOT / "config.json"
    cfg = os.environ.get("SMARTMEMORY_CONFIG") or str(default_cfg)
    cfg_abs = str(Path(cfg).expanduser().resolve())
    os.environ["SMARTMEMORY_CONFIG"] = cfg_abs
    return cfg_abs


def set_mode_env(mode: str, template: str = None, template_dir: str = None, provider: str = None,
                 model: str = None, api_base: str = None):
    """Configure environment flags for ontology inference stages and OntoGPT."""
    mode = (mode or "auto").lower()

    def _ontogpt_available():
        return shutil.which("ontogpt") is not None

    # Decide effective mode
    effective_mode = mode
    if mode == "auto":
        if _ontogpt_available() and (template or os.environ.get("ONTOGPT_TEMPLATE")):
            effective_mode = "ontogpt"
        else:
            effective_mode = "basic"

    if effective_mode == "basic":
        os.environ["ONTOLOGY_INFERENCE_ENGINE"] = "basic"
        # Ensure no env override forces ontogpt.enabled to truthy string
        for var in ("ONTOLOGY_INFERENCE_ONTOGPT_ENABLED", "ONTOLOGY_ONTOGPT_ENABLED"):
            if var in os.environ:
                del os.environ[var]
    else:
        os.environ["ONTOLOGY_INFERENCE_ENGINE"] = "ontogpt"
        # Correct flag matching config.json's ontology.inference.ontogpt.enabled
        os.environ["ONTOLOGY_ONTOGPT_ENABLED"] = "true"

    # Optional OntoGPT settings
    if template:
        os.environ["ONTOGPT_TEMPLATE"] = template
    if template_dir:
        os.environ["ONTOGPT_TEMPLATE_DIR"] = template_dir
    if provider:
        os.environ["ONTOGPT_PROVIDER"] = provider
    if model:
        os.environ["ONTOGPT_MODEL"] = model
    if api_base:
        os.environ["ONTOGPT_API_BASE"] = api_base

    return effective_mode


def print_ontology_summary(ontology):
    print("\n=== Ontology Summary ===")
    print(f"ID: {ontology.id}")
    print(f"Name: {ontology.name}")
    print(f"Created by: {ontology.created_by}")
    print(f"Entity types: {len(ontology.entity_types)} | Relationship types: {len(ontology.relationship_types)}")

    # Show a few entity types
    shown = 0
    for et_name, et in ontology.entity_types.items():
        print(f"- EntityType: {et_name}")
        if getattr(et, "required_properties", None):
            print(f"  required: {sorted(list(et.required_properties))}")
        if getattr(et, "properties", None):
            props = list(et.properties.keys())
            print(f"  props: {props[:6]}{'...' if len(props) > 6 else ''}")
        shown += 1
        if shown >= 5:
            break

    # Show relationship types
    for rt_name, rt in ontology.relationship_types.items():
        st = sorted(list(rt.source_types)) if getattr(rt, "source_types", None) else []
        tt = sorted(list(rt.target_types)) if getattr(rt, "target_types", None) else []
        print(f"- RelType: {rt_name} (source: {st or '*'}, target: {tt or '*'})")


def main():
    parser = argparse.ArgumentParser(description="E2E Ontology Inference Demo (OntoGPT + fallback)")
    parser.add_argument("--mode", choices=["basic", "ontogpt", "auto"], default="auto",
                        help="Select inference stages mode (default: auto)")
    parser.add_argument("--template", help="OntoGPT template name/path", default=None)
    parser.add_argument("--template-dir", help="Directory containing OntoGPT templates", default=None)
    parser.add_argument("--provider", help="LLM provider for OntoGPT (e.g., openai)", default=None)
    parser.add_argument("--model", help="Model name for OntoGPT", default=None)
    parser.add_argument("--api-base", help="API base URL for OntoGPT provider", default=None)
    parser.add_argument("--ontology-name", default="e2e_inferred", help="Name for the inferred ontology")

    args = parser.parse_args()

    cfg_path = resolve_config_path()
    print(f"Using config: {cfg_path}")

    effective_mode = set_mode_env(
        mode=args.mode,
        template=args.template,
        template_dir=args.template_dir,
        provider=args.provider,
        model=args.model,
        api_base=args.api_base,
    )
    print(f"Requested mode: {args.mode} | Effective mode: {effective_mode}")

    # Build demo extraction history and run inference
    history = build_sample_extraction_history()
    mgr = OntologyManager()
    ontology = mgr.infer_ontology_from_extractions(history, ontology_name=args.ontology_name)

    # Print summary and the stages actually used (from provenance)
    engine_used = ontology.created_by
    print(f"Engine used (from ontology.created_by): {engine_used}")
    print_ontology_summary(ontology)

    # Show where the ontology was saved (filesystem storage)
    storage_dir = Path("ontologies").resolve()
    print(f"\nSaved ontologies directory: {storage_dir}")
    if storage_dir.exists():
        files = sorted([p.name for p in storage_dir.glob("*.json")])
        if files:
            print(f"Artifacts: {files[-3:]}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"E2E failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
