#!/usr/bin/env python3
"""
Pipeline V2 Example: Breakpoints and Stage Control

Demonstrates the Pipeline v2 system in SmartMemory:
1. Using PipelineRunner for step-by-step execution
2. run_to() breakpoints for stopping at specific stages
3. run_from() for resuming from checkpoints
4. undo_to() for rolling back stages
5. Inspecting pipeline state and stage timings

Pipeline V2 Stages:
- classify: Detect memory type
- coreference: Resolve pronouns (e.g., "he" → "John")
- simplify: Break complex sentences into atomic facts
- entity_ruler: Rule-based entity extraction
- llm_extract: LLM-based entity/relation extraction
- ontology_constrain: Apply ontology constraints
- store: Persist to graph
- link: Create relationships
- enrich: Add sentiment, temporal, topic metadata
- evolve: Run evolution and clustering

Run this example:
    PYTHONPATH=. python examples/pipeline_v2_example.py
"""

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from smartmemory.smart_memory import SmartMemory
    from smartmemory.pipeline.config import (
        PipelineConfig,
        RetryConfig,
        StageRetryPolicy,
        LLMExtractConfig,
        SimplifyConfig,
    )
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info("This example requires the full SmartMemory system to be available.")
    exit(1)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def demo_pipeline_config():
    """Demonstrate PipelineConfig options."""
    print_section("1. Pipeline Configuration")
    
    # Default configuration
    default_config = PipelineConfig.default(workspace_id="demo_workspace")
    print("📋 Default Configuration:")
    print(f"   Mode: {default_config.mode}")
    print(f"   Workspace: {default_config.workspace_id}")
    print(f"   Retry: max_retries={default_config.retry.max_retries}, on_failure={default_config.retry.on_failure}")
    
    # Preview mode (no storage side-effects)
    preview_config = PipelineConfig.preview()
    print("\n🔍 Preview Mode:")
    print(f"   Mode: {preview_config.mode}")
    print(f"   Evolution: {preview_config.evolve.run_evolution}")
    print(f"   Clustering: {preview_config.evolve.run_clustering}")
    
    # Custom configuration
    custom_config = PipelineConfig(
        workspace_id="custom_workspace",
        mode="sync",
        retry=RetryConfig(max_retries=3, on_failure="skip"),
        stage_retry_policies={
            "llm_extract": StageRetryPolicy(max_retries=5, backoff_seconds=2.0),
        },
        simplify=SimplifyConfig(
            enabled=True,
            split_clauses=True,
            passive_to_active=True,
        ),
        extraction=__import__('smartmemory.pipeline.config', fromlist=['ExtractionConfig']).ExtractionConfig(
            llm_extract=LLMExtractConfig(
                max_entities=15,
                max_relations=50,
            ),
        ),
    )
    
    print("\n⚙️ Custom Configuration:")
    print(f"   Mode: {custom_config.mode}")
    print(f"   Global retry: max_retries={custom_config.retry.max_retries}")
    print(f"   LLM extract retry: max_retries={custom_config.stage_retry_policies['llm_extract'].max_retries}")
    print(f"   Simplify enabled: {custom_config.simplify.enabled}")
    print(f"   Max entities: {custom_config.extraction.llm_extract.max_entities}")
    
    return custom_config


def demo_pipeline_runner():
    """Demonstrate creating and using PipelineRunner."""
    print_section("2. PipelineRunner: Full Pipeline Run")
    
    memory = SmartMemory()
    
    # Get the pipeline runner from SmartMemory
    runner = memory.create_pipeline_runner()
    
    print("🔧 Pipeline Runner created")
    print(f"   Number of stages: {len(runner.stages)}")
    print(f"   Stage names: {[s.name for s in runner.stages]}")
    
    # Run the full pipeline
    text = "John Smith is a software engineer at Google. He works on machine learning projects."
    config = PipelineConfig.preview()  # Use preview mode for demo
    
    print("\n📝 Running full pipeline on:")
    print(f"   '{text}'")
    
    try:
        state = runner.run(text, config)
        print("\n✅ Pipeline completed!")
        print(f"   Stages completed: {state.stage_history}")
        print(f"   Total time: {sum(state.stage_timings.values()):.2f}ms")
    except Exception as e:
        print(f"   ⚠️ Pipeline error (expected in demo): {e}")
    
    return memory, runner


def demo_breakpoints():
    """Demonstrate run_to() breakpoints."""
    print_section("3. Breakpoints: run_to() for Step-by-Step Execution")
    
    memory = SmartMemory()
    runner = memory.create_pipeline_runner()
    
    text = "The quick brown fox jumps over the lazy dog."
    config = PipelineConfig.preview()
    
    print("📍 Running to 'simplify' stage (breakpoint)...")
    
    try:
        # Run only up to the simplify stage
        state = runner.run_to(text, config, stop_after="simplify")
        
        print("\n✅ Stopped at breakpoint!")
        print(f"   Stages completed: {state.stage_history}")
        print(f"   Current text: {state.text[:50]}...")
        
        # Inspect extracted entities so far
        if state.entities:
            print(f"   Entities extracted: {len(state.entities)}")
        
        return state, runner, config
    except Exception as e:
        print(f"   ⚠️ Breakpoint error (expected in demo): {e}")
        return None, runner, config


def demo_resume(state, runner, config):
    """Demonstrate run_from() resumption."""
    print_section("4. Resume: run_from() to Continue from Checkpoint")
    
    if state is None:
        print("⚠️ No state to resume from (previous step may have failed)")
        return
    
    print(f"📍 Current state: {state.stage_history}")
    print("🔄 Resuming from next stage...")
    
    try:
        # Resume and run to the next breakpoint
        resumed_state = runner.run_from(
            state, 
            config, 
            start_from=None,  # Auto-detect next stage
            stop_after="llm_extract",
        )
        
        print("\n✅ Resumed and stopped at next breakpoint!")
        print(f"   Stages completed: {resumed_state.stage_history}")
        
        return resumed_state
    except Exception as e:
        print(f"   ⚠️ Resume error (expected in demo): {e}")


def demo_undo():
    """Demonstrate undo_to() rollback."""
    print_section("5. Rollback: undo_to() for Stage Reversal")
    
    print("""
🔄 undo_to() allows rolling back pipeline stages:

    # Roll back to before 'enrich' stage
    state = runner.undo_to(state, target="store")
    
This is useful for:
- Correcting extraction errors
- Re-running stages with different config
- Debugging pipeline issues
- A/B testing stage implementations

The rollback is stage-aware — each stage defines its own
undo() method to properly clean up its effects.
    """)


def demo_stage_timings():
    """Demonstrate inspecting stage timings."""
    print_section("6. Stage Timings and Metrics")
    
    memory = SmartMemory()
    runner = memory.create_pipeline_runner()
    
    text = "Python is a programming language used for AI and machine learning."
    config = PipelineConfig.preview()
    
    print("⏱️ Running pipeline with timing enabled...")
    
    try:
        state = runner.run(text, config)
        
        print("\n📊 Stage Timings:")
        for stage_name, timing_ms in state.stage_timings.items():
            bar = "█" * int(timing_ms / 10)
            print(f"   {stage_name:20} {timing_ms:8.2f}ms {bar}")
        
        total = sum(state.stage_timings.values())
        print(f"   {'TOTAL':20} {total:8.2f}ms")
        print(f"\n   Started: {state.started_at}")
        print(f"   Completed: {state.completed_at}")
    except Exception as e:
        print(f"   ⚠️ Timing error (expected in demo): {e}")


def demo_pipeline_state():
    """Demonstrate PipelineState inspection."""
    print_section("7. PipelineState: Inspecting Extraction Results")
    
    print("""
📦 PipelineState contains all intermediate data:

    state.text              # Original or simplified text
    state.entities          # List of extracted entities
    state.relations         # List of extracted relations
    state.stage_history     # Completed stages
    state.stage_timings     # Timing per stage
    state.mode              # 'sync' | 'async' | 'preview'
    state.workspace_id      # Tenant isolation
    state.raw_metadata      # User-provided metadata
    state.started_at        # Pipeline start time
    state.completed_at      # Pipeline end time

Example entity structure:
    {
        'name': 'Python',
        'type': 'programming_language',
        'confidence': 0.95,
        'source': 'llm_extract',
    }

Example relation structure:
    {
        'subject': 'Python',
        'predicate': 'used_for',
        'object': 'machine learning',
        'confidence': 0.88,
    }
    """)


def demo_practical_workflow():
    """Demonstrate a practical debugging workflow."""
    print_section("8. Practical Workflow: Debugging Extraction")
    
    print("""
🔧 Debugging workflow using Pipeline V2:

1. Run to extraction breakpoint:
   
   state = runner.run_to(text, config, stop_after="llm_extract")
   print(state.entities)  # Inspect extracted entities

2. If entities look wrong, adjust config and re-run:
   
   config.extraction.llm_extract.max_entities = 20
   state = runner.run(text, config)

3. Use undo to roll back and try again:
   
   state = runner.undo_to(state, target="simplify")
   state = runner.run_from(state, config)

4. Compare different extraction strategies:
   
   # Run with entity_ruler only
   config.extraction.llm_extract.enabled = False
   state_ruler = runner.run(text, config)
   
   # Run with LLM only
   config.extraction.entity_ruler.enabled = False
   config.extraction.llm_extract.enabled = True
   state_llm = runner.run(text, config)

5. Inspect timing to identify bottlenecks:
   
   for stage, ms in state.stage_timings.items():
       print(f"{stage}: {ms:.2f}ms")
    """)


def main():
    """Run all demonstrations."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║            SmartMemory Pipeline V2 - Demo                            ║
║                                                                      ║
║  Pipeline V2 provides fine-grained control over the ingestion       ║
║  process with breakpoints, resumption, and rollback support.        ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Run demonstrations
        demo_pipeline_config()
        memory, runner = demo_pipeline_runner()
        state, runner, config = demo_breakpoints()
        demo_resume(state, runner, config)
        demo_undo()
        demo_stage_timings()
        demo_pipeline_state()
        demo_practical_workflow()
        
        print_section("Demo Complete!")
        print("""
🎉 You've seen the core features of SmartMemory's Pipeline V2!

Key APIs:
  • PipelineConfig.default() - Standard production config
  • PipelineConfig.preview() - No storage side-effects
  • runner.run() - Execute full pipeline
  • runner.run_to() - Stop at breakpoint
  • runner.run_from() - Resume from checkpoint
  • runner.undo_to() - Roll back stages

Use cases:
  • Step-by-step debugging of extraction
  • A/B testing extraction strategies
  • Fine-tuning stage parameters
  • Performance profiling
  • Preview mode for testing
        """)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
