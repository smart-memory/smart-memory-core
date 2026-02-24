#!/usr/bin/env python3
"""
Reasoning Trace Example (System 2 Memory)

Demonstrates the Reasoning Trace system in SmartMemory:
1. Creating ReasoningStep and ReasoningTrace objects
2. Using TaskContext for task metadata
3. Storing reasoning traces as memories
4. Quality evaluation with ReasoningEvaluation
5. Querying "why" decisions were made

Reasoning traces capture chain-of-thought reasoning from agent conversations,
enabling retrieval of "why" decisions were made, not just the outcomes.

Step types:
- thought: Internal reasoning ("I think we should...")
- action: Action taken ("Let me search for...")
- observation: Result observed ("The search returned...")
- decision: Decision point reached ("Based on this, I'll...")
- conclusion: Final answer ("Therefore, the solution is...")
- reflection: Meta-reasoning ("This approach worked because...")

Run this example:
    PYTHONPATH=. python examples/reasoning_trace_example.py
"""

import logging
import uuid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from smartmemory.smart_memory import SmartMemory
    from smartmemory.models.memory_item import MemoryItem
    from smartmemory.models.reasoning import (
        ReasoningStep,
        ReasoningTrace,
        ReasoningEvaluation,
        TaskContext,
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


def demo_reasoning_steps():
    """Demonstrate creating individual reasoning steps."""
    print_section("1. Creating Reasoning Steps")
    
    # Create different types of reasoning steps
    steps = [
        ReasoningStep(type="thought", content="The user is asking about Python async patterns. I should explain asyncio."),
        ReasoningStep(type="action", content="Let me check the codebase for existing async implementations."),
        ReasoningStep(type="observation", content="Found 3 files using asyncio: main.py, worker.py, and api.py."),
        ReasoningStep(type="decision", content="Based on the existing patterns, I'll recommend using async/await with asyncio.gather()."),
        ReasoningStep(type="conclusion", content="Use asyncio.gather() for concurrent API calls to improve performance by 3x."),
        ReasoningStep(type="reflection", content="This approach worked because the user's code was I/O-bound, not CPU-bound."),
    ]
    
    print("📝 Reasoning Steps:")
    for step in steps:
        print(f"   [{step.type.upper():12}] {step.content[:60]}...")
    
    # Serialize and deserialize
    print("\n🔄 Serialization test:")
    step_dict = steps[0].to_dict()
    reconstructed = ReasoningStep.from_dict(step_dict)
    print(f"   Original: {steps[0].content[:50]}...")
    print(f"   Reconstructed: {reconstructed.content[:50]}...")
    print(f"   Match: {'✅' if steps[0].content == reconstructed.content else '❌'}")
    
    return steps


def demo_task_context():
    """Demonstrate TaskContext for task metadata."""
    print_section("2. Task Context")
    
    context = TaskContext(
        goal="Debug and fix the authentication error",
        input="Why am I getting a 401 error when calling the API?",
        task_type="debugging",
        domain="python",
        complexity="medium",
    )
    
    print("📋 Task Context:")
    print(f"   Goal: {context.goal}")
    print(f"   Input: {context.input}")
    print(f"   Task Type: {context.task_type}")
    print(f"   Domain: {context.domain}")
    print(f"   Complexity: {context.complexity}")
    
    # Useful for filtered retrieval
    print("\n💡 Use Case: Filter reasoning traces by:")
    print("   - task_type: Find all 'debugging' traces")
    print("   - domain: Find all 'python' traces")
    print("   - complexity: Find 'high' complexity examples")
    
    return context


def demo_reasoning_trace(steps, context):
    """Demonstrate creating a complete reasoning trace."""
    print_section("3. Complete Reasoning Trace")
    
    trace = ReasoningTrace(
        trace_id=f"trace_{uuid.uuid4().hex[:12]}",
        steps=steps,
        task_context=context,
        session_id="session_abc123",
        has_explicit_markup=False,
        artifact_ids=["mem_fix_auth_001", "mem_solution_002"],
    )
    
    print(f"🧠 Reasoning Trace: {trace.trace_id}")
    print(f"   Step count: {trace.step_count}")
    print(f"   Session ID: {trace.session_id}")
    print(f"   Has explicit markup: {trace.has_explicit_markup}")
    print(f"   Artifact IDs: {trace.artifact_ids}")
    
    # Generate content for vector embedding
    print("\n📄 Generated content for embedding:")
    content = trace.content
    print(f"   {content[:200]}...")
    
    return trace


def demo_quality_evaluation():
    """Demonstrate reasoning quality evaluation."""
    print_section("4. Quality Evaluation")
    
    # High-quality trace evaluation
    good_eval = ReasoningEvaluation(
        quality_score=0.85,
        has_loops=False,
        has_redundancy=False,
        step_diversity=0.8,
        issues=[],
        suggestions=["Consider adding more observations"],
    )
    
    print("✅ High-Quality Trace Evaluation:")
    print(f"   Quality Score: {good_eval.quality_score:.2f}")
    print(f"   Has Loops: {good_eval.has_loops}")
    print(f"   Has Redundancy: {good_eval.has_redundancy}")
    print(f"   Step Diversity: {good_eval.step_diversity:.2f}")
    print(f"   Should Store: {good_eval.should_store}")
    
    # Low-quality trace evaluation
    bad_eval = ReasoningEvaluation(
        quality_score=0.3,
        has_loops=True,
        has_redundancy=True,
        step_diversity=0.2,
        issues=[
            {"type": "repetition", "description": "Steps 3-5 repeat the same idea", "severity": "medium"},
            {"type": "loop", "description": "Circular reasoning detected", "severity": "high"},
        ],
        suggestions=["Break out of the loop", "Add concrete observations"],
    )
    
    print("\n❌ Low-Quality Trace Evaluation:")
    print(f"   Quality Score: {bad_eval.quality_score:.2f}")
    print(f"   Has Loops: {bad_eval.has_loops}")
    print(f"   Should Store: {bad_eval.should_store}")
    print("   Issues:")
    for issue in bad_eval.issues:
        print(f"      [{issue['severity'].upper()}] {issue['description']}")
    
    return good_eval


def demo_storing_traces():
    """Demonstrate storing reasoning traces as memories."""
    print_section("5. Storing Reasoning Traces")
    
    memory = SmartMemory()
    
    # Create a reasoning trace
    steps = [
        ReasoningStep(type="thought", content="I need to analyze the error message"),
        ReasoningStep(type="observation", content="The error indicates a missing import"),
        ReasoningStep(type="decision", content="Add the missing import statement"),
        ReasoningStep(type="conclusion", content="Fixed by adding 'from typing import Optional'"),
    ]
    
    trace = ReasoningTrace(
        trace_id=f"trace_{uuid.uuid4().hex[:12]}",
        steps=steps,
        task_context=TaskContext(
            goal="Fix import error",
            task_type="debugging",
            domain="python",
        ),
    )
    
    # Store as a reasoning memory
    item = MemoryItem(
        content=trace.content,
        memory_type="reasoning",
        item_id=trace.trace_id,
        metadata={
            "trace": trace.to_dict(),
            "domain": trace.task_context.domain if trace.task_context else None,
            "task_type": trace.task_context.task_type if trace.task_context else None,
        },
    )
    
    # Use add() for direct storage (reasoning traces are already processed)
    item_id = memory.add(item)
    
    print(f"✅ Stored reasoning trace: {item_id}")
    print("   Memory type: reasoning")
    print(f"   Content preview: {trace.content[:100]}...")
    
    return memory, trace


def demo_querying_why():
    """Demonstrate querying 'why' decisions were made."""
    print_section("6. Querying 'Why' Decisions Were Made")
    
    memory = SmartMemory()
    
    # Create and store multiple reasoning traces
    traces_data = [
        {
            "goal": "Choose database for the project",
            "steps": [
                ReasoningStep(type="thought", content="Need to consider data relationships"),
                ReasoningStep(type="observation", content="The data model has complex relations"),
                ReasoningStep(type="decision", content="PostgreSQL is best for relational data"),
            ],
        },
        {
            "goal": "Select API design pattern",
            "steps": [
                ReasoningStep(type="thought", content="Considering REST vs GraphQL"),
                ReasoningStep(type="observation", content="Frontend needs flexible queries"),
                ReasoningStep(type="decision", content="GraphQL provides query flexibility"),
            ],
        },
    ]
    
    for data in traces_data:
        trace = ReasoningTrace(
            trace_id=f"trace_{uuid.uuid4().hex[:12]}",
            steps=data["steps"],
            task_context=TaskContext(goal=data["goal"], task_type="choice"),
        )
        item = MemoryItem(
            content=trace.content,
            memory_type="reasoning",
            item_id=trace.trace_id,
            metadata={"trace": trace.to_dict()},
        )
        memory.add(item)
    
    print("📝 Stored 2 reasoning traces")
    
    # Query for reasoning about a decision
    print("\n🔍 Query: 'Why did we choose the database?'")
    results = memory.search("database choice decision", memory_type="reasoning", top_k=1)
    
    if results:
        print(f"   Found {len(results)} relevant trace(s)")
        for result in results:
            print(f"   → {getattr(result, 'content', str(result))[:100]}...")
    else:
        print("   (No results - may need FalkorDB running)")


def demo_practical_workflow():
    """Demonstrate a practical workflow for capturing reasoning."""
    print_section("7. Practical Workflow: Capturing Agent Reasoning")
    
    print("""
📋 Workflow for capturing reasoning in your agent:

1. During task execution, collect reasoning steps:
   
   steps = []
   steps.append(ReasoningStep(type="thought", content="..."))
   steps.append(ReasoningStep(type="action", content="..."))
   steps.append(ReasoningStep(type="observation", content="..."))
   steps.append(ReasoningStep(type="conclusion", content="..."))

2. After task completion, create a ReasoningTrace:
   
   trace = ReasoningTrace(
       trace_id=generate_id(),
       steps=steps,
       task_context=TaskContext(
           goal="User's original request",
           task_type="debugging",
           domain="python",
       ),
       artifact_ids=[id_of_memory_created],
   )

3. Evaluate quality before storing:
   
   evaluation = evaluate_trace(trace)  # Your evaluation logic
   if evaluation.should_store:
       store_trace(trace)

4. Query reasoning later:
   
   results = memory.search(
       "why did we choose X",
       memory_type="reasoning",
   )
    """)


def main():
    """Run all demonstrations."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║         SmartMemory Reasoning Traces (System 2 Memory) Demo         ║
║                                                                      ║
║  Reasoning traces capture chain-of-thought reasoning, enabling      ║
║  retrieval of "why" decisions were made, not just outcomes.         ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Run demonstrations
        steps = demo_reasoning_steps()
        context = demo_task_context()
        trace = demo_reasoning_trace(steps, context)
        demo_quality_evaluation()
        demo_storing_traces()
        demo_querying_why()
        demo_practical_workflow()
        
        print_section("Demo Complete!")
        print("""
🎉 You've seen the core features of SmartMemory's Reasoning Traces!

Key concepts:
  • ReasoningStep captures individual reasoning moments
  • ReasoningTrace collects steps into a complete trace
  • TaskContext provides metadata for filtered retrieval
  • ReasoningEvaluation gates low-quality traces
  • Store as memory_type="reasoning" for searchability

Use cases:
  • "Why did the agent make this decision?"
  • "Show me similar debugging sessions"
  • "How was this problem solved before?"
  • Learning from successful reasoning patterns
        """)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
