"""
Complete Zettelkasten System Demonstration

This example demonstrates all features of SmartMemory's Zettelkasten system:
1. Creating interconnected notes
2. Bidirectional linking
3. Knowledge cluster detection
4. Discovery engine features
5. System analytics
6. Best practices

Run this example:
    PYTHONPATH=. python examples/zettelkasten_example.py
"""

from smartmemory.memory.types.zettel_memory import ZettelMemory
from smartmemory.models.memory_item import MemoryItem


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def demo_basic_operations():
    """Demonstrate basic Zettelkasten operations."""
    print_section("1. Basic Operations: Creating and Linking Notes")
    
    # Initialize Zettelkasten
    zettel = ZettelMemory()
    
    # Create interconnected notes about AI/ML
    notes = [
        MemoryItem(
            content="""# Machine Learning

Machine learning is a subset of artificial intelligence that enables systems 
to learn and improve from experience without being explicitly programmed.

Key concepts: supervised learning, unsupervised learning, reinforcement learning.
            """,
            metadata={
                'title': 'Machine Learning Basics',
                'tags': ['ai', 'machine-learning', 'fundamentals'],
                'concepts': ['Learning', 'Data', 'Algorithms', 'AI']
            },
            item_id='ml_basics'
        ),
        MemoryItem(
            content="""# Neural Networks

Neural networks are computing systems inspired by biological neural networks.
They consist of interconnected nodes (neurons) organized in layers.

Foundation for deep learning and modern AI systems.
            """,
            metadata={
                'title': 'Neural Networks',
                'tags': ['ai', 'neural-networks', 'deep-learning'],
                'concepts': ['Neurons', 'Layers', 'Deep Learning', 'AI']
            },
            item_id='neural_networks'
        ),
        MemoryItem(
            content="""# Deep Learning

Deep learning uses neural networks with multiple layers to learn 
hierarchical representations of data.

Applications: computer vision, NLP, speech recognition.
            """,
            metadata={
                'title': 'Deep Learning',
                'tags': ['ai', 'deep-learning', 'neural-networks'],
                'concepts': ['Neural Networks', 'Hierarchical Learning', 'AI']
            },
            item_id='deep_learning'
        ),
        MemoryItem(
            content="""# Natural Language Processing

NLP enables computers to understand, interpret, and generate human language.
Modern NLP heavily relies on deep learning and transformer architectures.

Applications: chatbots, translation, sentiment analysis.
            """,
            metadata={
                'title': 'Natural Language Processing',
                'tags': ['ai', 'nlp', 'language'],
                'concepts': ['Language', 'Transformers', 'AI', 'Deep Learning']
            },
            item_id='nlp'
        ),
        MemoryItem(
            content="""# Computer Vision

Computer vision enables machines to interpret and understand visual information.
Uses convolutional neural networks (CNNs) for image processing.

Applications: object detection, facial recognition, autonomous vehicles.
            """,
            metadata={
                'title': 'Computer Vision',
                'tags': ['ai', 'computer-vision', 'image-processing'],
                'concepts': ['Vision', 'CNN', 'AI', 'Deep Learning']
            },
            item_id='computer_vision'
        ),
        MemoryItem(
            content="""# Reinforcement Learning

RL is learning through interaction with an environment to maximize rewards.
Agent learns optimal behavior through trial and error.

Applications: game playing, robotics, resource management.
            """,
            metadata={
                'title': 'Reinforcement Learning',
                'tags': ['ai', 'reinforcement-learning', 'agents'],
                'concepts': ['Rewards', 'Agents', 'Environment', 'AI']
            },
            item_id='reinforcement_learning'
        )
    ]
    
    # Add all notes
    print("Adding notes to Zettelkasten...")
    for note in notes:
        result = zettel.add(note)
        if result:
            print(f"  ✅ Added: {note.metadata.get('title')}")
    
    # Create manual bidirectional links
    print("\nCreating bidirectional links...")
    links = [
        ('ml_basics', 'neural_networks', 'FOUNDATION_FOR'),
        ('neural_networks', 'deep_learning', 'ENABLES'),
        ('deep_learning', 'nlp', 'USED_IN'),
        ('deep_learning', 'computer_vision', 'USED_IN'),
        ('ml_basics', 'reinforcement_learning', 'INCLUDES')
    ]
    
    for source, target, link_type in links:
        zettel.create_bidirectional_link(source, target, link_type)
        source_note = zettel.get(source)
        target_note = zettel.get(target)
        print(f"  🔗 {source_note.metadata.get('title')} → {target_note.metadata.get('title')}")
    
    return zettel


def demo_bidirectional_navigation(zettel: ZettelMemory):
    """Demonstrate bidirectional navigation features."""
    print_section("2. Bidirectional Navigation: Exploring Connections")
    
    # Get backlinks for a note
    note_id = 'deep_learning'
    print("Exploring connections for: Deep Learning\n")
    
    backlinks = zettel.get_backlinks(note_id)
    print(f"📥 Backlinks (notes linking TO Deep Learning): {len(backlinks)}")
    for note in backlinks:
        print(f"  ← {note.metadata.get('title')}")
    
    # Get complete bidirectional view
    connections = zettel.get_bidirectional_connections(note_id)
    
    print(f"\n📤 Forward links (Deep Learning links TO): {len(connections['forward_links'])}")
    for note in connections['forward_links']:
        print(f"  → {note.metadata.get('title')}")
    
    print(f"\n🏷️  Tag-related notes: {len(connections['related_by_tags'])}")
    for note in connections['related_by_tags'][:3]:
        print(f"  # {note.metadata.get('title')}")
    
    print(f"\n💡 Concept-related notes: {len(connections['related_by_concepts'])}")
    for note in connections['related_by_concepts'][:3]:
        print(f"  ((  {note.metadata.get('title')}")
    
    total = sum(len(conn_list) for conn_list in connections.values())
    print(f"\n📊 Total connections: {total}")


def demo_knowledge_clusters(zettel: ZettelMemory):
    """Demonstrate knowledge cluster detection."""
    print_section("3. Emergent Structure: Knowledge Clusters")
    
    # Detect knowledge clusters
    clusters = zettel.detect_knowledge_clusters(min_cluster_size=2)
    
    print(f"Found {len(clusters)} knowledge clusters:\n")
    
    for i, cluster in enumerate(clusters[:3], 1):
        print(f"🌐 Cluster {i}: {cluster.cluster_id}")
        print(f"   Notes: {len(cluster.note_ids)}")
        print(f"   Central concepts: {', '.join(cluster.central_concepts[:5])}")
        print(f"   Connection density: {cluster.connection_density:.3f}")
        print(f"   Emergence score: {cluster.emergence_score:.2f}")
        
        # Show notes in cluster
        print("   Members:")
        for note_id in cluster.note_ids[:5]:
            note = zettel.get(note_id)
            if note:
                print(f"     • {note.metadata.get('title')}")
        print()


def demo_knowledge_bridges(zettel: ZettelMemory):
    """Demonstrate knowledge bridge detection."""
    print_section("4. Knowledge Bridges: Connecting Domains")
    
    bridges = zettel.find_knowledge_bridges()
    
    if bridges:
        print(f"Found {len(bridges)} knowledge bridges:\n")
        
        for note_id, connected_domains in bridges[:3]:
            note = zettel.get(note_id)
            print(f"🌉 Bridge: {note.metadata.get('title')}")
            print(f"   Connects domains: {', '.join(connected_domains[:5])}")
            print("   This note is crucial for knowledge integration!\n")
    else:
        print("No knowledge bridges detected (need more interconnected notes)")


def demo_concept_emergence(zettel: ZettelMemory):
    """Demonstrate concept emergence detection."""
    print_section("5. Concept Emergence: Trending Ideas")
    
    emerging = zettel.detect_concept_emergence()
    
    print("🌱 Emerging concepts (by importance):\n")
    
    for concept, score in list(emerging.items())[:10]:
        # Visual bar
        bar_length = int(score * 10)
        bar = '█' * bar_length + '░' * (10 - bar_length)
        print(f"   {concept:25s} {bar} {score:.3f}")


def demo_knowledge_paths(zettel: ZettelMemory):
    """Demonstrate knowledge path finding."""
    print_section("6. Discovery Engine: Finding Knowledge Paths")
    
    start = 'ml_basics'
    end = 'computer_vision'
    
    start_note = zettel.get(start)
    end_note = zettel.get(end)
    
    print(f"Finding paths from '{start_note.metadata.get('title')}' to '{end_note.metadata.get('title')}'...\n")
    
    paths = zettel.find_knowledge_paths(start, end, max_depth=5)
    
    if paths:
        print(f"Found {len(paths)} paths:\n")
        
        for i, path in enumerate(paths[:3], 1):
            print(f"📍 Path {i} ({path.discovery_type}):")
            
            # Build path string with note titles
            path_titles = []
            for note_id in path.path_notes:
                note = zettel.get(note_id)
                if note:
                    path_titles.append(note.metadata.get('title'))
            
            print(f"   {' → '.join(path_titles)}")
            print(f"   Strength: {path.path_strength:.2f}")
            
            if path.insights:
                print(f"   Insights: {', '.join(path.insights)}")
            print()
    else:
        print("No paths found between these notes")


def demo_related_suggestions(zettel: ZettelMemory):
    """Demonstrate related note suggestions."""
    print_section("7. Discovery Engine: Related Note Suggestions")
    
    note_id = 'neural_networks'
    note = zettel.get(note_id)
    
    print(f"Suggestions for: {note.metadata.get('title')}\n")
    
    suggestions = zettel.suggest_related_notes(note_id, suggestion_count=5)
    
    if suggestions:
        print(f"💡 Found {len(suggestions)} related notes:\n")
        
        for i, (suggested_note, score, reason) in enumerate(suggestions, 1):
            print(f"{i}. {suggested_note.metadata.get('title')}")
            print(f"   Relevance: {score:.2f}")
            print(f"   Reason: {reason}\n")
    else:
        print("No suggestions available")


def demo_missing_connections(zettel: ZettelMemory):
    """Demonstrate missing connection discovery."""
    print_section("8. Discovery Engine: Missing Connections")
    
    note_id = 'reinforcement_learning'
    note = zettel.get(note_id)
    
    print(f"Analyzing potential connections for: {note.metadata.get('title')}\n")
    
    missing = zettel.discover_missing_connections(note_id)
    
    if missing:
        print(f"🔗 Suggested connections ({len(missing)} found):\n")
        
        for target_id, strength, reason in missing[:5]:
            target = zettel.get(target_id)
            print(f"→ {target.metadata.get('title')}")
            print(f"  Strength: {strength:.2f}")
            print(f"  Reason: {reason}\n")
    else:
        print("No missing connections detected")


def demo_random_walk(zettel: ZettelMemory):
    """Demonstrate random walk discovery."""
    print_section("9. Discovery Engine: Random Walk Exploration")
    
    start_id = 'ml_basics'
    start_note = zettel.get(start_id)
    
    print(f"Starting random walk from: {start_note.metadata.get('title')}\n")
    
    walk_path = zettel.random_walk_discovery(start_id, walk_length=5)
    
    print("🚶 Random walk path:\n")
    for i, note_id in enumerate(walk_path, 1):
        note = zettel.get(note_id)
        if note:
            arrow = "   ↓" if i < len(walk_path) else ""
            print(f"{i}. {note.metadata.get('title')}{arrow}")
    
    print("\n💭 This path might reveal unexpected connections!")


def demo_system_overview(zettel: ZettelMemory):
    """Demonstrate system overview and analytics."""
    print_section("10. System Overview: Health & Analytics")
    
    overview = zettel.get_zettelkasten_overview()
    
    print("📊 Zettelkasten Statistics:\n")
    print(f"   Total notes: {overview.get('total_notes', 0)}")
    print(f"   Total connections: {overview.get('total_connections', 0)}")
    print(f"   Connection density: {overview.get('connection_density', 0):.4f}")
    print(f"   Knowledge clusters: {overview.get('knowledge_clusters', 0)}")
    print(f"   System health: {overview.get('system_health', 'unknown').upper()}")
    
    # Top clusters
    top_clusters = overview.get('top_clusters', [])
    if top_clusters:
        print("\n🏆 Top Knowledge Clusters:\n")
        for cluster in top_clusters[:3]:
            print(f"   {cluster.get('id')}:")
            print(f"     Size: {cluster.get('size')} notes")
            print(f"     Concepts: {', '.join(cluster.get('concepts', [])[:3])}")
            print(f"     Emergence: {cluster.get('emergence_score', 0):.2f}\n")
    
    # Emerging concepts
    emerging = overview.get('emerging_concepts', {})
    if emerging:
        print("🌱 Top Emerging Concepts:\n")
        for concept, score in list(emerging.items())[:5]:
            print(f"   {concept}: {score:.3f}")


def demo_best_practices():
    """Demonstrate Zettelkasten best practices."""
    print_section("11. Best Practices: Building Your Knowledge Base")
    
    print("""
✅ DO:
   • Keep notes atomic (one idea per note)
   • Use descriptive titles
   • Add rich metadata (tags, concepts)
   • Create bidirectional links
   • Review and refine connections regularly
   • Use discovery features to find gaps
   
❌ DON'T:
   • Create notes with multiple unrelated ideas
   • Leave notes isolated (no connections)
   • Forget to add tags and concepts
   • Ignore suggested connections
   • Let your Zettelkasten stagnate
   
💡 TIPS:
   • Start with 10-20 interconnected notes
   • Use random walks for inspiration
   • Review emerging concepts monthly
   • Create links between distant domains
   • Let structure emerge naturally
    """)


def demo_practical_workflow():
    """Demonstrate a practical Zettelkasten workflow."""
    print_section("12. Practical Workflow: Daily Usage")
    
    zettel = ZettelMemory()
    
    print("📝 Scenario: Learning about a new topic (Transformers)\n")
    
    # Step 1: Create new note
    print("Step 1: Create atomic note")
    new_note = MemoryItem(
        content="""# Transformer Architecture

Transformers use self-attention mechanisms to process sequential data in parallel.
Introduced in 'Attention is All You Need' paper (2017).

Key innovation: eliminates recurrence, enables parallelization.
        """,
        metadata={
            'title': 'Transformer Architecture',
            'tags': ['ai', 'nlp', 'architecture', 'attention'],
            'concepts': ['Attention', 'Self-Attention', 'Parallelization', 'NLP'],
            'source': 'Vaswani et al., 2017'
        },
        item_id='transformers'
    )
    zettel.add(new_note)
    print("  ✅ Note created\n")
    
    # Step 2: Find related notes
    print("Step 2: Discover related notes")
    suggestions = zettel.suggest_related_notes('transformers', suggestion_count=3)
    if suggestions:
        for note, score, reason in suggestions:
            print(f"  💡 {note.metadata.get('title')} (score: {score:.2f})")
    print()
    
    # Step 3: Create connections
    print("Step 3: Create meaningful connections")
    # In a real workflow, you'd link to related notes found in step 2
    print("  🔗 Link to related concepts")
    print("  🔗 Add bidirectional references\n")
    
    # Step 4: Review system health
    print("Step 4: Check system health")
    overview = zettel.get_zettelkasten_overview()
    print(f"  📊 System health: {overview.get('system_health', 'unknown')}")
    print(f"  📊 Total notes: {overview.get('total_notes', 0)}")
    print(f"  📊 Total connections: {overview.get('total_connections', 0)}\n")
    
    print("✨ Workflow complete! Repeat daily for best results.")


def main():
    """Run all demonstrations."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║           SmartMemory Zettelkasten System - Complete Demo           ║
║                                                                      ║
║  This demonstration showcases all features of the Zettelkasten      ║
║  system including bidirectional linking, knowledge discovery,       ║
║  cluster detection, and AI-powered suggestions.                     ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run demonstrations
    zettel = demo_basic_operations()
    demo_bidirectional_navigation(zettel)
    demo_knowledge_clusters(zettel)
    demo_knowledge_bridges(zettel)
    demo_concept_emergence(zettel)
    demo_knowledge_paths(zettel)
    demo_related_suggestions(zettel)
    demo_missing_connections(zettel)
    demo_random_walk(zettel)
    demo_system_overview(zettel)
    demo_best_practices()
    demo_practical_workflow()
    
    print_section("Demo Complete!")
    print("""
🎉 You've seen all the features of SmartMemory's Zettelkasten system!

Next steps:
  1. Read the full documentation: docs/ZETTELKASTEN.md
  2. Build your own knowledge base
  3. Experiment with discovery features
  4. Integrate with your workflow

Happy note-taking! 📝
    """)


if __name__ == "__main__":
    main()
