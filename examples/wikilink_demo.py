"""
Wikilink Parser Demo - Automatic Bidirectional Linking

This example demonstrates SmartMemory's automatic wikilink parsing feature
for the Zettelkasten system.

Supported syntax:
- [[Note Title]] - Wikilink to another note
- [[Note Title|Alias]] - Wikilink with display alias
- ((Concept)) - Concept mention
- #tag - Hashtag

Run this example:
    PYTHONPATH=. python examples/wikilink_demo.py
"""

from smartmemory.memory.types.zettel_memory import ZettelMemory
from smartmemory.models.memory_item import MemoryItem


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def demo_basic_wikilinks():
    """Demonstrate basic wikilink parsing and auto-linking."""
    print_section("1. Basic Wikilinks: Automatic Bidirectional Linking")
    
    # Initialize Zettelkasten with auto-linking enabled (default)
    zettel = ZettelMemory(auto_link=True)
    
    print("Creating notes with wikilink syntax...\n")
    
    # Create first note
    note1 = MemoryItem(
        content="""# Machine Learning

Machine learning is a subset of [[Artificial Intelligence]] that enables 
systems to learn from data. It uses ((Algorithms)) and ((Statistical Methods))
to improve performance.

Related topics: #ai #data-science #algorithms
        """,
        metadata={'title': 'Machine Learning'},
        item_id='machine_learning'
    )
    
    # Create second note (referenced by first)
    note2 = MemoryItem(
        content="""# Artificial Intelligence

AI is the simulation of human intelligence by machines. It encompasses
[[Machine Learning]], [[Deep Learning]], and other techniques.

Key concepts: ((Reasoning)), ((Problem Solving)), ((Learning))

Tags: #ai #computer-science
        """,
        metadata={'title': 'Artificial Intelligence'},
        item_id='artificial_intelligence'
    )
    
    # Add notes - wikilinks will be automatically parsed and linked
    zettel.add(note1)
    zettel.add(note2)
    
    print("✅ Notes added with automatic wikilink parsing\n")
    
    # Verify bidirectional links were created
    print("Checking bidirectional connections...\n")
    
    connections1 = zettel.get_bidirectional_connections('machine_learning')
    print("Machine Learning:")
    print(f"  Forward links: {len(connections1['forward_links'])}")
    print(f"  Backlinks: {len(connections1['backlinks'])}")
    print(f"  Tags: {connections1['forward_links'][0].metadata.get('tags') if connections1['forward_links'] else []}")
    print(f"  Concepts: {connections1['forward_links'][0].metadata.get('concepts') if connections1['forward_links'] else []}")
    
    connections2 = zettel.get_bidirectional_connections('artificial_intelligence')
    print("\nArtificial Intelligence:")
    print(f"  Forward links: {len(connections2['forward_links'])}")
    print(f"  Backlinks: {len(connections2['backlinks'])}")
    
    print("\n🎉 Bidirectional links created automatically from wikilink syntax!")
    
    return zettel


def demo_wikilink_parsing():
    """Demonstrate wikilink parsing without adding to graph."""
    print_section("2. Wikilink Parsing: Extract Links from Content")
    
    zettel = ZettelMemory()
    
    content = """# Deep Learning

Deep learning uses [[Neural Networks]] with multiple layers. It's a subset
of [[Machine Learning]] that has revolutionized [[Computer Vision]] and
[[Natural Language Processing]].

Key concepts: ((Backpropagation)), ((Gradient Descent)), ((Convolutional Networks))

Applications include #image-recognition #nlp #speech-recognition
    """
    
    print("Content to parse:")
    print("-" * 70)
    print(content)
    print("-" * 70)
    
    # Parse wikilinks
    parsed = zettel.parse_wikilinks(content)
    
    print("\nParsed links:\n")
    print(f"📝 Wikilinks ({len(parsed['wikilinks'])}):")
    for link in parsed['wikilinks']:
        print(f"   → [[{link}]]")
    
    print(f"\n💡 Concepts ({len(parsed['concepts'])}):")
    for concept in parsed['concepts']:
        print(f"   → (({concept}))")
    
    print(f"\n🏷️  Hashtags ({len(parsed['hashtags'])}):")
    for tag in parsed['hashtags']:
        print(f"   → #{tag}")


def demo_wikilink_aliases():
    """Demonstrate wikilink aliases."""
    print_section("3. Wikilink Aliases: Display Text vs. Link Target")
    
    zettel = ZettelMemory()
    
    content = """# Research Notes

I'm studying [[Machine Learning|ML]] and [[Artificial Intelligence|AI]].
The [[Neural Networks|NN architecture]] is fascinating.

See also: [[Deep Learning|DL techniques]]
    """
    
    print("Content with aliases:")
    print("-" * 70)
    print(content)
    print("-" * 70)
    
    parsed = zettel.parse_wikilinks(content)
    
    print("\nExtracted note references (aliases removed):\n")
    for link in parsed['wikilinks']:
        print(f"   → {link}")
    
    print("\n💡 The display text (after |) is ignored - links point to actual note titles")


def demo_knowledge_graph_building():
    """Demonstrate building an interconnected knowledge graph with wikilinks."""
    print_section("4. Building a Knowledge Graph with Wikilinks")
    
    zettel = ZettelMemory(auto_link=True)
    
    print("Creating an interconnected knowledge base...\n")
    
    notes = [
        MemoryItem(
            content="""# Python Programming

Python is a high-level programming language. It's widely used in
[[Data Science]], [[Machine Learning]], and [[Web Development]].

Key features: ((Readability)), ((Simplicity)), ((Versatility))

#programming #python #language
            """,
            metadata={'title': 'Python Programming'},
            item_id='python'
        ),
        MemoryItem(
            content="""# Data Science

Data science combines [[Statistics]], [[Programming]], and domain knowledge.
Common tools include [[Python Programming|Python]], R, and SQL.

Core skills: ((Data Analysis)), ((Visualization)), ((Machine Learning))

#data-science #analytics #statistics
            """,
            metadata={'title': 'Data Science'},
            item_id='data_science'
        ),
        MemoryItem(
            content="""# Machine Learning

ML is a key component of [[Data Science]] and [[Artificial Intelligence]].
Often implemented using [[Python Programming|Python]] libraries.

Techniques: ((Supervised Learning)), ((Unsupervised Learning)), ((Deep Learning))

#machine-learning #ai #algorithms
            """,
            metadata={'title': 'Machine Learning'},
            item_id='machine_learning'
        ),
        MemoryItem(
            content="""# Web Development

Building web applications using various technologies. [[Python Programming|Python]]
frameworks like Django and Flask are popular choices.

Concepts: ((Frontend)), ((Backend)), ((APIs)), ((Databases))

#web-development #programming #fullstack
            """,
            metadata={'title': 'Web Development'},
            item_id='web_dev'
        )
    ]
    
    # Add all notes
    for note in notes:
        zettel.add(note)
        print(f"  ✅ Added: {note.metadata['title']}")
    
    print("\n📊 Knowledge graph statistics:\n")
    
    overview = zettel.get_zettelkasten_overview()
    print(f"   Total notes: {overview['total_notes']}")
    print(f"   Total connections: {overview['total_connections']}")
    print(f"   Auto-linking: {'✅ Enabled' if overview['auto_linking_enabled'] else '❌ Disabled'}")
    
    # Show connections for Python
    print("\n🔗 Connections for 'Python Programming':\n")
    connections = zettel.get_bidirectional_connections('python')
    
    print(f"   Forward links ({len(connections['forward_links'])}):")
    for note in connections['forward_links']:
        print(f"     → {note.metadata.get('title')}")
    
    print(f"\n   Backlinks ({len(connections['backlinks'])}):")
    for note in connections['backlinks']:
        print(f"     ← {note.metadata.get('title')}")
    
    print(f"\n   Tag-related ({len(connections['related_by_tags'])}):")
    for note in connections['related_by_tags'][:3]:
        print(f"     # {note.metadata.get('title')}")


def demo_manual_control():
    """Demonstrate manual control over auto-linking."""
    print_section("5. Manual Control: Enable/Disable Auto-Linking")
    
    # Start with auto-linking disabled
    zettel = ZettelMemory(auto_link=False)
    
    print("Auto-linking disabled by default\n")
    
    note1 = MemoryItem(
        content="This note mentions [[Another Note]] but won't auto-link.",
        metadata={'title': 'First Note'},
        item_id='note1'
    )
    
    zettel.add(note1)
    print("✅ Added note without auto-linking")
    
    # Check metadata
    retrieved = zettel.get('note1')
    print(f"   Wikilinks in metadata: {retrieved.metadata.get('wikilinks', 'None')}")
    
    # Enable auto-linking
    print("\nEnabling auto-linking...")
    zettel.enable_auto_linking()
    
    note2 = MemoryItem(
        content="This note mentions [[First Note]] and WILL auto-link.",
        metadata={'title': 'Second Note'},
        item_id='note2'
    )
    
    zettel.add(note2)
    print("✅ Added note with auto-linking enabled")
    
    # Check metadata
    retrieved2 = zettel.get('note2')
    print(f"   Wikilinks in metadata: {retrieved2.metadata.get('wikilinks', [])}")
    
    # Disable again
    print("\nDisabling auto-linking...")
    zettel.disable_auto_linking()
    
    print("\n💡 You can toggle auto-linking on/off as needed!")


def demo_complex_example():
    """Demonstrate a complex real-world example."""
    print_section("6. Real-World Example: Research Notes")
    
    zettel = ZettelMemory(auto_link=True)
    
    print("Building a research knowledge base...\n")
    
    research_note = MemoryItem(
        content="""# Transformer Architecture Research

## Overview
The Transformer architecture, introduced in [[Attention Is All You Need]], 
revolutionized [[Natural Language Processing]]. It relies entirely on 
((Self-Attention)) mechanisms, eliminating the need for recurrence.

## Key Components
- **Encoder-Decoder Structure**: Uses stacked layers
- **Multi-Head Attention**: Parallel ((Attention Mechanisms))
- **Positional Encoding**: Injects sequence information

## Applications
Transformers power modern models like:
- [[BERT]] - Bidirectional encoding
- [[GPT]] - Generative pre-training
- [[T5]] - Text-to-text framework

## Related Concepts
The architecture builds on earlier work in [[Neural Networks]] and 
[[Deep Learning]], particularly ((Attention Mechanisms)) from 
[[Sequence-to-Sequence Models]].

## Impact
Transformers enabled breakthrough performance in:
- [[Machine Translation]]
- [[Question Answering]]
- [[Text Summarization]]

Key innovations: ((Parallelization)), ((Scalability)), ((Transfer Learning))

#transformers #nlp #deep-learning #attention #research
        """,
        metadata={
            'title': 'Transformer Architecture Research',
            'author': 'Research Team',
            'date': '2025-10-04',
            'source': 'Vaswani et al., 2017'
        },
        item_id='transformer_research'
    )
    
    zettel.add(research_note)
    
    print("✅ Added comprehensive research note\n")
    
    # Parse and display what was extracted
    retrieved = zettel.get('transformer_research')
    
    print("📊 Automatically extracted:\n")
    print(f"   Wikilinks: {len(retrieved.metadata.get('wikilinks', []))}")
    for link in retrieved.metadata.get('wikilinks', [])[:5]:
        print(f"     → [[{link}]]")
    if len(retrieved.metadata.get('wikilinks', [])) > 5:
        print(f"     ... and {len(retrieved.metadata.get('wikilinks', [])) - 5} more")
    
    print(f"\n   Concepts: {len(retrieved.metadata.get('concepts', []))}")
    for concept in retrieved.metadata.get('concepts', [])[:5]:
        print(f"     → (({concept}))")
    
    print(f"\n   Tags: {len(retrieved.metadata.get('tags', []))}")
    for tag in retrieved.metadata.get('tags', []):
        print(f"     → #{tag}")
    
    print("\n💡 All links, concepts, and tags extracted automatically!")


def demo_best_practices():
    """Demonstrate best practices for using wikilinks."""
    print_section("7. Best Practices for Wikilinks")
    
    print("""
✅ DO:
   • Use descriptive note titles that match wikilink text
   • Be consistent with note naming (e.g., "Machine Learning" not "ML")
   • Use aliases [[Full Name|Short]] when needed for readability
   • Add notes in logical order (referenced notes first when possible)
   • Use concepts (()) for ideas that span multiple notes
   • Use hashtags for categorization and filtering

❌ DON'T:
   • Link to notes that don't exist (creates broken links)
   • Use inconsistent capitalization in wikilinks
   • Overuse wikilinks (link only meaningful connections)
   • Forget to add the target note eventually
   • Mix different naming conventions

💡 TIPS:
   • Create a note immediately when you reference it
   • Use the same title in metadata and wikilinks
   • Review wikilinks metadata to verify parsing
   • Use parse_wikilinks() to preview before adding
   • Keep a consistent naming scheme across your knowledge base
    """)


def main():
    """Run all demonstrations."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║              Wikilink Parser - Automatic Linking Demo               ║
║                                                                      ║
║  This demonstration shows how SmartMemory automatically parses      ║
║  wikilinks, concepts, and hashtags to create bidirectional          ║
║  connections in your Zettelkasten.                                  ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run demonstrations
    demo_basic_wikilinks()
    demo_wikilink_parsing()
    demo_wikilink_aliases()
    demo_knowledge_graph_building()
    demo_manual_control()
    demo_complex_example()
    demo_best_practices()
    
    print_section("Demo Complete!")
    print("""
🎉 You've learned how to use wikilinks in SmartMemory!

Supported syntax:
  [[Note Title]]        → Link to another note
  [[Note|Alias]]        → Link with display alias
  ((Concept))           → Concept mention
  #tag                  → Hashtag

Next steps:
  1. Build your knowledge base with wikilinks
  2. Let SmartMemory create bidirectional connections automatically
  3. Use discovery features to explore your knowledge graph
  4. See docs/ZETTELKASTEN.md for complete documentation

Happy linking! 🔗
    """)


if __name__ == "__main__":
    main()
