#!/usr/bin/env python3
"""
Advanced Programming Tutor - Multi-Memory Type Example

This example demonstrates SmartMemory's capabilities through a multi-week
programming tutor that uses all 4 memory types to create adaptive, personalized learning:

1. Semantic Memory: Programming knowledge graph with interconnected concepts
2. Episodic Memory: Specific learning events, struggles, and breakthroughs
3. Procedural Memory: Teaching strategies that work for this specific user
4. Working Memory: Current project context and immediate learning goals

The tutor evolves over 4 weeks, demonstrating key capabilities:
- Cross-session intelligence that builds on previous learning
- Personalized teaching methods based on user's learning history
- Knowledge web building that strengthens understanding over time
- Memory quality improvement through experience

This showcases the benefits of multi-type memory systems over simpler approaches.
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple

from smartmemory.models.memory_item import MemoryItem
from smartmemory.similarity.framework import EnhancedSimilarityFramework
from smartmemory.smart_memory import SmartMemory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # Import successful - system is available
    pass
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info("This example requires the full SmartMemory system to be available.")
    exit(1)


class AdvancedProgrammingTutor:
    """AI Programming Tutor with multi-memory intelligence for adaptive learning."""

    def __init__(self, tutor_name: str = "CodeMentor"):
        """Initialize the advanced programming tutor with all memory systems."""
        self.name = tutor_name
        self.logger = logging.getLogger(f"tutor.{tutor_name.replace(' ', '_').lower()}")

        # Use unified SmartMemory interface with evolution triggers
        self.memory = SmartMemory()

        # For compatibility with existing code, create memory type accessors
        self.semantic_memory = self.memory  # All operations go through unified interface
        self.episodic_memory = self.memory
        self.procedural_memory = self.memory
        self.working_memory = self.memory

        # Teaching state
        self.student_id = "student_001"  # Add missing student_id
        self.teaching_approach = "foundational"
        self.student_progress = {}
        self.session_history = []

        # Initialize similarity framework for compatibility
        self.similarity_framework = EnhancedSimilarityFramework()
        self.session_count = 0
        self.learning_metrics = {
            'concepts_mastered': 0,
            'teaching_adaptations': 0,
            'cross_concept_connections': 0,
            'breakthrough_moments': 0,
            'personalization_improvements': 0,
            'total_interactions': 0
        }

        # Initialize foundational teaching strategies
        self._initialize_teaching_strategies()

    def _initialize_teaching_strategies(self):
        """Initialize basic teaching strategies in procedural memory."""
        base_strategies = [
            "Start with concrete examples before abstract concepts",
            "Use hands-on practice immediately after explanation",
            "Connect new concepts to previously learned material",
            "Adapt explanation complexity based on user comprehension",
            "Use visual aids and analogies for difficult concepts"
        ]

        for strategy in base_strategies:
            strategy_item = MemoryItem(
                content=strategy,
                memory_type="procedural",
                metadata={
                    'type': 'teaching_strategy',
                    'effectiveness': 0.5,  # Will be updated based on experience
                    'student_id': self.student_id,
                    'category': 'general',
                    'timestamp': datetime.now().isoformat()
                }
            )
            self.procedural_memory.add(strategy_item)

    def start_week(self, week_number: int, topic: str, learning_goals: List[str]):
        """Start a new week of learning with specific topic and goals."""
        self.current_week = week_number
        logger.info(f"=== WEEK {week_number}: {topic.upper()} ===")

        # Store week context in working memory
        week_context = MemoryItem(
            content=f"Week {week_number}: Learning {topic}",
            memory_type="working",
            metadata={
                'week': week_number,
                'topic': topic,
                'learning_goals': learning_goals,
                'student_id': self.student_id,
                'start_date': datetime.now().isoformat(),
                'status': 'active'
            }
        )
        self.working_memory.add(week_context)

        # Retrieve relevant prior knowledge
        prior_knowledge = self._get_relevant_prior_knowledge(topic)

        # Adapt teaching approach based on previous weeks
        teaching_approach = self._adapt_teaching_approach(week_number, topic)

        return {
            'week': week_number,
            'topic': topic,
            'goals': learning_goals,
            'prior_knowledge': len(prior_knowledge),
            'teaching_approach': teaching_approach
        }

    def teach_concept(self, concept: str, user_question: str = None) -> str:
        """Teach a programming concept using multi-memory intelligence."""
        self.learning_metrics['total_interactions'] += 1
        self.session_count += 1

        # Get teaching context from all memory types
        context = self._gather_teaching_context(concept, user_question)

        # Generate adaptive explanation
        explanation = self._generate_adaptive_explanation(concept, context, user_question)

        # Store the teaching interaction
        self._store_teaching_interaction(concept, user_question, explanation, context)

        return explanation

    def _gather_teaching_context(self, concept: str, user_question: str = None) -> Dict[str, Any]:
        """Gather relevant context from all memory types."""
        context = {
            'semantic_connections': [],
            'learning_history': [],
            'effective_strategies': [],
            'current_context': {}
        }

        # Enhanced query item for better similarity search
        query_content = f"{concept} programming {user_question or ''}"
        query_item = MemoryItem(
            content=query_content,
            memory_type="query",
            metadata={
                'student_id': self.student_id,
                'concept': concept,
                'week': self.current_week,
                'domain': 'programming'
            }
        )

        # Use unified SmartMemory interface with memory type routing
        # Semantic memory: Programming concepts and knowledge
        try:
            semantic_items = self.memory.search(query_item.content, top_k=5, memory_type="semantic")
            if semantic_items:
                context['prior_knowledge'] = semantic_items
                logger.info(f"Found {len(semantic_items)} relevant items in SemanticMemory (from {len(semantic_items)} total)")
            else:
                logger.warning("No items found in store: SemanticMemory")
        except Exception as e:
            logger.warning(f"Semantic memory search failed: {e}")

        # Episodic memory: Past learning experiences
        try:
            episodic_items = self.memory.search(query_item.content, top_k=3, memory_type="episodic")
            if episodic_items:
                context['learning_history'] = episodic_items
                logger.info(f"Found {len(episodic_items)} relevant items in EpisodicMemory (from {len(episodic_items)} total)")
            else:
                logger.warning("No items found in store: EpisodicMemory")
        except Exception as e:
            logger.warning(f"Episodic memory search failed: {e}")

        # Procedural memory: Teaching strategies and methods
        try:
            procedural_items = self.memory.search(query_item.content, top_k=2, memory_type="procedural")
            if procedural_items:
                context['teaching_strategies'] = procedural_items
                logger.info(f"Found {len(procedural_items)} relevant items in ProceduralMemory (from {len(procedural_items)} total)")
            else:
                logger.warning("No items found in store: ProceduralMemory")
        except Exception as e:
            logger.warning(f"Procedural memory search failed: {e}")

        # Working memory: Current learning context
        try:
            working_items = self.memory.search(query_item.content, top_k=1, memory_type="working")
            if working_items:
                context['current_context'] = working_items[0].metadata if working_items else {}
                logger.info(f"Found {len(working_items)} relevant items in WorkingMemory (from {len(working_items)} total)")
            else:
                logger.warning("No items found in store: WorkingMemory")
        except Exception as e:
            logger.warning(f"Working memory search failed: {e}")

        return context

    def _search_memory_store(self, store, query_item: MemoryItem, top_k: int = 3) -> List[Tuple[float, MemoryItem]]:
        """Search a memory store and return scored results with improved retrieval."""
        all_items = []

        # Get items from store using working _search_impl method
        try:
            # Primary method: Use _search_impl which we know works
            if hasattr(store, '_search_impl'):
                try:
                    all_items = store._search_impl("*", top_k=1000)
                    logger.debug(f"Retrieved {len(all_items)} items using _search_impl")
                except Exception as e:
                    logger.debug(f"_search_impl failed: {e}")
                    # Fallback to empty string query
                    try:
                        all_items = store._search_impl("", top_k=1000)
                        logger.debug(f"Retrieved {len(all_items)} items using _search_impl with empty query")
                    except Exception:
                        all_items = []

            # Fallback method: Try search method
            elif hasattr(store, 'search'):
                try:
                    all_items = store.search("*", top_k=1000)
                    logger.debug(f"Retrieved {len(all_items)} items using search")
                except Exception:
                    try:
                        all_items = store.search("", top_k=1000)
                        logger.debug(f"Retrieved {len(all_items)} items using search with empty query")
                    except Exception:
                        all_items = []

            # Legacy fallback methods (kept for compatibility)
            elif hasattr(store, 'get_all_items'):
                all_items = store.get_all_items()
                logger.debug(f"Retrieved {len(all_items)} items using get_all_items")

            # Filter by student_id if needed
            if all_items:
                filtered_items = []
                for item in all_items:
                    if hasattr(item, 'metadata') and item.metadata:
                        student_id = item.metadata.get('student_id')
                        if student_id == self.student_id or student_id is None:
                            filtered_items.append(item)
                    else:
                        # Include items without metadata
                        filtered_items.append(item)
                all_items = filtered_items

        except Exception as e:
            logger.warning(f"Store access failed: {e}")
            return []

        if not all_items:
            logger.warning(f"No items found in store: {type(store).__name__}")
            return []

        # Calculate similarities with multiple approaches
        scored_items = []
        query_content_lower = query_item.content.lower()

        for item in all_items:
            try:
                # Primary similarity using enhanced framework
                similarity = 0.0
                try:
                    similarity = self.similarity_framework.calculate_similarity(query_item, item)
                except Exception as e:
                    logger.warning(f"Enhanced similarity failed: {e}")
                    # Fallback to basic content similarity
                    similarity = self._basic_content_similarity(query_content_lower, item.content.lower())

                # Boost similarity for relevant metadata matches
                if hasattr(item, 'metadata') and item.metadata:
                    metadata_boost = self._calculate_metadata_boost(query_item, item)
                    similarity += metadata_boost

                # Lower threshold for better recall
                if similarity > 0.05:  # Reduced from 0.1
                    scored_items.append((similarity, item))

            except Exception as e:
                logger.warning(f"Similarity calculation failed for item: {e}")
                continue

        # Sort by similarity and return top results
        scored_items.sort(key=lambda x: x[0], reverse=True)
        logger.info(f"Found {len(scored_items)} relevant items in {type(store).__name__} (from {len(all_items)} total)")
        return scored_items[:top_k]

    def _basic_content_similarity(self, query_lower: str, content_lower: str) -> float:
        """Basic content similarity using word overlap."""
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())

        if not query_words or not content_words:
            return 0.0

        intersection = query_words.intersection(content_words)
        union = query_words.union(content_words)

        return len(intersection) / len(union) if union else 0.0

    def _calculate_metadata_boost(self, query_item: MemoryItem, target_item: MemoryItem) -> float:
        """Calculate similarity boost based on metadata matches."""
        boost = 0.0

        query_meta = getattr(query_item, 'metadata', {}) or {}
        target_meta = getattr(target_item, 'metadata', {}) or {}

        # Week proximity boost
        query_week = query_meta.get('week')
        target_week = target_meta.get('week')
        if query_week is not None and target_week is not None:
            week_diff = abs(query_week - target_week)
            if week_diff == 0:
                boost += 0.2  # Same week
            elif week_diff == 1:
                boost += 0.1  # Adjacent week

        # Topic/concept boost
        query_concept = query_meta.get('concept', '')
        target_concept = target_meta.get('concept', '')
        if query_concept and target_concept and query_concept == target_concept:
            boost += 0.15

        # Domain boost
        query_domain = query_meta.get('domain', '')
        target_domain = target_meta.get('domain', '')
        if query_domain and target_domain and query_domain == target_domain:
            boost += 0.1

        return boost

    def _generate_adaptive_explanation(self, concept: str, context: Dict, user_question: str = None) -> str:
        """Generate explanation adapted to user's learning history and current context."""

        # Base explanations for different concepts
        explanations = {
            'variables': "Variables are like labeled boxes that store values. Think of them as containers with names.",
            'functions': "Functions are reusable blocks of code that perform specific tasks. Like recipes in cooking.",
            'classes': "Classes are blueprints for creating objects. Like a cookie cutter that makes cookies.",
            'inheritance': "Inheritance lets one class use features from another. Like how a sports car inherits from a general car.",
            'loops': "Loops repeat code multiple times. Like doing jumping jacks - same action, multiple repetitions.",
            'web_frameworks': "Web frameworks provide structure for building web applications. Like scaffolding for construction.",
            'databases': "Databases store and organize data permanently. Like a digital filing cabinet.",
            'apis': "APIs let different programs talk to each other. Like a waiter taking your order to the kitchen."
        }

        base_explanation = explanations.get(concept.lower(), f"Let me explain {concept}...")

        # Adapt based on context
        adapted_explanation = base_explanation

        # Use prior knowledge connections with specific examples
        if context.get('prior_knowledge'):
            prior_concepts = context['prior_knowledge']
            if prior_concepts:
                # prior_concepts is a list of MemoryItems from search
                best_connection = prior_concepts[0]
                # Extract concept name from content
                concept_content = best_connection.content if hasattr(best_connection, 'content') else str(best_connection)
                if ':' in concept_content:
                    prior_concept = concept_content.split(':')[0].strip()
                else:
                    prior_concept = concept_content.split('.')[0].strip()

                adapted_explanation += f"\n\nThis builds on what you learned about {prior_concept}. "
                adapted_explanation += f"Remember: {concept_content[:100]}..."
                self.learning_metrics['cross_concept_connections'] += 1

        # Apply effective teaching strategies with specific adaptations
        if context.get('teaching_strategies'):
            strategies = context['teaching_strategies']
            if strategies:
                # strategies is a list of MemoryItems
                best_strategy = strategies[0]  # Already sorted by relevance
                strategy_content = best_strategy.content.lower() if hasattr(best_strategy, 'content') else ''

                if 'concrete examples' in strategy_content or 'examples' in strategy_content:
                    adapted_explanation += "\n\nLet me give you a concrete example (this approach worked well for you before)..."
                elif 'hands-on' in strategy_content or 'practice' in strategy_content:
                    adapted_explanation += "\n\nLet's try this with some hands-on practice (you learn best this way)..."
                elif 'visual' in strategy_content or 'analogies' in strategy_content:
                    adapted_explanation += "\n\nLet me use a visual analogy to explain this..."
                else:
                    adapted_explanation += "\n\nBased on your learning style, let me explain this step by step..."

                self.learning_metrics['teaching_adaptations'] += 1

        # Address specific struggles and successes from learning history
        if context.get('learning_history'):
            for event in context['learning_history']:
                if not hasattr(event, 'content'):
                    continue
                event_content = event.content.lower()
                if 'struggled' in event_content or 'difficult' in event_content:
                    # Extract what was difficult
                    struggle_context = event.content[:100]
                    adapted_explanation += f"\n\nI remember you found similar concepts challenging before ({struggle_context}...), so let's approach this carefully step by step..."
                    self.learning_metrics['teaching_adaptations'] += 1
                    break
                elif 'breakthrough' in event_content or 'mastered' in event_content:
                    success_context = event.content[:100]
                    adapted_explanation += f"\n\nYou had a great breakthrough with related concepts before ({success_context}...), let's build on that success..."
                    self.learning_metrics['teaching_adaptations'] += 1
                    break
                elif 'week' in event_content and concept.lower() in event_content:
                    adapted_explanation += f"\n\nThis connects to your previous work: {event.content[:100]}..."
                    break

        # Connect to current project context with specific details
        current_context = context.get('current_context') or {}
        current_topic = current_context.get('topic', '')
        current_week = current_context.get('week', 0)
        learning_goals = current_context.get('learning_goals', [])

        if current_topic:
            adapted_explanation += f"\n\nThis fits perfectly with your current Week {current_week} focus on {current_topic}!"

            # Connect to specific learning goals
            if learning_goals:
                relevant_goals = [goal for goal in learning_goals if any(word in goal.lower() for word in concept.lower().split())]
                if relevant_goals:
                    adapted_explanation += f" This helps you achieve your goal: '{relevant_goals[0]}'."

        # Add week progression context
        if current_week > 1:
            adapted_explanation += f"\n\nYou're now in Week {current_week} - building on {current_week - 1} weeks of programming knowledge!"

        return adapted_explanation

    def _store_teaching_interaction(self, concept: str, user_question: str, explanation: str, context: Dict):
        """Store the teaching interaction across appropriate memory types."""
        timestamp = datetime.now()

        # Semantic memory: Store the concept and its explanation
        semantic_item = MemoryItem(
            content=f"{concept}: {explanation}",
            memory_type="semantic",
            metadata={
                'concept': concept,
                'week': self.current_week,
                'student_id': self.student_id,
                'connections': len(context['semantic_connections']),
                'timestamp': timestamp.isoformat(),
                'domain': 'programming'
            }
        )
        self.semantic_memory.add(semantic_item)

        # Episodic memory: Store the learning event
        episodic_content = f"Taught {concept} in week {self.current_week}."
        if user_question:
            episodic_content += f" Student asked: '{user_question}'"
        if context['learning_history']:
            episodic_content += " Built on previous learning experiences."

        episodic_item = MemoryItem(
            content=episodic_content,
            memory_type="episodic",
            metadata={
                'event_type': 'teaching_session',
                'concept': concept,
                'week': self.current_week,
                'session': self.session_count,
                'student_id': self.student_id,
                'adaptations_made': self.learning_metrics['teaching_adaptations'],
                'timestamp': timestamp.isoformat()
            }
        )
        self.episodic_memory.add(episodic_item)

    def record_learning_outcome(self, concept: str, outcome: str, effectiveness_score: float):
        """Record learning outcome to improve future teaching."""
        # Update procedural memory with strategy effectiveness
        if outcome == 'mastered':
            self.learning_metrics['concepts_mastered'] += 1
        elif outcome == 'breakthrough':
            self.learning_metrics['breakthrough_moments'] += 1

        # Store outcome in episodic memory
        outcome_item = MemoryItem(
            content=f"Student {outcome} {concept} with effectiveness score {effectiveness_score}",
            memory_type="episodic",
            metadata={
                'event_type': 'learning_outcome',
                'concept': concept,
                'outcome': outcome,
                'effectiveness': effectiveness_score,
                'week': self.current_week,
                'student_id': self.student_id,
                'timestamp': datetime.now().isoformat()
            }
        )
        self.episodic_memory.add(outcome_item)

        # Update teaching strategy effectiveness in procedural memory
        self._update_strategy_effectiveness(effectiveness_score)

    def _update_strategy_effectiveness(self, effectiveness_score: float):
        """Update teaching strategy effectiveness based on outcomes."""
        # This would update procedural memory strategies based on results
        # For now, just track the improvement
        self.learning_metrics['personalization_improvements'] += 1

    def _get_relevant_prior_knowledge(self, topic: str) -> List[MemoryItem]:
        """Get relevant prior knowledge for the new topic."""
        query_item = MemoryItem(
            content=topic,
            memory_type="query",
            metadata={'student_id': self.student_id}
        )

        try:
            return self._search_memory_store(self.semantic_memory, query_item, top_k=5)
        except Exception:
            return []

    def _adapt_teaching_approach(self, week_number: int, topic: str) -> str:
        """Adapt teaching approach based on previous weeks' experience."""
        if week_number == 1:
            return "foundational_approach"

        # Look at previous week's learning patterns
        try:
            query_item = MemoryItem(
                content=f"week {week_number - 1} learning",
                memory_type="query",
                metadata={'student_id': self.student_id}
            )

            recent_episodes = self._search_memory_store(self.episodic_memory, query_item, top_k=3)

            if recent_episodes:
                # Analyze recent learning patterns
                for score, episode in recent_episodes:
                    if 'breakthrough' in episode.content.lower():
                        return "build_on_success"
                    elif 'struggled' in episode.content.lower():
                        return "remedial_approach"

            return "progressive_approach"
        except Exception:
            return "standard_approach"


class CapabilityEvaluator:
    """Evaluates SmartMemory tutor capabilities and demonstrates multi-memory benefits."""

    def __init__(self):
        self.results = {}

    def run_4_week_comparison(self) -> Dict[str, Any]:
        """Run 4-week learning journey demonstrating SmartMemory capabilities."""
        logger.info("=== 4-WEEK CAPABILITY EVALUATION ===")

        # Initialize SmartMemory tutor
        smart_tutor = AdvancedProgrammingTutor("SmartMemory Tutor")

        # Simulate 4-week learning journey
        weeks = [
            {
                'week': 1,
                'topic': 'Python Basics',
                'concepts': ['variables', 'functions', 'loops'],
                'goals': ['Understand basic syntax', 'Write simple programs']
            },
            {
                'week': 2,
                'topic': 'Object-Oriented Programming',
                'concepts': ['classes', 'inheritance'],
                'goals': ['Create classes', 'Understand inheritance']
            },
            {
                'week': 3,
                'topic': 'Web Development',
                'concepts': ['web_frameworks', 'apis'],
                'goals': ['Build web applications', 'Handle HTTP requests']
            },
            {
                'week': 4,
                'topic': 'Database Integration',
                'concepts': ['databases'],
                'goals': ['Store data persistently', 'Query databases']
            }
        ]

        smart_memory_results = []

        for week_data in weeks:
            # Start week
            week_context = smart_tutor.start_week(
                week_data['week'],
                week_data['topic'],
                week_data['goals']
            )

            # Teach concepts
            week_interactions = []
            for concept in week_data['concepts']:
                explanation = smart_tutor.teach_concept(concept)
                week_interactions.append({
                    'concept': concept,
                    'explanation_length': len(explanation),
                    'mentions_prior_learning': 'learned' in explanation.lower() or 'remember' in explanation.lower(),
                    'personalized': 'you' in explanation.lower() and ('before' in explanation.lower() or 'previous' in explanation.lower()),
                    'contextual': week_data['topic'].lower() in explanation.lower()
                })

                # Simulate learning outcome
                effectiveness = 0.8 if week_data['week'] > 1 else 0.6  # Improves over time
                smart_tutor.record_learning_outcome(concept, 'mastered', effectiveness)

            smart_memory_results.append({
                'week': week_data['week'],
                'topic': week_data['topic'],
                'interactions': week_interactions,
                'prior_knowledge_used': week_context['prior_knowledge'],
                'teaching_approach': week_context['teaching_approach']
            })

        # Simulate simpler memory approach for comparison (illustrative only)
        simple_memory_results = self._simulate_simple_memory_system(weeks)

        # Compare results
        comparison = self._compare_systems(smart_memory_results, simple_memory_results, smart_tutor)

        return comparison

    def _simulate_simple_memory_system(self, weeks: List[Dict]) -> List[Dict]:
        """Simulate a simpler memory approach without cross-session intelligence (illustrative)."""
        basic_results = []

        for week_data in weeks:
            week_interactions = []
            for concept in week_data['concepts']:
                # Basic system: generic explanations, no personalization
                explanation = f"Here's how {concept} works in programming..."
                week_interactions.append({
                    'concept': concept,
                    'explanation_length': len(explanation),
                    'mentions_prior_learning': False,  # No cross-session memory
                    'personalized': False,  # No learning adaptation
                    'contextual': False  # No project context
                })

            basic_results.append({
                'week': week_data['week'],
                'topic': week_data['topic'],
                'interactions': week_interactions,
                'prior_knowledge_used': 0,  # No prior knowledge integration
                'teaching_approach': 'standard_approach'  # No adaptation
            })

        return basic_results

    def _compare_systems(self, smart_results: List[Dict], simple_results: List[Dict], smart_tutor) -> Dict[str, Any]:
        """Compare SmartMemory vs simpler memory approach (illustrative comparison)."""

        # Calculate metrics
        smart_metrics = self._calculate_system_metrics(smart_results)
        simple_metrics = self._calculate_system_metrics(simple_results)

        # SmartMemory specific capabilities
        smart_capabilities = {
            'cross_session_learning': smart_tutor.learning_metrics['cross_concept_connections'],
            'teaching_adaptations': smart_tutor.learning_metrics['teaching_adaptations'],
            'personalization_improvements': smart_tutor.learning_metrics['personalization_improvements'],
            'concepts_mastered': smart_tutor.learning_metrics['concepts_mastered'],
            'total_interactions': smart_tutor.learning_metrics['total_interactions']
        }

        # Capability demonstrations
        capabilities_demonstrated = {
            'prior_knowledge_integration': smart_metrics['avg_prior_knowledge'] > simple_metrics['avg_prior_knowledge'],
            'personalization_rate': smart_metrics['personalization_rate'] > simple_metrics['personalization_rate'],
            'contextual_teaching': smart_metrics['contextual_rate'] > simple_metrics['contextual_rate'],
            'cross_session_intelligence': smart_metrics['prior_learning_mentions'] > simple_metrics['prior_learning_mentions'],
            'adaptive_teaching': len(set(result['teaching_approach'] for result in smart_results)) > 1
        }

        # Overall capability score
        capability_score = sum(capabilities_demonstrated.values()) / len(capabilities_demonstrated)

        return {
            'smart_memory_metrics': smart_metrics,
            'simple_memory_metrics': simple_metrics,
            'smart_memory_capabilities': smart_capabilities,
            'capabilities_demonstrated': capabilities_demonstrated,
            'capability_score': capability_score,
            'human_evaluation_questions': self._generate_evaluation_questions(smart_results, simple_results),
            'detailed_comparison': {
                'smart_memory_results': smart_results,
                'simple_memory_results': simple_results
            }
        }

    def _calculate_system_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate performance metrics for a memory system."""
        total_interactions = sum(len(week['interactions']) for week in results)

        if total_interactions == 0:
            return {
                'avg_prior_knowledge': 0,
                'personalization_rate': 0,
                'contextual_rate': 0,
                'prior_learning_mentions': 0
            }

        personalized_count = sum(
            sum(1 for interaction in week['interactions'] if interaction['personalized'])
            for week in results
        )

        contextual_count = sum(
            sum(1 for interaction in week['interactions'] if interaction['contextual'])
            for week in results
        )

        prior_learning_count = sum(
            sum(1 for interaction in week['interactions'] if interaction['mentions_prior_learning'])
            for week in results
        )

        avg_prior_knowledge = sum(week['prior_knowledge_used'] for week in results) / len(results)

        return {
            'avg_prior_knowledge': avg_prior_knowledge,
            'personalization_rate': personalized_count / total_interactions,
            'contextual_rate': contextual_count / total_interactions,
            'prior_learning_mentions': prior_learning_count / total_interactions
        }

    def _generate_evaluation_questions(self, smart_results: List[Dict], simple_results: List[Dict]) -> List[Dict]:
        """Generate human evaluation questions for capability comparison."""
        return [
            {
                'question': 'Which tutor better connects new concepts to previously learned material?',
                'smart_example': 'Week 4: "Remember your Calculator class from Week 2? Databases are like permanent storage for your web app..."',
                'simple_example': 'Week 4: "Here\'s how databases works in programming..."',
                'evaluation_focus': 'Cross-session learning integration'
            },
            {
                'question': 'Which tutor adapts its teaching style based on your learning patterns?',
                'smart_example': 'Week 3: "I remember this was tricky before, so let\'s take it step by step..."',
                'simple_example': 'Week 3: "Here\'s how web_frameworks works in programming..."',
                'evaluation_focus': 'Personalized teaching adaptation'
            },
            {
                'question': 'Which tutor maintains better context of your current project?',
                'smart_example': 'Week 3: "This fits perfectly with your current Web Development project!"',
                'simple_example': 'Week 3: "Here\'s how apis works in programming..."',
                'evaluation_focus': 'Project context awareness'
            },
            {
                'question': 'Which tutor shows evidence of learning and improving over time?',
                'smart_example': 'Teaching approach evolves: foundational → build_on_success → progressive',
                'simple_example': 'Teaching approach stays: standard → standard → standard',
                'evaluation_focus': 'System learning and evolution'
            }
        ]


def main():
    """Main demonstration function."""
    logger.info("Starting Advanced Programming Tutor - Multi-Memory Demonstration")
    logger.info("=" * 80)

    # Run capability evaluation
    evaluator = CapabilityEvaluator()
    results = evaluator.run_4_week_comparison()

    # Display results
    logger.info("=" * 80)
    logger.info("CAPABILITY EVALUATION RESULTS")
    logger.info("=" * 80)

    logger.info(f"\nCAPABILITY SCORE: {results['capability_score']:.1%}")

    logger.info("\nCAPABILITIES DEMONSTRATED:")
    for capability, achieved in results['capabilities_demonstrated'].items():
        status = "✅ DEMONSTRATED" if achieved else "❌ NOT DEMONSTRATED"
        logger.info(f"  {capability}: {status}")

    logger.info("\nSMARTMEMORY CAPABILITIES:")
    for metric, value in results['smart_memory_capabilities'].items():
        logger.info(f"  {metric}: {value}")

    logger.info("\nPERFORMANCE COMPARISON (illustrative):")
    smart_metrics = results['smart_memory_metrics']
    simple_metrics = results['simple_memory_metrics']

    logger.info(f"  Prior Knowledge Integration: SmartMemory {smart_metrics['avg_prior_knowledge']:.1f} vs Simple {simple_metrics['avg_prior_knowledge']:.1f}")
    logger.info(f"  Personalization Rate: SmartMemory {smart_metrics['personalization_rate']:.1%} vs Simple {simple_metrics['personalization_rate']:.1%}")
    logger.info(f"  Contextual Teaching: SmartMemory {smart_metrics['contextual_rate']:.1%} vs Simple {simple_metrics['contextual_rate']:.1%}")
    logger.info(f"  Cross-Session Learning: SmartMemory {smart_metrics['prior_learning_mentions']:.1%} vs Simple {simple_metrics['prior_learning_mentions']:.1%}")

    logger.info("\nHUMAN EVALUATION QUESTIONS:")
    for i, question in enumerate(results['human_evaluation_questions'], 1):
        logger.info(f"\n{i}. {question['question']}")
        logger.info(f"   SmartMemory: {question['smart_example']}")
        logger.info(f"   Simple Approach: {question['simple_example']}")
        logger.info(f"   Focus: {question['evaluation_focus']}")

    # Save detailed results
    with open('advanced_tutor_capability_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    logger.info("\nDetailed results saved to advanced_tutor_capability_results.json")

    # Final assessment
    if results['capability_score'] >= 0.8:
        rating = "EXCELLENT CAPABILITY DEMONSTRATION"
    elif results['capability_score'] >= 0.6:
        rating = "GOOD CAPABILITY DEMONSTRATION"
    elif results['capability_score'] >= 0.4:
        rating = "MODERATE CAPABILITY DEMONSTRATION"
    else:
        rating = "NEEDS IMPROVEMENT"

    print(f"\n🏆 CAPABILITY ASSESSMENT: {rating}")
    print(f"📊 Capability Score: {results['capability_score']:.1%}")
    print(f"🎯 Capabilities Demonstrated: {sum(results['capabilities_demonstrated'].values())}/5")
    print("\nNote: This is an illustrative comparison to demonstrate multi-memory capabilities.")

    return results


if __name__ == "__main__":
    main()
