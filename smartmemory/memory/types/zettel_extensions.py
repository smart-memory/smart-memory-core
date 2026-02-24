"""
Zettelkasten Extensions - Core missing functionality for complete Zettelkasten system.

Implements:
- Backlink system (bidirectional navigation)
- Emergent structure detection
- Discovery engine
- Knowledge evolution
- Thinking support features
"""

import logging
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any

from smartmemory.models.memory_item import MemoryItem

logger = logging.getLogger(__name__)

# NetworkX import with fallback for graph analysis
try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    logger.warning("NetworkX not available - some advanced graph analysis features will be disabled")


@dataclass
class ConnectionStrength:
    """Represents the strength and metadata of a connection between notes."""
    source_id: str
    target_id: str
    connection_type: str
    strength: float = 1.0
    usage_count: int = 0
    last_accessed: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class KnowledgeCluster:
    """Represents a cluster of related notes discovered through connection analysis."""
    cluster_id: str
    note_ids: List[str]
    central_concepts: List[str]
    connection_density: float
    cluster_strength: float
    emergence_score: float


@dataclass
class DiscoveryPath:
    """Represents a path between notes for knowledge discovery."""
    start_note_id: str
    end_note_id: str
    path_notes: List[str]
    path_strength: float
    discovery_type: str  # 'semantic', 'conceptual', 'temporal'
    insights: List[str]


class ZettelBacklinkSystem:
    """
    Implements bidirectional linking and backlink navigation for Zettelkasten.
    
    Core missing functionality: "What links here?" and bidirectional navigation.
    """

    def __init__(self, zettel_memory):
        self.zettel_memory = zettel_memory
        self.connection_cache = {}
        self.backlink_cache = {}

    def get_backlinks(self, note_id: str) -> List[MemoryItem]:
        """
        Get all notes that link TO this note (backlinks).
        
        This is the core missing functionality in current implementation.
        """
        try:
            # Check cache first
            if note_id in self.backlink_cache:
                return self.backlink_cache[note_id]

            backlinks = []

            # Find all incoming LINKS_TO relations
            incoming_links_raw = self.zettel_memory.graph.graph.get_incoming_neighbors(
                note_id, edge_type="LINKS_TO"
            )
            
            # get_incoming_neighbors already returns MemoryItems
            for src in incoming_links_raw:
                if isinstance(src, MemoryItem):
                    backlinks.append(src)
                elif isinstance(src, str):
                    # If it's just an ID, fetch the item
                    source_item = self.zettel_memory.get(src)
                    if source_item:
                        backlinks.append(source_item)
                elif isinstance(src, dict):
                    sid = src.get('item_id')
                    if sid:
                        source_item = self.zettel_memory.get(sid)
                        if source_item:
                            backlinks.append(source_item)
                elif hasattr(src, 'item_id'):
                    # Has item_id attribute
                    backlinks.append(src)

            # Cache the result
            self.backlink_cache[note_id] = backlinks
            return backlinks

        except Exception as e:
            logger.error(f"Failed to get backlinks for {note_id}: {e}")
            return []

    def get_bidirectional_connections(self, note_id: str) -> Dict[str, List[MemoryItem]]:
        """
        Get both forward links and backlinks for complete bidirectional view.
        """
        return {
            'forward_links': self.get_forward_links(note_id),
            'backlinks': self.get_backlinks(note_id),
            'related_by_tags': self.get_tag_related_notes(note_id),
            'related_by_concepts': self.get_concept_related_notes(note_id)
        }

    def get_all_connections(self, note_id: str) -> List[Dict[str, Any]]:
        """
        Get all connections (backlinks + forward links) as a flat list.

        Each connection includes the note and the connection type.
        Returns: List of dicts with note data and connection_type.
        """
        try:
            connections = []
            seen_ids: Set[str] = set()

            # Get bidirectional connections
            bidir = self.get_bidirectional_connections(note_id)

            # Map connection types to the dict keys
            type_mapping = {
                'forward_links': 'forward_link',
                'backlinks': 'backlink',
                'related_by_tags': 'tag_related',
                'related_by_concepts': 'concept_related'
            }

            for conn_type, notes in bidir.items():
                for note in notes:
                    note_id_val = note.item_id if hasattr(note, 'item_id') else str(note)
                    if note_id_val not in seen_ids:
                        seen_ids.add(note_id_val)
                        connections.append({
                            'note_id': note_id_val,
                            'connection_type': type_mapping.get(conn_type, conn_type),
                            'content': note.content[:100] if hasattr(note, 'content') and note.content else '',
                            'metadata': dict(note.metadata) if hasattr(note, 'metadata') and note.metadata else {}
                        })

            return connections

        except Exception as e:
            logger.error(f"Failed to get all connections for {note_id}: {e}")
            return []

    def get_forward_links(self, note_id: str) -> List[MemoryItem]:
        """Get all notes this note links TO (forward links)."""
        try:
            forward_links = []

            # Find all outgoing LINKS_TO relations
            outgoing_raw = self.zettel_memory.graph.graph.get_neighbors(note_id, edge_type="LINKS_TO")
            
            # get_neighbors already returns MemoryItems
            for tgt in outgoing_raw:
                if isinstance(tgt, MemoryItem):
                    forward_links.append(tgt)
                elif isinstance(tgt, str):
                    # If it's just an ID, fetch the item
                    target_item = self.zettel_memory.get(tgt)
                    if target_item:
                        forward_links.append(target_item)
                elif isinstance(tgt, dict):
                    tid = tgt.get('item_id')
                    if tid:
                        target_item = self.zettel_memory.get(tid)
                        if target_item:
                            forward_links.append(target_item)
                elif hasattr(tgt, 'item_id'):
                    # Has item_id attribute
                    forward_links.append(tgt)

            return forward_links

        except Exception as e:
            logger.error(f"Failed to get forward links for {note_id}: {e}")
            return []

    def get_tag_related_notes(self, note_id: str) -> List[MemoryItem]:
        """Get notes related through shared tags."""
        try:
            current_note = self.zettel_memory.get(note_id)
            if not current_note:
                return []

            current_tags = current_note.metadata.get('tags', [])
            if not current_tags:
                return []

            related_notes = []

            # Find notes with shared tags using CORRECT SmartGraph access
            for tag in current_tags:
                tag_related = self.zettel_memory.graph.graph.search_nodes({'tags': tag})
                for related_note_data in tag_related:
                    if related_note_data.get('item_id') != note_id:
                        related_item = self.zettel_memory.get(related_note_data.get('item_id'))
                        if related_item:
                            related_notes.append(related_item)

            return list(set(related_notes))  # Remove duplicates

        except Exception as e:
            logger.error(f"Failed to get tag-related notes for {note_id}: {e}")
            return []

    def get_concept_related_notes(self, note_id: str) -> List[MemoryItem]:
        """Get notes related through shared concepts."""
        try:
            current_note = self.zettel_memory.get(note_id)
            if not current_note:
                return []

            current_concepts = current_note.metadata.get('concepts', [])
            if not current_concepts:
                return []

            related_notes = []

            # Find notes mentioning same concepts using CORRECT SmartGraph access
            for concept in current_concepts:
                concept_related = self.zettel_memory.graph.graph.search_nodes({'concepts': concept})
                for related_note_data in concept_related:
                    if related_note_data.get('item_id') != note_id:
                        related_item = self.zettel_memory.get(related_note_data.get('item_id'))
                        if related_item:
                            related_notes.append(related_item)

            return list(set(related_notes))  # Remove duplicates

        except Exception as e:
            logger.error(f"Failed to get concept-related notes for {note_id}: {e}")
            return []

    def create_bidirectional_link(self, source_id: str, target_id: str, link_type: str = "LINKS_TO"):
        """
        Create bidirectional links automatically when a wikilink is detected.
        
        This ensures every [[link]] creates both forward and backward connections.
        """
        try:
            # Create forward link using CORRECT SmartGraph access
            self.zettel_memory.graph.graph.add_edge(
                source_id, target_id, link_type,
                {'direction': 'forward', 'auto_created': True}
            )

            # Create backward link
            self.zettel_memory.graph.graph.add_edge(
                target_id, source_id, f"{link_type}_BACK",
                {'direction': 'backward', 'auto_created': True}
            )

            # Clear caches
            self.backlink_cache.clear()
            self.connection_cache.clear()

            logger.info(f"Created bidirectional link: {source_id} <-> {target_id}")

        except Exception as e:
            logger.error(f"Failed to create bidirectional link {source_id} <-> {target_id}: {e}")


class ZettelEmergentStructure:
    """
    Detects emergent knowledge structures from connection patterns.
    
    Missing functionality: Knowledge emergence from connection analysis.
    """

    def __init__(self, zettel_memory):
        self.zettel_memory = zettel_memory
        self.structure_cache = {}

    def detect_knowledge_clusters(self, min_cluster_size: int = 3) -> List[KnowledgeCluster]:
        """
        Detect clusters of highly connected notes that form knowledge domains.
        """
        try:
            # Build connection graph
            graph = self._build_connection_graph()
            # Guard: if graph too small or edgeless, no clusters
            if graph.number_of_nodes() < min_cluster_size or graph.number_of_edges() == 0:
                return []

            # Use community detection to find clusters
            clusters = []
            try:
                communities = nx.community.greedy_modularity_communities(graph)
            except Exception as e:
                # Fallback to a method without any cutoff/resolution pitfalls
                try:
                    communities = list(nx.community.label_propagation_communities(graph))
                except Exception:
                    # If all methods fail, re-raise original
                    raise e

            for i, community in enumerate(communities):
                if len(community) >= min_cluster_size:
                    cluster_notes = list(community)

                    # Calculate cluster metrics
                    subgraph = graph.subgraph(cluster_notes)
                    density = nx.density(subgraph)

                    # Extract central concepts
                    central_concepts = self._extract_cluster_concepts(cluster_notes)

                    # Calculate emergence score
                    emergence_score = self._calculate_emergence_score(cluster_notes, central_concepts)

                    cluster = KnowledgeCluster(
                        cluster_id=f"cluster_{i}",
                        note_ids=cluster_notes,
                        central_concepts=central_concepts,
                        connection_density=density,
                        cluster_strength=len(cluster_notes) * density,
                        emergence_score=emergence_score
                    )
                    clusters.append(cluster)

            return sorted(clusters, key=lambda c: c.emergence_score, reverse=True)

        except Exception as e:
            logger.error(f"Failed to detect knowledge clusters: {e}")
            return []

    def find_knowledge_bridges(self) -> List[Tuple[str, List[str]]]:
        """
        Find notes that bridge different knowledge domains.
        
        These are notes with high betweenness centrality.
        """
        try:
            graph = self._build_connection_graph()

            # Calculate betweenness centrality
            centrality = nx.betweenness_centrality(graph)

            # Find high-centrality notes (bridges)
            bridges = []
            threshold = 0.1  # Top 10% centrality

            for note_id, centrality_score in centrality.items():
                if centrality_score > threshold:
                    # Find the domains this note connects
                    neighbors = list(graph.neighbors(note_id))
                    connected_clusters = self._identify_connected_clusters(note_id, neighbors)
                    bridges.append((note_id, connected_clusters))

            return sorted(bridges, key=lambda b: centrality[b[0]], reverse=True)

        except Exception as e:
            logger.error(f"Failed to find knowledge bridges: {e}")
            return []

    def detect_concept_emergence(self) -> Dict[str, float]:
        """
        Detect emerging concepts based on connection patterns and frequency.
        """
        try:
            concept_connections = defaultdict(int)
            concept_frequency = Counter()

            # Analyze all notes for concept patterns
            all_notes = self._get_all_notes()

            for note in all_notes:
                concepts = note.metadata.get('concepts', [])
                concept_frequency.update(concepts)

                # Count co-occurrence patterns
                for i, concept1 in enumerate(concepts):
                    for concept2 in concepts[i + 1:]:
                        concept_connections[(concept1, concept2)] += 1

            # Calculate emergence scores
            emergence_scores = {}
            for concept, frequency in concept_frequency.items():
                # Concepts that appear frequently and have many connections
                connection_count = sum(1 for pair in concept_connections if concept in pair)
                emergence_score = frequency * connection_count / len(all_notes)
                emergence_scores[concept] = emergence_score

            return dict(sorted(emergence_scores.items(), key=lambda x: x[1], reverse=True))

        except Exception as e:
            logger.error(f"Failed to detect concept emergence: {e}")
            return {}

    def _build_connection_graph(self) -> nx.Graph:
        """Build NetworkX graph from Zettelkasten connections."""
        graph = nx.Graph()

        try:
            # Add all notes as nodes
            all_notes = self._get_all_notes()
            for note in all_notes:
                graph.add_node(note.item_id, **note.metadata)

            # Add connections as edges
            for note in all_notes:
                # Add wikilink connections
                forward_links = self._get_note_links(note.item_id)
                for link_target in forward_links:
                    graph.add_edge(note.item_id, link_target, connection_type='wikilink')

                # Add tag-based connections
                tag_related = self._get_tag_connections(note.item_id)
                for related_note in tag_related:
                    graph.add_edge(note.item_id, related_note, connection_type='tag')

            return graph

        except Exception as e:
            logger.error(f"Failed to build connection graph: {e}")
            return nx.Graph()

    def _extract_cluster_concepts(self, note_ids: List[str]) -> List[str]:
        """Extract the most common concepts in a cluster."""
        concept_counter = Counter()

        for note_id in note_ids:
            note = self.zettel_memory.get(note_id)
            if note:
                concepts = note.metadata.get('concepts', [])
                concept_counter.update(concepts)

        # Return top 5 concepts
        return [concept for concept, count in concept_counter.most_common(5)]

    def _calculate_emergence_score(self, note_ids: List[str], concepts: List[str]) -> float:
        """Calculate how much new knowledge emerges from this cluster."""
        # Simple heuristic: cluster size * concept diversity * connection density
        cluster_size = len(note_ids)
        concept_diversity = len(concepts)

        # Calculate internal connections
        internal_connections = 0
        for note_id in note_ids:
            links = self._get_note_links(note_id)
            internal_connections += len([link for link in links if link in note_ids])

        max_possible_connections = cluster_size * (cluster_size - 1) / 2
        connection_ratio = internal_connections / max(max_possible_connections, 1)

        return cluster_size * concept_diversity * connection_ratio

    def _identify_connected_clusters(self, bridge_note: str, neighbors: List[str]) -> List[str]:
        """Identify which clusters a bridge note connects."""
        # Simplified: return unique concept domains of neighbors
        cluster_concepts = set()

        for neighbor_id in neighbors:
            neighbor = self.zettel_memory.get(neighbor_id)
            if neighbor:
                concepts = neighbor.metadata.get('concepts', [])
                cluster_concepts.update(concepts[:2])  # Top 2 concepts per neighbor

        return list(cluster_concepts)

    def _get_all_notes(self) -> List[MemoryItem]:
        """Get all notes from the Zettelkasten."""
        try:
            # Get all nodes using CORRECT SmartGraph access
            all_node_data = self.zettel_memory.graph.graph.get_all_nodes()
            notes: List[MemoryItem] = []

            for nd in all_node_data:
                item_id: Optional[str] = None
                mem_type: Optional[str] = None
                # Support dict-like
                if isinstance(nd, dict):
                    item_id = nd.get('item_id')
                    mem_type = nd.get('memory_type') or nd.get('label')
                # Support MemoryItem-like
                elif hasattr(nd, 'item_id'):
                    item_id = nd.item_id
                    md = getattr(nd, 'metadata', {}) or {}
                    mem_type = md.get('memory_type') or md.get('label')
                # Support backend Node objects
                elif hasattr(nd, 'properties'):
                    props = dict(nd.properties)
                    item_id = props.get('item_id')
                    mem_type = props.get('memory_type') or props.get('label')

                # Accept notes with label 'Note' or memory_type 'zettel'
                # ZettelMemory stores notes with label='Note'
                if item_id and (mem_type == 'zettel' or mem_type == 'Note'):
                    note = self.zettel_memory.get(item_id)
                    if note:
                        notes.append(note)

            return notes

        except Exception as e:
            logger.error(f"Failed to get all notes: {e}")
            return []

    def _get_note_links(self, note_id: str) -> List[str]:
        """Get all notes this note links to."""
        try:
            neighbors = self.zettel_memory.graph.graph.get_neighbors(note_id, edge_type="LINKS_TO")
            # Normalize return values to list[str]
            norm: List[str] = []
            for n in neighbors:
                if isinstance(n, MemoryItem):
                    norm.append(n.item_id)
                elif isinstance(n, str):
                    norm.append(n)
                elif isinstance(n, dict):
                    nid = n.get('item_id')
                    if nid:
                        norm.append(nid)
                elif hasattr(n, 'item_id'):
                    norm.append(n.item_id)
                elif hasattr(n, 'properties'):
                    props = dict(n.properties)
                    nid = props.get('item_id')
                    if nid:
                        norm.append(nid)
            return norm
        except Exception as e:
            logger.error(f"Failed to get links for {note_id}: {e}")
            return []

    def _get_tag_connections(self, note_id: str) -> List[str]:
        """Get notes connected through shared tags."""
        try:
            note = self.zettel_memory.get(note_id)
            if not note:
                return []

            tags = note.metadata.get('tags', [])
            connected_notes = []

            for tag in tags:
                tag_notes = self.zettel_memory.graph.graph.search_nodes({'tags': tag})
                for tnd in tag_notes:
                    if isinstance(tnd, dict):
                        tag_note_id = tnd.get('item_id')
                    elif hasattr(tnd, 'properties'):
                        props = dict(tnd.properties)
                        tag_note_id = props.get('item_id')
                    elif hasattr(tnd, 'item_id'):
                        tag_note_id = tnd.item_id
                    else:
                        tag_note_id = None
                    if tag_note_id and tag_note_id != note_id:
                        connected_notes.append(tag_note_id)

            return list(set(connected_notes))

        except Exception as e:
            logger.error(f"Failed to get tag connections for {note_id}: {e}")
            return []


class ZettelDiscoveryEngine:
    """
    Implements serendipitous discovery and knowledge path finding.
    
    Missing functionality: "How do I get from concept A to concept B?"
    """

    def __init__(self, zettel_memory):
        self.zettel_memory = zettel_memory
        self.backlink_system = ZettelBacklinkSystem(zettel_memory)
        self.structure_system = ZettelEmergentStructure(zettel_memory)

    def find_knowledge_paths(self, start_note_id: str, end_note_id: str, max_depth: int = 5) -> List[DiscoveryPath]:
        """
        Find paths between two notes for knowledge discovery.
        
        This answers: "How do I get from concept A to concept B?"
        """
        try:
            graph = self.structure_system._build_connection_graph()

            # Find all simple paths between notes
            paths = []
            try:
                simple_paths = nx.all_simple_paths(graph, start_note_id, end_note_id, cutoff=max_depth)

                for path_nodes in simple_paths:
                    if len(path_nodes) <= max_depth + 1:
                        # Calculate path strength
                        path_strength = self._calculate_path_strength(path_nodes, graph)

                        # Determine discovery type
                        discovery_type = self._determine_discovery_type(path_nodes)

                        # Generate insights
                        insights = self._generate_path_insights(path_nodes)

                        discovery_path = DiscoveryPath(
                            start_note_id=start_note_id,
                            end_note_id=end_note_id,
                            path_notes=path_nodes,
                            path_strength=path_strength,
                            discovery_type=discovery_type,
                            insights=insights
                        )
                        paths.append(discovery_path)

            except nx.NetworkXNoPath:
                logger.info(f"No path found between {start_note_id} and {end_note_id}")

            return sorted(paths, key=lambda p: p.path_strength, reverse=True)

        except Exception as e:
            logger.error(f"Failed to find knowledge paths: {e}")
            return []

    def suggest_related_notes(self, note_id: str, suggestion_count: int = 5) -> List[Tuple[MemoryItem, float, str]]:
        """
        Suggest related notes for serendipitous discovery.
        
        Returns: List of (note, relevance_score, reason)
        """
        try:
            suggestions = []

            # Get current note
            current_note = self.zettel_memory.get(note_id)
            if not current_note:
                return []

            # Get bidirectional connections
            connections = self.backlink_system.get_bidirectional_connections(note_id)

            # Score and rank suggestions
            candidate_notes = set()

            # Add direct connections
            for connection_type, notes in connections.items():
                for note in notes:
                    candidate_notes.add(note.item_id)

            # Add second-degree connections
            for connection_type, notes in connections.items():
                for note in notes[:3]:  # Limit to top 3 per type
                    second_degree = self.backlink_system.get_bidirectional_connections(note.item_id)
                    for sd_type, sd_notes in second_degree.items():
                        for sd_note in sd_notes[:2]:  # Limit second degree
                            if sd_note.item_id != note_id:
                                candidate_notes.add(sd_note.item_id)

            # Score candidates
            for candidate_id in candidate_notes:
                if candidate_id != note_id:
                    candidate_note = self.zettel_memory.get(candidate_id)
                    if candidate_note:
                        score, reason = self._calculate_relevance_score(current_note, candidate_note)
                        suggestions.append((candidate_note, score, reason))

            # Sort by relevance and return top suggestions
            suggestions.sort(key=lambda x: x[1], reverse=True)
            return suggestions[:suggestion_count]

        except Exception as e:
            logger.error(f"Failed to suggest related notes for {note_id}: {e}")
            return []

    def discover_missing_connections(self, note_id: str) -> List[Tuple[str, float, str]]:
        """
        Suggest connections that might be missing based on content analysis.
        
        Returns: List of (target_note_id, connection_strength, reason)
        """
        try:
            current_note = self.zettel_memory.get(note_id)
            if not current_note:
                return []

            missing_connections = []

            # Get current concepts and tags
            current_concepts = set(current_note.metadata.get('concepts', []))
            current_tags = set(current_note.metadata.get('tags', []))

            # Find notes with overlapping concepts/tags that aren't linked
            all_notes = self.structure_system._get_all_notes()
            current_links = set(self.backlink_system.get_forward_links(note_id))
            current_link_ids = {note.item_id for note in current_links}

            for candidate_note in all_notes:
                if candidate_note.item_id == note_id or candidate_note.item_id in current_link_ids:
                    continue

                candidate_concepts = set(candidate_note.metadata.get('concepts', []))
                candidate_tags = set(candidate_note.metadata.get('tags', []))

                # Calculate overlap
                concept_overlap = len(current_concepts & candidate_concepts)
                tag_overlap = len(current_tags & candidate_tags)

                if concept_overlap > 0 or tag_overlap > 0:
                    # Calculate connection strength
                    strength = (concept_overlap * 2 + tag_overlap) / max(len(current_concepts | candidate_concepts), 1)

                    # Generate reason
                    reasons = []
                    if concept_overlap > 0:
                        shared_concepts = list(current_concepts & candidate_concepts)
                        reasons.append(f"Shared concepts: {', '.join(shared_concepts[:3])}")
                    if tag_overlap > 0:
                        shared_tags = list(current_tags & candidate_tags)
                        reasons.append(f"Shared tags: {', '.join(shared_tags[:3])}")

                    reason = "; ".join(reasons)
                    missing_connections.append((candidate_note.item_id, strength, reason))

            return sorted(missing_connections, key=lambda x: x[1], reverse=True)[:10]

        except Exception as e:
            logger.error(f"Failed to discover missing connections for {note_id}: {e}")
            return []

    def random_walk_discovery(self, start_note_id: str, walk_length: int = 5) -> List[str]:
        """
        Perform random walk for serendipitous discovery.
        
        Returns path of note IDs discovered through random traversal.
        """
        try:
            graph = self.structure_system._build_connection_graph()

            if start_note_id not in graph:
                return [start_note_id]

            walk_path = [start_note_id]
            current_node = start_note_id

            for _ in range(walk_length - 1):
                neighbors = list(graph.neighbors(current_node))
                if not neighbors:
                    break

                # Weighted random selection (prefer stronger connections)
                weights = []
                for neighbor in neighbors:
                    edge_data = graph.get_edge_data(current_node, neighbor)
                    weight = edge_data.get('weight', 1.0) if edge_data else 1.0
                    weights.append(weight)

                # Simple weighted choice (could be improved with numpy)
                total_weight = sum(weights)
                if total_weight > 0:
                    import random
                    r = random.uniform(0, total_weight)
                    cumulative = 0
                    for i, weight in enumerate(weights):
                        cumulative += weight
                        if r <= cumulative:
                            current_node = neighbors[i]
                            break
                else:
                    import random
                    current_node = random.choice(neighbors)

                walk_path.append(current_node)

            return walk_path

        except Exception as e:
            logger.error(f"Failed to perform random walk from {start_note_id}: {e}")
            return [start_note_id]

    def _calculate_path_strength(self, path_nodes: List[str], graph: nx.Graph) -> float:
        """Calculate the strength of a knowledge path."""
        if len(path_nodes) < 2:
            return 0.0

        total_strength = 0.0
        for i in range(len(path_nodes) - 1):
            edge_data = graph.get_edge_data(path_nodes[i], path_nodes[i + 1])
            edge_weight = edge_data.get('weight', 1.0) if edge_data else 1.0
            total_strength += edge_weight

        # Normalize by path length
        return total_strength / (len(path_nodes) - 1)

    def _determine_discovery_type(self, path_nodes: List[str]) -> str:
        """Determine the type of discovery this path represents."""
        # Simple heuristic based on path length and node types
        if len(path_nodes) <= 2:
            return 'direct'
        elif len(path_nodes) <= 4:
            return 'conceptual'
        else:
            return 'serendipitous'

    def _generate_path_insights(self, path_nodes: List[str]) -> List[str]:
        """Generate insights about what this path reveals."""
        insights = []

        if len(path_nodes) > 2:
            insights.append(f"Connection path through {len(path_nodes) - 2} intermediate concepts")

        if len(path_nodes) > 4:
            insights.append("Serendipitous connection - may reveal unexpected relationships")

        # Could add more sophisticated insight generation based on node content
        return insights

    def _calculate_relevance_score(self, current_note: MemoryItem, candidate_note: MemoryItem) -> Tuple[float, str]:
        """Calculate relevance score and reason for suggestion."""
        score = 0.0
        reasons = []

        # Concept overlap
        current_concepts = set(current_note.metadata.get('concepts', []))
        candidate_concepts = set(candidate_note.metadata.get('concepts', []))
        concept_overlap = len(current_concepts & candidate_concepts)

        if concept_overlap > 0:
            score += concept_overlap * 2
            shared = list(current_concepts & candidate_concepts)
            reasons.append(f"Shared concepts: {', '.join(shared[:2])}")

        # Tag overlap
        current_tags = set(current_note.metadata.get('tags', []))
        candidate_tags = set(candidate_note.metadata.get('tags', []))
        tag_overlap = len(current_tags & candidate_tags)

        if tag_overlap > 0:
            score += tag_overlap
            shared = list(current_tags & candidate_tags)
            reasons.append(f"Shared tags: {', '.join(shared[:2])}")

        # Content similarity (simple word overlap)
        current_words = set(current_note.content.lower().split())
        candidate_words = set(candidate_note.content.lower().split())
        word_overlap = len(current_words & candidate_words)

        if word_overlap > 5:  # Threshold for meaningful overlap
            score += word_overlap / 10
            reasons.append("Similar content")

        reason = "; ".join(reasons) if reasons else "General relevance"
        return score, reason
