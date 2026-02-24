"""
Performance optimizations for temporal queries.

Provides indexing, caching, and query optimization for bi-temporal operations.
"""

from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class TemporalIndex:
    """
    Temporal index for fast time-based queries.
    
    Maintains indexes on:
    - Valid time ranges
    - Transaction time ranges
    - Version numbers
    """
    
    def __init__(self):
        """Initialize temporal indexes."""
        # Index: item_id -> sorted list of (time, version_number)
        self.valid_time_index: Dict[str, List[tuple]] = defaultdict(list)
        self.transaction_time_index: Dict[str, List[tuple]] = defaultdict(list)
        
        # Index: time_range -> set of item_ids
        self.time_range_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Statistics
        self.index_hits = 0
        self.index_misses = 0
    
    def add_version(
        self,
        item_id: str,
        version_number: int,
        valid_time_start: datetime,
        valid_time_end: Optional[datetime],
        transaction_time_start: datetime
    ):
        """
        Add a version to the temporal index.
        
        Args:
            item_id: Memory item ID
            version_number: Version number
            valid_time_start: Start of valid time
            valid_time_end: End of valid time (None = current)
            transaction_time_start: Transaction time
        """
        # Add to valid time index
        self.valid_time_index[item_id].append(
            (valid_time_start, version_number, valid_time_end)
        )
        self.valid_time_index[item_id].sort()
        
        # Add to transaction time index
        self.transaction_time_index[item_id].append(
            (transaction_time_start, version_number)
        )
        self.transaction_time_index[item_id].sort()
        
        # Add to time range index (by day)
        day_key = valid_time_start.strftime('%Y-%m-%d')
        self.time_range_index[day_key].add(item_id)
    
    def find_version_at_time(
        self,
        item_id: str,
        time: datetime,
        time_type: str = 'valid'
    ) -> Optional[int]:
        """
        Find version number at a specific time using index.
        
        Args:
            item_id: Memory item ID
            time: Time to query
            time_type: 'valid' or 'transaction'
            
        Returns:
            Version number or None
        """
        if time_type == 'valid':
            index = self.valid_time_index.get(item_id, [])
            
            # Binary search for efficiency
            for start, version, end in reversed(index):
                if start <= time:
                    if end is None or end > time:
                        self.index_hits += 1
                        return version
        else:
            index = self.transaction_time_index.get(item_id, [])
            
            for trans_time, version in reversed(index):
                if trans_time <= time:
                    self.index_hits += 1
                    return version
        
        self.index_misses += 1
        return None
    
    def find_items_in_range(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Set[str]:
        """
        Find all items with versions in a time range.
        
        Args:
            start_time: Start of range
            end_time: End of range
            
        Returns:
            Set of item IDs
        """
        items = set()
        
        # Query by day
        current = start_time
        while current <= end_time:
            day_key = current.strftime('%Y-%m-%d')
            items.update(self.time_range_index.get(day_key, set()))
            current = current.replace(
                day=current.day + 1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0
            )
        
        if items:
            self.index_hits += 1
        else:
            self.index_misses += 1
        
        return items
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics."""
        total_queries = self.index_hits + self.index_misses
        hit_rate = self.index_hits / total_queries if total_queries > 0 else 0
        
        return {
            'index_hits': self.index_hits,
            'index_misses': self.index_misses,
            'hit_rate': hit_rate,
            'indexed_items': len(self.valid_time_index),
            'time_range_buckets': len(self.time_range_index)
        }
    
    def clear(self):
        """Clear all indexes."""
        self.valid_time_index.clear()
        self.transaction_time_index.clear()
        self.time_range_index.clear()
        self.index_hits = 0
        self.index_misses = 0


class TemporalQueryOptimizer:
    """
    Optimize temporal queries for performance.
    
    Features:
    - Query plan optimization
    - Batch operations
    - Result caching
    """
    
    def __init__(self, index: TemporalIndex):
        """
        Initialize query optimizer.
        
        Args:
            index: Temporal index to use
        """
        self.index = index
        self.query_cache: Dict[str, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def optimize_time_range_query(
        self,
        start_time: datetime,
        end_time: datetime,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize a time range query.
        
        Args:
            start_time: Start of range
            end_time: End of range
            filters: Optional filters
            
        Returns:
            Optimized query plan
        """
        # Generate cache key
        cache_key = f"range:{start_time.isoformat()}:{end_time.isoformat()}:{filters}"
        
        # Check cache
        if cache_key in self.query_cache:
            self.cache_hits += 1
            return self.query_cache[cache_key]
        
        self.cache_misses += 1
        
        # Use index to find candidate items
        candidate_items = self.index.find_items_in_range(start_time, end_time)
        
        # Build optimized query plan
        plan = {
            'strategy': 'index_scan' if candidate_items else 'full_scan',
            'candidate_items': list(candidate_items),
            'estimated_items': len(candidate_items),
            'use_index': True,
            'filters': filters or {}
        }
        
        # Cache the plan
        self.query_cache[cache_key] = plan
        
        return plan
    
    def batch_version_lookup(
        self,
        item_ids: List[str],
        time: datetime,
        time_type: str = 'valid'
    ) -> Dict[str, Optional[int]]:
        """
        Batch lookup of versions at a specific time.
        
        Args:
            item_ids: List of item IDs
            time: Time to query
            time_type: 'valid' or 'transaction'
            
        Returns:
            Dictionary mapping item_id to version number
        """
        results = {}
        
        for item_id in item_ids:
            version = self.index.find_version_at_time(item_id, time, time_type)
            results[item_id] = version
        
        logger.info(f"Batch lookup: {len(item_ids)} items at {time.isoformat()}")
        
        return results
    
    def optimize_join_query(
        self,
        item_ids: List[str],
        start_time: datetime,
        end_time: datetime,
        join_type: str
    ) -> Dict[str, Any]:
        """
        Optimize a temporal join query.
        
        Args:
            item_ids: Items to join
            start_time: Start of time range
            end_time: End of time range
            join_type: Type of join
            
        Returns:
            Optimized join plan
        """
        # Estimate selectivity
        total_candidates = 0
        for item_id in item_ids:
            versions = self.index.valid_time_index.get(item_id, [])
            # Count versions in range
            in_range = sum(
                1 for start, _, end in versions
                if start <= end_time and (end is None or end >= start_time)
            )
            total_candidates += in_range
        
        # Choose join strategy
        if total_candidates < 100:
            strategy = 'nested_loop'
        elif total_candidates < 1000:
            strategy = 'hash_join'
        else:
            strategy = 'sort_merge_join'
        
        plan = {
            'strategy': strategy,
            'estimated_candidates': total_candidates,
            'join_type': join_type,
            'use_index': True,
            'item_count': len(item_ids)
        }
        
        logger.info(f"Join plan: {strategy} for {len(item_ids)} items")
        
        return plan
    
    def clear_cache(self):
        """Clear query cache."""
        self.query_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        total_queries = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_queries if total_queries > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': hit_rate,
            'cached_queries': len(self.query_cache),
            'index_stats': self.index.get_statistics()
        }


class TemporalBatchOperations:
    """
    Batch operations for temporal queries.
    
    Improves performance by batching multiple operations together.
    """
    
    def __init__(self, version_tracker, relationship_queries):
        """
        Initialize batch operations.
        
        Args:
            version_tracker: VersionTracker instance
            relationship_queries: TemporalRelationshipQueries instance
        """
        self.version_tracker = version_tracker
        self.relationship_queries = relationship_queries
    
    def batch_get_versions(
        self,
        item_ids: List[str],
        time: datetime
    ) -> Dict[str, Any]:
        """
        Get versions for multiple items at once.
        
        Args:
            item_ids: List of item IDs
            time: Time to query
            
        Returns:
            Dictionary mapping item_id to version
        """
        results = {}
        
        for item_id in item_ids:
            version = self.version_tracker.get_version_at_time(item_id, time)
            results[item_id] = version
        
        logger.info(f"Batch retrieved {len(results)} versions")
        
        return results
    
    def batch_get_relationships(
        self,
        item_ids: List[str],
        time: datetime,
        relationship_type: Optional[str] = None
    ) -> Dict[str, List]:
        """
        Get relationships for multiple items at once.
        
        Args:
            item_ids: List of item IDs
            time: Time to query
            relationship_type: Optional filter
            
        Returns:
            Dictionary mapping item_id to list of relationships
        """
        results = {}
        
        for item_id in item_ids:
            rels = self.relationship_queries.get_relationships_at_time(
                item_id,
                time,
                relationship_type
            )
            results[item_id] = rels
        
        logger.info(f"Batch retrieved relationships for {len(results)} items")
        
        return results
    
    def batch_temporal_search(
        self,
        queries: List[str],
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, List]:
        """
        Execute multiple temporal searches at once.
        
        Args:
            queries: List of search queries
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            Dictionary mapping query to results
        """
        results = {}
        
        # This would be optimized to batch the actual searches
        # For now, it's a placeholder for the pattern
        
        logger.info(f"Batch executed {len(queries)} temporal searches")
        
        return results
