"""
Analytics and monitoring for memory drift and bias detection.
"""
import numpy as np
import re
from collections import defaultdict, Counter
from datetime import datetime, timedelta, timezone
from scipy import stats
from typing import Dict, List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from smartmemory.configuration import MemoryConfig

# Optional heavy deps for topic modeling
try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    _SKLEARN_AVAILABLE = True
except Exception:  # ImportError or other env issues
    _SKLEARN_AVAILABLE = False


class MemoryAnalytics:
    """
    Advanced analytics for detecting memory drift and bias.
    """

    def __init__(self, graph):
        self._graph = graph
        self._drift_windows = [7, 30, 90]  # Days to track for drift
        # Initialize sentiment analyzer once
        self._sentiment_analyzer = SentimentIntensityAnalyzer()
        # Load analytics config (env-driven)
        try:
            cfg = MemoryConfig()
            self._analytics_cfg = cfg.analytics or {}
        except Exception:
            self._analytics_cfg = {}

    @staticmethod
    def _as_bool(val, default: bool) -> bool:
        if val is None:
            return default
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            v = val.strip().lower()
            if v in {"true", "1", "yes", "on"}:
                return True
            if v in {"false", "0", "no", "off"}:
                return False
        return default

    @staticmethod
    def _as_int(val, default: int) -> int:
        try:
            return int(val)
        except Exception:
            return default

    @staticmethod
    def _as_float(val, default: float) -> float:
        try:
            return float(val)
        except Exception:
            return default

    def detect_concept_drift(self, time_window_days: int = 30) -> Dict:
        """
        Detect concept drift by comparing recent memory patterns with historical data.
        
        Args:
            time_window_days: Number of recent days to consider as the 'current' period
            
        Returns:
            Dict containing drift scores and significant changes
        """
        now = datetime.now()
        recent_cutoff = now - timedelta(days=time_window_days)

        # Get recent and historical items
        # search_nodes() returns MemoryItem objects (via SmartGraph) or dicts (via raw backend).
        # Normalize to dicts so downstream code can use .get() and `in` uniformly.
        all_items = [item.to_dict() if hasattr(item, 'to_dict') else item
                     for item in self._graph.search_nodes({})]
        recent_items = [
            item for item in all_items
            if self._get_item_timestamp(item) >= recent_cutoff
        ]
        historical_items = [
            item for item in all_items
            if self._get_item_timestamp(item) < recent_cutoff
        ]

        if not historical_items or not recent_items:
            return {"status": "insufficient_data", "message": "Not enough data for drift detection"}

        # Simple keyword-based drift detection
        recent_keywords = self._extract_keywords(recent_items)
        historical_keywords = self._extract_keywords(historical_items)

        # Calculate KL divergence between keyword distributions
        all_keywords = set(recent_keywords) | set(historical_keywords)
        p = np.array([recent_keywords.get(k, 1e-10) for k in all_keywords])
        q = np.array([historical_keywords.get(k, 1e-10) for k in all_keywords])

        # Normalize to probability distributions
        p = p / p.sum()
        q = q / q.sum()

        # Calculate metrics
        kl_divergence = stats.entropy(p, q)
        js_distance = self._jensen_shannon(p, q)

        # Identify significant changes
        significant_changes = {
            k: (recent_keywords.get(k, 0), historical_keywords.get(k, 0))
            for k in set(recent_keywords) | set(historical_keywords)
            if abs((recent_keywords.get(k, 0) - historical_keywords.get(k, 0)) /
                   (historical_keywords.get(k, 1) or 1)) > 0.5  # 50% change threshold
        }

        return {
            "status": "success",
            "kl_divergence": float(kl_divergence),
            "js_distance": float(js_distance),
            "significant_changes": significant_changes,
            "recent_item_count": len(recent_items),
            "historical_item_count": len(historical_items),
            "time_window_days": time_window_days
        }

    def find_similar_items(self, embedding, top_k: int = 5, prop_key: str = 'embedding') -> List[Dict]:
        """Return top_k items most similar to the given embedding.

        Args:
            embedding: Vector embedding of the query item.
            top_k: How many similar items to return.
            prop_key: Property key under which embeddings are stored in node properties.

        Returns:
            List of item dicts augmented with a `similarity` score (1.0 is identical).
        """
        # vector_similarity_search lives on the backend (e.g. FalkorDBBackend),
        # not on SmartGraph itself.  Check backend first, then _graph for backward
        # compatibility when a raw backend is passed as the graph argument.
        backend = getattr(self._graph, "backend", None)
        if backend is not None and hasattr(backend, "vector_similarity_search"):
            results = backend.vector_similarity_search(embedding, top_k, prop_key)
        elif hasattr(self._graph, "vector_similarity_search"):
            results = self._graph.vector_similarity_search(embedding, top_k, prop_key)
        else:
            return []
        out: List[Dict] = []
        for item_id, score in results:
            node = self._graph.get_node(item_id)
            if node is None:
                continue
            if hasattr(node, 'dict'):
                node_dict = node.to_dict()
            else:
                node_dict = node
            node_dict['similarity'] = score
            out.append(node_dict)
        return out

    def detect_bias(self, protected_attributes: List[str] = None,
                    sentiment_analysis: bool | None = None,
                    topic_analysis: bool | None = None) -> Dict:
        """
        Detect potential bias in memory content using multiple analysis techniques.
        
        Args:
            protected_attributes: List of attributes to check for bias (e.g., ['gender', 'race'])
            sentiment_analysis: Whether to perform sentiment analysis for bias detection
            topic_analysis: Whether to analyze topic distribution across protected attributes
            
        Returns:
            Dict containing comprehensive bias metrics and analysis
        """
        if protected_attributes is None:
            # From config default or fallback
            attr_default = (
                (self._analytics_cfg.get('bias') or {} or {}).get('protected_attributes_default')
                if isinstance(self._analytics_cfg, dict) else None
            )
            if isinstance(attr_default, str):
                protected_attributes = [a.strip() for a in attr_default.split(',') if a.strip()]
            elif isinstance(attr_default, list):
                protected_attributes = attr_default
            else:
                protected_attributes = ['gender', 'race', 'age_group', 'nationality', 'religion']

        items = [item.to_dict() if hasattr(item, 'to_dict') else item
                 for item in self._graph.search_nodes({})]
        if not items:
            return {"status": "no_data", "message": "No items found for analysis"}

        # Basic attribute-based bias analysis
        bias_report = {}
        for attr in protected_attributes:
            bias_report[attr] = self._analyze_attribute_bias(items, attr)

        # Determine defaults from config
        sent_enabled_default = self._as_bool((self._analytics_cfg.get('sentiment') or {} or {}).get('enabled'), True)
        topic_enabled_default = self._as_bool((self._analytics_cfg.get('topic') or {} or {}).get('enabled'), True)

        # Sentiment bias analysis
        sentiment_bias = {}
        if self._as_bool(sentiment_analysis, sent_enabled_default):
            for attr in protected_attributes:
                sentiment_bias[attr] = self._analyze_sentiment_bias(items, attr)

        # Topic modeling for bias detection
        topic_bias = {}
        # Need sufficient data for meaningful topics
        min_docs = self._as_int((self._analytics_cfg.get('topic') or {} or {}).get('min_docs'), 10)
        if self._as_bool(topic_analysis, topic_enabled_default) and len(items) > max(9, min_docs - 1):
            topic_bias = self._analyze_topic_bias(items, protected_attributes)

        # Statistical bias metrics
        statistical_bias = self._calculate_statistical_biases(items, protected_attributes)

        return {
            "status": "success",
            "attribute_bias": bias_report,
            "sentiment_bias": sentiment_bias,
            "topic_bias": topic_bias,
            "statistical_biases": statistical_bias,
            "total_items_analyzed": len(items),
            "analysis_timestamp": datetime.now(timezone.utc).isoformat()
        }

    def _analyze_attribute_bias(self, items: List[Dict], attribute: str) -> Dict:
        """
        Analyze bias for a specific attribute using multiple detection methods.
        
        Args:
            items: List of memory items to analyze
            attribute: The attribute to analyze (e.g., 'gender', 'race')
            
        Returns:
            Dict containing bias metrics for the specified attribute
        """
        # Initialize counters and trackers
        counts = Counter()
        sentiment_scores = defaultdict(list)

        # Define attribute detection patterns (simplified - consider NER in production)
        attribute_patterns = {
            'gender': {
                'male': [r'\bhe\b', r'\bhim\b', r'\bhis\b', 'man', 'boy', 'male'],
                'female': [r'\bshe\b', r'\bher\b', r'\bhers\b', 'woman', 'girl', 'female']
            },
            'race': {
                'white': ['caucasian', 'white', 'european'],
                'black': ['african american', 'black', 'afro-'],
                'asian': ['asian', 'chinese', 'japanese', 'korean', 'vietnamese'],
                'hispanic': ['hispanic', 'latino', 'latina', 'latinx']
            },
            'age_group': {
                'child': ['child', 'kid', 'baby', 'toddler', 'youngster'],
                'teen': ['teen', 'adolescent', 'youth', 'young adult'],
                'adult': ['adult', 'grown-up', 'mature'],
                'senior': ['senior', 'elderly', 'retired', 'pensioner']
            },
            'nationality': {
                'us': ['american', 'u\.s\.', 'usa', 'united states'],
                'uk': ['british', 'english', 'u\.k\.', 'united kingdom'],
                'european': ['european', 'europe'],
                'asian': ['asian', 'east asian', 'south asian']
            },
            'religion': {
                'christian': ['christian', 'catholic', 'protestant', 'baptist'],
                'muslim': ['muslim', 'islam', 'moslem'],
                'jewish': ['jewish', 'judaism', 'hebrew'],
                'hindu': ['hindu', 'hinduism'],
                'buddhist': ['buddhist', 'buddhism']
            }
        }

        # Analyze each item
        for item in items:
            content = str(item.get("content", "")).lower()

            # Skip empty content
            if not content.strip():
                continue

            # Check for attribute matches
            if attribute in attribute_patterns:
                for value, patterns in attribute_patterns[attribute].items():
                    if any(re.search(pattern, content) for pattern in patterns):
                        counts[value] += 1

                        # Sentiment analysis for this mention
                        sentiment = self._get_sentiment(content)
                        sentiment_scores[value].append(sentiment)

        # Calculate statistics
        total = sum(counts.values())
        distribution = {k: v / total for k, v in counts.items()} if total > 0 else {}

        # Calculate sentiment bias
        sentiment_biases = {}
        if sentiment_scores:
            for value, scores in sentiment_scores.items():
                if scores:
                    sentiment_biases[value] = {
                        'mean_sentiment': np.mean(scores),
                        'std_sentiment': np.std(scores),
                        'count': len(scores)
                    }

        # Calculate representation fairness
        fairness_metrics = {}
        if len(counts) > 1:
            fairness_metrics = {
                'entropy': stats.entropy(list(counts.values())) if counts else 0,
                'gini_coefficient': self._gini(list(counts.values())),
                'relative_entropy': self._relative_entropy(
                    [c / total for c in counts.values()],
                    [1 / len(counts)] * len(counts)  # Uniform distribution
                )
            }

        return {
            'counts': dict(counts),
            'distribution': distribution,
            'fairness_metrics': fairness_metrics,
            'sentiment_biases': sentiment_biases,
            'attribute': attribute,
            'total_mentions': total
        }

    def _extract_keywords(self, items: List[Dict], top_n: int = 100) -> Dict[str, float]:
        """Extract and rank keywords from items."""
        word_counts = Counter()
        for item in items:
            content = str(item.get("content", "")).lower()
            # Simple word splitting - consider using TF-IDF or similar in production
            words = [w for w in content.split() if len(w) > 3]  # Filter out short words
            word_counts.update(words)

        total = sum(word_counts.values())
        return {k: v / total for k, v in word_counts.most_common(top_n)}

    def _analyze_sentiment_bias(self, items: List[Dict], attribute: str) -> Dict:
        """
        Analyze sentiment bias across different attribute values.
        
        Args:
            items: List of memory items
            attribute: Attribute to analyze sentiment bias for
            
        Returns:
            Dict containing sentiment bias analysis
        """
        sentiment_by_value = defaultdict(list)

        for item in items:
            content = str(item.get("content", "")).lower()
            if not content.strip():
                continue

            # Get sentiment score (-1 to 1)
            sentiment = self._get_sentiment(content)

            # Get attribute values mentioned in the content
            values = self._extract_attribute_values(content, attribute)

            # Record sentiment for each mentioned value
            for value in values:
                sentiment_by_value[value].append(sentiment)

        # Calculate statistics
        result = {}
        for value, sentiments in sentiment_by_value.items():
            if sentiments:
                result[value] = {
                    'count': len(sentiments),
                    'mean_sentiment': float(np.mean(sentiments)),
                    'std_sentiment': float(np.std(sentiments)),
                    'min_sentiment': float(min(sentiments)),
                    'max_sentiment': float(max(sentiments))
                }

        return result

    def _analyze_topic_bias(self, items: List[Dict], attributes: List[str]) -> Dict:
        """
        Analyze topic distribution across different attribute values.
        
        Args:
            items: List of memory items
            attributes: Attributes to analyze topic distribution for
            
        Returns:
            Dict containing topic bias analysis
        """
        # Guard: ensure sklearn is available and we have sufficient data
        if not _SKLEARN_AVAILABLE:
            return {
                'status': 'unavailable',
                'message': 'scikit-learn is not available; install scikit-learn to enable topic modeling'
            }

        # Collect documents
        docs: List[str] = []
        doc_attr_values: List[Dict[str, List[str]]] = []
        for item in items:
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            docs.append(content)
            values_per_attr: Dict[str, List[str]] = {}
            for attr in attributes:
                values_per_attr[attr] = self._extract_attribute_values(content.lower(), attr)
            doc_attr_values.append(values_per_attr)

        if len(docs) < 10:
            return {"status": "insufficient_data", "message": "Need at least 10 documents for topic modeling", "doc_count": len(docs)}

        # Vectorize (configurable)
        topic_cfg = (self._analytics_cfg.get('topic') or {} or {})
        min_df = self._as_int(topic_cfg.get('min_df'), 2)
        max_features = self._as_int(topic_cfg.get('max_features'), 20000)
        vectorizer = CountVectorizer(stop_words='english', min_df=min_df, max_features=max_features)
        X = vectorizer.fit_transform(docs)

        # Choose number of topics (configurable)
        n_components_cfg = self._as_int(topic_cfg.get('n_topics'), 0)
        if n_components_cfg > 0:
            n_components = n_components_cfg
        else:
            n_components = 15 if len(docs) >= 200 else 10
        lda = LatentDirichletAllocation(
            n_components=n_components,
            learning_method='online',
            random_state=0,
            n_jobs=None
        )
        doc_topic = lda.fit_transform(X)  # shape: (n_docs, n_topics)

        # Build topic descriptors
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for k, topic_dist in enumerate(lda.components_):
            top_idx = np.argsort(topic_dist)[-10:][::-1]
            topics.append({
                'id': int(k),
                'top_terms': [str(feature_names[i]) for i in top_idx],
            })

        # Aggregate per-attribute group distributions
        attr_results: Dict[str, Dict] = {}
        for attr in attributes:
            # Map value -> list of doc_topic rows
            buckets: Dict[str, List[np.ndarray]] = defaultdict(list)
            for i, vals in enumerate(doc_attr_values):
                values = vals.get(attr, [])
                for v in values:
                    buckets[v].append(doc_topic[i])

            if not buckets:
                continue

            # Compute mean topic distribution per value
            group_distributions: Dict[str, List[float]] = {}
            for value, rows in buckets.items():
                mat = np.vstack(rows)
                mean_dist = mat.mean(axis=0)
                mean_dist = mean_dist / (mean_dist.sum() + 1e-12)
                group_distributions[value] = [float(x) for x in mean_dist]

            # Pairwise JS divergence between groups
            values = list(group_distributions.keys())
            divergences: Dict[str, float] = {}
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    p = np.array(group_distributions[values[i]], dtype=np.float64)
                    q = np.array(group_distributions[values[j]], dtype=np.float64)
                    divergences[f"{values[i]}||{values[j]}"] = float(self._jensen_shannon(p, q))

            # Identify topics with highest variance across groups
            # Compute variance across groups per topic index
            if len(group_distributions) > 1:
                group_mat = np.array(list(group_distributions.values()))  # (n_groups, n_topics)
                topic_variance = group_mat.var(axis=0)
                top_bias_topic_idx = np.argsort(topic_variance)[-5:][::-1]
                biased_topics = [int(idx) for idx in top_bias_topic_idx]
            else:
                topic_variance = np.zeros(doc_topic.shape[1], dtype=np.float64)
                biased_topics = []

            attr_results[attr] = {
                'groups': group_distributions,
                'divergences_js': divergences,
                'biased_topics': biased_topics,
            }

        return {
            'status': 'success',
            'n_docs': len(docs),
            'n_topics': int(n_components),
            'topics': topics,
            'by_attribute': attr_results,
        }

    def _calculate_statistical_biases(self, items: List[Dict], attributes: List[str]) -> Dict:
        """
        Calculate various statistical bias metrics.
        
        Args:
            items: List of memory items
            attributes: Attributes to calculate biases for
            
        Returns:
            Dict containing statistical bias metrics
        """
        results = {}

        for attr in attributes:
            # Count occurrences of each attribute value
            counts = Counter()
            for item in items:
                content = str(item.get("content", "")).lower()
                values = self._extract_attribute_values(content, attr)
                counts.update(values)

            if not counts:
                continue

            # Calculate various bias metrics
            values = list(counts.values())
            total = sum(values)
            proportions = [v / total for v in values]

            results[attr] = {
                'entropy': float(stats.entropy(proportions)),
                'gini': float(self._gini(values)),
                'max_min_ratio': max(values) / min(values) if min(values) > 0 else float('inf'),
                'counts': dict(counts),
                'proportions': {k: v / total for k, v in counts.items()}
            }

        return results

    def _extract_attribute_values(self, content: str, attribute: str) -> List[str]:
        """Extract values for a given attribute from content."""
        # This is a simplified implementation - in practice, use NER or similar
        values = []

        # Define patterns for different attributes
        patterns = {
            'gender': {
                'male': [r'\bhe\b', r'\bhim\b', r'\bhis\b', 'man', 'boy', 'male'],
                'female': [r'\bshe\b', r'\bher\b', r'\bhers\b', 'woman', 'girl', 'female']
            },
            # Add more attribute patterns as needed
        }

        if attribute in patterns:
            for value, value_patterns in patterns[attribute].items():
                if any(re.search(p, content) for p in value_patterns):
                    values.append(value)

        return values

    @staticmethod
    def _get_sentiment(text: str) -> float:
        """
        Get sentiment score for text (-1 to 1).
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score between -1 (negative) and 1 (positive)
        """
        # Use VADER compound score in [-1, 1]
        # Note: although static, we access the analyzer via a lightweight singleton pattern
        # to keep interface stable without refactoring call sites.
        if not hasattr(MemoryAnalytics, "_vader_singleton"):
            try:
                MemoryAnalytics._vader_singleton = SentimentIntensityAnalyzer()
            except Exception:
                MemoryAnalytics._vader_singleton = None

        analyzer = getattr(MemoryAnalytics, "_vader_singleton", None)
        if analyzer is None:
            return 0.0
        score = analyzer.polarity_scores(text or "")
        return float(score.get('compound', 0.0))

    @staticmethod
    def _gini(x: List[float]) -> float:
        """Calculate Gini coefficient of a distribution."""
        if not x:
            return 0.0

        x = np.array(x, dtype=np.float64)
        x = np.sort(x)
        n = len(x)
        index = np.arange(1, n + 1)
        return float((np.sum((2 * index - n - 1) * x)) / (n * np.sum(x)))

    @staticmethod
    def _relative_entropy(p: List[float], q: List[float]) -> float:
        """Calculate relative entropy (KL divergence) between two distributions."""
        p = np.array(p, dtype=np.float64)
        q = np.array(q, dtype=np.float64)

        # Add small constant to avoid division by zero
        p = p + 1e-10
        q = q + 1e-10

        # Normalize
        p = p / p.sum()
        q = q / q.sum()

        return float(np.sum(p * np.log(p / q)))

    @staticmethod
    def _jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate Jensen-Shannon divergence between two distributions.
        
        Args:
            p: First probability distribution
            q: Second probability distribution
            
        Returns:
            JS divergence between p and q
        """
        m = 0.5 * (p + q)
        return float(0.5 * (stats.entropy(p, m) + stats.entropy(q, m)))

    def _get_item_timestamp(self, item: Dict) -> datetime:
        """Extract timestamp from item, falling back to current time if not found."""
        for field in ['created_at', 'timestamp', 'date']:
            if field in item:
                ts = item[field]
                if isinstance(ts, str):
                    try:
                        return datetime.fromisoformat(ts)
                    except (ValueError, TypeError):
                        continue
                elif isinstance(ts, (int, float)):
                    return datetime.fromtimestamp(ts)
                elif isinstance(ts, datetime):
                    return ts
        return datetime.now()  # Default to current time if no timestamp found
