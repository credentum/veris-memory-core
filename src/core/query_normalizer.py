"""
Semantic query normalization for consistent retrieval.

Normalizes semantically equivalent queries to canonical forms
to ensure consistent caching and retrieval behavior.

This addresses the S3-Paraphrase-Robustness issue where different
phrasings of the same query would take different code paths and
return different results.
"""

import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Detected query intents."""

    CONFIGURATION = "configuration"
    TROUBLESHOOTING = "troubleshooting"
    HOWTO = "howto"
    CONCEPTUAL = "conceptual"
    LOOKUP = "lookup"
    UNKNOWN = "unknown"


@dataclass
class NormalizedQuery:
    """Result of query normalization."""

    original: str
    normalized: str
    intent: QueryIntent
    entities: List[str]
    confidence: float


@dataclass
class QueryNormalizerConfig:
    """Configuration for query normalizer."""

    enabled: bool = True
    confidence_threshold: float = 0.5
    metrics_enabled: bool = True


class QueryNormalizer:
    """
    Normalize queries to canonical semantic forms.

    Uses pattern matching, keyword detection, and intent classification
    to map paraphrased queries to consistent forms. This ensures that
    semantically equivalent queries are treated identically throughout
    the retrieval pipeline.

    Example:
        "How to setup Neo4j?" -> "How do I configure Neo4j database settings?"
        "What are the steps to set up Neo4j database configuration?" ->
            "How do I configure Neo4j database settings?"
    """

    def __init__(self, config: Optional[QueryNormalizerConfig] = None):
        """
        Initialize the query normalizer.

        Args:
            config: Configuration options. If None, uses defaults from environment.
        """
        if config is None:
            config = QueryNormalizerConfig(
                enabled=os.getenv("QUERY_NORMALIZATION_ENABLED", "true").lower() == "true",
                confidence_threshold=float(os.getenv("QUERY_NORMALIZATION_CONFIDENCE", "0.5")),
            )

        self.config = config
        self._metrics = {
            "total_normalizations": 0,
            "queries_normalized": 0,
            "queries_unchanged": 0,
            "intent_counts": {intent.value: 0 for intent in QueryIntent},
            "average_confidence": 0.0,
        }

        # Canonical query patterns (intent -> keywords -> canonical form)
        self.canonical_patterns: Dict[QueryIntent, Dict[str, str]] = {
            QueryIntent.CONFIGURATION: {
                "neo4j": "How do I configure Neo4j database settings?",
                "neo4j database": "How do I configure Neo4j database settings?",
                "neo4j connection": "How do I configure Neo4j database connection?",
                "database connection": "How do I configure database connection settings?",
                "qdrant": "How do I configure Qdrant vector database?",
                "redis": "How do I configure Redis cache settings?",
                "redis cache": "How do I configure Redis cache settings?",
                "embedding": "How do I configure embedding model settings?",
                "embedding model": "How do I configure embedding model settings?",
                "veris memory": "How do I configure Veris Memory?",
            },
            QueryIntent.TROUBLESHOOTING: {
                "neo4j timeout": "How do I troubleshoot Neo4j connection timeout errors?",
                "neo4j error": "How do I troubleshoot Neo4j errors?",
                "connection error": "How do I troubleshoot database connection errors?",
                "connection timeout": "How do I troubleshoot connection timeout errors?",
                "performance": "How do I troubleshoot performance issues?",
                "slow query": "How do I troubleshoot slow query performance?",
            },
            QueryIntent.HOWTO: {
                "store context": "How do I store context in Veris Memory?",
                "save context": "How do I store context in Veris Memory?",
                "retrieve context": "How do I retrieve context from Veris Memory?",
                "get context": "How do I retrieve context from Veris Memory?",
                "query graph": "How do I query the Neo4j graph database?",
                "search": "How do I search for context in Veris Memory?",
            },
            QueryIntent.CONCEPTUAL: {
                "microservices": "What are microservices and their benefits?",
                "vector database": "What is a vector database?",
                "embedding": "What are embeddings in machine learning?",
                "mcp protocol": "What is the MCP protocol?",
            },
        }

        # Intent detection patterns (regex patterns for each intent)
        self.intent_patterns: Dict[QueryIntent, List[str]] = {
            QueryIntent.CONFIGURATION: [
                r"\bconfigur\w*\b",
                r"\bset\s*up\b",
                r"\bsettings?\b",
                r"\bparameters?\b",
                r"\binitializ\w*\b",
                r"\bsetup\b",
            ],
            QueryIntent.TROUBLESHOOTING: [
                r"\btroubleshoot\w*\b",
                r"\bfix\b",
                r"\bresolv\w*\b",
                r"\berrors?\b",
                r"\bfail\w*\b",
                r"\bissues?\b",
                r"\bproblems?\b",
                r"\bdebug\w*\b",
            ],
            QueryIntent.HOWTO: [
                r"\bhow\s+(?:to|do|can)\b",
                r"\bsteps?\s+to\b",
                r"\bguide\b",
                r"\btutorial\b",
                r"\bwalk\s*through\b",
            ],
            QueryIntent.CONCEPTUAL: [
                r"\bwhat\s+(?:is|are)\b",
                r"\bexplain\b",
                r"\bdescribe\b",
                r"\bunderstand\b",
                r"\bdefin\w*\b",
            ],
            QueryIntent.LOOKUP: [
                r"\bfind\b",
                r"\bsearch\b",
                r"\blook\s*up\b",
                r"\bget\b",
                r"\bretrieve\b",
                r"\blist\b",
            ],
        }

        # Entity extraction patterns (technical terms to detect)
        self.entity_patterns: List[str] = [
            r"\bneo4j\b",
            r"\bqdrant\b",
            r"\bredis\b",
            r"\bmcp\b",
            r"\bveris\s*memory\b",
            r"\bembedding\b",
            r"\bvector\b",
            r"\bgraph\b",
            r"\bdatabase\b",
            r"\bcontext\b",
            r"\bquery\b",
            r"\bsearch\b",
            r"\bconnection\b",
            r"\bcache\b",
        ]

        logger.info(
            f"QueryNormalizer initialized: enabled={config.enabled}, "
            f"confidence_threshold={config.confidence_threshold}"
        )

    def normalize(self, query: str) -> NormalizedQuery:
        """
        Normalize a query to its canonical form.

        Args:
            query: Raw user query

        Returns:
            NormalizedQuery with normalized form and metadata
        """
        self._metrics["total_normalizations"] += 1

        if not self.config.enabled:
            return NormalizedQuery(
                original=query,
                normalized=query,
                intent=QueryIntent.UNKNOWN,
                entities=[],
                confidence=0.0,
            )

        query_lower = query.lower().strip()

        # Detect intent
        intent, intent_confidence = self._detect_intent(query_lower)
        self._metrics["intent_counts"][intent.value] += 1

        # Extract entities
        entities = self._extract_entities(query_lower)

        # Find canonical form
        canonical, match_confidence = self._find_canonical(query_lower, intent, entities)

        # Calculate overall confidence
        confidence = (intent_confidence + match_confidence) / 2

        # Update metrics
        self._update_average_confidence(confidence)

        # Determine if normalization should be applied
        if confidence > self.config.confidence_threshold and canonical != query:
            normalized_query = canonical
            self._metrics["queries_normalized"] += 1
        else:
            normalized_query = query
            self._metrics["queries_unchanged"] += 1

        logger.debug(
            f"Query normalization: '{query[:50]}...' -> '{normalized_query[:50]}...' "
            f"(intent={intent.value}, entities={entities}, confidence={confidence:.2f})"
        )

        return NormalizedQuery(
            original=query,
            normalized=normalized_query,
            intent=intent,
            entities=entities,
            confidence=confidence,
        )

    def _detect_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """
        Detect the intent of a query.

        Args:
            query: Lowercased query string

        Returns:
            Tuple of (detected intent, confidence score)
        """
        intent_scores: Dict[QueryIntent, float] = {}

        for intent, patterns in self.intent_patterns.items():
            matches = sum(
                1 for pattern in patterns if re.search(pattern, query, re.IGNORECASE)
            )
            if patterns:
                score = matches / len(patterns)
                intent_scores[intent] = score

        if not intent_scores or max(intent_scores.values()) == 0:
            return QueryIntent.UNKNOWN, 0.0

        best_intent = max(intent_scores, key=lambda k: intent_scores[k])
        return best_intent, intent_scores[best_intent]

    def _extract_entities(self, query: str) -> List[str]:
        """
        Extract technical entities from query.

        Args:
            query: Lowercased query string

        Returns:
            List of detected entity names
        """
        entities = []
        for pattern in self.entity_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities.extend([m.lower().strip() for m in matches])
        return list(set(entities))

    def _find_canonical(
        self,
        query: str,
        intent: QueryIntent,
        entities: List[str],
    ) -> Tuple[str, float]:
        """
        Find the canonical form for the query.

        Args:
            query: Lowercased query string
            intent: Detected query intent
            entities: List of detected entities

        Returns:
            Tuple of (canonical query form, match confidence)
        """
        if intent not in self.canonical_patterns:
            return query, 0.0

        intent_patterns = self.canonical_patterns[intent]

        best_match: Optional[str] = None
        best_score: float = 0.0

        for key, canonical in intent_patterns.items():
            # Score based on keyword presence
            keywords = key.lower().split()
            matches = sum(1 for kw in keywords if kw in query)
            score = matches / len(keywords) if keywords else 0.0

            # Bonus for entity overlap
            canonical_lower = canonical.lower()
            entity_bonus = sum(0.1 for entity in entities if entity in canonical_lower)

            total_score = min(1.0, score + entity_bonus)

            if total_score > best_score:
                best_score = total_score
                best_match = canonical

        return best_match or query, best_score

    def _update_average_confidence(self, new_confidence: float) -> None:
        """Update running average confidence."""
        total = self._metrics["total_normalizations"]
        if total > 0:
            current_avg = self._metrics["average_confidence"]
            self._metrics["average_confidence"] = (
                current_avg * (total - 1) + new_confidence
            ) / total

    def get_metrics(self) -> Dict:
        """Get normalization metrics."""
        total = self._metrics["total_normalizations"]
        return {
            **self._metrics,
            "normalization_rate": (
                self._metrics["queries_normalized"] / total if total > 0 else 0.0
            ),
            "config": {
                "enabled": self.config.enabled,
                "confidence_threshold": self.config.confidence_threshold,
            },
        }

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self._metrics = {
            "total_normalizations": 0,
            "queries_normalized": 0,
            "queries_unchanged": 0,
            "intent_counts": {intent.value: 0 for intent in QueryIntent},
            "average_confidence": 0.0,
        }


# Global instance
_query_normalizer: Optional[QueryNormalizer] = None


def get_query_normalizer() -> QueryNormalizer:
    """
    Get or create the global query normalizer.

    Returns:
        Global QueryNormalizer instance
    """
    global _query_normalizer
    if _query_normalizer is None:
        _query_normalizer = QueryNormalizer()
    return _query_normalizer


def reset_query_normalizer() -> None:
    """Reset the global query normalizer (useful for testing)."""
    global _query_normalizer
    _query_normalizer = None
