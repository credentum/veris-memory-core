#!/usr/bin/env python3
"""
Query-Specific Relevance Scorer for Veris Memory

This module addresses LIM-004 from GitHub issue #127:
"Search returns generic results regardless of specific query terms"

The scorer analyzes query patterns and applies query-specific relevance scoring
to ensure results vary meaningfully based on the actual search terms.
"""

import re
import math
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries based on intent and content."""
    TECHNICAL_SPECIFIC = "technical_specific"  # "redis hash leaderboard queue"
    CODE_SEARCH = "code_search"  # "function getUserData"
    CONCEPT_LOOKUP = "concept_lookup"  # "how to implement authentication"
    TROUBLESHOOTING = "troubleshooting"  # "error connecting to database"
    GENERIC = "generic"  # "help with project"


@dataclass
class QueryAnalysis:
    """Analysis results for a query."""
    query_type: QueryType
    key_terms: Set[str]
    technical_terms: Set[str]
    intent_strength: float  # 0.0-1.0, how specific the query is
    domain_focus: Optional[str]  # e.g., "database", "authentication", "storage"


class QueryRelevanceScorer:
    """
    Query-specific relevance scorer that understands query intent
    and adjusts result scoring accordingly.
    """
    
    def __init__(self):
        # Technical domain mappings
        self.domain_terms = {
            "database": {
                "redis", "sql", "nosql", "query", "schema", "table", "index", 
                "mongodb", "postgresql", "mysql", "database", "db", "cache",
                "hash", "set", "list", "sorted", "key", "value", "store"
            },
            "authentication": {
                "auth", "login", "token", "jwt", "session", "cookie", "oauth",
                "password", "user", "permission", "role", "access", "security"
            },
            "api": {
                "api", "endpoint", "rest", "graphql", "http", "https", "request",
                "response", "json", "xml", "webhook", "route", "controller"
            },
            "storage": {
                "storage", "file", "disk", "memory", "cache", "persist", "save",
                "load", "backup", "sync", "async", "queue", "buffer"
            },
            "search": {
                "search", "query", "index", "rank", "relevance", "match",
                "filter", "sort", "vector", "embedding", "similarity"
            },
            "infrastructure": {
                "docker", "kubernetes", "deployment", "container", "service",
                "server", "client", "proxy", "load", "balancer", "nginx"
            }
        }
        
        # Code-specific patterns
        self.code_patterns = [
            r'\b[a-zA-Z_][a-zA-Z0-9_]*\(\)',  # function calls
            r'\bclass\s+[A-Z][a-zA-Z0-9_]*',  # class definitions
            r'\bdef\s+[a-z][a-zA-Z0-9_]*',    # function definitions
            r'\b[A-Z_][A-Z0-9_]*\b',          # constants
            r'\.[a-z][a-zA-Z0-9_]*',          # method calls
            r'\b[a-z][a-zA-Z0-9_]*\.[a-z]',   # module.method
        ]
        
        # Troubleshooting indicators
        self.trouble_terms = {
            "error", "bug", "issue", "problem", "fail", "broken", "crash",
            "exception", "traceback", "debug", "fix", "solve", "resolve"
        }

    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze a query to understand its type and intent.
        
        Args:
            query: The search query string
            
        Returns:
            QueryAnalysis with query type and key terms
        """
        query_lower = query.lower().strip()
        words = set(re.findall(r'\b\w+\b', query_lower))
        
        # Detect query type
        query_type = self._detect_query_type(query, words)
        
        # Extract key terms (remove common stop words)
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can"}
        key_terms = words - stop_words
        
        # Identify technical terms
        technical_terms = set()
        for domain, domain_words in self.domain_terms.items():
            technical_terms.update(words.intersection(domain_words))
        
        # Calculate intent strength (how specific the query is)
        intent_strength = self._calculate_intent_strength(query, key_terms, technical_terms)
        
        # Determine domain focus
        domain_focus = self._determine_domain_focus(words)
        
        return QueryAnalysis(
            query_type=query_type,
            key_terms=key_terms,
            technical_terms=technical_terms,
            intent_strength=intent_strength,
            domain_focus=domain_focus
        )

    def _detect_query_type(self, query: str, words: Set[str]) -> QueryType:
        """Detect the type of query based on patterns and terms."""
        
        # Check for code patterns
        for pattern in self.code_patterns:
            if re.search(pattern, query):
                return QueryType.CODE_SEARCH
        
        # Check for troubleshooting terms
        if words.intersection(self.trouble_terms):
            return QueryType.TROUBLESHOOTING
        
        # Check for technical specificity
        technical_count = 0
        for domain_words in self.domain_terms.values():
            technical_count += len(words.intersection(domain_words))
        
        if technical_count >= 3:  # Multiple technical terms
            return QueryType.TECHNICAL_SPECIFIC
        elif technical_count >= 1:
            return QueryType.CONCEPT_LOOKUP
        elif len(words) <= 3 and not any(word in ["help", "how", "what", "why"] for word in words):
            return QueryType.CONCEPT_LOOKUP
        else:
            return QueryType.GENERIC

    def _calculate_intent_strength(self, query: str, key_terms: Set[str], technical_terms: Set[str]) -> float:
        """Calculate how specific/intentional the query is."""
        
        # Base strength from query length and complexity
        word_count = len(query.split())
        base_strength = min(word_count / 8.0, 0.8)  # Longer queries are more specific
        
        # Boost for technical terms
        tech_boost = min(len(technical_terms) * 0.25, 0.6)
        
        # Boost for specific patterns (especially code patterns)
        pattern_boost = 0.0
        if any(re.search(pattern, query) for pattern in self.code_patterns):
            pattern_boost = 0.4  # Higher boost for code patterns
        
        # Special boost for function-like patterns
        if re.search(r'\w+\(\)', query):
            pattern_boost = max(pattern_boost, 0.5)
        
        # Penalty for very generic terms
        generic_penalty = 0.0
        generic_words = {"help", "how", "what", "why", "best", "good", "bad"}
        if key_terms.intersection(generic_words):
            generic_penalty = 0.3
        
        strength = base_strength + tech_boost + pattern_boost - generic_penalty
        return max(0.0, min(1.0, strength))

    def _determine_domain_focus(self, words: Set[str]) -> Optional[str]:
        """Determine which technical domain the query focuses on."""
        domain_scores = {}
        
        for domain, domain_words in self.domain_terms.items():
            overlap = len(words.intersection(domain_words))
            if overlap > 0:
                domain_scores[domain] = overlap
        
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        return None

    def score_result_relevance(self, query_analysis: QueryAnalysis, result: Dict[str, Any]) -> float:
        """
        Score how relevant a result is to the analyzed query.
        
        Args:
            query_analysis: The analyzed query
            result: A search result dictionary
            
        Returns:
            Relevance score multiplier (1.0 = no change, >1.0 = more relevant)
        """
        
        # Extract result content
        payload = result.get('payload', {})
        content = payload.get('content', {})
        metadata = payload.get('metadata', {})
        
        # Convert content to text for analysis
        content_text = self._extract_text_from_content(content, metadata)
        content_words = set(re.findall(r'\b\w+\b', content_text.lower()))
        
        relevance_score = 1.0
        
        # 1. Key term matching - higher weight for query-specific terms
        key_term_matches = len(query_analysis.key_terms.intersection(content_words))
        if query_analysis.key_terms:
            key_term_ratio = key_term_matches / len(query_analysis.key_terms)
            relevance_score *= (1.0 + key_term_ratio * 2.0)  # Up to 3x boost
        
        # 2. Technical term matching - especially important for technical queries
        tech_term_matches = len(query_analysis.technical_terms.intersection(content_words))
        if query_analysis.technical_terms:
            tech_ratio = tech_term_matches / len(query_analysis.technical_terms)
            tech_boost = tech_ratio * query_analysis.intent_strength * 1.5
            relevance_score *= (1.0 + tech_boost)
        
        # 3. Domain focus matching
        if query_analysis.domain_focus:
            domain_terms = self.domain_terms[query_analysis.domain_focus]
            domain_matches = len(domain_terms.intersection(content_words))
            if domain_matches > 0:
                domain_boost = min(domain_matches * 0.3, 1.0)
                relevance_score *= (1.0 + domain_boost)
        
        # 4. Query type specific scoring
        type_boost = self._get_query_type_boost(query_analysis.query_type, content, metadata)
        relevance_score *= type_boost
        
        # 5. Penalize results that match too generically
        generic_penalty = self._calculate_generic_penalty(query_analysis, content_words)
        relevance_score *= generic_penalty
        
        return relevance_score

    def _extract_text_from_content(self, content: Any, metadata: Dict) -> str:
        """Extract searchable text from result content."""
        text_parts = []
        
        # Handle different content types
        if isinstance(content, dict):
            for key, value in content.items():
                if isinstance(value, str):
                    text_parts.append(value)
                elif key in ['title', 'description', 'summary']:
                    text_parts.append(str(value))
        elif isinstance(content, str):
            text_parts.append(content)
        else:
            text_parts.append(str(content))
        
        # Add metadata text
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                if key in ['title', 'description', 'tags', 'category', 'type']:
                    text_parts.append(str(value))
        
        return ' '.join(text_parts)

    def _get_query_type_boost(self, query_type: QueryType, content: Any, metadata: Dict) -> float:
        """Get boost factor based on query type and result type matching."""
        
        content_type = metadata.get('type', '').lower()
        content_str = str(content).lower()
        
        if query_type == QueryType.CODE_SEARCH:
            if content_type in ['code', 'log'] or 'function' in content_str or 'class' in content_str:
                return 1.5
            return 0.8
            
        elif query_type == QueryType.TECHNICAL_SPECIFIC:
            if content_type in ['code', 'log', 'trace'] or any(tech in content_str for tech in ['api', 'database', 'server']):
                return 1.3
            return 0.9
            
        elif query_type == QueryType.TROUBLESHOOTING:
            if content_type in ['log', 'trace'] or any(word in content_str for word in ['error', 'exception', 'fail']):
                return 1.4
            return 0.85
            
        elif query_type == QueryType.CONCEPT_LOOKUP:
            if content_type in ['design', 'decision'] or 'how to' in content_str:
                return 1.2
            return 0.95
        
        return 1.0

    def _calculate_generic_penalty(self, query_analysis: QueryAnalysis, content_words: Set[str]) -> float:
        """Penalize results that match too generically when query is specific."""
        
        if query_analysis.intent_strength < 0.3:
            return 1.0  # Don't penalize for generic queries
        
        # If query is specific but result only matches generic terms
        specific_matches = len(query_analysis.technical_terms.intersection(content_words))
        total_matches = len(query_analysis.key_terms.intersection(content_words))
        
        if total_matches > 0 and specific_matches == 0:
            # Only generic matches for a specific query
            penalty = 0.7  # 30% penalty
        elif total_matches > specific_matches * 3:
            # Mostly generic matches
            penalty = 0.85  # 15% penalty
        else:
            penalty = 1.0
        
        return penalty

    def enhance_search_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main method to enhance search results with query-specific relevance.
        
        Args:
            query: The search query
            results: List of search results
            
        Returns:
            Enhanced and re-ranked results
        """
        if not results:
            return results
        
        # Analyze the query
        query_analysis = self.analyze_query(query)
        logger.info(f"Query analysis: type={query_analysis.query_type.value}, intent_strength={query_analysis.intent_strength:.2f}, domain={query_analysis.domain_focus}")
        
        enhanced_results = []
        
        for result in results:
            # Calculate query-specific relevance
            relevance_multiplier = self.score_result_relevance(query_analysis, result)
            
            # Apply to existing score
            original_score = result.get('score', 0.0)
            enhanced_score = original_score * relevance_multiplier
            
            # Create enhanced result
            enhanced_result = result.copy()
            enhanced_result['query_relevance_score'] = enhanced_score
            enhanced_result['query_relevance_multiplier'] = relevance_multiplier
            enhanced_result['query_analysis'] = {
                'type': query_analysis.query_type.value,
                'intent_strength': query_analysis.intent_strength,
                'domain_focus': query_analysis.domain_focus,
                'key_terms': list(query_analysis.key_terms)[:5],  # Limit for response size
                'technical_terms': list(query_analysis.technical_terms)
            }
            
            enhanced_results.append(enhanced_result)
        
        # Re-sort by enhanced relevance score
        enhanced_results.sort(key=lambda x: x['query_relevance_score'], reverse=True)
        
        logger.info(f"Enhanced {len(results)} results with query-specific scoring")
        return enhanced_results