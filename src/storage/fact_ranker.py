"""
Fact-aware ranking system with answer priority boosts.

This module enhances the existing reranking pipeline with fact-specific
scoring to prioritize declarative answers over interrogative questions
and improve fact recall accuracy.
"""

import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import math
import logging

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Types of content for ranking purposes."""
    DECLARATIVE_ANSWER = "declarative_answer"
    INTERROGATIVE_QUESTION = "interrogative_question"
    MIXED_CONTENT = "mixed_content"
    NEUTRAL_CONTENT = "neutral_content"


@dataclass
class RankingResult:
    """Result with ranking metadata."""
    content: str
    original_score: float
    fact_boost: float
    final_score: float
    content_type: ContentType
    matched_patterns: List[str]
    metadata: Dict[str, Any]


class FactAwareRanker:
    """
    Enhanced ranker with fact-specific scoring patterns.
    
    Applies answer priority boosts to improve fact recall by promoting
    declarative statements over questions in search results.
    """

    def __init__(self) -> None:
        # Declarative patterns that should be boosted (answer-like content)
        self.boost_patterns = [
            # Personal identity declarations
            (r"\b(?:my name is|i'?m (?:called )?|call me|you can call me)\s+\w+", 0.15, "name_declaration"),
            (r"\bi go by\s+\w+", 0.12, "name_alias"),
            
            # Contact information declarations
            (r"\bmy email (?:address )?is\s+[\w._%+-]+@[\w.-]+\.\w+", 0.15, "email_declaration"),
            (r"\bmy (?:phone|number) is\s+[\d\s\-\(\)\+]+", 0.12, "phone_declaration"),
            
            # Location declarations
            (r"\bi live in\s+[A-Za-z\s,]+", 0.12, "location_declaration"),
            (r"\bi'?m (?:from|located in)\s+[A-Za-z\s,]+", 0.10, "origin_declaration"),
            
            # Preference declarations
            (r"\bi like\s+[^.!?]+", 0.10, "preference_declaration"),
            (r"\bi love\s+[^.!?]+", 0.08, "strong_preference"),
            (r"\bi prefer\s+[^.!?]+", 0.10, "preference_statement"),
            (r"\bmy favorite\s+\w+\s+is\s+[^.!?]+", 0.12, "favorite_declaration"),
            
            # Professional declarations
            (r"\bi work (?:as|at)\s+[^.!?]+", 0.10, "job_declaration"),
            (r"\bmy job is\s+[^.!?]+", 0.12, "job_statement"),
            (r"\bi'?m a\s+[^.!?]+", 0.08, "role_declaration"),
            
            # Age and personal info
            (r"\bi'?m\s+\d{1,3}\s+years old", 0.12, "age_declaration"),
            (r"\bmy age is\s+\d{1,3}", 0.12, "age_statement"),
            (r"\bi was born (?:in|on)\s+[^.!?]+", 0.10, "birth_declaration"),
            
            # General declarative patterns
            (r"\bmy\s+\w+\s+is\s+[^.!?]+", 0.08, "possession_declaration"),
            (r"\bi am\s+[^.!?]+", 0.06, "identity_statement"),
            (r"\bi have\s+[^.!?]+", 0.05, "possession_statement"),
        ]

        # Interrogative patterns that should be demoted (question-like content)
        self.demote_patterns = [
            # Direct fact questions
            (r"\bwhat'?s my name\b", -0.10, "name_question"),
            (r"\bwho am i\b", -0.08, "identity_question"),
            (r"\bwhat'?s my email\b", -0.10, "email_question"),
            (r"\bwhere do i live\b", -0.08, "location_question"),
            (r"\bhow old am i\b", -0.08, "age_question"),
            
            # General fact inquiry patterns
            (r"\bwhat (?:food|music) do i like\b", -0.08, "preference_question"),
            (r"\bwhat'?s my favorite\s+\w+", -0.08, "favorite_question"),
            (r"\bdo you (?:know|remember) my\s+\w+", -0.10, "recall_question"),
            (r"\bwhat do you call me\b", -0.08, "name_inquiry"),
            
            # Generic question patterns
            (r"\bwhat'?s my\s+\w+", -0.06, "generic_my_question"),
            (r"\bdo you know\s+(?:my|about)\s+\w+", -0.06, "knowledge_question"),
            (r"\bcan you tell me\s+(?:my|about)\s+\w+", -0.05, "request_question"),
            (r"\bwhat do i\s+\w+", -0.04, "activity_question"),
            
            # Question markers
            (r"\?\s*$", -0.03, "question_mark"),
            (r"\b(?:what|who|where|when|why|how)\b.*\?", -0.04, "wh_question"),
        ]

        # Compiled patterns for efficiency
        self._compiled_boost = [(re.compile(pattern, re.IGNORECASE), score, label) 
                               for pattern, score, label in self.boost_patterns]
        self._compiled_demote = [(re.compile(pattern, re.IGNORECASE), score, label) 
                                for pattern, score, label in self.demote_patterns]

    def apply_fact_ranking(self, results: List[Dict[str, Any]], query: str = "") -> List[RankingResult]:
        """
        Apply fact-aware ranking to search results.
        
        Args:
            results: List of search results with 'content' and 'score' fields
            query: Original query for context-aware ranking
            
        Returns:
            List of RankingResult objects with enhanced scoring
        """
        ranked_results = []
        
        for result in results:
            content = result.get('content', '')
            original_score = result.get('score', 0.0)
            
            # Apply fact-specific scoring
            ranking_result = self._score_content(content, original_score, query)
            ranking_result.metadata.update(result.get('metadata', {}))
            
            ranked_results.append(ranking_result)
        
        # Sort by final score (descending)
        ranked_results.sort(key=lambda x: x.final_score, reverse=True)
        
        logger.debug(f"Applied fact ranking to {len(results)} results")
        return ranked_results

    def _score_content(self, content: str, original_score: float, query: str = "") -> RankingResult:
        """Score individual content with fact-aware patterns."""
        boost_score = 0.0
        matched_patterns = []
        
        # Apply boost patterns (declarative statements)
        for pattern, score_delta, label in self._compiled_boost:
            if pattern.search(content):
                boost_score += score_delta
                matched_patterns.append(f"+{label}")
        
        # Apply demote patterns (interrogative questions)
        for pattern, score_delta, label in self._compiled_demote:
            if pattern.search(content):
                boost_score += score_delta  # score_delta is negative
                matched_patterns.append(f"-{label}")
        
        # Apply query-specific boosts
        query_boost = self._calculate_query_boost(content, query)
        boost_score += query_boost
        
        # Determine content type
        content_type = self._classify_content_type(content, matched_patterns)
        
        # Calculate final score
        final_score = original_score + boost_score
        
        # Apply score bounds
        final_score = max(0.0, min(1.0, final_score))
        
        return RankingResult(
            content=content,
            original_score=original_score,
            fact_boost=boost_score,
            final_score=final_score,
            content_type=content_type,
            matched_patterns=matched_patterns,
            metadata={}
        )

    def _calculate_query_boost(self, content: str, query: str) -> float:
        """Calculate query-specific boost for content."""
        if not query:
            return 0.0
        
        boost = 0.0
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Boost for query-answer alignment
        if re.search(r"\bwhat'?s my name\b", query_lower):
            if re.search(r'\bmy name is\b', content_lower):
                boost += 0.08
        
        if re.search(r"\bwhat'?s my email\b", query_lower):
            if re.search(r'\bmy email.*is\b', content_lower):
                boost += 0.08
        
        if re.search(r'\bwhere do i live\b', query_lower):
            if re.search(r'\bi live in\b', content_lower):
                boost += 0.08
        
        # General fact-query alignment
        if re.search(r'\bwhat.*my\b', query_lower):
            if re.search(r'\bmy.*is\b', content_lower):
                boost += 0.04
        
        return boost

    def _classify_content_type(self, content: str, matched_patterns: List[str]) -> ContentType:
        """Classify content type based on matched patterns."""
        boost_patterns = [p for p in matched_patterns if p.startswith('+')]
        demote_patterns = [p for p in matched_patterns if p.startswith('-')]
        
        if boost_patterns and not demote_patterns:
            return ContentType.DECLARATIVE_ANSWER
        elif demote_patterns and not boost_patterns:
            return ContentType.INTERROGATIVE_QUESTION
        elif boost_patterns and demote_patterns:
            return ContentType.MIXED_CONTENT
        else:
            return ContentType.NEUTRAL_CONTENT

    def boost_declarative_patterns(self, text: str) -> float:
        """Get boost score for declarative patterns in text."""
        boost = 0.0
        for pattern, score_delta, _ in self._compiled_boost:
            if pattern.search(text):
                boost += score_delta
        return boost

    def demote_interrogative_patterns(self, text: str) -> float:
        """Get demote score for interrogative patterns in text."""
        demote = 0.0
        for pattern, score_delta, _ in self._compiled_demote:
            if pattern.search(text):
                demote += score_delta  # score_delta is negative
        return demote

    def explain_ranking(self, content: str, query: str = "") -> Dict[str, Any]:
        """
        Provide detailed explanation of ranking decision.
        
        Useful for debugging and understanding ranking behavior.
        """
        ranking_result = self._score_content(content, 0.5, query)  # Use 0.5 as baseline
        
        return {
            "content": content,
            "query": query,
            "original_score": ranking_result.original_score,
            "fact_boost": ranking_result.fact_boost,
            "final_score": ranking_result.final_score,
            "content_type": ranking_result.content_type.value,
            "matched_patterns": ranking_result.matched_patterns,
            "boost_breakdown": {
                "declarative_boost": self.boost_declarative_patterns(content),
                "interrogative_demote": self.demote_interrogative_patterns(content),
                "query_boost": self._calculate_query_boost(content, query)
            },
            "supported_patterns": {
                "boost_patterns": len(self.boost_patterns),
                "demote_patterns": len(self.demote_patterns)
            }
        }

    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get statistics about ranking patterns."""
        return {
            "boost_patterns": len(self.boost_patterns),
            "demote_patterns": len(self.demote_patterns),
            "boost_categories": [label for _, _, label in self.boost_patterns],
            "demote_categories": [label for _, _, label in self.demote_patterns]
        }

    def create_ranking_prior(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create ranking prior that can be used with existing reranker.
        
        This applies fact-aware scoring as a pre-processing step that can
        be combined with cross-encoder reranking.
        """
        enhanced_results = []
        
        for result in results:
            content = result.get('content', '')
            original_score = result.get('score', 0.0)
            
            # Calculate fact-aware boost
            ranking_result = self._score_content(content, original_score)
            
            # Create enhanced result with prior score
            enhanced_result = result.copy()
            enhanced_result['fact_prior_score'] = ranking_result.fact_boost
            enhanced_result['enhanced_score'] = ranking_result.final_score
            enhanced_result['content_type'] = ranking_result.content_type.value
            enhanced_result['fact_patterns'] = ranking_result.matched_patterns
            
            enhanced_results.append(enhanced_result)
        
        return enhanced_results

    def integrate_with_cross_encoder(self, 
                                   results: List[Dict[str, Any]], 
                                   cross_encoder_scores: List[float],
                                   alpha: float = 0.7,
                                   beta: float = 0.3) -> List[Dict[str, Any]]:
        """
        Integrate fact-aware ranking with cross-encoder scores.
        
        Args:
            results: Search results
            cross_encoder_scores: Scores from cross-encoder reranker
            alpha: Weight for cross-encoder scores (0.7)
            beta: Weight for fact-aware scores (0.3)
            
        Returns:
            Results with combined scoring
        """
        if len(results) != len(cross_encoder_scores):
            logger.warning("Mismatch between results and cross-encoder scores")
            return results
        
        enhanced_results = []
        
        for i, result in enumerate(results):
            content = result.get('content', '')
            cross_encoder_score = cross_encoder_scores[i]
            
            # Get fact-aware scoring
            ranking_result = self._score_content(content, 0.0)  # Use 0 baseline
            fact_score = ranking_result.fact_boost + 0.5  # Normalize to [0, 1] range
            
            # Combine scores
            combined_score = alpha * cross_encoder_score + beta * fact_score
            
            # Create enhanced result
            enhanced_result = result.copy()
            enhanced_result['cross_encoder_score'] = cross_encoder_score
            enhanced_result['fact_score'] = fact_score
            enhanced_result['combined_score'] = combined_score
            enhanced_result['content_type'] = ranking_result.content_type.value
            enhanced_result['fact_patterns'] = ranking_result.matched_patterns
            
            enhanced_results.append(enhanced_result)
        
        # Sort by combined score
        enhanced_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        logger.info(f"Integrated fact-aware ranking with cross-encoder for {len(results)} results")
        return enhanced_results