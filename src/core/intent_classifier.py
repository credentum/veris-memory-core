"""
Intent classification for identifying fact lookup queries.

This module determines whether a query is asking for stored facts vs. general conversation,
enabling the system to route fact queries to the deterministic fact store.
"""

import re
from enum import Enum
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Query intent types."""
    FACT_LOOKUP = "fact_lookup"
    GENERAL_QUERY = "general_query"
    STORE_FACT = "store_fact"
    UPDATE_FACT = "update_fact"


@dataclass
class IntentResult:
    """Result of intent classification."""
    intent: IntentType
    confidence: float
    attribute: Optional[str] = None
    reasoning: str = ""


class IntentClassifier:
    """
    Lightweight intent detector for fact vs. general queries.
    
    Uses pattern matching and linguistic cues to identify when users
    are asking for stored facts like "What's my name?" vs. general queries.
    """

    def __init__(self):
        # Fact lookup patterns (asking for stored info)
        self.fact_lookup_patterns = [
            # Name queries
            (r"what'?s my name", "name", 0.95),
            (r"who am i", "name", 0.9),
            (r"do you (?:know|remember) my name", "name", 0.9),
            (r"what do you call me", "name", 0.85),
            
            # Email queries
            (r"what'?s my email", "email", 0.95),
            (r"what'?s my email address", "email", 0.95),
            (r"do you (?:know|have) my email", "email", 0.9),
            
            # Preference queries
            (r"what (?:food|foods) do i like", "preferences.food", 0.9),
            (r"what'?s my favorite food", "preferences.food", 0.9),
            (r"what do i prefer to eat", "preferences.food", 0.85),
            (r"what music do i like", "preferences.music", 0.9),
            (r"what'?s my favorite (?:music|song|artist)", "preferences.music", 0.9),
            
            # Location queries
            (r"where do i live", "location", 0.9),
            (r"what'?s my (?:address|location)", "location", 0.9),
            (r"where am i (?:from|located)", "location", 0.85),
            
            # Contact info
            (r"what'?s my (?:phone|number)", "phone", 0.9),
            (r"do you have my (?:phone|number)", "phone", 0.85),
            
            # Personal info
            (r"how old am i", "age", 0.9),
            (r"what'?s my age", "age", 0.9),
            (r"when (?:was i born|is my birthday)", "birthday", 0.9),
            
            # General fact recall patterns
            (r"what did i (?:say|tell you) about", "general", 0.8),
            (r"do you remember (?:what i said|when i told you)", "general", 0.8),
            (r"what do you know about my", "general", 0.75),
        ]

        # Fact storage patterns (providing new info)
        self.fact_storage_patterns = [
            # Name statements
            (r"my name is (\w+)", "name", 0.95),
            (r"i'?m (?:called )?(\w+)", "name", 0.9),
            (r"call me (\w+)", "name", 0.85),
            (r"you can call me (\w+)", "name", 0.85),
            
            # Email statements
            (r"my email (?:is|address is) ([\w._%+-]+@[\w.-]+\.\w+)", "email", 0.95),
            (r"here'?s my email:? ([\w._%+-]+@[\w.-]+\.\w+)", "email", 0.9),
            
            # Preference statements
            (r"i like ([^.!?]+)", "preferences.general", 0.8),
            (r"i prefer ([^.!?]+)", "preferences.general", 0.8),
            (r"my favorite (\w+) is ([^.!?]+)", "preferences", 0.85),
            (r"i love ([^.!?]+)", "preferences.general", 0.75),
            (r"i enjoy ([^.!?]+)", "preferences.general", 0.75),
            
            # Location statements
            (r"i live in ([^.!?]+)", "location", 0.9),
            (r"i'?m (?:from|located in) ([^.!?]+)", "location", 0.85),
            (r"my (?:address|location) is ([^.!?]+)", "location", 0.9),
            
            # Contact statements
            (r"my (?:phone|number) is ([\d\-\+\(\)\s]+)", "phone", 0.9),
            
            # Personal info statements
            (r"i'?m (\d+) years old", "age", 0.9),
            (r"i was born (?:in|on) ([^.!?]+)", "birthday", 0.85),
        ]

        # Update patterns (corrections)
        self.update_patterns = [
            (r"(?:actually|no),?\s*(?:my name is|i'?m|call me) (\w+)", "name", 0.9),
            (r"(?:correction|actually),?\s*(.+)", "general", 0.8),
            (r"that'?s wrong,?\s*(.+)", "general", 0.8),
            (r"let me (?:correct|fix) that", "general", 0.7),
        ]

        # Compile patterns for efficiency
        self._compiled_lookup = [(re.compile(pattern, re.IGNORECASE), attr, conf) 
                                for pattern, attr, conf in self.fact_lookup_patterns]
        self._compiled_storage = [(re.compile(pattern, re.IGNORECASE), attr, conf) 
                                 for pattern, attr, conf in self.fact_storage_patterns]
        self._compiled_update = [(re.compile(pattern, re.IGNORECASE), attr, conf) 
                                for pattern, attr, conf in self.update_patterns]

    def classify(self, query: str) -> IntentResult:
        """
        Classify query intent and extract relevant attributes.
        
        Args:
            query: User query text
            
        Returns:
            IntentResult with intent type, confidence, and extracted attribute
        """
        if not query or not query.strip():
            return IntentResult(
                intent=IntentType.GENERAL_QUERY,
                confidence=0.0,
                reasoning="Empty query"
            )

        query = query.strip()
        
        # Check for update/correction patterns first
        update_result = self._check_patterns(query, self._compiled_update, IntentType.UPDATE_FACT)
        if update_result.confidence > 0.7:
            return update_result

        # Check for fact storage patterns
        storage_result = self._check_patterns(query, self._compiled_storage, IntentType.STORE_FACT)
        if storage_result.confidence > 0.7:
            return storage_result

        # Check for fact lookup patterns
        lookup_result = self._check_patterns(query, self._compiled_lookup, IntentType.FACT_LOOKUP)
        if lookup_result.confidence > 0.7:
            return lookup_result

        # Default to general query
        return IntentResult(
            intent=IntentType.GENERAL_QUERY,
            confidence=0.8,
            reasoning="No fact patterns matched"
        )

    def _check_patterns(self, query: str, patterns: List[Tuple], intent_type: IntentType) -> IntentResult:
        """Check query against a set of compiled patterns."""
        best_match = None
        best_confidence = 0.0
        best_attribute = None

        for pattern, attribute, confidence in patterns:
            match = pattern.search(query)
            if match and confidence > best_confidence:
                best_match = match
                best_confidence = confidence
                best_attribute = attribute

        if best_match:
            return IntentResult(
                intent=intent_type,
                confidence=best_confidence,
                attribute=best_attribute,
                reasoning=f"Matched pattern for {best_attribute}"
            )

        return IntentResult(
            intent=IntentType.GENERAL_QUERY,
            confidence=0.0,
            reasoning="No patterns matched"
        )

    def extract_fact_attribute(self, query: str) -> Optional[str]:
        """
        Extract the specific fact attribute being requested.
        
        Returns None if no specific attribute can be determined.
        """
        result = self.classify(query)
        return result.attribute if result.confidence > 0.7 else None

    def extract_fact_value(self, statement: str) -> Optional[Tuple[str, str]]:
        """
        Extract attribute and value from a fact storage statement.
        
        Returns (attribute, value) tuple or None if no extraction possible.
        """
        result = self.classify(statement)
        
        if result.intent not in [IntentType.STORE_FACT, IntentType.UPDATE_FACT]:
            return None

        if not result.attribute:
            return None

        # Try to extract the value using the same patterns
        for pattern, attribute, confidence in self.fact_storage_patterns:
            if attribute == result.attribute:
                match = re.search(pattern, statement, re.IGNORECASE)
                if match and len(match.groups()) > 0:
                    value = match.group(1).strip()
                    return (attribute, value)

        return None

    def is_fact_query(self, query: str, confidence_threshold: float = 0.7) -> bool:
        """
        Simple boolean check if query is fact-related.
        
        Args:
            query: Query text
            confidence_threshold: Minimum confidence to consider fact query
            
        Returns:
            True if query is likely asking for or storing facts
        """
        result = self.classify(query)
        return (result.intent in [IntentType.FACT_LOOKUP, IntentType.STORE_FACT, IntentType.UPDATE_FACT] 
                and result.confidence >= confidence_threshold)

    def get_supported_attributes(self) -> List[str]:
        """Get list of fact attributes the classifier can detect."""
        attributes = set()
        
        for _, attr, _ in self.fact_lookup_patterns:
            attributes.add(attr)
        for _, attr, _ in self.fact_storage_patterns:
            attributes.add(attr)
            
        return sorted(list(attributes))

    def explain_classification(self, query: str) -> Dict[str, Any]:
        """
        Provide detailed explanation of classification decision.
        
        Useful for debugging and understanding classifier behavior.
        """
        result = self.classify(query)
        
        # Test against all pattern types
        lookup_result = self._check_patterns(query, self._compiled_lookup, IntentType.FACT_LOOKUP)
        storage_result = self._check_patterns(query, self._compiled_storage, IntentType.STORE_FACT)
        update_result = self._check_patterns(query, self._compiled_update, IntentType.UPDATE_FACT)

        return {
            "query": query,
            "final_classification": {
                "intent": result.intent.value,
                "confidence": result.confidence,
                "attribute": result.attribute,
                "reasoning": result.reasoning
            },
            "pattern_scores": {
                "fact_lookup": lookup_result.confidence,
                "fact_storage": storage_result.confidence,
                "fact_update": update_result.confidence
            },
            "supported_attributes": self.get_supported_attributes()
        }