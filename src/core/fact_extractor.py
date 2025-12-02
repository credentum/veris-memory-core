"""
Fact extraction from conversation context for automatic fact storage.

This module analyzes stored context to identify and extract structured facts
that can be stored in the deterministic fact store for reliable retrieval.
"""

import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging

try:
    from .intent_classifier import IntentClassifier, IntentType
except ImportError:
    from intent_classifier import IntentClassifier, IntentType

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFact:
    """A fact extracted from text with metadata."""
    attribute: str
    value: Any
    confidence: float
    source_text: str
    extraction_method: str
    position: int = 0  # Position in text where fact was found


class FactExtractor:
    """
    Extracts structured facts from conversation text.
    
    Works with the IntentClassifier to identify fact statements and extract
    attribute-value pairs for storage in the fact store.
    """

    def __init__(self):
        self.intent_classifier = IntentClassifier()
        
        # Enhanced extraction patterns with capture groups
        self.extraction_patterns = [
            # Personal identity
            {
                "attribute": "name",
                "patterns": [
                    r"(?:my name is|i'?m (?:called )?|call me|you can call me)\s+([A-Za-z][A-Za-z\s]{1,30})",
                    r"i go by\s+([A-Za-z][A-Za-z\s]{1,30})",
                    r"people call me\s+([A-Za-z][A-Za-z\s]{1,30})"
                ],
                "confidence": 0.95,
                "validator": self._validate_name
            },
            
            # Contact information
            {
                "attribute": "email",
                "patterns": [
                    r"(?:my email(?: address)? is|email:?)\s*([\w._%+-]+@[\w.-]+\.\w{2,})",
                    r"(?:reach me at|contact me at|send (?:to|at))\s*([\w._%+-]+@[\w.-]+\.\w{2,})"
                ],
                "confidence": 0.95,
                "validator": self._validate_email
            },
            
            {
                "attribute": "phone",
                "patterns": [
                    r"(?:my (?:phone|number) is|call me at|phone:?)\s*([\+]?[\d\s\-\(\)]{7,20})",
                    r"(?:reach me at|contact me at)\s*([\+]?[\d\s\-\(\)]{7,20})"
                ],
                "confidence": 0.9,
                "validator": self._validate_phone
            },
            
            # Location information
            {
                "attribute": "location",
                "patterns": [
                    r"(?:i live in|i'?m (?:from|located in)|my (?:address|location) is)\s+([A-Za-z][A-Za-z\s,]{2,50})",
                    r"(?:based in|located in)\s+([A-Za-z][A-Za-z\s,]{2,50})"
                ],
                "confidence": 0.9,
                "validator": self._validate_location
            },
            
            # Food preferences
            {
                "attribute": "preferences.food",
                "patterns": [
                    r"(?:i like|i love|i enjoy|i prefer)\s+([^.!?]+?)(?:\s+food)?(?:[.!?]|$)",
                    r"my favorite food is\s+([^.!?]+)",
                    r"i'?m a (?:big )?fan of\s+([^.!?]+?)(?:\s+food)?(?:[.!?]|$)"
                ],
                "confidence": 0.85,
                "validator": self._validate_food_preference
            },
            
            # Music preferences  
            {
                "attribute": "preferences.music",
                "patterns": [
                    r"(?:i like|i love|i enjoy|i prefer)\s+([^.!?]+?)(?:\s+music)?(?:[.!?]|$)",
                    r"my favorite (?:music|artist|band|song) is\s+([^.!?]+)",
                    r"i listen to\s+([^.!?]+)",
                    r"i'?m into\s+([^.!?]+?)(?:\s+music)?(?:[.!?]|$)"
                ],
                "confidence": 0.85,
                "validator": self._validate_music_preference
            },
            
            # Personal details
            {
                "attribute": "age",
                "patterns": [
                    r"i'?m\s+(\d{1,3})\s+years old",
                    r"my age is\s+(\d{1,3})",
                    r"i'?m\s+(\d{1,3})"
                ],
                "confidence": 0.9,
                "validator": self._validate_age
            },
            
            {
                "attribute": "birthday",
                "patterns": [
                    r"(?:i was born (?:on|in)|my birthday is)\s+([A-Za-z0-9\s,\/\-]{5,30})",
                    r"born\s+([A-Za-z0-9\s,\/\-]{5,30})"
                ],
                "confidence": 0.85,
                "validator": self._validate_birthday
            },
            
            # Professional information
            {
                "attribute": "job",
                "patterns": [
                    r"(?:i work (?:as|at)|i'?m a|my job is)\s+([^.!?]{3,50})",
                    r"(?:i'?m employed (?:as|at)|my profession is)\s+([^.!?]{3,50})"
                ],
                "confidence": 0.8,
                "validator": self._validate_job
            },
            
            # Relationship status
            {
                "attribute": "relationship",
                "patterns": [
                    r"i'?m\s+(married|single|dating|engaged|divorced|widowed)",
                    r"my (?:relationship status|status) is\s+(married|single|dating|engaged|divorced|widowed)"
                ],
                "confidence": 0.8,
                "validator": self._validate_relationship
            }
        ]

    def extract_facts_from_text(self, text: str, source_turn_id: str = "") -> List[ExtractedFact]:
        """
        Extract all identifiable facts from a text passage.
        
        Args:
            text: Text to analyze for facts
            source_turn_id: Identifier for the source conversation turn
            
        Returns:
            List of extracted facts with confidence scores
        """
        if not text or not text.strip():
            return []

        facts = []
        text = text.strip()

        # First, use intent classifier to identify fact statements
        intent_result = self.intent_classifier.classify(text)
        
        if intent_result.intent in [IntentType.STORE_FACT, IntentType.UPDATE_FACT]:
            # Try direct extraction using intent classifier
            fact_extraction = self.intent_classifier.extract_fact_value(text)
            if fact_extraction:
                attribute, value = fact_extraction
                facts.append(ExtractedFact(
                    attribute=attribute,
                    value=value,
                    confidence=intent_result.confidence,
                    source_text=text,
                    extraction_method="intent_classifier",
                    position=0
                ))

        # Then apply pattern-based extraction for additional facts
        pattern_facts = self._extract_with_patterns(text)
        facts.extend(pattern_facts)

        # Remove duplicates (same attribute)
        unique_facts = {}
        for fact in facts:
            if fact.attribute not in unique_facts or fact.confidence > unique_facts[fact.attribute].confidence:
                unique_facts[fact.attribute] = fact

        return list(unique_facts.values())

    def _extract_with_patterns(self, text: str) -> List[ExtractedFact]:
        """Extract facts using pattern matching."""
        facts = []
        
        for pattern_group in self.extraction_patterns:
            attribute = pattern_group["attribute"]
            patterns = pattern_group["patterns"]
            base_confidence = pattern_group["confidence"]
            validator = pattern_group.get("validator")
            
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                
                for match in matches:
                    if match.groups():
                        value = match.group(1).strip()
                        
                        # Apply validator if present
                        if validator:
                            is_valid, cleaned_value = validator(value)
                            if not is_valid:
                                continue
                            value = cleaned_value
                        
                        # Adjust confidence based on context
                        confidence = self._adjust_confidence(base_confidence, text, match)
                        
                        facts.append(ExtractedFact(
                            attribute=attribute,
                            value=value,
                            confidence=confidence,
                            source_text=match.group(0),
                            extraction_method="pattern_matching",
                            position=match.start()
                        ))
        
        return facts

    def _adjust_confidence(self, base_confidence: float, text: str, match: re.Match) -> float:
        """Adjust confidence based on context and quality indicators."""
        confidence = base_confidence
        
        # Boost confidence for first-person statements
        if re.search(r'\b(?:my|i\'?m|i\s+(?:am|was|have))\b', text, re.IGNORECASE):
            confidence = min(0.99, confidence + 0.05)
        
        # Reduce confidence for questions or uncertain language
        if re.search(r'\?|maybe|perhaps|might|could be', text, re.IGNORECASE):
            confidence = max(0.1, confidence - 0.2)
        
        # Boost confidence for explicit statements
        if re.search(r'my .* is|i am|i\'m called', text, re.IGNORECASE):
            confidence = min(0.99, confidence + 0.03)
        
        return confidence

    # Validators
    def _validate_name(self, value: str) -> Tuple[bool, str]:
        """Validate and clean extracted name."""
        cleaned = re.sub(r'[^\w\s]', '', value).strip()
        if len(cleaned) < 1 or len(cleaned) > 50:
            return False, value
        return True, cleaned.title()

    def _validate_email(self, value: str) -> Tuple[bool, str]:
        """Validate email format."""
        email_pattern = r'^[\w._%+-]+@[\w.-]+\.\w{2,}$'
        if re.match(email_pattern, value.strip()):
            return True, value.strip().lower()
        return False, value

    def _validate_phone(self, value: str) -> Tuple[bool, str]:
        """Validate phone number format."""
        # Remove common formatting
        cleaned = re.sub(r'[\s\-\(\)]', '', value)
        if len(cleaned) >= 7 and cleaned.replace('+', '').isdigit():
            return True, cleaned
        return False, value

    def _validate_location(self, value: str) -> Tuple[bool, str]:
        """Validate location string."""
        cleaned = value.strip()
        if len(cleaned) < 2 or len(cleaned) > 100:
            return False, value
        return True, cleaned.title()

    def _validate_food_preference(self, value: str) -> Tuple[bool, str]:
        """Validate food preference."""
        cleaned = value.strip()
        # Filter out very generic or non-food terms
        if len(cleaned) < 3 or cleaned.lower() in ['things', 'stuff', 'it', 'them']:
            return False, value
        return True, cleaned.lower()

    def _validate_music_preference(self, value: str) -> Tuple[bool, str]:
        """Validate music preference."""
        cleaned = value.strip()
        if len(cleaned) < 3 or cleaned.lower() in ['things', 'stuff', 'it', 'them']:
            return False, value
        return True, cleaned.lower()

    def _validate_age(self, value: str) -> Tuple[bool, str]:
        """Validate age."""
        try:
            age = int(value)
            if 0 <= age <= 150:
                return True, str(age)
        except ValueError:
            pass
        return False, value

    def _validate_birthday(self, value: str) -> Tuple[bool, str]:
        """Validate birthday string."""
        cleaned = value.strip()
        if len(cleaned) < 5:
            return False, value
        return True, cleaned

    def _validate_job(self, value: str) -> Tuple[bool, str]:
        """Validate job description."""
        cleaned = value.strip()
        if len(cleaned) < 3 or len(cleaned) > 100:
            return False, value
        return True, cleaned.lower()

    def _validate_relationship(self, value: str) -> Tuple[bool, str]:
        """Validate relationship status."""
        valid_statuses = ['married', 'single', 'dating', 'engaged', 'divorced', 'widowed']
        cleaned = value.strip().lower()
        if cleaned in valid_statuses:
            return True, cleaned
        return False, value

    def extract_facts_from_context(self, context: str, metadata: Dict[str, Any] = None) -> List[ExtractedFact]:
        """
        Extract facts from stored context with additional metadata.
        
        Args:
            context: Context text to analyze
            metadata: Additional context metadata
            
        Returns:
            List of extracted facts
        """
        source_turn_id = metadata.get('turn_id', '') if metadata else ''
        return self.extract_facts_from_text(context, source_turn_id)

    def get_supported_attributes(self) -> List[str]:
        """Get list of fact attributes this extractor can identify."""
        return [pattern["attribute"] for pattern in self.extraction_patterns]

    def explain_extraction(self, text: str) -> Dict[str, Any]:
        """
        Provide detailed explanation of fact extraction process.
        
        Useful for debugging and understanding extraction decisions.
        """
        facts = self.extract_facts_from_text(text)
        intent_result = self.intent_classifier.classify(text)
        
        return {
            "text": text,
            "intent_classification": {
                "intent": intent_result.intent.value,
                "confidence": intent_result.confidence,
                "attribute": intent_result.attribute
            },
            "extracted_facts": [
                {
                    "attribute": fact.attribute,
                    "value": fact.value,
                    "confidence": fact.confidence,
                    "method": fact.extraction_method,
                    "source": fact.source_text,
                    "position": fact.position
                }
                for fact in facts
            ],
            "supported_attributes": self.get_supported_attributes()
        }