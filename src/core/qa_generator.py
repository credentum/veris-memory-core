"""
Question-Answer pair generation for enhanced fact retrieval.

This module generates Q&A pairs from declarative statements to improve
fact recall by creating stitched units that can be jointly indexed
for both vector and lexical search.
"""

import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class QAPair:
    """A question-answer pair with metadata."""
    question: str
    answer: str
    fact_attribute: str
    confidence: float
    generation_method: str
    original_statement: str


@dataclass  
class StitchedUnit:
    """A stitched Q&A unit for joint indexing."""
    content: str  # Combined Q&A text for embedding
    question: str
    answer: str
    fact_attribute: str
    confidence: float
    metadata: Dict[str, Any]
    

class QAPairGenerator:
    """
    Generates question-answer pairs from declarative statements.
    
    Creates multiple question variants for each fact to improve recall
    and generates stitched units for joint vector/lexical indexing.
    """

    def __init__(self):
        # Question templates for different fact types
        self.question_templates = {
            "name": [
                "What is my name?",
                "What's my name?", 
                "Who am I?",
                "Do you know my name?",
                "What do you call me?",
                "What should I be called?"
            ],
            "email": [
                "What is my email?",
                "What's my email address?",
                "Do you know my email?",
                "How can you email me?",
                "What's my contact email?"
            ],
            "location": [
                "Where do I live?",
                "What's my location?",
                "Where am I from?",
                "What's my address?",
                "Where am I located?"
            ],
            "preferences.food": [
                "What food do I like?",
                "What's my favorite food?", 
                "What do I prefer to eat?",
                "What kind of food do I enjoy?",
                "What food am I into?"
            ],
            "preferences.music": [
                "What music do I like?",
                "What's my favorite music?",
                "What do I listen to?",
                "What kind of music do I enjoy?",
                "What's my music taste?"
            ],
            "age": [
                "How old am I?",
                "What's my age?",
                "What age am I?",
                "Do you know how old I am?"
            ],
            "birthday": [
                "When is my birthday?",
                "When was I born?",
                "What's my birth date?",
                "Do you know my birthday?"
            ],
            "job": [
                "What do I do for work?",
                "What's my job?",
                "Where do I work?",
                "What's my profession?",
                "What do I do professionally?"
            ],
            "phone": [
                "What's my phone number?",
                "How can you call me?",
                "Do you have my number?",
                "What's my contact number?"
            ],
            "relationship": [
                "What's my relationship status?",
                "Am I married?",
                "Am I single?",
                "What's my status?"
            ]
        }

        # Answer patterns for extracting values from statements
        self.answer_patterns = {
            "name": [
                r"(?:my name is|i'?m (?:called )?|call me|you can call me)\s+([A-Za-z][A-Za-z\s]{1,30})",
                r"i go by\s+([A-Za-z][A-Za-z\s]{1,30})"
            ],
            "email": [
                r"(?:my email(?: address)? is|email:?)\s*([\w._%+-]+@[\w.-]+\.\w{2,})"
            ],
            "location": [
                r"(?:i live in|i'?m (?:from|located in)|my (?:address|location) is)\s+([A-Za-z][A-Za-z\s,]{2,50})"
            ],
            "preferences.food": [
                r"(?:i like|i love|i enjoy|i prefer)\s+([^.!?]+?)(?:\s+food)?(?:[.!?]|$)",
                r"my favorite food is\s+([^.!?]+)"
            ],
            "preferences.music": [
                r"(?:i like|i love|i enjoy|i prefer)\s+([^.!?]+?)(?:\s+music)?(?:[.!?]|$)",
                r"my favorite (?:music|artist|band|song) is\s+([^.!?]+)"
            ],
            "age": [
                r"i'?m\s+(\d{1,3})\s+years old",
                r"my age is\s+(\d{1,3})"
            ],
            "birthday": [
                r"(?:i was born (?:on|in)|my birthday is)\s+([A-Za-z0-9\s,\/\-]{5,30})"
            ],
            "job": [
                r"(?:i work (?:as|at)|i'?m a|my job is)\s+([^.!?]{3,50})"
            ],
            "phone": [
                r"(?:my (?:phone|number) is|call me at|phone:?)\s*([\+]?[\d\s\-\(\)]{7,20})"
            ],
            "relationship": [
                r"i'?m\s+(married|single|dating|engaged|divorced|widowed)"
            ]
        }

    def generate_qa_pairs(self, statement: str, fact_attribute: str, fact_value: str) -> List[QAPair]:
        """
        Generate Q&A pairs for a fact statement.
        
        Args:
            statement: Original declarative statement
            fact_attribute: Type of fact (name, email, etc.)
            fact_value: Extracted fact value
            
        Returns:
            List of generated Q&A pairs
        """
        pairs = []
        
        # Get question templates for this fact type
        questions = self.question_templates.get(fact_attribute, [])
        if not questions:
            # Generate generic questions for unknown fact types
            questions = self._generate_generic_questions(fact_attribute)
        
        # Create Q&A pairs
        for question in questions:
            pair = QAPair(
                question=question,
                answer=fact_value,
                fact_attribute=fact_attribute,
                confidence=self._calculate_confidence(statement, fact_attribute, fact_value),
                generation_method="template_based",
                original_statement=statement
            )
            pairs.append(pair)
        
        logger.debug(f"Generated {len(pairs)} Q&A pairs for {fact_attribute}: {fact_value}")
        return pairs

    def generate_qa_pairs_from_statement(self, statement: str) -> List[QAPair]:
        """
        Extract facts from statement and generate Q&A pairs.
        
        Args:
            statement: Declarative statement to process
            
        Returns:
            List of generated Q&A pairs
        """
        pairs = []
        
        # Try to extract facts using patterns
        for fact_type, patterns in self.answer_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, statement, re.IGNORECASE)
                for match in matches:
                    if match.groups():
                        fact_value = match.group(1).strip()
                        fact_pairs = self.generate_qa_pairs(statement, fact_type, fact_value)
                        pairs.extend(fact_pairs)
        
        return pairs

    def create_stitched_unit(self, qa_pair: QAPair, include_variants: bool = True) -> StitchedUnit:
        """
        Create a stitched unit for joint indexing.
        
        Args:
            qa_pair: Q&A pair to create stitched unit from
            include_variants: Include question variants in content
            
        Returns:
            StitchedUnit for indexing
        """
        # Create combined content for embedding
        base_content = f"Q: {qa_pair.question} A: {qa_pair.answer}"
        
        if include_variants:
            # Add related questions to improve recall
            related_questions = self.question_templates.get(qa_pair.fact_attribute, [])
            if related_questions:
                # Include up to 3 additional question variants
                variants = [q for q in related_questions[:3] if q != qa_pair.question]
                if variants:
                    variant_text = " | ".join(variants)
                    base_content += f" | Related: {variant_text}"
        
        # Add fact attribute as searchable content
        content = f"{base_content} | Attribute: {qa_pair.fact_attribute}"
        
        metadata = {
            "fact_attribute": qa_pair.fact_attribute,
            "confidence": qa_pair.confidence,
            "generation_method": qa_pair.generation_method,
            "original_statement": qa_pair.original_statement,
            "created_at": datetime.utcnow().isoformat(),
            "content_type": "qa_pair"
        }
        
        return StitchedUnit(
            content=content,
            question=qa_pair.question,
            answer=qa_pair.answer,
            fact_attribute=qa_pair.fact_attribute,
            confidence=qa_pair.confidence,
            metadata=metadata
        )

    def create_multiple_stitched_units(self, statement: str) -> List[StitchedUnit]:
        """
        Create multiple stitched units from a statement.
        
        Args:
            statement: Declarative statement
            
        Returns:
            List of stitched units for indexing
        """
        qa_pairs = self.generate_qa_pairs_from_statement(statement)
        stitched_units = []
        
        for pair in qa_pairs:
            unit = self.create_stitched_unit(pair)
            stitched_units.append(unit)
        
        logger.info(f"Created {len(stitched_units)} stitched units from statement")
        return stitched_units

    def _generate_generic_questions(self, fact_attribute: str) -> List[str]:
        """Generate generic questions for unknown fact types."""
        base_attr = fact_attribute.split('.')[-1]  # Get last part of dotted attribute
        
        return [
            f"What is my {base_attr}?",
            f"What's my {base_attr}?", 
            f"Do you know my {base_attr}?",
            f"Can you tell me my {base_attr}?"
        ]

    def _calculate_confidence(self, statement: str, fact_attribute: str, fact_value: str) -> float:
        """Calculate confidence score for Q&A pair generation."""
        base_confidence = 0.85
        
        # Boost confidence for first-person statements
        if re.search(r'\b(?:my|i\'?m|i\s+(?:am|was|have))\b', statement, re.IGNORECASE):
            base_confidence += 0.1
        
        # Boost confidence for explicit statements
        if re.search(r'\bis\b', statement, re.IGNORECASE):
            base_confidence += 0.05
        
        # Reduce confidence for uncertain language
        if re.search(r'\b(?:maybe|perhaps|might|could be|think)\b', statement, re.IGNORECASE):
            base_confidence -= 0.2
        
        # Boost confidence for known high-quality fact types
        high_quality_types = ['name', 'email', 'age', 'birthday']
        if fact_attribute in high_quality_types:
            base_confidence += 0.05
        
        return min(0.99, max(0.1, base_confidence))

    def get_supported_fact_types(self) -> List[str]:
        """Get list of supported fact types for Q&A generation."""
        return list(self.question_templates.keys())

    def validate_qa_pair(self, qa_pair: QAPair) -> bool:
        """Validate a Q&A pair for quality."""
        # Basic validation
        if not qa_pair.question or not qa_pair.answer:
            return False
        
        if len(qa_pair.question) < 5 or len(qa_pair.answer) < 1:
            return False
        
        if qa_pair.confidence < 0.3:
            return False
        
        # Check for reasonable answer length
        if len(qa_pair.answer) > 200:
            return False
        
        return True

    def explain_generation(self, statement: str) -> Dict[str, Any]:
        """
        Provide detailed explanation of Q&A generation process.
        
        Useful for debugging and understanding generation decisions.
        """
        qa_pairs = self.generate_qa_pairs_from_statement(statement)
        stitched_units = [self.create_stitched_unit(pair) for pair in qa_pairs]
        
        return {
            "statement": statement,
            "generated_pairs": [
                {
                    "question": pair.question,
                    "answer": pair.answer,
                    "fact_attribute": pair.fact_attribute,
                    "confidence": pair.confidence,
                    "method": pair.generation_method
                }
                for pair in qa_pairs
            ],
            "stitched_units": [
                {
                    "content": unit.content,
                    "fact_attribute": unit.fact_attribute,
                    "confidence": unit.confidence
                }
                for unit in stitched_units
            ],
            "supported_fact_types": self.get_supported_fact_types(),
            "total_pairs_generated": len(qa_pairs),
            "total_units_created": len(stitched_units)
        }