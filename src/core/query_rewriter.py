"""
Query rewriting system for improved fact recall.

This module rewrites fact queries to generate multiple variants that
improve recall by expanding queries with alternative phrasings and
fact-specific templates.
"""

import re
from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RewriteMethod(Enum):
    """Methods used for query rewriting."""
    TEMPLATE_EXPANSION = "template_expansion"
    PARAPHRASE_GENERATION = "paraphrase_generation"
    FACT_ATTRIBUTE_EXPANSION = "fact_attribute_expansion"
    QUESTION_NORMALIZATION = "question_normalization"


@dataclass
class RewrittenQuery:
    """A rewritten query variant with metadata."""
    query: str
    confidence: float
    method: RewriteMethod
    fact_attribute: Optional[str]
    original_query: str


class FactQueryRewriter:
    """
    Rewrites fact queries to improve recall using template expansion.
    
    Generates bounded query variants (max 3) to improve fact retrieval
    while maintaining performance and avoiding query explosion.
    """

    def __init__(self, max_variants: int = 3):
        self.max_variants = max_variants
        
        # Fact-specific rewrite templates
        self.fact_templates = {
            "name": {
                "declarative_patterns": [
                    "my name is {value}",
                    "i am called {value}",
                    "i'm {value}",
                    "call me {value}",
                    "you can call me {value}",
                    "people call me {value}",
                    "i go by {value}"
                ],
                "question_variants": [
                    "what is my name",
                    "what's my name", 
                    "who am i",
                    "do you know my name",
                    "what do you call me",
                    "what should i be called",
                    "can you tell me my name"
                ]
            },
            "email": {
                "declarative_patterns": [
                    "my email is {value}",
                    "my email address is {value}",
                    "contact me at {value}",
                    "reach me at {value}",
                    "email me at {value}",
                    "you can email me at {value}"
                ],
                "question_variants": [
                    "what is my email",
                    "what's my email address",
                    "do you know my email",
                    "how can you email me",
                    "what's my contact email",
                    "where can you reach me"
                ]
            },
            "location": {
                "declarative_patterns": [
                    "i live in {value}",
                    "i'm from {value}",
                    "i am located in {value}",
                    "my location is {value}",
                    "my address is {value}",
                    "i'm based in {value}"
                ],
                "question_variants": [
                    "where do i live",
                    "what's my location",
                    "where am i from",
                    "what's my address",
                    "where am i located",
                    "where am i based"
                ]
            },
            "preferences.food": {
                "declarative_patterns": [
                    "i like {value}",
                    "i love {value}",
                    "i enjoy {value}",
                    "i prefer {value}",
                    "my favorite food is {value}",
                    "i'm into {value}",
                    "i really like {value}"
                ],
                "question_variants": [
                    "what food do i like",
                    "what's my favorite food",
                    "what do i prefer to eat",
                    "what kind of food do i enjoy",
                    "what food am i into",
                    "what do i like to eat"
                ]
            },
            "preferences.music": {
                "declarative_patterns": [
                    "i like {value}",
                    "i love {value}",
                    "i listen to {value}",
                    "my favorite music is {value}",
                    "my favorite artist is {value}",
                    "i'm into {value}",
                    "i enjoy {value}"
                ],
                "question_variants": [
                    "what music do i like",
                    "what's my favorite music",
                    "what do i listen to",
                    "what kind of music do i enjoy",
                    "what's my music taste",
                    "what artist do i like"
                ]
            },
            "age": {
                "declarative_patterns": [
                    "i am {value} years old",
                    "i'm {value} years old",
                    "my age is {value}",
                    "i'm {value}"
                ],
                "question_variants": [
                    "how old am i",
                    "what's my age",
                    "what age am i",
                    "do you know how old i am"
                ]
            },
            "birthday": {
                "declarative_patterns": [
                    "i was born on {value}",
                    "i was born in {value}",
                    "my birthday is {value}",
                    "my birth date is {value}"
                ],
                "question_variants": [
                    "when is my birthday",
                    "when was i born",
                    "what's my birth date",
                    "do you know my birthday"
                ]
            },
            "job": {
                "declarative_patterns": [
                    "i work as {value}",
                    "i work at {value}",
                    "i'm a {value}",
                    "my job is {value}",
                    "my profession is {value}",
                    "i'm employed as {value}"
                ],
                "question_variants": [
                    "what do i do for work",
                    "what's my job",
                    "where do i work",
                    "what's my profession",
                    "what do i do professionally"
                ]
            },
            "phone": {
                "declarative_patterns": [
                    "my phone number is {value}",
                    "my number is {value}",
                    "call me at {value}",
                    "you can reach me at {value}"
                ],
                "question_variants": [
                    "what's my phone number",
                    "how can you call me",
                    "do you have my number",
                    "what's my contact number"
                ]
            }
        }

        # Generic question-to-statement conversion patterns
        self.question_to_statement_patterns = [
            (r"what'?s my (\w+)", r"my \1 is"),
            (r"who am i", r"my name is"),
            (r"where do i live", r"i live in"),
            (r"how old am i", r"i am"),
            (r"when (?:is my birthday|was i born)", r"i was born"),
            (r"what do i (\w+)", r"i \1"),
            (r"do you know my (\w+)", r"my \1 is")
        ]

        # Common question word substitutions
        self.question_substitutions = {
            "what's": ["what is"],
            "who's": ["who is"],
            "where's": ["where is"],
            "how's": ["how is"],
            "when's": ["when is"]
        }

    def rewrite_fact_query(self, query: str, intent_attribute: Optional[str] = None) -> List[RewrittenQuery]:
        """
        Rewrite a fact query to improve recall.
        
        Args:
            query: Original query
            intent_attribute: Detected fact attribute (if any)
            
        Returns:
            List of rewritten query variants (bounded by max_variants)
        """
        rewrites = []
        query_lower = query.lower().strip()
        
        # Method 1: Template-based expansion using fact attribute
        if intent_attribute and intent_attribute in self.fact_templates:
            template_rewrites = self._expand_with_templates(query_lower, intent_attribute)
            rewrites.extend(template_rewrites)
        
        # Method 2: Question normalization (contractions, etc.)
        normalized_rewrites = self._normalize_question(query_lower)
        rewrites.extend(normalized_rewrites)
        
        # Method 3: Question-to-statement conversion
        statement_rewrites = self._convert_to_statements(query_lower)
        rewrites.extend(statement_rewrites)
        
        # Method 4: Generic paraphrasing for unknown attributes
        if not intent_attribute:
            paraphrase_rewrites = self._generate_paraphrases(query_lower)
            rewrites.extend(paraphrase_rewrites)
        
        # Remove duplicates and limit to max_variants
        unique_rewrites = self._deduplicate_and_limit(rewrites, query)
        
        logger.debug(f"Rewrote query '{query}' into {len(unique_rewrites)} variants")
        return unique_rewrites

    def _expand_with_templates(self, query: str, fact_attribute: str) -> List[RewrittenQuery]:
        """Expand query using fact-specific templates."""
        rewrites = []
        templates = self.fact_templates.get(fact_attribute, {})
        
        # Generate question variants
        question_variants = templates.get("question_variants", [])
        for variant in question_variants[:2]:  # Limit to 2 variants
            if variant.lower() != query:
                rewrites.append(RewrittenQuery(
                    query=variant,
                    confidence=0.9,
                    method=RewriteMethod.TEMPLATE_EXPANSION,
                    fact_attribute=fact_attribute,
                    original_query=query
                ))
        
        # Generate declarative patterns (without {value} placeholder)
        declarative_patterns = templates.get("declarative_patterns", [])
        for pattern in declarative_patterns[:1]:  # Limit to 1 declarative
            # Remove {value} placeholder for search
            search_pattern = pattern.replace(" {value}", "").replace("{value} ", "").replace("{value}", "")
            if search_pattern and search_pattern.lower() != query:
                rewrites.append(RewrittenQuery(
                    query=search_pattern,
                    confidence=0.85,
                    method=RewriteMethod.FACT_ATTRIBUTE_EXPANSION,
                    fact_attribute=fact_attribute,
                    original_query=query
                ))
        
        return rewrites

    def _normalize_question(self, query: str) -> List[RewrittenQuery]:
        """Normalize question contractions and variations."""
        rewrites = []
        
        # Expand contractions
        for contraction, expansions in self.question_substitutions.items():
            if contraction in query:
                for expansion in expansions:
                    normalized = query.replace(contraction, expansion)
                    if normalized != query:
                        rewrites.append(RewrittenQuery(
                            query=normalized,
                            confidence=0.95,
                            method=RewriteMethod.QUESTION_NORMALIZATION,
                            fact_attribute=None,
                            original_query=query
                        ))
        
        return rewrites

    def _convert_to_statements(self, query: str) -> List[RewrittenQuery]:
        """Convert questions to declarative statement patterns."""
        rewrites = []
        
        for question_pattern, statement_pattern in self.question_to_statement_patterns:
            match = re.search(question_pattern, query, re.IGNORECASE)
            if match:
                if match.groups():
                    # Use captured group
                    statement = re.sub(question_pattern, statement_pattern, query, flags=re.IGNORECASE)
                else:
                    # Direct replacement
                    statement = statement_pattern
                
                if statement != query:
                    rewrites.append(RewrittenQuery(
                        query=statement,
                        confidence=0.8,
                        method=RewriteMethod.PARAPHRASE_GENERATION,
                        fact_attribute=None,
                        original_query=query
                    ))
        
        return rewrites

    def _generate_paraphrases(self, query: str) -> List[RewrittenQuery]:
        """Generate paraphrases for queries without detected attributes."""
        rewrites = []
        
        # Simple substitution-based paraphrasing
        substitutions = [
            ("do you know", "what is"),
            ("can you tell me", "what is"),
            ("tell me about", "what is"),
            ("i want to know", "what is"),
            ("what about", "what is")
        ]
        
        for old_phrase, new_phrase in substitutions:
            if old_phrase in query:
                paraphrase = query.replace(old_phrase, new_phrase)
                if paraphrase != query:
                    rewrites.append(RewrittenQuery(
                        query=paraphrase,
                        confidence=0.7,
                        method=RewriteMethod.PARAPHRASE_GENERATION,
                        fact_attribute=None,
                        original_query=query
                    ))
        
        return rewrites

    def _deduplicate_and_limit(self, rewrites: List[RewrittenQuery], original_query: str) -> List[RewrittenQuery]:
        """Remove duplicates and limit to max_variants."""
        # Remove duplicates by query text
        seen_queries = {original_query.lower()}
        unique_rewrites = []
        
        # Sort by confidence (descending)
        rewrites.sort(key=lambda x: x.confidence, reverse=True)
        
        for rewrite in rewrites:
            if rewrite.query.lower() not in seen_queries and len(unique_rewrites) < self.max_variants:
                seen_queries.add(rewrite.query.lower())
                unique_rewrites.append(rewrite)
        
        return unique_rewrites

    def generate_fact_templates(self, attribute: str, value: str = "") -> List[str]:
        """
        Generate fact-specific templates for a given attribute.
        
        Args:
            attribute: Fact attribute (name, email, etc.)
            value: Optional fact value to fill templates
            
        Returns:
            List of template strings
        """
        templates = []
        
        if attribute in self.fact_templates:
            declarative_patterns = self.fact_templates[attribute].get("declarative_patterns", [])
            question_variants = self.fact_templates[attribute].get("question_variants", [])
            
            # Add declarative patterns
            for pattern in declarative_patterns:
                if value:
                    template = pattern.replace("{value}", value)
                else:
                    template = pattern.replace(" {value}", "").replace("{value} ", "").replace("{value}", "")
                templates.append(template)
            
            # Add question variants
            templates.extend(question_variants)
        
        return templates

    def explain_rewriting(self, query: str, intent_attribute: Optional[str] = None) -> Dict[str, Any]:
        """
        Provide detailed explanation of query rewriting process.
        
        Useful for debugging and understanding rewriting decisions.
        """
        rewrites = self.rewrite_fact_query(query, intent_attribute)
        
        return {
            "original_query": query,
            "intent_attribute": intent_attribute,
            "rewritten_queries": [
                {
                    "query": rewrite.query,
                    "confidence": rewrite.confidence,
                    "method": rewrite.method.value,
                    "fact_attribute": rewrite.fact_attribute
                }
                for rewrite in rewrites
            ],
            "total_variants": len(rewrites),
            "max_variants": self.max_variants,
            "available_templates": list(self.fact_templates.keys()),
            "methods_used": list(set(rewrite.method.value for rewrite in rewrites))
        }

    def get_supported_attributes(self) -> List[str]:
        """Get list of fact attributes with template support."""
        return list(self.fact_templates.keys())

    def add_fact_template(self, attribute: str, declarative_patterns: List[str], question_variants: List[str]):
        """
        Add new fact template for custom attributes.
        
        Args:
            attribute: Fact attribute name
            declarative_patterns: List of declarative statement patterns
            question_variants: List of question variants
        """
        self.fact_templates[attribute] = {
            "declarative_patterns": declarative_patterns,
            "question_variants": question_variants
        }
        logger.info(f"Added fact template for attribute: {attribute}")

    def get_template_stats(self) -> Dict[str, Any]:
        """Get statistics about available templates."""
        stats = {
            "total_attributes": len(self.fact_templates),
            "attribute_details": {}
        }
        
        for attr, templates in self.fact_templates.items():
            stats["attribute_details"][attr] = {
                "declarative_patterns": len(templates.get("declarative_patterns", [])),
                "question_variants": len(templates.get("question_variants", []))
            }
        
        return stats

    def create_query_expansion(self, query: str, intent_attribute: Optional[str] = None) -> List[str]:
        """
        Create query expansion for use with search systems.
        
        Returns just the query strings for easy integration.
        """
        rewrites = self.rewrite_fact_query(query, intent_attribute)
        return [rewrite.query for rewrite in rewrites]