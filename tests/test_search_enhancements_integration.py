#!/usr/bin/env python3
"""
Integration tests for search enhancements in the retrieval pipeline.

Tests the integration of apply_search_enhancements with the main
retrieve_context endpoint for improved paraphrase robustness.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from mcp_server.search_enhancements import (
    apply_search_enhancements,
    calculate_exact_match_boost,
    apply_context_type_weight,
    calculate_recency_decay,
    calculate_technical_boost,
    is_technical_query,
    CONTEXT_TYPE_WEIGHTS,
    TECHNICAL_TERMS,
)


class TestExactMatchBoost(unittest.TestCase):
    """Test exact match boosting functionality."""

    def test_exact_filename_match(self):
        """Test that exact filename matches get highest boost."""
        boost = calculate_exact_match_boost(
            query="server.py",
            content="Server implementation code",
            metadata={"file_path": "/src/server.py"},
        )

        self.assertEqual(boost, 5.0)

    def test_partial_filename_match(self):
        """Test that partial filename matches get medium boost."""
        boost = calculate_exact_match_boost(
            query="server",
            content="Server implementation code",
            metadata={"file_path": "/src/server.py"},
        )

        self.assertEqual(boost, 3.0)

    def test_title_match(self):
        """Test that title matches get boost."""
        boost = calculate_exact_match_boost(
            query="neo4j configuration",
            content="Content about database settings",
            metadata={"title": "Neo4j Configuration Guide"},
        )

        self.assertEqual(boost, 2.0)

    def test_exact_phrase_in_content(self):
        """Test that exact phrase matches in content get boost."""
        boost = calculate_exact_match_boost(
            query="database configuration",
            content="This is about database configuration settings",
            metadata={},
        )

        self.assertEqual(boost, 1.5)

    def test_all_keywords_present(self):
        """Test that all keywords present get small boost."""
        boost = calculate_exact_match_boost(
            query="neo4j vector database",
            content="Setting up neo4j for vector database storage",
            metadata={},
        )

        self.assertGreater(boost, 1.0)

    def test_no_match_no_boost(self):
        """Test that no match returns no boost."""
        boost = calculate_exact_match_boost(
            query="something completely different",
            content="Unrelated content here",
            metadata={},
        )

        self.assertEqual(boost, 1.0)

    def test_dict_content_handling(self):
        """Test that dict content is handled correctly."""
        boost = calculate_exact_match_boost(
            query="server",
            content={"text": "Server implementation", "content": "More about server"},
            metadata={},
        )

        self.assertGreater(boost, 1.0)


class TestContextTypeWeight(unittest.TestCase):
    """Test context type weighting functionality."""

    def test_code_type_highest_weight(self):
        """Test that code type gets highest weight."""
        result = {"payload": {"type": "code"}}
        weight = apply_context_type_weight(result)

        self.assertEqual(weight, 2.0)

    def test_log_type_weight(self):
        """Test that log type gets appropriate weight."""
        result = {"payload": {"type": "log"}}
        weight = apply_context_type_weight(result)

        self.assertEqual(weight, 1.5)

    def test_documentation_type_weight(self):
        """Test that documentation type gets appropriate weight."""
        result = {"payload": {"type": "documentation"}}
        weight = apply_context_type_weight(result)

        self.assertEqual(weight, 1.3)

    def test_conversation_type_lowest_weight(self):
        """Test that conversation type gets lowest weight."""
        result = {"payload": {"type": "conversation"}}
        weight = apply_context_type_weight(result)

        self.assertEqual(weight, 0.5)

    def test_unknown_type_default_weight(self):
        """Test that unknown type gets default weight."""
        result = {"payload": {"type": "random_type"}}
        weight = apply_context_type_weight(result)

        self.assertEqual(weight, 1.0)

    def test_category_overrides_type(self):
        """Test that category field overrides type field."""
        result = {"payload": {"type": "conversation", "category": "code"}}
        weight = apply_context_type_weight(result)

        self.assertEqual(weight, 2.0)  # Code category overrides conversation type

    def test_python_code_category(self):
        """Test that python_code category is treated as code."""
        result = {"payload": {"category": "python_code"}}
        weight = apply_context_type_weight(result)

        self.assertEqual(weight, 2.0)


class TestRecencyDecay(unittest.TestCase):
    """Test recency decay functionality."""

    def test_recent_content_full_score(self):
        """Test that very recent content gets full score."""
        now = datetime.now()
        score = calculate_recency_decay(now.isoformat(), 1.0)

        self.assertAlmostEqual(score, 1.0, places=1)

    def test_week_old_content_decayed(self):
        """Test that week-old content is decayed by about half."""
        week_ago = datetime.now() - timedelta(days=7)
        score = calculate_recency_decay(week_ago.isoformat(), 1.0, decay_rate=7.0)

        # With decay_rate=7, after 7 days should be ~0.37 (exp(-1))
        self.assertLess(score, 0.5)
        self.assertGreater(score, 0.3)

    def test_old_content_minimum_score(self):
        """Test that very old content still has minimum score."""
        very_old = datetime.now() - timedelta(days=365)
        score = calculate_recency_decay(very_old.isoformat(), 1.0)

        # Should not go below 10%
        self.assertEqual(score, 0.1)

    def test_no_timestamp_unchanged(self):
        """Test that no timestamp returns unchanged score."""
        score = calculate_recency_decay(None, 0.8)

        self.assertEqual(score, 0.8)

    def test_invalid_timestamp_unchanged(self):
        """Test that invalid timestamp returns unchanged score."""
        score = calculate_recency_decay("not a date", 0.8)

        self.assertEqual(score, 0.8)

    def test_string_timestamp_handling(self):
        """Test that ISO string timestamps are handled."""
        recent = datetime.now() - timedelta(hours=1)
        score = calculate_recency_decay(recent.isoformat(), 1.0)

        self.assertGreater(score, 0.9)


class TestTechnicalBoost(unittest.TestCase):
    """Test technical term boosting functionality."""

    def test_technical_query_with_technical_content(self):
        """Test that technical queries with technical content get boost."""
        boost = calculate_technical_boost(
            query="python database query",
            content="This is a Python function that executes SQL database queries",
        )

        self.assertGreater(boost, 1.0)

    def test_technical_query_with_nontechnical_content(self):
        """Test that technical queries with non-technical content get less boost."""
        boost = calculate_technical_boost(
            query="python database query",
            content="This is some random conversational text about nothing technical",
        )

        # Should still be 1.0 since content has no technical terms
        self.assertAlmostEqual(boost, 1.0, places=2)

    def test_file_extension_query(self):
        """Test that file extension queries are treated as technical."""
        boost = calculate_technical_boost(
            query="server.py",
            content="Python code with import statements and function definitions",
        )

        self.assertGreater(boost, 1.0)

    def test_non_technical_query(self):
        """Test that non-technical queries don't get extra boost."""
        boost = calculate_technical_boost(
            query="how to use the app",
            content="Instructions for using the application",
        )

        self.assertEqual(boost, 1.0)

    def test_dict_content_handling(self):
        """Test that dict content is handled correctly."""
        boost = calculate_technical_boost(
            query="python function",
            content={"text": "def hello():", "content": "Python function example"},
        )

        self.assertGreater(boost, 1.0)


class TestIsTechnicalQuery(unittest.TestCase):
    """Test technical query detection."""

    def test_technical_terms_detected(self):
        """Test that technical terms are detected."""
        self.assertTrue(is_technical_query("python function"))
        self.assertTrue(is_technical_query("database query"))
        self.assertTrue(is_technical_query("docker container"))

    def test_file_extension_detected(self):
        """Test that file extensions are detected."""
        self.assertTrue(is_technical_query("server.py"))
        self.assertTrue(is_technical_query("config.json"))
        self.assertTrue(is_technical_query("README.md"))

    def test_code_patterns_detected(self):
        """Test that code patterns are detected."""
        self.assertTrue(is_technical_query("function()"))
        self.assertTrue(is_technical_query("class definition"))
        self.assertTrue(is_technical_query("def hello"))

    def test_non_technical_query(self):
        """Test that non-technical queries are correctly identified."""
        self.assertFalse(is_technical_query("how are you"))
        self.assertFalse(is_technical_query("what time is it"))
        self.assertFalse(is_technical_query("hello world"))


class TestApplySearchEnhancements(unittest.TestCase):
    """Test the main apply_search_enhancements function."""

    def setUp(self):
        """Set up test data."""
        self.test_results = [
            {
                "id": "doc1",
                "score": 0.8,
                "payload": {
                    "content": "Python server implementation with database queries",
                    "type": "code",
                    "metadata": {"file_path": "/src/server.py"},
                },
            },
            {
                "id": "doc2",
                "score": 0.9,
                "payload": {
                    "content": "Conversation about the server",
                    "type": "conversation",
                    "metadata": {},
                },
            },
            {
                "id": "doc3",
                "score": 0.85,
                "payload": {
                    "content": "Server configuration documentation",
                    "type": "documentation",
                    "metadata": {"title": "Server Configuration"},
                },
            },
        ]

    def test_code_context_prioritized_over_conversation(self):
        """Test that code context is prioritized over conversation for technical queries."""
        enhanced = apply_search_enhancements(
            results=self.test_results,
            query="server.py",
            enable_exact_match=True,
            enable_type_weighting=True,
            enable_recency_decay=False,
            enable_technical_boost=True,
        )

        # Code result should be first (has filename match + code type weight)
        self.assertEqual(enhanced[0]["id"], "doc1")

    def test_all_results_have_enhanced_scores(self):
        """Test that all results get enhanced score metadata."""
        enhanced = apply_search_enhancements(
            results=self.test_results,
            query="server",
        )

        for result in enhanced:
            self.assertIn("enhanced_score", result)
            self.assertIn("original_score", result)
            self.assertIn("score_boosts", result)

    def test_results_sorted_by_enhanced_score(self):
        """Test that results are sorted by enhanced score descending."""
        enhanced = apply_search_enhancements(
            results=self.test_results,
            query="server.py",
        )

        scores = [r["enhanced_score"] for r in enhanced]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_individual_boosts_tracked(self):
        """Test that individual boost factors are tracked."""
        enhanced = apply_search_enhancements(
            results=self.test_results[:1],
            query="server.py",
        )

        boosts = enhanced[0]["score_boosts"]
        self.assertIn("exact_match", boosts)
        self.assertIn("type_weight", boosts)
        self.assertIn("technical", boosts)

    def test_disabling_enhancements(self):
        """Test that individual enhancements can be disabled."""
        enhanced = apply_search_enhancements(
            results=self.test_results,
            query="server.py",
            enable_exact_match=False,
            enable_type_weighting=False,
            enable_recency_decay=False,
            enable_technical_boost=False,
        )

        # All boost factors should be 1.0
        for result in enhanced:
            boosts = result["score_boosts"]
            self.assertEqual(boosts["exact_match"], 1.0)
            self.assertEqual(boosts["type_weight"], 1.0)
            self.assertEqual(boosts["technical"], 1.0)

    def test_empty_results_handled(self):
        """Test that empty results are handled gracefully."""
        enhanced = apply_search_enhancements(
            results=[],
            query="test",
        )

        self.assertEqual(enhanced, [])


class TestSearchEnhancementsIntegration(unittest.TestCase):
    """Integration tests simulating real-world scenarios."""

    def test_neo4j_configuration_paraphrase_consistency(self):
        """Test that different Neo4j config queries prioritize same content."""
        config_doc = {
            "id": "neo4j_config",
            "score": 0.75,
            "payload": {
                "content": "Neo4j database configuration settings with connection parameters",
                "type": "documentation",
                "metadata": {"title": "Neo4j Configuration Guide"},
            },
        }

        conversation_doc = {
            "id": "chat_about_neo4j",
            "score": 0.8,
            "payload": {
                "content": "I was talking about neo4j earlier",
                "type": "conversation",
                "metadata": {},
            },
        }

        results = [config_doc, conversation_doc]

        # Test with first paraphrase
        enhanced1 = apply_search_enhancements(
            results=results.copy(),
            query="How do I configure Neo4j?",
        )

        # Test with second paraphrase
        enhanced2 = apply_search_enhancements(
            results=results.copy(),
            query="What are the steps to set up Neo4j database?",
        )

        # Both should prioritize the config doc over conversation
        self.assertEqual(enhanced1[0]["id"], "neo4j_config")
        self.assertEqual(enhanced2[0]["id"], "neo4j_config")

    def test_code_file_search_prioritizes_code(self):
        """Test that searching for code files prioritizes actual code."""
        code_file = {
            "id": "main_py",
            "score": 0.7,
            "payload": {
                "content": "def main():\n    print('Hello')",
                "type": "code",
                "metadata": {"file_path": "/src/main.py"},
            },
        }

        discussion = {
            "id": "discussion",
            "score": 0.85,
            "payload": {
                "content": "We should modify main.py to add logging",
                "type": "conversation",
                "metadata": {},
            },
        }

        results = [code_file, discussion]

        enhanced = apply_search_enhancements(
            results=results,
            query="main.py",
        )

        # Code file should be first despite lower original score
        self.assertEqual(enhanced[0]["id"], "main_py")

    def test_technical_query_boosts_technical_content(self):
        """Test that technical queries boost technical content."""
        technical_doc = {
            "id": "api_doc",
            "score": 0.75,
            "payload": {
                "content": "REST API endpoint returns JSON response with database schema",
                "type": "documentation",
                "metadata": {},
            },
        }

        general_doc = {
            "id": "general",
            "score": 0.8,
            "payload": {
                "content": "The system is designed to be user-friendly",
                "type": "documentation",
                "metadata": {},
            },
        }

        results = [technical_doc, general_doc]

        enhanced = apply_search_enhancements(
            results=results,
            query="REST API database query",
        )

        # Technical doc should be first
        self.assertEqual(enhanced[0]["id"], "api_doc")


class TestConstantsAndConfiguration(unittest.TestCase):
    """Test that constants and configuration are properly defined."""

    def test_context_type_weights_defined(self):
        """Test that all expected context type weights are defined."""
        expected_types = [
            "code",
            "log",
            "documentation",
            "conversation",
            "unknown",
        ]

        for type_name in expected_types:
            self.assertIn(type_name, CONTEXT_TYPE_WEIGHTS)

    def test_technical_terms_include_key_terms(self):
        """Test that key technical terms are included."""
        key_terms = [
            "python",
            "database",
            "neo4j",
            "qdrant",
            "redis",
            "api",
            "function",
            "class",
            "docker",
        ]

        for term in key_terms:
            self.assertIn(term, TECHNICAL_TERMS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
