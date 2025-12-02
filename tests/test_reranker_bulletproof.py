#!/usr/bin/env python3
"""
Comprehensive unit tests for bulletproof reranker implementation
Tests text extraction, reranking logic, fail-safes, and edge cases
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Handle optional dependencies for numpy/torch
try:
    import numpy as np
    import torch
    NUMERICAL_DEPS_AVAILABLE = True
except ImportError:
    NUMERICAL_DEPS_AVAILABLE = False

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

# Handle optional dependencies gracefully
try:
    from src.storage.reranker_bulletproof import (
        extract_chunk_text, 
        clamp_for_rerank, 
        BulletproofReranker, 
        RerankerMetrics
    )
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    import warnings
    warnings.warn(f"Skipping reranker tests due to missing dependencies: {e}")


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Reranker dependencies not available")
class TestTextExtraction(unittest.TestCase):
    """Test robust text extraction from various payload shapes"""
    
    def test_extract_direct_text(self):
        """Test extraction from direct text field"""
        payload = {"text": "Direct text content"}
        result = extract_chunk_text(payload)
        self.assertEqual(result, "Direct text content")
    
    def test_extract_nested_content_text(self):
        """Test extraction from nested content.text"""
        payload = {"content": {"text": "Nested content text"}}
        result = extract_chunk_text(payload)
        self.assertEqual(result, "Nested content text")
    
    def test_extract_string_content(self):
        """Test extraction from plain string content"""
        payload = {"content": "Plain string content"}
        result = extract_chunk_text(payload)
        self.assertEqual(result, "Plain string content")
    
    def test_extract_tool_array_style(self):
        """Test extraction from tool-style content array"""
        payload = {
            "content": [
                {"type": "text", "text": "First part"},
                {"type": "text", "text": "Second part"},
                {"type": "other", "data": "ignored"}
            ]
        }
        result = extract_chunk_text(payload)
        self.assertEqual(result, "First part\nSecond part")
    
    def test_extract_nested_payload(self):
        """Test extraction from nested payload structure (MCP style)"""
        payload = {
            "payload": {
                "content": {"text": "MCP nested content"}
            }
        }
        result = extract_chunk_text(payload)
        self.assertEqual(result, "MCP nested content")
    
    def test_extract_alternative_fields(self):
        """Test extraction from alternative field names"""
        test_cases = [
            ({"body": "Body content"}, "Body content"),
            ({"markdown": "Markdown content"}, "Markdown content"),
            ({"description": "Description content"}, "Description content"),
            ({"summary": "Summary content"}, "Summary content"),
            ({"title": "Title content"}, "Title content"),
        ]
        
        for payload, expected in test_cases:
            with self.subTest(field=list(payload.keys())[0]):
                result = extract_chunk_text(payload)
                self.assertEqual(result, expected)
    
    def test_extract_empty_payload(self):
        """Test extraction from empty payload"""
        self.assertEqual(extract_chunk_text({}), "")
        self.assertEqual(extract_chunk_text(None), "")
    
    def test_extract_fallback_to_string(self):
        """Test fallback to string conversion"""
        payload = {"unknown_field": "value", "other": 123}
        result = extract_chunk_text(payload)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
    
    def test_extract_priority_order(self):
        """Test that extraction follows correct priority order"""
        # text field should take priority over content
        payload = {"text": "Direct text", "content": "Content text"}
        result = extract_chunk_text(payload)
        self.assertEqual(result, "Direct text")
    
    def test_extract_handles_none_values(self):
        """Test extraction handles None values gracefully"""
        payload = {"content": None, "text": None, "body": "Valid content"}
        result = extract_chunk_text(payload)
        self.assertEqual(result, "Valid content")


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Reranker dependencies not available")
class TestTextClamping(unittest.TestCase):
    """Test text clamping functionality"""
    
    def test_clamp_short_text(self):
        """Test clamping preserves short text"""
        text = "Short text"
        result = clamp_for_rerank(text)
        self.assertEqual(result, "Short text")
    
    def test_clamp_long_text(self):
        """Test clamping truncates long text"""
        text = "x" * 5000
        result = clamp_for_rerank(text, max_chars=1000)
        self.assertEqual(len(result), 1000)
    
    def test_clamp_strips_empty_lines(self):
        """Test clamping removes short/empty lines"""
        text = "Line 1\n\n  \nLine 2\nx\nLong line content"
        result = clamp_for_rerank(text)
        lines = result.split('\n')
        # Should remove empty lines and very short lines
        self.assertNotIn("", lines)
        self.assertNotIn("x", lines)  # Too short
    
    def test_clamp_handles_none(self):
        """Test clamping handles None input"""
        result = clamp_for_rerank(None)
        self.assertEqual(result, "")
    
    def test_clamp_strips_whitespace(self):
        """Test clamping strips leading/trailing whitespace"""
        text = "  \n  Content with spaces  \n  "
        result = clamp_for_rerank(text)
        self.assertEqual(result, "Content with spaces")


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Reranker dependencies not available")
class TestRerankerMetrics(unittest.TestCase):
    """Test metrics tracking functionality"""
    
    def setUp(self):
        self.metrics = RerankerMetrics()
    
    def test_counter_initialization(self):
        """Test counter initialization and increment"""
        counter = self.metrics.counter("test_counter")
        self.assertIsInstance(counter, RerankerMetrics)
        
        counter.inc()
        stats = self.metrics.get_stats()
        self.assertEqual(stats["counters"]["test_counter"], 1)
    
    def test_timer_tracking(self):
        """Test latency timer tracking"""
        self.metrics.timers = [100.0, 200.0, 150.0]
        stats = self.metrics.get_stats()
        self.assertEqual(stats["avg_latency_ms"], 150.0)
    
    def test_empty_metrics(self):
        """Test empty metrics return valid structure"""
        stats = self.metrics.get_stats()
        self.assertIn("counters", stats)
        self.assertIn("avg_latency_ms", stats)
        self.assertEqual(stats["avg_latency_ms"], 0.0)


@unittest.skipUnless(DEPENDENCIES_AVAILABLE and NUMERICAL_DEPS_AVAILABLE, "Reranker dependencies not available")
class TestBulletproofReranker(unittest.TestCase):
    """Test bulletproof reranker functionality"""
    
    def setUp(self):
        """Setup test fixtures"""
        # Mock the sentence_transformers import
        self.mock_cross_encoder = Mock()
        self.mock_model = Mock()
        self.mock_cross_encoder.return_value = self.mock_model
        
        with patch('storage.reranker_bulletproof.CrossEncoder', self.mock_cross_encoder):
            self.reranker = BulletproofReranker(debug_mode=True)
    
    def test_reranker_initialization(self):
        """Test reranker initializes correctly"""
        self.assertTrue(self.reranker.enabled)
        self.assertEqual(self.reranker.request_count, 0)
        self.assertTrue(self.reranker.debug_mode)
    
    def test_rerank_empty_candidates(self):
        """Test reranking with empty candidates list"""
        result = self.reranker.rerank("test query", [])
        self.assertEqual(result, [])
    
    def test_rerank_disabled_reranker(self):
        """Test reranking when reranker is disabled"""
        self.reranker.enabled = False
        candidates = [{"id": "test", "payload": {"text": "test"}}]
        result = self.reranker.rerank("query", candidates)
        self.assertEqual(result, candidates)
    
    def test_rerank_all_empty_text_fallback(self):
        """Test reranking falls back when all candidates have empty text"""
        candidates = [
            {"id": "empty1", "payload": {}},
            {"id": "empty2", "payload": {"content": None}},
            {"id": "empty3", "payload": {"text": ""}}
        ]
        
        result = self.reranker.rerank("query", candidates)
        self.assertEqual(result, candidates)  # Should return original order
    
    def test_rerank_successful_scoring(self):
        """Test successful reranking with valid scores"""
        candidates = [
            {"id": "doc1", "payload": {"text": "Database content"}, "score": 0.7},
            {"id": "doc2", "payload": {"text": "Microservices content"}, "score": 0.6},
            {"id": "doc3", "payload": {"text": "OAuth content"}, "score": 0.5}
        ]
        
        # Mock model to return scores that should reorder candidates
        self.mock_model.predict.return_value = [0.3, 0.9, 0.1]  # doc2 should be first
        
        result = self.reranker.rerank("microservices query", candidates)
        
        # Verify reordering
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["id"], "doc2")  # Highest rerank score
        self.assertEqual(result[1]["id"], "doc1")
        self.assertEqual(result[2]["id"], "doc3")
        
        # Verify scores are added
        self.assertEqual(result[0]["rerank_score"], 0.9)
        self.assertEqual(result[0]["original_score"], 0.6)
    
    def test_rerank_all_zero_scores_fallback(self):
        """Test reranking falls back when model returns all zeros"""
        candidates = [
            {"id": "doc1", "payload": {"text": "Content 1"}, "score": 0.7},
            {"id": "doc2", "payload": {"text": "Content 2"}, "score": 0.6}
        ]
        
        # Mock model to return all zeros (failure case)
        self.mock_model.predict.return_value = [0.0, 0.0]
        
        result = self.reranker.rerank("query", candidates)
        self.assertEqual(result, candidates)  # Should return original order
    
    def test_rerank_exception_handling(self):
        """Test reranking handles exceptions gracefully"""
        candidates = [{"id": "doc1", "payload": {"text": "Content"}}]
        
        # Mock model to raise exception
        self.mock_model.predict.side_effect = Exception("Model error")
        
        result = self.reranker.rerank("query", candidates)
        self.assertEqual(result, candidates)  # Should return original order
    
    def test_debug_rerank_functionality(self):
        """Test debug rerank provides detailed information"""
        candidates = [
            {"id": "doc1", "payload": {"text": "Test content"}, "score": 0.8},
            {"id": "doc2", "payload": {}, "score": 0.6}  # Empty payload
        ]
        
        # Mock model for debug scoring
        self.mock_model.predict.return_value = [0.5]
        
        result = self.reranker.debug_rerank("test query", candidates)
        
        self.assertEqual(len(result), 2)
        
        # Check first result (with text)
        self.assertEqual(result[0]["id"], "doc1")
        self.assertGreater(result[0]["text_len"], 0)
        self.assertEqual(result[0]["rerank_score"], 0.5)
        
        # Check second result (empty text)  
        self.assertEqual(result[1]["id"], "doc2")
        self.assertEqual(result[1]["text_len"], 0)
    
    def test_get_stats(self):
        """Test getting reranker statistics"""
        stats = self.reranker.get_stats()
        
        expected_keys = [
            "enabled", "model_name", "model_loaded", "request_count", 
            "debug_mode", "auto_disable_threshold", "metrics"
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
        
        self.assertTrue(stats["enabled"])
        self.assertEqual(stats["request_count"], 0)


@unittest.skipUnless(DEPENDENCIES_AVAILABLE and NUMERICAL_DEPS_AVAILABLE, "Reranker dependencies not available")
class TestIntegrationScenarios(unittest.TestCase):
    """Test real-world integration scenarios"""
    
    def setUp(self):
        self.mock_cross_encoder = Mock()
        self.mock_model = Mock()
        self.mock_cross_encoder.return_value = self.mock_model
    
    def test_mcp_response_format(self):
        """Test handling of typical MCP response format"""
        mcp_candidates = [
            {
                "context_id": "ctx_123",
                "payload": {
                    "content": {
                        "text": "Microservices provide better scalability and fault isolation"
                    }
                },
                "score": 0.85
            },
            {
                "context_id": "ctx_456", 
                "payload": {
                    "content": "Database indexing improves query performance"
                },
                "score": 0.75
            }
        ]
        
        with patch('storage.reranker_bulletproof.CrossEncoder', self.mock_cross_encoder):
            reranker = BulletproofReranker()
            
            # Mock successful reranking
            self.mock_model.predict.return_value = [0.9, 0.3]
            
            result = reranker.rerank("microservices benefits", mcp_candidates)
            
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["context_id"], "ctx_123")  # Higher score first
            self.assertIn("rerank_score", result[0])
            self.assertIn("original_score", result[0])
    
    def test_phase2_failure_scenario(self):
        """Test the exact Phase 2 T-102 failure scenario"""
        # This is the payload format that was causing "all zeros"
        problematic_candidates = [
            {"id": "doc_23", "payload": {"content": "Database indexing..."}, "score": 0.73},
            {"id": "doc_45", "payload": {"content": "Microservices provide..."}, "score": 0.69},  # Should rank higher
            {"id": "doc_91", "payload": {"content": "OAuth 2.0 provides..."}, "score": 0.71},
        ]
        
        with patch('storage.reranker_bulletproof.CrossEncoder', self.mock_cross_encoder):
            reranker = BulletproofReranker()
            
            # Mock scores that should promote microservices content
            self.mock_model.predict.return_value = [-0.5, 0.8, -0.2]
            
            result = reranker.rerank("microservices architecture benefits", problematic_candidates)
            
            # Verify the gold document (doc_45) moves to top
            self.assertEqual(result[0]["id"], "doc_45")
            self.assertGreater(result[0]["rerank_score"], 0.5)
            
            # Verify no all-zero bug
            scores = [r["rerank_score"] for r in result]
            self.assertFalse(all(abs(s) < 1e-9 for s in scores))


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)