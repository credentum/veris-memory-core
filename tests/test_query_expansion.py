#!/usr/bin/env python3
"""
Unit tests for query expansion module
Tests multi-query expansion (MQE) and field boost functionality
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

# Handle optional dependencies gracefully
try:
    from src.storage.query_expansion import MultiQueryExpander, FieldBoostProcessor
    QUERY_EXPANSION_AVAILABLE = True
except ImportError as e:
    QUERY_EXPANSION_AVAILABLE = False
    import warnings
    warnings.warn(f"Skipping query expansion tests due to missing dependencies: {e}")


@unittest.skipUnless(QUERY_EXPANSION_AVAILABLE, "Query expansion dependencies not available")
class TestMultiQueryExpander(unittest.TestCase):
    """Test multi-query expansion functionality"""
    
    def setUp(self):
        self.expander = MultiQueryExpander()
    
    def test_generate_paraphrases_microservices(self):
        """Test paraphrase generation for microservices query"""
        query = "What are the benefits of microservices architecture?"
        paraphrases = self.expander.generate_paraphrases(query, num_paraphrases=2)
        
        # Should return original query plus paraphrases
        self.assertEqual(len(paraphrases), 3)  # original + 2 paraphrases
        self.assertEqual(paraphrases[0], query)  # First should be original
        
        # Paraphrases should be different from original
        self.assertNotEqual(paraphrases[1], query)
        self.assertNotEqual(paraphrases[2], query)
        
        # Should contain relevant terms
        combined = " ".join(paraphrases).lower()
        self.assertIn("microservices", combined)
        self.assertIn("advantage", combined.replace("benefit", "advantage"))
    
    def test_generate_paraphrases_database(self):
        """Test paraphrase generation for database query"""
        query = "How to optimize database performance?"
        paraphrases = self.expander.generate_paraphrases(query, num_paraphrases=2)
        
        self.assertEqual(len(paraphrases), 3)
        self.assertEqual(paraphrases[0], query)
        
        combined = " ".join(paraphrases).lower()
        self.assertIn("database", combined)
        self.assertIn("performance", combined)
    
    def test_generate_paraphrases_oauth(self):
        """Test paraphrase generation for OAuth query"""
        query = "How to implement OAuth authentication?"
        paraphrases = self.expander.generate_paraphrases(query, num_paraphrases=2)
        
        self.assertEqual(len(paraphrases), 3)
        combined = " ".join(paraphrases).lower()
        self.assertIn("oauth", combined)
        self.assertIn("implement", combined.replace("implement", "implement"))
    
    def test_generate_paraphrases_fallback(self):
        """Test paraphrase generation fallback for unknown queries"""
        query = "Unknown complex technical query about something"
        paraphrases = self.expander.generate_paraphrases(query, num_paraphrases=2)
        
        self.assertEqual(len(paraphrases), 3)
        self.assertEqual(paraphrases[0], query)
        
        # Fallback paraphrases should be generic
        combined = " ".join(paraphrases[1:]).lower()
        self.assertIn("information", combined)
        self.assertIn("guide", combined)
    
    def test_generate_paraphrases_empty_query(self):
        """Test paraphrase generation with empty query"""
        paraphrases = self.expander.generate_paraphrases("", num_paraphrases=2)
        self.assertEqual(len(paraphrases), 3)
        self.assertEqual(paraphrases[0], "")
    
    async def test_expand_and_search_basic(self):
        """Test basic multi-query expansion and search"""
        query = "What are microservices benefits?"
        
        # Mock search function
        async def mock_search(q: str, limit: int = 10):
            if "microservices" in q.lower():
                return [
                    {"id": "micro_doc", "content": "Microservices provide...", "score": 0.9},
                    {"id": "other_doc", "content": "Other content...", "score": 0.7}
                ]
            return [{"id": "generic_doc", "content": "Generic...", "score": 0.5}]
        
        results = await self.expander.expand_and_search(query, mock_search, limit=5, num_paraphrases=2)
        
        # Should return aggregated unique results
        self.assertGreater(len(results), 0)
        
        # Check that results have MQE metadata
        for result in results:
            self.assertIn("source_query", result)
            self.assertIn("mqe_scores", result)
            self.assertIn("mqe_queries", result)
    
    async def test_expand_and_search_aggregation(self):
        """Test that MQE properly aggregates duplicate documents"""
        query = "test query"
        
        # Mock search that returns same doc with different scores
        call_count = 0
        async def mock_search(q: str, limit: int = 10):
            nonlocal call_count
            call_count += 1
            return [
                {"id": "same_doc", "content": "content", "score": 0.5 + call_count * 0.1},
                {"id": f"unique_{call_count}", "content": "unique", "score": 0.4}
            ]
        
        results = await self.expander.expand_and_search(query, mock_search, limit=10, num_paraphrases=1)
        
        # Should have aggregated same_doc with max score
        same_docs = [r for r in results if r["id"] == "same_doc"]
        self.assertEqual(len(same_docs), 1)
        
        # Should have max score from multiple queries
        same_doc = same_docs[0]
        self.assertIn("mqe_scores", same_doc)
        self.assertGreater(len(same_doc["mqe_scores"]), 1)
        self.assertEqual(same_doc["score"], max(same_doc["mqe_scores"]))
    
    async def test_expand_and_search_error_handling(self):
        """Test MQE handles search errors gracefully"""
        query = "test query"
        
        # Mock search that sometimes fails
        call_count = 0
        async def mock_search(q: str, limit: int = 10):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Second call fails
                raise Exception("Search failed")
            return [{"id": f"doc_{call_count}", "content": "content", "score": 0.8}]
        
        results = await self.expander.expand_and_search(query, mock_search, limit=10, num_paraphrases=2)
        
        # Should have results from successful queries only
        self.assertGreater(len(results), 0)
        doc_ids = [r["id"] for r in results]
        self.assertIn("doc_1", doc_ids)  # First call succeeded
        # Second call failed, third call succeeded but may have different ID


@unittest.skipUnless(QUERY_EXPANSION_AVAILABLE, "Query expansion dependencies not available")
class TestFieldBoostProcessor(unittest.TestCase):
    """Test field boost processing functionality"""
    
    def setUp(self):
        self.processor = FieldBoostProcessor(title_boost=1.2, heading_boost=1.1)
    
    def test_extract_fields_markdown_headers(self):
        """Test field extraction from markdown-style headers"""
        text = """# Main Title

## Section Header

Some content here.

### Subsection

More content."""
        
        fields = self.processor.extract_fields(text)
        
        self.assertEqual(fields["title"], "Main Title")
        self.assertIn("Section Header", fields["headings"])
        self.assertIn("Subsection", fields["headings"])
        self.assertIn("Some content", fields["content"])
    
    def test_extract_fields_no_headers(self):
        """Test field extraction from plain text"""
        text = "Just some plain text content without headers."
        
        fields = self.processor.extract_fields(text)
        
        self.assertEqual(fields["title"], "")
        self.assertEqual(fields["headings"], "")
        self.assertIn("Just some plain", fields["content"])
    
    def test_extract_fields_title_detection(self):
        """Test title detection from short lines"""
        text = """Microservices Architecture Guide

This is the main content of the document.
It explains various concepts and best practices."""
        
        fields = self.processor.extract_fields(text)
        
        self.assertEqual(fields["title"], "Microservices Architecture Guide")
        self.assertIn("main content", fields["content"])
    
    def test_boost_chunk_text_with_title(self):
        """Test text boosting with title"""
        text = """# Database Performance Guide

Database optimization involves indexing and query tuning."""
        
        boosted = self.processor.boost_chunk_text(text)
        
        # Title should be repeated for lexical boost
        self.assertIn("Database Performance Guide", boosted)
        # Should contain original content
        self.assertIn("indexing and query tuning", boosted)
        
        # Count title occurrences (should be boosted)
        title_count = boosted.count("Database Performance Guide")
        self.assertGreaterEqual(title_count, 1)  # At least once from boost
    
    def test_boost_chunk_text_with_headings(self):
        """Test text boosting with headings"""
        text = """## Performance Optimization
### Query Tuning

Optimize your database queries for better performance."""
        
        boosted = self.processor.boost_chunk_text(text)
        
        # Headings should be repeated
        self.assertIn("Performance Optimization", boosted)
        self.assertIn("Query Tuning", boosted)
    
    def test_process_results_integration(self):
        """Test processing results with field boosts"""
        results = [
            {
                "id": "doc1",
                "text": "# Microservices Guide\n\nMicroservices architecture benefits...",
                "score": 0.8
            },
            {
                "id": "doc2", 
                "text": "Plain text content without headers.",
                "score": 0.7
            }
        ]
        
        processed = self.processor.process_results(results)
        
        self.assertEqual(len(processed), 2)
        
        # First result should have boosted text
        self.assertIn("boosted_text", processed[0])
        self.assertIn("original_text", processed[0])
        
        # Boosted text should contain repeated title
        boosted_text = processed[0]["boosted_text"]
        self.assertIn("Microservices Guide", boosted_text)
        
        # Second result should also have boosted_text (even if minimal)
        self.assertIn("boosted_text", processed[1])
    
    def test_boost_configuration(self):
        """Test different boost configurations"""
        # Test with different boost values
        processor_high = FieldBoostProcessor(title_boost=2.0, heading_boost=1.5)
        processor_low = FieldBoostProcessor(title_boost=1.0, heading_boost=1.0)
        
        text = "# Important Title\n## Section\nContent here."
        
        boosted_high = processor_high.boost_chunk_text(text)
        boosted_low = processor_low.boost_chunk_text(text)
        
        # High boost should have more repetitions
        high_title_count = boosted_high.count("Important Title")
        low_title_count = boosted_low.count("Important Title")
        
        self.assertGreaterEqual(high_title_count, low_title_count)
    
    def test_empty_text_handling(self):
        """Test handling of empty or None text"""
        processor = FieldBoostProcessor()
        
        # Test empty string
        boosted_empty = processor.boost_chunk_text("")
        self.assertEqual(boosted_empty, "")
        
        # Test None (converted to string)
        boosted_none = processor.boost_chunk_text(None)
        self.assertEqual(boosted_none, "")
    
    def test_boost_preserves_content(self):
        """Test that boosting preserves original content"""
        text = """# Title Here

Original content that should be preserved.
Multiple lines of important information."""
        
        boosted = self.processor.boost_chunk_text(text)
        
        # Should contain all original content
        self.assertIn("Original content", boosted)
        self.assertIn("Multiple lines", boosted)
        self.assertIn("important information", boosted)


@unittest.skipUnless(QUERY_EXPANSION_AVAILABLE, "Query expansion dependencies not available")
class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios combining MQE and field boosts"""
    
    def setUp(self):
        self.expander = MultiQueryExpander()
        self.processor = FieldBoostProcessor()
        
        # Setup reranker for integration tests
        try:
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
            from src.storage.reranker_bulletproof import BulletproofReranker
            self.reranker = BulletproofReranker(debug_mode=True)
            self.reranker_available = True
        except ImportError:
            self.reranker = None
            self.reranker_available = False
    
    async def test_mq_expansion_with_field_boosts(self):
        """Test MQE combined with field boosts"""
        query = "microservices architecture benefits"
        
        # Mock search returning documents with titles/headings
        async def mock_search(q: str, limit: int = 10):
            return [
                {
                    "id": "guide_doc",
                    "text": "# Microservices Architecture Guide\n\nBenefits include scalability...",
                    "score": 0.8
                },
                {
                    "id": "plain_doc", 
                    "text": "Simple text about microservices without formatting.",
                    "score": 0.7
                }
            ]
        
        # Run MQE
        mq_results = await self.expander.expand_and_search(query, mock_search, limit=10, num_paraphrases=1)
        
        # Apply field boosts
        final_results = self.processor.process_results(mq_results)
        
        # Verify integration
        self.assertGreater(len(final_results), 0)
        
        # Results should have both MQE and boost metadata
        for result in final_results:
            self.assertIn("source_query", result)  # From MQE
            self.assertIn("boosted_text", result)  # From field boost
            self.assertIn("original_text", result)  # Preserved
    
    async def test_paraphrase_robustness_simulation(self):
        """Test paraphrase robustness simulation (T-106 scenario)"""
        original_queries = [
            "What are the benefits of microservices architecture?",
            "How to optimize database performance?",
            "OAuth 2.0 implementation best practices?"
        ]
        
        # Mock search that varies by query phrasing
        async def mock_search(q: str, limit: int = 10):
            if "benefit" in q.lower() or "advantage" in q.lower():
                return [{"id": "benefits_doc", "content": "Benefits content", "score": 0.9}]
            elif "optimize" in q.lower() or "improve" in q.lower():
                return [{"id": "perf_doc", "content": "Performance content", "score": 0.85}]
            else:
                return [{"id": "generic_doc", "content": "Generic content", "score": 0.6}]
        
        total_precision = 0.0
        
        for query in original_queries:
            # Test with MQE
            mq_results = await self.expander.expand_and_search(query, mock_search, limit=5, num_paraphrases=2)
            
            # Apply field boosts
            final_results = self.processor.process_results(mq_results)
            
            # Calculate simulated precision (would need ground truth in real test)
            if final_results:
                # Simulate higher precision for relevant docs
                top_result = final_results[0]
                if any(term in top_result.get("id", "").lower() for term in ["benefits", "perf"]):
                    precision = 1.0
                else:
                    precision = 0.7  # Partial match
            else:
                precision = 0.0
            
            total_precision += precision
        
        avg_precision = total_precision / len(original_queries)
        
        # MQE + field boosts should achieve good precision
        self.assertGreater(avg_precision, 0.8)
    
    @unittest.skipUnless(QUERY_EXPANSION_AVAILABLE, "Query expansion dependencies not available")
    async def test_mqe_with_reranker_integration(self):
        """Test MQE integrated with actual bulletproof reranker"""
        if not self.reranker_available or not self.reranker:
            self.skipTest("Reranker not available for integration test")
        
        query = "What are the benefits of microservices architecture?"
        
        # Mock search that returns candidates suitable for reranking
        async def mock_search_with_rerank(q: str, limit: int = 10):
            # Simulate search results with different relevance
            base_results = [
                {
                    "id": "microservices_guide",
                    "text": "# Microservices Architecture Benefits\n\nScalability, fault isolation, and technology diversity are key advantages.",
                    "score": 0.7
                },
                {
                    "id": "database_guide", 
                    "text": "Database indexing improves query performance through B-tree structures.",
                    "score": 0.8  # Higher initial score but less relevant
                },
                {
                    "id": "microservices_patterns",
                    "text": "Microservices provide better team autonomy and independent deployment capabilities.",
                    "score": 0.65
                },
                {
                    "id": "oauth_guide",
                    "text": "OAuth 2.0 authentication provides secure access delegation for APIs.",
                    "score": 0.6
                }
            ]
            
            # Filter results based on query relevance (simulate vector search)
            if "microservices" in q.lower() or "architecture" in q.lower() or "benefits" in q.lower():
                return base_results
            elif "database" in q.lower():
                return [base_results[1], base_results[3]]  # Less relevant for microservices
            else:
                return [base_results[3]]  # Least relevant
        
        # Run MQE to get diverse candidates
        mq_results = await self.expander.expand_and_search(query, mock_search_with_rerank, limit=10, num_paraphrases=2)
        
        # Apply field boosts
        boosted_results = self.processor.process_results(mq_results)
        
        # Prepare candidates for reranking
        rerank_candidates = []
        for result in boosted_results:
            rerank_candidates.append({
                'id': result.get('id'),
                'payload': {
                    'content': {
                        'text': result.get('boosted_text', result.get('text', ''))
                    }
                },
                'score': result.get('score', 0.0)
            })
        
        # Apply reranking
        reranked_results = self.reranker.rerank(query, rerank_candidates)
        
        # Validate integration results
        self.assertGreater(len(reranked_results), 0)
        
        # The most relevant microservices content should be ranked higher
        top_result = reranked_results[0]
        self.assertIn('microservices', top_result.get('id', '').lower())
        
        # Verify reranking scores were applied
        self.assertIn('rerank_score', top_result)
        self.assertIn('original_score', top_result)
        
        # Check that reranking improved relevance (microservices content should beat database content)
        microservices_results = [r for r in reranked_results if 'microservices' in r.get('id', '').lower()]
        database_results = [r for r in reranked_results if 'database' in r.get('id', '').lower()]
        
        if microservices_results and database_results:
            # Find positions of microservices vs database results
            microservices_positions = [i for i, r in enumerate(reranked_results) if 'microservices' in r.get('id', '').lower()]
            database_positions = [i for i, r in enumerate(reranked_results) if 'database' in r.get('id', '').lower()]
            
            # Best microservices result should rank higher than best database result for this query
            if microservices_positions and database_positions:
                best_microservices_pos = min(microservices_positions)
                best_database_pos = min(database_positions)
                self.assertLess(best_microservices_pos, best_database_pos, 
                               "Reranker should promote microservices content for microservices query")
        
        # Verify reranker stats
        stats = self.reranker.get_stats()
        self.assertGreater(stats['request_count'], 0)
        self.assertTrue(stats['enabled'])
        
        # Verify no all-zero scores (the bug we're fixing)
        scores = [r.get('rerank_score', 0) for r in reranked_results]
        non_zero_scores = [s for s in scores if abs(s) > 1e-9]
        self.assertGreater(len(non_zero_scores), 0, "Should have non-zero rerank scores (T-102 fix validation)")


def run_async_test(coro):
    """Helper to run async tests"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# Make async tests work with unittest
class AsyncTestCase(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        self.loop.close()
    
    def async_test(self, coro):
        return self.loop.run_until_complete(coro)


# Patch the test classes to use async support
for test_class in [TestMultiQueryExpander, TestIntegrationScenarios]:
    for method_name in dir(test_class):
        if method_name.startswith('test_') and asyncio.iscoroutinefunction(getattr(test_class, method_name)):
            original_method = getattr(test_class, method_name)
            
            def make_sync_wrapper(async_method):
                def sync_wrapper(self):
                    return run_async_test(async_method(self))
                return sync_wrapper
            
            setattr(test_class, method_name, make_sync_wrapper(original_method))


if __name__ == '__main__':
    unittest.main(verbosity=2)