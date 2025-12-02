#!/usr/bin/env python3
"""
test_multi_chunk_narrative.py: Sprint 11 Phase 3 Multi-Chunk Narrative Test

Tests Sprint 11 Phase 3 Task 3 requirements:
- Answer requires 2+ chunks; ensure ordering & dedup
- ≥2 sources cited in narrative response
- No duplicated spans; chunk drift < 5%
"""

import asyncio
import pytest
import logging
import os
import sys
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
from typing import List, Dict, Any, Set

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

try:
    from src.mcp_server.server import store_context_tool, retrieve_context_tool
    from src.core.config import Config
    from src.storage.reranker import get_reranker
    from src.storage.hybrid_scorer import HybridScorer, ScoringMode
except ImportError as e:
    print(f"Import error: {e}")
    pytest.skip("Required modules not available", allow_module_level=True)

# Setup logging to capture narrative construction
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestMultiChunkNarrative:
    """Sprint 11 Phase 3 Task 3: Multi-Chunk Narrative Tests"""
    
    @pytest.fixture
    def multi_chunk_knowledge_base(self):
        """Knowledge base requiring multiple chunks to answer comprehensively"""
        return [
            {
                "content": {
                    "id": "sprint11_overview",
                    "title": "Sprint 11: Veris-Memory Cleanup & Interface Tightening",
                    "description": "This sprint focuses on freezing the v1.0 API contract and validating system integrity",
                    "goals": [
                        "Freeze a simple v1 API contract (breaking changes OK)",
                        "Validate schema/index integrity (no dimension drift; correct HNSW params)"
                    ],
                    "phase": "planning",
                    "duration": "5 days"
                },
                "type": "sprint",
                "metadata": {
                    "source": "sprint_planning",
                    "tags": ["sprint11", "api", "planning"],
                    "priority": "high",
                    "chunk_id": "chunk_001"
                }
            },
            {
                "content": {
                    "id": "sprint11_phase1",
                    "title": "Phase 1: API Contract Freeze (Greenfield)",
                    "description": "Define and lock a minimal v1 for MCP tools and HTTP API; remove legacy paths",
                    "tasks": [
                        "Define Minimal v1 Spec - Document request/response for store_context, retrieve_context, query_graph",
                        "Remove Legacy/Deprecated Endpoints - Delete unused routes, legacy field adapters",
                        "Error Semantics - Standardize error codes: ERR_TIMEOUT, ERR_AUTH, ERR_RATE_LIMIT"
                    ],
                    "duration_days": 1,
                    "priority": "high"
                },
                "type": "sprint",
                "metadata": {
                    "source": "sprint_planning",
                    "tags": ["sprint11", "phase1", "api-contract"],
                    "priority": "high",
                    "chunk_id": "chunk_002"
                }
            },
            {
                "content": {
                    "id": "sprint11_phase2",
                    "title": "Phase 2: Data Model & Index Integrity",
                    "description": "Assert canonical dims and index params; block ingest on drift",
                    "tasks": [
                        "Embedding Dimension Guard - Enforce Qdrant 384 dims across collections",
                        "HNSW/ef Params Check - Verify M=32, ef_search=256 default",
                        "Idempotent Migrations - Run migration twice; no dup nodes/edges"
                    ],
                    "critical_requirement": "384 dimensions must be enforced across all vector collections",
                    "duration_days": 1,
                    "priority": "high"
                },
                "type": "sprint", 
                "metadata": {
                    "source": "sprint_planning",
                    "tags": ["sprint11", "phase2", "data-integrity"],
                    "priority": "critical",
                    "chunk_id": "chunk_003"
                }
            },
            {
                "content": {
                    "id": "sprint11_phase3",
                    "title": "Phase 3: Functional Golden Paths",
                    "description": "E2E tests through SDK only (no legacy clients)",
                    "tasks": [
                        "Fact Recall (Name) E2E - Store 'My name is Matt' → ask 'What is my name?' + 3 paraphrases",
                        "Code Search Sanity - Ingest small repo; query 'server.py' + a function",
                        "Multi-Chunk Narrative - Answer requires 2+ chunks; ensure ordering & dedup"
                    ],
                    "success_criteria": "P@1 = 1.0 for 4/4 queries in fact recall",
                    "duration_days": 1,
                    "priority": "high"
                },
                "type": "sprint",
                "metadata": {
                    "source": "sprint_planning", 
                    "tags": ["sprint11", "phase3", "testing"],
                    "priority": "high",
                    "chunk_id": "chunk_004"
                }
            },
            {
                "content": {
                    "id": "sprint11_success_metrics",
                    "title": "Sprint 11 Success Metrics",
                    "description": "Key performance indicators and success criteria for sprint completion",
                    "metrics": [
                        {
                            "metric": "api_contract_locked",
                            "target": 1,
                            "unit": "boolean",
                            "description": "v1 spec merged & published"
                        },
                        {
                            "metric": "golden_paths_passing", 
                            "target": 3,
                            "unit": "count",
                            "description": "All phase-3 tests green"
                        },
                        {
                            "metric": "index_integrity",
                            "target": 1,
                            "unit": "boolean",
                            "description": "Dims/params verified"
                        }
                    ],
                    "overall_goal": "Eliminate all 'NoneType' object has no attribute 'call_tool' errors"
                },
                "type": "sprint",
                "metadata": {
                    "source": "sprint_planning",
                    "tags": ["sprint11", "metrics", "success-criteria"],
                    "priority": "medium",
                    "chunk_id": "chunk_005"
                }
            }
        ]
    
    @pytest.fixture
    def narrative_queries(self):
        """Queries requiring synthesis from multiple chunks"""
        return [
            {
                "query": "What are the main goals and phases of Sprint 11?",
                "expected_chunks": ["chunk_001", "chunk_002", "chunk_003", "chunk_004"],
                "expected_sources": 4,
                "narrative_type": "comprehensive_overview"
            },
            {
                "query": "Describe the data integrity requirements and dimension changes in Sprint 11",
                "expected_chunks": ["chunk_002", "chunk_003"],
                "expected_sources": 2,
                "narrative_type": "technical_deep_dive"
            },
            {
                "query": "How will Sprint 11 success be measured and what are the key deliverables?",
                "expected_chunks": ["chunk_001", "chunk_004", "chunk_005"],
                "expected_sources": 3,
                "narrative_type": "success_criteria_narrative"
            }
        ]
    
    @pytest.mark.asyncio
    async def test_knowledge_base_ingestion(self, multi_chunk_knowledge_base):
        """Test ingestion of multi-chunk knowledge base"""
        
        # Mock storage to track ingested chunks
        ingested_chunks = []
        
        def mock_store_context(context_data):
            chunk_id = context_data["metadata"]["chunk_id"]
            ingested_chunks.append(chunk_id)
            return {"success": True, "id": f"ctx_{chunk_id}"}
        
        with patch('src.mcp_server.server.kv_store') as mock_kv, \
             patch('src.mcp_server.server.vector_db') as mock_vector, \
             patch('src.mcp_server.server.graph_db') as mock_graph:
            
            # Setup mocks
            mock_kv.store_context.side_effect = mock_store_context
            mock_vector.store_embeddings.return_value = {"success": True}
            mock_graph.create_nodes.return_value = {"success": True}
            
            # Ingest all knowledge chunks
            for chunk_data in multi_chunk_knowledge_base:
                result = await store_context_tool(chunk_data)
                assert result["success"] is True
                
                chunk_id = chunk_data["metadata"]["chunk_id"]
                logger.info(f"✅ Ingested chunk: {chunk_id} - {chunk_data['content']['title']}")
        
        # Verify all chunks were ingested
        assert len(ingested_chunks) == len(multi_chunk_knowledge_base)
        assert set(ingested_chunks) == {f"chunk_{i:03d}" for i in range(1, 6)}
        
        logger.info(f"✅ Knowledge base ingestion complete: {len(multi_chunk_knowledge_base)} chunks")
        return multi_chunk_knowledge_base
    
    def _calculate_chunk_overlap(self, chunk1_content: str, chunk2_content: str) -> float:
        """Calculate content overlap between two chunks (for deduplication)"""
        import re
        
        # Tokenize content (simple word-based)
        words1 = set(re.findall(r'\w+', chunk1_content.lower()))
        words2 = set(re.findall(r'\w+', chunk2_content.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _detect_duplicate_spans(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect duplicate content spans in multi-chunk results"""
        duplicate_info = {
            "total_chunks": len(results),
            "duplicate_pairs": [],
            "max_overlap": 0.0,
            "avg_overlap": 0.0,
            "has_duplicates": False
        }
        
        overlaps = []
        
        # Compare each pair of chunks
        for i, result1 in enumerate(results):
            for j, result2 in enumerate(results[i+1:], i+1):
                content1 = str(result1["content"])
                content2 = str(result2["content"])
                
                overlap = self._calculate_chunk_overlap(content1, content2)
                overlaps.append(overlap)
                
                if overlap > 0.1:  # 10% overlap threshold
                    duplicate_info["duplicate_pairs"].append({
                        "chunk1_idx": i,
                        "chunk2_idx": j,
                        "overlap": overlap,
                        "chunk1_id": result1.get("metadata", {}).get("chunk_id", f"unknown_{i}"),
                        "chunk2_id": result2.get("metadata", {}).get("chunk_id", f"unknown_{j}")
                    })
        
        if overlaps:
            duplicate_info["max_overlap"] = max(overlaps)
            duplicate_info["avg_overlap"] = sum(overlaps) / len(overlaps)
            duplicate_info["has_duplicates"] = duplicate_info["max_overlap"] > 0.1
        
        return duplicate_info
    
    @pytest.mark.asyncio
    async def test_multi_chunk_narrative_queries(self, multi_chunk_knowledge_base, narrative_queries):
        """Test narrative queries requiring multiple chunks"""
        
        # First ensure knowledge base is ingested
        await self.test_knowledge_base_ingestion(multi_chunk_knowledge_base)
        
        narrative_results = []
        
        for query_data in narrative_queries:
            query = query_data["query"]
            expected_chunks = query_data["expected_chunks"] 
            expected_sources = query_data["expected_sources"]
            narrative_type = query_data["narrative_type"]
            
            logger.info(f"Testing {narrative_type} query: '{query}'")
            
            # Mock multi-chunk retrieval
            with patch('src.mcp_server.server.vector_db') as mock_vector, \
                 patch('src.mcp_server.server.graph_db') as mock_graph, \
                 patch('src.mcp_server.server.kv_store') as mock_kv:
                
                # Create mock results from expected chunks
                mock_results = []
                for chunk_id in expected_chunks:
                    # Find the corresponding chunk data
                    chunk_data = next(
                        (chunk for chunk in multi_chunk_knowledge_base 
                         if chunk["metadata"]["chunk_id"] == chunk_id), 
                        None
                    )
                    
                    if chunk_data:
                        mock_results.append({
                            "id": f"ctx_{chunk_id}",
                            "content": chunk_data["content"],
                            "score": 0.85 + (0.1 * len(mock_results)),  # Varying scores
                            "type": chunk_data["type"],
                            "metadata": chunk_data["metadata"],
                            "chunk_order": len(mock_results) + 1
                        })
                
                # Setup mocks
                mock_vector.search.return_value = mock_results
                mock_graph.traverse.return_value = mock_results
                mock_kv.get_context.return_value = mock_results[0] if mock_results else None
                
                # Execute narrative query
                search_request = {
                    "query": query,
                    "type": "sprint",
                    "search_mode": "hybrid",
                    "limit": 20,  # Allow multiple chunks
                    "include_relationships": True,
                    "sort_by": "relevance"
                }
                
                result = await retrieve_context_tool(search_request)
            
            # Verify query was successful
            assert result["success"] is True
            assert len(result["results"]) >= 2, f"Multi-chunk narrative requires ≥2 chunks, got {len(result['results'])}"
            
            # Sprint 11 requirement: ≥2 sources cited
            sources_count = len(result["results"])
            assert sources_count >= expected_sources, f"Expected ≥{expected_sources} sources, got {sources_count}"
            
            # Sprint 11 requirement: No duplicated spans; chunk drift <5%
            duplicate_info = self._detect_duplicate_spans(result["results"])
            chunk_drift = duplicate_info["max_overlap"]
            
            assert chunk_drift < 0.05, f"Chunk drift {chunk_drift:.3f} exceeds 5% threshold"
            assert not duplicate_info["has_duplicates"], f"Duplicate spans detected: {duplicate_info['duplicate_pairs']}"
            
            # Verify proper ordering (by relevance score)
            scores = [r["score"] for r in result["results"]]
            is_properly_ordered = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
            assert is_properly_ordered, f"Results not properly ordered by relevance: {scores}"
            
            # Log narrative construction details
            logger.info(f"  Sources used: {sources_count}")
            logger.info(f"  Chunk drift: {chunk_drift:.3f} (<5% ✓)")
            logger.info(f"  Properly ordered: {is_properly_ordered} ✓")
            logger.info(f"  Average relevance: {sum(scores)/len(scores):.3f}")
            
            narrative_results.append({
                "query": query,
                "narrative_type": narrative_type,
                "sources_count": sources_count,
                "chunk_drift": chunk_drift,
                "properly_ordered": is_properly_ordered,
                "duplicate_info": duplicate_info,
                "avg_relevance": sum(scores) / len(scores)
            })
            
            logger.info(f"✅ {narrative_type} query passed: {sources_count} sources, {chunk_drift:.3f} drift")
        
        # Verify overall Sprint 11 requirements
        all_multi_chunk = all(r["sources_count"] >= 2 for r in narrative_results)
        all_low_drift = all(r["chunk_drift"] < 0.05 for r in narrative_results)
        all_ordered = all(r["properly_ordered"] for r in narrative_results)
        
        assert all_multi_chunk, "Not all queries used ≥2 sources"
        assert all_low_drift, "Some queries exceeded 5% chunk drift"
        assert all_ordered, "Some results not properly ordered"
        
        logger.info(f"🎯 Multi-Chunk Narrative PASSED: {len(narrative_results)} queries successful")
        return narrative_results
    
    @pytest.mark.asyncio
    async def test_narrative_deduplication(self, multi_chunk_knowledge_base):
        """Test that narrative construction properly deduplicates overlapping content"""
        
        # Create test scenario with intentionally overlapping chunks
        overlapping_chunks = [
            {
                "content": {
                    "id": "overlap_test_1",
                    "title": "Sprint 11 Overview",
                    "description": "Sprint 11 focuses on API contract freeze and system validation. The main goal is to eliminate NoneType errors and establish v1.0 compliance."
                },
                "type": "sprint",
                "metadata": {"chunk_id": "overlap_001"}
            },
            {
                "content": {
                    "id": "overlap_test_2", 
                    "title": "Sprint 11 Goals",
                    "description": "The primary objectives of Sprint 11 include API contract freeze, system validation, and eliminating NoneType errors through v1.0 compliance."
                },
                "type": "sprint", 
                "metadata": {"chunk_id": "overlap_002"}
            },
            {
                "content": {
                    "id": "unique_test_3",
                    "title": "Sprint 11 Technical Details",
                    "description": "Technical implementation requires 384-dimension vectors, HNSW parameter validation, and comprehensive E2E testing."
                },
                "type": "sprint",
                "metadata": {"chunk_id": "unique_003"}
            }
        ]
        
        # Test overlap detection
        overlap_12 = self._calculate_chunk_overlap(
            str(overlapping_chunks[0]["content"]),
            str(overlapping_chunks[1]["content"])
        )
        
        overlap_13 = self._calculate_chunk_overlap(
            str(overlapping_chunks[0]["content"]),
            str(overlapping_chunks[2]["content"])
        )
        
        # Should detect high overlap between chunks 1 and 2
        assert overlap_12 > 0.3, f"Expected high overlap between similar chunks, got {overlap_12:.3f}"
        
        # Should detect low overlap between chunks 1 and 3  
        assert overlap_13 < 0.2, f"Expected low overlap between different chunks, got {overlap_13:.3f}"
        
        # Test duplicate detection in mock results
        mock_results = [
            {"content": overlapping_chunks[0]["content"], "metadata": overlapping_chunks[0]["metadata"]},
            {"content": overlapping_chunks[1]["content"], "metadata": overlapping_chunks[1]["metadata"]},
            {"content": overlapping_chunks[2]["content"], "metadata": overlapping_chunks[2]["metadata"]}
        ]
        
        duplicate_info = self._detect_duplicate_spans(mock_results)
        
        # Should detect the overlapping pair
        assert duplicate_info["has_duplicates"], "Failed to detect duplicate content spans"
        assert len(duplicate_info["duplicate_pairs"]) > 0, "No duplicate pairs detected"
        assert duplicate_info["max_overlap"] > 0.1, f"Max overlap {duplicate_info['max_overlap']:.3f} too low"
        
        logger.info(f"✅ Deduplication test passed: detected {len(duplicate_info['duplicate_pairs'])} duplicate pairs")
        logger.info(f"   Max overlap: {duplicate_info['max_overlap']:.3f}")
    
    @pytest.mark.asyncio
    async def test_narrative_ordering_verification(self, multi_chunk_knowledge_base):
        """Test that multi-chunk results maintain proper relevance ordering"""
        
        # Create test results with various relevance scores
        test_results = [
            {"id": "ctx_001", "content": {"title": "High relevance"}, "score": 0.95},
            {"id": "ctx_002", "content": {"title": "Medium relevance"}, "score": 0.87}, 
            {"id": "ctx_003", "content": {"title": "Lower relevance"}, "score": 0.82},
            {"id": "ctx_004", "content": {"title": "Lowest relevance"}, "score": 0.79}
        ]
        
        # Test proper ordering
        scores = [r["score"] for r in test_results]
        is_descending = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
        assert is_descending, f"Results not in descending order: {scores}"
        
        # Test ordering preservation in narrative construction
        with patch('src.mcp_server.server.vector_db') as mock_vector:
            mock_vector.search.return_value = test_results
            
            result = await retrieve_context_tool({
                "query": "test narrative ordering",
                "type": "all",
                "search_mode": "hybrid", 
                "limit": 10,
                "sort_by": "relevance"
            })
            
            # Verify ordering maintained
            result_scores = [r["score"] for r in result["results"]]
            ordering_preserved = all(result_scores[i] >= result_scores[i+1] for i in range(len(result_scores)-1))
            
            assert ordering_preserved, f"Ordering not preserved in results: {result_scores}"
            
            logger.info(f"✅ Ordering verification passed: {len(result_scores)} results properly ordered")
    
    @pytest.mark.asyncio
    async def test_end_to_end_multi_chunk_narrative(self, multi_chunk_knowledge_base, narrative_queries):
        """Complete E2E test for multi-chunk narrative construction"""
        
        logger.info("🚀 Starting Sprint 11 Phase 3 Multi-Chunk Narrative E2E Test")
        
        # Step 1: Knowledge Base Ingestion
        logger.info("Step 1: Ingesting multi-chunk knowledge base")
        await self.test_knowledge_base_ingestion(multi_chunk_knowledge_base)
        logger.info(f"✅ Step 1 complete: {len(multi_chunk_knowledge_base)} chunks ingested")
        
        # Step 2: Multi-Chunk Narrative Queries
        logger.info("Step 2: Testing multi-chunk narrative queries")
        narrative_results = await self.test_multi_chunk_narrative_queries(multi_chunk_knowledge_base, narrative_queries)
        logger.info(f"✅ Step 2 complete: {len(narrative_results)} narrative queries tested")
        
        # Step 3: Deduplication Verification
        logger.info("Step 3: Testing deduplication capabilities")
        await self.test_narrative_deduplication(multi_chunk_knowledge_base)
        logger.info("✅ Step 3 complete: Deduplication verified")
        
        # Step 4: Ordering Verification
        logger.info("Step 4: Testing result ordering")
        await self.test_narrative_ordering_verification(multi_chunk_knowledge_base)
        logger.info("✅ Step 4 complete: Ordering verified")
        
        # Step 5: Overall Sprint 11 Requirements Check
        logger.info("Step 5: Verifying Sprint 11 requirements")
        
        # Verify ≥2 sources for all narratives
        multi_source_queries = len([r for r in narrative_results if r["sources_count"] >= 2])
        assert multi_source_queries == len(narrative_results), f"Not all queries used ≥2 sources: {multi_source_queries}/{len(narrative_results)}"
        
        # Verify chunk drift <5% for all narratives
        low_drift_queries = len([r for r in narrative_results if r["chunk_drift"] < 0.05])
        assert low_drift_queries == len(narrative_results), f"Some queries exceeded 5% drift: {low_drift_queries}/{len(narrative_results)}"
        
        # Verify proper ordering for all narratives
        ordered_queries = len([r for r in narrative_results if r["properly_ordered"]])
        assert ordered_queries == len(narrative_results), f"Some queries not properly ordered: {ordered_queries}/{len(narrative_results)}"
        
        logger.info("✅ Step 5 complete: All Sprint 11 requirements verified")
        
        # Summary
        avg_sources = sum(r["sources_count"] for r in narrative_results) / len(narrative_results)
        max_drift = max(r["chunk_drift"] for r in narrative_results)
        avg_relevance = sum(r["avg_relevance"] for r in narrative_results) / len(narrative_results)
        
        logger.info(f"🎯 Multi-Chunk Narrative E2E Test PASSED:")
        logger.info(f"   - Narrative queries tested: {len(narrative_results)}")
        logger.info(f"   - Average sources per query: {avg_sources:.1f}")
        logger.info(f"   - Maximum chunk drift: {max_drift:.3f} (<5% ✓)")
        logger.info(f"   - Average relevance score: {avg_relevance:.3f}")
        logger.info(f"   - All results properly ordered: ✓")
        logger.info(f"   - No duplicate spans detected: ✓")


if __name__ == "__main__":
    # Run the multi-chunk narrative tests
    pytest.main([__file__, "-v", "-s"])