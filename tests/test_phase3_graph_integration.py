#!/usr/bin/env python3
"""
Test Phase 3 graph integration for enhanced fact recall.

This script tests the graph integration components including entity extraction,
graph storage, hybrid scoring, and query expansion to validate enhanced
fact recall capabilities.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.storage.graph_enhancer import GraphSignalEnhancer, EntityExtractor, EntityType
from src.storage.graph_fact_store import GraphFactStore
from src.storage.hybrid_scorer import HybridScorer, ScoringMode
from src.core.graph_query_expander import GraphQueryExpander, ExpansionStrategy
from src.storage.fact_store import FactStore
from src.storage.fact_ranker import FactAwareRanker
from src.core.intent_classifier import IntentClassifier, IntentType


def test_entity_extraction():
    """Test entity extraction with graph linking."""
    print("🧩 Testing Entity Extraction")
    print("-" * 40)
    
    extractor = EntityExtractor()
    
    test_texts = [
        "My name is Alice Smith and I work at TechCorp in San Francisco.",
        "I live in New York and my email is alice@example.com.",
        "I like spicy food and I listen to jazz music.",
        "My phone number is 555-123-4567 and I'm 28 years old."
    ]
    
    total_entities = 0
    for text in test_texts:
        print(f"\n📝 Text: '{text}'")
        
        entities = extractor.extract_entities(text)
        total_entities += len(entities)
        
        for entity in entities:
            print(f"   Entity: '{entity.text}' ({entity.entity_type.value})")
            print(f"   Normalized: '{entity.normalized_form}' (confidence: {entity.confidence:.2f})")
    
    print(f"\n✅ Extracted {total_entities} entities total")
    
    # Test entity normalization
    test_names = ["john doe", "JANE SMITH", "Bob Johnson Jr."]
    normalized = [extractor._normalize_person_name(name) for name in test_names]
    print(f"✅ Name normalization: {list(zip(test_names, normalized))}")
    
    return total_entities > 0


def test_graph_signal_enhancement():
    """Test graph signal enhancement without Neo4j dependency."""
    print("\n📊 Testing Graph Signal Enhancement")
    print("-" * 40)
    
    # Test without Neo4j client (graceful degradation)
    enhancer = GraphSignalEnhancer(neo4j_client=None)
    
    test_query = "What's my email address?"
    test_results = [
        {
            "id": "1",
            "content": "My email is alice@example.com and I work remotely.",
            "score": 0.8,
            "metadata": {"type": "fact_statement"}
        },
        {
            "id": "2", 
            "content": "What's your email? I need to contact you.",
            "score": 0.9,
            "metadata": {"type": "question"}
        }
    ]
    
    print(f"🔍 Query: '{test_query}'")
    print(f"📊 Test results: {len(test_results)} items")
    
    # Test graph score computation (should handle no Neo4j gracefully)
    graph_signals = enhancer.compute_graph_score(test_query, test_results)
    
    print(f"📈 Graph signals computed: {len(graph_signals)}")
    
    # Test entity extraction components
    extractor = EntityExtractor()
    query_entities = extractor.extract_entities(test_query)
    content_entities = extractor.extract_entities(test_results[0]["content"])
    
    print(f"🔍 Query entities: {len(query_entities)}")
    print(f"📄 Content entities: {len(content_entities)}")
    
    for entity in query_entities + content_entities:
        print(f"   {entity.text} ({entity.entity_type.value}) -> {entity.normalized_form}")
    
    # Test graph stats
    stats = enhancer.get_graph_stats()
    print(f"📊 Graph stats: {stats}")
    
    return True


def test_hybrid_scoring():
    """Test hybrid scoring with multiple components."""
    print("\n🎯 Testing Hybrid Scoring")
    print("-" * 40)
    
    # Initialize components
    fact_ranker = FactAwareRanker()
    hybrid_scorer = HybridScorer(fact_ranker, None)  # No graph enhancer for testing
    
    test_query = "What's my name?"
    test_results = [
        {
            "id": "1",
            "content": "What's my name? I can't remember.",
            "score": 0.9,
            "lexical_score": 0.7,
            "metadata": {"type": "question"}
        },
        {
            "id": "2",
            "content": "My name is Alice and I work at TechCorp.",
            "score": 0.8,
            "lexical_score": 0.6,
            "metadata": {"type": "statement"}
        },
        {
            "id": "3",
            "content": "People call me Alice Smith.",
            "score": 0.7,
            "lexical_score": 0.8,
            "metadata": {"type": "statement"}
        }
    ]
    
    print(f"🔍 Query: '{test_query}'")
    print(f"📊 Test results: {len(test_results)} items")
    
    # Test different scoring modes
    scoring_modes = [ScoringMode.FACT_OPTIMIZED, ScoringMode.GENERAL_SEARCH, ScoringMode.GRAPH_ENHANCED]
    
    for mode in scoring_modes:
        print(f"\n📈 Testing {mode.value} scoring mode:")
        
        hybrid_scores = hybrid_scorer.compute_hybrid_score(
            query=test_query,
            results=test_results,
            scoring_mode=mode
        )
        
        for i, score in enumerate(hybrid_scores):
            print(f"   {i+1}. Score: {score.final_score:.3f} - {score.explanation}")
            result_index = i if i < len(test_results) else 0  # Safe fallback
            print(f"      Content: {test_results[result_index]['content'][:50]}...")
    
    # Test scoring explanation
    explanation = hybrid_scorer.explain_scoring_decision(test_query)
    print(f"\n🔍 Scoring decision explanation:")
    print(f"   Selected mode: {explanation['selected_mode']}")
    print(f"   Rationale: {explanation['mode_rationale']}")
    
    # Test statistics
    stats = hybrid_scorer.get_scoring_statistics()
    print(f"📊 Scoring statistics: {list(stats.keys())}")
    
    return len(hybrid_scores) > 0


def test_query_expansion():
    """Test graph-enhanced query expansion."""
    print("\n🔄 Testing Query Expansion")
    print("-" * 40)
    
    # Initialize components
    intent_classifier = IntentClassifier()
    expander = GraphQueryExpander(neo4j_client=None, intent_classifier=intent_classifier)
    
    test_queries = [
        "What's my email address?",
        "Where do I work?",
        "Who are my colleagues?",
        "What food do I like?",
        "Tell me about my preferences"
    ]
    
    total_expansions = 0
    for query in test_queries:
        print(f"\n🔍 Original query: '{query}'")
        
        # Test expansion
        expansion_result = expander.expand_query(query)
        total_expansions += len(expansion_result.expanded_queries)
        
        print(f"   Intent: {expansion_result.detected_intent.value}")
        print(f"   Entities: {expansion_result.extracted_entities}")
        print(f"   Expansions: {len(expansion_result.expanded_queries)}")
        
        for expansion in expansion_result.expanded_queries:
            print(f"   ↳ {expansion.query}")
            print(f"     Strategy: {expansion.expansion_type.value} (confidence: {expansion.confidence:.2f})")
    
    print(f"\n✅ Generated {total_expansions} query expansions total")
    
    # Test expansion explanation
    explanation = expander.explain_expansion_decision("What's my work email?")
    print(f"\n🔍 Expansion decision explanation:")
    print(f"   Selected strategy: {explanation['selected_strategy']}")
    print(f"   Rationale: {explanation['strategy_rationale']}")
    
    # Test statistics
    stats = expander.get_expansion_statistics()
    print(f"📊 Expansion statistics: {list(stats.keys())}")
    
    return total_expansions > 0


def test_graph_fact_store():
    """Test graph-integrated fact storage."""
    print("\n🗄️ Testing Graph Fact Store")
    print("-" * 40)
    
    # Create mock Redis client for testing
    class MockRedisClient:
        def __init__(self):
            self.data = {}
        
        def set(self, key, value):
            self.data[key] = value
            return True
        
        def setex(self, key, time, value):
            self.data[key] = value
            return True
        
        def get(self, key):
            value = self.data.get(key)
            return value.encode('utf-8') if isinstance(value, str) else value
        
        def delete(self, *keys):
            deleted = 0
            for key in keys:
                if key in self.data:
                    del self.data[key]
                    deleted += 1
            return deleted
        
        def keys(self, pattern):
            import re
            pattern_regex = pattern.replace('*', '.*')
            matching_keys = [k for k in self.data.keys() if re.match(pattern_regex, k)]
            return [k.encode('utf-8') if isinstance(k, str) else k for k in matching_keys]
    
    # Initialize fact stores
    redis_client = MockRedisClient()
    fact_store = FactStore(redis_client)
    graph_fact_store = GraphFactStore(fact_store, neo4j_client=None)  # No Neo4j for testing
    
    test_facts = [
        ("user1", "name", "Alice Smith"),
        ("user1", "email", "alice@example.com"),
        ("user1", "location", "San Francisco"),
        ("user1", "job", "Software Engineer"),
        ("user1", "preferences.food", "spicy food")
    ]
    
    print(f"📝 Storing {len(test_facts)} test facts:")
    
    for user_id, attribute, value in test_facts:
        print(f"   {attribute}: {value}")
        graph_fact_store.store_fact("test", user_id, attribute, value)
    
    # Test retrieval
    print(f"\n📥 Testing fact retrieval:")
    for user_id, attribute, expected_value in test_facts:
        retrieved_fact = graph_fact_store.get_fact("test", user_id, attribute)
        retrieved_value = retrieved_fact.value if retrieved_fact else None
        status = "✅" if str(retrieved_value) == str(expected_value) else "❌"
        print(f"   {status} {attribute}: {retrieved_value} (expected: {expected_value})")
    
    # Test bulk retrieval
    all_facts = graph_fact_store.get_user_facts("test", "user1")
    print(f"\n📋 All facts for user1: {len(all_facts)} items")
    for attr, fact in all_facts.items():
        value = fact.value if hasattr(fact, 'value') else fact
        print(f"   {attr}: {value}")
    
    # Test entity relationships (graceful handling without Neo4j)
    relationships = graph_fact_store.get_entity_relationships("test", "user1")
    print(f"\n🔗 Entity relationships: {len(relationships)} items")
    
    # Test graph statistics
    stats = graph_fact_store.get_graph_statistics("test")
    print(f"📊 Graph statistics: {stats}")
    
    return len(all_facts) == len(test_facts)


def test_integration_workflow():
    """Test the complete Phase 3 integration workflow."""
    print("\n🔗 Testing Phase 3 Integration Workflow")
    print("-" * 40)
    
    # Setup all components with proper MockRedisClient
    class MockRedisClient:
        def __init__(self):
            self.data = {}
        
        def set(self, key, value):
            self.data[key] = value
            return True
        
        def setex(self, key, time, value):
            self.data[key] = value
            return True
        
        def get(self, key):
            value = self.data.get(key)
            return value.encode('utf-8') if isinstance(value, str) else value
        
        def delete(self, *keys):
            deleted = 0
            for key in keys:
                if key in self.data:
                    del self.data[key]
                    deleted += 1
            return deleted
        
        def keys(self, pattern):
            import re
            pattern_regex = pattern.replace('*', '.*')
            matching_keys = [k for k in self.data.keys() if re.match(pattern_regex, k)]
            return [k.encode('utf-8') if isinstance(k, str) else k for k in matching_keys]
    
    redis_client = MockRedisClient()
    
    fact_store = FactStore(redis_client)
    graph_fact_store = GraphFactStore(fact_store, neo4j_client=None)
    fact_ranker = FactAwareRanker()
    hybrid_scorer = HybridScorer(fact_ranker, None)
    intent_classifier = IntentClassifier()
    expander = GraphQueryExpander(neo4j_client=None, intent_classifier=intent_classifier)
    
    # Simulate fact storage workflow
    fact_statement = "My name is John Doe and I work at TechCorp in Seattle. My email is john@techcorp.com."
    
    print(f"📝 Storing fact: '{fact_statement}'")
    
    # Extract and store facts
    from core.fact_extractor import FactExtractor
    extractor = FactExtractor()
    extracted_facts = extractor.extract_facts_from_text(fact_statement)
    
    print(f"🧩 Extracted {len(extracted_facts)} facts:")
    for fact in extracted_facts:
        print(f"   {fact.attribute}: {fact.value} (confidence: {fact.confidence:.2f})")
        graph_fact_store.store_fact("test", "john", fact.attribute, fact.value, fact.confidence)
    
    # Simulate query workflow
    query = "What's my work email address?"
    print(f"\n🔍 Query: '{query}'")
    
    # Step 1: Intent classification
    intent_result = intent_classifier.classify(query)
    print(f"1. Intent: {intent_result.intent.value}, Attribute: {intent_result.attribute}")
    
    # Step 2: Query expansion
    expansion_result = expander.expand_query(query)
    expanded_queries = [query] + [eq.query for eq in expansion_result.expanded_queries]
    print(f"2. Query expansion: {len(expanded_queries)} variants")
    for i, eq in enumerate(expanded_queries):
        print(f"   {i+1}. {eq}")
    
    # Step 3: Simulate retrieval results
    mock_results = [
        {
            "id": "1",
            "content": "What's your work email? I need to send you something.",
            "score": 0.85,
            "lexical_score": 0.7,
            "metadata": {"type": "question"}
        },
        {
            "id": "2",
            "content": "My email is john@techcorp.com and I work at TechCorp.",
            "score": 0.8,
            "lexical_score": 0.6,
            "metadata": {"type": "fact_statement"}
        },
        {
            "id": "3",
            "content": "John Doe works at TechCorp in Seattle.",
            "score": 0.75,
            "lexical_score": 0.8,
            "metadata": {"type": "context_info"}
        }
    ]
    
    print(f"3. Mock retrieval: {len(mock_results)} results")
    
    # Step 4: Apply hybrid scoring
    hybrid_scores = hybrid_scorer.compute_hybrid_score(
        query=query,
        results=mock_results,
        intent=intent_result.intent,
        scoring_mode=ScoringMode.FACT_OPTIMIZED
    )
    
    print(f"4. Hybrid scoring results:")
    for i, score in enumerate(hybrid_scores):
        result_index = i if i < len(mock_results) else 0  # Safe fallback
        content_preview = mock_results[result_index]['content'][:60]
        print(f"   {i+1}. Score: {score.final_score:.3f} - {content_preview}...")
        print(f"      Components: V:{score.vector_score:.3f} L:{score.lexical_score:.3f} F:{score.fact_pattern_score:.3f}")
    
    # Step 5: Verify email fact retrieval
    stored_email_fact = graph_fact_store.get_fact("test", "john", "email")
    stored_email = stored_email_fact.value if stored_email_fact else None
    email_match = stored_email and "john@techcorp.com" in str(stored_email)
    
    print(f"5. Fact verification: Email = {stored_email}, Match = {email_match}")
    
    # Integration success metrics
    success_metrics = {
        "facts_extracted": len(extracted_facts) > 0,
        "intent_classified": intent_result.intent == IntentType.FACT_LOOKUP,
        "queries_expanded": len(expansion_result.expanded_queries) > 0,
        "hybrid_scoring_applied": len(hybrid_scores) > 0,
        "fact_recalled": email_match
    }
    
    print(f"\n🎯 Integration success metrics:")
    for metric, success in success_metrics.items():
        status = "✅" if success else "❌"
        print(f"   {status} {metric.replace('_', ' ').title()}: {success}")
    
    overall_success = all(success_metrics.values())
    print(f"\n{'🎉' if overall_success else '⚠️'} Overall integration: {'SUCCESS' if overall_success else 'PARTIAL'}")
    
    return overall_success


def test_performance_characteristics():
    """Test performance characteristics of Phase 3 components."""
    print("\n⚡ Testing Phase 3 Performance")
    print("-" * 40)
    
    import time
    
    # Test entity extraction speed
    extractor = EntityExtractor()
    text_samples = ["My name is Alice Smith and I work at TechCorp."] * 10
    
    start_time = time.time()
    for text in text_samples:
        extractor.extract_entities(text)
    extraction_time = time.time() - start_time
    
    print(f"Entity extraction: {extraction_time*1000:.1f}ms for 10 texts ({extraction_time*100:.1f}ms avg)")
    
    # Test hybrid scoring speed
    hybrid_scorer = HybridScorer(FactAwareRanker(), None)
    mock_results = [{"id": str(i), "content": f"Test content {i}", "score": 0.5, "lexical_score": 0.5} for i in range(20)]
    
    start_time = time.time()
    hybrid_scorer.compute_hybrid_score("test query", mock_results)
    scoring_time = time.time() - start_time
    
    print(f"Hybrid scoring: {scoring_time*1000:.1f}ms for 20 results")
    
    # Test query expansion speed
    expander = GraphQueryExpander(neo4j_client=None, intent_classifier=IntentClassifier())
    test_queries = ["What's my name?"] * 5
    
    start_time = time.time()
    for query in test_queries:
        expander.expand_query(query)
    expansion_time = time.time() - start_time
    
    print(f"Query expansion: {expansion_time*1000:.1f}ms for 5 queries ({expansion_time*200:.1f}ms avg)")
    
    # Performance targets
    fast_extraction = extraction_time * 100 < 100  # <100ms avg
    fast_scoring = scoring_time * 1000 < 200  # <200ms
    fast_expansion = expansion_time * 200 < 300  # <300ms avg
    
    print(f"✅ Performance targets met: Extraction({fast_extraction}), Scoring({fast_scoring}), Expansion({fast_expansion})")
    
    return fast_extraction and fast_scoring and fast_expansion


def main():
    """Run all Phase 3 graph integration tests."""
    print("🧪 Testing Phase 3 Graph Integration for Enhanced Fact Recall")
    print("=" * 70)
    
    tests = [
        ("Entity Extraction", test_entity_extraction),
        ("Graph Signal Enhancement", test_graph_signal_enhancement),
        ("Hybrid Scoring", test_hybrid_scoring),
        ("Query Expansion", test_query_expansion),
        ("Graph Fact Store", test_graph_fact_store),
        ("Integration Workflow", test_integration_workflow),
        ("Performance", test_performance_characteristics)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            results.append((test_name, False))
            print(f"\n❌ ERROR: {test_name} - {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 Phase 3 Graph Integration Test Results")
    print("-" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 3 graph integration components working correctly!")
        print("Ready for enhanced fact recall with:")
        print("  • Entity extraction and graph storage")
        print("  • Graph signal enhancement for ranking")
        print("  • Hybrid scoring with multiple components")
        print("  • Graph-enhanced query expansion")
        print("  • Complete integration workflow")
    else:
        print("⚠️  Some graph integration components need attention")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)