#!/usr/bin/env python3
"""
Test Phase 2 fact recall enhancements.

This script tests the Q&A generation, fact-aware ranking, and query rewriting
components to validate improved fact recall capabilities.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.qa_generator import QAPairGenerator
from src.storage.fact_ranker import FactAwareRanker
from src.core.query_rewriter import FactQueryRewriter
from src.core.intent_classifier import IntentClassifier, IntentType


def test_qa_generation():
    """Test Q&A pair generation from statements."""
    print("🔀 Testing Q&A Pair Generation")
    print("-" * 40)
    
    generator = QAPairGenerator()
    
    test_statements = [
        "My name is John Doe and I work as a software engineer.",
        "I live in San Francisco and my email is john@example.com.",
        "I like spicy food and I listen to jazz music.",
        "I am 28 years old and my birthday is March 15th."
    ]
    
    total_pairs = 0
    for statement in test_statements:
        print(f"\n📝 Statement: '{statement}'")
        
        qa_pairs = generator.generate_qa_pairs_from_statement(statement)
        total_pairs += len(qa_pairs)
        
        for pair in qa_pairs:
            print(f"   Q: {pair.question}")
            print(f"   A: {pair.answer}")
            print(f"   Attribute: {pair.fact_attribute} (confidence: {pair.confidence:.2f})")
            
            # Test stitched unit creation
            stitched = generator.create_stitched_unit(pair)
            print(f"   Stitched: {stitched.content[:80]}...")
            print()
    
    print(f"✅ Generated {total_pairs} Q&A pairs total")
    return total_pairs > 0


def test_fact_aware_ranking():
    """Test fact-aware ranking with answer priority boosts."""
    print("\n📊 Testing Fact-Aware Ranking")
    print("-" * 40)
    
    ranker = FactAwareRanker()
    
    # Simulate search results with mix of questions and answers
    test_results = [
        {
            "content": "What's my name? I can't remember.",
            "score": 0.9,
            "metadata": {"type": "question"}
        },
        {
            "content": "My name is Alice and I work at TechCorp.",
            "score": 0.8,
            "metadata": {"type": "statement"}
        },
        {
            "content": "Do you know what my email address is?",
            "score": 0.85,
            "metadata": {"type": "question"}
        },
        {
            "content": "My email is alice@techcorp.com and I prefer working remotely.",
            "score": 0.75,
            "metadata": {"type": "statement"}
        }
    ]
    
    print("📥 Original results:")
    for i, result in enumerate(test_results):
        print(f"   {i+1}. Score: {result['score']:.2f} - {result['content'][:60]}...")
    
    # Apply fact-aware ranking
    ranked_results = ranker.apply_fact_ranking(test_results, "What's my name?")
    
    print("\n📤 After fact-aware ranking:")
    for i, result in enumerate(ranked_results):
        boost_info = f" (boost: {result.fact_boost:+.2f})" if abs(result.fact_boost) > 0.01 else ""
        print(f"   {i+1}. Score: {result.final_score:.2f}{boost_info} - {result.content[:60]}...")
        print(f"      Type: {result.content_type.value}, Patterns: {result.matched_patterns}")
    
    # Check that declarative statements are boosted
    declarative_boosted = any(r.content_type.value == "declarative_answer" and r.fact_boost > 0 for r in ranked_results)
    interrogative_demoted = any(r.content_type.value == "interrogative_question" and r.fact_boost < 0 for r in ranked_results)
    
    print(f"✅ Declarative statements boosted: {declarative_boosted}")
    print(f"✅ Interrogative questions demoted: {interrogative_demoted}")
    
    return declarative_boosted and interrogative_demoted


def test_query_rewriting():
    """Test query rewriting for improved fact recall."""
    print("\n🔄 Testing Query Rewriting")
    print("-" * 40)
    
    rewriter = FactQueryRewriter()
    classifier = IntentClassifier()
    
    test_queries = [
        "What's my name?",
        "What's my email address?",
        "Where do I live?",
        "What food do I like?",
        "Do you know my phone number?"
    ]
    
    total_rewrites = 0
    for query in test_queries:
        print(f"\n🔍 Original query: '{query}'")
        
        # Classify intent first
        intent = classifier.classify(query)
        print(f"   Intent: {intent.intent.value}, Attribute: {intent.attribute}")
        
        # Generate rewrites
        rewrites = rewriter.rewrite_fact_query(query, intent.attribute)
        total_rewrites += len(rewrites)
        
        for rewrite in rewrites:
            print(f"   ↳ {rewrite.query} (method: {rewrite.method.value}, conf: {rewrite.confidence:.2f})")
    
    print(f"\n✅ Generated {total_rewrites} query rewrites total")
    return total_rewrites > 0


def test_integration_workflow():
    """Test the complete enhanced retrieval workflow."""
    print("\n🔗 Testing Integration Workflow")
    print("-" * 40)
    
    # Simulate the enhanced retrieval process
    query = "What's my email?"
    
    print(f"🔍 User query: '{query}'")
    
    # Step 1: Intent classification
    classifier = IntentClassifier()
    intent = classifier.classify(query)
    print(f"1. Intent: {intent.intent.value}, Attribute: {intent.attribute}")
    
    # Step 2: Query rewriting
    rewriter = FactQueryRewriter()
    rewrites = rewriter.rewrite_fact_query(query, intent.attribute)
    enhanced_queries = [query] + [r.query for r in rewrites]
    print(f"2. Enhanced queries: {len(enhanced_queries)} variants")
    for eq in enhanced_queries:
        print(f"   - {eq}")
    
    # Step 3: Simulate storage with Q&A generation
    fact_statement = "My email is john@example.com and I work remotely."
    generator = QAPairGenerator()
    qa_pairs = generator.generate_qa_pairs_from_statement(fact_statement)
    
    print(f"3. Generated {len(qa_pairs)} Q&A pairs from storage:")
    for pair in qa_pairs:
        if pair.fact_attribute == "email":
            print(f"   Q: {pair.question} -> A: {pair.answer}")
    
    # Step 4: Simulate retrieval results
    mock_results = [
        {
            "content": "What's my email address? I forgot it.",
            "score": 0.85,
            "metadata": {"type": "question"}
        },
        {
            "content": "My email is john@example.com and I work remotely.",
            "score": 0.8,
            "metadata": {"type": "qa_pair", "fact_attribute": "email"}
        },
        {
            "content": "Q: What is my email? A: john@example.com | Related: What's my email address? | Attribute: email",
            "score": 0.9,
            "metadata": {"type": "qa_pair", "fact_attribute": "email", "content_type": "qa_pair"}
        }
    ]
    
    # Step 5: Apply fact-aware ranking
    ranker = FactAwareRanker()
    ranked = ranker.apply_fact_ranking(mock_results, query)
    
    print("4. Final ranked results:")
    for i, result in enumerate(ranked):
        print(f"   {i+1}. Score: {result.final_score:.2f} - {result.content[:60]}...")
        print(f"      Type: {result.content_type.value}")
    
    # Check if email answer is at the top
    top_result = ranked[0] if ranked else None
    email_at_top = top_result and "john@example.com" in top_result.content
    
    print(f"✅ Email answer ranked at top: {email_at_top}")
    
    return email_at_top


def test_performance_characteristics():
    """Test performance characteristics of enhanced system."""
    print("\n⚡ Testing Performance Characteristics")
    print("-" * 40)
    
    import time
    
    # Test intent classification speed
    classifier = IntentClassifier()
    queries = ["What's my name?"] * 100
    
    start_time = time.time()
    for query in queries:
        classifier.classify(query)
    intent_time = time.time() - start_time
    
    print(f"Intent classification: {intent_time*1000:.1f}ms for 100 queries ({intent_time*10:.2f}ms avg)")
    
    # Test Q&A generation speed
    generator = QAPairGenerator()
    statements = ["My name is John Doe and I work as a software engineer."] * 10
    
    start_time = time.time()
    for statement in statements:
        generator.generate_qa_pairs_from_statement(statement)
    qa_time = time.time() - start_time
    
    print(f"Q&A generation: {qa_time*1000:.1f}ms for 10 statements ({qa_time*100:.1f}ms avg)")
    
    # Test ranking speed
    ranker = FactAwareRanker()
    mock_results = [{"content": f"Test content {i}", "score": 0.5} for i in range(50)]
    
    start_time = time.time()
    ranker.apply_fact_ranking(mock_results, "test query")
    ranking_time = time.time() - start_time
    
    print(f"Fact ranking: {ranking_time*1000:.1f}ms for 50 results")
    
    # Performance targets
    fast_intent = intent_time * 10 < 5  # <5ms avg
    fast_qa = qa_time * 100 < 50  # <50ms avg  
    fast_ranking = ranking_time * 1000 < 100  # <100ms
    
    print(f"✅ Performance targets met: Intent({fast_intent}), Q&A({fast_qa}), Ranking({fast_ranking})")
    
    return fast_intent and fast_qa and fast_ranking


def main():
    """Run all Phase 2 enhancement tests."""
    print("🧪 Testing Phase 2 Fact Recall Enhancements")
    print("=" * 60)
    
    tests = [
        ("Q&A Generation", test_qa_generation),
        ("Fact-Aware Ranking", test_fact_aware_ranking),
        ("Query Rewriting", test_query_rewriting),
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
    print("\n" + "=" * 60)
    print("📋 Phase 2 Enhancement Test Results")
    print("-" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 2 enhancements working correctly!")
        print("Ready for enhanced fact recall with:")
        print("  • Q&A pair generation and stitched indexing")
        print("  • Answer-priority ranking with pattern boosts")
        print("  • Query rewriting for improved recall")
        print("  • Integrated multi-query enhanced retrieval")
    else:
        print("⚠️  Some enhancements need attention before production")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)