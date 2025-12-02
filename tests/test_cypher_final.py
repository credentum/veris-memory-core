#!/usr/bin/env python3
"""
Final Cypher Comment Normalization Test
Sprint 10 Phase 3 - Verify comment normalization working correctly
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from src.security.cypher_validator import CypherValidator

def test_cypher_comment_normalization():
    """Test cypher comment normalization functionality"""
    
    validator = CypherValidator()
    
    # Test cases for comment normalization
    test_cases = [
        {
            "name": "Single-line comment",
            "cypher": "MATCH (n) // This is a comment\nRETURN n",
            "should_pass": True
        },
        {
            "name": "Multi-line comment", 
            "cypher": "MATCH (n) /* This is a\nmulti-line comment */\nRETURN n",
            "should_pass": True
        },
        {
            "name": "Nested comment blocks (suspicious pattern)",
            "cypher": "MATCH (n) /* comment */ /* another */ RETURN n",
            "should_pass": False  # Multiple adjacent comment blocks are flagged as suspicious
        },
        {
            "name": "Injection attempt with comment",
            "cypher": "MATCH (n) /*; DROP TABLE users; */ RETURN n",
            "should_pass": False  # Should be blocked due to injection pattern
        }
    ]
    
    print("=== Cypher Comment Normalization Test ===")
    
    passed_count = 0
    total_count = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case['name']}")
        print(f"   Query: {test_case['cypher'][:60]}{'...' if len(test_case['cypher']) > 60 else ''}")
        
        result = validator.validate_query(test_case['cypher'])
        
        if result.is_valid == test_case['should_pass']:
            print(f"   ✅ CORRECT: {'Allowed' if result.is_valid else 'Blocked'}")
            passed_count += 1
        else:
            print(f"   ❌ INCORRECT: Expected {'allow' if test_case['should_pass'] else 'block'}, got {'allow' if result.is_valid else 'block'}")
            if not result.is_valid:
                print(f"      Error: {result.error_message}")
    
    success_rate = (passed_count / total_count) * 100
    print(f"\n=== CYPHER VALIDATION SUMMARY ===")
    print(f"Correct results: {passed_count}/{total_count}")
    print(f"Success rate: {success_rate:.1f}%")
    
    if success_rate == 100:
        print("🎉 PERFECT: Comment normalization working correctly!")
        return True
    else:
        print("⚠️  ISSUES: Comment normalization needs adjustment")
        return False

if __name__ == "__main__":
    success = test_cypher_comment_normalization()
    exit(0 if success else 1)