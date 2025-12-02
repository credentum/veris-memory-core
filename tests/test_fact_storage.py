"""
Unit tests for fact storage system.

Tests the fact store, intent classifier, fact extractor, and scope validator
to ensure reliable fact storage and retrieval functionality.
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import Dict, Any

# Import the modules we're testing
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.storage.fact_store import FactStore, Fact
from src.core.intent_classifier import IntentClassifier, IntentType, IntentResult
from src.core.fact_extractor import FactExtractor, ExtractedFact
from middleware.scope_validator import ScopeValidator, ScopeContext, ScopeValidationError


class TestFactStore:
    """Test the FactStore class."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client for testing."""
        redis = Mock()
        redis.get.return_value = None
        redis.setex.return_value = True
        redis.delete.return_value = 1
        redis.keys.return_value = []
        redis.lpush.return_value = 1
        redis.expire.return_value = True
        redis.lrange.return_value = []
        return redis

    @pytest.fixture
    def fact_store(self, mock_redis):
        """Create FactStore instance with mock Redis."""
        return FactStore(mock_redis)

    def test_store_fact_basic(self, fact_store, mock_redis):
        """Test basic fact storage."""
        fact_store.store_fact(
            namespace="test_agent",
            user_id="user123",
            attribute="name",
            value="John Doe",
            confidence=0.95,
            source_turn_id="turn_001"
        )

        # Verify Redis operations
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args[0]
        
        # Check key format
        assert call_args[0] == "facts:test_agent:user123:name"
        
        # Check stored data
        stored_data = json.loads(call_args[2])
        assert stored_data["value"] == "John Doe"
        assert stored_data["confidence"] == 0.95
        assert stored_data["attribute"] == "name"

    def test_store_fact_validation(self, fact_store):
        """Test fact storage validation."""
        with pytest.raises(ValueError):
            fact_store.store_fact("", "user123", "name", "John")
        
        with pytest.raises(ValueError):
            fact_store.store_fact("agent", "", "name", "John")
        
        with pytest.raises(ValueError):
            fact_store.store_fact("agent", "user", "", "John")

    def test_get_fact_success(self, fact_store, mock_redis):
        """Test successful fact retrieval."""
        # Mock Redis to return a fact
        fact_data = {
            "value": "John Doe",
            "confidence": 0.95,
            "source_turn_id": "turn_001",
            "updated_at": datetime.utcnow().isoformat(),
            "provenance": "user_input",
            "attribute": "name",
            "user_id": "user123",
            "namespace": "test_agent"
        }
        mock_redis.get.return_value = json.dumps(fact_data).encode('utf-8')

        fact = fact_store.get_fact("test_agent", "user123", "name")
        
        assert fact is not None
        assert fact.value == "John Doe"
        assert fact.confidence == 0.95
        assert fact.attribute == "name"

    def test_get_fact_not_found(self, fact_store, mock_redis):
        """Test fact retrieval when fact doesn't exist."""
        mock_redis.get.return_value = None
        
        fact = fact_store.get_fact("test_agent", "user123", "nonexistent")
        assert fact is None

    def test_get_user_facts(self, fact_store, mock_redis):
        """Test retrieving all facts for a user."""
        # Mock multiple facts
        facts_data = {
            "facts:test_agent:user123:name": {
                "value": "John Doe",
                "confidence": 0.95,
                "attribute": "name",
                "user_id": "user123",
                "namespace": "test_agent",
                "source_turn_id": "turn_001",
                "updated_at": datetime.utcnow().isoformat(),
                "provenance": "user_input"
            },
            "facts:test_agent:user123:email": {
                "value": "john@example.com",
                "confidence": 0.98,
                "attribute": "email",
                "user_id": "user123",
                "namespace": "test_agent",
                "source_turn_id": "turn_002",
                "updated_at": datetime.utcnow().isoformat(),
                "provenance": "user_input"
            }
        }

        # Mock Redis keys and get operations
        mock_redis.keys.return_value = [k.encode('utf-8') for k in facts_data.keys()]
        mock_redis.get.side_effect = lambda key: json.dumps(facts_data[key.decode('utf-8')]).encode('utf-8')

        user_facts = fact_store.get_user_facts("test_agent", "user123")
        
        assert len(user_facts) == 2
        assert "name" in user_facts
        assert "email" in user_facts
        assert user_facts["name"].value == "John Doe"
        assert user_facts["email"].value == "john@example.com"

    def test_delete_user_facts(self, fact_store, mock_redis):
        """Test deleting all facts for a user."""
        mock_redis.keys.return_value = [b"facts:test_agent:user123:name", b"facts:test_agent:user123:email"]
        mock_redis.delete.return_value = 4  # 2 fact keys + 2 history keys

        deleted_count = fact_store.delete_user_facts("test_agent", "user123")
        
        assert deleted_count == 4
        mock_redis.delete.assert_called_once()

    def test_fact_history_tracking(self, fact_store, mock_redis):
        """Test that fact updates are tracked in history."""
        # Mock existing fact
        existing_fact = json.dumps({"value": "Old Name"}).encode('utf-8')
        mock_redis.get.return_value = existing_fact

        # Store new fact
        fact_store.store_fact("test_agent", "user123", "name", "New Name")

        # Verify history was updated
        mock_redis.lpush.assert_called_once()
        history_call = mock_redis.lpush.call_args[0]
        assert "fact_history:test_agent:user123:name" in history_call[0]


class TestIntentClassifier:
    """Test the IntentClassifier class."""

    @pytest.fixture
    def classifier(self):
        """Create IntentClassifier instance."""
        return IntentClassifier()

    def test_fact_lookup_classification(self, classifier):
        """Test classification of fact lookup queries."""
        test_cases = [
            ("What's my name?", IntentType.FACT_LOOKUP, "name"),
            ("Do you know my email?", IntentType.FACT_LOOKUP, "email"),
            ("What food do I like?", IntentType.FACT_LOOKUP, "preferences.food"),
            ("Where do I live?", IntentType.FACT_LOOKUP, "location"),
            ("Who am I?", IntentType.FACT_LOOKUP, "name"),
        ]

        for query, expected_intent, expected_attr in test_cases:
            result = classifier.classify(query)
            assert result.intent == expected_intent, f"Failed for: {query}"
            assert result.attribute == expected_attr, f"Wrong attribute for: {query}"
            assert result.confidence > 0.7, f"Low confidence for: {query}"

    def test_fact_storage_classification(self, classifier):
        """Test classification of fact storage statements."""
        test_cases = [
            ("My name is John", IntentType.STORE_FACT, "name"),
            ("I'm called Mike", IntentType.STORE_FACT, "name"),
            ("My email is john@example.com", IntentType.STORE_FACT, "email"),
            ("I live in New York", IntentType.STORE_FACT, "location"),
            ("I like pizza", IntentType.STORE_FACT, "preferences.general"),
        ]

        for statement, expected_intent, expected_attr in test_cases:
            result = classifier.classify(statement)
            assert result.intent == expected_intent, f"Failed for: {statement}"
            assert result.attribute == expected_attr, f"Wrong attribute for: {statement}"
            assert result.confidence > 0.7, f"Low confidence for: {statement}"

    def test_fact_update_classification(self, classifier):
        """Test classification of fact update statements."""
        test_cases = [
            ("Actually, my name is Mike", IntentType.UPDATE_FACT, "name"),
            ("Correction, I live in Boston", IntentType.UPDATE_FACT, "general"),
            ("That's wrong, I prefer tea", IntentType.UPDATE_FACT, "general"),
        ]

        for statement, expected_intent, expected_attr in test_cases:
            result = classifier.classify(statement)
            assert result.intent == expected_intent, f"Failed for: {statement}"
            assert result.confidence > 0.7, f"Low confidence for: {statement}"

    def test_general_query_classification(self, classifier):
        """Test classification of general queries."""
        test_cases = [
            "How's the weather?",
            "What is Python?",
            "Tell me a joke",
            "How do I cook pasta?",
        ]

        for query in test_cases:
            result = classifier.classify(query)
            assert result.intent == IntentType.GENERAL_QUERY, f"Should be general: {query}"

    def test_extract_fact_value(self, classifier):
        """Test extraction of fact values from statements."""
        test_cases = [
            ("My name is John Doe", ("name", "John Doe")),
            ("My email is test@example.com", ("email", "test@example.com")),
            ("I live in San Francisco", ("location", "San Francisco")),
        ]

        for statement, expected in test_cases:
            result = classifier.extract_fact_value(statement)
            assert result == expected, f"Failed extraction for: {statement}"

    def test_is_fact_query(self, classifier):
        """Test boolean fact query detection."""
        fact_queries = [
            "What's my name?",
            "My name is John",
            "Actually, it's Mike"
        ]

        general_queries = [
            "How's the weather?",
            "What is AI?",
            "Tell me about Python"
        ]

        for query in fact_queries:
            assert classifier.is_fact_query(query), f"Should be fact query: {query}"

        for query in general_queries:
            assert not classifier.is_fact_query(query), f"Should not be fact query: {query}"


class TestFactExtractor:
    """Test the FactExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create FactExtractor instance."""
        return FactExtractor()

    def test_extract_name(self, extractor):
        """Test name extraction."""
        test_cases = [
            "My name is John Doe",
            "I'm called Michael",
            "You can call me Sarah",
            "People call me Alex"
        ]

        for text in test_cases:
            facts = extractor.extract_facts_from_text(text)
            name_facts = [f for f in facts if f.attribute == "name"]
            assert len(name_facts) > 0, f"No name extracted from: {text}"
            assert name_facts[0].confidence > 0.8, f"Low confidence for: {text}"

    def test_extract_email(self, extractor):
        """Test email extraction."""
        test_cases = [
            "My email is john@example.com",
            "Contact me at sarah.doe@company.org",
            "Reach me at test123@gmail.com"
        ]

        for text in test_cases:
            facts = extractor.extract_facts_from_text(text)
            email_facts = [f for f in facts if f.attribute == "email"]
            assert len(email_facts) > 0, f"No email extracted from: {text}"
            assert "@" in email_facts[0].value, f"Invalid email from: {text}"

    def test_extract_preferences(self, extractor):
        """Test preference extraction."""
        test_cases = [
            ("I like spicy food", "preferences.food"),
            ("I love rock music", "preferences.music"),
            ("My favorite food is pizza", "preferences.food"),
            ("I'm into jazz music", "preferences.music")
        ]

        for text, expected_attr in test_cases:
            facts = extractor.extract_facts_from_text(text)
            pref_facts = [f for f in facts if expected_attr in f.attribute]
            assert len(pref_facts) > 0, f"No preference extracted from: {text}"

    def test_extract_multiple_facts(self, extractor):
        """Test extraction of multiple facts from one text."""
        text = "My name is John Doe and my email is john@example.com. I live in New York and I like pizza."
        
        facts = extractor.extract_facts_from_text(text)
        
        # Should extract name, email, location, and food preference
        attributes = [f.attribute for f in facts]
        assert "name" in attributes
        assert "email" in attributes
        assert "location" in attributes
        assert any("food" in attr for attr in attributes)

    def test_confidence_adjustment(self, extractor):
        """Test that confidence is adjusted based on context."""
        certain_text = "My name is definitely John"
        uncertain_text = "My name might be John"
        
        certain_facts = extractor.extract_facts_from_text(certain_text)
        uncertain_facts = extractor.extract_facts_from_text(uncertain_text)
        
        if certain_facts and uncertain_facts:
            certain_name = [f for f in certain_facts if f.attribute == "name"][0]
            uncertain_name = [f for f in uncertain_facts if f.attribute == "name"][0]
            # Uncertain should have lower confidence (though this is a simple test)
            assert certain_name.confidence >= uncertain_name.confidence

    def test_validation_filters(self, extractor):
        """Test that validators filter out invalid extractions."""
        invalid_cases = [
            "My name is X",  # Too short
            "My email is notanemail",  # Invalid format
            "I'm 999 years old",  # Invalid age
        ]

        for text in invalid_cases:
            facts = extractor.extract_facts_from_text(text)
            # Should either extract nothing or extract with low confidence
            if facts:
                assert all(f.confidence < 0.9 for f in facts), f"Should reject: {text}"


class TestScopeValidator:
    """Test the ScopeValidator class."""

    @pytest.fixture
    def validator(self):
        """Create ScopeValidator instance."""
        return ScopeValidator()

    def test_extract_scope_context(self, validator):
        """Test scope context extraction from request data."""
        request_data = {
            "namespace": "test_agent",
            "user_id": "user123",
            "tenant_id": "tenant456"
        }

        scope = validator.extract_scope_context(request_data)
        
        assert scope.namespace == "test_agent"
        assert scope.user_id == "user123"
        assert scope.tenant_id == "tenant456"

    def test_validate_scope_success(self, validator):
        """Test successful scope validation."""
        scope = ScopeContext(
            namespace="test_agent",
            user_id="user123"
        )

        # Should not raise an exception
        validator.validate_scope("store_fact", scope)

    def test_validate_scope_missing_fields(self, validator):
        """Test scope validation with missing required fields."""
        scope = ScopeContext(namespace="test_agent")  # Missing user_id

        with pytest.raises(ScopeValidationError) as exc_info:
            validator.validate_scope("store_fact", scope)
        
        assert "missing_required_fields" in str(exc_info.value.error_type)

    def test_apply_scope_filter(self, validator):
        """Test applying scope filters to query parameters."""
        query_params = {"query": "test"}
        scope = ScopeContext(
            namespace="test_agent",
            user_id="user123",
            tenant_id="tenant456"
        )

        filtered = validator.apply_scope_filter(query_params, scope)
        
        assert filtered["namespace"] == "test_agent"
        assert filtered["user_id"] == "user123"
        assert filtered["tenant_id"] == "tenant456"
        assert filtered["query"] == "test"  # Original params preserved

    def test_check_cross_scope_access(self, validator):
        """Test cross-scope access validation."""
        scope1 = ScopeContext(namespace="agent1", user_id="user1")
        scope2 = ScopeContext(namespace="agent1", user_id="user1")  # Same
        scope3 = ScopeContext(namespace="agent1", user_id="user2")  # Different user
        scope4 = ScopeContext(namespace="agent2", user_id="user1")  # Different namespace

        assert validator.check_cross_scope_access(scope1, scope2) == True
        assert validator.check_cross_scope_access(scope1, scope3) == False
        assert validator.check_cross_scope_access(scope1, scope4) == False

    def test_validate_fact_access(self, validator):
        """Test fact-specific access validation."""
        scope = ScopeContext(namespace="test_agent", user_id="user123")

        # Same namespace and user - should allow
        assert validator.validate_fact_access(scope, "test_agent", "user123") == True
        
        # Different namespace - should deny
        assert validator.validate_fact_access(scope, "other_agent", "user123") == False
        
        # Different user - should deny
        assert validator.validate_fact_access(scope, "test_agent", "user456") == False

    def test_create_scoped_key(self, validator):
        """Test scoped key creation."""
        scope = ScopeContext(
            namespace="test_agent",
            user_id="user123",
            tenant_id="tenant456"
        )

        key = validator.create_scoped_key(scope, "store_fact", "name")
        
        assert "test_agent" in key
        assert "user123" in key
        assert "tenant456" in key
        assert "store_fact" in key
        assert "name" in key

    def test_require_scope_decorator(self):
        """Test the require_scope decorator."""
        from middleware.scope_validator import require_scope

        @require_scope("namespace", "user_id")
        def test_function(data, **kwargs):
            return kwargs.get('validated_scope')

        # Valid scope
        valid_data = {"namespace": "test", "user_id": "user123"}
        result = test_function(valid_data)
        assert result.namespace == "test"
        assert result.user_id == "user123"

        # Invalid scope - missing user_id
        invalid_data = {"namespace": "test"}
        with pytest.raises(ScopeValidationError):
            test_function(invalid_data)


class TestIntegration:
    """Integration tests for the complete fact system."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis for integration tests."""
        redis = Mock()
        redis.get.return_value = None
        redis.setex.return_value = True
        redis.delete.return_value = 1
        redis.keys.return_value = []
        return redis

    @pytest.fixture
    def fact_system(self, mock_redis):
        """Create complete fact system for integration testing."""
        return {
            'store': FactStore(mock_redis),
            'classifier': IntentClassifier(),
            'extractor': FactExtractor(),
            'validator': ScopeValidator()
        }

    def test_complete_fact_storage_flow(self, fact_system, mock_redis):
        """Test complete flow from extraction to storage."""
        text = "My name is John Doe and my email is john@example.com"
        namespace = "test_agent"
        user_id = "user123"

        # Extract facts
        facts = fact_system['extractor'].extract_facts_from_text(text)
        assert len(facts) >= 2  # Name and email

        # Validate scope
        scope_data = {"namespace": namespace, "user_id": user_id}
        scope = fact_system['validator'].extract_scope_context(scope_data)
        fact_system['validator'].validate_scope("store_fact", scope)

        # Store facts
        for fact in facts:
            fact_system['store'].store_fact(
                namespace=namespace,
                user_id=user_id,
                attribute=fact.attribute,
                value=fact.value,
                confidence=fact.confidence
            )

        # Verify storage calls
        assert mock_redis.setex.call_count >= 2

    def test_fact_query_intent_flow(self, fact_system):
        """Test fact query classification and attribute extraction."""
        queries = [
            "What's my name?",
            "Do you know my email?",
            "What food do I like?"
        ]

        for query in queries:
            # Classify intent
            result = fact_system['classifier'].classify(query)
            
            if result.intent == IntentType.FACT_LOOKUP:
                # Should be able to extract attribute
                attribute = fact_system['classifier'].extract_fact_attribute(query)
                assert attribute is not None, f"No attribute for: {query}"
                assert result.confidence > 0.7, f"Low confidence for: {query}"

    def test_scope_isolation(self, fact_system):
        """Test that scope isolation prevents cross-user access."""
        validator = fact_system['validator']

        # Different users should not access each other's facts
        user1_scope = ScopeContext(namespace="agent", user_id="user1")
        user2_scope = ScopeContext(namespace="agent", user_id="user2")

        assert not validator.check_cross_scope_access(user1_scope, user2_scope)
        assert not validator.validate_fact_access(user1_scope, "agent", "user2")

    def test_error_handling(self, fact_system):
        """Test error handling across the system."""
        # Invalid scope should raise validation error
        with pytest.raises(ScopeValidationError):
            scope = ScopeContext(namespace="", user_id="")
            fact_system['validator'].validate_scope("store_fact", scope)

        # Invalid fact store parameters should raise ValueError
        with pytest.raises(ValueError):
            fact_system['store'].store_fact("", "", "", "")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])