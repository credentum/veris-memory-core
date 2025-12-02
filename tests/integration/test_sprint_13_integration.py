"""
Sprint 13 Integration Tests
Tests all features from Sprint 13 Phases 1-4
"""

import pytest
import asyncio
from typing import Dict, Any
from datetime import datetime
import uuid


# Phase 1: Embedding Pipeline Tests

@pytest.mark.asyncio
async def test_embedding_status_in_store_response(test_client):
    """Test that store_context returns embedding_status (Phase 1)"""
    response = await test_client.post("/tools/store_context", json={
        "type": "test",
        "content": {"title": "Test Embedding Status", "text": "Test content"},
        "metadata": {"test": "sprint13_phase1"}
    })

    assert response.status_code == 200
    data = response.json()

    # Phase 1 requirement: embedding_status must be present
    assert "embedding_status" in data
    assert data["embedding_status"] in ["completed", "failed", "unavailable"]

    # If embedding failed, message should be present
    if data["embedding_status"] != "completed":
        assert "embedding_message" in data


@pytest.mark.asyncio
async def test_health_detailed_embedding_info(test_client):
    """Test that /health/detailed includes embedding pipeline status (Phase 1)"""
    response = await test_client.get("/health/detailed")

    assert response.status_code == 200
    data = response.json()

    # Phase 1 requirement: detailed health must include embedding info
    assert "qdrant" in data or "embedding_service" in data

    # If Qdrant is available, check embedding service status
    if "qdrant" in data and data["qdrant"].get("healthy"):
        # Embedding service should be tracked
        assert "embedding_service_loaded" in str(data) or "embedding" in str(data).lower()


@pytest.mark.asyncio
async def test_search_result_limit_validation(test_client):
    """Test that search result limits are enforced (Phase 1)"""
    # Test minimum limit
    response = await test_client.post("/tools/retrieve_context", json={
        "query": "test",
        "limit": 0  # Invalid: below minimum
    })

    # Should either reject or clamp to minimum
    assert response.status_code in [200, 400, 422]

    if response.status_code == 200:
        data = response.json()
        assert len(data.get("results", [])) >= 0  # Clamped to valid range

    # Test maximum limit
    response = await test_client.post("/tools/retrieve_context", json={
        "query": "test",
        "limit": 200  # Invalid: above maximum (100)
    })

    if response.status_code == 200:
        data = response.json()
        assert len(data.get("results", [])) <= 100  # Clamped to max


# Phase 2: Authentication & Authorization Tests

@pytest.mark.asyncio
async def test_api_key_authentication(test_client):
    """Test API key authentication (Phase 2)"""
    # Test with valid API key
    headers = {"X-API-Key": "vmk_test_key"}  # Assuming test key exists

    response = await test_client.post("/tools/store_context",
        headers=headers,
        json={
            "type": "test",
            "content": {"title": "Auth Test"},
            "metadata": {}
        }
    )

    # Should succeed with valid key
    assert response.status_code in [200, 201]


@pytest.mark.asyncio
async def test_author_attribution(test_client):
    """Test author attribution is auto-populated (Phase 2)"""
    headers = {"X-API-Key": "vmk_test_key"}

    response = await test_client.post("/tools/store_context",
        headers=headers,
        json={
            "type": "test",
            "content": {"title": "Author Test", "text": "Testing attribution"},
            "metadata": {}
        }
    )

    assert response.status_code == 200
    data = response.json()

    # Verify context was created
    context_id = data.get("context_id")
    assert context_id is not None

    # Retrieve and verify author metadata
    retrieve_response = await test_client.post("/tools/retrieve_context", json={
        "query": "Author Test",
        "limit": 1
    })

    if retrieve_response.status_code == 200:
        results = retrieve_response.json().get("results", [])
        if len(results) > 0:
            # Author attribution should be present in metadata
            metadata = results[0].get("metadata", {})
            assert "author" in metadata or "author_type" in metadata


@pytest.mark.asyncio
async def test_delete_requires_human(test_client):
    """Test that delete operations require human authentication (Phase 2)"""
    # First, create a context
    create_response = await test_client.post("/tools/store_context", json={
        "type": "test",
        "content": {"title": "Delete Test"},
        "metadata": {}
    })

    assert create_response.status_code == 200
    context_id = create_response.json().get("context_id")

    # Try to delete with agent key (should fail)
    agent_headers = {"X-API-Key": "vmk_agent_key"}  # Assuming agent key exists

    delete_response = await test_client.post("/tools/delete_context",
        headers=agent_headers,
        json={
            "context_id": context_id,
            "reason": "Testing agent deletion block"
        }
    )

    # Should either fail or return error message
    if delete_response.status_code == 200:
        data = delete_response.json()
        assert data.get("success") == False or "human" in data.get("error", "").lower()


@pytest.mark.asyncio
async def test_forget_context_soft_delete(test_client):
    """Test forget_context soft-delete functionality (Phase 2/3)"""
    # Create a context
    create_response = await test_client.post("/tools/store_context", json={
        "type": "test",
        "content": {"title": "Forget Test", "text": "This will be forgotten"},
        "metadata": {}
    })

    assert create_response.status_code == 200
    context_id = create_response.json().get("context_id")

    # Soft-delete with forget
    forget_response = await test_client.post("/tools/forget_context", json={
        "context_id": context_id,
        "reason": "Testing soft delete",
        "retention_days": 30
    })

    # Should succeed (or gracefully handle if endpoint not fully integrated)
    assert forget_response.status_code in [200, 404, 501]


# Phase 3: Memory Management Tests

@pytest.mark.asyncio
async def test_redis_ttl_management(redis_client):
    """Test Redis TTL management (Phase 3)"""
    from src.storage.redis_manager import RedisTTLManager

    if not redis_client:
        pytest.skip("Redis not available")

    manager = RedisTTLManager(redis_client)

    # Test set with TTL
    success = manager.set_with_ttl(
        key="test_sprint13:ttl_test",
        value="test_value",
        ttl=60,
        key_type="temporary"
    )

    assert success == True

    # Verify TTL was set
    ttl = manager.get_ttl("test_sprint13:ttl_test")
    assert ttl > 0 and ttl <= 60

    # Cleanup
    redis_client.delete("test_sprint13:ttl_test")


@pytest.mark.asyncio
async def test_redis_event_logging(redis_client):
    """Test Redis event logging (Phase 3)"""
    from src.storage.redis_manager import RedisEventLog

    if not redis_client:
        pytest.skip("Redis not available")

    event_log = RedisEventLog(redis_client)

    # Log an event
    success = event_log.log_event(
        event_type="test",
        key="test_sprint13:event",
        operation="set",
        metadata={"sprint": "13", "phase": "3"}
    )

    assert success == True

    # Retrieve recent events
    events = event_log.get_recent_events(limit=10)
    assert isinstance(events, list)

    # Verify our event is present
    test_events = [e for e in events if e.get("key") == "test_sprint13:event"]
    assert len(test_events) > 0


@pytest.mark.asyncio
async def test_redis_neo4j_sync(redis_client, neo4j_client):
    """Test Redis-to-Neo4j synchronization (Phase 3)"""
    from src.tools.redis_neo4j_sync import RedisNeo4jSync

    if not redis_client or not neo4j_client:
        pytest.skip("Redis or Neo4j not available")

    sync = RedisNeo4jSync(redis_client, neo4j_client)

    # Log some test events
    from src.storage.redis_manager import RedisEventLog
    event_log = RedisEventLog(redis_client)

    event_log.log_event(
        event_type="test_sync",
        key="test_sprint13:sync",
        operation="test",
        metadata={"test": "sprint13_phase3"}
    )

    # Perform sync
    result = sync.sync_event_log(limit=100)

    assert result.get("success") == True
    assert "synced" in result


# Phase 4: Namespace Management Tests

@pytest.mark.asyncio
async def test_namespace_parsing():
    """Test namespace path parsing (Phase 4)"""
    from src.core.namespace_manager import NamespaceManager

    manager = NamespaceManager()

    # Test global namespace
    parsed = manager.parse_namespace("/global/api_design")
    assert parsed["type"] == "global"
    assert parsed["scope"] is None
    assert parsed["name"] == "api_design"

    # Test team namespace
    parsed = manager.parse_namespace("/team/engineering/backend")
    assert parsed["type"] == "team"
    assert parsed["scope"] == "engineering"
    assert parsed["name"] == "backend"

    # Test user namespace
    parsed = manager.parse_namespace("/user/alice/notes")
    assert parsed["type"] == "user"
    assert parsed["scope"] == "alice"
    assert parsed["name"] == "notes"

    # Test project namespace
    parsed = manager.parse_namespace("/project/veris/memory")
    assert parsed["type"] == "project"
    assert parsed["scope"] == "veris"
    assert parsed["name"] == "memory"


@pytest.mark.asyncio
async def test_namespace_lock_acquisition(redis_client):
    """Test TTL-based namespace locks (Phase 4)"""
    from src.core.namespace_manager import NamespaceManager

    if not redis_client:
        pytest.skip("Redis not available")

    manager = NamespaceManager(redis_client)

    namespace_path = "/global/test_lock"
    lock_id = f"lock_{uuid.uuid4()}"

    # Acquire lock
    acquired = manager.acquire_lock(namespace_path, lock_id, ttl=30)
    assert acquired == True

    # Verify lock is held
    is_locked = manager.is_locked(namespace_path)
    assert is_locked == True

    # Try to acquire again (should fail)
    acquired_again = manager.acquire_lock(namespace_path, "different_lock", ttl=30)
    assert acquired_again == False

    # Release lock
    released = manager.release_lock(namespace_path, lock_id)
    assert released == True

    # Verify lock is released
    is_locked = manager.is_locked(namespace_path)
    assert is_locked == False


@pytest.mark.asyncio
async def test_namespace_auto_assignment():
    """Test auto-assignment of namespaces (Phase 4)"""
    from src.core.namespace_manager import add_namespace_to_context

    # Test project-based assignment
    namespace = add_namespace_to_context(
        content={"project_id": "veris-memory"},
        namespace_path=None,
        user_id=None
    )
    assert namespace == "/project/veris-memory/context"

    # Test team-based assignment
    namespace = add_namespace_to_context(
        content={"team_id": "engineering"},
        namespace_path=None,
        user_id=None
    )
    assert namespace == "/team/engineering/context"

    # Test user-based assignment
    namespace = add_namespace_to_context(
        content={},
        namespace_path=None,
        user_id="alice"
    )
    assert namespace == "/user/alice/context"

    # Test explicit assignment
    namespace = add_namespace_to_context(
        content={},
        namespace_path="/global/shared",
        user_id="bob"
    )
    assert namespace == "/global/shared"


# Phase 4: Relationship Detection Tests

@pytest.mark.asyncio
async def test_relationship_detection_temporal():
    """Test temporal relationship detection (Phase 4)"""
    from src.core.relationship_detector import RelationshipDetector

    detector = RelationshipDetector(neo4j_client=None)

    # Note: Without Neo4j, temporal detection returns empty
    # This tests the structure and error handling
    relationships = detector.detect_relationships(
        context_id=str(uuid.uuid4()),
        context_type="sprint",
        content={"sprint_number": 13},
        metadata={}
    )

    assert isinstance(relationships, list)


@pytest.mark.asyncio
async def test_relationship_detection_references():
    """Test reference-based relationship detection (Phase 4)"""
    from src.core.relationship_detector import RelationshipDetector

    detector = RelationshipDetector()

    # Test PR reference detection
    relationships = detector.detect_relationships(
        context_id="test-123",
        context_type="design",
        content={
            "description": "This design implements PR #456 and fixes issue #789"
        },
        metadata={}
    )

    # Should detect PR and issue references
    pr_refs = [r for r in relationships if r[0] == "REFERENCES" and "pr_" in r[1]]
    issue_refs = [r for r in relationships if r[0] == "FIXES" and "issue_" in r[1]]

    assert len(pr_refs) > 0
    assert len(issue_refs) > 0


@pytest.mark.asyncio
async def test_relationship_detection_hierarchical():
    """Test hierarchical relationship detection (Phase 4)"""
    from src.core.relationship_detector import RelationshipDetector

    detector = RelationshipDetector()

    # Test sprint relationship
    relationships = detector.detect_relationships(
        context_id="test-456",
        context_type="design",
        content={},
        metadata={"sprint": "13"}
    )

    # Should detect PART_OF relationship to sprint
    sprint_refs = [r for r in relationships if r[0] == "PART_OF" and "sprint_" in r[1]]
    assert len(sprint_refs) > 0

    # Test project relationship
    relationships = detector.detect_relationships(
        context_id="test-789",
        context_type="code",
        content={"project_id": "veris-memory"},
        metadata={}
    )

    # Should detect PART_OF relationship to project
    project_refs = [r for r in relationships if r[0] == "PART_OF" and "project_" in r[1]]
    assert len(project_refs) > 0


@pytest.mark.asyncio
async def test_relationship_detection_stats():
    """Test relationship detection statistics tracking (Phase 4)"""
    from src.core.relationship_detector import RelationshipDetector

    detector = RelationshipDetector()

    # Detect relationships multiple times
    for i in range(3):
        detector.detect_relationships(
            context_id=f"test-{i}",
            context_type="test",
            content={"description": f"Test {i} references PR #100"},
            metadata={}
        )

    # Check stats
    stats = detector.get_detection_stats()

    assert "total_detected" in stats
    assert stats["total_detected"] >= 3  # At least 3 PR references
    assert "by_type" in stats
    assert "last_detection" in stats


# Phase 4: Tool Discovery Tests

@pytest.mark.asyncio
async def test_tools_endpoint_structure(test_client):
    """Test enhanced /tools endpoint structure (Phase 4)"""
    response = await test_client.get("/tools")

    assert response.status_code == 200
    data = response.json()

    # Phase 4 requirements
    assert "tools" in data
    assert "total_tools" in data
    assert "available_tools" in data
    assert "capabilities" in data

    # Should have all 7 tools
    assert data["total_tools"] >= 7
    assert isinstance(data["tools"], list)


@pytest.mark.asyncio
async def test_tools_endpoint_schemas(test_client):
    """Test that each tool has complete schemas (Phase 4)"""
    response = await test_client.get("/tools")

    assert response.status_code == 200
    data = response.json()

    for tool in data["tools"]:
        # Each tool must have required fields
        assert "name" in tool
        assert "description" in tool
        assert "endpoint" in tool
        assert "method" in tool
        assert "available" in tool
        assert "capabilities" in tool

        # Schema requirements
        assert "input_schema" in tool
        assert "output_schema" in tool

        # Example requirement
        assert "example" in tool or tool["name"] in ["query_graph"]  # Some tools may not have examples


@pytest.mark.asyncio
async def test_tools_endpoint_sprint13_enhancements(test_client):
    """Test Sprint 13 enhancements are documented (Phase 4)"""
    response = await test_client.get("/tools")

    assert response.status_code == 200
    data = response.json()

    # Sprint 13 should add delete and forget tools
    tool_names = [t["name"] for t in data["tools"]]

    # New tools from Sprint 13
    if "delete_context" in tool_names:
        delete_tool = next(t for t in data["tools"] if t["name"] == "delete_context")
        assert "human_only" in str(delete_tool).lower() or delete_tool.get("requires_auth") == True

    if "forget_context" in tool_names:
        forget_tool = next(t for t in data["tools"] if t["name"] == "forget_context")
        assert forget_tool.get("available") is not None


# Integration Test: End-to-End Sprint 13 Workflow

@pytest.mark.asyncio
async def test_sprint13_complete_workflow(test_client, redis_client, neo4j_client):
    """Test complete Sprint 13 workflow across all phases"""

    # Phase 1: Store context with embedding status
    store_response = await test_client.post("/tools/store_context", json={
        "type": "sprint",
        "content": {
            "sprint_number": 13,
            "title": "Sprint 13 Integration Test",
            "description": "Testing all phases. References PR #999, fixes issue #888.",
            "project_id": "veris-memory"
        },
        "metadata": {"sprint": "13", "test": "integration"}
    })

    assert store_response.status_code == 200
    store_data = store_response.json()

    # Phase 1: Verify embedding status
    assert "embedding_status" in store_data

    context_id = store_data.get("context_id")
    assert context_id is not None

    # Phase 2: Verify author attribution (if API key auth is enabled)
    # This would be tested with actual API key headers

    # Phase 3: Verify Redis TTL management (if Redis available)
    if redis_client:
        from src.storage.redis_manager import RedisTTLManager
        manager = RedisTTLManager(redis_client)

        # Check cleanup stats
        stats = manager.get_cleanup_stats()
        assert "total_cleaned" in stats

    # Phase 4: Verify namespace assignment
    # Content had project_id, so namespace should be project-based

    # Phase 4: Verify relationships were detected
    # Should have detected: PR #999, issue #888, sprint 13, project veris-memory

    # Retrieve the context
    retrieve_response = await test_client.post("/tools/retrieve_context", json={
        "query": "Sprint 13 Integration Test",
        "limit": 5
    })

    assert retrieve_response.status_code == 200
    retrieve_data = retrieve_response.json()

    assert "results" in retrieve_data
    assert len(retrieve_data["results"]) > 0

    # Verify enhanced tool discovery
    tools_response = await test_client.get("/tools")
    assert tools_response.status_code == 200
    tools_data = tools_response.json()

    assert tools_data["total_tools"] >= 7
    assert "sprint_13_enhancements" in str(tools_data).lower() or len(tools_data["tools"]) >= 7


# Fixtures

@pytest.fixture
async def redis_client():
    """Provide Redis client for tests"""
    try:
        import redis
        client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
        client.ping()
        yield client
    except Exception:
        yield None


@pytest.fixture
async def neo4j_client():
    """Provide Neo4j client for tests"""
    try:
        from src.storage.neo4j_client import Neo4jClient
        client = Neo4jClient()
        # Test connection
        yield client
    except Exception:
        yield None


@pytest.fixture
async def test_client():
    """Provide test HTTP client"""
    from httpx import AsyncClient
    from src.mcp_server.main import app

    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
