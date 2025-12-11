"""
Unit tests for agent_tools MCP endpoints.

Tests:
- log_trajectory endpoint
- check_precedent endpoint
- discover_skills endpoint
- API key authentication
- Error handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from src.mcp_server.agent_tools import (
    router,
    register_routes,
    ensure_collections,
    get_qdrant,
    get_embedding,
    log_trajectory,
    check_precedent,
    discover_skills,
    store_skill,
    LogTrajectoryRequest,
    LogTrajectoryResponse,
    CheckPrecedentRequest,
    CheckPrecedentResponse,
    DiscoverSkillsRequest,
    DiscoverSkillsResponse,
    StoreSkillRequest,
    StoreSkillResponse,
    WorkPacketInput,
    ResultInput,
    Skill,
    PrecedentMatch,
    TRAJECTORY_COLLECTION,
    SKILLS_COLLECTION,
    FAILURE_SIMILARITY_THRESHOLD,
    SUCCESS_SIMILARITY_THRESHOLD,
    VECTOR_DIMENSION,
    _qdrant_client,
    _embedding_service,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client."""
    client = MagicMock()
    client.upsert = MagicMock()
    client.search = MagicMock(return_value=[])

    # Mock get_collection to return collection info with correct dimensions
    mock_collection_info = MagicMock()
    mock_collection_info.config.params.vectors.size = 384
    client.get_collection = MagicMock(return_value=mock_collection_info)

    return client


@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    service = AsyncMock()
    service.generate_embedding = AsyncMock(return_value=[0.1] * 384)
    return service


@pytest.fixture
def app_with_routes(mock_qdrant_client, mock_embedding_service):
    """Create FastAPI app with agent tools routes registered."""
    app = FastAPI()
    register_routes(app, mock_qdrant_client, mock_embedding_service)
    return app


@pytest.fixture
def client(app_with_routes):
    """Create test client."""
    return TestClient(app_with_routes)


# =============================================================================
# Pydantic Model Tests
# =============================================================================


class TestPydanticModels:
    """Tests for Pydantic request/response models."""

    def test_work_packet_input_defaults(self):
        """Test WorkPacketInput has correct defaults."""
        packet = WorkPacketInput()
        assert packet.description == ""
        assert packet.type == "unknown"
        assert packet.tech_stack == []

    def test_work_packet_input_with_values(self):
        """Test WorkPacketInput with provided values."""
        packet = WorkPacketInput(
            description="Test task",
            type="feature",
            tech_stack=["python", "fastapi"]
        )
        assert packet.description == "Test task"
        assert packet.type == "feature"
        assert packet.tech_stack == ["python", "fastapi"]

    def test_result_input_required_status(self):
        """Test ResultInput requires status field."""
        with pytest.raises(ValueError):
            ResultInput()

    def test_result_input_with_status(self):
        """Test ResultInput with required status."""
        result = ResultInput(status="SUCCESS")
        assert result.status == "SUCCESS"
        assert result.agent_id == "unknown"
        assert result.retry_count == 0

    def test_log_trajectory_request(self):
        """Test LogTrajectoryRequest model."""
        request = LogTrajectoryRequest(
            packet_id="test-123",
            work_packet=WorkPacketInput(description="Test"),
            result=ResultInput(status="SUCCESS")
        )
        assert request.packet_id == "test-123"
        assert request.work_packet.description == "Test"
        assert request.result.status == "SUCCESS"

    def test_check_precedent_request_defaults(self):
        """Test CheckPrecedentRequest has correct defaults."""
        request = CheckPrecedentRequest(plan_summary="Test plan")
        assert request.plan_summary == "Test plan"
        assert request.lookback_limit == 5

    def test_discover_skills_request_defaults(self):
        """Test DiscoverSkillsRequest has correct defaults."""
        request = DiscoverSkillsRequest()
        assert request.query is None
        assert request.tech_stack == []
        assert request.domain is None
        assert request.limit == 3

    def test_skill_model(self):
        """Test Skill model."""
        skill = Skill(
            skill_id="skill-1",
            title="Test Skill",
            domain="testing",
            content="# Test"
        )
        assert skill.skill_id == "skill-1"
        assert skill.title == "Test Skill"
        assert skill.relevance_score == 0.0

    def test_precedent_match_model(self):
        """Test PrecedentMatch model."""
        match = PrecedentMatch(
            packet_id="pkg-1",
            similarity=0.87
        )
        assert match.packet_id == "pkg-1"
        assert match.similarity == 0.87
        assert match.tech_stack == []


# =============================================================================
# Dependency Tests
# =============================================================================


class TestDependencies:
    """Tests for dependency injection functions."""

    def test_get_qdrant_not_initialized(self):
        """Test get_qdrant raises when not initialized."""
        import src.mcp_server.agent_tools as module
        original = module._qdrant_client
        module._qdrant_client = None

        with pytest.raises(HTTPException) as exc_info:
            get_qdrant()

        assert exc_info.value.status_code == 503
        assert "not initialized" in exc_info.value.detail

        module._qdrant_client = original

    def test_get_qdrant_when_initialized(self, mock_qdrant_client):
        """Test get_qdrant returns client when initialized."""
        import src.mcp_server.agent_tools as module
        original = module._qdrant_client
        module._qdrant_client = mock_qdrant_client

        result = get_qdrant()
        assert result == mock_qdrant_client

        module._qdrant_client = original

    @pytest.mark.asyncio
    async def test_get_embedding_not_initialized(self):
        """Test get_embedding raises when not initialized."""
        import src.mcp_server.agent_tools as module
        original = module._embedding_service
        module._embedding_service = None

        with pytest.raises(HTTPException) as exc_info:
            await get_embedding("test")

        assert exc_info.value.status_code == 503
        assert "not initialized" in exc_info.value.detail

        module._embedding_service = original

    @pytest.mark.asyncio
    async def test_get_embedding_when_initialized(self, mock_embedding_service):
        """Test get_embedding returns vector when initialized."""
        import src.mcp_server.agent_tools as module
        original = module._embedding_service
        module._embedding_service = mock_embedding_service

        result = await get_embedding("test")
        assert len(result) == 384
        mock_embedding_service.generate_embedding.assert_called_once_with("test")

        module._embedding_service = original


# =============================================================================
# Log Trajectory Endpoint Tests
# =============================================================================


class TestLogTrajectoryEndpoint:
    """Tests for log_trajectory endpoint."""

    def test_log_trajectory_success(self, client, mock_qdrant_client):
        """Test successful trajectory logging."""
        response = client.post(
            "/tools/log_trajectory",
            json={
                "packet_id": "test-123",
                "work_packet": {
                    "description": "Test task",
                    "type": "feature",
                    "tech_stack": ["python"]
                },
                "result": {
                    "status": "SUCCESS",
                    "agent_id": "agent-1"
                }
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "point_id" in data
        assert data["collection"] == TRAJECTORY_COLLECTION

        # Verify qdrant.upsert was called
        mock_qdrant_client.upsert.assert_called_once()

    def test_log_trajectory_failure_status(self, client, mock_qdrant_client):
        """Test logging a failure trajectory."""
        response = client.post(
            "/tools/log_trajectory",
            json={
                "packet_id": "test-456",
                "work_packet": {
                    "description": "Failed task",
                    "type": "bug"
                },
                "result": {
                    "status": "FAILURE",
                    "error_message": "Connection timeout",
                    "retry_count": 3
                }
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_log_trajectory_with_human_verdict(self, client, mock_qdrant_client):
        """Test logging trajectory with human verdict."""
        response = client.post(
            "/tools/log_trajectory",
            json={
                "packet_id": "test-789",
                "work_packet": {"description": "Disputed task"},
                "result": {
                    "status": "SUCCESS",
                    "human_verdict": "ARCHITECT_WINS",
                    "mitigation_applied": "Used alternative approach"
                }
            }
        )

        assert response.status_code == 200

    def test_log_trajectory_missing_required_field(self, client):
        """Test log_trajectory with missing required field."""
        response = client.post(
            "/tools/log_trajectory",
            json={
                "packet_id": "test-123",
                "work_packet": {"description": "Test"}
                # Missing 'result' field
            }
        )

        assert response.status_code == 422  # Validation error


# =============================================================================
# Check Precedent Endpoint Tests
# =============================================================================


class TestCheckPrecedentEndpoint:
    """Tests for check_precedent endpoint."""

    def test_check_precedent_clean(self, client, mock_qdrant_client):
        """Test precedent check with no matches (CLEAN verdict)."""
        mock_qdrant_client.search.return_value = []

        response = client.post(
            "/tools/check_precedent",
            json={"plan_summary": "Implement new feature X"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["verdict"] == "CLEAN"
        assert data["failures"] == []
        assert data["successes"] == []
        assert "standard review" in data["recommendation"].lower()

    def test_check_precedent_stop_on_failures(self, client, mock_qdrant_client):
        """Test precedent check returns STOP when failures found."""
        # Mock failure search results
        mock_hit = MagicMock()
        mock_hit.payload = {
            "packet_id": "failed-pkg-1",
            "error_reason": "Type error",
            "timestamp": "2025-12-10T10:00:00",
            "task_type": "feature",
            "tech_stack": ["python"]
        }
        mock_hit.score = 0.92

        # First call returns failures, second returns empty (successes)
        mock_qdrant_client.search.side_effect = [[mock_hit], []]

        response = client.post(
            "/tools/check_precedent",
            json={"plan_summary": "Similar failed plan"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["verdict"] == "STOP"
        assert len(data["failures"]) == 1
        assert data["failures"][0]["packet_id"] == "failed-pkg-1"
        assert "REJECT" in data["recommendation"]

    def test_check_precedent_positive_on_successes(self, client, mock_qdrant_client):
        """Test precedent check returns POSITIVE when only successes found."""
        # Mock success search results
        mock_hit = MagicMock()
        mock_hit.payload = {
            "packet_id": "success-pkg-1",
            "timestamp": "2025-12-10T10:00:00",
            "task_type": "feature",
            "tech_stack": ["python"]
        }
        mock_hit.score = 0.95

        # First call returns empty (failures), second returns successes
        mock_qdrant_client.search.side_effect = [[], [mock_hit]]

        response = client.post(
            "/tools/check_precedent",
            json={"plan_summary": "Similar successful plan"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["verdict"] == "POSITIVE"
        assert len(data["successes"]) == 1
        assert "confidence boost" in data["recommendation"].lower()

    def test_check_precedent_custom_limit(self, client, mock_qdrant_client):
        """Test precedent check with custom lookback_limit."""
        mock_qdrant_client.search.return_value = []

        response = client.post(
            "/tools/check_precedent",
            json={
                "plan_summary": "Test plan",
                "lookback_limit": 10
            }
        )

        assert response.status_code == 200
        # Verify the limit was passed to search
        call_args = mock_qdrant_client.search.call_args_list[0]
        assert call_args.kwargs.get("limit") == 10


# =============================================================================
# Discover Skills Endpoint Tests
# =============================================================================


class TestDiscoverSkillsEndpoint:
    """Tests for discover_skills endpoint."""

    def test_discover_skills_with_query(self, client, mock_qdrant_client):
        """Test skill discovery with query."""
        mock_hit = MagicMock()
        mock_hit.payload = {
            "skill_id": "skill-1",
            "title": "FastAPI Best Practices",
            "domain": "backend",
            "trigger": ["fastapi", "api"],
            "content": "# FastAPI Guide",
            "file_path": "/skills/fastapi.md"
        }
        mock_hit.score = 0.88
        mock_qdrant_client.search.return_value = [mock_hit]

        response = client.post(
            "/tools/discover_skills",
            json={"query": "How to build FastAPI endpoints"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_found"] == 1
        assert len(data["skills"]) == 1
        assert data["skills"][0]["skill_id"] == "skill-1"
        assert data["skills"][0]["relevance_score"] == 0.88

    def test_discover_skills_with_tech_stack(self, client, mock_qdrant_client):
        """Test skill discovery with tech_stack filter."""
        mock_qdrant_client.search.return_value = []

        response = client.post(
            "/tools/discover_skills",
            json={"tech_stack": ["python", "redis", "qdrant"]}
        )

        assert response.status_code == 200
        data = response.json()
        assert "python" in data["query"].lower() or "redis" in data["query"].lower()

    def test_discover_skills_with_domain_filter(self, client, mock_qdrant_client):
        """Test skill discovery with domain filter."""
        mock_qdrant_client.search.return_value = []

        response = client.post(
            "/tools/discover_skills",
            json={
                "query": "authentication",
                "domain": "security"
            }
        )

        assert response.status_code == 200
        # Verify filter was applied
        call_args = mock_qdrant_client.search.call_args
        assert call_args.kwargs.get("query_filter") is not None

    def test_discover_skills_empty_request(self, client):
        """Test skill discovery with empty request returns empty results."""
        response = client.post(
            "/tools/discover_skills",
            json={}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_found"] == 0
        assert data["skills"] == []
        assert data["query"] == ""

    def test_discover_skills_custom_limit(self, client, mock_qdrant_client):
        """Test skill discovery with custom limit."""
        mock_qdrant_client.search.return_value = []

        response = client.post(
            "/tools/discover_skills",
            json={
                "query": "testing",
                "limit": 10
            }
        )

        assert response.status_code == 200
        call_args = mock_qdrant_client.search.call_args
        assert call_args.kwargs.get("limit") == 10


# =============================================================================
# Registration Tests
# =============================================================================


class TestRouteRegistration:
    """Tests for route registration."""

    def test_register_routes(self, mock_qdrant_client, mock_embedding_service):
        """Test routes are registered correctly."""
        app = FastAPI()
        register_routes(app, mock_qdrant_client, mock_embedding_service)

        # Check routes exist
        routes = [route.path for route in app.routes]
        assert "/tools/log_trajectory" in routes
        assert "/tools/check_precedent" in routes
        assert "/tools/discover_skills" in routes

    def test_register_routes_sets_globals(self, mock_qdrant_client, mock_embedding_service):
        """Test register_routes sets global clients."""
        import src.mcp_server.agent_tools as module

        app = FastAPI()
        register_routes(app, mock_qdrant_client, mock_embedding_service)

        assert module._qdrant_client == mock_qdrant_client
        assert module._embedding_service == mock_embedding_service


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_log_trajectory_qdrant_error(self, client, mock_qdrant_client):
        """Test log_trajectory handles Qdrant errors."""
        mock_qdrant_client.upsert.side_effect = Exception("Qdrant connection lost")

        response = client.post(
            "/tools/log_trajectory",
            json={
                "packet_id": "test-123",
                "work_packet": {"description": "Test"},
                "result": {"status": "SUCCESS"}
            }
        )

        assert response.status_code == 500
        assert "Failed to log trajectory" in response.json()["detail"]

    def test_check_precedent_qdrant_error(self, client, mock_qdrant_client):
        """Test check_precedent handles Qdrant errors."""
        mock_qdrant_client.search.side_effect = Exception("Search failed")

        response = client.post(
            "/tools/check_precedent",
            json={"plan_summary": "Test plan"}
        )

        assert response.status_code == 500
        assert "Failed to check precedent" in response.json()["detail"]

    def test_discover_skills_qdrant_error(self, client, mock_qdrant_client):
        """Test discover_skills handles Qdrant errors."""
        mock_qdrant_client.search.side_effect = Exception("Connection refused")

        response = client.post(
            "/tools/discover_skills",
            json={"query": "test"}
        )

        assert response.status_code == 500
        assert "Failed to discover skills" in response.json()["detail"]


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_trajectory_collection_name(self):
        """Test trajectory collection name is correct."""
        assert TRAJECTORY_COLLECTION == "trajectory_logs"

    def test_skills_collection_name(self):
        """Test skills collection name is correct."""
        assert SKILLS_COLLECTION == "veris_skills"

    def test_failure_threshold(self):
        """Test failure similarity threshold is set correctly."""
        assert FAILURE_SIMILARITY_THRESHOLD == 0.85

    def test_success_threshold(self):
        """Test success similarity threshold is set correctly."""
        assert SUCCESS_SIMILARITY_THRESHOLD == 0.90
        assert SUCCESS_SIMILARITY_THRESHOLD > FAILURE_SIMILARITY_THRESHOLD

    def test_vector_dimension(self):
        """Test vector dimension constant is set correctly."""
        assert VECTOR_DIMENSION == 384


# =============================================================================
# Collection Management Tests
# =============================================================================


class TestEnsureCollections:
    """Tests for ensure_collections function."""

    def test_creates_missing_collections(self):
        """Test that missing collections are created."""
        mock_client = MagicMock()
        # Simulate collections not existing
        mock_client.get_collection.side_effect = Exception("Not found: collection doesn't exist")
        mock_client.create_collection.return_value = None

        with patch("src.mcp_server.agent_tools.QDRANT_AVAILABLE", True):
            with patch("src.mcp_server.agent_tools.qdrant_models") as mock_models:
                mock_models.VectorParams = MagicMock()
                mock_models.Distance.COSINE = "Cosine"

                ensure_collections(mock_client)

        # Should have tried to create both collections
        assert mock_client.create_collection.call_count == 2

        # Verify correct collection names
        call_args = [call[1]["collection_name"] for call in mock_client.create_collection.call_args_list]
        assert TRAJECTORY_COLLECTION in call_args
        assert SKILLS_COLLECTION in call_args

    def test_skips_existing_collections_with_correct_dims(self):
        """Test that existing collections with correct dimensions are skipped."""
        mock_client = MagicMock()

        # Simulate collections existing with correct dimensions
        mock_collection_info = MagicMock()
        mock_collection_info.config.params.vectors.size = 384
        mock_client.get_collection.return_value = mock_collection_info

        with patch("src.mcp_server.agent_tools.QDRANT_AVAILABLE", True):
            with patch("src.mcp_server.agent_tools.qdrant_models") as mock_models:
                ensure_collections(mock_client)

        # Should not have tried to create any collections
        mock_client.create_collection.assert_not_called()

    def test_raises_on_wrong_dimensions(self):
        """Test that wrong dimensions raise ValueError."""
        mock_client = MagicMock()

        # Simulate collection existing with wrong dimensions
        mock_collection_info = MagicMock()
        mock_collection_info.config.params.vectors.size = 1536  # Wrong!
        mock_client.get_collection.return_value = mock_collection_info

        with patch("src.mcp_server.agent_tools.QDRANT_AVAILABLE", True):
            with patch("src.mcp_server.agent_tools.qdrant_models"):
                with pytest.raises(ValueError) as exc_info:
                    ensure_collections(mock_client)

        assert "wrong dimensions" in str(exc_info.value)
        assert "expected 384" in str(exc_info.value)
        assert "got 1536" in str(exc_info.value)

    def test_skips_when_qdrant_unavailable(self):
        """Test that function skips gracefully when Qdrant is unavailable."""
        mock_client = MagicMock()

        with patch("src.mcp_server.agent_tools.QDRANT_AVAILABLE", False):
            ensure_collections(mock_client)

        # Should not have called any Qdrant methods
        mock_client.get_collection.assert_not_called()
        mock_client.create_collection.assert_not_called()

    def test_raises_on_create_failure(self):
        """Test that creation failure propagates."""
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("Not found: collection doesn't exist")
        mock_client.create_collection.side_effect = Exception("Connection refused")

        with patch("src.mcp_server.agent_tools.QDRANT_AVAILABLE", True):
            with patch("src.mcp_server.agent_tools.qdrant_models") as mock_models:
                mock_models.VectorParams = MagicMock()
                mock_models.Distance.COSINE = "Cosine"

                with pytest.raises(Exception) as exc_info:
                    ensure_collections(mock_client)

        assert "Connection refused" in str(exc_info.value)


class TestRegisterRoutesWithCollections:
    """Tests for register_routes integration with ensure_collections."""

    def test_register_routes_calls_ensure_collections(self):
        """Test that register_routes calls ensure_collections."""
        app = FastAPI()
        mock_client = MagicMock()
        mock_embedding = MagicMock()

        # Mock collection info to exist with correct dimensions
        mock_collection_info = MagicMock()
        mock_collection_info.config.params.vectors.size = 384
        mock_client.get_collection.return_value = mock_collection_info

        with patch("src.mcp_server.agent_tools.QDRANT_AVAILABLE", True):
            with patch("src.mcp_server.agent_tools.qdrant_models"):
                register_routes(app, mock_client, mock_embedding)

        # Should have checked both collections
        assert mock_client.get_collection.call_count == 2

    def test_register_routes_fails_on_ensure_error(self):
        """Test that register_routes fails if ensure_collections fails."""
        app = FastAPI()
        mock_client = MagicMock()
        mock_embedding = MagicMock()

        # Mock collection with wrong dimensions
        mock_collection_info = MagicMock()
        mock_collection_info.config.params.vectors.size = 1536
        mock_client.get_collection.return_value = mock_collection_info

        with patch("src.mcp_server.agent_tools.QDRANT_AVAILABLE", True):
            with patch("src.mcp_server.agent_tools.qdrant_models"):
                with pytest.raises(ValueError):
                    register_routes(app, mock_client, mock_embedding)


# =============================================================================
# Store Skill Model Tests
# =============================================================================


class TestStoreSkillModels:
    """Tests for StoreSkillRequest and StoreSkillResponse models."""

    def test_store_skill_request_valid(self):
        """Test valid StoreSkillRequest creation."""
        request = StoreSkillRequest(
            title="Python FastAPI Endpoint",
            domain="api",
            trigger=["add endpoint", "create route"],
            tech_stack=["python", "fastapi"],
            content="# How to create a FastAPI endpoint...",
        )
        assert request.title == "Python FastAPI Endpoint"
        assert request.domain == "api"
        assert len(request.trigger) == 2
        assert "python" in request.tech_stack
        assert request.file_path == ""

    def test_store_skill_request_minimal(self):
        """Test StoreSkillRequest with only required fields."""
        request = StoreSkillRequest(
            title="Simple Skill",
            domain="general",
            content="Basic content",
        )
        assert request.title == "Simple Skill"
        assert request.trigger == []
        assert request.tech_stack == []
        assert request.file_path == ""

    def test_store_skill_request_with_file_path(self):
        """Test StoreSkillRequest with optional file_path."""
        request = StoreSkillRequest(
            title="Test Skill",
            domain="testing",
            content="Content here",
            file_path="/skills/testing/pytest.md",
        )
        assert request.file_path == "/skills/testing/pytest.md"

    def test_store_skill_response(self):
        """Test StoreSkillResponse creation."""
        response = StoreSkillResponse(
            success=True,
            skill_id="abc123def456",
            message="Skill stored successfully",
        )
        assert response.success is True
        assert response.skill_id == "abc123def456"
        assert "stored successfully" in response.message


# =============================================================================
# Store Skill Endpoint Tests
# =============================================================================


class TestStoreSkillEndpoint:
    """Tests for the store_skill endpoint."""

    @pytest.mark.asyncio
    async def test_store_skill_generates_correct_skill_id(self):
        """Test that skill_id is MD5 of title."""
        import hashlib

        mock_qdrant = MagicMock()
        mock_embedding = AsyncMock(return_value=[0.1] * 384)

        request = StoreSkillRequest(
            title="Test Skill Title",
            domain="testing",
            content="Test content",
        )

        expected_skill_id = hashlib.md5("Test Skill Title".encode()).hexdigest()

        with patch("src.mcp_server.agent_tools._qdrant_client", mock_qdrant):
            with patch("src.mcp_server.agent_tools._embedding_service") as mock_svc:
                mock_svc.generate_embedding = mock_embedding
                with patch("src.mcp_server.agent_tools.qdrant_models") as mock_models:
                    mock_models.PointStruct = MagicMock()

                    response = await store_skill(request, mock_qdrant, None)

        assert response.skill_id == expected_skill_id
        assert response.success is True

    @pytest.mark.asyncio
    async def test_store_skill_builds_correct_embedding_text(self):
        """Test that embedding is generated from title + domain + triggers + content[:500]."""
        mock_qdrant = MagicMock()
        captured_text = None

        async def capture_embedding(text):
            nonlocal captured_text
            captured_text = text
            return [0.1] * 384

        request = StoreSkillRequest(
            title="FastAPI Endpoints",
            domain="api",
            trigger=["create endpoint", "add route"],
            content="This is the skill content that explains how to do things.",
        )

        with patch("src.mcp_server.agent_tools._qdrant_client", mock_qdrant):
            with patch("src.mcp_server.agent_tools._embedding_service") as mock_svc:
                mock_svc.generate_embedding = capture_embedding
                with patch("src.mcp_server.agent_tools.qdrant_models") as mock_models:
                    mock_models.PointStruct = MagicMock()

                    await store_skill(request, mock_qdrant, None)

        # Verify embedding text contains all components
        assert "FastAPI Endpoints" in captured_text
        assert "api" in captured_text
        assert "create endpoint" in captured_text
        assert "add route" in captured_text
        assert "skill content" in captured_text

    @pytest.mark.asyncio
    async def test_store_skill_truncates_long_content(self):
        """Test that content is truncated to 500 chars for embedding."""
        mock_qdrant = MagicMock()
        captured_text = None

        async def capture_embedding(text):
            nonlocal captured_text
            captured_text = text
            return [0.1] * 384

        long_content = "A" * 1000  # 1000 chars

        request = StoreSkillRequest(
            title="Long Content Skill",
            domain="test",
            content=long_content,
        )

        with patch("src.mcp_server.agent_tools._qdrant_client", mock_qdrant):
            with patch("src.mcp_server.agent_tools._embedding_service") as mock_svc:
                mock_svc.generate_embedding = capture_embedding
                with patch("src.mcp_server.agent_tools.qdrant_models") as mock_models:
                    mock_models.PointStruct = MagicMock()

                    await store_skill(request, mock_qdrant, None)

        # Content in embedding should be truncated
        # Full text = title + domain + content[:500]
        assert captured_text.count("A") == 500  # Only 500 A's, not 1000

    @pytest.mark.asyncio
    async def test_store_skill_upserts_to_correct_collection(self):
        """Test that skill is upserted to veris_skills collection."""
        mock_qdrant = MagicMock()

        request = StoreSkillRequest(
            title="Collection Test",
            domain="test",
            content="Content",
        )

        with patch("src.mcp_server.agent_tools._qdrant_client", mock_qdrant):
            with patch("src.mcp_server.agent_tools._embedding_service") as mock_svc:
                mock_svc.generate_embedding = AsyncMock(return_value=[0.1] * 384)
                with patch("src.mcp_server.agent_tools.qdrant_models") as mock_models:
                    mock_models.PointStruct = MagicMock()

                    await store_skill(request, mock_qdrant, None)

        # Verify upsert was called with correct collection
        mock_qdrant.upsert.assert_called_once()
        call_kwargs = mock_qdrant.upsert.call_args[1]
        assert call_kwargs["collection_name"] == "veris_skills"

    @pytest.mark.asyncio
    async def test_store_skill_payload_contains_all_fields(self):
        """Test that payload contains all required fields."""
        mock_qdrant = MagicMock()
        captured_payload = None

        def capture_upsert(**kwargs):
            nonlocal captured_payload
            point = kwargs["points"][0]
            captured_payload = point.payload

        mock_qdrant.upsert = capture_upsert

        request = StoreSkillRequest(
            title="Payload Test",
            domain="testing",
            trigger=["test trigger"],
            tech_stack=["python"],
            content="Test content",
            file_path="/test/path.md",
        )

        with patch("src.mcp_server.agent_tools._qdrant_client", mock_qdrant):
            with patch("src.mcp_server.agent_tools._embedding_service") as mock_svc:
                mock_svc.generate_embedding = AsyncMock(return_value=[0.1] * 384)
                with patch("src.mcp_server.agent_tools.qdrant_models") as mock_models:
                    mock_models.PointStruct = lambda id, vector, payload: MagicMock(
                        id=id, vector=vector, payload=payload
                    )

                    await store_skill(request, mock_qdrant, None)

        assert captured_payload["title"] == "Payload Test"
        assert captured_payload["domain"] == "testing"
        assert captured_payload["trigger"] == ["test trigger"]
        assert captured_payload["tech_stack"] == ["python"]
        assert captured_payload["content"] == "Test content"
        assert captured_payload["file_path"] == "/test/path.md"
        assert "timestamp" in captured_payload
        assert "skill_id" in captured_payload

    @pytest.mark.asyncio
    async def test_store_skill_idempotent_updates(self):
        """Test that same title produces same skill_id for updates."""
        import hashlib

        mock_qdrant = MagicMock()

        request1 = StoreSkillRequest(
            title="Same Title",
            domain="test",
            content="First version",
        )
        request2 = StoreSkillRequest(
            title="Same Title",
            domain="test",
            content="Updated version",
        )

        with patch("src.mcp_server.agent_tools._qdrant_client", mock_qdrant):
            with patch("src.mcp_server.agent_tools._embedding_service") as mock_svc:
                mock_svc.generate_embedding = AsyncMock(return_value=[0.1] * 384)
                with patch("src.mcp_server.agent_tools.qdrant_models") as mock_models:
                    mock_models.PointStruct = MagicMock()

                    response1 = await store_skill(request1, mock_qdrant, None)
                    response2 = await store_skill(request2, mock_qdrant, None)

        # Same title should produce same skill_id
        assert response1.skill_id == response2.skill_id
        expected_id = hashlib.md5("Same Title".encode()).hexdigest()
        assert response1.skill_id == expected_id
