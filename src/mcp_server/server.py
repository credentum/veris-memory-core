#!/usr/bin/env python3
"""
Context Store MCP Server using the official MCP Python SDK.

This module implements the Model Context Protocol server for the context store
using the official MCP Python SDK for proper protocol compliance.
"""
# flake8: noqa: E501

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

# MCP SDK imports
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import EmbeddedResource, ImageContent, Resource, TextContent, Tool

# Storage client imports
try:
    # Try absolute imports from src package
    from src.core.agent_namespace import AgentNamespace
    from src.core.config import Config
    from src.core.embedding_config import create_embedding_generator
    from src.core.rate_limiter import rate_limit_check
    from src.core.ssl_config import SSLConfigManager
    from src.security.cypher_validator import CypherValidator
    from src.storage.kv_store import ContextKV
    from src.storage.neo4j_client import Neo4jInitializer
    from src.storage.qdrant_client import VectorDBInitializer
    from src.storage.reranker import get_reranker
    from src.validators.config_validator import validate_all_configs
    from src.security.input_validator import InputValidator, ContentTypeValidator
    # Fact system imports
    from src.storage.fact_store import FactStore
    from src.core.intent_classifier import IntentClassifier, IntentType
    from src.core.fact_extractor import FactExtractor
    from src.core.qa_generator import QAPairGenerator
    from src.storage.fact_ranker import FactAwareRanker
    from src.core.query_rewriter import FactQueryRewriter
    from src.middleware.scope_validator import ScopeValidator, ScopeMiddleware
    # Phase 3: Graph integration imports
    from src.storage.graph_enhancer import GraphSignalEnhancer
    from src.storage.graph_fact_store import GraphFactStore
    from src.storage.hybrid_scorer import HybridScorer, ScoringMode
    from src.core.graph_query_expander import GraphQueryExpander
    # Metrics collection import
    from src.monitoring.request_metrics import get_metrics_collector
    # Response validation import
    from src.core.response_validator import validate_mcp_response
    # Search enhancement imports
    from src.mcp_server.search_enhancements import (
        apply_search_enhancements,
        is_technical_query
    )
    from src.mcp_server.query_relevance_scorer import QueryRelevanceScorer
    # Circuit breaker and error handling imports
    from src.storage.circuit_breaker import with_mcp_circuit_breaker, get_mcp_service_health, CircuitBreakerError
    from src.core.error_codes import ErrorCode, create_error_response
except ImportError:
    # Fallback for different import contexts
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    
    from core.agent_namespace import AgentNamespace
    from core.config import Config
    from core.embedding_config import create_embedding_generator
    from core.rate_limiter import rate_limit_check
    from core.ssl_config import SSLConfigManager
    from security.cypher_validator import CypherValidator
    # Metrics collection import for fallback path
    from monitoring.request_metrics import get_metrics_collector
    from storage.kv_store import ContextKV
    from storage.neo4j_client import Neo4jInitializer
    from storage.qdrant_client import VectorDBInitializer
    from storage.reranker import get_reranker
    from validators.config_validator import validate_all_configs
    from security.input_validator import InputValidator, ContentTypeValidator
    # Fact system imports
    from storage.fact_store import FactStore
    from core.intent_classifier import IntentClassifier, IntentType
    from core.fact_extractor import FactExtractor
    from core.qa_generator import QAPairGenerator
    from storage.fact_ranker import FactAwareRanker
    from core.query_rewriter import FactQueryRewriter
    from middleware.scope_validator import ScopeValidator, ScopeMiddleware
    # Metrics collection import
    from monitoring.request_metrics import get_metrics_collector
    # Search enhancement imports
    from mcp_server.search_enhancements import (
        apply_search_enhancements,
        is_technical_query
    )
    from mcp_server.query_relevance_scorer import QueryRelevanceScorer
    # Circuit breaker and error handling imports
    from storage.circuit_breaker import with_mcp_circuit_breaker, get_mcp_service_health, CircuitBreakerError
    from core.error_codes import ErrorCode, create_error_response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create MCP server
server = Server("context-store")

# Global storage client instances
neo4j_client = None
qdrant_client = None
kv_store = None
embedding_generator = None

# Global metrics collector instance
metrics_collector = None

# Global fact system instances
fact_store = None
intent_classifier = IntentClassifier()
fact_extractor = FactExtractor()
qa_generator = QAPairGenerator()
fact_ranker = FactAwareRanker()
query_rewriter = FactQueryRewriter()
scope_validator = ScopeValidator()
scope_middleware = None
# Phase 3: Graph integration instances
graph_fact_store = None
graph_enhancer = None
hybrid_scorer = None
graph_query_expander = None

# Global security and namespace instances
cypher_validator = CypherValidator()
agent_namespace = AgentNamespace()
input_validator = InputValidator()
content_type_validator = ContentTypeValidator()

# Tool selector bridge import
try:
    from .tool_selector_bridge import get_tool_selector_bridge

    tool_selector_bridge = get_tool_selector_bridge()
except ImportError as e:
    logger.warning(f"Tool selector bridge not available: {e}")
    tool_selector_bridge = None


async def initialize_storage_clients() -> Dict[str, Any]:
    """Initialize storage clients with SSL/TLS support."""
    global neo4j_client, qdrant_client, kv_store, embedding_generator, fact_store, scope_middleware, metrics_collector

    try:
        # Initialize metrics collector
        try:
            metrics_collector = get_metrics_collector()
            await metrics_collector.start_queue_processor()
            logger.info("✅ Metrics collector initialized")
        except Exception as metrics_error:
            logger.warning(f"⚠️ Failed to initialize metrics collector: {metrics_error}")
            metrics_collector = None  # Continue without metrics
        
        # Validate configuration
        config_result = validate_all_configs()
        base_config = config_result.get("config", {})

        if not config_result.get("valid", False):
            logger.warning(f"⚠️ Configuration validation failed: {config_result}")
            # Continue with initialization even if validation fails for missing optional configs

        # Initialize SSL configuration manager
        ssl_manager = SSLConfigManager(base_config)

        # Validate SSL certificates if configured
        ssl_validation = ssl_manager.validate_ssl_certificates()
        for backend, valid in ssl_validation.items():
            if not valid:
                logger.warning(f"⚠️ SSL certificate validation failed for {backend}")

        # Initialize Neo4j with SSL support
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        if neo4j_password:
            # Get Neo4j URI from environment
            neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            neo4j_username = os.getenv("NEO4J_USER", "neo4j")
            
            # Parse URI to extract host and port
            import urllib.parse
            parsed = urllib.parse.urlparse(neo4j_uri)
            neo4j_host = parsed.hostname or "localhost"
            neo4j_port = parsed.port or 7687
            
            # Create config for Neo4jInitializer
            neo4j_config = {
                "neo4j": {
                    "host": neo4j_host,
                    "port": neo4j_port,
                    "database": "neo4j",
                    "ssl": neo4j_uri.startswith("bolt+s") or neo4j_uri.startswith("neo4j+s")
                }
            }
            
            neo4j_client = Neo4jInitializer(config=neo4j_config)
            neo4j_ssl_config = ssl_manager.get_neo4j_ssl_config()

            # Neo4j.connect only accepts username and password
            # SSL config is handled internally by Neo4j client
            if neo4j_client.connect(username=neo4j_username, password=neo4j_password):
                ssl_status = "with SSL" if neo4j_ssl_config.get("encrypted") else "without SSL"
                logger.info(f"✅ Neo4j client initialized at {neo4j_uri} {ssl_status}")
            else:
                neo4j_client = None
                logger.warning(f"⚠️ Neo4j connection failed at {neo4j_uri}")
        else:
            neo4j_client = None
            logger.warning("⚠️ Neo4j disabled: NEO4J_PASSWORD not set")

        # Initialize Qdrant with SSL support
        qdrant_url = os.getenv("QDRANT_URL")
        if qdrant_url:
            # Parse Qdrant URL to extract host and port
            import urllib.parse
            parsed_qdrant = urllib.parse.urlparse(qdrant_url)
            qdrant_host = parsed_qdrant.hostname or "localhost"
            qdrant_port = parsed_qdrant.port or 6333
            
            # Create config for VectorDBInitializer
            qdrant_config = {
                "qdrant": {
                    "host": qdrant_host,
                    "port": qdrant_port,
                    "ssl": qdrant_url.startswith("https"),
                    "timeout": 5
                }
            }
            
            qdrant_client = VectorDBInitializer(config=qdrant_config)
            qdrant_ssl_config = ssl_manager.get_qdrant_ssl_config()

            if qdrant_client.connect():
                ssl_status = "with HTTPS" if qdrant_ssl_config.get("https") else "without SSL"
                logger.info(f"✅ Qdrant client initialized at {qdrant_url} {ssl_status}")
            else:
                qdrant_client = None
                logger.warning(f"⚠️ Qdrant connection failed at {qdrant_url}")
        else:
            qdrant_client = None
            logger.warning("⚠️ Qdrant disabled: QDRANT_URL not set")

        # Initialize KV Store with SSL support
        redis_url = os.getenv("REDIS_URL")
        redis_password = os.getenv("REDIS_PASSWORD")
        if redis_url:
            # Parse Redis URL to extract host and port
            import urllib.parse
            parsed_redis = urllib.parse.urlparse(redis_url)
            redis_host = parsed_redis.hostname or "localhost"
            redis_port = parsed_redis.port or 6379
            
            # Create config for ContextKV
            redis_config = {
                "redis": {
                    "host": redis_host,
                    "port": redis_port,
                    "database": 0,
                    "ssl": redis_url.startswith("rediss")
                }
            }
            
            kv_store = ContextKV(config=redis_config)
            redis_ssl_config = ssl_manager.get_redis_ssl_config()

            # Pass password if available
            if kv_store.connect(redis_password=redis_password):
                ssl_status = "with SSL" if redis_ssl_config.get("ssl") else "without SSL"
                logger.info(f"✅ KV store initialized at {redis_url} {ssl_status}")
            else:
                kv_store = None
                logger.warning(f"⚠️ KV store connection failed at {redis_url}")
        else:
            kv_store = None
            logger.warning("⚠️ KV store disabled: REDIS_URL not set")

        # Initialize embedding generator
        try:
            embedding_generator = await create_embedding_generator(base_config)
            logger.info("✅ Embedding generator initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize embedding generator: {e}")
            embedding_generator = None

        # Initialize fact system
        try:
            if kv_store and kv_store.redis_client:
                fact_store = FactStore(kv_store.redis_client)
                scope_middleware = ScopeMiddleware(scope_validator)
                
                # Phase 3: Initialize graph integration
                try:
                    # Initialize graph enhancer with Neo4j client
                    graph_enhancer = GraphSignalEnhancer(neo4j_client, config)
                    
                    # Initialize graph-enabled fact store
                    graph_fact_store = GraphFactStore(fact_store, neo4j_client)
                    
                    # Initialize hybrid scorer
                    hybrid_scorer = HybridScorer(fact_ranker, graph_enhancer, config)
                    
                    # Initialize graph query expander
                    graph_query_expander = GraphQueryExpander(
                        neo4j_client, intent_classifier, query_rewriter, config
                    )
                    
                    logger.info("✅ Graph integration initialized")
                except Exception as graph_error:
                    logger.warning(f"⚠️ Graph integration failed: {graph_error}")
                    # Fallback to non-graph mode
                    graph_enhancer = None
                    graph_fact_store = None
                    hybrid_scorer = HybridScorer(fact_ranker, None, config)
                    graph_query_expander = None
                
                logger.info("✅ Fact system initialized")
            else:
                fact_store = None
                scope_middleware = None
                graph_fact_store = None
                graph_enhancer = None
                hybrid_scorer = None
                graph_query_expander = None
                logger.warning("⚠️ Fact system disabled: Redis not available")
        except Exception as e:
            logger.error(f"❌ Failed to initialize fact system: {e}")
            fact_store = None
            scope_middleware = None
            graph_fact_store = None
            graph_enhancer = None
            hybrid_scorer = None
            graph_query_expander = None

        logger.info("✅ Storage clients initialization completed")

        return {
            "success": True,
            "neo4j_initialized": neo4j_client is not None,
            "qdrant_initialized": qdrant_client is not None,
            "kv_store_initialized": kv_store is not None,
            "embedding_initialized": embedding_generator is not None,
            "fact_system_initialized": fact_store is not None,
            "message": "Storage clients initialized successfully",
        }

    except Exception as e:
        logger.error(f"❌ Failed to initialize storage clients: {e}")
        # Don't raise to allow server to start with partial functionality
        neo4j_client = None
        qdrant_client = None
        kv_store = None
        embedding_generator = None

        return {
            "success": False,
            "neo4j_initialized": False,
            "qdrant_initialized": False,
            "kv_store_initialized": False,
            "embedding_initialized": False,
            "fact_system_initialized": False,
            "message": f"Failed to initialize storage clients: {str(e)}",
        }


async def cleanup_storage_clients() -> None:
    """Clean up storage clients."""
    global neo4j_client, qdrant_client, kv_store

    if neo4j_client:
        neo4j_client.close()
        logger.info("Neo4j client closed")

    if kv_store:
        kv_store.close()
        logger.info("KV store closed")

    # Qdrant VectorDBInitializer doesn't have a close method in the current implementation
    # but we set it to None for cleanup
    if qdrant_client:
        qdrant_client = None
        logger.info("Qdrant client cleaned up")

    logger.info("Storage clients cleaned up")


@server.list_resources()
async def list_resources() -> List[Resource]:
    """List available resources."""
    return [
        Resource(
            uri="context://health",
            name="Health Status",
            description="Server and service health status",
            mimeType="application/json",
        ),
        Resource(
            uri="context://tools",
            name="Available Tools",
            description="List of available MCP tools",
            mimeType="application/json",
        ),
    ]


@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read resource content."""
    if uri == "context://health":
        health_status = await get_health_status()
        return json.dumps(health_status, indent=2)

    elif uri == "context://tools":
        tools_info = await get_tools_info()
        return json.dumps(tools_info, indent=2)

    else:
        raise ValueError(f"Unknown resource: {uri}")


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools."""
    return [
        Tool(
            name="store_context",
            description="Store context data with vector embeddings and graph relationships",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "object",
                        "description": "Context content to store",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["design", "decision", "trace", "sprint", "log", "test"],
                        "description": "Type of context",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Additional metadata",
                    },
                    "relationships": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "target": {"type": "string"},
                                "type": {"type": "string"},
                            },
                        },
                        "description": "Graph relationships to create",
                    },
                },
                "required": ["content", "type"],
            },
        ),
        Tool(
            name="retrieve_context",
            description="Retrieve context using hybrid vector and graph search",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "type": {
                        "type": "string",
                        "default": "all",
                        "description": "Context type filter",
                    },
                    "search_mode": {
                        "type": "string",
                        "enum": ["vector", "graph", "hybrid"],
                        "default": "hybrid",
                        "description": "Search strategy",
                    },
                    "retrieval_mode": {
                        "type": "string",
                        "enum": ["vector", "graph", "hybrid"],
                        "default": "hybrid",
                        "description": "Retrieval mode for hybrid vector and graph search",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 10,
                        "description": "Maximum results to return",
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["timestamp", "relevance"],
                        "default": "timestamp",
                        "description": "Sort results by timestamp (most recent first) or relevance score",
                    },
                    "max_hops": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "default": 2,
                        "description": "Maximum hops for graph traversal",
                    },
                    "relationship_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter relationships by type (e.g., ['LINK', 'REFERENCES'])",
                    },
                    "include_reasoning_path": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include reasoning path in results",
                    },
                    "enable_community_detection": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include community detection results",
                    },
                    "metadata_filters": {
                        "type": "object",
                        "description": "Filter results by metadata key-value pairs (e.g., {'project': 'api-v2', 'priority': 'high'})",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="query_graph",
            description="Execute read-only Cypher queries on the graph database",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Cypher query (read-only)",
                    },
                    "parameters": {"type": "object", "description": "Query parameters"},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 100,
                        "description": "Maximum results to return",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="update_scratchpad",
            description="Update agent scratchpad with transient storage and TTL support",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "description": "Agent identifier"},
                    "key": {"type": "string", "description": "Scratchpad data key (not secret)"},
                    "content": {"type": "string", "description": "Content to store"},
                    "mode": {
                        "type": "string",
                        "enum": ["overwrite", "append"],
                        "default": "overwrite",
                        "description": "Update mode",
                    },
                    "ttl": {
                        "type": "integer",
                        "minimum": 60,
                        "maximum": 86400,
                        "default": 3600,
                        "description": "Time to live in seconds (1 hour default)",
                    },
                },
                "required": ["agent_id", "key", "content"],
            },
        ),
        Tool(
            name="get_agent_state",
            description="Retrieve agent state with namespace isolation",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "description": "Agent identifier"},
                    "key": {
                        "type": "string",
                        "description": "State data key (optional for all, not secret)",
                    },
                    "prefix": {
                        "type": "string",
                        "enum": ["state", "scratchpad", "memory", "config"],
                        "default": "state",
                        "description": "State type to retrieve",
                    },
                },
                "required": ["agent_id"],
            },
        ),
        Tool(
            name="detect_communities",
            description="Detect communities in the knowledge graph using GraphRAG",
            inputSchema={
                "type": "object",
                "properties": {
                    "algorithm": {
                        "type": "string",
                        "enum": ["louvain", "leiden", "modularity"],
                        "default": "louvain",
                        "description": "Community detection algorithm",
                    },
                    "min_community_size": {
                        "type": "integer",
                        "minimum": 2,
                        "maximum": 50,
                        "default": 3,
                        "description": "Minimum size for communities",
                    },
                    "resolution": {
                        "type": "number",
                        "minimum": 0.1,
                        "maximum": 2.0,
                        "default": 1.0,
                        "description": "Resolution parameter for community detection",
                    },
                    "include_members": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include community member details",
                    },
                },
            },
        ),
        Tool(
            name="select_tools",
            description="Select relevant tools based on query using multiple selection algorithms",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query or context description for tool selection",
                    },
                    "max_tools": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5,
                        "description": "Maximum number of tools to return (≤ N)",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["dot_product", "rule_based", "hybrid"],
                        "default": "hybrid",
                        "description": "Selection method to use",
                    },
                    "category_filter": {
                        "type": "string",
                        "description": "Filter tools by category (optional)",
                    },
                    "include_scores": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include relevance scores in response",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="list_available_tools",
            description="List all available tools in the tool selector system",
            inputSchema={
                "type": "object",
                "properties": {
                    "category_filter": {
                        "type": "string",
                        "description": "Filter tools by category (optional)",
                    },
                    "include_metadata": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include tool metadata in response",
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["name", "category", "keywords"],
                        "default": "name",
                        "description": "Sort tools by specified field",
                    },
                },
            },
        ),
        Tool(
            name="get_tool_info",
            description="Get detailed information about a specific tool",
            inputSchema={
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "Name of the tool to get information about",
                    },
                    "include_usage_examples": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include usage examples if available",
                    },
                },
                "required": ["tool_name"],
            },
        ),
        Tool(
            name="store_fact",
            description="Store a structured fact with automatic extraction and validation",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text containing facts to extract and store",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Agent/namespace identifier for isolation",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User identifier for fact ownership",
                    },
                    "source_turn_id": {
                        "type": "string",
                        "description": "Source conversation turn identifier",
                    },
                    "attribute": {
                        "type": "string",
                        "description": "Specific fact attribute (optional, for manual storage)",
                    },
                    "value": {
                        "type": "string",
                        "description": "Fact value (optional, for manual storage)",
                    },
                },
                "required": ["text", "namespace", "user_id"],
            },
        ),
        Tool(
            name="retrieve_fact",
            description="Retrieve stored facts using intent classification and deterministic lookup",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Fact query (e.g., 'What's my name?')",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Agent/namespace identifier",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User identifier",
                    },
                    "attribute": {
                        "type": "string",
                        "description": "Specific fact attribute to retrieve (optional)",
                    },
                    "fallback_to_context": {
                        "type": "boolean",
                        "default": True,
                        "description": "Fall back to context search if fact not found",
                    },
                },
                "required": ["query", "namespace", "user_id"],
            },
        ),
        Tool(
            name="list_user_facts",
            description="List all stored facts for a user",
            inputSchema={
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "description": "Agent/namespace identifier",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User identifier",
                    },
                    "include_history": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include fact update history",
                    },
                },
                "required": ["namespace", "user_id"],
            },
        ),
        Tool(
            name="delete_user_facts",
            description="Delete all facts for a user (forget-me functionality)",
            inputSchema={
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "description": "Agent/namespace identifier",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User identifier",
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": "Confirmation required for deletion",
                    },
                },
                "required": ["namespace", "user_id", "confirm"],
            },
        ),
        Tool(
            name="classify_intent",
            description="Classify query intent for debugging and development",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query text to classify",
                    },
                    "include_explanation": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include detailed classification explanation",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="delete_context",
            description="Delete a stored context by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "context_id": {
                        "type": "string",
                        "description": "ID of the context to delete",
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": "Confirmation that you want to delete this context",
                    },
                },
                "required": ["context_id", "confirm"],
            },
        ),
        Tool(
            name="list_context_types",
            description="Get available context types and their descriptions",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_descriptions": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include detailed descriptions of each context type",
                    },
                },
            },
        ),
        Tool(
            name="redis_get",
            description="Get a value from Redis by key",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Redis key to retrieve",
                    },
                },
                "required": ["key"],
            },
        ),
        Tool(
            name="redis_set",
            description="Set a key-value pair in Redis",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Redis key to set",
                    },
                    "value": {
                        "type": "string",
                        "description": "Value to store",
                    },
                    "ex": {
                        "type": "integer",
                        "description": "Expiration time in seconds (optional)",
                    },
                },
                "required": ["key", "value"],
            },
        ),
        Tool(
            name="redis_hget",
            description="Get a field value from a Redis hash",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Redis hash key",
                    },
                    "field": {
                        "type": "string",
                        "description": "Hash field name",
                    },
                },
                "required": ["key", "field"],
            },
        ),
        Tool(
            name="redis_hset",
            description="Set a field value in a Redis hash",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Redis hash key",
                    },
                    "field": {
                        "type": "string",
                        "description": "Hash field name",
                    },
                    "value": {
                        "type": "string",
                        "description": "Field value to set",
                    },
                },
                "required": ["key", "field", "value"],
            },
        ),
        Tool(
            name="redis_lpush",
            description="Push an element to the head of a Redis list",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Redis list key",
                    },
                    "value": {
                        "type": "string",
                        "description": "Value to push to list",
                    },
                },
                "required": ["key", "value"],
            },
        ),
        Tool(
            name="redis_lrange",
            description="Get a range of elements from a Redis list",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Redis list key",
                    },
                    "start": {
                        "type": "integer",
                        "default": 0,
                        "description": "Start index (0-based)",
                    },
                    "stop": {
                        "type": "integer",
                        "default": -1,
                        "description": "Stop index (-1 for end)",
                    },
                },
                "required": ["key"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(
    name: str, arguments: Dict[str, Any]
) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """Handle tool calls with metrics tracking."""
    start_time = time.time()
    status_code = 200
    error_msg = None
    
    try:
        if name == "store_context":
            result = await store_context_tool(arguments)
        elif name == "retrieve_context":
            result = await retrieve_context_tool(arguments)
        elif name == "query_graph":
            result = await query_graph_tool(arguments)
        elif name == "update_scratchpad":
            result = await update_scratchpad_tool(arguments)
        elif name == "get_agent_state":
            result = await get_agent_state_tool(arguments)
        elif name == "detect_communities":
            result = await detect_communities_tool(arguments)
        elif name == "select_tools":
            result = await select_tools_tool(arguments)
        elif name == "list_available_tools":
            result = await list_available_tools_tool(arguments)
        elif name == "get_tool_info":
            result = await get_tool_info_tool(arguments)
        elif name == "store_fact":
            result = await store_fact_tool(arguments)
        elif name == "retrieve_fact":
            result = await retrieve_fact_tool(arguments)
        elif name == "list_user_facts":
            result = await list_user_facts_tool(arguments)
        elif name == "delete_user_facts":
            result = await delete_user_facts_tool(arguments)
        elif name == "classify_intent":
            result = await classify_intent_tool(arguments)
        elif name == "delete_context":
            result = await delete_context_tool(arguments)
        elif name == "list_context_types":
            result = await list_context_types_tool(arguments)
        elif name == "redis_get":
            result = await redis_get_tool(arguments)
        elif name == "redis_set":
            result = await redis_set_tool(arguments)
        elif name == "redis_hget":
            result = await redis_hget_tool(arguments)
        elif name == "redis_hset":
            result = await redis_hset_tool(arguments)
        elif name == "redis_lpush":
            result = await redis_lpush_tool(arguments)
        elif name == "redis_lrange":
            result = await redis_lrange_tool(arguments)
        else:
            status_code = 400  # Client error for unknown tool
            error_msg = f"Unknown tool: {name}"
            raise ValueError(error_msg)
        
        # Check if the result indicates failure (client error)
        if isinstance(result, dict) and not result.get("success", True):
            status_code = 400  # Client error for tool validation failures
            error_msg = result.get("error", "Tool execution failed")
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except ValueError as e:
        # Client errors (bad input, unknown tool, validation failures)
        if status_code == 200:
            status_code = 400
        error_msg = str(e)
        raise
    except Exception as e:
        # Server errors (internal failures, database issues)
        if status_code == 200:
            status_code = 500
            error_msg = str(e)
        raise
    finally:
        # Record metrics for the MCP tool call
        duration_ms = (time.time() - start_time) * 1000
        if metrics_collector:
            await metrics_collector.record_request(
                method="MCP",
                path=f"/tools/{name}",
                status_code=status_code,
                duration_ms=duration_ms,
                error=error_msg
            )


async def store_context_tool(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Store context data with vector embeddings and graph relationships.

    This function stores context data in both vector and graph databases,
    creating embeddings for semantic search and establishing relationships
    for graph traversal. It supports graceful degradation when storage
    backends are unavailable.

    Args:
        arguments: Dictionary containing:
            - content (dict): The context content to store (required)
            - type (str): Context type - one of 'design', 'decision', 'trace', 'sprint', 'log'
            - metadata (dict, optional): Additional metadata for the context
            - relationships (list, optional): List of graph relationships to create
                Each relationship should have 'type' and 'target' keys

    Returns:
        Dict containing:
            - success (bool): Whether the operation succeeded
            - id (str): Unique identifier for the stored context
            - vector_id (str, optional): Vector database ID if stored
            - graph_id (str, optional): Graph database node ID if stored
            - message (str): Success or error message
            - validation_errors (list, optional): Any validation errors

    Example:
        >>> arguments = {
        ...     "content": {"title": "API Design", "description": "REST API specification"},
        ...     "type": "design",
        ...     "metadata": {"author": "developer", "priority": "high"},
        ...     "relationships": [{"type": "implements", "target": "req-001"}]
        ... }
        >>> result = await store_context_tool(arguments)
        >>> print(result["success"])  # True
        >>> print(result["id"])       # ctx_abc123...
    """
    # Rate limiting check
    allowed, rate_limit_msg = await rate_limit_check("store_context")
    if not allowed:
        return {
            "success": False,
            "id": None,
            "message": f"Rate limit exceeded: {rate_limit_msg}",
            "error_type": "rate_limit",
        }

    try:
        # Generate unique ID
        import uuid

        context_id = f"ctx_{uuid.uuid4().hex[:12]}"

        # Validate input arguments with comprehensive security checks
        content = arguments["content"]
        context_type = arguments["type"]
        metadata = arguments.get("metadata", {})
        relationships = arguments.get("relationships", [])

        # Validate content size and structure
        content_str = json.dumps(content, sort_keys=True, ensure_ascii=False)
        content_validation = input_validator.validate_input(content_str, "content")
        if not content_validation.valid:
            return {
                "success": False,
                "message": f"Content validation failed: {content_validation.error}",
                "error_type": content_validation.error,
            }

        # Validate JSON structure if content is dict/list
        if isinstance(content, (dict, list)):
            json_validation = input_validator.validate_json_input(content)
            if not json_validation.valid:
                return {
                    "success": False,
                    "message": f"JSON structure validation failed: {json_validation.error}",
                    "error_type": json_validation.error,
                }

        # Validate context_type
        type_validation = input_validator.validate_input(context_type, "query")
        if not type_validation.valid:
            return {
                "success": False,
                "message": f"Type validation failed: {type_validation.error}",
                "error_type": type_validation.error,
            }

        # Generate Q&A pairs for enhanced fact retrieval (Phase 2 enhancement)
        qa_pairs_generated = []
        stitched_units = []
        try:
            # Extract text content for Q&A generation
            if isinstance(content, dict):
                # Try to extract meaningful text from various fields
                text_fields = []
                for key, value in content.items():
                    if isinstance(value, str) and len(value.strip()) > 10:
                        text_fields.append(value.strip())
                
                combined_text = " ".join(text_fields)
                
                if combined_text and len(combined_text) > 20:
                    # Generate Q&A pairs from the content
                    qa_pairs = qa_generator.generate_qa_pairs_from_statement(combined_text)
                    qa_pairs_generated = qa_pairs
                    
                    # Create stitched units for joint indexing
                    for qa_pair in qa_pairs:
                        stitched_unit = qa_generator.create_stitched_unit(qa_pair)
                        stitched_units.append(stitched_unit)
                    
                    logger.info(f"Generated {len(qa_pairs)} Q&A pairs and {len(stitched_units)} stitched units")
            
            elif isinstance(content, str) and len(content.strip()) > 20:
                # Direct string content
                qa_pairs = qa_generator.generate_qa_pairs_from_statement(content.strip())
                qa_pairs_generated = qa_pairs
                
                for qa_pair in qa_pairs:
                    stitched_unit = qa_generator.create_stitched_unit(qa_pair)
                    stitched_units.append(stitched_unit)
                
                logger.info(f"Generated {len(qa_pairs)} Q&A pairs and {len(stitched_units)} stitched units")
        
        except Exception as e:
            logger.warning(f"Q&A generation failed: {e}")
            # Continue with normal storage even if Q&A generation fails

        # Store in vector database
        vector_id = None
        if qdrant_client and qdrant_client.client:
            try:
                from qdrant_client.models import PointStruct

                # Generate proper semantic embedding from content
                content_str = json.dumps(content, sort_keys=True, ensure_ascii=False)
                if embedding_generator:
                    embedding = await embedding_generator.generate_embedding(content_str)
                    logger.info("Generated semantic embedding using configured model")
                else:
                    # Fallback to improved hash-based embedding for development
                    logger.warning("No embedding generator available, using fallback method")
                    import hashlib

                    hash_obj = hashlib.sha256(content_str.encode())
                    hash_bytes = hash_obj.digest()

                    # Get dimensions from Qdrant config or default
                    dimensions = qdrant_client.config.get("qdrant", {}).get(
                        "dimensions", Config.EMBEDDING_DIMENSIONS
                    )
                    embedding = []
                    for i in range(dimensions):
                        byte_idx = i % len(hash_bytes)
                        # Normalize to [-1, 1] for better vector space properties
                        normalized_value = (float(hash_bytes[byte_idx]) / 255.0) * 2.0 - 1.0
                        embedding.append(normalized_value)

                # Use the collection name from config or default
                collection_name = qdrant_client.config.get("qdrant", {}).get(
                    "collection_name", "context_embeddings"
                )

                # Store the vector using VectorDBInitializer.store_vector method with circuit breaker
                vector_metadata = {
                    "content": content,
                    "type": context_type,
                    "metadata": metadata,
                }
                try:
                    await with_mcp_circuit_breaker(
                        lambda: qdrant_client.store_vector(
                            vector_id=context_id,
                            embedding=embedding,
                            metadata=vector_metadata,
                        )
                    )
                except (ConnectionError, CircuitBreakerError, asyncio.TimeoutError) as e:
                    logger.error(f"MCP circuit breaker: Vector storage failed - {e}")
                    return create_error_response(
                        ErrorCode.DEPENDENCY_DOWN,
                        f"Vector storage unavailable: {str(e)}",
                        context={"operation": "vector_storage", "context_id": context_id}
                    )
                
                # Verify the vector was actually stored
                try:
                    import time
                    time.sleep(0.1)  # Brief wait for write to complete
                    stored_points = qdrant_client.client.retrieve(
                        collection_name=qdrant_client.collection_name,
                        ids=[context_id],
                        with_payload=False,
                        with_vectors=False
                    )
                    if stored_points and len(stored_points) > 0:
                        vector_id = context_id
                        logger.info(f"✅ Vector storage verified - stored and retrievable: {vector_id}")
                    else:
                        vector_id = None
                        logger.error(f"❌ Vector storage failed - data not retrievable after write: {context_id}")
                except Exception as verify_e:
                    logger.warning(f"Vector storage verification failed: {verify_e}, assuming success")
                    vector_id = context_id
                
                # Store stitched Q&A units as additional vectors (Phase 2 enhancement)
                qa_vector_ids = []
                for i, stitched_unit in enumerate(stitched_units):
                    try:
                        qa_id = f"{context_id}_qa_{i}"
                        qa_content = stitched_unit.content
                        
                        if embedding_generator:
                            qa_embedding = await embedding_generator.generate_embedding(qa_content)
                        else:
                            # Fallback hash-based embedding
                            import hashlib
                            hash_obj = hashlib.sha256(qa_content.encode())
                            hash_bytes = hash_obj.digest()
                            dimensions = qdrant_client.config.get("qdrant", {}).get(
                                "dimensions", Config.EMBEDDING_DIMENSIONS
                            )
                            qa_embedding = []
                            for j in range(dimensions):
                                byte_idx = j % len(hash_bytes)
                                normalized_value = (float(hash_bytes[byte_idx]) / 255.0) * 2.0 - 1.0
                                qa_embedding.append(normalized_value)
                        
                        # Store Q&A unit with enhanced metadata
                        qa_metadata = {
                            "content": qa_content,
                            "type": "qa_pair",
                            "question": stitched_unit.question,
                            "answer": stitched_unit.answer,
                            "fact_attribute": stitched_unit.fact_attribute,
                            "confidence": stitched_unit.confidence,
                            "parent_context_id": context_id,
                            "metadata": stitched_unit.metadata
                        }
                        
                        qdrant_client.store_vector(
                            vector_id=qa_id,
                            embedding=qa_embedding,
                            metadata=qa_metadata,
                        )
                        qa_vector_ids.append(qa_id)
                        
                    except Exception as qa_e:
                        logger.warning(f"Failed to store Q&A unit {i}: {qa_e}")
                
                if qa_vector_ids:
                    logger.info(f"Stored {len(qa_vector_ids)} Q&A vector units")
                    
            except Exception as e:
                logger.error(f"Failed to store vector: {e}")
                vector_id = None

        # Store in graph database
        graph_id = None
        if neo4j_client and neo4j_client.driver:
            try:
                with neo4j_client.driver.session(database=neo4j_client.database) as session:
                    # Create context node
                    create_query = """
                    CREATE (c:Context {
                        id: $id,
                        type: $type,
                        content: $content,
                        metadata: $metadata,
                        created_at: datetime()
                    })
                    RETURN c.id as node_id
                    """
                    result = session.run(
                        create_query,
                        id=context_id,
                        type=context_type,
                        content=json.dumps(content),
                        metadata=json.dumps(metadata),
                    )
                    record = result.single()
                    if record:
                        graph_id = record["node_id"]
                        logger.info(f"Created graph node with ID: {graph_id}")

                        # Create relationships if specified
                        for rel in relationships:
                            try:
                                rel_query = """
                                MATCH (a:Context {id: $from_id})
                                MATCH (b:Context {id: $to_id})
                                CREATE (a)-[r:RELATES_TO {type: $rel_type}]->(b)
                                RETURN r
                                """
                                session.run(
                                    rel_query,
                                    from_id=context_id,
                                    to_id=rel["target"],
                                    rel_type=rel["type"],
                                )
                                logger.info(
                                    f"Created relationship: {context_id} -> "
                                    f"{rel['target']} ({rel['type']})"
                                )
                            except Exception as rel_e:
                                logger.warning(f"Failed to create relationship: {rel_e}")
            except Exception as e:
                logger.error(f"Failed to store in graph database: {e}")
                graph_id = None

        # Build success message with backend status
        backend_status = []
        if vector_id:
            backend_status.append("vector")
        if graph_id:
            backend_status.append("graph")

        # Determine if storage was actually successful
        storage_successful = len(backend_status) > 0  # At least one backend must succeed
        
        if not backend_status:
            failure_response = {
                "success": False,
                "id": None,  # Primary field expected by MCP contract
                "context_id": None,  # Backward compatibility
                "vector_id": vector_id,
                "graph_id": graph_id,
                "message": "Storage failed: No backends available or all storage operations failed",
                "backend_status": {
                    "vector": "failed",
                    "graph": "failed",
                },
                "error_type": "storage_failure"
            }
            
            # Validate response before returning
            validation_result = validate_mcp_response("store_context", failure_response, log_results=True)
            if not validation_result:
                logger.error(f"store_context failure response failed validation: {[e.message for e in validation_result.errors]}")
            
            return failure_response
        elif len(backend_status) == 2:
            message = "Context stored successfully in all backends"
        else:
            available = ", ".join(backend_status)
            failed = "graph" if vector_id and not graph_id else "vector"
            message = (
                f"Context stored successfully in {available} (warning: {failed} backend failed)"
            )

        success_response = {
            "success": True,
            "id": context_id,  # Primary field expected by MCP contract
            "context_id": context_id,  # Backward compatibility
            "vector_id": vector_id,
            "graph_id": graph_id,
            "message": message,
            "backend_status": {
                "vector": "success" if vector_id else "failed",
                "graph": "success" if graph_id else "failed",
            },
            "qa_enhancement": {
                "qa_pairs_generated": len(qa_pairs_generated),
                "stitched_units_stored": len(stitched_units),
                "fact_attributes": [pair.fact_attribute for pair in qa_pairs_generated] if qa_pairs_generated else []
            }
        }
        
        # Validate response before returning
        validation_result = validate_mcp_response("store_context", success_response, log_results=True)
        if not validation_result:
            logger.error(f"store_context success response failed validation: {[e.message for e in validation_result.errors]}")
        
        return success_response

    except Exception as e:
        logger.error(f"Error storing context: {e}")
        error_response = {
            "success": False,
            "id": None,  # Primary field expected by MCP contract
            "context_id": None,  # Backward compatibility
            "message": f"Failed to store context: {str(e)}",
            "error_type": "exception"
        }
        
        # Validate response before returning
        validation_result = validate_mcp_response("store_context", error_response, log_results=True)
        if not validation_result:
            logger.error(f"store_context error response failed validation: {[e.message for e in validation_result.errors]}")
        
        return error_response


async def retrieve_context_tool(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve context using hybrid vector and graph search.

    This function performs semantic search across stored contexts using
    vector similarity and graph traversal. It supports multiple search modes
    and filters to find relevant context information.

    Args:
        arguments: Dictionary containing:
            - query (str): Search query text (required)
            - type (str, optional): Filter by context type ('design', 'decision', etc.)
                Default: 'all' (no filtering)
            - search_mode (str, optional): Search strategy - 'vector', 'graph', or 'hybrid'
                Default: 'hybrid'
            - retrieval_mode (str, optional): Retrieval mode for hybrid search - 'vector', 'graph', or 'hybrid'
                Default: 'hybrid'
            - limit (int, optional): Maximum number of results to return (1-100)
                Default: 10
            - sort_by (str, optional): Sort results by 'timestamp' or 'relevance'
                Default: 'timestamp' (most recent first)

    Returns:
        Dict containing:
            - success (bool): Whether the search succeeded
            - results (list): List of matching context objects
            - total_count (int): Total number of results found
            - search_mode_used (str): The search mode that was actually used
            - retrieval_mode_used (str): The retrieval mode that was actually used
            - message (str): Success or error message

    Example:
        >>> arguments = {
        ...     "query": "API authentication design",
        ...     "type": "design",
        ...     "search_mode": "hybrid",
        ...     "retrieval_mode": "hybrid",
        ...     "limit": 5
        ... }
        >>> result = await retrieve_context_tool(arguments)
        >>> print(len(result["results"]))  # Number of matching contexts
        >>> for ctx in result["results"]:
        ...     print(ctx["content"]["title"])  # Context titles
    """
    # Rate limiting check
    allowed, rate_limit_msg = await rate_limit_check("retrieve_context")
    if not allowed:
        return {
            "success": False,
            "results": [],
            "message": f"Rate limit exceeded: {rate_limit_msg}",
            "error_type": "rate_limit",
        }

    try:
        # Extract and validate query input
        query = arguments["query"]
        context_type = arguments.get("type", "all")
        search_mode = arguments.get("search_mode", "hybrid")
        retrieval_mode = arguments.get("retrieval_mode", "hybrid")
        limit = arguments.get("limit", 10)
        sort_by = arguments.get("sort_by", "timestamp")  # Default to timestamp since relevance was broken
        
        # Validate sort_by parameter
        if sort_by not in ["timestamp", "relevance"]:
            return {
                "success": False,
                "results": [],
                "message": f"Invalid sort_by value: '{sort_by}'. Must be 'timestamp' or 'relevance'",
                "error_type": "invalid_parameter",
            }

        # Validate query with comprehensive security checks
        query_validation = input_validator.validate_input(query, "query")
        if not query_validation.valid:
            return {
                "success": False,
                "results": [],
                "message": f"Query validation failed: {query_validation.error}",
                "error_type": query_validation.error,
            }

        # Validate context_type if provided
        if context_type != "all":
            type_validation = input_validator.validate_input(context_type, "query")
            if not type_validation.valid:
                return {
                    "success": False,
                    "results": [],
                    "message": f"Type validation failed: {type_validation.error}",
                    "error_type": type_validation.error,
                }

        # Validate limit parameter
        if not isinstance(limit, int) or limit < 1 or limit > 100:
            return {
                "success": False,
                "results": [],
                "message": "Limit must be an integer between 1 and 100",
                "error_type": "invalid_parameter",
            }

        # Phase 2 & 3 Enhancement: Query analysis and expansion
        original_query = query
        enhanced_queries = [query]  # Start with original query
        intent_result = None
        query_expansion_metadata = {}
        
        try:
            # Classify query intent
            intent_result = intent_classifier.classify(query)
            logger.debug(f"Query intent: {intent_result.intent.value}, attribute: {intent_result.attribute}")
            
            # Generate query rewrites for better recall
            if intent_result.intent == IntentType.FACT_LOOKUP:
                rewritten_queries = query_rewriter.rewrite_fact_query(query, intent_result.attribute)
                additional_queries = [rq.query for rq in rewritten_queries]
                enhanced_queries.extend(additional_queries)
                logger.info(f"Generated {len(additional_queries)} query variants for fact lookup")
            
            # Phase 3: Graph-enhanced query expansion
            if graph_query_expander:
                try:
                    expansion_result = graph_query_expander.expand_query(query)
                    graph_expanded_queries = [eq.query for eq in expansion_result.expanded_queries]
                    enhanced_queries.extend(graph_expanded_queries)
                    
                    # Store expansion metadata for response
                    query_expansion_metadata = {
                        'strategy_used': expansion_result.expansion_metadata.get('strategy_used', 'none'),
                        'entities_found': expansion_result.expansion_metadata.get('entities_found', 0),
                        'graph_expansions': len(graph_expanded_queries),
                        'detected_entities': expansion_result.extracted_entities
                    }
                    
                    logger.info(f"Generated {len(graph_expanded_queries)} graph-enhanced query expansions")
                except Exception as graph_exp_error:
                    logger.warning(f"Graph query expansion failed: {graph_exp_error}")
            
            # Limit total queries to avoid performance issues
            enhanced_queries = enhanced_queries[:6]  # Original + up to 5 variants (fact + graph)
            
        except Exception as e:
            logger.warning(f"Query enhancement failed: {e}")
            # Continue with original query

        # New GraphRAG parameters
        max_hops = arguments.get("max_hops", 2)
        relationship_types = arguments.get("relationship_types", None)
        include_reasoning_path = arguments.get("include_reasoning_path", False)
        enable_community_detection = arguments.get("enable_community_detection", False)
        metadata_filters = arguments.get("metadata_filters", {})

        # Use retrieval_mode if specified, otherwise fall back to search_mode
        effective_mode = retrieval_mode

        results = []

        if effective_mode in ["vector", "hybrid"] and qdrant_client and qdrant_client.client:
            try:
                # Phase 2 Enhancement: Multi-query vector search with fact-aware ranking
                all_vector_results = []
                
                for enhanced_query in enhanced_queries:
                    # Generate proper semantic embedding for each query variant
                    if embedding_generator:
                        query_vector = await embedding_generator.generate_embedding(enhanced_query)
                        logger.debug(f"Generated embedding for query: {enhanced_query[:50]}...")
                    else:
                        # Fallback to hash-based query embedding for development
                        import hashlib
                        query_hash = hashlib.sha256(enhanced_query.encode()).digest()
                        dimensions = qdrant_client.config.get("qdrant", {}).get(
                            "dimensions", Config.EMBEDDING_DIMENSIONS
                        )
                        query_vector = []
                        for i in range(dimensions):
                            byte_idx = i % len(query_hash)
                            normalized_value = (float(query_hash[byte_idx]) / 255.0) * 2.0 - 1.0
                            query_vector.append(normalized_value)

                    # Search with each query variant - run async for better performance
                    collection_name = qdrant_client.config.get("qdrant", {}).get(
                        "collection_name", "context_embeddings"
                    )

                    # Build Qdrant filter from metadata_filters and context_type
                    filter_dict = None
                    if metadata_filters or context_type != "all":
                        filter_conditions = []
                        
                        # Add context type filter
                        if context_type != "all":
                            filter_conditions.append({
                                "key": "type",
                                "match": {"value": context_type}
                            })
                        
                        # Add metadata filters - strict matching
                        for key, value in metadata_filters.items():
                            # Support for nested metadata fields
                            metadata_key = f"metadata.{key}"
                            filter_conditions.append({
                                "key": metadata_key,
                                "match": {"value": value}
                            })
                        
                        if filter_conditions:
                            filter_dict = {"must": filter_conditions}
                    
                    # Run vector search asynchronously to avoid blocking
                    variant_results = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: qdrant_client.search(
                            query_vector=query_vector,
                            limit=limit * 2,  # Get more results for better ranking
                            filter_dict=filter_dict,
                        )
                    )
                    
                    # Add query source tracking
                    for result in variant_results:
                        result["query_variant"] = enhanced_query
                        result["is_original_query"] = enhanced_query == original_query
                    
                    all_vector_results.extend(variant_results)

                # Apply fact-aware ranking to combined results
                if all_vector_results:
                    # Convert to format expected by fact ranker
                    ranking_input = []
                    for result in all_vector_results:
                        content = ""
                        if "content" in result.get("payload", {}):
                            content = str(result["payload"]["content"])
                        elif "payload" in result:
                            content = str(result["payload"])
                        
                        ranking_input.append({
                            "content": content,
                            "score": result["score"],
                            "metadata": result.get("payload", {})
                        })
                    
                    # Apply hybrid scoring if available, otherwise fall back to fact-aware ranking
                    if hybrid_scorer:
                        # Classify intent for scoring mode selection
                        intent_result = intent_classifier.classify(original_query)
                        
                        # Apply hybrid scoring with graph integration
                        hybrid_scores = hybrid_scorer.compute_hybrid_score(
                            query=original_query,
                            results=ranking_input,
                            intent=intent_result.intent
                        )
                        
                        # Convert hybrid scores back to ranking results format
                        ranked_results = []
                        for hybrid_score in hybrid_scores:
                            # Find the original ranking input item
                            result_id = hybrid_score.metadata.get('result_id', '')
                            original_item = None
                            for item in ranking_input:
                                if item.get('id', str(ranking_input.index(item))) == result_id:
                                    original_item = item
                                    break
                            
                            if original_item:
                                # Create a ranking result-like object
                                from types import SimpleNamespace
                                ranking_result = SimpleNamespace()
                                ranking_result.content = original_item['content']
                                ranking_result.final_score = hybrid_score.final_score
                                ranking_result.fact_boost = hybrid_score.fact_pattern_score
                                ranking_result.content_type = SimpleNamespace()
                                ranking_result.content_type.value = "hybrid_enhanced"
                                ranking_result.matched_patterns = [hybrid_score.explanation]
                                ranking_result.metadata = {**original_item.get('metadata', {}), **hybrid_score.metadata}
                                ranked_results.append(ranking_result)
                        
                        logger.info(f"Applied hybrid scoring to {len(ranked_results)} results")
                    else:
                        # Fallback to fact-aware ranking only
                        # Apply fact ranking asynchronously for large result sets
                        ranked_results = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: fact_ranker.apply_fact_ranking(ranking_input, original_query)
                        )
                    
                    # Convert back and deduplicate by ID
                    seen_ids = set()
                    for i, ranked_result in enumerate(ranked_results):
                        if i >= limit:  # Respect the limit
                            break
                        
                        # Find corresponding original result
                        original_result = all_vector_results[ranking_input.index({
                            "content": ranked_result.content,
                            "score": ranked_result.original_score,
                            "metadata": ranked_result.metadata
                        })]
                        
                        result_id = original_result["id"]
                        if result_id not in seen_ids:
                            seen_ids.add(result_id)
                            results.append({
                                "id": result_id,
                                "score": ranked_result.final_score,
                                "original_score": ranked_result.original_score,
                                "fact_boost": ranked_result.fact_boost,
                                "content_type": ranked_result.content_type.value,
                                "source": "vector_enhanced",
                                "payload": original_result.get("payload", {}),
                                "query_variant": original_result.get("query_variant", ""),
                                "ranking_patterns": ranked_result.matched_patterns
                            })
                    
                    logger.info(f"Applied fact-aware ranking to {len(all_vector_results)} vector results, returned {len(results)}")
                else:
                    logger.warning("No vector results found for enhanced queries")

                logger.info(f"Enhanced vector search completed")
            except Exception as e:
                logger.error(f"Vector search failed: {e}")

        if effective_mode in ["graph", "hybrid"] and neo4j_client and neo4j_client.driver:
            try:
                # Perform graph search
                with neo4j_client.driver.session(database=neo4j_client.database) as session:
                    # Enhanced graph search with configurable hop distance and relationship filtering
                    if effective_mode == "hybrid" or max_hops > 1:
                        # Build relationship type filter
                        rel_filter = ""
                        if relationship_types:
                            rel_types = "|".join(relationship_types)
                            rel_filter = f":{rel_types}"
                        else:
                            rel_filter = ""  # Allow all relationship types

                        # Multi-hop traversal query with configurable parameters
                        # Now includes hop distance for proper scoring
                        # Performance Note: ORDER BY hop_distance may impact performance on large graphs.
                        # Consider these optimizations if query performance degrades:
                        # 1. Create index: CREATE INDEX FOR (n:Context) ON (n.created_at)
                        # 2. Limit max_hops to 2-3 for most use cases
                        # 3. Use SKIP/LIMIT pagination for large result sets
                        # 4. Consider caching frequently accessed paths
                        cypher_query = f"""
                        MATCH path = (n:Context)-[r{rel_filter}*1..{max_hops}]->(m)
                        WHERE (n.type = $type OR $type = 'all')
                        AND (n.content CONTAINS $query OR n.metadata CONTAINS $query OR
                             m.content CONTAINS $query OR m.metadata CONTAINS $query)
                        RETURN DISTINCT m.id as id, m.type as type, m.content as content,
                               m.metadata as metadata, m.created_at as created_at,
                               length(path) as hop_distance
                        ORDER BY hop_distance ASC
                        LIMIT $limit
                        """
                    else:
                        # Standard graph search for single-hop modes
                        metadata_conditions = []
                        if metadata_filters:
                            for key, value in metadata_filters.items():
                                metadata_conditions.append(f"n.metadata.{key} = ${key}")
                        
                        metadata_clause = ""
                        if metadata_conditions:
                            metadata_clause = "AND " + " AND ".join(metadata_conditions)
                        
                        cypher_query = f"""
                        MATCH (n:Context)
                        WHERE (n.type = $type OR $type = 'all')
                        AND (n.content CONTAINS $query OR n.metadata CONTAINS $query)
                        {metadata_clause}
                        RETURN n.id as id, n.type as type, n.content as content,
                               n.metadata as metadata, n.created_at as created_at
                        LIMIT $limit
                        """
                    # Prepare parameters including metadata filters
                    parameters = {
                        "type": context_type, 
                        "query": query, 
                        "limit": limit
                    }
                    # Add metadata filter values to parameters
                    if metadata_filters:
                        parameters.update(metadata_filters)
                    
                    # Run graph query asynchronously to avoid blocking
                    query_result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: neo4j_client.query(
                            cypher_query,
                            parameters=parameters,
                        )
                    )

                    for record in query_result:
                        try:
                            content = json.loads(record["content"]) if record["content"] else {}
                            metadata = json.loads(record["metadata"]) if record["metadata"] else {}
                        except json.JSONDecodeError:
                            content = record["content"]
                            metadata = record["metadata"]

                        # Calculate score based on hop distance (closer = higher score)
                        # Score ranges from 1.0 (direct connection) to lower values for distant nodes
                        hop_distance = record.get("hop_distance", 1)
                        graph_score = 1.0 / (hop_distance + 0.5)  # +0.5 to avoid division issues and smooth the curve

                        results.append(
                            {
                                "id": record["id"],
                                "type": record["type"],
                                "content": content,
                                "metadata": metadata,
                                "created_at": (
                                    str(record["created_at"]) if record["created_at"] else None
                                ),
                                "score": graph_score,
                                "hop_distance": hop_distance,
                                "source": "graph",
                            }
                        )

                graph_results_count = len([r for r in results if r.get("source") == "graph"])
                logger.info(f"Graph search returned {graph_results_count} results")
            except Exception as e:
                logger.error(f"Graph search failed: {e}")

        # Apply search enhancements before sorting (Phase 1 improvements)
        # These enhancements address workspace 001's identified issues:
        # 1. Exact match boosting for filenames
        # 2. Context type weighting to prioritize code over conversations
        # 3. Recency balancing to prevent new content from overshadowing older relevant content
        # 4. Technical term recognition for better code discovery
        
        # Determine if this is a technical query
        technical_query = is_technical_query(query)
        
        # Apply enhancements with all features enabled
        # These can be made configurable via arguments in the future
        if results:
            logger.info(f"Applying search enhancements to {len(results)} results (technical_query={technical_query})")
            results = apply_search_enhancements(
                results=results,
                query=query,
                enable_exact_match=True,  # Phase 1: Exact match boosting
                enable_type_weighting=True,  # Phase 1: Context type weighting
                enable_recency_decay=True,  # Phase 2: Recency balancing
                enable_technical_boost=technical_query  # Phase 2: Only boost technical terms for technical queries
            )
            logger.info(f"Search enhancements applied, top result score: {results[0].get('enhanced_score', 0):.3f}")
            
            # Phase 3: Apply query-specific relevance scoring to address LIM-004
            # This ensures results vary meaningfully based on actual search terms
            try:
                query_scorer = QueryRelevanceScorer()
                results = query_scorer.enhance_search_results(original_query, results)
                logger.info(f"Query-specific relevance scoring applied, top result score: {results[0].get('query_relevance_score', 0):.3f}")
            except Exception as e:
                logger.warning(f"Query relevance scoring failed: {e}")
                # Continue with basic enhanced results
        
        # Apply sorting based on sort_by parameter
        if sort_by == "timestamp":
            # Sort by created_at timestamp (most recent first)
            results.sort(
                key=lambda x: x.get("created_at", "") or "",
                reverse=True  # Most recent first
            )
            logger.info(f"Applied timestamp sorting to {len(results)} results")
        elif sort_by == "relevance":
            # Sort by enhanced score if available, otherwise original score
            results.sort(
                key=lambda x: x.get("enhanced_score", x.get("score", 0)),
                reverse=True  # Highest score first
            )
            logger.info(f"Applied relevance sorting to {len(results)} enhanced results")
        
        # Enhance results with GraphRAG features if requested
        enhanced_results = results[:limit]
        graphrag_metadata = {
            "max_hops_used": max_hops,
            "relationship_types_used": relationship_types,
            "reasoning_path_included": include_reasoning_path,
            "community_detection_enabled": enable_community_detection,
        }

        # Add reasoning path if requested
        if include_reasoning_path and enhanced_results:
            # Simple reasoning path implementation - can be enhanced with full GraphRAG integration
            for result in enhanced_results:
                result[
                    "reasoning_path"
                ] = f"Found via {result.get('source', 'unknown')} search with {max_hops}-hop traversal"

        # Add community detection placeholder - will be implemented in community detection tool
        if enable_community_detection:
            graphrag_metadata[
                "community_info"
            ] = "Community detection available via detect_communities tool"
        
        # Apply strict metadata filtering as post-processing
        # This ensures exact matching regardless of which backend was used
        if metadata_filters and enhanced_results:
            filtered_results = []
            for result in enhanced_results:
                # Fix: Access metadata from correct nesting level in payload structure
                # User metadata is stored at payload.metadata, not at top level
                metadata = result.get('payload', {}).get('metadata', {})

                # Check if all metadata filters match exactly
                match = True
                for filter_key, filter_value in metadata_filters.items():
                    if metadata.get(filter_key) != filter_value:
                        match = False
                        break

                if match:
                    filtered_results.append(result)

            enhanced_results = filtered_results
            logger.info(f"Strict metadata filtering applied: {len(enhanced_results)} results remaining")

        # Apply reranking to improve precision
        reranker_config = qdrant_client.config.get("reranker", {}) if qdrant_client else {}
        reranker = get_reranker(reranker_config)
        if reranker.enabled and enhanced_results:
            logger.info(f"Applying reranker to {len(enhanced_results)} results")
            reranked_results = reranker.rerank(query, enhanced_results)
            
            # Add reranker metadata
            graphrag_metadata["reranker"] = {
                "enabled": True,
                "model": reranker.model_name,
                "original_count": len(enhanced_results),
                "reranked_count": len(reranked_results),
                "top_k": reranker.top_k,
                "return_k": reranker.return_k
            }
            enhanced_results = reranked_results
        else:
            graphrag_metadata["reranker"] = {"enabled": False}

        return {
            "success": True,
            "results": enhanced_results,
            "total_count": len(results),
            "search_mode_used": search_mode,
            "retrieval_mode_used": effective_mode,
            "graphrag_metadata": graphrag_metadata,
            "message": f"Found {len(enhanced_results)} matching contexts using GraphRAG-enhanced {'reranked ' if reranker.enabled else ''}search",
            "phase2_enhancements": {
                "intent_detected": intent_result.intent.value if intent_result else "unknown",
                "fact_attribute": intent_result.attribute if intent_result else None,
                "query_variants_used": len(enhanced_queries),
            },
            "phase3_enhancements": {
                "graph_expansion_enabled": graph_query_expander is not None,
                "hybrid_scoring_enabled": hybrid_scorer is not None,
                "graph_expansion_metadata": query_expansion_metadata,
                "original_query": original_query,
                "enhanced_queries": enhanced_queries[:3] if len(enhanced_queries) > 1 else [],
                "fact_aware_ranking_applied": any("fact_boost" in r for r in enhanced_results)
            },
            "search_enhancements": {
                "technical_query_detected": technical_query,
                "exact_match_boosting": True,
                "context_type_weighting": True,
                "recency_balancing": True,
                "technical_term_boosting": technical_query,
                "top_result_boosts": enhanced_results[0].get("score_boosts", {}) if enhanced_results else {}
            }
        }

    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return {
            "success": False,
            "results": [],
            "message": f"Failed to retrieve context: {str(e)}",
        }


async def query_graph_tool(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute read-only Cypher queries on the graph database.

    This function allows executing custom Cypher queries for advanced
    graph analysis and traversal. For security, only read operations
    are permitted - write operations like CREATE, DELETE, SET are blocked.

    Args:
        arguments: Dictionary containing:
            - query (str): Cypher query to execute (required)
                Must be read-only (no CREATE/DELETE/SET/REMOVE/MERGE/DROP)
            - parameters (dict, optional): Query parameters for parameterized queries
                Default: {}
            - limit (int, optional): Maximum number of results to return (1-1000)
                Default: 100

    Returns:
        Dict containing:
            - success (bool): Whether the query succeeded
            - results (list): Query results as list of dictionaries
            - row_count (int): Number of rows returned
            - error (str, optional): Error message if query failed

    Security:
        - Only read operations are allowed (MATCH, RETURN, WHERE, etc.)
        - Write operations (CREATE, DELETE, SET, REMOVE, MERGE, DROP) are blocked
        - Query validation prevents SQL injection-style attacks

    Example:
        >>> arguments = {
        ...     "query": "MATCH (n:Context)-[:IMPLEMENTS]->(r:Requirement) "
        ...              "WHERE n.type = $type RETURN n.title, r.id",
        ...     "parameters": {"type": "design"},
        ...     "limit": 20
        ... }
        >>> result = await query_graph_tool(arguments)
        >>> for row in result["results"]:
        ...     print(f"Design: {row['n.title']} implements {row['r.id']}")
    """
    # Rate limiting check
    allowed, rate_limit_msg = await rate_limit_check("query_graph")
    if not allowed:
        return {
            "success": False,
            "results": [],
            "message": f"Rate limit exceeded: {rate_limit_msg}",
            "error_type": "rate_limit",
        }

    try:
        query = arguments["query"]
        parameters = arguments.get("parameters", {})
        limit = arguments.get("limit", 100)

        # Enhanced security check with comprehensive validation
        validation_result = cypher_validator.validate_query(query, parameters)
        if not validation_result.is_valid:
            logger.warning(f"Query validation failed: {validation_result.error_message}")
            return {
                "success": False,
                "error": f"Query validation failed: {validation_result.error_message}",
                "error_type": validation_result.error_type,
            }

        # Log warnings if any
        if validation_result.warnings:
            logger.info(f"Query validation warnings: {validation_result.warnings}")

        logger.info(
            f"Query validation passed with complexity score: {validation_result.complexity_score}"
        )

        # Use read-only client for enhanced security
        try:
            from ..storage.neo4j_readonly import get_readonly_client
            readonly_client = get_readonly_client(config.get_all_config() if config else None)
            
            if not readonly_client.connect():
                # Fallback to main client if read-only client unavailable
                if not neo4j_client or not neo4j_client.driver:
                    return {"success": False, "error": "Graph database not available"}
                logger.warning("Read-only client unavailable, using main client")
                results = neo4j_client.query(query, parameters)
            else:
                # Use secure read-only client
                logger.info("Using read-only client for graph query")
                results = readonly_client.query(query, parameters)
                
        except ImportError as e:
            logger.warning(f"Read-only client not available: {e}, using main client")
            if not neo4j_client or not neo4j_client.driver:
                return {"success": False, "error": "Graph database not available"}
            results = neo4j_client.query(query, parameters)
        except Exception as e:
            logger.error(f"Graph query execution failed: {e}")
            return {"success": False, "error": str(e)}

        return {"success": True, "results": results[:limit], "row_count": len(results)}

    except Exception as e:
        logger.error(f"Error executing graph query: {e}")
        return {"success": False, "error": str(e)}


async def update_scratchpad_tool(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update agent scratchpad with transient storage and TTL support.

    This function provides a scratchpad for agents to store temporary data
    with automatic expiration. Each agent has an isolated namespace to prevent
    data leakage between agents.

    Args:
        arguments: Dictionary containing:
            - agent_id (str): Agent identifier (required)
            - key (str): Scratchpad key (required)
            - content (str): Content to store (required)
            - mode (str, optional): Update mode 'overwrite' or 'append' (default: 'overwrite')
            - ttl (int, optional): Time to live in seconds (default: 3600, max: 86400)

    Returns:
        Dict containing:
            - success (bool): Whether the operation succeeded
            - message (str): Success or error message
            - key (str): The namespaced key used for storage
            - ttl (int): TTL applied in seconds

    Example:
        >>> arguments = {
        ...     "agent_id": "agent-123",
        ...     "key": "working_memory",
        ...     "content": "Current task: analyze data",
        ...     "mode": "append",
        ...     "ttl": 7200
        ... }
        >>> result = await update_scratchpad_tool(arguments)
        >>> print(result["success"])  # True
    """
    # Rate limiting check
    allowed, rate_limit_msg = await rate_limit_check("update_scratchpad")
    if not allowed:
        return {
            "success": False,
            "message": f"Rate limit exceeded: {rate_limit_msg}",
            "error_type": "rate_limit",
        }

    try:
        agent_id = arguments["agent_id"]
        key = arguments["key"]
        content = arguments["content"]
        mode = arguments.get("mode", "overwrite")
        ttl = arguments.get("ttl", 3600)  # 1 hour default

        # Validate agent ID and key using namespace manager
        if not agent_namespace.validate_agent_id(agent_id):
            return {
                "success": False,
                "message": f"Invalid agent ID format: {agent_id}",
                "error_type": "invalid_agent_id",
            }

        if not agent_namespace.validate_key(key):
            return {
                "success": False,
                "message": f"Invalid key format: {key}",
                "error_type": "invalid_key",
            }

        # Validate content
        if not isinstance(content, str):
            return {
                "success": False,
                "message": "Content must be a string",
                "error_type": "invalid_content_type",
            }

        # Use comprehensive input validation
        content_validation = input_validator.validate_input(content, "content")
        if not content_validation.valid:
            error_messages = {
                "input_too_large": "Content exceeds maximum size (100KB)",
                "null_byte_detected": "Content contains null bytes",
                "control_characters_detected": "Content contains invalid control characters"
            }
            return {
                "success": False,
                "message": error_messages.get(content_validation.error, f"Content validation failed: {content_validation.error}"),
                "error_type": content_validation.error,
            }

        # Validate TTL
        if ttl < 60 or ttl > 86400:  # 1 minute to 24 hours
            return {
                "success": False,
                "message": "TTL must be between 60 and 86400 seconds",
                "error_type": "invalid_ttl",
            }

        # Create namespaced key
        try:
            namespaced_key = agent_namespace.create_namespaced_key(agent_id, "scratchpad", key)
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to create namespaced key: {str(e)}",
                "error_type": "namespace_error",
            }

        # Check Redis availability
        if not kv_store or not kv_store.redis:
            return {
                "success": False,
                "message": "Redis storage not available",
                "error_type": "storage_unavailable",
            }

        try:
            # Handle append mode
            if mode == "append":
                existing_content = kv_store.redis.get(namespaced_key)
                if existing_content:
                    content = existing_content + "\n" + content

            # Store content with TTL
            success = kv_store.redis.setex(namespaced_key, ttl, content)

            if success:
                logger.info(f"Updated scratchpad for agent {agent_id}, key: {key}, TTL: {ttl}s")
                return {
                    "success": True,
                    "message": f"Scratchpad updated successfully (mode: {mode})",
                    "key": namespaced_key,
                    "ttl": ttl,
                    "content_size": len(content),
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to store content in Redis",
                    "error_type": "storage_error",
                }

        except Exception as e:
            logger.error(f"Redis operation failed: {e}")
            return {
                "success": False,
                "message": f"Storage operation failed: {str(e)}",
                "error_type": "storage_exception",
            }

    except KeyError as e:
        return {
            "success": False,
            "message": f"Missing required parameter: {str(e)}",
            "error_type": "missing_parameter",
        }
    except Exception as e:
        logger.error(f"Error updating scratchpad: {e}")
        return {
            "success": False,
            "message": f"Failed to update scratchpad: {str(e)}",
            "error_type": "unexpected_error",
        }


async def get_agent_state_tool(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve agent state with namespace isolation.

    This function retrieves agent state data from the appropriate storage
    backend while ensuring namespace isolation. Agents can only access
    their own state data.

    Args:
        arguments: Dictionary containing:
            - agent_id (str): Agent identifier (required)
            - key (str, optional): Specific state key to retrieve
            - prefix (str, optional): State type - 'state', 'scratchpad', 'memory', 'config' (default: 'state')

    Returns:
        Dict containing:
            - success (bool): Whether the operation succeeded
            - data (dict): Retrieved state data
            - keys (list, optional): Available keys if no specific key requested
            - message (str): Success or error message

    Example:
        >>> arguments = {
        ...     "agent_id": "agent-123",
        ...     "key": "working_memory",
        ...     "prefix": "scratchpad"
        ... }
        >>> result = await get_agent_state_tool(arguments)
        >>> print(result["data"])  # Retrieved content
    """
    # Rate limiting check
    allowed, rate_limit_msg = await rate_limit_check("get_agent_state")
    if not allowed:
        return {
            "success": False,
            "data": {},
            "message": f"Rate limit exceeded: {rate_limit_msg}",
            "error_type": "rate_limit",
        }

    try:
        agent_id = arguments["agent_id"]
        key = arguments.get("key")
        prefix = arguments.get("prefix", "state")

        # Validate agent ID
        if not agent_namespace.validate_agent_id(agent_id):
            return {
                "success": False,
                "data": {},
                "message": f"Invalid agent ID format: {agent_id}",
                "error_type": "invalid_agent_id",
            }

        # Validate prefix
        if not agent_namespace.validate_prefix(prefix):
            return {
                "success": False,
                "data": {},
                "message": f"Invalid prefix: {prefix}",
                "error_type": "invalid_prefix",
            }

        # Validate key if provided
        if key and not agent_namespace.validate_key(key):
            return {
                "success": False,
                "data": {},
                "message": f"Invalid key format: {key}",
                "error_type": "invalid_key",
            }

        # Check Redis availability
        if not kv_store or not kv_store.redis:
            return {
                "success": False,
                "data": {},
                "message": "Redis storage not available",
                "error_type": "storage_unavailable",
            }

        try:
            if key:
                # Retrieve specific key
                namespaced_key = agent_namespace.create_namespaced_key(agent_id, prefix, key)

                # Verify agent has access to this key
                if not agent_namespace.verify_agent_access(agent_id, namespaced_key):
                    return {
                        "success": False,
                        "data": {},
                        "message": "Access denied to requested resource",
                        "error_type": "access_denied",
                    }

                content = kv_store.redis.get(namespaced_key)
                if content is None:
                    return {
                        "success": False,
                        "data": {},
                        "message": f"Key '{key}' not found",
                        "error_type": "key_not_found",
                    }

                logger.info(f"Retrieved state for agent {agent_id}, key: {key}")
                return {
                    "success": True,
                    "data": {
                        "key": key,
                        "content": content,
                        "namespaced_key": namespaced_key,
                    },
                    "message": "State retrieved successfully",
                }

            else:
                # List all keys for the agent with the given prefix
                pattern = f"agent:{agent_id}:{prefix}:*"
                keys = kv_store.redis.keys(pattern)

                if not keys:
                    return {
                        "success": True,
                        "data": {},
                        "keys": [],
                        "message": f"No {prefix} data found for agent",
                    }

                # Retrieve all matching keys (limit to reasonable amount)
                max_keys = 100
                data = {}
                retrieved_keys = []

                for namespaced_key in keys[:max_keys]:
                    try:
                        # Parse the key to get the actual key name
                        _, _, _, actual_key = namespaced_key.split(":", 3)
                        content = kv_store.redis.get(namespaced_key)
                        if content is not None:
                            data[actual_key] = content
                            retrieved_keys.append(actual_key)
                    except Exception as e:
                        logger.warning(f"Failed to retrieve key {namespaced_key}: {e}")
                        continue

                logger.info(f"Retrieved {len(retrieved_keys)} {prefix} keys for agent {agent_id}")
                return {
                    "success": True,
                    "data": data,
                    "keys": retrieved_keys,
                    "message": f"Retrieved {len(retrieved_keys)} {prefix} entries",
                    "total_available": len(keys),
                }

        except Exception as e:
            logger.error(f"Redis operation failed: {e}")
            return {
                "success": False,
                "data": {},
                "message": f"Storage operation failed: {str(e)}",
                "error_type": "storage_exception",
            }

    except KeyError as e:
        return {
            "success": False,
            "data": {},
            "message": f"Missing required parameter: {str(e)}",
            "error_type": "missing_parameter",
        }
    except Exception as e:
        logger.error(f"Error retrieving agent state: {e}")
        return {
            "success": False,
            "data": {},
            "message": f"Failed to retrieve state: {str(e)}",
            "error_type": "unexpected_error",
        }


async def get_health_status() -> Dict[str, Any]:
    """Get server health status."""
    health_status = {
        "status": "healthy",
        "services": {"neo4j": "unknown", "qdrant": "unknown", "redis": "unknown"},
    }

    # Check Neo4j using Neo4jInitializer.query method
    if neo4j_client and neo4j_client.driver:
        try:
            neo4j_client.query("RETURN 1")
            health_status["services"]["neo4j"] = "healthy"
        except Exception as e:
            health_status["services"]["neo4j"] = "unhealthy"
            health_status["status"] = "degraded"
            logger.warning(f"Neo4j health check failed: {e}")

    # Check Qdrant using VectorDBInitializer.get_collections method
    if qdrant_client and qdrant_client.client:
        try:
            qdrant_client.get_collections()
            health_status["services"]["qdrant"] = "healthy"
        except Exception as e:
            health_status["services"]["qdrant"] = "unhealthy"
            health_status["status"] = "degraded"
            logger.warning(f"Qdrant health check failed: {e}")

    # Check Redis/KV Store
    if kv_store:
        try:
            # Test Redis connection through ContextKV
            if kv_store.redis.redis_client:
                kv_store.redis.redis_client.ping()
                health_status["services"]["redis"] = "healthy"
            else:
                health_status["services"]["redis"] = "disconnected"
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["services"]["redis"] = "unhealthy"
            health_status["status"] = "degraded"
            logger.warning(f"Redis health check failed: {e}")

    return health_status


async def detect_communities_tool(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Detect communities in the knowledge graph using GraphRAG.

    This tool performs community detection on the knowledge graph to identify
    clusters of related documents, entities, or concepts. It uses various
    community detection algorithms to analyze the graph structure and find
    meaningful groupings.

    Args:
        arguments: Dictionary containing:
            - algorithm (str, optional): Community detection algorithm ('louvain', 'leiden', 'modularity')
                Default: 'louvain'
            - min_community_size (int, optional): Minimum size for communities (2-50)
                Default: 3
            - resolution (float, optional): Resolution parameter for algorithm (0.1-2.0)
                Default: 1.0
            - include_members (bool, optional): Include community member details
                Default: True

    Returns:
        Dict containing:
            - success (bool): Whether community detection succeeded
            - communities (list): List of detected communities
            - algorithm_used (str): Algorithm that was used
            - total_communities (int): Total number of communities found
            - message (str): Success or error message

    Example:
        >>> arguments = {
        ...     "algorithm": "louvain",
        ...     "min_community_size": 3,
        ...     "resolution": 1.0,
        ...     "include_members": True
        ... }
        >>> result = await detect_communities_tool(arguments)
        >>> print(f"Found {result['total_communities']} communities")
    """
    # Import GraphRAG bridge
    try:
        from .graphrag_bridge import get_graphrag_bridge
    except ImportError:
        return {
            "success": False,
            "communities": [],
            "message": "GraphRAG bridge not available",
            "error_type": "import_error",
        }

    # Rate limiting check
    allowed, rate_limit_msg = await rate_limit_check("detect_communities")
    if not allowed:
        return {
            "success": False,
            "communities": [],
            "message": f"Rate limit exceeded: {rate_limit_msg}",
            "error_type": "rate_limit",
        }

    try:
        # Extract parameters
        algorithm = arguments.get("algorithm", "louvain")
        min_community_size = arguments.get("min_community_size", 3)
        resolution = arguments.get("resolution", 1.0)
        include_members = arguments.get("include_members", True)

        # Validate parameters
        if algorithm not in ["louvain", "leiden", "modularity"]:
            return {
                "success": False,
                "communities": [],
                "message": f"Invalid algorithm: {algorithm}. Must be one of: louvain, leiden, modularity",
                "error_type": "invalid_algorithm",
            }

        if not (2 <= min_community_size <= 50):
            return {
                "success": False,
                "communities": [],
                "message": f"Invalid min_community_size: {min_community_size}. Must be between 2 and 50",
                "error_type": "invalid_parameter",
            }

        if not (0.1 <= resolution <= 2.0):
            return {
                "success": False,
                "communities": [],
                "message": f"Invalid resolution: {resolution}. Must be between 0.1 and 2.0",
                "error_type": "invalid_parameter",
            }

        # Get GraphRAG bridge and perform community detection
        bridge = get_graphrag_bridge()

        if not bridge.is_available():
            return {
                "success": False,
                "communities": [],
                "message": "GraphRAG integration not available",
                "error_type": "graphrag_unavailable",
            }

        # Perform community detection
        result = await bridge.detect_communities(
            algorithm=algorithm, min_community_size=min_community_size, resolution=resolution
        )

        # Filter communities by size if requested
        if result["success"] and min_community_size > 1:
            filtered_communities = [
                community
                for community in result["communities"]
                if community.get("size", 0) >= min_community_size
            ]
            result["communities"] = filtered_communities
            result["total_communities"] = len(filtered_communities)

        # Remove member details if not requested
        if not include_members and result["success"]:
            for community in result["communities"]:
                if "members" in community:
                    community["member_count"] = len(community["members"])
                    del community["members"]

        # Add GraphRAG metadata
        result["graphrag_metadata"] = {
            "algorithm_used": algorithm,
            "min_community_size": min_community_size,
            "resolution": resolution,
            "include_members": include_members,
            "bridge_status": bridge.get_status(),
        }

        return result

    except Exception as e:
        logger.error(f"Error in community detection: {e}")
        return {
            "success": False,
            "communities": [],
            "message": f"Community detection failed: {str(e)}",
            "error_type": "execution_error",
        }


async def get_tools_info() -> Dict[str, Any]:
    """Get information about available tools."""
    contracts_dir = Path(__file__).parent.parent.parent / "contracts"
    tools = []

    if contracts_dir.exists():
        for contract_file in contracts_dir.glob("*.json"):
            try:
                with open(contract_file) as f:
                    contract = json.load(f)
                    tools.append(
                        {
                            "name": contract.get("name"),
                            "description": contract.get("description"),
                            "version": contract.get("version"),
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to load contract {contract_file}: {e}")

    return {"tools": tools, "server_version": "1.0.0", "mcp_version": "1.0"}


async def select_tools_tool(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Select relevant tools based on query using multiple selection algorithms.

    This tool provides intelligent tool selection using numpy dot-product relevance,
    rule-based heuristics, or hybrid approaches. It returns ≤ N relevant tools
    and logs selection results in YAML format.

    Args:
        arguments: Dictionary containing:
            - query (str): Search query or context description (required)
            - max_tools (int, optional): Maximum number of tools to return (default: 5)
            - method (str, optional): Selection method (default: "hybrid")
            - category_filter (str, optional): Filter by category
            - include_scores (bool, optional): Include relevance scores (default: True)

    Returns:
        Dict containing selected tools, metadata, and YAML selection log
    """
    if not tool_selector_bridge:
        return {
            "success": False,
            "tools": [],
            "total_available": 0,
            "message": "Tool selector bridge not available",
            "error_type": "bridge_unavailable",
        }

    # Rate limiting check
    allowed, rate_limit_msg = await rate_limit_check("select_tools")
    if not allowed:
        return {
            "success": False,
            "tools": [],
            "total_available": 0,
            "message": f"Rate limit exceeded: {rate_limit_msg}",
            "error_type": "rate_limit",
        }

    try:
        result = await tool_selector_bridge.select_tools(arguments)
        logger.info(f"Tool selection completed: {result.get('message', 'No message')}")
        return result

    except Exception as e:
        logger.error(f"Error in select_tools_tool: {e}")
        return {
            "success": False,
            "tools": [],
            "total_available": 0,
            "message": f"Tool selection failed: {str(e)}",
            "error_type": "execution_error",
        }


async def list_available_tools_tool(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    List all available tools in the tool selector system.

    This tool provides a comprehensive list of all tools available in the
    tool selector system, with optional filtering and sorting capabilities.

    Args:
        arguments: Dictionary containing:
            - category_filter (str, optional): Filter by category
            - include_metadata (bool, optional): Include metadata (default: True)
            - sort_by (str, optional): Sort by field (default: "name")

    Returns:
        Dict containing list of tools, categories, and metadata
    """
    if not tool_selector_bridge:
        return {
            "success": False,
            "tools": [],
            "total_count": 0,
            "categories": [],
            "message": "Tool selector bridge not available",
            "error_type": "bridge_unavailable",
        }

    # Rate limiting check
    allowed, rate_limit_msg = await rate_limit_check("list_available_tools")
    if not allowed:
        return {
            "success": False,
            "tools": [],
            "total_count": 0,
            "categories": [],
            "message": f"Rate limit exceeded: {rate_limit_msg}",
            "error_type": "rate_limit",
        }

    try:
        result = await tool_selector_bridge.list_available_tools(arguments)
        logger.info(f"Tool listing completed: {result.get('message', 'No message')}")
        return result

    except Exception as e:
        logger.error(f"Error in list_available_tools_tool: {e}")
        return {
            "success": False,
            "tools": [],
            "total_count": 0,
            "categories": [],
            "message": f"Tool listing failed: {str(e)}",
            "error_type": "execution_error",
        }


async def get_tool_info_tool(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get detailed information about a specific tool.

    This tool provides comprehensive information about a specific tool,
    including its documentation, usage examples, and metadata.

    Args:
        arguments: Dictionary containing:
            - tool_name (str): Name of the tool to get info about (required)
            - include_usage_examples (bool, optional): Include usage examples (default: True)

    Returns:
        Dict containing detailed tool information
    """
    if not tool_selector_bridge:
        return {
            "success": False,
            "message": "Tool selector bridge not available",
            "error_type": "bridge_unavailable",
        }

    # Rate limiting check
    allowed, rate_limit_msg = await rate_limit_check("get_tool_info")
    if not allowed:
        return {
            "success": False,
            "message": f"Rate limit exceeded: {rate_limit_msg}",
            "error_type": "rate_limit",
        }

    try:
        result = await tool_selector_bridge.get_tool_info(arguments)
        logger.info(f"Tool info retrieval completed: {result.get('message', 'No message')}")
        return result

    except Exception as e:
        logger.error(f"Error in get_tool_info_tool: {e}")
        return {
            "success": False,
            "message": f"Tool info retrieval failed: {str(e)}",
            "error_type": "execution_error",
        }


async def store_fact_tool(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Store facts with automatic extraction and validation."""
    if not fact_store or not scope_middleware:
        return {
            "success": False,
            "message": "Fact system not initialized",
            "error_type": "system_unavailable",
        }

    # Rate limiting check
    allowed, rate_limit_msg = await rate_limit_check("store_fact")
    if not allowed:
        return {
            "success": False,
            "message": f"Rate limit exceeded: {rate_limit_msg}",
            "error_type": "rate_limit",
        }

    try:
        # Process request through scope middleware
        processed_request = scope_middleware.process_request("store_fact", arguments)
        scope = processed_request["_validated_scope"]

        text = arguments.get("text", "")
        source_turn_id = arguments.get("source_turn_id", "")
        
        # Check for manual fact storage
        if arguments.get("attribute") and arguments.get("value"):
            # Manual fact storage - use graph fact store if available
            active_fact_store = graph_fact_store if graph_fact_store else fact_store
            
            # Run potentially long-running storage operation asynchronously
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: active_fact_store.store_fact(
                        namespace=scope.namespace,
                        user_id=scope.user_id,
                        attribute=arguments["attribute"],
                        value=arguments["value"],
                        source_turn_id=source_turn_id
                    )
                )
                stored_facts = [{"attribute": arguments["attribute"], "value": arguments["value"]}]
            except Exception as e:
                logger.error(f"Error storing fact: {e}")
                return {
                    "success": False,
                    "message": f"Failed to store fact: {str(e)}",
                    "error_type": "storage_error"
                }
        else:
            # Automatic fact extraction - run async for long-running operations
            try:
                # Extract facts asynchronously for long texts
                extracted_facts = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: fact_extractor.extract_facts_from_text(text, source_turn_id)
                )
                stored_facts = []
                
                # Store each fact asynchronously
                active_fact_store = graph_fact_store if graph_fact_store else fact_store
                store_tasks = []
                
                for fact in extracted_facts:
                    # Create async task for each fact storage operation
                    task = asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda f=fact: active_fact_store.store_fact(
                            namespace=scope.namespace,
                            user_id=scope.user_id,
                            attribute=f.attribute,
                            value=f.value,
                            confidence=f.confidence,
                            source_turn_id=source_turn_id
                        )
                    )
                    store_tasks.append(task)
                    stored_facts.append({
                        "attribute": fact.attribute,
                        "value": fact.value,
                        "confidence": fact.confidence
                    })
                
                # Wait for all storage operations to complete
                if store_tasks:
                    await asyncio.gather(*store_tasks)
                    
            except Exception as e:
                logger.error(f"Error in automatic fact extraction: {e}")
                return {
                    "success": False,
                    "message": f"Failed to extract and store facts: {str(e)}",
                    "error_type": "extraction_error"
                }

        logger.info(f"Stored {len(stored_facts)} facts for {scope.namespace}:{scope.user_id}")
        
        return {
            "success": True,
            "message": f"Stored {len(stored_facts)} facts successfully",
            "stored_facts": stored_facts,
            "namespace": scope.namespace,
            "user_id": scope.user_id
        }

    except Exception as e:
        logger.error(f"Error in store_fact_tool: {e}")
        return {
            "success": False,
            "message": f"Fact storage failed: {str(e)}",
            "error_type": "execution_error",
        }


async def retrieve_fact_tool(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve facts using intent classification and deterministic lookup."""
    if not fact_store or not scope_middleware:
        return {
            "success": False,
            "message": "Fact system not initialized",
            "error_type": "system_unavailable",
        }

    # Rate limiting check
    allowed, rate_limit_msg = await rate_limit_check("retrieve_fact")
    if not allowed:
        return {
            "success": False,
            "message": f"Rate limit exceeded: {rate_limit_msg}",
            "error_type": "rate_limit",
        }

    try:
        # Process request through scope middleware
        processed_request = scope_middleware.process_request("retrieve_fact", arguments)
        scope = processed_request["_validated_scope"]

        query = arguments.get("query", "")
        attribute = arguments.get("attribute")
        fallback_to_context = arguments.get("fallback_to_context", True)

        # Classify intent
        intent_result = intent_classifier.classify(query)
        
        if attribute:
            # Direct attribute lookup
            fact = fact_store.get_fact(scope.namespace, scope.user_id, attribute)
            if fact:
                return {
                    "success": True,
                    "message": "Fact retrieved successfully",
                    "fact": {
                        "attribute": fact.attribute,
                        "value": fact.value,
                        "confidence": fact.confidence,
                        "updated_at": fact.updated_at
                    },
                    "intent": intent_result.intent.value,
                    "method": "direct_lookup"
                }
        elif intent_result.attribute:
            # Intent-based lookup
            fact = fact_store.get_fact(scope.namespace, scope.user_id, intent_result.attribute)
            if fact:
                return {
                    "success": True,
                    "message": "Fact retrieved successfully",
                    "fact": {
                        "attribute": fact.attribute,
                        "value": fact.value,
                        "confidence": fact.confidence,
                        "updated_at": fact.updated_at
                    },
                    "intent": intent_result.intent.value,
                    "method": "intent_lookup"
                }

        # Fact not found
        if fallback_to_context:
            # Fall back to context search
            context_args = {
                "query": query,
                "limit": 5,
                "search_mode": "hybrid"
            }
            context_result = await retrieve_context_tool(context_args)
            
            return {
                "success": True,
                "message": "Fact not found, returned context search results",
                "fact": None,
                "intent": intent_result.intent.value,
                "method": "context_fallback",
                "context_results": context_result.get("results", [])
            }
        else:
            return {
                "success": False,
                "message": "Fact not found",
                "fact": None,
                "intent": intent_result.intent.value,
                "method": "fact_lookup_only"
            }

    except Exception as e:
        logger.error(f"Error in retrieve_fact_tool: {e}")
        return {
            "success": False,
            "message": f"Fact retrieval failed: {str(e)}",
            "error_type": "execution_error",
        }


async def list_user_facts_tool(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """List all stored facts for a user."""
    if not fact_store or not scope_middleware:
        return {
            "success": False,
            "message": "Fact system not initialized",
            "error_type": "system_unavailable",
        }

    # Rate limiting check
    allowed, rate_limit_msg = await rate_limit_check("list_user_facts")
    if not allowed:
        return {
            "success": False,
            "message": f"Rate limit exceeded: {rate_limit_msg}",
            "error_type": "rate_limit",
        }

    try:
        # Process request through scope middleware
        processed_request = scope_middleware.process_request("list_facts", arguments)
        scope = processed_request["_validated_scope"]

        include_history = arguments.get("include_history", False)

        # Get all facts for user
        user_facts = fact_store.get_user_facts(scope.namespace, scope.user_id)
        
        facts_data = []
        for attribute, fact in user_facts.items():
            fact_data = {
                "attribute": fact.attribute,
                "value": fact.value,
                "confidence": fact.confidence,
                "updated_at": fact.updated_at,
                "provenance": fact.provenance,
                "source_turn_id": fact.source_turn_id
            }
            
            if include_history:
                history = fact_store.get_fact_history(scope.namespace, scope.user_id, attribute)
                fact_data["history"] = history
            
            facts_data.append(fact_data)

        logger.info(f"Listed {len(facts_data)} facts for {scope.namespace}:{scope.user_id}")
        
        return {
            "success": True,
            "message": f"Found {len(facts_data)} facts",
            "facts": facts_data,
            "namespace": scope.namespace,
            "user_id": scope.user_id,
            "include_history": include_history
        }

    except Exception as e:
        logger.error(f"Error in list_user_facts_tool: {e}")
        return {
            "success": False,
            "message": f"Fact listing failed: {str(e)}",
            "error_type": "execution_error",
        }


async def delete_user_facts_tool(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Delete all facts for a user (forget-me functionality)."""
    if not fact_store or not scope_middleware:
        return {
            "success": False,
            "message": "Fact system not initialized",
            "error_type": "system_unavailable",
        }

    # Rate limiting check
    allowed, rate_limit_msg = await rate_limit_check("delete_user_facts")
    if not allowed:
        return {
            "success": False,
            "message": f"Rate limit exceeded: {rate_limit_msg}",
            "error_type": "rate_limit",
        }

    try:
        # Process request through scope middleware
        processed_request = scope_middleware.process_request("delete_fact", arguments)
        scope = processed_request["_validated_scope"]

        confirm = arguments.get("confirm", False)
        
        if not confirm:
            return {
                "success": False,
                "message": "Deletion requires explicit confirmation",
                "error_type": "confirmation_required",
            }

        # Delete all facts for user - run async for potentially large deletions
        deleted_count = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: fact_store.delete_user_facts(scope.namespace, scope.user_id)
        )
        
        logger.info(f"Deleted {deleted_count} fact entries for {scope.namespace}:{scope.user_id}")
        
        return {
            "success": True,
            "message": f"Deleted {deleted_count} fact entries successfully",
            "deleted_count": deleted_count,
            "namespace": scope.namespace,
            "user_id": scope.user_id
        }

    except Exception as e:
        logger.error(f"Error in delete_user_facts_tool: {e}")
        return {
            "success": False,
            "message": f"Fact deletion failed: {str(e)}",
            "error_type": "execution_error",
        }


async def classify_intent_tool(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Classify query intent for debugging and development."""
    # Rate limiting check
    allowed, rate_limit_msg = await rate_limit_check("classify_intent")
    if not allowed:
        return {
            "success": False,
            "message": f"Rate limit exceeded: {rate_limit_msg}",
            "error_type": "rate_limit",
        }

    try:
        query = arguments.get("query", "")
        include_explanation = arguments.get("include_explanation", False)

        if include_explanation:
            result = intent_classifier.explain_classification(query)
        else:
            intent_result = intent_classifier.classify(query)
            result = {
                "query": query,
                "intent": intent_result.intent.value,
                "confidence": intent_result.confidence,
                "attribute": intent_result.attribute,
                "reasoning": intent_result.reasoning
            }

        return {
            "success": True,
            "message": "Intent classification completed",
            "classification": result
        }

    except Exception as e:
        logger.error(f"Error in classify_intent_tool: {e}")
        return {
            "success": False,
            "message": f"Intent classification failed: {str(e)}",
            "error_type": "execution_error",
        }


async def delete_context_tool(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Delete a stored context by ID."""
    # Rate limiting check
    allowed, rate_limit_msg = await rate_limit_check("delete_context")
    if not allowed:
        return {
            "success": False,
            "message": f"Rate limit exceeded: {rate_limit_msg}",
            "error_type": "rate_limit",
        }

    try:
        context_id = arguments.get("context_id", "")
        confirm = arguments.get("confirm", False)

        if not context_id:
            return {
                "success": False,
                "message": "context_id is required",
                "error_type": "validation_error",
            }

        if not confirm:
            return {
                "success": False,
                "message": "confirm must be true to delete context",
                "error_type": "validation_error",
            }

        # Use the existing delete_context method from ContextKV
        deleted = kv_store.delete_context(context_id)

        if deleted:
            return {
                "success": True,
                "message": f"Context {context_id} deleted successfully",
                "context_id": context_id
            }
        else:
            return {
                "success": False,
                "message": f"Context {context_id} not found or could not be deleted",
                "error_type": "not_found",
            }

    except Exception as e:
        logger.error(f"Error in delete_context_tool: {e}")
        return {
            "success": False,
            "message": f"Context deletion failed: {str(e)}",
            "error_type": "execution_error",
        }


async def list_context_types_tool(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Get available context types and their descriptions."""
    # Rate limiting check
    allowed, rate_limit_msg = await rate_limit_check("list_context_types")
    if not allowed:
        return {
            "success": False,
            "message": f"Rate limit exceeded: {rate_limit_msg}",
            "error_type": "rate_limit",
        }

    try:
        include_descriptions = arguments.get("include_descriptions", True)

        # Define available context types with descriptions
        context_types = {
            "design": "Design documents, specifications, and architectural decisions",
            "decision": "Decision records, choices made, and their rationale",
            "trace": "Execution traces, debugging information, and system behavior",
            "sprint": "Sprint planning, retrospectives, and iteration artifacts",
            "log": "System logs, events, and operational information",
            "test": "Test cases, test results, and testing artifacts"  # Added 'test' type
        }

        if include_descriptions:
            result = {
                "context_types": [
                    {"type": type_name, "description": description}
                    for type_name, description in context_types.items()
                ]
            }
        else:
            result = {
                "context_types": list(context_types.keys())
            }

        return {
            "success": True,
            "message": "Context types retrieved successfully",
            **result
        }

    except Exception as e:
        logger.error(f"Error in list_context_types_tool: {e}")
        return {
            "success": False,
            "message": f"Error listing context types: {str(e)}",
            "error_type": "execution_error",
        }


async def redis_get_tool(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Get a value from Redis by key."""
    # Rate limiting check
    allowed, rate_limit_msg = await rate_limit_check("redis_get")
    if not allowed:
        return {
            "success": False,
            "message": f"Rate limit exceeded: {rate_limit_msg}",
            "error_type": "rate_limit",
        }

    try:
        key = arguments.get("key", "")
        
        if not key:
            return {
                "success": False,
                "message": "key is required",
                "error_type": "validation_error",
            }

        # Validate key format
        key_validation = input_validator.validate_input(key, "query")
        if not key_validation.valid:
            return {
                "success": False,
                "message": f"Key validation failed: {key_validation.error}",
                "error_type": key_validation.error,
            }

        # Use kv_store to get value from Redis
        value = kv_store.redis.get(key)
        
        return {
            "success": True,
            "key": key,
            "value": value,
            "exists": value is not None,
            "message": f"Retrieved key '{key}'" if value is not None else f"Key '{key}' not found"
        }

    except Exception as e:
        logger.error(f"Error in redis_get_tool: {e}")
        return {
            "success": False,
            "message": f"Redis GET failed: {str(e)}",
            "error_type": "execution_error",
        }


async def redis_set_tool(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Set a key-value pair in Redis."""
    # Rate limiting check
    allowed, rate_limit_msg = await rate_limit_check("redis_set")
    if not allowed:
        return {
            "success": False,
            "message": f"Rate limit exceeded: {rate_limit_msg}",
            "error_type": "rate_limit",
        }

    try:
        key = arguments.get("key", "")
        value = arguments.get("value", "")
        ex = arguments.get("ex")
        
        if not key or not value:
            return {
                "success": False,
                "message": "key and value are required",
                "error_type": "validation_error",
            }

        # Validate key and value
        key_validation = input_validator.validate_input(key, "query")
        if not key_validation.valid:
            return {
                "success": False,
                "message": f"Key validation failed: {key_validation.error}",
                "error_type": key_validation.error,
            }

        value_validation = input_validator.validate_input(value, "content")
        if not value_validation.valid:
            return {
                "success": False,
                "message": f"Value validation failed: {value_validation.error}",
                "error_type": value_validation.error,
            }

        # Use kv_store to set value in Redis
        success = kv_store.redis.set(key, value, ex=ex)
        
        return {
            "success": success,
            "key": key,
            "value": value,
            "expiration": ex,
            "message": f"Set key '{key}' with value" + (f" (expires in {ex}s)" if ex else "")
        }

    except Exception as e:
        logger.error(f"Error in redis_set_tool: {e}")
        return {
            "success": False,
            "message": f"Redis SET failed: {str(e)}",
            "error_type": "execution_error",
        }


async def redis_hget_tool(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Get a field value from a Redis hash."""
    # Rate limiting check
    allowed, rate_limit_msg = await rate_limit_check("redis_hget")
    if not allowed:
        return {
            "success": False,
            "message": f"Rate limit exceeded: {rate_limit_msg}",
            "error_type": "rate_limit",
        }

    try:
        key = arguments.get("key", "")
        field = arguments.get("field", "")
        
        if not key or not field:
            return {
                "success": False,
                "message": "key and field are required",
                "error_type": "validation_error",
            }

        # Use kv_store Redis client for hash operations
        redis_client = kv_store.redis.redis_client
        if not redis_client:
            return {
                "success": False,
                "message": "Redis client not available",
                "error_type": "connection_error",
            }

        value = redis_client.hget(key, field)
        if value:
            value = value.decode('utf-8') if isinstance(value, bytes) else str(value)
        
        return {
            "success": True,
            "key": key,
            "field": field,
            "value": value,
            "exists": value is not None,
            "message": f"Retrieved field '{field}' from hash '{key}'" if value is not None else f"Field '{field}' not found in hash '{key}'"
        }

    except Exception as e:
        logger.error(f"Error in redis_hget_tool: {e}")
        return {
            "success": False,
            "message": f"Redis HGET failed: {str(e)}",
            "error_type": "execution_error",
        }


async def redis_hset_tool(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Set a field value in a Redis hash."""
    # Rate limiting check
    allowed, rate_limit_msg = await rate_limit_check("redis_hset")
    if not allowed:
        return {
            "success": False,
            "message": f"Rate limit exceeded: {rate_limit_msg}",
            "error_type": "rate_limit",
        }

    try:
        key = arguments.get("key", "")
        field = arguments.get("field", "")
        value = arguments.get("value", "")
        
        if not key or not field or not value:
            return {
                "success": False,
                "message": "key, field, and value are required",
                "error_type": "validation_error",
            }

        # Use kv_store Redis client for hash operations
        redis_client = kv_store.redis.redis_client
        if not redis_client:
            return {
                "success": False,
                "message": "Redis client not available",
                "error_type": "connection_error",
            }

        result = redis_client.hset(key, field, value)
        
        return {
            "success": True,
            "key": key,
            "field": field,
            "value": value,
            "new_field": result == 1,
            "message": f"Set field '{field}' in hash '{key}' to '{value}'"
        }

    except Exception as e:
        logger.error(f"Error in redis_hset_tool: {e}")
        return {
            "success": False,
            "message": f"Redis HSET failed: {str(e)}",
            "error_type": "execution_error",
        }


async def redis_lpush_tool(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Push an element to the head of a Redis list."""
    # Rate limiting check
    allowed, rate_limit_msg = await rate_limit_check("redis_lpush")
    if not allowed:
        return {
            "success": False,
            "message": f"Rate limit exceeded: {rate_limit_msg}",
            "error_type": "rate_limit",
        }

    try:
        key = arguments.get("key", "")
        value = arguments.get("value", "")
        
        if not key or not value:
            return {
                "success": False,
                "message": "key and value are required",
                "error_type": "validation_error",
            }

        # Use kv_store Redis client for list operations
        redis_client = kv_store.redis.redis_client
        if not redis_client:
            return {
                "success": False,
                "message": "Redis client not available",
                "error_type": "connection_error",
            }

        list_length = redis_client.lpush(key, value)
        
        return {
            "success": True,
            "key": key,
            "value": value,
            "list_length": list_length,
            "message": f"Pushed '{value}' to list '{key}', new length: {list_length}"
        }

    except Exception as e:
        logger.error(f"Error in redis_lpush_tool: {e}")
        return {
            "success": False,
            "message": f"Redis LPUSH failed: {str(e)}",
            "error_type": "execution_error",
        }


async def redis_lrange_tool(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Get a range of elements from a Redis list."""
    # Rate limiting check
    allowed, rate_limit_msg = await rate_limit_check("redis_lrange")
    if not allowed:
        return {
            "success": False,
            "message": f"Rate limit exceeded: {rate_limit_msg}",
            "error_type": "rate_limit",
        }

    try:
        key = arguments.get("key", "")
        start = arguments.get("start", 0)
        stop = arguments.get("stop", -1)
        
        if not key:
            return {
                "success": False,
                "message": "key is required",
                "error_type": "validation_error",
            }

        # Use kv_store Redis client for list operations
        redis_client = kv_store.redis.redis_client
        if not redis_client:
            return {
                "success": False,
                "message": "Redis client not available",
                "error_type": "connection_error",
            }

        elements = redis_client.lrange(key, start, stop)
        # Decode bytes to strings
        elements = [elem.decode('utf-8') if isinstance(elem, bytes) else str(elem) for elem in elements]
        
        return {
            "success": True,
            "key": key,
            "start": start,
            "stop": stop,
            "elements": elements,
            "count": len(elements),
            "message": f"Retrieved {len(elements)} elements from list '{key}'"
        }

    except Exception as e:
        logger.error(f"Error in redis_lrange_tool: {e}")
        return {
            "success": False,
            "message": f"Redis LRANGE failed: {str(e)}",
            "error_type": "execution_error",
        }


async def main():
    """Main server entry point."""
    # Initialize storage clients
    await initialize_storage_clients()

    try:
        # Run the server using stdio transport
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="context-store",
                    server_version="1.0.0",
                    capabilities={}
                ),
            )
    finally:
        # Clean up
        await cleanup_storage_clients()


if __name__ == "__main__":
    asyncio.run(main())
