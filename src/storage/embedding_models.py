#!/usr/bin/env python3
"""
Stronger embedding models test configuration
Support for bge-large, text-embedding-3-large, and other high-performance models
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingModelConfig:
    """Configuration for embedding models"""
    name: str
    model_id: str
    dimensions: int
    max_sequence_length: int
    normalization: bool = True
    expected_performance: str = "baseline"  # baseline, good, excellent

# Model configurations for testing
EMBEDDING_MODELS = {
    # Current baseline (Phase 1.5)
    "sentence-transformers/all-MiniLM-L6-v2": EmbeddingModelConfig(
        name="MiniLM-L6-v2",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=384,
        max_sequence_length=256,
        expected_performance="baseline"
    ),
    
    # BGE models (higher quality)
    "BAAI/bge-base-en-v1.5": EmbeddingModelConfig(
        name="BGE-Base-EN",
        model_id="BAAI/bge-base-en-v1.5",
        dimensions=768,
        max_sequence_length=512,
        expected_performance="good"
    ),
    
    "BAAI/bge-large-en-v1.5": EmbeddingModelConfig(
        name="BGE-Large-EN",
        model_id="BAAI/bge-large-en-v1.5",
        dimensions=1024,
        max_sequence_length=512,
        expected_performance="excellent"
    ),
    
    # OpenAI models (via API)
    "text-embedding-3-small": EmbeddingModelConfig(
        name="OpenAI-3-Small",
        model_id="text-embedding-3-small",
        dimensions=1536,
        max_sequence_length=8191,
        expected_performance="good"
    ),
    
    "text-embedding-3-large": EmbeddingModelConfig(
        name="OpenAI-3-Large", 
        model_id="text-embedding-3-large",
        dimensions=3072,
        max_sequence_length=8191,
        expected_performance="excellent"
    ),
    
    # E5 models
    "intfloat/e5-base-v2": EmbeddingModelConfig(
        name="E5-Base-v2",
        model_id="intfloat/e5-base-v2",
        dimensions=768,
        max_sequence_length=512,
        expected_performance="good"
    ),
    
    "intfloat/e5-large-v2": EmbeddingModelConfig(
        name="E5-Large-v2",
        model_id="intfloat/e5-large-v2", 
        dimensions=1024,
        max_sequence_length=512,
        expected_performance="excellent"
    )
}

class EmbeddingModelTester:
    """Test different embedding models for performance comparison"""
    
    def __init__(self):
        self.test_queries = [
            "What are the benefits of microservices architecture?",
            "How to implement OAuth 2.0 authentication?", 
            "Best practices for database indexing",
            "Docker container optimization techniques",
            "REST API design patterns",
        ]
        
        self.test_documents = [
            {
                "id": "microservices_guide",
                "title": "Microservices Architecture Guide", 
                "content": "Microservices architecture provides scalability, fault isolation, and technology diversity. Key benefits include independent deployment, better fault tolerance, and team autonomy."
            },
            {
                "id": "oauth_tutorial",
                "title": "OAuth 2.0 Implementation Tutorial",
                "content": "OAuth 2.0 is an authorization framework that enables secure API access. Implementation requires authorization server setup, client registration, and token validation."
            },
            {
                "id": "db_indexing_guide", 
                "title": "Database Indexing Best Practices",
                "content": "Database indexing improves query performance through B-tree structures, composite indices, and query optimization techniques. Proper indexing reduces table scan overhead."
            },
            {
                "id": "docker_optimization",
                "title": "Docker Container Optimization",
                "content": "Docker optimization includes multi-stage builds, layer caching, minimal base images, and resource limits. Proper container design reduces startup time and memory usage."
            },
            {
                "id": "rest_api_patterns",
                "title": "REST API Design Patterns", 
                "content": "REST API design follows stateless principles, resource-based URLs, HTTP methods, and proper status codes. Good API design includes versioning and documentation."
            }
        ]
    
    def get_recommended_models_for_testing(self) -> List[str]:
        """Get recommended models for performance testing"""
        return [
            "sentence-transformers/all-MiniLM-L6-v2",  # Current baseline
            "BAAI/bge-base-en-v1.5",                   # Better quality, same compute
            "BAAI/bge-large-en-v1.5",                  # Best quality, more compute
        ]
    
    def create_model_test_config(self, model_key: str) -> Dict[str, Any]:
        """Create configuration for testing a specific model"""
        if model_key not in EMBEDDING_MODELS:
            raise ValueError(f"Unknown model: {model_key}")
        
        config = EMBEDDING_MODELS[model_key]
        
        return {
            "embeddings": {
                "model": config.model_id,
                "dimensions": config.dimensions,
                "normalize": config.normalization,
                "max_sequence_length": config.max_sequence_length
            },
            "vector_store": {
                "collection_name": f"test_{config.name.lower().replace('-', '_')}",
                "distance": "cosine"
            },
            "test_metadata": {
                "model_name": config.name,
                "expected_performance": config.expected_performance,
                "test_queries": len(self.test_queries),
                "test_documents": len(self.test_documents)
            }
        }
    
    def generate_test_script(self, models_to_test: Optional[List[str]] = None) -> str:
        """Generate a test script for comparing embedding models"""
        if not models_to_test:
            models_to_test = self.get_recommended_models_for_testing()
        
        script_lines = [
            "#!/bin/bash",
            "# Embedding model comparison test script",
            "# Generated automatically for Phase 2 triage improvements",
            "",
            "set -euo pipefail",
            "",
            "echo '🧪 Testing embedding models for Phase 2 improvements...'",
            "echo '=' * 60",
            "",
        ]
        
        for model_key in models_to_test:
            if model_key in EMBEDDING_MODELS:
                config = EMBEDDING_MODELS[model_key]
                script_lines.extend([
                    f"echo '🔄 Testing {config.name}...'",
                    f"export EMBEDDING_MODEL='{config.model_id}'",
                    f"export EMBEDDING_DIMENSIONS='{config.dimensions}'",
                    f"export MAX_SEQUENCE_LENGTH='{config.max_sequence_length}'",
                    "",
                    "# Run test suite with this model",
                    "python3 -m src.mcp_server.main --test-mode &",
                    "SERVER_PID=$!",
                    "sleep 10  # Wait for startup",
                    "",
                    "# Run Phase 2 tests",
                    "python3 phase2-test-suite.py --model-test",
                    "",
                    "kill $SERVER_PID",
                    "wait $SERVER_PID 2>/dev/null || true",
                    "",
                    f"echo '✅ {config.name} test completed'",
                    "echo",
                    ""
                ])
        
        script_lines.extend([
            "echo '📊 All model tests completed!'",
            "echo 'Check phase2-results-*.json files for comparison'",
        ])
        
        return "\n".join(script_lines)

def create_embedding_model_configs():
    """Create configuration files for different embedding models"""
    tester = EmbeddingModelTester()
    
    configs = {}
    for model_key in tester.get_recommended_models_for_testing():
        config = tester.create_model_test_config(model_key)
        config_name = EMBEDDING_MODELS[model_key].name.lower().replace('-', '_')
        configs[config_name] = config
    
    return configs

def print_model_recommendations():
    """Print model recommendations for Phase 2 improvements"""
    print("🎯 Embedding Model Recommendations for Phase 2:")
    print("=" * 50)
    
    for model_key, config in EMBEDDING_MODELS.items():
        if config.expected_performance in ['good', 'excellent']:
            print(f"\n📈 {config.name}")
            print(f"   Model: {config.model_id}")
            print(f"   Dimensions: {config.dimensions}")
            print(f"   Max Length: {config.max_sequence_length}")
            print(f"   Performance: {config.expected_performance}")
            
            if config.expected_performance == 'excellent':
                print("   ⭐ Recommended for production")

if __name__ == "__main__":
    print_model_recommendations()
    
    # Generate test script
    tester = EmbeddingModelTester()
    script = tester.generate_test_script()
    
    with open("/workspaces/agent-context-template/test_embedding_models.sh", "w") as f:
        f.write(script)
    
    print(f"\n📝 Test script generated: test_embedding_models.sh")