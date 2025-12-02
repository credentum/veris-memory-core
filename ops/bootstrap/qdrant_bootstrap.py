#!/usr/bin/env python3
"""
Qdrant Bootstrap Guard - Idempotent collection creation with text index.
Ensures Qdrant collection exists with correct configuration.
"""

import sys
import time
import argparse
import json
from typing import Dict, Optional
import requests
from dataclasses import dataclass


@dataclass
class QdrantConfig:
    """Qdrant collection configuration."""
    collection_name: str = "context_embeddings"
    vector_size: int = 384
    distance: str = "Cosine"
    qdrant_url: str = "http://localhost:6333"
    text_field: str = "content"
    on_disk_payload: bool = True
    quantization: Optional[Dict] = None


class QdrantBootstrap:
    def __init__(self, config: QdrantConfig):
        self.config = config
        self.base_url = config.qdrant_url.rstrip('/')
        
    def check_health(self) -> bool:
        """Check if Qdrant is healthy."""
        try:
            # Qdrant doesn't have /health, but root endpoint returns info
            response = requests.get(f"{self.base_url}/", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"❌ Qdrant health check failed: {e}")
            return False
    
    def collection_exists(self) -> bool:
        """Check if collection already exists."""
        try:
            response = requests.get(
                f"{self.base_url}/collections/{self.config.collection_name}",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                return data.get('status') == 'ok'
            return False
        except requests.exceptions.RequestException:
            return False
    
    def get_collection_config(self) -> Optional[Dict]:
        """Get existing collection configuration."""
        try:
            response = requests.get(
                f"{self.base_url}/collections/{self.config.collection_name}",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'ok':
                    return data.get('result', {}).get('config', {})
            return None
        except requests.exceptions.RequestException:
            return None
    
    def validate_existing_config(self, existing_config: Dict) -> tuple[bool, list]:
        """Validate existing collection configuration against expected."""
        issues = []
        
        # Check vector size
        params = existing_config.get('params', {})
        vectors = params.get('vectors', {})
        
        if isinstance(vectors, dict):
            # Named vectors format
            size = vectors.get('size', 0)
            distance = vectors.get('distance', '')
        else:
            # Single vector format
            size = vectors
            distance = params.get('distance', '')
        
        if size != self.config.vector_size:
            issues.append(f"Vector size mismatch: expected {self.config.vector_size}, got {size}")
        
        # Distance is case-insensitive comparison
        if distance.lower() != self.config.distance.lower():
            issues.append(f"Distance mismatch: expected {self.config.distance}, got {distance}")
        
        return len(issues) == 0, issues
    
    def create_collection(self) -> bool:
        """Create new collection with specified configuration."""
        collection_config = {
            "vectors": {
                "size": self.config.vector_size,
                "distance": self.config.distance
            },
            "on_disk_payload": self.config.on_disk_payload
        }
        
        # Add quantization if specified
        if self.config.quantization:
            collection_config["quantization_config"] = self.config.quantization
        
        try:
            response = requests.put(
                f"{self.base_url}/collections/{self.config.collection_name}",
                json=collection_config,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('status') == 'ok'
            else:
                print(f"❌ Failed to create collection: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error creating collection: {e}")
            return False
    
    def create_text_index(self) -> bool:
        """Create text index on content field."""
        index_config = {
            "field_name": self.config.text_field,
            "field_schema": {
                "type": "text",
                "tokenizer": "word",
                "min_token_len": 2,
                "max_token_len": 20,
                "lowercase": True
            }
        }
        
        try:
            response = requests.put(
                f"{self.base_url}/collections/{self.config.collection_name}/index",
                json=index_config,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('status') == 'ok'
            elif response.status_code == 400:
                # Index might already exist
                if "already exists" in response.text.lower():
                    print(f"ℹ️  Text index already exists on '{self.config.text_field}'")
                    return True
            
            print(f"⚠️  Failed to create text index: {response.text}")
            return False
            
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Error creating text index: {e}")
            return False
    
    def check_text_index(self) -> bool:
        """Check if text index exists on content field."""
        try:
            response = requests.get(
                f"{self.base_url}/collections/{self.config.collection_name}",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                result = data.get('result', {})
                payload_schema = result.get('payload_schema', {})
                
                # Check if text field has index
                if self.config.text_field in payload_schema:
                    field_config = payload_schema[self.config.text_field]
                    # Check if it's configured as text with index
                    if isinstance(field_config, dict):
                        data_type = field_config.get('data_type')
                        indexed = field_config.get('indexed', False)
                        if data_type == 'text' or indexed:
                            return True
                
            return False
            
        except requests.exceptions.RequestException:
            return False
    
    def bootstrap(self, ensure_collection: bool = False, require_text_index: bool = True) -> bool:
        """
        Bootstrap Qdrant collection with proper configuration.
        
        Args:
            ensure_collection: If True, create collection if it doesn't exist
            require_text_index: If True, ensure text index exists
            
        Returns:
            True if bootstrap successful, False otherwise
        """
        print(f"🔧 Bootstrapping Qdrant collection '{self.config.collection_name}'...")
        
        # Step 1: Check Qdrant health
        if not self.check_health():
            print("❌ Qdrant is not healthy. Please ensure Qdrant is running.")
            return False
        
        print("✅ Qdrant is healthy")
        
        # Step 2: Check if collection exists
        exists = self.collection_exists()
        
        if exists:
            print(f"ℹ️  Collection '{self.config.collection_name}' already exists")
            
            # Validate configuration
            existing_config = self.get_collection_config()
            if existing_config:
                valid, issues = self.validate_existing_config(existing_config)
                
                if not valid:
                    print("❌ Existing collection configuration mismatch:")
                    for issue in issues:
                        print(f"   - {issue}")
                    print("\n⚠️  Manual intervention required:")
                    print(f"   1. Delete collection: DELETE {self.base_url}/collections/{self.config.collection_name}")
                    print(f"   2. Re-run bootstrap to create with correct config")
                    return False
                else:
                    print(f"✅ Collection configuration validated (dim={self.config.vector_size}, distance={self.config.distance})")
        
        elif ensure_collection:
            print(f"📦 Creating collection '{self.config.collection_name}'...")
            
            if not self.create_collection():
                print("❌ Failed to create collection")
                return False
            
            print(f"✅ Collection created (dim={self.config.vector_size}, distance={self.config.distance})")
            
            # Wait for collection to be ready
            time.sleep(2)
        
        else:
            print(f"⚠️  Collection '{self.config.collection_name}' does not exist")
            print("   Use --ensure-collection flag to create it")
            return False
        
        # Step 3: Handle text index
        if require_text_index:
            has_index = self.check_text_index()
            
            if not has_index:
                print(f"📝 Creating text index on '{self.config.text_field}' field...")
                
                if self.create_text_index():
                    print(f"✅ Text index created on '{self.config.text_field}'")
                else:
                    print(f"⚠️  Could not create text index (may already exist)")
            else:
                print(f"✅ Text index exists on '{self.config.text_field}'")
        
        print(f"\n✅ Bootstrap complete for collection '{self.config.collection_name}'")
        return True
    
    def get_collection_stats(self) -> Optional[Dict]:
        """Get collection statistics."""
        try:
            response = requests.get(
                f"{self.base_url}/collections/{self.config.collection_name}",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                result = data.get('result', {})
                
                stats = {
                    'vectors_count': result.get('vectors_count', 0),
                    'indexed_vectors_count': result.get('indexed_vectors_count', 0),
                    'points_count': result.get('points_count', 0),
                    'segments_count': result.get('segments_count', 0),
                    'status': result.get('status', 'unknown')
                }
                
                return stats
            
            return None
            
        except requests.exceptions.RequestException:
            return None


def main():
    parser = argparse.ArgumentParser(description='Bootstrap Qdrant collection with proper configuration')
    parser.add_argument('--collection', default='context_embeddings',
                        help='Collection name (default: context_embeddings)')
    parser.add_argument('--dimensions', type=int, default=384,
                        help='Vector dimensions (default: 384)')
    parser.add_argument('--distance', default='Cosine',
                        choices=['Cosine', 'Euclidean', 'Dot'],
                        help='Distance metric (default: Cosine)')
    parser.add_argument('--qdrant-url', default='http://localhost:6333',
                        help='Qdrant URL (default: http://localhost:6333)')
    parser.add_argument('--text-field', default='content',
                        help='Field name for text index (default: content)')
    parser.add_argument('--ensure-collection', action='store_true',
                        help='Create collection if it does not exist')
    parser.add_argument('--skip-text-index', action='store_true',
                        help='Skip text index creation')
    parser.add_argument('--stats', action='store_true',
                        help='Show collection statistics after bootstrap')
    
    args = parser.parse_args()
    
    # Create configuration
    config = QdrantConfig(
        collection_name=args.collection,
        vector_size=args.dimensions,
        distance=args.distance,
        qdrant_url=args.qdrant_url,
        text_field=args.text_field
    )
    
    # Run bootstrap
    bootstrap = QdrantBootstrap(config)
    success = bootstrap.bootstrap(
        ensure_collection=args.ensure_collection,
        require_text_index=not args.skip_text_index
    )
    
    if success and args.stats:
        print("\n📊 Collection Statistics:")
        stats = bootstrap.get_collection_stats()
        if stats:
            for key, value in stats.items():
                print(f"   {key}: {value}")
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()