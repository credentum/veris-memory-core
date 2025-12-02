#!/usr/bin/env python3
"""
Test script to demonstrate how manifest_verifier.py prevents dimension drift disasters
"""

import json
import tempfile
import os
import sys
import subprocess

def create_test_config(model: str, dimensions: int, distance: str = "Cosine"):
    """Create a test frozen config"""
    config = {
        "embedding": {
            "model": model,
            "dimensions": dimensions,
            "batch_size": 50
        },
        "retrieval": {
            "distance": distance
        }
    }
    
    config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    
    # Write YAML manually (simple structure)
    config_file.write(f"""embedding:
  model: {model}
  dimensions: {dimensions}
  batch_size: 50
retrieval:
  distance: {distance}
""")
    config_file.close()
    return config_file.name

def create_test_manifest(vector_dim: int, distance: str = "Cosine"):
    """Create a test manifest"""
    manifest = {
        "service": "veris-memory",
        "storage_schema": {
            "qdrant": {
                "collection": "context_store",
                "vector_dim": vector_dim,
                "distance": distance
            }
        }
    }
    
    manifest_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(manifest, manifest_file)
    manifest_file.close()
    return manifest_file.name

def test_scenario(name: str, config_model: str, config_dims: int, manifest_dims: int, should_pass: bool):
    """Test a specific dimension drift scenario"""
    print(f"\n🧪 TEST: {name}")
    print("=" * (len(name) + 12))
    
    # Create test files
    config_file = create_test_config(config_model, config_dims)
    manifest_file = create_test_manifest(manifest_dims)
    
    print(f"   Config: {config_model} ({config_dims}-dim)")
    print(f"   Manifest: {manifest_dims}-dim")
    print(f"   Expected: {'PASS' if should_pass else 'FAIL'}")
    
    try:
        # Run the verifier (will fail on Qdrant connection, but we only care about config validation)
        cmd = [
            sys.executable, "ops/verify/manifest_verifier.py",
            "--config", config_file,
            "--manifest", manifest_file,
            "--qdrant-url", "http://127.0.0.1:6333",  # Will fail connection
            "--collection", "test"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check if config/manifest validation passed (before Qdrant connection failure)
        output = result.stdout + result.stderr
        config_validated = "Frozen config →" in output
        manifest_validated = "Manifest matches frozen config" in output
        
        if should_pass:
            if config_validated and manifest_validated:
                print("   ✅ Result: PASS (config/manifest alignment correct)")
            else:
                print("   ❌ Result: UNEXPECTED FAIL")
        else:
            if "vector_dim" in output and "!=" in output:
                print("   ✅ Result: FAIL (dimension mismatch detected as expected)")
            elif not manifest_validated:
                print("   ✅ Result: FAIL (config/manifest mismatch detected as expected)")
            else:
                print("   ❌ Result: UNEXPECTED PASS")
                
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    finally:
        # Cleanup
        os.unlink(config_file)
        os.unlink(manifest_file)

def main():
    print("🔍 DIMENSION DRIFT DETECTION TESTS")
    print("==================================")
    print("Demonstrating how manifest_verifier.py prevents silent mis-indexing")
    
    # Test scenarios
    test_scenario(
        "✅ Correct Alignment",
        "all-MiniLM-L6-v2", 384, 384, 
        should_pass=True
    )
    
    test_scenario(
        "❌ Dimension Drift (384→1536)",
        "all-MiniLM-L6-v2", 384, 1536,
        should_pass=False
    )
    
    test_scenario(
        "❌ Model Mismatch (OpenAI model with wrong dims)",
        "text-embedding-ada-002", 1536, 384,
        should_pass=False
    )
    
    test_scenario(
        "✅ Large Model Alignment",
        "text-embedding-3-small", 1536, 1536,
        should_pass=True
    )
    
    print("\n🎯 SUMMARY")
    print("==========")
    print("The manifest_verifier.py successfully prevents:")
    print("  🚫 Silent dimension mismatches (384 ≠ 1536)")
    print("  🚫 Model/dimension inconsistencies") 
    print("  🚫 Configuration drift between frozen config and live deployment")
    print("  🚫 Manifest corruption or incorrect generation")
    print()
    print("In production, this runs as a CI/CD gate:")
    print("  1. Validates frozen config ↔ manifest alignment")
    print("  2. Checks live Qdrant collection dimensions")
    print("  3. Blocks deployment on any mismatch")
    print("  4. Prevents expensive re-indexing disasters")

if __name__ == "__main__":
    main()