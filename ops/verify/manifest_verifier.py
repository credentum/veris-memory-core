#!/usr/bin/env python3
# ops/verify/manifest_verifier.py
import argparse, json, os, sys
from typing import Any, Dict
import requests
try:
    import yaml
except ImportError:
    print("ERROR: pyyaml not installed. pip install pyyaml", file=sys.stderr)
    sys.exit(2)

# Common model → expected dim map (override with config if present)
MODEL_DIMS = {
    "all-MiniLM-L6-v2": 384,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "bge-base-en-v1.5": 768,
    "bge-large-en-v1.5": 1024,
    "intfloat/e5-large-v2": 1024,
    "text-embedding-ada-002": 1536,
}

def fail(msg: str) -> None:
    print(f"VERIFIER: ❌ {msg}")
    sys.exit(1)

def ok(msg: str) -> None:
    print(f"VERIFIER: ✅ {msg}")

def get_qdrant_collection(qbase: str, collection: str) -> Dict[str, Any]:
    url = f"{qbase.rstrip('/')}/collections/{collection}"
    print(f"DEBUG: Fetching Qdrant collection from: {url}")
    r = requests.get(url)
    print(f"DEBUG: Response status: {r.status_code}")
    if r.status_code != 200:
        print(f"DEBUG: Response text: {r.text[:500] if r.text else 'No response text'}")
        fail(f"Qdrant: cannot fetch collection '{collection}' ({r.status_code})")
    return r.json().get("result", {})

def get_qdrant_indexes(qbase: str, collection: str) -> Dict[str, Any]:
    url = f"{qbase.rstrip('/')}/collections/{collection}/indexes"
    print(f"DEBUG: Fetching indexes from: {url}")
    r = requests.get(url)
    print(f"DEBUG: Indexes response status: {r.status_code}")
    if r.status_code == 404:
        # Indexes endpoint may not exist in newer Qdrant versions
        print(f"INFO: Indexes endpoint not available (404) - skipping text index check")
        return []
    if r.status_code != 200:
        fail(f"Qdrant: cannot fetch indexes for '{collection}' ({r.status_code})")
    return r.json().get("result", [])

def extract_vec_params(coll: Dict[str, Any]) -> Dict[str, Any]:
    """
    Qdrant returns:
      result.config.params.vectors = {"size":..., "distance":"Cosine"}
    or named vectors:
      {"vectors": {"my_vec": {"size":..., "distance":"Cosine"}}}
    We handle both.
    """
    cfg = coll.get("config", {})
    params = cfg.get("params", {}) or cfg  # older versions
    vectors = params.get("vectors") or params.get("vector_config") or {}

    # If plain dict with size/distance
    if isinstance(vectors, dict) and "size" in vectors and "distance" in vectors:
        return {"name": None, "size": vectors["size"], "distance": vectors["distance"]}

    # If named vectors
    if isinstance(vectors, dict):
        # Prefer 'default' if present, else take the first
        if "default" in vectors:
            v = vectors["default"]
            return {"name": "default", "size": v["size"], "distance": v["distance"]}
        # arbitrary first key
        k, v = next(iter(vectors.items()))
        return {"name": k, "size": v["size"], "distance": v["distance"]}

    fail("Qdrant: could not parse vector params from collection config")

def main():
    ap = argparse.ArgumentParser(description="Verify Qdrant vector params against frozen config/manifest.")
    ap.add_argument("--config", default="production_locked_config.yaml", help="Path to frozen config YAML")
    ap.add_argument("--manifest", default=None, help="Optional deployments/manifest-*.json to cross-check")
    ap.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL","http://127.0.0.1:6333"), help="Qdrant base URL")
    ap.add_argument("--collection", default=os.getenv("VECTOR_COLLECTION","context_store"), help="Qdrant collection name")
    ap.add_argument("--require-text-index", action="store_true", help="Fail if 'text' payload index not present")
    ap.add_argument("--expected-distance", default=None, help="Override expected distance (e.g., Cosine)")
    args = ap.parse_args()

    # Load frozen config
    try:
        cfg = yaml.safe_load(open(args.config))
    except FileNotFoundError:
        fail(f"Config not found: {args.config}")

    # Infer expected model + dim from config; allow explicit dim override in config
    # Expected structure (examples):
    # embedding:
    #   model_name: all-MiniLM-L6-v2 (or model: all-MiniLM-L6-v2)
    #   dimensions: 384 (or dim: 384)
    # retrieval:
    #   distance: Cosine
    embedding = (cfg.get("embedding") or {})
    expected_model = embedding.get("model_name") or embedding.get("model")
    expected_dim = embedding.get("dimensions") or embedding.get("dim")
    if expected_dim is None and expected_model:
        expected_dim = MODEL_DIMS.get(expected_model)

    if expected_dim is None:
        fail("Frozen config must specify embedding.dimensions or embedding.model (known).")

    # Expected distance
    expected_distance = args.expected_distance or (cfg.get("retrieval", {}) or {}).get("distance") or "Cosine"

    ok(f"Frozen config → model={expected_model} dim={expected_dim} distance={expected_distance}")

    # Optional manifest cross-check
    if args.manifest:
        try:
            man = json.load(open(args.manifest))
            m_vecdim = (((man.get("storage_schema") or {}).get("qdrant") or {}).get("vector_dim"))
            if m_vecdim is not None and int(m_vecdim) != int(expected_dim):
                fail(f"Manifest vector_dim ({m_vecdim}) != frozen config ({expected_dim})")
            ok("Manifest matches frozen config (vector_dim)")
        except FileNotFoundError:
            fail(f"Manifest not found: {args.manifest}")

    # Query Qdrant live collection
    coll = get_qdrant_collection(args.qdrant_url, args.collection)
    vp = extract_vec_params(coll)
    ok(f"Qdrant live → collection={args.collection} vec_name={vp['name']} size={vp['size']} distance={vp['distance']}")

    # Compare size
    if int(vp["size"]) != int(expected_dim):
        fail(f"Qdrant vector size {vp['size']} != expected {expected_dim}")

    # Compare distance (case-insensitive)
    if str(vp["distance"]).lower() != str(expected_distance).lower():
        fail(f"Qdrant distance {vp['distance']} != expected {expected_distance}")

    # Optional text index requirement
    if args.require_text_index:
        idx = get_qdrant_indexes(args.qdrant_url, args.collection)
        # Qdrant returns list of indexes; each item includes "field_name" and "field_schema"
        has_text = any(
            (it.get("field_name") == "text") or
            (isinstance(it.get("field_name"), dict) and it["field_name"].get("text") == "text")
            for it in idx
        )
        if not has_text:
            fail("Payload text index not present (require-text-index set)")
        ok("Payload text index present")

    ok("Manifest verification PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()