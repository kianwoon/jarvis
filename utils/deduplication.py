import hashlib
from typing import List, Set
from pymilvus import Collection

def hash_text(text: str) -> str:
    """Return SHA256 hash of the input text (case-insensitive)."""
    # Convert to lowercase for consistent hashing
    normalized_text = text.lower().strip()
    return hashlib.sha256(normalized_text.encode()).hexdigest()

def get_existing_hashes(collection: Collection, limit: int = 10000) -> Set[str]:
    """Query all existing chunk hashes from the collection."""
    try:
        # Use a proper expression that matches all records
        results = collection.query(expr="hash != ''", output_fields=["hash"], limit=limit)
        return set(r["hash"] for r in results if "hash" in r and r["hash"])
    except Exception as e:
        print(f"Warning: Failed to query existing hashes: {e}")
        return set()

def get_existing_doc_ids(collection: Collection, limit: int = 10000) -> Set[str]:
    """Query all existing doc_ids from the collection."""
    try:
        # Use a proper expression that matches all records
        results = collection.query(expr="doc_id != ''", output_fields=["doc_id"], limit=limit)
        return set(r["doc_id"] for r in results if "doc_id" in r and r["doc_id"])
    except Exception as e:
        print(f"Warning: Failed to query existing doc_ids: {e}")
        return set()

def filter_new_chunks(chunks, existing_hashes: Set[str], existing_doc_ids: Set[str]) -> List:
    """Return only chunks that are not present in the collection by hash or doc_id."""
    return [
        chunk for chunk in chunks
        if chunk.metadata.get("hash") not in existing_hashes and chunk.metadata.get("doc_id") not in existing_doc_ids
    ] 