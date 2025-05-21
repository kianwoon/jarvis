import hashlib
from typing import List, Set
from pymilvus import Collection

def hash_text(text: str) -> str:
    """Return SHA256 hash of the input text."""
    return hashlib.sha256(text.encode()).hexdigest()

def get_existing_hashes(collection: Collection, limit: int = 10000) -> Set[str]:
    """Query all existing chunk hashes from the collection."""
    results = collection.query(expr="", output_fields=["hash"], limit=limit)
    return set(r["hash"] for r in results if "hash" in r)

def get_existing_doc_ids(collection: Collection, limit: int = 10000) -> Set[str]:
    """Query all existing doc_ids from the collection."""
    results = collection.query(expr="", output_fields=["doc_id"], limit=limit)
    return set(r["doc_id"] for r in results if "doc_id" in r)

def filter_new_chunks(chunks, existing_hashes: Set[str], existing_doc_ids: Set[str]) -> List:
    """Return only chunks that are not present in the collection by hash or doc_id."""
    return [
        chunk for chunk in chunks
        if chunk.metadata.get("hash") not in existing_hashes and chunk.metadata.get("doc_id") not in existing_doc_ids
    ] 