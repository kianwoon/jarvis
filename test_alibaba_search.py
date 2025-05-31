#!/usr/bin/env python3
"""
Test script to investigate why "bys & alibaba partnership.docx" isn't found when searching
"""

import os
import json
from pymilvus import connections, Collection, utility
from app.core.vector_db_settings_cache import get_vector_db_settings
from app.core.embedding_settings_cache import get_embedding_settings
from app.api.v1.endpoints.document import HTTPEmbeddingFunction
from langchain_community.embeddings import HuggingFaceEmbeddings

def test_milvus_search():
    """Test direct Milvus search for alibaba documents"""
    print("=== Testing Milvus Search for Alibaba Documents ===\n")
    
    # Get configuration
    vector_db_cfg = get_vector_db_settings()
    milvus_cfg = vector_db_cfg["milvus"]
    uri = milvus_cfg.get("MILVUS_URI")
    token = milvus_cfg.get("MILVUS_TOKEN")
    collection_name = milvus_cfg.get("MILVUS_DEFAULT_COLLECTION", "default_knowledge")
    
    print(f"Connecting to Milvus collection: {collection_name}")
    print(f"URI: {uri}\n")
    
    # Connect to Milvus
    connections.connect(uri=uri, token=token)
    
    if not utility.has_collection(collection_name):
        print(f"ERROR: Collection '{collection_name}' does not exist!")
        return
    
    collection = Collection(collection_name)
    collection.load()
    
    print(f"Collection loaded. Total entities: {collection.num_entities}\n")
    
    # Test 1: Search for documents with "alibaba" in source field
    print("Test 1: Searching for 'alibaba' in source field...")
    expr = 'source like "%alibaba%"'
    
    try:
        results = collection.query(
            expr=expr,
            output_fields=["content", "source", "page", "hash", "doc_id"],
            limit=100
        )
        
        print(f"Found {len(results)} documents with 'alibaba' in source field:\n")
        
        # Group by source
        sources = {}
        for r in results:
            source = r.get("source", "unknown")
            if source not in sources:
                sources[source] = []
            sources[source].append(r)
        
        for source, docs in sources.items():
            print(f"\nSource: {source}")
            print(f"  Documents: {len(docs)}")
            if docs:
                print(f"  Sample content: {docs[0].get('content', '')[:200]}...")
                
    except Exception as e:
        print(f"Error in Test 1: {str(e)}")
    
    # Test 2: Search for variations of the filename
    print("\n\nTest 2: Searching for variations of the filename...")
    variations = [
        'bys & alibaba partnership.docx',
        'bys alibaba partnership.docx',
        'bys&alibaba partnership.docx',
        'bys and alibaba partnership.docx',
        'alibaba partnership.docx',
        'bys alibaba.docx'
    ]
    
    for variant in variations:
        # Try both original and lowercase
        for test_variant in [variant, variant.lower()]:
            expr = f'source = "{test_variant}"'
            try:
                results = collection.query(
                    expr=expr,
                    output_fields=["content", "source", "page"],
                    limit=10
                )
                if results:
                    print(f"  ✓ Found {len(results)} docs for: {test_variant}")
                    break
            except Exception as e:
                print(f"  ✗ Error searching for '{test_variant}': {str(e)}")
    
    # Test 3: List all unique sources
    print("\n\nTest 3: Listing all unique sources containing 'bys' or 'alibaba'...")
    
    # Search for documents with these keywords in source
    for keyword in ['bys', 'alibaba', 'partnership']:
        expr = f'source like "%{keyword}%"'
        try:
            results = collection.query(
                expr=expr,
                output_fields=["source"],
                limit=1000
            )
            
            unique_sources = set(r.get("source", "") for r in results)
            if unique_sources:
                print(f"\nSources containing '{keyword}':")
                for source in sorted(unique_sources):
                    print(f"  - {source}")
        except Exception as e:
            print(f"\nError searching for '{keyword}': {str(e)}")
    
    # Test 4: Search in content for alibaba-related terms
    print("\n\nTest 4: Searching for 'alibaba' in content...")
    expr = 'content like "%alibaba%"'
    
    try:
        results = collection.query(
            expr=expr,
            output_fields=["content", "source", "page"],
            limit=20
        )
        
        print(f"Found {len(results)} documents with 'alibaba' in content:")
        
        # Group by source
        content_sources = {}
        for r in results:
            source = r.get("source", "unknown")
            if source not in content_sources:
                content_sources[source] = 0
            content_sources[source] += 1
        
        for source, count in content_sources.items():
            print(f"  - {source}: {count} chunks")
            
    except Exception as e:
        print(f"Error in Test 4: {str(e)}")
    
    # Test 5: Vector search for "alibaba organisation structure"
    print("\n\nTest 5: Vector search for 'alibaba organisation structure'...")
    
    # Set up embeddings
    embedding_cfg = get_embedding_settings()
    embedding_endpoint = embedding_cfg.get("embedding_endpoint")
    
    if embedding_endpoint:
        embeddings = HTTPEmbeddingFunction(embedding_endpoint)
    else:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_cfg.get("embedding_model", "BAAI/bge-base-en-v1.5"))
    
    # Generate query embedding
    query = "alibaba organisation structure"
    query_embedding = embeddings.embed_query(query.lower())
    
    # Vector search
    search_results = collection.search(
        data=[query_embedding],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=10,
        output_fields=["content", "source", "page"]
    )
    
    print(f"\nVector search results for '{query}':")
    for i, hit in enumerate(search_results[0]):
        print(f"\n{i+1}. Score: {hit.score:.4f}")
        print(f"   Source: {hit.entity.get('source', 'unknown')}")
        print(f"   Page: {hit.entity.get('page', 'N/A')}")
        print(f"   Content: {hit.entity.get('content', '')[:200]}...")
    
    # Disconnect
    connections.disconnect()

def check_filesystem():
    """Check if the document exists in the filesystem"""
    print("\n\n=== Checking Filesystem ===")
    
    # Common upload directories
    dirs_to_check = [
        "/tmp",
        "/Users/kianwoonwong/Downloads/jarvis/uploads",
        "/Users/kianwoonwong/Downloads/jarvis/documents",
        "/Users/kianwoonwong/Downloads/jarvis"
    ]
    
    filename_patterns = [
        "bys & alibaba partnership.docx",
        "bys alibaba partnership.docx",
        "*alibaba*.docx",
        "*bys*.docx"
    ]
    
    import glob
    
    for directory in dirs_to_check:
        if os.path.exists(directory):
            print(f"\nChecking {directory}:")
            for pattern in filename_patterns:
                matches = glob.glob(os.path.join(directory, pattern))
                if matches:
                    print(f"  Found: {matches}")

if __name__ == "__main__":
    test_milvus_search()
    check_filesystem()