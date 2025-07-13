#!/usr/bin/env python3
"""
Direct Milvus debug script to investigate partnership collection content
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def debug_milvus_partnership():
    """Debug the partnership collection directly"""
    
    try:
        # Import required modules
        from app.core.vector_db_settings_cache import get_vector_db_settings
        from pymilvus import Collection, connections
        
        # Get vector DB settings
        print("Getting vector DB settings...")
        vector_db_settings = get_vector_db_settings()
        
        # Extract Milvus configuration
        if "milvus" in vector_db_settings:
            milvus_config = vector_db_settings["milvus"]
            uri = milvus_config.get("MILVUS_URI")
            token = milvus_config.get("MILVUS_TOKEN")
        else:
            # New format
            milvus_config = None
            for db in vector_db_settings.get("databases", []):
                if db.get("id") == "milvus" and db.get("enabled", False):
                    milvus_config = db.get("config", {})
                    break
            
            uri = milvus_config.get("MILVUS_URI") if milvus_config else None
            token = milvus_config.get("MILVUS_TOKEN") if milvus_config else None
        
        if not uri:
            print("No Milvus URI configured!")
            return
            
        print(f"Connecting to Milvus at: {uri}")
        
        # Connect to Milvus
        connections.connect(uri=uri, token=token, alias="debug_connection")
        collection = Collection("partnership", using="debug_connection")
        collection.load()
        
        print(f"Successfully connected to partnership collection")
        
        # Get collection info
        print(f"Collection entities count: {collection.num_entities}")
        
        # Test 1: Search for documents containing "tencent"
        print("\n1. Searching for documents containing 'tencent':")
        tencent_results = collection.query(
            expr='content like "%tencent%" or content like "%Tencent%"',
            output_fields=["content", "source", "doc_id", "page"],
            limit=10
        )
        
        print(f"Found {len(tencent_results)} documents with 'tencent':")
        for i, result in enumerate(tencent_results):
            print(f"  Doc {i+1}:")
            print(f"    Source: {result.get('source', 'Unknown')}")
            print(f"    Doc ID: {result.get('doc_id', 'Unknown')}")
            print(f"    Page: {result.get('page', 'Unknown')}")
            content = result.get('content', '')
            print(f"    Content preview: {content[:300]}...")
            print()
        
        # Test 2: Search for documents with "bys" in filename
        print("\n2. Searching for documents with 'bys' in source:")
        bys_results = collection.query(
            expr='source like "%bys%" or source like "%BYS%"',
            output_fields=["content", "source", "doc_id", "page"],
            limit=10
        )
        
        print(f"Found {len(bys_results)} documents with 'bys' in source:")
        for i, result in enumerate(bys_results):
            print(f"  Doc {i+1}:")
            print(f"    Source: {result.get('source', 'Unknown')}")
            print(f"    Doc ID: {result.get('doc_id', 'Unknown')}")
            print(f"    Page: {result.get('page', 'Unknown')}")
            content = result.get('content', '')
            print(f"    Content preview: {content[:300]}...")
            print()
        
        # Test 3: Search for all PDF documents
        print("\n3. Searching for all PDF documents:")
        pdf_results = collection.query(
            expr='source like "%.pdf"',
            output_fields=["source", "doc_id"],
            limit=20
        )
        
        print(f"Found {len(pdf_results)} PDF documents:")
        for i, result in enumerate(pdf_results):
            source = result.get('source', 'Unknown')
            print(f"  PDF {i+1}: {source}")
            # Check if this is the document we're looking for
            if 'tencent' in source.lower() or 'bys' in source.lower():
                print(f"    *** This might be our target document! ***")
        
        # Test 4: Search with broader terms
        print("\n4. Searching for documents containing 'partnership':")
        partnership_results = collection.query(
            expr='content like "%partnership%" or content like "%Partnership%"',
            output_fields=["content", "source", "doc_id", "page"],
            limit=10
        )
        
        print(f"Found {len(partnership_results)} documents with 'partnership':")
        for i, result in enumerate(partnership_results):
            source = result.get('source', 'Unknown')
            content = result.get('content', '')
            print(f"  Doc {i+1}: {source}")
            
            # Check if this mentions Tencent in the content
            if 'tencent' in content.lower():
                print(f"    *** Contains Tencent reference! ***")
                print(f"    Content: {content[:500]}...")
                print()
        
        # Test 5: Vector search simulation with embeddings
        print("\n5. Testing vector search for 'Tencent partnership':")
        try:
            from app.api.v1.endpoints.document import HTTPEmbeddingFunction
            from app.core.embedding_settings_cache import get_embedding_settings
            
            embedding_cfg = get_embedding_settings()
            embedding_endpoint = embedding_cfg.get("embedding_endpoint")
            
            if embedding_endpoint:
                embeddings = HTTPEmbeddingFunction(embedding_endpoint)
                query_embedding = embeddings.embed_query("Tencent partnership")
                
                search_results = collection.search(
                    data=[query_embedding],
                    anns_field="vector",
                    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                    limit=5,
                    output_fields=["content", "source", "doc_id", "page"]
                )
                
                print(f"Vector search found {len(search_results[0])} results:")
                for i, result in enumerate(search_results[0]):
                    print(f"  Result {i+1}:")
                    print(f"    Score: {result.score}")
                    print(f"    Source: {result.entity.get('source', 'Unknown')}")
                    print(f"    Content: {result.entity.get('content', '')[:300]}...")
                    print()
            else:
                print("No embedding endpoint configured")
                
        except Exception as e:
            print(f"Vector search failed: {e}")
        
        # Disconnect
        connections.disconnect(alias="debug_connection")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_milvus_partnership()