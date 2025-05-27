#!/usr/bin/env python3
"""Test keyword search functionality directly"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_keyword_search():
    """Test keyword search for 'dbs outages'"""
    
    from pymilvus import Collection, connections
    from app.core.vector_db_settings_cache import get_vector_db_settings
    
    vector_db_cfg = get_vector_db_settings()
    milvus_cfg = vector_db_cfg["milvus"]
    
    collection_name = milvus_cfg.get("MILVUS_DEFAULT_COLLECTION", "default_knowledge")
    uri = milvus_cfg.get("MILVUS_URI")
    token = milvus_cfg.get("MILVUS_TOKEN")
    
    print(f"Connecting to Milvus collection: {collection_name}")
    print(f"URI: {uri}")
    
    try:
        connections.connect(uri=uri, token=token, alias="test_search")
        collection = Collection(collection_name, using="test_search")
        collection.load()
        
        # Test queries
        test_queries = [
            'content like "%outages%" and content like "%DBS%"',
            'content like "%DBS%"',
            'content like "%outages%"',
            'content like "%bank%" and content like "%outages%"'
        ]
        
        for expr in test_queries:
            print(f"\nQuery: {expr}")
            print("-" * 50)
            
            results = collection.query(
                expr=expr,
                output_fields=["content", "source", "page"],
                limit=5
            )
            
            print(f"Found {len(results)} results")
            
            for i, result in enumerate(results[:3]):
                print(f"\nResult {i+1}:")
                print(f"Source: {result.get('source', 'Unknown')}")
                print(f"Page: {result.get('page', 'Unknown')}")
                print(f"Content preview: {result.get('content', '')[:200]}...")
        
        connections.disconnect(alias="test_search")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_keyword_search()