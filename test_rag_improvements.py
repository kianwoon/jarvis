#!/usr/bin/env python3
"""Test script to verify RAG improvements for queries like 'dbs outages'"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_rag_queries():
    """Test various queries to ensure RAG improvements work"""
    
    # Import here to avoid circular import
    from app.langchain.service import rag_answer
    
    test_queries = [
        "dbs outages",
        "DBS outages",
        "tell me about DBS bank outages",
        "what happened with DBS",
        "banking outages Singapore"
    ]
    
    print("Testing RAG improvements with various queries...\n")
    
    for query in test_queries:
        print(f"Query: '{query}'")
        print("-" * 50)
        
        try:
            # Test without LangGraph (original implementation)
            result = rag_answer(query, thinking=False, stream=False, use_langgraph=False)
            
            if isinstance(result, dict):
                print(f"Source: {result.get('source', 'Unknown')}")
                print(f"Context found: {'Yes' if result.get('context') else 'No'}")
                if result.get('context'):
                    print(f"Context preview: {result['context'][:200]}...")
                print(f"Answer preview: {result.get('answer', '')[:200]}...")
            else:
                print("Streaming response received")
                
        except Exception as e:
            print(f"Error: {str(e)}")
        
        print("\n")
    
    # Test keyword search directly
    print("\nTesting direct keyword search for 'dbs outages':")
    print("-" * 50)
    
    try:
        from app.langchain.service import keyword_search_milvus
        from app.core.vector_db_settings_cache import get_vector_db_settings
        
        vector_db_cfg = get_vector_db_settings()
        milvus_cfg = vector_db_cfg["milvus"]
        
        docs = keyword_search_milvus(
            "dbs outages",
            milvus_cfg.get("MILVUS_DEFAULT_COLLECTION", "default_knowledge"),
            uri=milvus_cfg.get("MILVUS_URI"),
            token=milvus_cfg.get("MILVUS_TOKEN")
        )
        
        print(f"Found {len(docs)} documents with keyword search")
        for i, doc in enumerate(docs[:3]):
            print(f"\nDoc {i+1}:")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            
    except Exception as e:
        print(f"Keyword search error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_rag_queries())