#!/usr/bin/env python3
"""
Direct test of the RAG search with proper embedding
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_direct_rag_search():
    """Test RAG search functionality directly"""
    print("=== Direct RAG Search Test ===")
    
    try:
        from app.core.rag_fallback import HTTPEmbedder, search_milvus_collection
        from app.core.vector_db_settings_cache import get_vector_db_settings
        
        # Create HTTP embedder with localhost endpoint
        print("1. Creating HTTP embedder...")
        embedder = HTTPEmbedder("http://localhost:8050/embed")
        
        # Test embedding generation
        print("2. Testing embedding generation...")
        query = "BeyondSoft and Alibaba partnership"
        embeddings = embedder.encode([query])
        
        if embeddings:
            embedding_dim = len(embeddings[0]) if embeddings[0] else 0
            print(f"Generated embedding with {embedding_dim} dimensions")
            
            if embedding_dim == 2560:
                print("✓ Embedding dimensions match expected (2560)")
                
                # Get vector database settings
                print("3. Getting vector database settings...")
                vector_settings = get_vector_db_settings()
                
                # Search Milvus collection
                print("4. Searching partnership collection...")
                results = search_milvus_collection(
                    collection_name="partnership",
                    query_embedding=embeddings[0],
                    vector_settings=vector_settings,
                    max_docs=5
                )
                
                print(f"Found {len(results)} results:")
                for i, result in enumerate(results):
                    print(f"\nResult {i+1}:")
                    print(f"  Title: {result['title']}")
                    print(f"  Score: {result['score']:.3f}")
                    content = result['content'][:300]
                    print(f"  Content: {content}...")
                    print(f"  Source: {result['metadata'].get('source', 'N/A')}")
                
                if results:
                    print(f"\n✓ RAG search successfully found {len(results)} documents!")
                    print("The BeyondSoft-Alibaba partnership information IS in the knowledge base.")
                else:
                    print("\n⚠ No results found, but embeddings are working correctly")
                
            else:
                print(f"✗ Embedding dimensions mismatch: got {embedding_dim}, expected 2560")
        else:
            print("✗ Failed to generate embeddings")
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct_rag_search()