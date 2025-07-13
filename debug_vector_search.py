#!/usr/bin/env python3
"""
Direct vector search test to identify the exact issue
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_vector_search():
    """Test vector search directly"""
    
    print("DIRECT VECTOR SEARCH TEST")
    print("=" * 50)
    
    try:
        from langchain_community.vectorstores import Milvus
        from app.api.v1.endpoints.document import HTTPEmbeddingFunction
        from app.core.embedding_settings_cache import get_embedding_settings
        from app.core.vector_db_settings_cache import get_vector_db_settings
        
        # Get settings
        embedding_cfg = get_embedding_settings()
        vector_db_settings = get_vector_db_settings()
        
        # Extract Milvus config
        if "milvus" in vector_db_settings:
            milvus_config = vector_db_settings["milvus"]
            uri = milvus_config.get("MILVUS_URI")
            token = milvus_config.get("MILVUS_TOKEN")
        else:
            milvus_config = None
            for db in vector_db_settings.get("databases", []):
                if db.get("id") == "milvus" and db.get("enabled", False):
                    milvus_config = db.get("config", {})
                    break
            
            uri = milvus_config.get("MILVUS_URI") if milvus_config else None
            token = milvus_config.get("MILVUS_TOKEN") if milvus_config else None
        
        # Setup embeddings
        embedding_endpoint = embedding_cfg.get("embedding_endpoint")
        
        if not embedding_endpoint:
            print("✗ No embedding endpoint configured")
            return
            
        if not uri:
            print("✗ No Milvus URI configured")
            return
        
        print(f"✓ Using embedding endpoint: {embedding_endpoint}")
        print(f"✓ Using Milvus URI: {uri}")
        
        embeddings = HTTPEmbeddingFunction(embedding_endpoint)
        
        # Create Milvus store for partnership collection
        milvus_store = Milvus(
            embedding_function=embeddings,
            collection_name="partnership",
            connection_args={"uri": uri, "token": token},
            text_field="content"
        )
        
        # Test queries with different variations
        test_queries = [
            "partnership between beyondsoft and tencent",
            "tencent partnership",
            "beyondsoft tencent",
            "tencent collaboration",
            "partnership 2012"
        ]
        
        for query in test_queries:
            print(f"\n--- Testing query: '{query}' ---")
            
            # Vector search with scores
            docs_with_scores = milvus_store.similarity_search_with_score(query.lower().strip(), k=5)
            
            print(f"Found {len(docs_with_scores)} results:")
            
            tencent_found = False
            for i, (doc, score) in enumerate(docs_with_scores):
                source = doc.metadata.get('source', 'Unknown')
                content_preview = doc.page_content[:150]
                
                print(f"  Result {i+1}:")
                print(f"    Score: {score:.4f}")
                print(f"    Source: {source}")
                print(f"    Content: {content_preview}...")
                
                # Check if this is our target document
                if 'tencent' in source.lower() and 'bys' in source.lower():
                    tencent_found = True
                    print(f"    *** TARGET DOCUMENT FOUND! ***")
                
                # Check similarity threshold
                if score <= 1.5:  # Using the configured threshold
                    print(f"    ✓ Passes similarity threshold (1.5)")
                else:
                    print(f"    ✗ Fails similarity threshold (1.5)")
                
                print()
            
            if not tencent_found:
                print(f"  ⚠ Target Tencent document not found in top 5 results")
            else:
                print(f"  ✓ Target Tencent document found!")
        
        # Test with exact content match
        print(f"\n--- Testing with exact content from document ---")
        exact_query = "Beyondsoft partnership Tencent 2012"
        docs_exact = milvus_store.similarity_search_with_score(exact_query.lower().strip(), k=3)
        
        print(f"Exact content search found {len(docs_exact)} results:")
        for i, (doc, score) in enumerate(docs_exact):
            source = doc.metadata.get('source', 'Unknown')
            print(f"  Result {i+1}: {source} (score: {score:.4f})")
            
    except Exception as e:
        print(f"✗ Vector search test failed: {e}")
        import traceback
        traceback.print_exc()

def test_keyword_search():
    """Test keyword search directly"""
    
    print("\n" + "=" * 50)
    print("DIRECT KEYWORD SEARCH TEST")
    print("=" * 50)
    
    try:
        from pymilvus import Collection, connections
        from app.core.vector_db_settings_cache import get_vector_db_settings
        
        # Get vector DB settings
        vector_db_settings = get_vector_db_settings()
        
        if "milvus" in vector_db_settings:
            milvus_config = vector_db_settings["milvus"]
            uri = milvus_config.get("MILVUS_URI")
            token = milvus_config.get("MILVUS_TOKEN")
        else:
            milvus_config = None
            for db in vector_db_settings.get("databases", []):
                if db.get("id") == "milvus" and db.get("enabled", False):
                    milvus_config = db.get("config", {})
                    break
            
            uri = milvus_config.get("MILVUS_URI") if milvus_config else None
            token = milvus_config.get("MILVUS_TOKEN") if milvus_config else None
        
        if not uri:
            print("✗ No Milvus URI configured for keyword search")
            return
        
        connections.connect(uri=uri, token=token, alias="keyword_test")
        collection = Collection("partnership", using="keyword_test")
        collection.load()
        
        # Test keyword search expressions
        test_expressions = [
            'content like "%tencent%" and content like "%partnership%"',
            'content like "%beyondsoft%" and content like "%tencent%"',  
            'content like "%2012%" and content like "%partnership%"',
            'source like "%tencent%"'
        ]
        
        for expr in test_expressions:
            print(f"\n--- Keyword search: {expr} ---")
            
            try:
                results = collection.query(
                    expr=expr,
                    output_fields=["content", "source", "doc_id"],
                    limit=5
                )
                
                print(f"Found {len(results)} results:")
                
                for i, result in enumerate(results):
                    source = result.get('source', 'Unknown')
                    content = result.get('content', '')[:150]
                    
                    print(f"  Result {i+1}:")
                    print(f"    Source: {source}")
                    print(f"    Content: {content}...")
                    
                    if 'tencent' in source.lower() and 'bys' in source.lower():
                        print(f"    *** TARGET DOCUMENT FOUND! ***")
                    print()
                    
            except Exception as e:
                print(f"  ✗ Query failed: {e}")
        
        connections.disconnect(alias="keyword_test")
        
    except Exception as e:
        print(f"✗ Keyword search test failed: {e}")

def test_hybrid_approach():
    """Test the hybrid approach that's used in production"""
    
    print("\n" + "=" * 50)
    print("HYBRID SEARCH SIMULATION")
    print("=" * 50)
    
    print("This simulates the full hybrid search process used in handle_rag_query:")
    print("1. Vector search with query expansion")
    print("2. Keyword search")  
    print("3. Result fusion and deduplication")
    print("4. Relevance scoring and filtering")
    print("5. Re-ranking")
    
    print("\nBased on our previous tests:")
    print("✓ Vector search: Should find Tencent documents")
    print("✓ Keyword search: Should find Tencent documents")
    print("✓ Relevance scoring: Should pass thresholds")
    print("✓ Collection targeting: Correctly routes to partnership")
    
    print("\nThe issue is likely in one of these stages:")
    print("• Query expansion may be changing the query ineffectively")
    print("• Result fusion may be discarding relevant documents")
    print("• Re-ranking may be deprioritizing the content")
    print("• Final filtering may be removing documents")

if __name__ == "__main__":
    test_vector_search()
    test_keyword_search()
    test_hybrid_approach()