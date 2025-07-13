#!/usr/bin/env python3
"""
Debug script to trace the full RAG pipeline and understand why Tencent partnership content isn't appearing
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def debug_rag_pipeline():
    """Debug the full RAG pipeline step by step"""
    
    try:
        # Test query exactly as mentioned in the issue
        test_query = "partnership between beyondsoft and tencent"
        print(f"Testing query: '{test_query}'")
        print("=" * 70)
        
        # Import the core document classifier to see how collections are selected
        print("\n1. Testing document classification and collection selection:")
        try:
            from app.core.document_classifier import get_document_classifier
            classifier = get_document_classifier()
            
            # Use query analysis to determine collection type
            collection_type = classifier.classify_document(test_query, {"query": True})
            target_collection = classifier.get_target_collection(collection_type)
            
            print(f"Classified query type: {collection_type}")
            print(f"Target collection: {target_collection}")
            
        except Exception as e:
            print(f"Classification error: {e}")
        
        # Test keyword search directly
        print("\n2. Testing keyword search directly in partnership collection:")
        try:
            from app.langchain.service import keyword_search_milvus
            from app.core.vector_db_settings_cache import get_vector_db_settings
            
            vector_db_settings = get_vector_db_settings()
            
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
            
            if uri:
                keyword_docs = keyword_search_milvus(
                    query=test_query,
                    collection_name="partnership",
                    uri=uri,
                    token=token,
                    limit=10
                )
                
                print(f"Keyword search found {len(keyword_docs)} documents:")
                for i, doc in enumerate(keyword_docs[:3]):
                    print(f"  Doc {i+1}:")
                    print(f"    Source: {getattr(doc, 'metadata', {}).get('source', 'Unknown')}")
                    print(f"    Content: {doc.page_content[:200]}...")
                    print()
            else:
                print("No Milvus URI configured")
                
        except Exception as e:
            print(f"Keyword search error: {e}")
            import traceback
            traceback.print_exc()
        
        # Test vector search using simplified approach
        print("\n3. Testing vector search in partnership collection:")
        try:
            from langchain_community.vectorstores import Milvus
            from app.api.v1.endpoints.document import HTTPEmbeddingFunction
            from app.core.embedding_settings_cache import get_embedding_settings
            
            embedding_cfg = get_embedding_settings()
            embedding_endpoint = embedding_cfg.get("embedding_endpoint")
            
            if embedding_endpoint:
                embeddings = HTTPEmbeddingFunction(embedding_endpoint)
                
                # Create Milvus store for partnership collection
                milvus_store = Milvus(
                    embedding_function=embeddings,
                    collection_name="partnership",
                    connection_args={"uri": uri, "token": token},
                    text_field="content"
                )
                
                # Test similarity search
                docs = milvus_store.similarity_search_with_score(test_query.lower().strip(), k=10)
                
                print(f"Vector search found {len(docs)} documents:")
                for i, (doc, score) in enumerate(docs[:3]):
                    print(f"  Doc {i+1}:")
                    print(f"    Score: {score:.4f}")
                    print(f"    Source: {doc.metadata.get('source', 'Unknown')}")
                    print(f"    Content: {doc.page_content[:200]}...")
                    print()
            else:
                print("No embedding endpoint configured")
                
        except Exception as e:
            print(f"Vector search error: {e}")
            import traceback
            traceback.print_exc()
        
        # Test relevance scoring
        print("\n4. Testing relevance scoring:")
        try:
            from app.langchain.service import calculate_relevance_score
            
            # Sample content from our Tencent document
            sample_content = """Beyondsoft's Partnership with Tencent Executive Summary Since the inception of our partnership in 2012, Beyondsoft has evolved from a service provider into a trusted strategic partner for Tencent. Over the years, our deep integration across Tencent's technology ecosystem—including Tencent Cloud, TDSQL, and enterprise-level distributed systems—has enabled us to drive large-scale database management, high-throughput messaging solutions, and cloud-native application deployments."""
            
            relevance_score = calculate_relevance_score(test_query, sample_content)
            print(f"Relevance score for sample Tencent content: {relevance_score:.4f}")
            
            # Test with irrelevant content
            irrelevant_content = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."""
            irrelevant_score = calculate_relevance_score(test_query, irrelevant_content)
            print(f"Relevance score for irrelevant content: {irrelevant_score:.4f}")
            
        except Exception as e:
            print(f"Relevance scoring error: {e}")
        
        # Test threshold settings
        print("\n5. Testing RAG settings and thresholds:")
        try:
            from app.core.rag_settings_cache import get_document_retrieval_settings, get_agent_settings
            
            doc_settings = get_document_retrieval_settings()
            agent_settings = get_agent_settings()
            
            print(f"Document retrieval settings:")
            print(f"  - similarity_threshold: {doc_settings.get('similarity_threshold', 'Not set')}")
            print(f"  - num_docs_retrieve: {doc_settings.get('num_docs_retrieve', 'Not set')}")
            print(f"  - max_documents_mcp: {doc_settings.get('max_documents_mcp', 'Not set')}")
            
            print(f"\nAgent settings:")
            print(f"  - min_relevance_score: {agent_settings.get('min_relevance_score', 'Not set')}")
            print(f"  - complex_query_threshold: {agent_settings.get('complex_query_threshold', 'Not set')}")
            
        except Exception as e:
            print(f"Settings error: {e}")
        
        # Test query analysis
        print("\n6. Testing query analysis:")
        try:
            from app.langchain.service import analyze_query_type
            
            query_analysis = analyze_query_type(test_query)
            print(f"Query analysis result: {query_analysis}")
            
        except Exception as e:
            print(f"Query analysis error: {e}")
        
        print("\n" + "=" * 70)
        print("Debug Summary:")
        print("- Partnership collection exists and contains 'bys & tencent partnership.pdf'")
        print("- Document contains detailed Tencent partnership information")  
        print("- Need to check if document retrieval, scoring, or filtering is causing issues")
        
    except Exception as e:
        print(f"General error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_rag_pipeline()