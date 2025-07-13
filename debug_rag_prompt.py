#!/usr/bin/env python3
"""
Debug script to examine what content is actually being retrieved for Tencent partnership queries
Uses the correct Milvus-based document storage system
"""
import sys
import os

# Add the app directory and utils to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))
sys.path.insert(0, os.path.dirname(__file__))

def test_tencent_partnership_retrieval():
    """Test what content is actually retrieved for Tencent partnership queries"""
    try:
        print("🔍 Testing Tencent Partnership Content Retrieval")
        print("=" * 60)
        
        # Test query
        test_query = "search internal knowledge base. relationship between beyondsoft and tencent in details"
        
        print(f"Query: {test_query}")
        print(f"Target collection: partnership")
        
        # Test Milvus connection and collection access
        print("\n📄 Testing Milvus collection access...")
        
        from app.core.vector_db_settings_cache import get_vector_db_settings
        from app.core.embedding_settings_cache import get_embedding_settings
        from app.core.collection_registry_cache import get_all_collections
        
        # Get configurations
        vector_db_cfg = get_vector_db_settings()
        embedding_cfg = get_embedding_settings()
        collections = get_all_collections()
        
        print(f"Collections type: {type(collections)}")
        print(f"Collections raw: {collections}")
        
        # Handle both dict and list structures
        if isinstance(collections, list):
            collection_names = []
            for col in collections:
                if isinstance(col, dict):
                    # Try different possible name fields
                    name = col.get('name') or col.get('collection_name') or col.get('id', 'Unknown')
                    collection_names.append(name)
                else:
                    collection_names.append(str(col))
        else:
            collection_names = list(collections.keys()) if hasattr(collections, 'keys') else []
            
        print(f"Available collections: {collection_names}")
        
        partnership_found = 'partnership' in collection_names or any('partnership' in str(name).lower() for name in collection_names)
        
        if partnership_found:
            print("✅ Partnership collection found")
            
            # Try to connect to Milvus and query the partnership collection
            from pymilvus import Collection, connections
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import Milvus
            
            # Setup embeddings
            embedding_model = HuggingFaceEmbeddings(
                model_name=embedding_cfg['model_name'],
                model_kwargs={'device': embedding_cfg.get('device', 'cpu')},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Connect to Milvus
            milvus_host = vector_db_cfg['host']
            milvus_port = vector_db_cfg['port']
            
            connections.connect(host=milvus_host, port=milvus_port)
            
            # Get the partnership collection
            collection = Collection("partnership")
            collection.load()
            
            print(f"📊 Partnership collection stats:")
            print(f"  Total entities: {collection.num_entities}")
            
            # Search for Tencent-related content
            query_vector = embedding_model.embed_query(test_query)
            
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=10,
                output_fields=["content", "source", "doc_type", "page"]
            )
            
            print(f"\n🔍 Search Results (top 10):")
            for i, hit in enumerate(results[0]):
                content = hit.entity.get('content', 'No content')
                source = hit.entity.get('source', 'No source')
                score = hit.score
                
                print(f"\n  {i+1}. Score: {score:.4f}")
                print(f"     Source: {source}")
                print(f"     Content preview: {content[:200]}...")
                
                # Check for Tencent mentions
                tencent_count = content.lower().count('tencent')
                beyondsoft_count = content.lower().count('beyondsoft')
                print(f"     Tencent mentions: {tencent_count}")
                print(f"     Beyondsoft mentions: {beyondsoft_count}")
            
            # Also check what BM25 would retrieve
            print(f"\n🔍 Testing BM25 retrieval...")
            
            # Get all documents for BM25 analysis
            query_results = collection.query(
                expr="id != ''",
                output_fields=["content", "source", "bm25_tokens"],
                limit=100
            )
            
            tencent_docs = []
            for doc in query_results:
                content = doc.get('content', '')
                if 'tencent' in content.lower() or 'beyondsoft' in content.lower():
                    tencent_docs.append(doc)
            
            print(f"📊 Found {len(tencent_docs)} documents containing Tencent/Beyondsoft")
            
            for i, doc in enumerate(tencent_docs[:5]):
                content = doc.get('content', '')
                source = doc.get('source', 'Unknown')
                print(f"\n  {i+1}. Source: {source}")
                print(f"     Content length: {len(content)} chars")
                print(f"     Preview: {content[:150]}...")
                
        else:
            print("❌ Partnership collection not found")
            print("Available collections:", collection_names)
            
    except Exception as e:
        print(f"❌ Error in retrieval test: {e}")
        import traceback
        traceback.print_exc()

def check_rag_settings():
    """Check current RAG settings that might affect retrieval"""
    try:
        print(f"\n⚙️  RAG Settings Analysis")
        print("=" * 60)
        
        from app.core.rag_settings_cache import get_rag_settings
        
        settings = get_rag_settings()
        
        # Check document retrieval settings
        doc_retrieval = settings.get('document_retrieval', {})
        print(f"📄 Document Retrieval Settings:")
        print(f"  max_documents_per_collection: {doc_retrieval.get('max_documents_per_collection', 'Not set')}")
        print(f"  max_documents_mcp: {doc_retrieval.get('max_documents_mcp', 'Not set')}")
        print(f"  similarity_threshold: {doc_retrieval.get('similarity_threshold', 'Not set')}")
        print(f"  num_docs_retrieve: {doc_retrieval.get('num_docs_retrieve', 'Not set')}")
        
        # Check search strategy
        search_strategy = settings.get('search_strategy', {})
        print(f"\n🔍 Search Strategy Settings:")
        print(f"  hybrid_search_weight: {search_strategy.get('hybrid_search_weight', 'Not set')}")
        print(f"  bm25_weight: {search_strategy.get('bm25_weight', 'Not set')}")
        print(f"  vector_weight: {search_strategy.get('vector_weight', 'Not set')}")
        
        # Check BM25 settings
        bm25_settings = settings.get('bm25_scoring', {})
        print(f"\n📊 BM25 Settings:")
        print(f"  k1: {bm25_settings.get('k1', 'Not set')}")
        print(f"  b: {bm25_settings.get('b', 'Not set')}")
        print(f"  boost_keywords: {bm25_settings.get('boost_keywords', 'Not set')}")
        
        # Check reranking
        reranking = settings.get('reranking', {})
        print(f"\n🎯 Reranking Settings:")
        print(f"  enabled: {reranking.get('enabled', 'Not set')}")
        print(f"  model_path: {reranking.get('model_path', 'Not set')}")
        print(f"  top_k: {reranking.get('top_k', 'Not set')}")
        
    except Exception as e:
        print(f"❌ Error checking RAG settings: {e}")

def test_actual_rag_query():
    """Test the actual RAG query pipeline"""
    try:
        print(f"\n🔄 Testing Actual RAG Pipeline")
        print("=" * 60)
        
        from app.langchain.service import handle_rag_query
        
        test_query = "search internal knowledge base. relationship between beyondsoft and tencent in details"
        
        print(f"Query: {test_query}")
        print(f"Collections: ['partnership']")
        
        results = handle_rag_query(
            question=test_query,
            thinking=False,
            collections=['partnership'],
            collection_strategy="specific"
        )
        
        if results and len(results) >= 2:
            sources = results[0] if isinstance(results[0], list) else []
            context = results[1] if len(results) > 1 else ""
            
            print(f"\n✅ Pipeline Results:")
            print(f"  Sources retrieved: {len(sources)}")
            print(f"  Context length: {len(context)} characters")
            
            # Analyze content quality
            context_lower = context.lower()
            tencent_mentions = context_lower.count('tencent')
            beyondsoft_mentions = context_lower.count('beyondsoft')
            partnership_mentions = context_lower.count('partnership')
            
            print(f"\n📊 Content Analysis:")
            print(f"  Tencent mentions: {tencent_mentions}")
            print(f"  Beyondsoft mentions: {beyondsoft_mentions}")
            print(f"  Partnership mentions: {partnership_mentions}")
            
            print(f"\n📄 Context Preview (first 800 chars):")
            print("-" * 50)
            print(context[:800])
            print("-" * 50)
            
            # Show source analysis
            print(f"\n📚 Source Analysis:")
            for i, source in enumerate(sources[:3]):
                if hasattr(source, 'metadata'):
                    source_file = source.metadata.get('source', 'Unknown')
                    content = getattr(source, 'page_content', str(source))
                    content_preview = content[:100].replace('\n', ' ')
                    
                    print(f"  {i+1}. File: {source_file}")
                    print(f"     Content: {content_preview}...")
                    print(f"     Length: {len(content)} chars")
                
        else:
            print("❌ No results from RAG pipeline")
            
    except Exception as e:
        print(f"❌ Error in RAG pipeline test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tencent_partnership_retrieval()
    check_rag_settings()
    test_actual_rag_query()
    print(f"\n✨ Debug analysis completed")