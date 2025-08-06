#!/usr/bin/env python3
"""
Test script to debug RAG search functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_rag_search():
    """Test RAG search functionality"""
    print("=== RAG Search Debug Test ===")
    
    # Test 1: Check database collections
    print("\n1. Checking database collections...")
    try:
        from app.core.db import SessionLocal
        from app.core.db import Settings as SettingsModel
        
        db = SessionLocal()
        try:
            # Check collection statistics
            from app.core.db import SessionLocal
            import sqlalchemy as sa
            
            # Execute raw SQL to check collection statistics
            result = db.execute(sa.text("SELECT collection_name, document_count, total_chunks FROM collection_statistics ORDER BY document_count DESC"))
            stats = result.fetchall()
            
            print("Collection statistics:")
            for stat in stats:
                print(f"  - {stat[0]}: {stat[1]} docs, {stat[2]} chunks")
                
        finally:
            db.close()
            
    except Exception as e:
        print(f"Database check failed: {e}")
    
    # Test 2: Check vector database connectivity
    print("\n2. Checking vector database connectivity...")
    try:
        from app.core.vector_db_settings_cache import get_vector_db_settings
        settings = get_vector_db_settings()
        print(f"Vector DB settings: {settings}")
        
        # Try to connect to Milvus
        try:
            from pymilvus import connections, utility
            # Check multiple locations for Milvus config
            milvus_config = {}
            if 'databases' in settings:
                for db in settings['databases']:
                    if db.get('id') == 'milvus' and db.get('enabled'):
                        milvus_config = db.get('config', {})
                        break
            
            uri = milvus_config.get('MILVUS_URI') or milvus_config.get('uri')
            token = milvus_config.get('MILVUS_TOKEN') or milvus_config.get('token')
            print(f"Trying to connect to Milvus at: {uri}")
            print(f"Token available: {'Yes' if token else 'No'}")
            
            if uri:
                # Connect with token if available
                if token:
                    connections.connect(uri=uri, token=token)
                else:
                    connections.connect(uri=uri)
                    
                collections = utility.list_collections()
                print(f"Available Milvus collections: {collections}")
                
                # Check specific collections
                for collection in ['partnership', 'default_knowledge']:
                    if collection in collections:
                        from pymilvus import Collection
                        col = Collection(collection)
                        col.load()
                        count = col.num_entities
                        print(f"  - {collection}: {count} entities")
            else:
                print("No Milvus URI configured")
                
        except Exception as e:
            print(f"Milvus connection failed: {e}")
            
    except Exception as e:
        print(f"Vector DB check failed: {e}")
    
    # Test 3: Test RAG fallback directly
    print("\n3. Testing RAG fallback search...")
    try:
        from app.core.rag_fallback import simple_rag_search
        
        # Test search for BeyondSoft/Alibaba partnership
        result = simple_rag_search(
            query="BeyondSoft and Alibaba partnership",
            collections=["partnership"],
            max_documents=5,
            include_content=True
        )
        
        print(f"RAG search result:")
        print(f"  Success: {result.get('success')}")
        print(f"  Collections searched: {result.get('collections_searched')}")
        print(f"  Documents found: {result.get('total_documents_found')}")
        print(f"  Documents returned: {result.get('documents_returned')}")
        print(f"  Execution time: {result.get('execution_time_ms')}ms")
        
        if result.get('error'):
            print(f"  Error: {result.get('error')}")
            
        if result.get('documents'):
            print("  Documents:")
            for i, doc in enumerate(result.get('documents', [])[:3]):
                print(f"    {i+1}. Title: {doc.get('title', 'N/A')}")
                print(f"       Score: {doc.get('score', 0)}")
                content = doc.get('content', '')[:200]
                print(f"       Content: {content}...")
                print()
        
    except Exception as e:
        print(f"RAG fallback test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Check what's actually in the partnership collection
    print("\n4. Checking partnership collection content...")
    try:
        # Check if we can query the vector database directly
        from app.core.vector_db_settings_cache import get_vector_db_settings
        settings = get_vector_db_settings()
        
        # Get Milvus config properly
        milvus_config = {}
        if 'databases' in settings:
            for db in settings['databases']:
                if db.get('id') == 'milvus' and db.get('enabled'):
                    milvus_config = db.get('config', {})
                    break
        
        uri = milvus_config.get('MILVUS_URI') or milvus_config.get('uri')
        token = milvus_config.get('MILVUS_TOKEN') or milvus_config.get('token')
        
        if uri:
            from pymilvus import Collection, connections, utility
            # Connect with token if available
            if token:
                connections.connect(uri=uri, token=token)
            else:
                connections.connect(uri=uri)
            
            available_collections = utility.list_collections()
            if 'partnership' in available_collections:
                collection = Collection('partnership')
                collection.load()
                
                # Get collection schema to understand the fields
                print(f"Collection schema for partnership:")
                schema = collection.schema
                for field in schema.fields:
                    print(f"  - {field.name}: {field.dtype} (is_primary: {field.is_primary})")
                
                # Find the primary key field
                pk_field = None
                for field in schema.fields:
                    if field.is_primary:
                        pk_field = field.name
                        break
                
                print(f"Primary key field: {pk_field}")
                
                # Try to get some sample data using the correct primary key
                if pk_field:
                    try:
                        # Since primary key is VarChar, use a different expression
                        results = collection.query(
                            expr="",  # Empty expression gets all records
                            output_fields=["content", "source"],
                            limit=3
                        )
                        
                        print(f"Sample documents from partnership collection:")
                        for i, result in enumerate(results):
                            print(f"  Document {i+1}:")
                            content = str(result.get('content', ''))[:300]
                            source = result.get('source', 'N/A')
                            print(f"    Source: {source}")
                            print(f"    Content: {content}...")
                            print()
                            
                        # Search test with actual embedding
                        print(f"\n5. Testing vector search directly...")
                        from app.core.embedding_settings_cache import get_embedding_settings
                        embedding_settings = get_embedding_settings()
                        
                        # Create simple embedder
                        try:
                            from sentence_transformers import SentenceTransformer
                            model_name = embedding_settings.get('model_name', 'all-MiniLM-L6-v2')
                            embedder = SentenceTransformer(model_name)
                            
                            # Search for BeyondSoft Alibaba partnership
                            search_query = "BeyondSoft Alibaba partnership"
                            query_embedding = embedder.encode([search_query])[0].tolist()
                            
                            search_results = collection.search(
                                data=[query_embedding],
                                anns_field="vector",  # Use correct field name from schema
                                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                                limit=5,
                                output_fields=["content", "source"]
                            )
                            
                            print(f"Vector search results for '{search_query}':")
                            for hits in search_results:
                                for i, hit in enumerate(hits):
                                    print(f"  Result {i+1}: Score={hit.distance:.3f}")
                                    content = str(hit.entity.get('content', ''))[:300]
                                    source = hit.entity.get('source', 'N/A')
                                    print(f"    Source: {source}")
                                    print(f"    Content: {content}...")
                                    print()
                        except Exception as e:
                            print(f"Vector search failed: {e}")
                            
                    except Exception as e:
                        print(f"Query with primary key failed: {e}")
            else:
                print(f"Partnership collection not found. Available: {available_collections}")
        
    except Exception as e:
        print(f"Partnership collection check failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rag_search()