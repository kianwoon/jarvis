#!/usr/bin/env python3
"""
Debug script to investigate why Tencent partnership content extraction is failing
"""

import os
import sys
import asyncio
import json
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def debug_tencent_partnership():
    """Debug the Tencent partnership content extraction issue"""
    
    try:
        # Direct imports to avoid circular import
        from app.core.vector_db_settings_cache import get_vector_db_settings
        from app.core.embedding_settings_cache import get_embedding_settings
        from app.core.rag_settings_cache import get_rag_settings
        from pymilvus import Collection, connections
        
        print("=== TENCENT PARTNERSHIP DEBUG ===")
        print("Investigating why detailed content isn't being extracted")
        print()
        
        # Step 1: Check RAG settings
        print("1. CHECKING RAG SETTINGS:")
        try:
            rag_settings = get_rag_settings()
            doc_retrieval = rag_settings.get('document_retrieval', {})
            print(f"   - Similarity threshold: {doc_retrieval.get('similarity_threshold')}")
            print(f"   - Num docs retrieve: {doc_retrieval.get('num_docs_retrieve')}")
            print(f"   - Agent min relevance: {rag_settings.get('agent_settings', {}).get('min_relevance_score')}")
            print()
        except Exception as e:
            print(f"   Error getting RAG settings: {e}")
            print()
        
        # Step 2: Connect to Milvus and inspect partnership collection
        print("2. CONNECTING TO MILVUS:")
        try:
            vector_db_settings = get_vector_db_settings()
            
            # Get Milvus config
            milvus_config = None
            if "milvus" in vector_db_settings:
                milvus_config = vector_db_settings["milvus"]
                uri = milvus_config.get("MILVUS_URI")
                token = milvus_config.get("MILVUS_TOKEN")
            else:
                for db in vector_db_settings.get("databases", []):
                    if db.get("id") == "milvus" and db.get("enabled", False):
                        milvus_config = db.get("config", {})
                        break
                
                uri = milvus_config.get("MILVUS_URI") if milvus_config else None
                token = milvus_config.get("MILVUS_TOKEN") if milvus_config else None
            
            if not uri:
                print("   No Milvus URI found in settings")
                return
                
            connections.connect(uri=uri, token=token, alias="debug_connection")
            collection = Collection("partnership", using="debug_connection")
            collection.load()
            print(f"   Connected to partnership collection")
            
            # Step 3: Find all documents containing "tencent"
            print("\n3. SEARCHING FOR TENCENT DOCUMENTS:")
            try:
                # Search for documents with "tencent" in content or source
                results = collection.query(
                    expr='content like "%tencent%" or content like "%Tencent%" or source like "%tencent%"',
                    output_fields=["content", "source", "doc_id"],
                    limit=10
                )
                
                print(f"   Found {len(results)} chunks with 'tencent'")
                
                # Group by source file
                sources = {}
                for result in results:
                    source = result.get('source', 'Unknown')
                    if source not in sources:
                        sources[source] = []
                    sources[source].append(result)
                
                print(f"   Documents: {list(sources.keys())}")
                
                # Examine "bys & tencent partnership.pdf" specifically
                target_file = None
                for source in sources:
                    if "bys" in source.lower() and "tencent" in source.lower():
                        target_file = source
                        break
                
                if target_file:
                    print(f"\n   ANALYZING TARGET FILE: {target_file}")
                    chunks = sources[target_file]
                    print(f"   Number of chunks: {len(chunks)}")
                    
                    for i, chunk in enumerate(chunks[:3]):
                        print(f"\n   Chunk {i+1}:")
                        print(f"     Doc ID: {chunk.get('doc_id')}")
                        content = chunk.get('content', '')
                        print(f"     Content length: {len(content)}")
                        print(f"     Content preview: {content[:300]}...")
                        
                        # Look for specific partnership details
                        keywords = ['partnership', 'collaboration', 'team', 'size', 'revenue', 'outcome', 'project']
                        found_keywords = [kw for kw in keywords if kw.lower() in content.lower()]
                        print(f"     Keywords found: {found_keywords}")
                        print()
                else:
                    print("   Target file 'bys & tencent partnership.pdf' not found")
                    
            except Exception as e:
                print(f"   Error querying documents: {e}")
                import traceback
                traceback.print_exc()
            
            # Step 4: Test vector search with Tencent query
            print("\n4. TESTING VECTOR SEARCH:")
            try:
                from app.llm.embedding import get_embedding_client
                
                # Get embedding for the query
                embedding_settings = get_embedding_settings()
                embedding_client = get_embedding_client(embedding_settings)
                
                query = "Tencent partnership collaboration details"
                query_embedding = embedding_client.encode([query])[0].tolist()
                
                # Search with vector similarity
                search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
                results = collection.search(
                    data=[query_embedding],
                    anns_field="embeddings",
                    param=search_params,
                    limit=10,
                    output_fields=["content", "source", "doc_id"]
                )
                
                print(f"   Vector search returned {len(results[0])} results")
                
                for i, hit in enumerate(results[0][:5]):
                    print(f"\n   Result {i+1}:")
                    print(f"     Score: {hit.score:.4f}")
                    print(f"     Source: {hit.entity.get('source')}")
                    print(f"     Doc ID: {hit.entity.get('doc_id')}")
                    content = hit.entity.get('content', '')
                    print(f"     Content length: {len(content)}")
                    print(f"     Content preview: {content[:200]}...")
                    
                    # Check if this is from the target PDF
                    source = hit.entity.get('source', '')
                    if "bys" in source.lower() and "tencent" in source.lower():
                        print(f"     *** THIS IS FROM TARGET PDF ***")
                        
                        # Look for specific business details
                        business_terms = ['million', 'billion', 'revenue', 'employees', 'team size', 'projects', 'clients', 'years']
                        found_terms = [term for term in business_terms if term.lower() in content.lower()]
                        if found_terms:
                            print(f"     Business terms found: {found_terms}")
                        else:
                            print(f"     No specific business terms found")
                    print()
                    
            except Exception as e:
                print(f"   Error in vector search: {e}")
                import traceback
                traceback.print_exc()
            
            # Step 5: Check if there are content filtering issues
            print("\n5. CONTENT FILTERING ANALYSIS:")
            try:
                # Look for very short chunks that might be filtered out
                all_chunks = collection.query(
                    expr='source like "%tencent%"',
                    output_fields=["content", "source"],
                    limit=50
                )
                
                chunk_lengths = [len(chunk.get('content', '')) for chunk in all_chunks]
                if chunk_lengths:
                    print(f"   Chunk lengths - Min: {min(chunk_lengths)}, Max: {max(chunk_lengths)}, Avg: {sum(chunk_lengths)/len(chunk_lengths):.1f}")
                    
                    # Count very short chunks
                    short_chunks = [length for length in chunk_lengths if length < 100]
                    print(f"   Chunks under 100 chars: {len(short_chunks)}")
                    
                    # Look for chunks with just metadata
                    metadata_chunks = 0
                    for chunk in all_chunks:
                        content = chunk.get('content', '').lower()
                        if any(meta in content for meta in ['filename', 'page', 'header', 'footer']) and len(content) < 200:
                            metadata_chunks += 1
                    print(f"   Likely metadata chunks: {metadata_chunks}")
                else:
                    print("   No chunks found for analysis")
                    
            except Exception as e:
                print(f"   Error in content analysis: {e}")
            
            connections.disconnect(alias="debug_connection")
            
        except Exception as e:
            print(f"   Error connecting to Milvus: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n=== ANALYSIS COMPLETE ===")
        
    except Exception as e:
        print(f"General error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_tencent_partnership())