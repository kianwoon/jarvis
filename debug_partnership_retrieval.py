#!/usr/bin/env python3
"""
Debug script to investigate partnership collection document retrieval
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def debug_partnership_retrieval():
    """Debug the partnership collection document retrieval"""
    
    # Import required modules
    from app.langchain.service import handle_rag_query
    from app.core.vector_db_settings_cache import get_vector_db_settings
    from app.core.embedding_settings_cache import get_embedding_settings
    
    try:
        # Set up test query
        test_query = "Tencent partnership"
        print(f"Testing query: '{test_query}'")
        print("=" * 50)
        
        # Test 1: Direct partnership collection search
        print("\n1. Testing direct partnership collection search:")
        try:
            context, sources = handle_rag_query(
                question=test_query,
                thinking=False,
                collections=["partnership"],
                collection_strategy="specific"
            )
            
            print(f"Context length: {len(context) if context else 0}")
            print(f"Number of sources: {len(sources) if sources else 0}")
            
            if context:
                print(f"Context preview: {context[:300]}...")
            else:
                print("No context returned")
                
            if sources:
                print(f"\nSources found:")
                for i, source in enumerate(sources[:3]):
                    print(f"  Source {i+1}:")
                    print(f"    File: {source.get('file', 'Unknown')}")
                    print(f"    Score: {source.get('score', 'N/A')}")
                    print(f"    Collection: {source.get('collection', 'Unknown')}")
                    print(f"    Content preview: {source.get('content', '')[:200]}...")
                    print()
            else:
                print("No sources returned")
                
        except Exception as e:
            print(f"Error in direct search: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 2: Auto collection strategy 
        print("\n2. Testing auto collection strategy:")
        try:
            context, sources = handle_rag_query(
                question=test_query,
                thinking=False,
                collections=None,
                collection_strategy="auto"
            )
            
            print(f"Context length: {len(context) if context else 0}")
            print(f"Number of sources: {len(sources) if sources else 0}")
            
            if sources:
                collections_found = set(source.get('collection', 'Unknown') for source in sources)
                print(f"Collections searched: {list(collections_found)}")
                
        except Exception as e:
            print(f"Error in auto search: {e}")
        
        # Test 3: Search for specific document name
        print("\n3. Testing search for specific PDF name:")
        try:
            pdf_query = "bys tencent partnership pdf"
            context, sources = handle_rag_query(
                question=pdf_query,
                thinking=False,
                collections=["partnership"],
                collection_strategy="specific"
            )
            
            print(f"Query: '{pdf_query}'")
            print(f"Context length: {len(context) if context else 0}")
            print(f"Number of sources: {len(sources) if sources else 0}")
            
            if sources:
                print(f"\nSources found for PDF search:")
                for i, source in enumerate(sources[:3]):
                    print(f"  Source {i+1}:")
                    print(f"    File: {source.get('file', 'Unknown')}")
                    print(f"    Score: {source.get('score', 'N/A')}")
                    print(f"    Content preview: {source.get('content', '')[:200]}...")
                    print()
                    
        except Exception as e:
            print(f"Error in PDF search: {e}")
        
        # Test 4: Direct Milvus collection query to see what documents exist
        print("\n4. Testing direct Milvus collection inspection:")
        try:
            from pymilvus import Collection, connections
            
            # Get vector DB settings
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
                connections.connect(uri=uri, token=token, alias="debug_connection")
                collection = Collection("partnership", using="debug_connection")
                collection.load()
                
                # Query for documents containing "tencent" or "pdf"
                print("Searching for documents containing 'tencent':")
                results = collection.query(
                    expr='content like "%tencent%" or content like "%Tencent%"',
                    output_fields=["content", "source", "doc_id"],
                    limit=5
                )
                
                print(f"Found {len(results)} documents with 'tencent':")
                for i, result in enumerate(results):
                    print(f"  Doc {i+1}:")
                    print(f"    Source: {result.get('source', 'Unknown')}")
                    print(f"    Doc ID: {result.get('doc_id', 'Unknown')}")
                    print(f"    Content preview: {result.get('content', '')[:200]}...")
                    print()
                
                # Query for PDF files
                print("\nSearching for PDF files:")
                pdf_results = collection.query(
                    expr='source like "%.pdf"',
                    output_fields=["content", "source", "doc_id"],
                    limit=10
                )
                
                print(f"Found {len(pdf_results)} PDF documents:")
                for i, result in enumerate(pdf_results):
                    print(f"  PDF {i+1}: {result.get('source', 'Unknown')}")
                
                connections.disconnect(alias="debug_connection")
            else:
                print("No Milvus URI configured")
                
        except Exception as e:
            print(f"Error in direct Milvus query: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"General error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_partnership_retrieval())