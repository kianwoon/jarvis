#!/usr/bin/env python3
"""
Simple direct test of Milvus partnership collection
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

try:
    from pymilvus import connections, Collection
    from app.core.vector_db_settings_cache import get_vector_db_settings
    
    print("🔍 Direct Milvus Partnership Collection Test")
    print("=" * 50)
    
    # Get Milvus settings
    vector_db_cfg = get_vector_db_settings()
    print(f"Vector DB config: {vector_db_cfg}")
    
    # Get Milvus config from the active database
    active_db = vector_db_cfg.get('active', 'milvus')
    db_configs = {db['id']: db for db in vector_db_cfg.get('databases', [])}
    milvus_config = db_configs.get(active_db, {}).get('config', {})
    
    milvus_uri = milvus_config.get('MILVUS_URI')
    milvus_token = milvus_config.get('MILVUS_TOKEN')
    
    print(f"Connecting to Milvus URI: {milvus_uri}")
    
    connections.connect(
        uri=milvus_uri,
        token=milvus_token
    )
    
    # Access partnership collection
    collection = Collection("partnership")
    collection.load()
    
    print(f"📊 Partnership Collection Stats:")
    print(f"  Total documents: {collection.num_entities}")
    
    # Get sample documents
    results = collection.query(
        expr="id != ''",
        output_fields=["content", "source", "doc_type"],
        limit=5
    )
    
    print(f"\n📄 Sample Documents ({len(results)} shown):")
    for i, doc in enumerate(results):
        content = doc.get('content', '')
        source = doc.get('source', 'Unknown')
        
        print(f"\n{i+1}. Source: {source}")
        print(f"   Content length: {len(content)} chars")
        print(f"   Preview: {content[:200]}...")
        
        # Check for Tencent mentions
        tencent_count = content.lower().count('tencent')
        beyondsoft_count = content.lower().count('beyondsoft')
        
        print(f"   Tencent mentions: {tencent_count}")
        print(f"   Beyondsoft mentions: {beyondsoft_count}")
    
    # Get all Tencent documents for detailed analysis
    all_results = collection.query(
        expr="id != ''",
        output_fields=["content", "source"],
        limit=173  # All documents
    )
    
    tencent_docs = []
    for doc in all_results:
        content = doc.get('content', '').lower()
        if 'tencent' in content or 'beyondsoft' in content:
            tencent_docs.append(doc)
    
    print(f"\n🔍 All Tencent/Beyondsoft documents ({len(tencent_docs)} found):")
    
    total_content = ""
    for i, doc in enumerate(tencent_docs[:10]):  # Show first 10
        content = doc.get('content', '')
        source = doc.get('source', 'Unknown')
        
        total_content += content + "\n\n"
        
        print(f"\n{i+1}. Source: {source}")
        print(f"   Length: {len(content)} chars")
        print(f"   Content: {content[:400]}...")
    
    print(f"\n📊 TOTAL CONTENT ANALYSIS:")
    print(f"  Total relevant documents: {len(tencent_docs)}")
    print(f"  Combined content length: {len(total_content)} chars")
    print(f"  Total Tencent mentions: {total_content.lower().count('tencent')}")
    print(f"  Total Beyondsoft mentions: {total_content.lower().count('beyondsoft')}")
    
    # This is what should be returned by the RAG system
    print(f"\n🎯 EXPECTED RAG RESPONSE LENGTH: {len(total_content)} characters")
    print(f"This should produce comprehensive partnership details, not brief responses")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()