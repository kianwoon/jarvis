#!/usr/bin/env python3
"""
Test collection statistics functionality
"""

import sys
sys.path.append('/Users/kianwoonwong/Downloads/jarvis')

from app.core.collection_statistics import refresh_all_collection_statistics, get_collection_statistics
from app.core.vector_db_settings_cache import get_vector_db_settings

def test_refresh_statistics():
    """Test refreshing statistics from Milvus"""
    print("\n=== Testing Statistics Refresh ===")
    
    # Get Milvus settings
    vector_db_settings = get_vector_db_settings()
    milvus_config = vector_db_settings.get('milvus', {})
    uri = milvus_config.get('MILVUS_URI')
    token = milvus_config.get('MILVUS_TOKEN')
    
    print(f"Milvus URI: {uri}")
    
    try:
        # Refresh all statistics
        stats = refresh_all_collection_statistics(uri, token)
        
        print(f"\nRefreshed statistics for {len(stats)} collections:")
        for collection_name, collection_stats in stats.items():
            print(f"\n📁 {collection_name}:")
            print(f"   Documents: {collection_stats['document_count']}")
            print(f"   Chunks: {collection_stats['total_chunks']}")
            print(f"   Size: {collection_stats['storage_size_mb']:.2f} MB")
            
    except Exception as e:
        print(f"❌ Error refreshing statistics: {e}")
        import traceback
        traceback.print_exc()

def test_get_statistics():
    """Test getting statistics for a specific collection"""
    print("\n=== Testing Get Statistics ===")
    
    collection_name = "default_knowledge"
    stats = get_collection_statistics(collection_name)
    
    if stats:
        print(f"\nStatistics for {collection_name}:")
        print(f"   Documents: {stats['document_count']}")
        print(f"   Chunks: {stats['total_chunks']}")
        print(f"   Size: {stats['storage_size_mb']:.2f} MB")
        print(f"   Last Updated: {stats['last_updated']}")
    else:
        print(f"No statistics found for {collection_name}")

if __name__ == "__main__":
    print("=== Collection Statistics Test ===")
    
    # Run tests
    test_refresh_statistics()
    test_get_statistics()
    
    print("\n✅ Tests completed!")