#!/usr/bin/env python3
"""
Refresh collection statistics from Milvus to update document counts
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.collection_statistics import refresh_all_collection_statistics
from app.core.vector_db_settings_cache import get_vector_db_settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Refresh all collection statistics"""
    try:
        # Get Milvus settings
        vector_db_settings = get_vector_db_settings()
        milvus_settings = vector_db_settings.get("milvus", {})
        
        if not milvus_settings.get("status"):
            print("❌ Milvus is not enabled in settings")
            return
        
        uri = milvus_settings.get("MILVUS_URI")
        token = milvus_settings.get("MILVUS_TOKEN")
        
        if not uri or not token:
            print("❌ Missing Milvus URI or token")
            return
        
        print(f"🔄 Refreshing collection statistics from Milvus...")
        print(f"📍 Milvus URI: {uri}")
        
        # Refresh all statistics
        results = refresh_all_collection_statistics(uri, token)
        
        print(f"\n✅ Successfully refreshed statistics for {len(results)} collections:")
        
        for collection_name, stats in results.items():
            print(f"\n📂 {collection_name}:")
            print(f"   - Documents: {stats['document_count']:,}")
            print(f"   - Chunks: {stats['total_chunks']:,}")
            print(f"   - Storage: {stats['storage_size_mb']:.2f} MB")
        
        # Invalidate collection cache to force reload
        from app.core.collection_registry_cache import invalidate_all
        cache = invalidate_all()
        print(f"\n🔄 Cleared collection cache to force reload")
        
        print(f"\n✅ Collection statistics refresh complete!")
        
    except Exception as e:
        logger.error(f"Failed to refresh collection statistics: {e}")
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()