"""
Test script to verify collection statistics caching functionality
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000/api/v1"

def test_collections_api():
    """Test the collections API endpoints"""
    print("Testing Collections API...")
    
    # Test 1: Get collections without stats (should be fast)
    print("\n1. Fetching collections without statistics...")
    start_time = time.time()
    response = requests.get(f"{BASE_URL}/collections/")
    end_time = time.time()
    
    if response.status_code == 200:
        collections = response.json()
        print(f"✅ Success! Fetched {len(collections)} collections in {end_time - start_time:.2f}s")
        
        # Check if stats are marked as not_loaded
        for col in collections:
            if col.get('statistics', {}).get('status') == 'not_loaded':
                print(f"   - {col['collection_name']}: Statistics not loaded (as expected)")
    else:
        print(f"❌ Failed: {response.status_code} - {response.text}")
    
    # Test 2: Get collections with stats (should be slower)
    print("\n2. Fetching collections with statistics...")
    start_time = time.time()
    response = requests.get(f"{BASE_URL}/collections/?include_stats=true")
    end_time = time.time()
    
    if response.status_code == 200:
        collections = response.json()
        print(f"✅ Success! Fetched {len(collections)} collections with stats in {end_time - start_time:.2f}s")
        
        # Display statistics
        for col in collections:
            stats = col.get('statistics', {})
            print(f"   - {col['collection_name']}:")
            print(f"     Documents: {stats.get('document_count', 0)}")
            print(f"     Chunks: {stats.get('total_chunks', 0)}")
            print(f"     Storage: {stats.get('storage_size_mb', 0):.2f} MB")
    else:
        print(f"❌ Failed: {response.status_code} - {response.text}")
    
    # Test 3: Get individual collection stats
    if collections:
        collection_name = collections[0]['collection_name']
        print(f"\n3. Fetching statistics for '{collection_name}'...")
        
        response = requests.get(f"{BASE_URL}/collections/{collection_name}/statistics")
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Success! Got stats for {collection_name}:")
            print(f"   Documents: {stats.get('document_count', 0)}")
            print(f"   Chunks: {stats.get('total_chunks', 0)}")
            print(f"   Storage: {stats.get('storage_size_mb', 0):.2f} MB")
        else:
            print(f"❌ Failed: {response.status_code} - {response.text}")

if __name__ == "__main__":
    test_collections_api()