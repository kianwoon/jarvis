#!/usr/bin/env python3
"""
Test script for verifying RAG collections integration
"""

import requests
import json
from datetime import datetime

# API endpoints
BASE_URL = "http://localhost:8000/api/v1"

def test_collection_creation():
    """Test creating new collections"""
    print("\n=== Testing Collection Creation ===")
    
    collections = [
        {
            "collection_name": "technical_docs",
            "collection_type": "technical_docs",
            "description": "Technical documentation, API references, and code examples",
            "metadata_schema": {
                "programming_languages": ["string"],
                "frameworks": ["string"]
            },
            "search_config": {
                "strategy": "precise",
                "boost_recent": False
            },
            "access_config": {
                "default_permission": "read",
                "requires_auth": False
            }
        },
        {
            "collection_name": "policies_procedures",
            "collection_type": "policies_procedures",
            "description": "Company policies, procedures, and guidelines",
            "metadata_schema": {
                "department": "string",
                "effective_date": "date"
            },
            "search_config": {
                "strategy": "comprehensive",
                "boost_recent": True
            }
        }
    ]
    
    for collection in collections:
        response = requests.post(f"{BASE_URL}/collections", json=collection)
        if response.status_code == 200:
            print(f"✅ Created collection: {collection['collection_name']}")
        else:
            print(f"❌ Failed to create {collection['collection_name']}: {response.text}")

def test_document_upload_with_collection():
    """Test uploading documents to specific collections"""
    print("\n=== Testing Document Upload with Collection ===")
    
    # Test auto-classification
    print("\n1. Testing auto-classification:")
    files = {'file': ('test_api_doc.pdf', open('test_api_doc.pdf', 'rb'), 'application/pdf')}
    response = requests.post(
        f"{BASE_URL}/documents/upload_pdf",
        files=files,
        params={'auto_classify': True}
    )
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Document uploaded to collection: {result.get('collection')}")
        if result.get('auto_classified'):
            print(f"   Auto-classified as: {result.get('classified_type')}")
    else:
        print(f"❌ Upload failed: {response.text}")
    
    # Test specific collection
    print("\n2. Testing specific collection upload:")
    files = {'file': ('company_policy.pdf', open('company_policy.pdf', 'rb'), 'application/pdf')}
    response = requests.post(
        f"{BASE_URL}/documents/upload_pdf",
        files=files,
        params={
            'collection_name': 'policies_procedures',
            'auto_classify': False
        }
    )
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Document uploaded to specified collection: {result.get('collection')}")
    else:
        print(f"❌ Upload failed: {response.text}")

def test_rag_with_collections():
    """Test RAG queries with collection routing"""
    print("\n=== Testing RAG with Collections ===")
    
    test_queries = [
        {
            "question": "What is the API endpoint for user authentication?",
            "collection_strategy": "auto"
        },
        {
            "question": "What is the company policy on remote work?",
            "collections": ["policies_procedures"],
            "collection_strategy": "specific"
        },
        {
            "question": "Find all information about vacation policies",
            "collection_strategy": "all"
        }
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query['question']}")
        print(f"   Strategy: {query.get('collection_strategy', 'auto')}")
        if 'collections' in query:
            print(f"   Collections: {query['collections']}")
        
        response = requests.post(
            f"{BASE_URL}/langchain/rag",
            json=query,
            stream=True
        )
        
        if response.status_code == 200:
            # Process streaming response
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if 'token' in data:
                            full_response += data['token']
                        elif 'answer' in data:
                            print(f"✅ Answer: {data['answer'][:200]}...")
                            if 'source_collections' in data:
                                print(f"   Sources from: {data['source_collections']}")
                    except:
                        pass
        else:
            print(f"❌ Query failed: {response.text}")

def test_collection_statistics():
    """Test retrieving collection statistics"""
    print("\n=== Testing Collection Statistics ===")
    
    response = requests.get(f"{BASE_URL}/collections")
    if response.status_code == 200:
        collections = response.json()
        print(f"\nFound {len(collections)} collections:")
        for col in collections:
            print(f"\n📁 {col['collection_name']} ({col['collection_type']})")
            print(f"   Description: {col['description']}")
            if col.get('statistics'):
                stats = col['statistics']
                print(f"   Documents: {stats.get('document_count', 0)}")
                print(f"   Chunks: {stats.get('total_chunks', 0)}")
                print(f"   Size: {stats.get('storage_size_mb', 0):.2f} MB")
    else:
        print(f"❌ Failed to get collections: {response.text}")

if __name__ == "__main__":
    print("=== RAG Collections Integration Test ===")
    print(f"Starting test at {datetime.now()}")
    
    # Note: Ensure the API server is running before executing these tests
    
    try:
        # Run tests in sequence
        test_collection_creation()
        test_document_upload_with_collection()
        test_rag_with_collections()
        test_collection_statistics()
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()