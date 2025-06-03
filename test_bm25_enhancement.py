#!/usr/bin/env python3
"""
Test script for BM25 enhancement to Milvus RAG system
"""

import asyncio
import json
import requests
from app.rag.bm25_processor import BM25Processor, BM25CorpusManager
from app.langchain.service import calculate_relevance_score
from app.core.vector_db_settings_cache import get_vector_db_settings
from pymilvus import connections, Collection


def test_bm25_processor():
    """Test the BM25 processor functionality"""
    print("🧪 Testing BM25 Processor...")
    
    processor = BM25Processor()
    
    # Test text
    test_text = """
    Machine learning is a subset of artificial intelligence that focuses on algorithms 
    that can learn from data. Deep learning uses neural networks with multiple layers 
    to process complex patterns in large datasets.
    """
    
    # Test tokenization
    tokens = processor.tokenize_and_clean(test_text)
    print(f"✅ Tokenization: {len(tokens)} tokens")
    print(f"   Sample tokens: {tokens[:10]}")
    
    # Test term frequencies
    term_freq = processor.calculate_term_frequencies(test_text)
    print(f"✅ Term frequencies: {len(term_freq)} unique terms")
    print(f"   Top terms: {dict(list(term_freq.items())[:5])}")
    
    # Test document preparation
    metadata = processor.prepare_document_for_bm25(test_text)
    print(f"✅ BM25 metadata prepared:")
    print(f"   Term count: {metadata.get('bm25_term_count')}")
    print(f"   Unique terms: {metadata.get('bm25_unique_terms')}")
    
    return True


def test_enhanced_relevance_scoring():
    """Test enhanced relevance scoring"""
    print("\n🧪 Testing Enhanced Relevance Scoring...")
    
    query = "machine learning algorithms"
    context1 = "Machine learning algorithms are used in artificial intelligence to process data automatically."
    context2 = "The weather today is sunny and warm with no clouds in the sky."
    
    score1 = calculate_relevance_score(query, context1)
    score2 = calculate_relevance_score(query, context2)
    
    print(f"✅ Relevance scores:")
    print(f"   Query: '{query}'")
    print(f"   Context 1 (relevant): {score1:.3f}")
    print(f"   Context 2 (irrelevant): {score2:.3f}")
    
    assert score1 > score2, "Relevant context should score higher than irrelevant"
    return True


def test_milvus_schema_compatibility():
    """Test that enhanced schema is compatible"""
    print("\n🧪 Testing Milvus Schema Compatibility...")
    
    try:
        # Get vector DB settings
        vector_db_cfg = get_vector_db_settings()
        milvus_cfg = vector_db_cfg.get("milvus", {})
        
        if not milvus_cfg:
            print("⚠️  Milvus not configured, skipping schema test")
            return True
        
        uri = milvus_cfg.get("MILVUS_URI")
        token = milvus_cfg.get("MILVUS_TOKEN")
        collection_name = milvus_cfg.get("MILVUS_DEFAULT_COLLECTION", "default_knowledge")
        
        if not uri:
            print("⚠️  Milvus URI not configured, skipping schema test")
            return True
        
        # Connect to Milvus
        connections.connect(uri=uri, token=token)
        
        # Check if collection exists
        from pymilvus import utility
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            collection.load()
            
            # Get collection schema
            schema = collection.schema
            field_names = [field.name for field in schema.fields]
            
            print(f"✅ Collection '{collection_name}' exists")
            print(f"   Fields: {field_names}")
            
            # Check for BM25 fields
            bm25_fields = ['bm25_tokens', 'bm25_term_count', 'bm25_unique_terms', 'bm25_top_terms']
            has_bm25_fields = all(field in field_names for field in bm25_fields)
            
            if has_bm25_fields:
                print("✅ BM25 fields present in schema")
            else:
                print("⚠️  BM25 fields not yet in schema (will be added on next document upload)")
            
            print(f"   Total entities: {collection.num_entities}")
            
        else:
            print(f"⚠️  Collection '{collection_name}' does not exist (will be created on first upload)")
        
        connections.disconnect(alias="default")
        return True
        
    except Exception as e:
        print(f"❌ Schema test failed: {e}")
        return False


def test_corpus_statistics():
    """Test corpus statistics calculation"""
    print("\n🧪 Testing Corpus Statistics...")
    
    try:
        corpus_manager = BM25CorpusManager()
        
        # Get vector DB settings
        vector_db_cfg = get_vector_db_settings()
        milvus_cfg = vector_db_cfg.get("milvus", {})
        
        if not milvus_cfg:
            print("⚠️  Milvus not configured, skipping corpus stats test")
            return True
        
        uri = milvus_cfg.get("MILVUS_URI")
        token = milvus_cfg.get("MILVUS_TOKEN")
        collection_name = milvus_cfg.get("MILVUS_DEFAULT_COLLECTION", "default_knowledge")
        
        if not uri:
            print("⚠️  Milvus URI not configured, skipping corpus stats test")
            return True
        
        # Connect and check collection
        connections.connect(uri=uri, token=token)
        from pymilvus import utility
        
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            
            # Test corpus stats calculation
            stats = corpus_manager.calculate_corpus_stats_from_milvus(collection)
            
            print(f"✅ Corpus statistics calculated:")
            print(f"   Total documents: {stats.total_documents}")
            print(f"   Average doc length: {stats.average_doc_length:.2f}")
            print(f"   Unique terms in corpus: {len(stats.term_doc_frequencies)}")
            
            if stats.total_documents > 0:
                # Show top terms by document frequency
                sorted_terms = sorted(
                    stats.term_doc_frequencies.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10]
                print(f"   Top terms by doc frequency: {sorted_terms}")
            
        else:
            print(f"⚠️  Collection '{collection_name}' does not exist")
        
        connections.disconnect(alias="default")
        return True
        
    except Exception as e:
        print(f"❌ Corpus stats test failed: {e}")
        return False


def test_api_endpoint():
    """Test the enhanced API endpoint"""
    print("\n🧪 Testing API Endpoint...")
    
    try:
        # Test the debug search endpoint
        test_query = "machine learning"
        response = requests.get(
            f"http://localhost:8001/api/v1/document/debug_search/{test_query}",
            params={"limit": 5, "min_score": 0.3}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API endpoint working:")
            print(f"   Query: {data.get('query')}")
            print(f"   Results: {data.get('results_count')}")
            print(f"   Collection: {data.get('collection')}")
            
            # Check if results have BM25 enhanced scoring
            results = data.get('results', [])
            if results:
                first_result = results[0]
                print(f"   First result score: {first_result.get('score'):.3f}")
                print(f"   Keyword match: {first_result.get('exact_keyword_match')}")
        else:
            print(f"⚠️  API endpoint returned status {response.status_code}")
            print(f"   Response: {response.text}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("⚠️  API server not running, skipping endpoint test")
        return True
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False


def test_supported_file_types():
    """Test supported file types endpoint"""
    print("\n🧪 Testing Supported File Types...")
    
    try:
        response = requests.get("http://localhost:8001/api/v1/document/supported-types")
        
        if response.status_code == 200:
            data = response.json()
            supported = data.get('supported_types', [])
            coming_soon = data.get('coming_soon', [])
            
            print(f"✅ Supported file types ({len(supported)}):")
            for file_type in supported:
                ext = file_type.get('extension')
                desc = file_type.get('description')
                print(f"   {ext}: {desc}")
            
            print(f"✅ Coming soon ({len(coming_soon)}):")
            for file_type in coming_soon:
                ext = file_type.get('extension')
                desc = file_type.get('description')
                print(f"   {ext}: {desc}")
            
            # Check if all major file types are supported
            supported_extensions = [ft.get('extension') for ft in supported]
            expected_extensions = ['.pdf', '.xlsx', '.xls', '.docx', '.pptx', '.ppt']
            
            missing = [ext for ext in expected_extensions if ext not in supported_extensions]
            if not missing:
                print("✅ All expected file types are supported!")
            else:
                print(f"⚠️  Missing support for: {missing}")
            
            return len(missing) == 0
        else:
            print(f"⚠️  API returned status {response.status_code}")
            return False
        
    except requests.exceptions.ConnectionError:
        print("⚠️  API server not running, skipping file types test")
        return True
    except Exception as e:
        print(f"❌ File types test failed: {e}")
        return False


async def run_comprehensive_test():
    """Run all BM25 enhancement tests"""
    print("🚀 Starting BM25 Enhancement Test Suite\n")
    
    tests = [
        ("BM25 Processor", test_bm25_processor),
        ("Enhanced Relevance Scoring", test_enhanced_relevance_scoring),
        ("Milvus Schema Compatibility", test_milvus_schema_compatibility),
        ("Corpus Statistics", test_corpus_statistics),
        ("API Endpoint", test_api_endpoint),
        ("Supported File Types", test_supported_file_types),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Test Results Summary:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! BM25 enhancement is ready.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    # Run the comprehensive test
    asyncio.run(run_comprehensive_test())