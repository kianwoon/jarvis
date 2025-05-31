#!/usr/bin/env python3
"""
Test script to verify case-insensitive search functionality in Milvus
Tests hash generation, embedding generation, and search queries with case variations
"""

import asyncio
import json
from app.langchain.service import rag_answer, handle_rag_query, keyword_search_milvus
from app.core.vector_db_settings_cache import get_vector_db_settings
from utils.deduplication import hash_text
from app.api.v1.endpoints.document import HTTPEmbeddingFunction
from app.core.embedding_settings_cache import get_embedding_settings


def test_hash_generation():
    """Test that hash generation is case-insensitive"""
    print("\n=== Testing Hash Generation (Case Insensitive) ===")
    
    test_cases = [
        ("Hello World", "hello world"),
        ("Machine Learning", "machine learning"),
        ("DBS Bank AI Strategy", "dbs bank ai strategy"),
        ("   TRIM TEST   ", "trim test"),
    ]
    
    for original, lowercase in test_cases:
        hash1 = hash_text(original)
        hash2 = hash_text(lowercase)
        match = "✓" if hash1 == hash2 else "✗"
        print(f"{match} '{original}' vs '{lowercase}': {hash1 == hash2}")
        if hash1 != hash2:
            print(f"   Hash 1: {hash1}")
            print(f"   Hash 2: {hash2}")


def test_embedding_generation():
    """Test that embedding generation is case-insensitive"""
    print("\n=== Testing Embedding Generation (Case Insensitive) ===")
    
    try:
        # Get embedding configuration
        embedding_cfg = get_embedding_settings()
        embedding_endpoint = embedding_cfg.get('embedding_endpoint')
        
        if embedding_endpoint:
            embeddings = HTTPEmbeddingFunction(embedding_endpoint)
            
            test_cases = [
                ("Machine Learning", "machine learning"),
                ("DBS BANK", "dbs bank"),
                ("Artificial Intelligence", "artificial intelligence"),
            ]
            
            for original, lowercase in test_cases:
                try:
                    # Generate embeddings
                    emb1 = embeddings.embed_query(original)
                    emb2 = embeddings.embed_query(lowercase)
                    
                    # Check if embeddings are identical
                    are_identical = all(abs(a - b) < 1e-6 for a, b in zip(emb1, emb2))
                    match = "✓" if are_identical else "✗"
                    
                    print(f"{match} '{original}' vs '{lowercase}': Embeddings {'match' if are_identical else 'differ'}")
                    
                    if not are_identical:
                        # Calculate cosine similarity
                        import math
                        dot_product = sum(a * b for a, b in zip(emb1, emb2))
                        norm1 = math.sqrt(sum(a * a for a in emb1))
                        norm2 = math.sqrt(sum(b * b for b in emb2))
                        similarity = dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0
                        print(f"   Cosine similarity: {similarity:.6f}")
                        
                except Exception as e:
                    print(f"✗ Error testing '{original}': {str(e)}")
                    
        else:
            print("⚠️  No embedding endpoint configured, skipping embedding tests")
            
    except Exception as e:
        print(f"✗ Failed to initialize embeddings: {str(e)}")


def test_search_queries():
    """Test search functionality with case variations"""
    print("\n=== Testing Search Queries (Case Insensitive) ===")
    
    # Test queries with different cases
    test_queries = [
        ("DBS bank AI progress", "dbs bank ai progress"),
        ("Machine Learning Applications", "machine learning applications"),
        ("ARTIFICIAL INTELLIGENCE", "artificial intelligence"),
        ("What is RAG?", "what is rag?"),
    ]
    
    try:
        vector_db_cfg = get_vector_db_settings()
        milvus_cfg = vector_db_cfg.get("milvus", {})
        
        if not milvus_cfg.get("status"):
            print("⚠️  Milvus is not enabled, skipping search tests")
            return
            
        collection = milvus_cfg.get("MILVUS_DEFAULT_COLLECTION", "default_knowledge")
        uri = milvus_cfg.get("MILVUS_URI")
        token = milvus_cfg.get("MILVUS_TOKEN")
        
        for original_query, lowercase_query in test_queries:
            print(f"\nTesting: '{original_query}' vs '{lowercase_query}'")
            
            try:
                # Test keyword search
                print("  Keyword Search:")
                docs1 = keyword_search_milvus(original_query, collection, uri, token)
                docs2 = keyword_search_milvus(lowercase_query, collection, uri, token)
                
                # Compare results
                docs1_content = set(doc.page_content[:100] for doc in docs1[:5])
                docs2_content = set(doc.page_content[:100] for doc in docs2[:5])
                
                overlap = len(docs1_content.intersection(docs2_content))
                total = max(len(docs1_content), len(docs2_content))
                
                if total > 0:
                    similarity = overlap / total
                    match = "✓" if similarity > 0.8 else "≈" if similarity > 0.5 else "✗"
                    print(f"    {match} Result similarity: {similarity:.2%} ({overlap}/{total} docs match)")
                else:
                    print(f"    ⚠️  No results found for either query")
                    
            except Exception as e:
                print(f"    ✗ Error: {str(e)}")
                
    except Exception as e:
        print(f"✗ Failed to run search tests: {str(e)}")


async def test_rag_answer():
    """Test the full RAG answer pipeline with case variations"""
    print("\n=== Testing Full RAG Pipeline (Case Insensitive) ===")
    
    test_queries = [
        ("What is DBS doing in AI?", "what is dbs doing in ai?"),
        ("Machine Learning applications", "MACHINE LEARNING APPLICATIONS"),
    ]
    
    for original_query, case_variant in test_queries:
        print(f"\nTesting: '{original_query}' vs '{case_variant}'")
        
        try:
            # Get RAG context for both queries
            print("  Getting RAG context...")
            context1, _ = handle_rag_query(original_query, thinking=False)
            context2, _ = handle_rag_query(case_variant, thinking=False)
            
            # Compare contexts
            if context1 and context2:
                # Simple similarity check based on common words
                words1 = set(context1.lower().split())
                words2 = set(context2.lower().split())
                
                overlap = len(words1.intersection(words2))
                total = max(len(words1), len(words2))
                
                if total > 0:
                    similarity = overlap / total
                    match = "✓" if similarity > 0.7 else "≈" if similarity > 0.5 else "✗"
                    print(f"    {match} Context similarity: {similarity:.2%}")
                    print(f"    Context 1 length: {len(context1)} chars")
                    print(f"    Context 2 length: {len(context2)} chars")
                else:
                    print(f"    ⚠️  Empty contexts returned")
            else:
                print(f"    ⚠️  No context found for one or both queries")
                print(f"    Context 1: {'Found' if context1 else 'Empty'}")
                print(f"    Context 2: {'Found' if context2 else 'Empty'}")
                
        except Exception as e:
            print(f"    ✗ Error: {str(e)}")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Case-Insensitive Search Test Suite")
    print("=" * 60)
    
    # Test 1: Hash generation
    test_hash_generation()
    
    # Test 2: Embedding generation
    test_embedding_generation()
    
    # Test 3: Search queries
    test_search_queries()
    
    # Test 4: Full RAG pipeline
    asyncio.run(test_rag_answer())
    
    print("\n" + "=" * 60)
    print("Test suite completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()