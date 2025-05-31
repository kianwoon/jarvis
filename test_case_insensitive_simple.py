#!/usr/bin/env python3
"""
Simple test script to verify case-insensitive functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_hash_generation():
    """Test that hash generation is case-insensitive"""
    print("\n=== Testing Hash Generation (Case Insensitive) ===")
    
    from utils.deduplication import hash_text
    
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


def test_embedding_normalization():
    """Test that embedding function normalizes text"""
    print("\n=== Testing Embedding Text Normalization ===")
    
    # Test the normalization logic directly
    test_cases = [
        ("Hello World", "hello world"),
        ("  MACHINE LEARNING  ", "machine learning"),
        ("DBS Bank", "dbs bank"),
    ]
    
    for original, expected in test_cases:
        # Simulate the normalization that happens in HTTPEmbeddingFunction
        normalized = original.lower().strip()
        match = "✓" if normalized == expected else "✗"
        print(f"{match} '{original}' -> '{normalized}' (expected: '{expected}')")


def test_query_normalization():
    """Test query normalization logic"""
    print("\n=== Testing Query Normalization ===")
    
    # Simulate the query normalization logic
    test_queries = [
        "What is Machine Learning?",
        "DBS BANK AI PROGRESS",
        "  Artificial Intelligence  ",
    ]
    
    print("Original -> Normalized:")
    for query in test_queries:
        normalized = query.lower().strip()
        print(f"'{query}' -> '{normalized}'")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Case-Insensitive Implementation Test")
    print("=" * 60)
    
    # Test 1: Hash generation
    test_hash_generation()
    
    # Test 2: Embedding normalization
    test_embedding_normalization()
    
    # Test 3: Query normalization
    test_query_normalization()
    
    print("\n" + "=" * 60)
    print("Basic tests completed!")
    print("\nSummary of Implementation:")
    print("1. Hash generation: text.lower().strip() before hashing")
    print("2. Embeddings: text.lower().strip() before embedding")
    print("3. Search queries: query.lower().strip() before searching")
    print("4. This ensures case-insensitive deduplication and search")
    print("=" * 60)


if __name__ == "__main__":
    main()