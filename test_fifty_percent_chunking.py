#!/usr/bin/env python3
"""
Test the new 50% Model Utilization Chunking Strategy

This test verifies that the chunking system now uses a simple, predictable approach:
- 256K model → 50% = 128K tokens = ~512K characters target chunk size  
- 31K document should result in 1 chunk (fits within 512K capacity)
"""

import os
import sys
import json
from typing import List

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.dynamic_chunk_sizing import get_dynamic_chunk_sizer, optimize_chunks_for_model
from app.document_handlers.base import ExtractedChunk

def create_test_document(size_chars: int) -> List[ExtractedChunk]:
    """Create a test document of specified size"""
    # Create content that's exactly the specified size
    content = "This is test content. " * (size_chars // 22)
    content = content[:size_chars]  # Trim to exact size
    
    return [ExtractedChunk(
        content=content,
        metadata={'chunk_id': 'test_chunk_0', 'source': 'test_document'},
        quality_score=1.0
    )]

def test_fifty_percent_rule():
    """Test the 50% model utilization rule"""
    print("🧪 Testing 50% Model Utilization Chunking Strategy")
    print("=" * 60)
    
    # Get the chunk sizer
    sizer = get_dynamic_chunk_sizer()
    
    # Display model configuration
    print(f"📊 Model Configuration:")
    print(f"   Model: {sizer.model_name}")
    print(f"   Context Limit: {sizer.context_limit:,} tokens")
    print(f"   50% Target: {sizer.context_limit // 2:,} tokens")
    print(f"   Target Chunk Size: {sizer.optimal_chunk_size:,} characters")
    print()
    
    # Test cases
    test_cases = [
        ("31K document (should be 1 chunk)", 31_000),
        ("100K document (should be 1 chunk)", 100_000),
        ("500K document (should be 1 chunk)", 500_000),
        ("600K document (should be 2 chunks)", 600_000),
        ("1M document (should be 2 chunks)", 1_000_000),
    ]
    
    for test_name, doc_size in test_cases:
        print(f"🔍 Test: {test_name}")
        print(f"   Document size: {doc_size:,} characters")
        
        # Create test document
        chunks = create_test_document(doc_size)
        print(f"   Input chunks: {len(chunks)}")
        
        # Optimize chunks
        optimized = optimize_chunks_for_model(chunks, 'general', 'knowledge_graph')
        
        print(f"   Output chunks: {len(optimized)}")
        if optimized:
            avg_size = sum(len(c.content) for c in optimized) // len(optimized)
            total_size = sum(len(c.content) for c in optimized)
            print(f"   Average chunk size: {avg_size:,} characters")
            print(f"   Total size: {total_size:,} characters")
            
            # Calculate utilization
            utilization = (avg_size / (sizer.context_limit * 4)) * 100
            print(f"   Model utilization: {utilization:.1f}%")
            
            # Check optimization metadata
            for i, chunk in enumerate(optimized):
                optimization = chunk.metadata.get('optimization', 'unknown')
                print(f"   Chunk {i}: {len(chunk.content):,} chars ({optimization})")
        
        print()

def test_specific_31k_case():
    """Test the specific 31K document case mentioned in requirements"""
    print("🎯 Specific Test: 31K Document Case")
    print("=" * 40)
    
    sizer = get_dynamic_chunk_sizer()
    
    # Expected behavior for 256K model with 31K document:
    # - 256K tokens = 262,144 tokens
    # - 50% utilization = 131,072 tokens ≈ 524,288 characters
    # - 31K characters fits easily in 524K capacity
    # - Result: Should be 1 chunk
    
    expected_target_size = sizer.optimal_chunk_size
    print(f"📐 Expected target chunk size (50% of model): {expected_target_size:,} chars")
    
    # Create 31K document
    doc_size = 31_000
    chunks = create_test_document(doc_size)
    
    print(f"📄 Test document: {doc_size:,} characters")
    print(f"✅ Should fit in single chunk: {doc_size < expected_target_size}")
    
    # Process chunks
    optimized = optimize_chunks_for_model(chunks, 'general', 'knowledge_graph')
    
    print(f"\n📊 Results:")
    print(f"   Input chunks: {len(chunks)}")
    print(f"   Output chunks: {len(optimized)}")
    
    if len(optimized) == 1:
        print("✅ SUCCESS: 31K document correctly processed as single chunk!")
        chunk = optimized[0]
        print(f"   Chunk size: {len(chunk.content):,} characters")
        print(f"   Optimization: {chunk.metadata.get('optimization')}")
        utilization = (len(chunk.content) / (sizer.context_limit * 4)) * 100
        print(f"   Model utilization: {utilization:.1f}%")
    else:
        print(f"❌ FAILED: Expected 1 chunk, got {len(optimized)} chunks")
        for i, chunk in enumerate(optimized):
            print(f"   Chunk {i}: {len(chunk.content):,} chars")

def main():
    """Run all tests"""
    print("🚀 50% Model Utilization Chunking Tests")
    print("=" * 80)
    print()
    
    try:
        test_fifty_percent_rule()
        test_specific_31k_case()
        
        print("🎉 All tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())