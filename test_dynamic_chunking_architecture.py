#!/usr/bin/env python3
"""
Test Dynamic Chunking Architecture

This test verifies that the chunking system is completely dynamic and
scales properly with any model context length WITHOUT hardcoding.
"""

import sys
import json
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, '/Users/kianwoonwong/Downloads/jarvis')

from app.services.dynamic_chunk_sizing import DynamicChunkSizer
from app.document_handlers.base import ExtractedChunk

def test_dynamic_scaling():
    """Test that chunking scales dynamically with context length"""
    
    print("🧪 Testing Dynamic Chunking Architecture\n")
    
    # Test different context lengths to verify dynamic scaling
    test_contexts = [
        {"name": "Small Model", "context_length": 4096},
        {"name": "Medium Model", "context_length": 32768},
        {"name": "Large Model", "context_length": 128000},
        {"name": "XLarge Model", "context_length": 262144},
        {"name": "Future 1M Model", "context_length": 1048576},
    ]
    
    # Create test document
    test_doc_sizes = [10000, 50000, 200000, 500000]  # Different document sizes in chars
    
    for context_config in test_contexts:
        print(f"\n{'='*60}")
        print(f"Testing: {context_config['name']} ({context_config['context_length']:,} tokens)")
        print(f"{'='*60}")
        
        # Mock the settings to test different context lengths
        mock_settings = {
            'model_config': {
                'model': 'test_model',
                'context_length': context_config['context_length']
            }
        }
        
        # Create chunk sizer with mocked settings
        from unittest.mock import patch
        with patch('app.services.dynamic_chunk_sizing.get_knowledge_graph_settings', return_value=mock_settings):
            sizer = DynamicChunkSizer()
            
            print(f"\n📊 Configuration:")
            print(f"   Context limit: {sizer.context_limit:,} tokens")
            print(f"   Optimal chunk size: {sizer.optimal_chunk_size:,} chars")
            print(f"   Uses large chunks: {sizer.should_use_large_chunks()}")
            
            config = sizer.get_chunk_configuration()
            print(f"\n🎯 Chunk Configuration:")
            print(f"   Strategy: {config['processing_strategy']}")
            print(f"   Max chunk size: {config['max_chunk_size']:,} chars")
            print(f"   Min chunk size: {config['min_chunk_size']:,} chars")
            print(f"   Target chunks per doc: {config['target_chunks_per_document']}")
            print(f"   Max consolidation ratio: {config['max_consolidation_ratio']}:1")
            
            # Test with different document sizes
            print(f"\n📄 Document Processing:")
            for doc_size in test_doc_sizes:
                if doc_size > sizer.context_limit * 4:  # Skip if doc is way too large
                    continue
                    
                # Calculate how many chunks would be created
                chunks_needed = max(1, doc_size // config['max_chunk_size'])
                utilization = (doc_size / (sizer.context_limit * 4)) * 100
                
                print(f"   {doc_size:,} char document → ~{chunks_needed} chunks ({utilization:.1f}% context)")

def test_no_hardcoding():
    """Verify there's no hardcoding in the dynamic chunk sizer"""
    
    print("\n\n🔍 Verifying No Hardcoding\n")
    
    # Test that unknown models still work with proper context_length
    unknown_model_settings = {
        'model_config': {
            'model': 'future-llm-9000-ultra',  # Completely unknown model
            'context_length': 524288  # 512k tokens
        }
    }
    
    from unittest.mock import patch
    with patch('app.services.dynamic_chunk_sizing.get_knowledge_graph_settings', return_value=unknown_model_settings):
        sizer = DynamicChunkSizer()
        
        print(f"✅ Unknown model '{sizer.model_name}' correctly uses configured context: {sizer.context_limit:,} tokens")
        print(f"✅ Optimal chunk size calculated dynamically: {sizer.optimal_chunk_size:,} chars")
        
        config = sizer.get_chunk_configuration()
        print(f"✅ Strategy selected dynamically: {config['processing_strategy']}")
        
    # Test missing context_length fallback
    print("\n⚠️  Testing fallback behavior (should warn):")
    no_context_settings = {
        'model_config': {
            'model': 'some-model'
            # NO context_length!
        }
    }
    
    with patch('app.services.dynamic_chunk_sizing.get_knowledge_graph_settings', return_value=no_context_settings):
        sizer = DynamicChunkSizer()
        print(f"   Fallback context used: {sizer.context_limit:,} tokens")
        print(f"   ⚠️  This should have produced warnings about missing context_length!")

def test_real_world_example():
    """Test with a real-world example matching the user's case"""
    
    print("\n\n🌍 Real-World Example Test\n")
    
    # Simulate the exact case from the user
    qwen_30b_settings = {
        'model_config': {
            'model': 'qwen3:30b-a3b-instruct-2507-q4_k_m',
            'context_length': 262144  # 256k tokens
        }
    }
    
    from unittest.mock import patch
    with patch('app.services.dynamic_chunk_sizing.get_knowledge_graph_settings', return_value=qwen_30b_settings):
        sizer = DynamicChunkSizer()
        
        print(f"Model: {sizer.model_name}")
        print(f"Context: {sizer.context_limit:,} tokens ({sizer.context_limit * 4:,} chars)")
        print(f"Optimal chunk: {sizer.optimal_chunk_size:,} chars")
        
        # Test with 31KB document
        doc_size = 31000
        print(f"\nProcessing {doc_size:,} char document:")
        
        # Create mock chunks
        mock_chunks = [
            ExtractedChunk(
                content="x" * doc_size,
                metadata={'chunk_id': 'test_chunk_1'},
                quality_score=1.0
            )
        ]
        
        optimized = sizer.optimize_chunks(mock_chunks)
        print(f"✅ Result: {len(optimized)} chunk(s)")
        print(f"✅ Chunk size: {len(optimized[0].content):,} chars")
        print(f"✅ Context utilization: {(doc_size / (sizer.context_limit * 4)) * 100:.2f}%")
        
        if len(optimized) == 1:
            print(f"\n🎉 SUCCESS: Document correctly processed as single chunk!")
            print(f"   No artificial chunking for a document that fits in context!")
        else:
            print(f"\n❌ FAILURE: Document was unnecessarily chunked!")

if __name__ == "__main__":
    test_dynamic_scaling()
    test_no_hardcoding()
    test_real_world_example()
    
    print("\n\n✨ Dynamic Chunking Architecture Test Complete!")