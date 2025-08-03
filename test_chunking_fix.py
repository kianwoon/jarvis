#!/usr/bin/env python3
"""
Test that the chunking fix is working correctly for 256k context models
"""

from app.services.dynamic_chunk_sizing import DynamicChunkSizer
from app.document_handlers.base import ExtractedChunk

def test_chunking_with_256k_model():
    """Test chunking behavior with the fixed configuration"""
    
    print("🧪 Testing chunking with 256k context model...")
    
    # Initialize the chunk sizer (will load settings from database)
    sizer = DynamicChunkSizer()
    
    print(f"\n📊 Configuration loaded:")
    print(f"   Model: {sizer.model_name}")
    print(f"   Context limit: {sizer.context_limit:,} tokens")
    print(f"   Optimal chunk size: {sizer.optimal_chunk_size:,} characters")
    
    # Get chunk configuration
    config = sizer.get_chunk_configuration()
    print(f"\n🔧 Chunk configuration:")
    print(f"   Strategy: {config['processing_strategy']}")
    print(f"   Max chunk size: {config['max_chunk_size']:,} chars")
    print(f"   Target chunks per document: {config['target_chunks_per_document']}")
    print(f"   Max consolidation ratio: {config['max_consolidation_ratio']}")
    
    # Simulate a document with 31,238 characters (like the DBS document)
    doc_size = 31238
    num_initial_chunks = 27
    
    # Create fake chunks
    chunk_size = doc_size // num_initial_chunks
    chunks = []
    for i in range(num_initial_chunks):
        chunk = ExtractedChunk(
            content="x" * chunk_size,
            metadata={
                'chunk_id': f'chunk_{i}',
                'document_id': 'test_doc',
                'chunk_index': i
            },
            quality_score=0.95
        )
        chunks.append(chunk)
    
    print(f"\n📄 Test document:")
    print(f"   Total size: {doc_size:,} chars")
    print(f"   Initial chunks: {num_initial_chunks}")
    print(f"   Average chunk size: {chunk_size:,} chars")
    
    # Optimize chunks
    print(f"\n🚀 Optimizing chunks...")
    optimized_chunks = sizer.optimize_chunks(chunks)
    
    print(f"\n✅ Results:")
    print(f"   Original chunks: {len(chunks)}")
    print(f"   Optimized chunks: {len(optimized_chunks)}")
    print(f"   Consolidation ratio: {len(chunks) / len(optimized_chunks):.1f}:1")
    
    # Show chunk sizes
    print(f"\n📊 Optimized chunk sizes:")
    for i, chunk in enumerate(optimized_chunks):
        print(f"   Chunk {i}: {len(chunk.content):,} chars")
        if 'utilization_ratio' in chunk.metadata:
            print(f"      Context utilization: {chunk.metadata['utilization_ratio']}")
    
    # Check if we're using full context
    if len(optimized_chunks) == 1:
        print(f"\n🎉 SUCCESS! Document consolidated into 1 chunk using full model context!")
    else:
        print(f"\n⚠️  WARNING: Document still split into {len(optimized_chunks)} chunks")
        print(f"   This may indicate the fix needs adjustment")

if __name__ == "__main__":
    test_chunking_with_256k_model()