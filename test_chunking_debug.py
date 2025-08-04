#!/usr/bin/env python3
"""
Debug chunking strategy to see what's happening
"""

import sys
import os
import logging

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Set up detailed logging
logging.basicConfig(level=logging.INFO)

def test_chunking_debug():
    """Debug the chunking process"""
    try:
        from app.services.dynamic_chunk_sizing import get_dynamic_chunk_sizer
        from app.document_handlers.base import ExtractedChunk
        
        # Create chunk sizer instance
        chunk_sizer = get_dynamic_chunk_sizer()
        print(f"Model: {chunk_sizer.model_name}, Context: {chunk_sizer.context_limit:,}")
        
        # Test business document configuration
        business_config = chunk_sizer.get_chunk_configuration('technology_strategy')
        print(f"Business config strategy: {business_config['processing_strategy']}")
        
        # Create mock chunks
        business_content = "DBS Bank Technology Strategy content " * 50  # Small content
        
        mock_chunks = []
        chunk_size = 800  # Very small chunks to force consolidation
        for i in range(0, len(business_content), chunk_size):
            chunk_content = business_content[i:i + chunk_size]
            if len(chunk_content.strip()) > 50:
                chunk = ExtractedChunk(
                    content=chunk_content,
                    metadata={'chunk_id': f'debug_chunk_{i//chunk_size}'},
                    quality_score=0.8
                )
                mock_chunks.append(chunk)
        
        print(f"Created {len(mock_chunks)} chunks")
        for i, chunk in enumerate(mock_chunks):
            print(f"  Chunk {i}: {len(chunk.content)} chars")
        
        # Test chunking optimization
        print("\n--- Starting optimization ---")
        optimized_chunks = chunk_sizer.optimize_chunks(mock_chunks, 'technology_strategy')
        
        print(f"\nResults: {len(mock_chunks)} → {len(optimized_chunks)} chunks")
        for i, chunk in enumerate(optimized_chunks):
            print(f"  Optimized chunk {i}: {len(chunk.content)} chars")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_chunking_debug()