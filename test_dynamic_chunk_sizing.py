#!/usr/bin/env python3
"""Test dynamic chunk sizing based on current model"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.dynamic_chunk_sizing import get_dynamic_chunk_sizer, get_optimal_chunk_config
from app.core.knowledge_graph_settings_cache import get_knowledge_graph_settings
from app.document_handlers.base import ExtractedChunk

def test_dynamic_chunk_sizing():
    """Test dynamic chunk sizing with current model configuration"""
    
    print("🧪 Testing Dynamic Chunk Sizing")
    print("=" * 60)
    
    # Get current settings
    kg_settings = get_knowledge_graph_settings()
    print(f"📋 Current Knowledge Graph Settings Keys: {list(kg_settings.keys())}")
    
    if 'model_config' in kg_settings:
        model_config = kg_settings['model_config']
        print(f"🔧 Model Config Keys: {list(model_config.keys())}")
        if 'model' in model_config:
            print(f"🎯 Current Model: {model_config['model']}")
    
    # Test dynamic chunk sizer
    try:
        chunk_sizer = get_dynamic_chunk_sizer()
        
        print(f"\n🧠 Dynamic Chunk Sizer Results:")
        print(f"   Detected Model: {chunk_sizer.model_name}")
        print(f"   Context Limit: {chunk_sizer.context_limit:,} tokens")
        print(f"   Optimal Chunk Size: {chunk_sizer.optimal_chunk_size:,} characters")
        print(f"   Should Use Large Chunks: {chunk_sizer.should_use_large_chunks()}")
        
        # Get chunk configuration
        config = chunk_sizer.get_chunk_configuration()
        print(f"\n⚙️  Chunk Configuration:")
        for key, value in config.items():
            print(f"   {key}: {value}")
        
        # Test with sample chunks
        print(f"\n🔧 Testing Chunk Optimization:")
        
        # Create sample small chunks (like current 2KB chunks)
        sample_chunks = []
        for i in range(5):
            content = f"Sample chunk {i+1}. " + "This is sample content. " * 50  # ~1KB each
            chunk = ExtractedChunk(
                content=content,
                metadata={'chunk_id': f'sample_{i+1}', 'source': 'test'},
                quality_score=0.8
            )
            sample_chunks.append(chunk)
        
        print(f"   Input: {len(sample_chunks)} chunks")
        for i, chunk in enumerate(sample_chunks):
            print(f"     Chunk {i+1}: {len(chunk.content):,} characters")
        
        # Optimize chunks
        optimized_chunks = chunk_sizer.optimize_chunks(sample_chunks)
        
        print(f"   Output: {len(optimized_chunks)} chunks")
        for i, chunk in enumerate(optimized_chunks):
            optimization = chunk.metadata.get('optimization', 'none')
            original_count = chunk.metadata.get('original_chunk_count', 1)
            print(f"     Chunk {i+1}: {len(chunk.content):,} characters (optimization: {optimization}, combined: {original_count})")
        
        # Calculate efficiency gain
        if config['processing_strategy'] == 'large_context':
            processing_calls_before = len(sample_chunks)
            processing_calls_after = len(optimized_chunks)
            efficiency_gain = ((processing_calls_before - processing_calls_after) / processing_calls_before) * 100
            print(f"\n📈 Efficiency Analysis:")
            print(f"   LLM calls before: {processing_calls_before}")
            print(f"   LLM calls after: {processing_calls_after}")
            print(f"   Efficiency gain: {efficiency_gain:.1f}%")
            
            if efficiency_gain > 0:
                print(f"✅ Large chunk optimization is working! Fewer LLM calls needed.")
            else:
                print(f"ℹ️  No optimization needed for this chunk size.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing dynamic chunk sizing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🔬 Dynamic Chunk Sizing Test")
    print("Testing model-based chunk optimization")
    print("=" * 60)
    
    success = test_dynamic_chunk_sizing()
    
    if success:
        print("\n✅ Dynamic chunk sizing test completed successfully!")
        print("The system is now optimized for your current model.")
    else:
        print("\n❌ Dynamic chunk sizing test failed.")
        print("Check the error messages above.")

if __name__ == "__main__":
    main()