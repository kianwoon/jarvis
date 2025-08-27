#!/usr/bin/env python3
"""Debug embedding generation issues"""
import asyncio
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, '/Users/kianwoonwong/Downloads/jarvis')

async def test_embedding_simple():
    """Simple embedding test"""
    try:
        from app.services.notebook_rag_service import NotebookRAGService
        
        print("📍 Creating NotebookRAGService instance...")
        rag_service = NotebookRAGService()
        
        print("📍 Getting embedding function...")
        embedding_function = rag_service._get_embedding_function()
        print(f"✅ Got embedding function: {type(embedding_function)}")
        
        print("📍 Testing simple query embedding...")
        test_query = "test query"
        result = await rag_service._get_query_embedding(test_query, embedding_function)
        print(f"✅ Generated embedding of length: {len(result)}")
        print(f"✅ First 5 values: {result[:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_embedding_simple())
    if success:
        print("\n🎉 Embedding generation test passed!")
    else:
        print("\n💥 Embedding generation test failed!")
        sys.exit(1)