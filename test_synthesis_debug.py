#!/usr/bin/env python3

"""
Debug script to test SYNTHESIS mode execution in rag_answer function
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Set up logging first
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_synthesis_mode():
    """Test the exact execution flow that's failing"""
    
    # Import after path setup
    from app.langchain.service import rag_answer
    
    # Test parameters that match the failing case
    question = "Tell me about the strategic partnership between Beyondsoft and Alibaba"
    thinking = ""
    query_type = "SYNTHESIS"  # This is the key - SYNTHESIS mode
    conversation_id = "test_conv_123"
    collections = []
    collection_strategy = "auto"
    
    # Create hybrid context like the real scenario
    hybrid_context = {
        'pre_formatted_context': 'Test context about Beyondsoft-Alibaba partnership' * 50,  # ~4197 chars like real case
        'sources': [
            {
                'content': 'Beyondsoft has been a key partner of Alibaba for 16 years',
                'file': 'test.pdf',
                'page': 1,
                'score': 0.801,
                'collection': 'test'
            }
        ]
    }
    
    # Mock trace object
    class MockTrace:
        def __init__(self):
            self.id = "test_trace_123"
    
    trace = MockTrace()
    
    print(f"[TEST] Starting SYNTHESIS mode test")
    print(f"[TEST] question: {question}")
    print(f"[TEST] query_type: {query_type}")
    print(f"[TEST] hybrid_context length: {len(str(hybrid_context))}")
    
    try:
        # This should reproduce the exact execution path
        result = await rag_answer(
            question=question,
            thinking=thinking,
            query_type=query_type,
            conversation_id=conversation_id,
            collections=collections,
            collection_strategy=collection_strategy,
            hybrid_context=hybrid_context,
            trace=trace,
            stream=True  # This triggers the streaming path where it fails
        )
        
        print(f"[TEST] SUCCESS: Got result: {type(result)}")
        
        # If streaming, collect the stream
        if hasattr(result, '__aiter__'):
            chunks = []
            async for chunk in result:
                chunks.append(chunk)
                print(f"[TEST] Stream chunk: {chunk[:100]}...")
            print(f"[TEST] Total stream chunks: {len(chunks)}")
        else:
            print(f"[TEST] Non-streaming result: {result}")
            
    except Exception as e:
        print(f"[TEST] ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_synthesis_mode())