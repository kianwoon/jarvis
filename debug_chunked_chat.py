#!/usr/bin/env python3
"""
Debug chunked chat processing
"""

import sys
import os
sys.path.append('/Users/kianwoonwong/Downloads/jarvis')

def test_detection():
    """Test detection of large generation request"""
    print("🔍 Testing Large Generation Detection")
    print("=" * 50)
    
    from app.langchain.service import detect_large_output_potential, classify_query_type
    from app.core.llm_settings_cache import get_llm_settings
    
    test_question = "generate 50 interview questions, for role of AI director for bank."
    
    # Test detection
    detection_result = detect_large_output_potential(test_question)
    print(f"Question: {test_question}")
    print(f"Detection result: {detection_result}")
    
    # Test classification  
    try:
        llm_cfg = get_llm_settings()
        classification = classify_query_type(test_question, llm_cfg)
        print(f"Classification: {classification}")
        
        if classification == "LARGE_GENERATION":
            print("✅ Correctly detected as large generation")
        else:
            print("❌ Not detected as large generation")
            
    except Exception as e:
        print(f"❌ Classification failed: {e}")
        import traceback
        traceback.print_exc()

def test_chunked_system():
    """Test the chunked generation system directly"""
    print("\n🔧 Testing Chunked Generation System")
    print("=" * 50)
    
    try:
        from app.langchain.multi_agent_system_simple import MultiAgentSystem
        from app.agents.task_decomposer import TaskDecomposer
        
        print("✅ MultiAgentSystem import successful")
        
        # Test task decomposer
        decomposer = TaskDecomposer()
        print("✅ TaskDecomposer created")
        
        # Test async function (this might reveal the issue)
        import asyncio
        
        async def test_decomposition():
            chunks = await decomposer.decompose_large_task(
                query="Generate 50 interview questions for AI director at bank",
                target_count=50
            )
            print(f"✅ Task decomposed into {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                print(f"  Chunk {i+1}: {chunk.start_index}-{chunk.end_index} ({chunk.chunk_size} items)")
            return chunks
        
        # Run the test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            chunks = loop.run_until_complete(test_decomposition())
            print("✅ Task decomposition working")
        except Exception as e:
            print(f"❌ Task decomposition failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            loop.close()
            
    except Exception as e:
        print(f"❌ Chunked system test failed: {e}")
        import traceback
        traceback.print_exc()

def test_continuity_manager():
    """Test the continuity manager"""
    print("\n⚙️ Testing Continuity Manager")
    print("=" * 50)
    
    try:
        from app.agents.redis_continuation_manager import RedisContinuityManager
        from app.agents.task_decomposer import TaskDecomposer
        
        # Test Redis continuity manager
        manager = RedisContinuityManager(session_id="test_session_123")
        print("✅ RedisContinuityManager created")
        
        # Test decomposition
        decomposer = TaskDecomposer()
        
        import asyncio
        async def test_execution():
            # Create test chunks
            chunks = await decomposer.decompose_large_task(
                query="Generate 50 interview questions for AI director",
                target_count=50
            )
            
            print(f"✅ Created {len(chunks)} chunks for testing")
            
            # Test streaming (just first event)
            event_count = 0
            async for event in manager.execute_chunked_task(
                chunks=chunks,
                agent_name="continuation_agent"
            ):
                event_count += 1
                print(f"Event {event_count}: {event.get('type', 'unknown')} - {event.get('message', '')}")
                
                # Stop after a few events to avoid hanging
                if event_count >= 5:
                    print("🛑 Stopping after 5 events for testing")
                    break
                    
                if event.get("type") == "error":
                    print(f"❌ Error in chunked execution: {event.get('error')}")
                    break
        
        # Run test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(test_execution())
        except Exception as e:
            print(f"❌ Continuity manager test failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            loop.close()
            
    except Exception as e:
        print(f"❌ Continuity manager import failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Debug Chunked Chat Processing")
    print("Investigating why chunked generation stops")
    print("=" * 60)
    
    try:
        # Test 1: Detection
        test_detection()
        
        # Test 2: Chunked system
        test_chunked_system()
        
        # Test 3: Continuity manager
        test_continuity_manager()
        
        print("\n🎯 Debugging Summary:")
        print("- If detection works but chunked system fails, the issue is in MultiAgentSystem")
        print("- If continuity manager fails, the issue is in Redis or agent execution")
        print("- Check server logs for more detailed error messages")
        
    except Exception as e:
        print(f"\n❌ Debug test failed: {e}")
        import traceback
        traceback.print_exc()