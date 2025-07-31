#!/usr/bin/env python3
"""
Minimal test to reproduce the multi-agent streaming error.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_multi_agent_minimal():
    """Test the multi-agent streaming with minimal setup"""
    
    print("Testing multi-agent streaming initialization...")
    
    try:
        from app.langchain.fixed_multi_agent_streaming import fixed_multi_agent_streaming
        
        # Test with minimal parameters
        question = "Test question about AI automation"
        conversation_id = "test_conversation"
        
        print(f"Calling fixed_multi_agent_streaming with question: {question}")
        
        count = 0
        async for event in fixed_multi_agent_streaming(question, conversation_id):
            count += 1
            print(f"Event {count}: {event[:100]}...")
            
            # Stop after 5 events to avoid full execution
            if count >= 5:
                print("Stopping test after 5 events")
                break
                
        print(f"✅ Multi-agent streaming worked! Processed {count} events")
        return True
        
    except Exception as e:
        print(f"❌ Multi-agent streaming failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_components_individually():
    """Test individual components that might be causing the error"""
    
    print("\n" + "="*60)
    print("TESTING INDIVIDUAL COMPONENTS")
    print("="*60)
    
    # Test 1: LLM Settings
    try:
        from app.core.llm_settings_cache import get_llm_settings
        settings = get_llm_settings()
        print("✅ LLM settings loaded")
    except Exception as e:
        print(f"❌ LLM settings failed: {e}")
        return False
    
    # Test 2: Langfuse integration
    try:
        from app.core.langfuse_integration import get_tracer
        tracer = get_tracer()
        print("✅ Langfuse tracer loaded")
    except Exception as e:
        print(f"❌ Langfuse tracer failed: {e}")
        return False
    
    # Test 3: Multi-agent system
    try:
        from app.langchain.langgraph_multi_agent_system import LangGraphMultiAgentSystem
        system = LangGraphMultiAgentSystem("test_conversation")
        print("✅ Multi-agent system initialized")
    except Exception as e:
        print(f"❌ Multi-agent system failed: {e}")
        return False
    
    # Test 4: Agent selector
    try:
        from app.core.intelligent_agent_selector import IntelligentAgentSelector
        selector = IntelligentAgentSelector()
        print("✅ Agent selector initialized")
    except Exception as e:
        print(f"❌ Agent selector failed: {e}")
        return False
    
    print("✅ All components loaded successfully")
    return True

async def main():
    print("Multi-Agent Minimal Test")
    print("=" * 50)
    
    # First test components individually
    components_ok = await test_components_individually()
    
    if not components_ok:
        print("❌ Component test failed, skipping full test")
        return
    
    # Then test the full streaming
    await test_multi_agent_minimal()

if __name__ == "__main__":
    asyncio.run(main())