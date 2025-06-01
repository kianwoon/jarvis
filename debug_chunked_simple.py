#!/usr/bin/env python3
"""
Simple debug of chunked processing components
"""

import sys
import os
sys.path.append('/Users/kianwoonwong/Downloads/jarvis')

def test_detection_only():
    """Test just the detection logic"""
    print("🔍 Testing Large Generation Detection (isolated)")
    print("=" * 50)
    
    # Copy detection logic to avoid imports
    import re
    
    def isolated_detect_large_output_potential(question: str) -> dict:
        large_output_indicators = [
            "generate", "create", "list", "write", "develop", "design", "build",
            "comprehensive", "detailed", "complete", "full", "extensive", "thorough",
            "step by step", "step-by-step", "all", "many", "multiple", "various",
            "questions", "examples", "ideas", "recommendations", "strategies", "options",
            "points", "items", "factors", "aspects", "benefits", "advantages", "features"
        ]
        
        score = 0
        question_lower = question.lower()
        matched_indicators = []
        
        for indicator in large_output_indicators:
            if indicator in question_lower:
                score += 1
                matched_indicators.append(indicator)
        
        numbers = re.findall(r'\b(\d+)\b', question)
        max_number = max([int(n) for n in numbers], default=0)
        
        # Determine if likely large
        is_likely_large = False
        if max_number >= 30:
            is_likely_large = True
        elif max_number >= 20 and score >= 2:
            is_likely_large = True
        elif score >= 3 and any(keyword in question_lower for keyword in ["comprehensive", "detailed", "all", "many"]):
            is_likely_large = True
        elif max_number > 0 and max_number < 20:
            is_likely_large = False
        
        estimated_items = max_number if max_number > 10 else (score * 15 if score >= 3 else 10)
        
        return {
            "likely_large": is_likely_large,
            "estimated_items": estimated_items,
            "score": score,
            "max_number": max_number,
            "matched_indicators": matched_indicators
        }
    
    test_question = "generate 50 interview questions, for role of AI director for bank."
    result = isolated_detect_large_output_potential(test_question)
    
    print(f"Question: {test_question}")
    print(f"Result: {result}")
    
    if result["likely_large"] and result["estimated_items"] >= 20:
        print("✅ Should trigger chunked processing")
        return True
    else:
        print("❌ Won't trigger chunked processing")
        return False

def test_multi_agent_import():
    """Test if MultiAgentSystem can be imported"""
    print("\n🔧 Testing MultiAgentSystem Import")
    print("=" * 50)
    
    try:
        from app.langchain.multi_agent_system_simple import MultiAgentSystem
        print("✅ MultiAgentSystem imported successfully")
        
        # Try creating instance
        system = MultiAgentSystem(conversation_id="test_123")
        print("✅ MultiAgentSystem instance created")
        
        return True
    except Exception as e:
        print(f"❌ MultiAgentSystem import/creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_task_decomposer():
    """Test TaskDecomposer"""
    print("\n⚙️ Testing TaskDecomposer")
    print("=" * 50)
    
    try:
        from app.agents.task_decomposer import TaskDecomposer
        print("✅ TaskDecomposer imported")
        
        decomposer = TaskDecomposer()
        print("✅ TaskDecomposer created")
        
        # Test sync version if available or mock async
        import asyncio
        
        async def test_decompose():
            chunks = await decomposer.decompose_large_task(
                query="Generate 50 interview questions for AI director",
                target_count=50
            )
            return chunks
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            chunks = loop.run_until_complete(test_decompose())
            print(f"✅ Decomposed into {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                print(f"  Chunk {i+1}: items {chunk.start_index}-{chunk.end_index}")
            return True
        except Exception as e:
            print(f"❌ Decomposition failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            loop.close()
            
    except Exception as e:
        print(f"❌ TaskDecomposer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_redis_manager():
    """Test Redis Continuation Manager"""
    print("\n💾 Testing Redis Continuation Manager") 
    print("=" * 50)
    
    try:
        from app.agents.redis_continuation_manager import RedisContinuityManager
        print("✅ RedisContinuityManager imported")
        
        manager = RedisContinuityManager(session_id="debug_session_123")
        print("✅ RedisContinuityManager created")
        
        # Test Redis connection
        from app.core.redis_client import get_redis_client
        redis_client = get_redis_client()
        if redis_client:
            print("✅ Redis connection available")
        else:
            print("⚠️  Redis not available - will use fallback")
        
        return True
        
    except Exception as e:
        print(f"❌ Redis manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent_availability():
    """Test if continuation agent is available"""
    print("\n🤖 Testing Agent Availability")
    print("=" * 50)
    
    try:
        from app.core.langgraph_agents_cache import get_langgraph_agents, get_agent_by_role
        
        agents = get_langgraph_agents()
        print(f"✅ Retrieved {len(agents)} agents")
        
        # Look for continuation agent
        continuation_agent = get_agent_by_role("continuation_agent")
        if continuation_agent:
            print("✅ Continuation agent found")
            print(f"   Agent: {continuation_agent.get('name', 'unknown')}")
        else:
            print("❌ Continuation agent not found")
            print("   Available agents:")
            for agent in agents:
                print(f"   - {agent.get('role', 'unknown')}: {agent.get('name', 'unknown')}")
        
        return continuation_agent is not None
        
    except Exception as e:
        print(f"❌ Agent availability test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Simple Debug of Chunked Processing Components")
    print("Testing each component individually")
    print("=" * 60)
    
    results = {}
    
    try:
        # Test each component
        results['detection'] = test_detection_only()
        results['multi_agent'] = test_multi_agent_import()
        results['task_decomposer'] = test_task_decomposer()
        results['redis_manager'] = test_redis_manager()
        results['agent_availability'] = test_agent_availability()
        
        print("\n🎯 Summary:")
        print("=" * 30)
        for test, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test:20}: {status}")
        
        # Diagnosis
        print("\n🔍 Diagnosis:")
        if not results.get('detection', False):
            print("❌ Detection not working - check thresholds")
        elif not results.get('multi_agent', False):
            print("❌ MultiAgentSystem issue - check imports/dependencies")
        elif not results.get('task_decomposer', False):
            print("❌ TaskDecomposer issue - check async handling")
        elif not results.get('redis_manager', False):
            print("❌ Redis manager issue - check Redis connection")
        elif not results.get('agent_availability', False):
            print("❌ Continuation agent missing - check database/agents setup")
        else:
            print("✅ All components working - issue might be in streaming/async conversion")
            
    except Exception as e:
        print(f"\n❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()