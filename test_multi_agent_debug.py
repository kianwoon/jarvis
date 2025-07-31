#!/usr/bin/env python3
"""
Debug multi-agent system to identify why agents are falling back to template responses.
"""

import asyncio
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_agent_llm_direct():
    """Test agent LLM configuration directly"""
    
    print("="*80)
    print("TESTING MULTI-AGENT LLM CONFIGURATION")
    print("="*80)
    
    from app.llm.ollama import OllamaLLM
    from app.llm.base import LLMConfig
    from app.core.llm_settings_cache import get_second_llm_full_config
    from app.core.langgraph_agents_cache import get_agent_by_name, get_all_agents
    
    # Get all available agents
    all_agents = get_all_agents()
    print(f"Total agents available: {len(all_agents)}")
    
    # Test a specific agent configuration
    test_agent_name = "cto_agent"  # The one that's failing
    agent_data = get_agent_by_name(test_agent_name)
    
    if not agent_data:
        print(f"❌ Agent '{test_agent_name}' not found")
        return False
    
    print(f"\n🔍 Testing agent: {test_agent_name}")
    print(f"Agent role: {agent_data.get('role', 'No role')}")
    
    config = agent_data.get('config', {})
    print(f"Agent config keys: {list(config.keys())}")
    
    # Determine which LLM configuration to use (mimicking the multi-agent logic)
    if config.get('use_main_llm'):
        from app.core.llm_settings_cache import get_main_llm_full_config
        main_llm_config = get_main_llm_full_config()
        agent_model = main_llm_config.get('model')
        max_tokens = config.get('max_tokens', main_llm_config.get('max_tokens', 2000))
        temperature = config.get('temperature', main_llm_config.get('temperature', 0.7))
        print(f"✅ Using main_llm: {agent_model}")
    elif config.get('use_second_llm') or not config.get('model'):
        second_llm_config = get_second_llm_full_config()
        agent_model = second_llm_config.get('model')
        max_tokens = config.get('max_tokens', second_llm_config.get('max_tokens', 2000))
        temperature = config.get('temperature', second_llm_config.get('temperature', 0.7))
        print(f"✅ Using second_llm: {agent_model}")
    else:
        agent_model = config.get('model')
        max_tokens = config.get('max_tokens', 2000)
        temperature = config.get('temperature', 0.7)
        print(f"✅ Using specific model: {agent_model}")
    
    print(f"Final config - Model: {agent_model}, Temp: {temperature}, Max tokens: {max_tokens}")
    
    # Create the LLM configuration
    llm_config = LLMConfig(
        model_name=agent_model,
        temperature=temperature,
        top_p=config.get('top_p', 0.9),
        max_tokens=max_tokens
    )
    
    # Test with actual agent system prompt
    system_prompt = agent_data.get('system_prompt', 'You are an AI assistant.')
    question = "AI automation workflow challenge traditional automation workflow like UIpath and Ansible. Are these new Ai automation workflow like n8n and dify could replace them?"
    
    agent_prompt = f"""{system_prompt}

User Question: {question}

Your response should include detailed analysis with specific examples and evidence.

Instructions:
1. Provide comprehensive analysis using your expertise (minimum 3-4 detailed paragraphs)
2. Include specific examples, case studies, and evidence
3. End with actionable recommendations and next steps
4. Ensure your response is thorough and adds substantial value to the analysis

IMPORTANT: This is a professional analysis that requires depth and detail. Provide at least 300-500 words of substantive content.
"""
    
    print(f"\nSystem prompt length: {len(system_prompt)} characters")
    print(f"Full prompt length: {len(agent_prompt)} characters")
    print(f"System prompt preview: {system_prompt[:200]}...")
    
    try:
        # Test the LLM call
        llm = OllamaLLM(llm_config, base_url="http://localhost:11434")
        
        response_text = ""
        chunk_count = 0
        start_time = asyncio.get_event_loop().time()
        
        print(f"\n🚀 Starting LLM streaming...")
        
        async for response_chunk in llm.generate_stream(agent_prompt):
            chunk_count += 1
            chunk_text = response_chunk.text
            response_text += chunk_text
            
            # Log first few chunks
            if chunk_count <= 5:
                print(f"Chunk {chunk_count}: '{chunk_text}' (length: {len(chunk_text)})")
            
            # Early termination check
            if chunk_count > 10 and len(response_text) < 100:
                print(f"⚠️  Potential early termination at chunk {chunk_count} with {len(response_text)} chars")
        
        end_time = asyncio.get_event_loop().time()
        generation_time = end_time - start_time
        
        # Analysis
        print(f"\n📊 RESULTS:")
        print(f"Total chunks: {chunk_count}")
        print(f"Response length: {len(response_text)} characters")
        print(f"Word count: {len(response_text.split())} words")
        print(f"Generation time: {generation_time:.2f} seconds")
        
        # Check for thinking tags
        has_thinking = '<think>' in response_text.lower()
        print(f"Contains thinking tags: {has_thinking}")
        
        # Check if response is complete
        is_complete = len(response_text) > 300 and len(response_text.split()) > 50
        print(f"Appears complete: {'✅ YES' if is_complete else '❌ NO'}")
        
        # Show preview
        print(f"\nResponse preview:")
        print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
        
        return is_complete
        
    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        print(f"Exception type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

async def test_agent_selection():
    """Test if the agent selection is working correctly"""
    
    print(f"\n{'='*80}")
    print("TESTING AGENT SELECTION")
    print("="*80)
    
    from app.core.intelligent_agent_selector import IntelligentAgentSelector
    from app.core.mcp_tools_cache import get_enabled_mcp_tools
    
    question = "AI automation workflow challenge traditional automation workflow like UIpath and Ansible. Are these new Ai automation workflow like n8n and dify could replace them?"
    
    try:
        # Get available tools
        available_tools = get_enabled_mcp_tools()
        print(f"Available tools: {len(available_tools)} tools")
        
        # Test agent selection
        agent_selector = IntelligentAgentSelector()
        selected_agents, question_analysis = agent_selector.select_agents(question, available_tools)
        
        print(f"Selected agents: {selected_agents}")
        print(f"Question analysis:")
        print(f"  Complexity: {question_analysis.complexity}")
        print(f"  Domain: {question_analysis.domain}")
        print(f"  Required skills: {question_analysis.required_skills}")
        print(f"  Optimal agent count: {question_analysis.optimal_agent_count}")
        print(f"  Confidence: {question_analysis.confidence}")
        
        # Check if CTO agent is selected
        cto_selected = any('cto' in agent.lower() for agent in selected_agents)
        print(f"CTO agent selected: {'✅ YES' if cto_selected else '❌ NO'}")
        
        return len(selected_agents) > 0
        
    except Exception as e:
        print(f"❌ Agent selection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_dynamic_mode_detection():
    """Test if dynamic mode detection is working for second_llm"""
    
    print(f"\n{'='*80}")
    print("TESTING DYNAMIC MODE DETECTION FOR SECOND_LLM")
    print("="*80)
    
    from app.core.llm_settings_cache import get_second_llm_full_config
    
    # Test with and without dynamic detection
    second_llm_config = get_second_llm_full_config()
    
    print(f"Second LLM config:")
    print(f"  Model: {second_llm_config.get('model')}")
    print(f"  Mode: {second_llm_config.get('mode')}")
    print(f"  Effective mode: {second_llm_config.get('effective_mode')}")
    print(f"  Mode overridden: {second_llm_config.get('mode_overridden')}")
    print(f"  Temperature: {second_llm_config.get('temperature')}")
    print(f"  Top-p: {second_llm_config.get('top_p')}")
    print(f"  Max tokens: {second_llm_config.get('max_tokens')}")
    
    # Check if this is using the problematic model
    model = second_llm_config.get('model', '')
    is_problematic = 'qwen3:30b-a3b-instruct-2507-q4_K_M' in model
    
    print(f"Using problematic model: {'⚠️  YES' if is_problematic else '✅ NO'}")
    
    if is_problematic:
        print("🔍 This could be the issue! The multi-agent system is using the problematic model.")
        print("💡 Solution: Apply the same dynamic detection fix to second_llm that we applied to main_llm.")
    
    return True

async def main():
    """Run comprehensive multi-agent debugging"""
    
    print("Multi-Agent System Debug Suite")
    print("Investigating why agents show fallback responses instead of full analysis")
    
    tests = [
        ("Agent LLM Direct Test", test_agent_llm_direct),
        ("Agent Selection Test", test_agent_selection), 
        ("Dynamic Mode Detection", test_dynamic_mode_detection)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            results[test_name] = False
    
    # Final summary
    print(f"\n{'='*80}")
    print("MULTI-AGENT DEBUG SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 ROOT CAUSE ANALYSIS:")
    print("-" * 60)
    
    if not results.get("Agent LLM Direct Test", False):
        print("❌ ISSUE: Agent LLM calls are failing directly")
        print("🔧 FIX: Check model configuration, connectivity, or prompt issues")
    elif results.get("Agent LLM Direct Test", False):
        print("✅ Agent LLM calls work when tested directly")
        print("🔍 ISSUE: Problem is likely in the multi-agent streaming pipeline")
        print("🔧 FIX: Check exception handling, streaming logic, or response processing")
    
    print(f"\n📋 RECOMMENDED ACTIONS:")
    print("-" * 60)
    print("1. Check if second_llm is using the problematic model configuration")
    print("2. Apply dynamic mode detection to multi-agent system")
    print("3. Add more detailed error logging to multi-agent streaming")
    print("4. Test with different model configurations")

if __name__ == "__main__":
    asyncio.run(main())