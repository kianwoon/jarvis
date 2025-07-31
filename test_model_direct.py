#!/usr/bin/env python3
"""
Test the qwen3:30b-a3b-instruct-2507-q4_K_M model directly to diagnose the short response issue.
"""

import asyncio
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_model_direct():
    """Test the problematic model with various configurations"""
    
    print("=" * 80)
    print("TESTING qwen3:30b-a3b-instruct-2507-q4_K_M DIRECTLY")
    print("=" * 80)
    
    from app.llm.ollama import OllamaLLM
    from app.llm.base import LLMConfig
    from app.core.llm_settings_cache import get_llm_settings
    
    # Get current settings
    settings = get_llm_settings()
    main_llm = settings.get('main_llm', {})
    
    test_prompt = """You are Jarvis, an AI assistant. Please provide a detailed response explaining what Anthropic is and their recent funding developments. Be comprehensive and informative.

User question: Tell me about Anthropic and their recent Series E funding round.

Please provide a thorough response with multiple paragraphs explaining:
1. What Anthropic is
2. Their recent funding details  
3. What they plan to do with the funding
4. The significance in the AI industry"""

    # Test different configurations
    test_configs = [
        {
            "name": "Current Settings (Thinking Mode)",
            "config": {
                "model_name": "qwen3:30b-a3b-instruct-2507-q4_K_M",
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": 196608
            }
        },
        {
            "name": "Non-Thinking Mode Settings", 
            "config": {
                "model_name": "qwen3:30b-a3b-instruct-2507-q4_K_M",
                "temperature": 0.7,
                "top_p": 0.8,
                "max_tokens": 196608
            }
        },
        {
            "name": "Higher Temperature",
            "config": {
                "model_name": "qwen3:30b-a3b-instruct-2507-q4_K_M", 
                "temperature": 0.9,
                "top_p": 0.9,
                "max_tokens": 196608
            }
        },
        {
            "name": "Lower Max Tokens",
            "config": {
                "model_name": "qwen3:30b-a3b-instruct-2507-q4_K_M",
                "temperature": 0.7,
                "top_p": 0.9, 
                "max_tokens": 4000
            }
        }
    ]
    
    results = {}
    
    for test in test_configs:
        print(f"\n🔬 Testing: {test['name']}")
        print("-" * 60)
        
        try:
            config = LLMConfig(**test['config'])
            llm = OllamaLLM(config, base_url="http://localhost:11434")
            
            print(f"Config: temp={config.temperature}, top_p={config.top_p}, max_tokens={config.max_tokens}")
            
            # Generate response
            response_text = ""
            start_time = asyncio.get_event_loop().time()
            
            async for chunk in llm.generate_stream(test_prompt):
                response_text += chunk.text
            
            end_time = asyncio.get_event_loop().time()
            generation_time = end_time - start_time
            
            # Analyze response
            response_length = len(response_text)
            word_count = len(response_text.split())
            line_count = len(response_text.split('\n'))
            
            print(f"✅ SUCCESS")
            print(f"Response length: {response_length} characters")
            print(f"Word count: {word_count} words")
            print(f"Line count: {line_count} lines")
            print(f"Generation time: {generation_time:.2f} seconds")
            
            # Check for thinking tags
            has_thinking_tags = '<think>' in response_text.lower()
            print(f"Contains thinking tags: {has_thinking_tags}")
            
            # Show first 200 characters
            print(f"Preview: {response_text[:200]}...")
            
            # Check if response seems complete
            is_complete = response_length > 500 and word_count > 50
            print(f"Appears complete: {'✅ YES' if is_complete else '❌ NO'}")
            
            results[test['name']] = {
                'success': True,
                'length': response_length,
                'words': word_count,
                'complete': is_complete,
                'thinking_tags': has_thinking_tags,
                'generation_time': generation_time
            }
            
        except Exception as e:
            print(f"❌ FAILED: {e}")
            results[test['name']] = {
                'success': False,
                'error': str(e)
            }
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")  
    print("="*80)
    
    for test_name, result in results.items():
        if result['success']:
            status = "✅ PASS" if result['complete'] else "⚠️  SHORT"
            print(f"{status} {test_name}: {result['length']} chars, {result['words']} words")
        else:
            print(f"❌ FAIL {test_name}: {result['error']}")
    
    # Recommendations
    print(f"\n🎯 ANALYSIS")
    print("-" * 60)
    
    successful_tests = [name for name, result in results.items() if result.get('success') and result.get('complete')]
    short_tests = [name for name, result in results.items() if result.get('success') and not result.get('complete')]
    failed_tests = [name for name, result in results.items() if not result.get('success')]
    
    if successful_tests:
        print(f"✅ Configurations that work: {len(successful_tests)}")
        for name in successful_tests:
            print(f"   • {name}")
    
    if short_tests:
        print(f"⚠️  Configurations with short responses: {len(short_tests)}")
        for name in short_tests:
            print(f"   • {name}")
    
    if failed_tests:
        print(f"❌ Failed configurations: {len(failed_tests)}")
        for name in failed_tests:
            print(f"   • {name}")
    
    return results

async def test_working_model_comparison():
    """Compare with the working model for reference"""
    
    print(f"\n{'='*80}")
    print("COMPARING WITH WORKING MODEL")
    print("="*80)
    
    from app.llm.ollama import OllamaLLM
    from app.llm.base import LLMConfig
    
    test_prompt = "Explain what Anthropic is and their recent Series E funding. Be detailed and comprehensive."
    
    working_model_config = LLMConfig(
        model_name="qwen3:30b-a3b-q4_K_M",  # Working model
        temperature=0.6,
        top_p=0.95,
        max_tokens=4000
    )
    
    try:
        llm = OllamaLLM(working_model_config, base_url="http://localhost:11434")
        
        response_text = ""
        async for chunk in llm.generate_stream(test_prompt):
            response_text += chunk.text
        
        print(f"Working model (qwen3:30b-a3b-q4_K_M):")
        print(f"Response length: {len(response_text)} characters")
        print(f"Word count: {len(response_text.split())} words")
        print(f"Preview: {response_text[:200]}...")
        
        return len(response_text)
        
    except Exception as e:
        print(f"❌ Working model test failed: {e}")
        return 0

async def main():
    """Run comprehensive model testing"""
    
    print("Model Direct Testing Suite")
    print("Investigating qwen3:30b-a3b-instruct-2507-q4_K_M short response issue")
    
    # Test the problematic model
    problematic_results = await test_model_direct()
    
    # Test the working model for comparison
    working_length = await test_working_model_comparison()
    
    # Final analysis
    print(f"\n🔍 ROOT CAUSE ANALYSIS")
    print("-" * 60)
    
    successful_configs = [name for name, result in problematic_results.items() 
                         if result.get('success') and result.get('complete')]
    
    if successful_configs:
        print("✅ The model CAN generate long responses with the right configuration!")
        print("🎯 Root cause: Configuration parameters, not the model itself")
        print(f"🔧 Working configurations: {', '.join(successful_configs)}")
    else:
        print("❌ The model consistently generates short responses")
        print("🎯 Root cause: Model-specific behavior issue")
        print("🔧 Recommendation: Continue using fallback system or switch models")
    
    if working_length > 0:
        print(f"\n📊 Comparison:")
        print(f"Working model response: {working_length} characters")
        avg_problematic = sum(r.get('length', 0) for r in problematic_results.values() if r.get('success')) / max(1, len([r for r in problematic_results.values() if r.get('success')]))
        print(f"Problematic model average: {avg_problematic:.0f} characters")

if __name__ == "__main__":
    asyncio.run(main())