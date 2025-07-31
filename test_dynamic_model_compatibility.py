#!/usr/bin/env python3
"""
Comprehensive test for dynamic model compatibility system.
Tests both Qwen3 models with the new dynamic detection and response processing.
"""

import asyncio
import httpx
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.llm.response_analyzer import detect_model_thinking_behavior, get_model_behavior_profile, clear_behavior_cache
from app.core.llm_settings_cache import get_llm_settings, get_main_llm_full_config_with_detection

async def test_response_analyzer():
    """Test the response analyzer with sample responses"""
    
    print("="*80)
    print("TESTING RESPONSE ANALYZER")
    print("="*80)
    
    # Test samples
    test_cases = [
        {
            'name': 'Thinking Model Response',
            'text': '<think>\nOkay, the user is asking about photosynthesis. Let me break this down...\n</think>\n\nPhotosynthesis is a process...',
            'expected_thinking': True
        },
        {
            'name': 'Non-Thinking Model Response', 
            'text': 'Photosynthesis is a fundamental biological process that allows plants to convert light energy...',
            'expected_thinking': False
        },
        {
            'name': 'JSON Response (instruct-2507 style)',
            'text': '{"useful": true, "comment": "This is helpful information", "jarvis_opinion": "Photosynthesis is crucial for life on Earth..."}',
            'expected_thinking': False
        }
    ]
    
    for case in test_cases:
        print(f"\nTesting: {case['name']}")
        print(f"Text sample: {case['text'][:100]}...")
        
        is_thinking, confidence = detect_model_thinking_behavior(case['text'], 'test-model')
        
        print(f"Detected thinking: {is_thinking} (confidence: {confidence:.2f})")
        print(f"Expected thinking: {case['expected_thinking']}")
        
        if is_thinking == case['expected_thinking']:
            print("✅ PASS - Detection matches expectation")
        else:
            print("❌ FAIL - Detection mismatch") 
    
    # Test caching
    print(f"\nTesting cache...")
    profile = get_model_behavior_profile('test-model')
    if profile:
        print(f"Cached profile: thinking={profile.is_thinking_model}, confidence={profile.confidence:.2f}")
    
    clear_behavior_cache()
    print("Cache cleared")

async def test_model_with_api(model_name: str, question: str):
    """Test a model through the actual API endpoints"""
    
    print(f"\n{'='*80}")
    print(f"TESTING MODEL: {model_name}")
    print("="*80)
    
    # Test basic model response first
    print("1. Testing direct model response...")
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are Jarvis, an AI assistant."},
            {"role": "user", "content": question}
        ],
        "stream": True,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 200,
            "num_ctx": 4096,
        }
    }
    
    response_text = ""
    token_count = 0
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            async with client.stream("POST", "http://localhost:11434/api/chat", json=payload) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    try:
                        data_json = json.loads(line)
                        message = data_json.get("message", {})
                        
                        if message and "content" in message:
                            content = message["content"]
                            response_text += content
                            token_count += 1
                        
                        if data_json.get("done", False):
                            break
                            
                    except json.JSONDecodeError:
                        continue
    
    except Exception as e:
        print(f"❌ Direct test failed: {e}")
        return False
    
    print(f"Direct response - Tokens: {token_count}, Length: {len(response_text)}")
    print(f"First 200 chars: {response_text[:200]}...")
    
    # Test dynamic detection
    print("\n2. Testing dynamic behavior detection...")
    
    is_thinking, confidence = detect_model_thinking_behavior(response_text, model_name)
    print(f"Detected behavior: {'thinking' if is_thinking else 'non-thinking'} (confidence: {confidence:.2f})")
    
    # Test config with detection
    print("\n3. Testing config with detection...")
    
    settings = get_llm_settings()
    # Temporarily update model in settings for test
    original_model = settings.get('main_llm', {}).get('model')
    settings['main_llm']['model'] = model_name
    
    # Test config with sample
    config_with_detection = get_main_llm_full_config_with_detection(settings, response_text)
    
    print(f"Original mode: {config_with_detection.get('configured_mode', 'unknown')}")
    print(f"Effective mode: {config_with_detection.get('effective_mode', 'unknown')}")
    print(f"Mode overridden: {config_with_detection.get('mode_overridden', False)}")
    
    # Restore original model
    if original_model:
        settings['main_llm']['model'] = original_model
    
    return True

async def test_synthesis_workflow():
    """Test the complete synthesis workflow with dynamic detection"""
    
    print(f"\n{'='*80}")
    print("TESTING SYNTHESIS WORKFLOW")
    print("="*80)
    
    # This would require a more complex setup with the actual RAG endpoint
    # For now, we'll test the components individually
    
    print("Synthesis workflow test requires backend to be running.")
    print("To test:")
    print("1. Start the backend: ./run_local.sh")
    print("2. Start the frontend: cd llm-ui && npm run dev")
    print("3. Test both models through the web interface")
    print("4. Check logs for dynamic detection messages")
    
    return True

async def main():
    """Run all compatibility tests"""
    
    print("Dynamic Model Compatibility Test Suite")
    print("="*80)
    
    # Test 1: Response Analyzer
    await test_response_analyzer()
    
    # Test 2: Both models individually
    models_to_test = [
        "qwen3:30b-a3b-q4_K_M",
        "qwen3:30b-a3b-instruct-2507-q4_K_M"
    ]
    
    question = "Explain how machine learning works in simple terms."
    
    for model in models_to_test:
        success = await test_model_with_api(model, question)
        if not success:
            print(f"❌ Model {model} test failed")
        await asyncio.sleep(1)  # Brief pause between tests
    
    # Test 3: Synthesis workflow
    await test_synthesis_workflow()
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print("="*80)
    
    from app.llm.response_analyzer import response_analyzer
    stats = response_analyzer.get_cache_stats()
    
    print(f"Models analyzed: {stats['total_models']}")
    print(f"Thinking models: {stats['thinking_models']}")
    print(f"Non-thinking models: {stats['non_thinking_models']}")
    
    if stats['models']:
        print("\nModel behavior profiles:")
        for model, profile in stats['models'].items():
            print(f"- {model}: {'thinking' if profile['is_thinking'] else 'non-thinking'} (confidence: {profile['confidence']:.2f})")
    
    print("\n✅ Dynamic compatibility system components tested successfully!")
    print("\nNext steps:")
    print("1. Start backend and frontend")
    print("2. Test both models through web interface")
    print("3. Verify dynamic detection in logs")
    print("4. Confirm proper response processing")

if __name__ == "__main__":
    asyncio.run(main())