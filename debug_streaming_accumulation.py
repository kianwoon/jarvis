#!/usr/bin/env python3
"""
Debug script to understand why Service Delivery Manager streams 80 tokens
but full_response ends up empty before cleaning
"""

import asyncio
import json
import logging
import sys
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def debug_streaming_accumulation():
    """Debug the streaming token accumulation issue"""
    
    print("🔍 Debugging Service Delivery Manager streaming accumulation...")
    
    # Simulate the exact streaming logic from fixed_multi_agent_streaming.py lines 562-635
    def simulate_streaming_tokens():
        """Simulate tokens being streamed like Ollama would"""
        # Based on the log evidence, we streamed 80 tokens but full_response was empty
        # This suggests a problem in the accumulation logic
        
        # Let's simulate different scenarios
        scenarios = [
            {
                "name": "Normal case (like Infrastructure Agent)",
                "tokens": [
                    "Based ", "on ", "your ", "infrastructure ", "question", ", ",
                    "I ", "can ", "provide ", "comprehensive ", "analysis", "...\n\n",
                    "**Key ", "Infrastructure ", "Considerations:**\n",
                    "- ", "Scalability ", "requirements\n",
                    "- ", "Security ", "implications\n" 
                ],
                "expected_success": True
            },
            {
                "name": "Thinking model case (potential issue)",
                "tokens": [
                    "<think>", "Let ", "me ", "analyze ", "this ", "service ", "delivery ", 
                    "question ", "from ", "multiple ", "angles", "...</think>",
                    "\n\nFor ", "service ", "delivery ", "optimization", ", ",
                    "I ", "recommend ", "focusing ", "on", ":\n\n",
                    "1. ", "Process ", "automation\n",
                    "2. ", "Quality ", "metrics\n"
                ],
                "expected_success": True
            },
            {
                "name": "Empty content tokens (potential bug)",
                "tokens": ["", "", "", "", "", ""] * 15,  # 90 empty tokens
                "expected_success": False
            },
            {
                "name": "Mixed empty and content (edge case)",
                "tokens": [
                    "", "", "Based ", "", "on ", "", "", "service ", "", "delivery",
                    "", "", "best ", "", "practices", "", "", "I ", "", "suggest"
                ] + [""] * 60,  # Mix of content and empty
                "expected_success": False  # Should have some content but might be lost
            }
        ]
        
        return scenarios
    
    scenarios = simulate_streaming_tokens()
    
    for scenario in scenarios:
        print(f"\n🧪 Testing scenario: {scenario['name']}")
        print(f"   Total tokens: {len(scenario['tokens'])}")
        
        # Simulate the exact accumulation logic from the code
        full_response = ""
        thinking_content = ""
        in_thinking = False
        thinking_detected = False
        token_count = 0
        streamed_tokens = []
        
        # Process each token like the real streaming loop
        for chunk_text in scenario['tokens']:
            full_response += chunk_text
            token_count += 1
            streamed_tokens.append(chunk_text)
            
            # Log first few tokens like the real code
            if token_count <= 5:
                logger.debug(f"Token {token_count}: '{chunk_text[:50]}...'")
            
            # Thinking detection logic (same as real code)
            if "<think>" in chunk_text.lower():
                in_thinking = True
                thinking_detected = True
                start_idx = chunk_text.lower().find("<think>") + 7
                thinking_content += chunk_text[start_idx:]
                
            elif "</think>" in chunk_text.lower():
                in_thinking = False
                end_idx = chunk_text.lower().find("</think>")
                thinking_content += chunk_text[:end_idx]
                thinking_content = ""  # Reset like real code
                
            elif in_thinking:
                thinking_content += chunk_text
        
        # Simulate the _clean_llm_response logic
        def clean_llm_response(response_text: str) -> str:
            """Minimal cleaning like the real method"""
            if not response_text:
                return ""
            cleaned = response_text.strip()
            
            # Only clean up excessive newlines (more than 3 in a row)
            import re
            cleaned = re.sub(r'\n\s*\n\s*\n\s*\n+', '\n\n\n', cleaned)
            
            if len(cleaned) < 10:
                return cleaned if cleaned else response_text
                
            return cleaned
        
        agent_response = clean_llm_response(full_response)
        
        # Check for the validation failure condition
        validation_failed = token_count > 0 and not agent_response.strip()
        
        print(f"   Tokens streamed: {token_count}")
        print(f"   full_response length: {len(full_response)}")
        print(f"   full_response content: {repr(full_response[:100])}...")
        print(f"   agent_response length: {len(agent_response)}")
        print(f"   agent_response content: {repr(agent_response[:100])}...")
        print(f"   Thinking detected: {thinking_detected}")
        print(f"   Validation failed: {validation_failed}")
        
        # Check if this matches the Service Delivery Manager failure pattern
        if token_count > 0 and len(full_response) == 0:
            print(f"   🚨 CRITICAL ISSUE: Streamed {token_count} tokens but full_response is empty!")
            print(f"   This matches the Service Delivery Manager failure pattern!")
            
            # Debug: Show what tokens were processed
            print(f"   Token details:")
            for i, token in enumerate(streamed_tokens[:10]):
                print(f"     Token {i+1}: {repr(token)}")
        elif validation_failed:
            print(f"   ⚠️ VALIDATION ISSUE: Response empty after cleaning")
        elif scenario['expected_success'] and len(agent_response) > 10:
            print(f"   ✅ SUCCESS: Response accumulated and cleaned properly")
        elif not scenario['expected_success'] and validation_failed:
            print(f"   ✅ EXPECTED: Known problematic scenario behaved as expected")
        else:
            print(f"   ❓ UNEXPECTED: Scenario behaved differently than expected")

async def debug_real_llm_streaming():
    """Test with actual Ollama LLM to see if there are connection issues"""
    
    print("\n🔧 Testing real LLM streaming for Service Delivery Manager...")
    
    try:
        from app.llm.ollama import OllamaLLM
        from app.llm.base import LLMConfig
        from app.core.llm_settings_cache import get_second_llm_full_config
        
        # Get the exact configuration that Service Delivery Manager would use
        second_llm_config = get_second_llm_full_config()
        model_name = second_llm_config.get('model', 'qwen3:30b-a3b')
        
        print(f"   Testing with model: {model_name}")
        
        llm_config = LLMConfig(
            model_name=model_name,
            temperature=0.7,
            max_tokens=200,  # Short test
            top_p=0.9
        )
        
        # Create a simple test prompt
        test_prompt = """You are a Service Delivery Manager expert.

User Question: What are the key metrics for measuring service delivery effectiveness?

Instructions:
1. Provide comprehensive analysis using your expertise
2. Include specific examples and evidence  
3. End with actionable recommendations

This is a test prompt to debug streaming issues."""

        print(f"   Prompt length: {len(test_prompt)} characters")
        
        # Test streaming
        llm = OllamaLLM(llm_config, base_url="http://localhost:11434")
        
        full_response = ""
        token_count = 0
        tokens_received = []
        
        print(f"   Starting streaming test...")
        
        try:
            async for response_chunk in llm.generate_stream(test_prompt):
                chunk_text = response_chunk.text
                full_response += chunk_text
                token_count += 1
                tokens_received.append(chunk_text)
                
                if token_count <= 5:
                    print(f"     Token {token_count}: {repr(chunk_text)}")
                
                if token_count >= 100:  # Limit for testing
                    print(f"     Stopping at 100 tokens for testing...")
                    break
                    
        except Exception as e:
            print(f"   ❌ Streaming failed: {e}")
            return False
            
        print(f"   Streaming completed:")
        print(f"     Tokens received: {token_count}")
        print(f"     full_response length: {len(full_response)}")
        print(f"     full_response sample: {repr(full_response[:200])}...")
        
        # Check for the Service Delivery Manager issue
        if token_count > 0 and len(full_response) == 0:
            print(f"   🚨 REPRODUCED THE BUG: Streamed tokens but empty response!")
            return False
        elif token_count > 0 and len(full_response) > 0:
            print(f"   ✅ Streaming working correctly")
            return True
        else:
            print(f"   ❓ No tokens received - possible connection issue")
            return False
            
    except Exception as e:
        print(f"   ❌ Real LLM test failed: {e}")
        return False

async def main():
    """Run all debugging tests"""
    
    print("🚀 Service Delivery Manager Streaming Debug Analysis\n")
    
    print("=" * 60)
    await debug_streaming_accumulation()
    
    print("\n" + "=" * 60)
    streaming_works = await debug_real_llm_streaming()
    
    print(f"\n📊 Debug Analysis Results:")
    print(f"   Streaming accumulation logic: Analyzed multiple scenarios")
    print(f"   Real LLM streaming test: {'✅ PASSED' if streaming_works else '❌ FAILED'}")
    
    print(f"\n🎯 Key Findings:")
    print(f"   • The issue likely occurs when:")
    print(f"     1. Ollama streaming returns empty content chunks")
    print(f"     2. Network issues cause incomplete streaming")
    print(f"     3. Model-specific response format issues")
    print(f"     4. Timeout or connection drops during streaming")
    
    print(f"\n🔍 Next Steps:")
    print(f"   • Check Ollama service status and connectivity")
    print(f"   • Add more robust error handling in streaming loop")
    print(f"   • Add validation that chunk_text is not empty before accumulating")
    print(f"   • Consider adding retry logic for failed streaming")

if __name__ == "__main__":
    asyncio.run(main())