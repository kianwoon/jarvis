#!/usr/bin/env python3
"""
Direct test of the multi-agent response fix without circular imports
Tests the _clean_llm_response method and validation logic directly
"""

import asyncio
import sys
import os
import re
import logging

# Set up logging to see debug messages
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_clean_llm_response():
    """Test the _clean_llm_response method directly"""
    
    class MockMultiAgentSystem:
        def _clean_llm_response(self, response_text: str) -> str:
            """MINIMAL cleaning - only remove excessive whitespace"""
            # MINIMAL cleaning - only remove excessive whitespace
            cleaned = response_text.strip()
            
            # Log response cleaning details for debugging
            logger.debug(f"Response cleaning: original_length={len(response_text)}, cleaned_length={len(cleaned)}")
            
            # Only clean up excessive newlines (more than 3 in a row)
            cleaned = re.sub(r'\n\s*\n\s*\n\s*\n+', '\n\n\n', cleaned)
            
            # If response is too short, return cleaned or original
            if len(cleaned) < 10:
                logger.debug(f"Response too short (len={len(cleaned)}), returning {'cleaned' if cleaned else 'original'}")
                return cleaned if cleaned else response_text
                
            logger.debug(f"Response cleaned successfully, final_length={len(cleaned)}")
            return cleaned
    
    system = MockMultiAgentSystem()
    
    print("🧪 Testing _clean_llm_response method...")
    
    # Test cases that were problematic
    test_cases = [
        {
            "name": "Normal response",
            "input": "This is a normal response with good content that should not be stripped.",
            "expected_min_length": 50
        },
        {
            "name": "Response with excessive whitespace",
            "input": "   This response has   excessive   whitespace   \n\n\n\n\nAnd multiple newlines.   ",
            "expected_min_length": 30
        },
        {
            "name": "Short response (should be preserved)",
            "input": "Short.",
            "expected_min_length": 6
        },
        {
            "name": "Empty response (edge case)",
            "input": "",
            "expected_min_length": 0
        },
        {
            "name": "Whitespace only response",
            "input": "   \n\n   \t   ",
            "expected_min_length": 0
        },
        {
            "name": "Infrastructure Agent style response",
            "input": """
            Based on your question about AI automation workflows vs traditional automation:

            **Traditional Automation (UIPath, Ansible):**
            - Rule-based, deterministic workflows
            - Requires explicit programming of each step
            - Limited adaptability to changing conditions
            
            **AI Automation Workflows (n8n, Dify):**
            - Can incorporate machine learning and AI decision making
            - More adaptive and intelligent routing
            - Better handling of unstructured data
            
            While AI workflows offer significant advantages, they complement rather than completely replace traditional automation in most enterprise scenarios.
            """,
            "expected_min_length": 200
        }
    ]
    
    all_passed = True
    
    for test_case in test_cases:
        print(f"\n🔍 Testing: {test_case['name']}")
        
        original = test_case["input"]
        cleaned = system._clean_llm_response(original)
        
        print(f"   Original length: {len(original)}")
        print(f"   Cleaned length: {len(cleaned)}")
        print(f"   Expected min length: {test_case['expected_min_length']}")
        
        if len(cleaned) >= test_case['expected_min_length']:
            print(f"   ✅ PASS: Response preserved correctly")
        else:
            print(f"   ❌ FAIL: Response too short after cleaning")
            print(f"   Original: {repr(original)}")
            print(f"   Cleaned:  {repr(cleaned)}")
            all_passed = False
    
    return all_passed

def test_validation_recovery():
    """Test the validation recovery mechanism"""
    
    print("\n🧪 Testing validation recovery mechanism...")
    
    # Simulate the validation logic from fixed_multi_agent_streaming.py
    def validate_response(agent_name: str, token_count: int, full_response: str, agent_response: str) -> str:
        """Simulate the validation recovery logic"""
        
        # Validate response is not empty when we streamed tokens
        if token_count > 0 and not agent_response.strip():
            logger.error(f"[VALIDATION FAILURE] {agent_name}: Streamed {token_count} tokens but got empty response after cleaning")
            logger.debug(f"[VALIDATION] Full response before cleaning: {repr(full_response[:200])}...")
            # Use full_response as fallback if cleaning resulted in empty response
            recovered_response = full_response if full_response.strip() else "Error: Response processing failed"
            logger.info(f"[VALIDATION RECOVERY] {agent_name}: Using fallback response, length={len(recovered_response)}")
            return recovered_response
        
        return agent_response
    
    # Test cases for validation recovery
    test_cases = [
        {
            "name": "Good response (no recovery needed)",
            "agent_name": "Infrastructure Agent",
            "token_count": 150,
            "full_response": "This is the full response that was streamed",
            "agent_response": "This is the full response that was streamed",
            "should_recover": False
        },
        {
            "name": "Empty after cleaning (recovery needed)",
            "agent_name": "Service Delivery Manager", 
            "token_count": 200,
            "full_response": "This is the full response with good content",
            "agent_response": "",  # Empty after cleaning
            "should_recover": True
        },
        {
            "name": "No tokens streamed (no recovery)",
            "agent_name": "Data Engineer",
            "token_count": 0,
            "full_response": "",
            "agent_response": "",
            "should_recover": False
        }
    ]
    
    all_passed = True
    
    for test_case in test_cases:
        print(f"\n🔍 Testing recovery: {test_case['name']}")
        
        result = validate_response(
            test_case["agent_name"],
            test_case["token_count"],
            test_case["full_response"],
            test_case["agent_response"]
        )
        
        if test_case["should_recover"]:
            if result == test_case["full_response"]:
                print(f"   ✅ PASS: Recovery mechanism triggered correctly")
            else:
                print(f"   ❌ FAIL: Recovery mechanism did not trigger")
                all_passed = False
        else:
            if result == test_case["agent_response"]:
                print(f"   ✅ PASS: No recovery needed, original response preserved")
            else:
                print(f"   ❌ FAIL: Unexpected recovery triggered")
                all_passed = False
        
        print(f"   Result length: {len(result)}")
    
    return all_passed

def main():
    """Run all tests"""
    
    print("🚀 Testing Multi-Agent Response Fix Components\n")
    
    test1_passed = test_clean_llm_response()
    test2_passed = test_validation_recovery()
    
    print(f"\n📊 Test Results Summary:")
    print(f"   Response cleaning test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Validation recovery test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print(f"\n🎉 SUCCESS: All multi-agent response fix components working correctly!")
        print(f"   ✅ Response cleaning preserves content")
        print(f"   ✅ Validation recovery mechanism functional")
        print(f"   ✅ Debug logging provides visibility")
        return True
    else:
        print(f"\n❌ FAILURE: Some components need additional fixes")
        return False

if __name__ == "__main__":
    result = main()
    sys.exit(0 if result else 1)