#!/usr/bin/env python3
"""
Test script to verify GPT OSS thinking model fixes in multi-agent streaming
"""

import asyncio
import json
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_gpt_oss_streaming():
    """Test the fixed multi-agent streaming with GPT OSS models"""
    
    try:
        # Import the fixed streaming function
        from app.langchain.fixed_multi_agent_streaming import fixed_multi_agent_streaming
        
        # Test questions
        test_questions = [
            "What are the key benefits of microservices architecture?",
            "How can I improve API performance in a high-traffic system?",
            "Explain the differences between SQL and NoSQL databases"
        ]
        
        for question in test_questions:
            print(f"\n{'='*60}")
            print(f"Testing: {question}")
            print('='*60)
            
            response_stats = {
                'empty_chunks': 0,
                'tokens': 0,
                'thinking_detected': False,
                'agents_completed': [],
                'errors': [],
                'final_response_length': 0
            }
            
            try:
                # Call the streaming function
                stream = await fixed_multi_agent_streaming(
                    question=question,
                    conversation_id=f"test_{hash(question)}"
                )
                
                # Process the stream
                async for event_str in stream:
                    try:
                        event = json.loads(event_str.strip())
                        event_type = event.get('type')
                        
                        # Track statistics
                        if event_type == 'agent_token':
                            response_stats['tokens'] += 1
                            token = event.get('token', '')
                            if not token.strip():
                                response_stats['empty_chunks'] += 1
                        
                        elif event_type == 'agent_thinking_start':
                            response_stats['thinking_detected'] = True
                            agent = event.get('agent')
                            print(f"  💭 {agent} thinking...")
                        
                        elif event_type == 'agent_complete':
                            agent = event.get('agent')
                            content = event.get('content', '')
                            response_stats['agents_completed'].append(agent)
                            print(f"  ✅ {agent} completed: {len(content)} chars")
                        
                        elif event_type == 'error':
                            error = event.get('error')
                            response_stats['errors'].append(error)
                            print(f"  ❌ Error: {error}")
                        
                        elif event_type == 'final_response':
                            response = event.get('response', '')
                            response_stats['final_response_length'] = len(response)
                            print(f"\n  📊 Final Response: {len(response)} characters")
                            print(f"  Preview: {response[:200]}...")
                        
                        elif event_type == 'agent_selection':
                            agents = event.get('selected_agents', [])
                            print(f"  🎯 Selected agents: {', '.join(agents)}")
                    
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse event: {e}")
                        logger.error(f"Raw event: {event_str}")
                
                # Print statistics
                print(f"\n  📈 Statistics:")
                print(f"     - Tokens streamed: {response_stats['tokens']}")
                print(f"     - Empty chunks: {response_stats['empty_chunks']}")
                print(f"     - Thinking detected: {response_stats['thinking_detected']}")
                print(f"     - Agents completed: {len(response_stats['agents_completed'])}")
                print(f"     - Errors: {len(response_stats['errors'])}")
                print(f"     - Final response: {response_stats['final_response_length']} chars")
                
                # Validate results
                if response_stats['final_response_length'] == 0:
                    print("  ⚠️ WARNING: No final response generated!")
                elif response_stats['final_response_length'] < 100:
                    print("  ⚠️ WARNING: Response seems truncated!")
                else:
                    print("  ✅ Response generated successfully!")
                
            except Exception as e:
                logger.error(f"Test failed for question: {e}", exc_info=True)
                print(f"  ❌ Test failed: {e}")
    
    except ImportError as e:
        logger.error(f"Failed to import fixed_multi_agent_streaming: {e}")
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the project root directory")

async def test_thinking_extraction():
    """Test the thinking tag extraction logic"""
    
    print(f"\n{'='*60}")
    print("Testing Thinking Tag Extraction")
    print('='*60)
    
    import re
    
    test_cases = [
        {
            'name': 'Full thinking wrap',
            'content': '<think>This is my reasoning process...</think>',
            'expected_thinking': 'This is my reasoning process...',
            'expected_after': ''
        },
        {
            'name': 'Thinking with content after',
            'content': '<think>My reasoning...</think>\nHere is the actual response.',
            'expected_thinking': 'My reasoning...',
            'expected_after': 'Here is the actual response.'
        },
        {
            'name': 'Multiple thinking blocks',
            'content': '<think>First thought</think>\nSome content\n<think>Second thought</think>',
            'expected_thinking': ['First thought', 'Second thought'],
            'expected_after': ''
        },
        {
            'name': 'Empty thinking tags',
            'content': '<think></think>The actual response',
            'expected_thinking': '',
            'expected_after': 'The actual response'
        }
    ]
    
    for test in test_cases:
        print(f"\n  Test: {test['name']}")
        content = test['content']
        
        # Extract thinking content
        think_pattern = r'<think>(.*?)</think>'
        think_matches = re.findall(think_pattern, content, re.DOTALL | re.IGNORECASE)
        
        # Extract content after thinking
        after_think_pattern = r'</think>\s*(.+)'
        after_think_match = re.search(after_think_pattern, content, re.DOTALL | re.IGNORECASE)
        
        after_content = after_think_match.group(1).strip() if after_think_match else ''
        
        print(f"    Input: {repr(content[:50])}...")
        print(f"    Thinking found: {think_matches}")
        print(f"    After content: {repr(after_content[:50]) if after_content else 'None'}")
        
        # Validate
        if isinstance(test['expected_thinking'], list):
            if think_matches == test['expected_thinking']:
                print(f"    ✅ Thinking extraction correct")
            else:
                print(f"    ❌ Expected: {test['expected_thinking']}")
        else:
            single_thinking = think_matches[0] if think_matches else ''
            if single_thinking == test['expected_thinking']:
                print(f"    ✅ Thinking extraction correct")
            else:
                print(f"    ❌ Expected: {repr(test['expected_thinking'])}")

async def main():
    """Run all tests"""
    print("🚀 Starting GPT OSS Fix Tests")
    print("="*60)
    
    # Test thinking extraction logic
    await test_thinking_extraction()
    
    # Test actual streaming (requires running system)
    print("\n\n🔧 Testing Multi-Agent Streaming with GPT OSS Models")
    print("="*60)
    await test_gpt_oss_streaming()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())