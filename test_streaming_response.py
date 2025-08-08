#!/usr/bin/env python3
"""
Test streaming response to verify multi-agent fix without curl
"""

import asyncio
import aiohttp
import json
import sys
import time

async def test_multi_agent_streaming():
    """Test multi-agent streaming to verify response fix"""
    
    print("🧪 Testing Multi-Agent Streaming Response Fix...")
    
    # Test data
    test_data = {
        "question": "AI automation workflow challenge traditional automation workflow like UIpath and Ansible. Are these new Ai automation workflow like n8n and dify could replace them?",
        "conversation_id": "test-response-fix-streaming"
    }
    
    agent_responses = {}
    empty_responses = []
    streaming_tokens = {}
    
    try:
        timeout = aiohttp.ClientTimeout(total=180)  # 3 minute timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                'http://localhost:8000/api/v1/langchain/multi-agent',
                json=test_data,
                headers={'Content-Type': 'application/json'}
            ) as response:
                
                if response.status != 200:
                    print(f"❌ HTTP Error: {response.status}")
                    error_text = await response.text()
                    print(f"Error details: {error_text}")
                    return False
                
                print("✅ Connected to multi-agent endpoint successfully")
                print("🔄 Processing streaming response...\n")
                
                current_agent = None
                token_count = 0
                
                async for chunk in response.content.iter_any():
                    try:
                        chunk_text = chunk.decode('utf-8').strip()
                        if not chunk_text:
                            continue
                        
                        # Parse JSON response
                        data = json.loads(chunk_text)
                        event_type = data.get("type", "unknown")
                        
                        if event_type == "agent_start":
                            current_agent = data.get("agent")
                            token_count = 0
                            streaming_tokens[current_agent] = 0
                            print(f"🤖 {current_agent} started")
                            
                        elif event_type == "agent_token":
                            if current_agent:
                                streaming_tokens[current_agent] += 1
                                token_count += 1
                                
                        elif event_type == "agent_complete":
                            agent_name = data.get("agent", current_agent)
                            content = data.get("content", "")
                            content_length = len(content.strip()) if content else 0
                            tokens_streamed = streaming_tokens.get(agent_name, 0)
                            
                            print(f"📊 {agent_name}:")
                            print(f"   Tokens streamed: {tokens_streamed}")
                            print(f"   Final response length: {content_length}")
                            
                            if tokens_streamed > 0 and content_length == 0:
                                print(f"   ❌ BUG: Streamed {tokens_streamed} tokens but got empty response!")
                                empty_responses.append({
                                    "agent": agent_name,
                                    "tokens_streamed": tokens_streamed,
                                    "response_length": content_length
                                })
                            elif tokens_streamed > 0 and content_length > 0:
                                print(f"   ✅ SUCCESS: Tokens properly preserved in response")
                            elif tokens_streamed == 0:
                                print(f"   ⚠️  WARNING: No tokens streamed")
                            
                            if content:
                                agent_responses[agent_name] = content
                                print(f"   First 100 chars: {repr(content[:100])}")
                            
                            print()
                            
                    except json.JSONDecodeError:
                        # Skip non-JSON chunks
                        continue
                    except Exception as e:
                        print(f"⚠️  Error processing chunk: {e}")
                        continue
                
    except asyncio.TimeoutError:
        print("❌ Request timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    print("📈 Test Results Summary:")
    print(f"   Total agents responded: {len(agent_responses)}")
    print(f"   Agents with streaming tokens: {len([k for k, v in streaming_tokens.items() if v > 0])}")
    print(f"   Empty response bugs found: {len(empty_responses)}")
    
    if empty_responses:
        print("\n❌ VALIDATION FAILURES:")
        for bug in empty_responses:
            print(f"   - {bug['agent']}: {bug['tokens_streamed']} tokens → {bug['response_length']} response length")
        print("\n💡 These agents need the response fix validation recovery!")
    
    # Check specific agents that were problematic
    infrastructure_fixed = (
        "Infrastructure Agent" in agent_responses and 
        len(agent_responses["Infrastructure Agent"].strip()) > 0 and
        not any(bug["agent"] == "Infrastructure Agent" for bug in empty_responses)
    )
    
    service_mgr_fixed = (
        "Service Delivery Manager" in agent_responses and 
        len(agent_responses["Service Delivery Manager"].strip()) > 0 and
        not any(bug["agent"] == "Service Delivery Manager" for bug in empty_responses)
    )
    
    print(f"\n🎯 Specific Agent Status:")
    print(f"   Infrastructure Agent: {'✅ FIXED' if infrastructure_fixed else '❌ STILL BROKEN'}")
    print(f"   Service Delivery Manager: {'✅ FIXED' if service_mgr_fixed else '❌ STILL BROKEN'}")
    
    # Overall assessment
    if len(empty_responses) == 0 and len(agent_responses) >= 2:
        print(f"\n🎉 SUCCESS: Multi-agent response fix working!")
        print(f"   ✅ No empty response bugs detected")
        print(f"   ✅ All agents with tokens produced responses")
        return True
    elif len(empty_responses) < len(streaming_tokens):
        print(f"\n⚠️  PARTIAL SUCCESS: Fix working but some issues remain")
        print(f"   ✅ {len(agent_responses)} agents working correctly")
        print(f"   ❌ {len(empty_responses)} agents still have empty response bug")
        return False
    else:
        print(f"\n❌ FAILURE: Multi-agent response fix not working")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_multi_agent_streaming())
    sys.exit(0 if result else 1)