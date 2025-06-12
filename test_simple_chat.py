#!/usr/bin/env python3
"""
Test script for simplified chat system
"""

import asyncio
import json
import httpx
from typing import AsyncGenerator

async def test_simple_chat():
    """Test the simplified chat endpoint"""
    
    # Test questions
    test_questions = [
        "What time is it?",  # Should trigger tool call
        "What is Python?",   # Should use LLM knowledge
        "Tell me about our company policies",  # Should trigger RAG search
    ]
    
    base_url = "http://localhost:8000/api/v1"
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        
        # First, check system status
        print("=== Checking System Status ===")
        try:
            response = await client.get(f"{base_url}/chat/simple-chat/status")
            if response.status_code == 200:
                status = response.json()
                print(f"✅ Status: {status['status']}")
                print(f"📊 Available Tools: {status['available_tools']}")
                print(f"📚 Available Collections: {status['available_collections']}")
                print(f"🤖 Current Model: {status['current_model']}")
                print(f"🔧 Tools: {', '.join(status.get('tools', [])[:5])}")
                print(f"📖 Collections: {', '.join(status.get('collections', [])[:3])}")
            else:
                print(f"❌ Status check failed: {response.status_code}")
                return
        except Exception as e:
            print(f"❌ Failed to check status: {e}")
            return
        
        print("\n" + "="*60 + "\n")
        
        # Test each question
        for i, question in enumerate(test_questions, 1):
            print(f"=== Test {i}: {question} ===")
            
            try:
                # Send request to simple chat endpoint
                request_data = {
                    "question": question,
                    "conversation_id": f"test_conv_{i}",
                    "thinking": False
                }
                
                async with client.stream(
                    "POST",
                    f"{base_url}/chat/simple-chat",
                    json=request_data
                ) as response:
                    
                    if response.status_code != 200:
                        print(f"❌ Request failed: {response.status_code}")
                        print(await response.aread())
                        continue
                    
                    print("📥 Streaming response:")
                    
                    async for chunk in response.aiter_text():
                        if chunk.strip():
                            for line in chunk.strip().split('\n'):
                                if line.strip():
                                    try:
                                        event = json.loads(line)
                                        event_type = event.get('type')
                                        
                                        if event_type == 'chat_start':
                                            print(f"🚀 Started chat with model: {event.get('model')}")
                                        
                                        elif event_type == 'message':
                                            print(f"💬 LLM: {event.get('content')}")
                                        
                                        elif event_type == 'tools_start':
                                            print(f"🔧 Executing {event.get('tool_count')} tool(s)...")
                                        
                                        elif event_type == 'tool_result':
                                            tool_result = event.get('tool_result', {})
                                            if tool_result.get('success'):
                                                print(f"✅ Tool {tool_result.get('tool')}: {str(tool_result.get('result', {}))[:100]}...")
                                            else:
                                                print(f"❌ Tool {tool_result.get('tool')}: {tool_result.get('error')}")
                                        
                                        elif event_type == 'rag_start':
                                            print(f"📚 Searching {event.get('search_count')} collection(s)...")
                                        
                                        elif event_type == 'rag_result':
                                            rag_result = event.get('rag_result', {})
                                            if rag_result.get('success'):
                                                print(f"📖 Found {rag_result.get('document_count', 0)} documents")
                                            else:
                                                print(f"❌ RAG search failed: {rag_result.get('error')}")
                                        
                                        elif event_type == 'synthesis_start':
                                            print("🧠 Synthesizing final response...")
                                        
                                        elif event_type == 'final_response':
                                            print(f"🎯 Final: {event.get('content')}")
                                        
                                        elif event_type == 'chat_complete':
                                            tools = event.get('tool_count', 0)
                                            rag = event.get('rag_count', 0)
                                            print(f"✅ Complete! Used {tools} tools, {rag} RAG searches")
                                        
                                        elif event_type == 'error':
                                            print(f"❌ Error: {event.get('error')}")
                                        
                                    except json.JSONDecodeError:
                                        continue
                
                print(f"\n{'='*60}\n")
                
            except Exception as e:
                print(f"❌ Test failed: {e}")
                print(f"{'='*60}\n")

if __name__ == "__main__":
    print("🧪 Testing Simplified Chat System")
    print("="*60)
    asyncio.run(test_simple_chat())