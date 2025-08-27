#!/usr/bin/env python3

"""
Test the emergency bypass implementation for the failing conversation
"""

import asyncio
import requests
import json

async def test_emergency_bypass():
    """Test the emergency bypass for timeout issues."""
    
    test_conversation_id = "notebook-144c3ac3-0a16-4904-8cba-5adaa2cd86cd"
    notebook_id = "144c3ac3-0a16-4904-8cba-5adaa2cd86cd"  # Extract from conversation ID
    test_message = "order by end year, give me the list again"
    
    print(f"🧪 TESTING EMERGENCY BYPASS")
    print(f"📝 Conversation ID: {test_conversation_id}")
    print(f"📚 Notebook ID: {notebook_id}")
    print(f"💬 Message: '{test_message}'")
    print("=" * 80)
    
    # Test the chat endpoint
    chat_url = f"http://localhost:8000/api/v1/notebooks/{notebook_id}/chat"
    
    payload = {
        "message": test_message,
        "conversation_id": test_conversation_id,
        "include_context": True,
        "max_sources": 10
    }
    
    print(f"🌐 Making request to: {chat_url}")
    print(f"📤 Payload: {json.dumps(payload, indent=2)}")
    print()
    
    try:
        # Make streaming request
        response = requests.post(
            chat_url,
            json=payload,
            stream=True,
            timeout=60  # Give it a minute to see if bypass works
        )
        
        if response.status_code == 200:
            print("✅ Request successful, processing stream:")
            print("-" * 40)
            
            response_chunks = []
            for line in response.iter_lines(decode_unicode=True):
                if line.strip():
                    try:
                        chunk_data = json.loads(line)
                        response_chunks.append(chunk_data)
                        
                        # Print key information
                        if 'status' in chunk_data:
                            print(f"📊 Status: {chunk_data['status']}")
                            if 'routing_decision' in chunk_data:
                                print(f"🔀 Routing: {chunk_data['routing_decision']}")
                        
                        if 'chunk' in chunk_data:
                            print(f"📄 Response chunk: {chunk_data['chunk'][:100]}...")
                        
                        if 'response' in chunk_data:
                            print(f"✨ Full response: {chunk_data['response'][:200]}...")
                            
                    except json.JSONDecodeError as e:
                        print(f"⚠️ JSON decode error: {e}")
                        print(f"   Raw line: {line[:100]}...")
            
            print("-" * 40)
            print(f"📈 Total chunks received: {len(response_chunks)}")
            
            # Analyze the routing decision
            routing_decisions = [chunk.get('routing_decision') for chunk in response_chunks if 'routing_decision' in chunk]
            if routing_decisions:
                print(f"🎯 Routing decisions: {routing_decisions}")
                if 'emergency_bypass' in str(routing_decisions):
                    print("🚀 EMERGENCY BYPASS WAS ACTIVATED!")
                else:
                    print("⚠️ Emergency bypass was not used")
            
            # Check if we got a timeout
            timeout_chunks = [chunk for chunk in response_chunks if 'chunk' in chunk and 'timed out' in chunk.get('chunk', '')]
            if timeout_chunks:
                print("❌ STILL GETTING TIMEOUT ERRORS")
            else:
                print("✅ NO TIMEOUT ERRORS DETECTED")
        
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"   Error: {response.text}")
    
    except requests.exceptions.Timeout:
        print("❌ REQUEST TIMED OUT - Emergency bypass did not prevent timeout")
    except Exception as e:
        print(f"💥 Request failed: {e}")
    
    print("\n" + "=" * 80)
    print("🎯 TEST RESULTS ANALYSIS:")
    print("If emergency bypass worked, you should see:")
    print("- Routing decision containing 'emergency_bypass'")
    print("- Fast response without timeout")
    print("- Direct formatted list of results")

if __name__ == "__main__":
    asyncio.run(test_emergency_bypass())