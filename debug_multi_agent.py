#!/usr/bin/env python
"""Debug multi-agent event stream"""
import asyncio
import httpx
import json

async def debug_multi_agent():
    url = "http://localhost:8000/api/v1/langchain/multi-agent"
    data = {
        "question": "What is the capital of France?",
        "conversation_id": "debug-simple"
    }
    
    print("Sending request...")
    async with httpx.AsyncClient(timeout=300.0) as client:
        async with client.stream('POST', url, json=data) as response:
            print(f"Response status: {response.status_code}")
            
            event_count = 0
            async for line in response.aiter_lines():
                if line.strip():
                    event_count += 1
                    try:
                        event = json.loads(line)
                        event_type = event.get("type", "unknown")
                        agent = event.get("agent", "")
                        
                        if event_type == "agent_start":
                            print(f"\n[{event_count}] ✓ Agent Starting: {agent}")
                        elif event_type == "agent_complete":
                            content = event.get("content", "")[:100] + "..." if len(event.get("content", "")) > 100 else event.get("content", "")
                            print(f"[{event_count}] ✓ Agent Complete: {agent}")
                            print(f"     Response: {content}")
                        elif event_type == "final_response":
                            print(f"\n[{event_count}] ✓ Final Response Generated")
                            response_preview = event.get("response", "")[:200] + "..." if len(event.get("response", "")) > 200 else event.get("response", "")
                            print(f"     Preview: {response_preview}")
                        elif event_type == "error":
                            print(f"\n[{event_count}] ✗ Error: {event.get('error', 'Unknown error')}")
                        else:
                            print(f"[{event_count}] Event: {event_type} - {agent}")
                    except json.JSONDecodeError as e:
                        print(f"[{event_count}] JSON Error: {e}")
                        print(f"     Line: {line[:100]}...")
            
            print(f"\nTotal events received: {event_count}")

if __name__ == "__main__":
    asyncio.run(debug_multi_agent())