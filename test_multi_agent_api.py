"""
Test multi-agent API endpoint directly
"""

import asyncio
import json
import aiohttp

async def test_multi_agent_api():
    """Test the multi-agent endpoint via HTTP"""
    url = "http://localhost:8000/api/v1/langchain/multi-agent"
    
    test_queries = [
        {
            "question": "What are the benefits of using a multi-agent system?",
            "conversation_id": "test-123"
        },
        {
            "question": "Client OCBC bank needs 3 system engineers. How to propose managed services?",
            "conversation_id": "test-456",
            "selected_agents": ["sales_strategist", "financial_analyst"]
        }
    ]
    
    async with aiohttp.ClientSession() as session:
        for query_data in test_queries:
            print(f"\n{'='*80}")
            print(f"Testing: {query_data['question']}")
            print(f"{'='*80}\n")
            
            try:
                async with session.post(url, json=query_data) as response:
                    if response.status != 200:
                        print(f"❌ Error: HTTP {response.status}")
                        continue
                    
                    # Process SSE stream
                    async for line in response.content:
                        decoded_line = line.decode('utf-8').strip()
                        if decoded_line.startswith('data: '):
                            try:
                                event = json.loads(decoded_line[6:])
                                event_type = event.get('type')
                                agent = event.get('agent', '')
                                
                                if event_type == 'agent_start':
                                    print(f"🚀 Starting: {agent}")
                                elif event_type == 'agent_streaming':
                                    # Just show that agent is streaming
                                    pass
                                elif event_type == 'agent_complete':
                                    content = event.get('content', '')
                                    print(f"✅ Completed: {agent} ({len(content)} chars)")
                                elif event_type == 'routing':
                                    routing = event.get('routing', {})
                                    print(f"🎯 Routing: {routing.get('reasoning', 'No reason')}")
                                elif event_type == 'final_response':
                                    print(f"\n📊 Final Response Preview:")
                                    print("-" * 40)
                                    response_text = event.get('response', '')
                                    # Show first 200 chars
                                    preview = response_text[:200] + "..." if len(response_text) > 200 else response_text
                                    print(preview)
                                elif event_type == 'error':
                                    print(f"❌ Error: {event.get('error')}")
                                    
                            except json.JSONDecodeError:
                                pass
                                
            except Exception as e:
                print(f"❌ Request failed: {str(e)}")

async def test_ui_format():
    """Test that the response format works with the UI"""
    url = "http://localhost:8000/api/v1/langchain/multi-agent"
    
    query_data = {
        "question": "How does managed services compare to time and material billing?",
        "conversation_id": "ui-test-789"
    }
    
    print(f"\n{'='*80}")
    print("Testing UI-compatible format")
    print(f"{'='*80}\n")
    
    events_collected = []
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=query_data) as response:
            async for line in response.content:
                decoded_line = line.decode('utf-8').strip()
                if decoded_line.startswith('data: '):
                    try:
                        event = json.loads(decoded_line[6:])
                        events_collected.append(event)
                    except:
                        pass
    
    # Verify event structure
    print("📋 Event types collected:")
    event_types = {}
    for event in events_collected:
        event_type = event.get('type', 'unknown')
        event_types[event_type] = event_types.get(event_type, 0) + 1
    
    for event_type, count in event_types.items():
        print(f"  - {event_type}: {count}")
    
    # Check for required fields
    print("\n✅ Validation:")
    has_final = any(e.get('type') == 'final_response' for e in events_collected)
    has_streaming = any(e.get('type') == 'agent_streaming' for e in events_collected)
    has_complete = any(e.get('type') == 'agent_complete' for e in events_collected)
    
    print(f"  - Has final response: {'Yes' if has_final else 'No'}")
    print(f"  - Has streaming events: {'Yes' if has_streaming else 'No'}")
    print(f"  - Has completion events: {'Yes' if has_complete else 'No'}")

if __name__ == "__main__":
    print("Testing Multi-Agent Chat API...")
    print("Make sure the FastAPI server is running on http://localhost:8000")
    print()
    
    asyncio.run(test_multi_agent_api())
    asyncio.run(test_ui_format())