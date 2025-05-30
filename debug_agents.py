#!/usr/bin/env python
"""Debug why some agents don't respond"""
import asyncio
import httpx
import json

async def test_multi_agent():
    url = "http://localhost:8000/api/v1/langchain/multi-agent"
    data = {
        "question": "Help me evaluate migrating from MariaDB to OceanBase",
        "conversation_id": "debug-test"
    }
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(url, json=data)
        
        # Process streaming response
        agents_started = set()
        agents_completed = set()
        agents_errored = set()
        
        for line in response.iter_lines():
            if line.strip():
                try:
                    event = json.loads(line)
                    event_type = event.get("type")
                    agent = event.get("agent")
                    
                    if event_type == "agent_start":
                        agents_started.add(agent)
                        print(f"✓ Started: {agent}")
                    elif event_type == "agent_complete":
                        agents_completed.add(agent)
                        content_len = len(event.get("content", ""))
                        print(f"✓ Completed: {agent} (response length: {content_len})")
                    elif event_type == "agent_error":
                        agents_errored.add(agent)
                        error = event.get("error", "Unknown error")
                        print(f"✗ Error: {agent} - {error}")
                    elif event_type == "routing":
                        routing = event.get("routing", {})
                        selected_agents = routing.get("agents", [])
                        print(f"\nRouter selected agents: {selected_agents}")
                        print(f"Collaboration pattern: {routing.get('collaboration_pattern')}\n")
                except json.JSONDecodeError:
                    pass
        
        print("\n=== Summary ===")
        print(f"Agents started: {sorted(agents_started)}")
        print(f"Agents completed: {sorted(agents_completed)}")
        print(f"Agents with errors: {sorted(agents_errored)}")
        
        # Find agents that started but didn't complete
        missing = agents_started - agents_completed - agents_errored - {"router", "synthesizer"}
        if missing:
            print(f"\n⚠️  Agents that started but didn't complete: {sorted(missing)}")

if __name__ == "__main__":
    asyncio.run(test_multi_agent())