#!/usr/bin/env python3
"""
Test script to verify multi-agent sequential execution and output passing
"""
import asyncio
import json
from datetime import datetime

# Add the app directory to Python path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.langchain.multi_agent_system_simple import MultiAgentSystem
from app.core.redis_client import get_redis_client

async def test_sequential_execution():
    """Test if agents properly pass outputs in sequential mode"""
    
    print("=" * 80)
    print("Multi-Agent Sequential Execution Test")
    print("=" * 80)
    
    # Initialize the multi-agent system
    multi_agent_system = MultiAgentSystem(
        conversation_id=f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Test query that should trigger sequential execution
    test_query = "First analyze the Python logging module documentation, then create a comprehensive tutorial based on that analysis"
    
    print(f"\nTest Query: {test_query}")
    print("\nExecuting multi-agent system...")
    print("-" * 80)
    
    # Track agent outputs
    agent_outputs = {}
    agent_order = []
    communication_events = []
    
    # Execute the multi-agent system
    event_count = 0
    async for event in multi_agent_system.stream_events(test_query):
        event_count += 1
        
        if isinstance(event, dict):
            event_type = event.get("type")
            
            if event_type == "collaboration_pattern":
                pattern = event.get("pattern", "unknown")
                agents = event.get("agents", [])
                print(f"\n[COLLABORATION] Pattern: {pattern}")
                print(f"[COLLABORATION] Agents: {', '.join(agents)}")
                
            elif event_type == "agent_start":
                agent_name = event.get("agent", "unknown")
                agent_order.append(agent_name)
                print(f"\n[AGENT START] {agent_name}")
                
            elif event_type == "agent_token":
                # Skip token events for cleaner output
                pass
                
            elif event_type == "agent_complete":
                agent_name = event.get("agent", "unknown")
                content = event.get("content", "")
                agent_outputs[agent_name] = content
                print(f"\n[AGENT COMPLETE] {agent_name}")
                print(f"[OUTPUT LENGTH] {len(content)} characters")
                print(f"[OUTPUT PREVIEW] {content[:200]}..." if len(content) > 200 else f"[OUTPUT] {content}")
                
            elif event_type == "agent_communication":
                from_agent = event.get("from_agent", "unknown")
                to_agent = event.get("to_agent", "unknown")
                message = event.get("message", "")
                communication_events.append({
                    "from": from_agent,
                    "to": to_agent,
                    "message": message
                })
                print(f"\n[COMMUNICATION] {from_agent} -> {to_agent}")
                print(f"[MESSAGE] {message[:100]}...")
                
            elif event_type == "final_response":
                print(f"\n[FINAL RESPONSE] Synthesis complete")
    
    print("\n" + "=" * 80)
    print("Test Results")
    print("=" * 80)
    
    # Analyze results
    print(f"\n1. Total Events: {event_count}")
    print(f"\n2. Agent Execution Order:")
    for i, agent in enumerate(agent_order, 1):
        print(f"   {i}. {agent}")
    
    print(f"\n3. Agent Outputs Collected: {len(agent_outputs)}")
    for agent, output in agent_outputs.items():
        print(f"   - {agent}: {len(output)} characters")
    
    print(f"\n4. Inter-Agent Communications: {len(communication_events)}")
    for comm in communication_events:
        print(f"   - {comm['from']} -> {comm['to']}: {comm['message'][:50]}...")
    
    # Verify sequential execution
    print("\n5. Sequential Execution Verification:")
    if len(agent_order) > 1:
        print("   ✓ Multiple agents executed")
        if len(communication_events) > 0:
            print("   ✓ Agent communication events detected")
        else:
            print("   ✗ No agent communication events detected")
    else:
        print("   ✗ Only one agent executed")
    
    # Check if outputs were passed
    print("\n6. Output Passing Verification:")
    if len(agent_order) >= 2 and len(agent_outputs) >= 2:
        first_agent = agent_order[0]
        second_agent = agent_order[1]
        
        # Check if second agent references first agent's output
        if first_agent in agent_outputs and second_agent in agent_outputs:
            second_output = agent_outputs[second_agent].lower()
            # Look for references to analysis, previous agent, or based on
            references = ["based on", "analysis", "previous", "above", "from the"]
            has_reference = any(ref in second_output for ref in references)
            
            if has_reference:
                print("   ✓ Second agent appears to reference first agent's work")
            else:
                print("   ? Cannot confirm if second agent used first agent's output")
        else:
            print("   ✗ Missing agent outputs")
    else:
        print("   ✗ Not enough agents executed for sequential verification")
    
    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_sequential_execution())