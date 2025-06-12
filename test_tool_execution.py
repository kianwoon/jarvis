#!/usr/bin/env python3
"""Test that tools are actually executed in the simplified system"""

import asyncio
import json
import httpx

async def test_tool_execution():
    """Test that LLM actually executes tools"""
    
    test_query = "What is the current date and time?"
    
    print(f"Testing query: {test_query}")
    print("="*60)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "http://localhost:8000/api/v1/langchain/rag",
            json={
                "question": test_query,
                "thinking": False,
                "skip_classification": True  # Use simplified system
            },
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            tool_executed = False
            tool_results = []
            final_answer = ""
            
            for line in response.text.strip().split('\n'):
                if line:
                    try:
                        data = json.loads(line)
                        
                        # Check for tool execution
                        if "tool_execution" in data:
                            tool_executed = True
                            tool_info = data["tool_execution"]
                            print(f"\n✅ Tool Executed: {tool_info['tool']}")
                            print(f"   Success: {tool_info['success']}")
                            print(f"   Result: {json.dumps(tool_info['result'], indent=2)}")
                            tool_results.append(tool_info)
                        
                        # Get final answer
                        if "answer" in data:
                            final_answer = data["answer"]
                            source = data.get("source", "")
                            print(f"\nSource: {source}")
                            
                    except json.JSONDecodeError:
                        pass
            
            print("\n" + "="*60)
            print(f"Tool Execution Status: {'✅ SUCCESS' if tool_executed else '❌ FAILED'}")
            print(f"Tools Executed: {len(tool_results)}")
            
            if tool_executed:
                print("\nTool Results Summary:")
                for tr in tool_results:
                    print(f"  - {tr['tool']}: {'Success' if tr['success'] else 'Failed'}")
            
            print("\nFinal Answer Preview:")
            print(final_answer[:500] + "..." if len(final_answer) > 500 else final_answer)
            
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

if __name__ == "__main__":
    asyncio.run(test_tool_execution())