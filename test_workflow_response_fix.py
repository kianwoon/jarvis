#!/usr/bin/env python3
"""
Test script to verify the workflow response fix - ensuring comprehensive responses instead of raw tool JSON
"""

import asyncio
import json
import time
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_workflow_response_fix():
    """Test that workflow generates comprehensive responses instead of raw tool JSON"""
    
    print("\n" + "="*80)
    print("TESTING WORKFLOW RESPONSE FIX")
    print("="*80)
    
    # Test query that uses get_datetime tool
    test_query = "What is the current date and time?"
    
    print(f"\n📝 Test Query: {test_query}")
    print("-"*80)
    
    # Initialize workflow executor
    from app.automation.core.agent_workflow_executor import AgentWorkflowExecutor
    executor = AgentWorkflowExecutor()
    
    # Create workflow config with datetime tool
    workflow_config = {
        "workflow_id": 15,
        "execution_pattern": "sequential",
        "nodes": [
            {
                "id": "agent_1",
                "type": "agentnode",
                "data": {
                    "type": "AgentNode",
                    "node": {
                        "agent_name": "Research Analyst",
                        "name": "Research Analyst",
                        "role": "Senior Research Analyst",
                        "description": "Expert at analyzing documents and extracting insights",
                        "system_prompt": "You are a senior research analyst. Provide detailed, comprehensive analysis with specific metrics and data points.",
                        "tools": ["get_datetime"],
                        "config": {
                            "temperature": 0.7,
                            "max_tokens": 2000,
                            "timeout": 30,
                            "model": "qwen3:30b-a3b"
                        }
                    }
                }
            }
        ],
        "edges": [],
        "cache_nodes": [],
        "transform_nodes": [],
        "parallel_nodes": [],
        "condition_nodes": [],
        "state_nodes": [],
        "api_nodes": []
    }
    
    print(f"\n🔍 Workflow Config Debug:")
    print(f"   Node type: {workflow_config['nodes'][0]['type']}")
    print(f"   Data type: {workflow_config['nodes'][0]['data']['type']}")
    print(f"   Agent name: {workflow_config['nodes'][0]['data']['node']['agent_name']}")
    
    print("\n🚀 Executing workflow with datetime tool...")
    print("-"*40)
    
    # Execute workflow and capture response
    workflow_response = ""
    tool_calls_found = 0
    
    async for update in executor.execute_agent_workflow(
        workflow_id=15,
        execution_id=f"test_response_fix_{int(time.time())}",
        workflow_config=workflow_config,
        message=test_query
    ):
        update_type = update.get("type", "")
        
        if update_type == "agent_complete":
            content = update.get("content", "")
            workflow_response = content
            
            print(f"✅ Agent completed")
            print(f"   Response length: {len(content)} chars")
            print(f"   Response preview: {content[:200]}...")
            print(f"   Update keys: {list(update.keys())}")
            
        elif update_type == "workflow_complete":
            print(f"✅ Workflow completed!")
    
    # Analyze the response
    print("\n" + "="*80)
    print("📊 RESPONSE ANALYSIS")
    print("="*80)
    
    # Check if response contains raw JSON tool calls
    import re
    tool_json_pattern = r'\{"tool"\s*:\s*"[^"]+"\s*,\s*"parameters"\s*:\s*\{.*?\}\}'
    raw_json_found = re.search(tool_json_pattern, workflow_response)
    
    print(f"\n1️⃣ Raw Tool JSON Check:")
    if raw_json_found:
        print("   ❌ FAILED: Response contains raw tool JSON")
        print(f"   Found: {raw_json_found.group()}")
    else:
        print("   ✅ SUCCESS: No raw tool JSON found")
    
    print(f"\n2️⃣ Response Content Check:")
    content_checks = {
        "Contains current time info": "current" in workflow_response.lower() and ("time" in workflow_response.lower() or "date" in workflow_response.lower()),
        "Contains structured formatting": "**" in workflow_response or "•" in workflow_response,
        "Contains datetime details": any(term in workflow_response.lower() for term in ["wednesday", "tuesday", "monday", "thursday", "friday", "saturday", "sunday", "2025", "2024"]),
        "Has comprehensive content": len(workflow_response) > 200,
        "No tool call remnants": not any(phrase in workflow_response for phrase in ['"tool":', '"parameters":', '{"tool"'])
    }
    
    all_passed = True
    for check, passed in content_checks.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {check}")
        if not passed:
            all_passed = False
    
    # Show the actual response
    print(f"\n3️⃣ Full Response:")
    print("-"*40)
    print(workflow_response)
    print("-"*40)
    
    # Overall assessment
    print(f"\n4️⃣ Overall Assessment:")
    if all_passed and not raw_json_found:
        print("✅ FIX SUCCESSFUL: Workflow now generates comprehensive responses!")
        return True
    else:
        print("❌ FIX INCOMPLETE: Issues still present")
        return False

if __name__ == "__main__":
    print("🔧 Testing Workflow Response Fix...")
    print(f"📅 Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        success = asyncio.run(test_workflow_response_fix())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)