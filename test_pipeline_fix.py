#!/usr/bin/env python3
"""
Test script to verify pipeline fixes work correctly.
"""

import asyncio
import sys
import requests
import json
from typing import Optional

API_BASE_URL = "http://localhost:8000/api/v1"

async def test_pipeline_fix(pipeline_id: int = 5):
    """Test the fixed pipeline execution"""
    
    url = f"{API_BASE_URL}/pipelines/{pipeline_id}/execute"
    
    payload = {
        "query": "Find emails with subject 'ENQUIRY: AGENTIC AI 2' and send a professional response",
        "conversation_history": [],
        "trigger_type": "manual",
        "debug_mode": True,  # Enable debug mode to see detailed logs
        "additional_params": {}
    }
    
    print(f"\n{'='*80}")
    print(f"TESTING PIPELINE FIX - Pipeline {pipeline_id}")
    print(f"Query: {payload['query']}")
    print(f"Debug Mode: ENABLED")
    print(f"{'='*80}\n")
    
    try:
        response = requests.post(url, json=payload, timeout=300)  # 5 minute timeout
        response.raise_for_status()
        
        result = response.json()
        
        print("\n[TEST RESULTS]")
        print(json.dumps(result, indent=2))
        
        # Check if we have proper output
        if "result" in result:
            pipeline_result = result["result"]
            
            final_output = pipeline_result.get("final_output", "")
            agent_outputs = pipeline_result.get("agent_outputs", [])
            
            print(f"\n[OUTPUT ANALYSIS]")
            print(f"Final output length: {len(final_output)} chars")
            print(f"Number of agent outputs: {len(agent_outputs)}")
            
            # Test SUCCESS criteria
            success = True
            issues = []
            
            if len(final_output) == 0:
                success = False
                issues.append("❌ Final output is empty")
            else:
                print(f"✅ Final output has content ({len(final_output)} chars)")
            
            if len(agent_outputs) == 0:
                success = False
                issues.append("❌ No agent outputs")
            else:
                print(f"✅ Found {len(agent_outputs)} agent outputs")
            
            # Check each agent output
            for i, agent_output in enumerate(agent_outputs):
                agent_name = agent_output.get("agent", f"Agent {i+1}")
                output_content = agent_output.get("output") or agent_output.get("content", "")
                
                if len(output_content) == 0:
                    success = False
                    issues.append(f"❌ Agent '{agent_name}' has empty output")
                else:
                    print(f"✅ Agent '{agent_name}' has output ({len(output_content)} chars)")
            
            print(f"\n[TEST SUMMARY]")
            if success:
                print("🎉 TEST PASSED - Pipeline is working correctly!")
                print(f"Final output preview: {final_output[:200]}...")
            else:
                print("❌ TEST FAILED - Issues found:")
                for issue in issues:
                    print(f"  {issue}")
        else:
            print("❌ TEST FAILED - No result in response")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ TEST FAILED - API Error: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"❌ TEST FAILED - Unexpected error: {e}")

def main():
    if len(sys.argv) > 1:
        pipeline_id = int(sys.argv[1])
    else:
        pipeline_id = 5  # Default to customer service pipeline
    
    asyncio.run(test_pipeline_fix(pipeline_id))

if __name__ == "__main__":
    main()