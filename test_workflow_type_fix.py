#!/usr/bin/env python3
"""
Test script to verify workflow 39 type detection fix.
This simulates the frontend loading workflow 39 and checks if the workflow type would be correctly set.
"""
import json
import requests

def test_workflow_type_detection():
    # Fetch workflow 39 from backend
    response = requests.get('http://127.0.0.1:8000/api/v1/automation/workflows/39')
    
    if response.status_code != 200:
        print(f"❌ Failed to fetch workflow 39: {response.status_code}")
        return
    
    workflow = response.json()
    langflow_config = workflow.get('langflow_config', {})
    
    print("🔍 Testing workflow type detection logic...")
    print(f"📋 Workflow Name: {workflow.get('name', 'Unknown')}")
    
    # Test the same logic as we added to the frontend
    detected_type = None
    detection_method = None
    
    # Method 1: Check top-level workflow_type
    if langflow_config.get('workflow_type'):
        detected_type = langflow_config['workflow_type']
        detection_method = "top-level workflow_type"
    
    # Method 2: Check metadata.workflow_type
    elif langflow_config.get('metadata', {}).get('workflow_type'):
        detected_type = langflow_config['metadata']['workflow_type']
        detection_method = "metadata.workflow_type"
    
    # Method 3: Auto-detect based on nodes
    else:
        nodes = langflow_config.get('nodes', [])
        has_agent_nodes = any(
            node.get('type') == 'agentnode' or 
            node.get('data', {}).get('type') == 'AgentNode'
            for node in nodes
        )
        detected_type = 'agent_based' if has_agent_nodes else 'legacy'
        detection_method = "auto-detection based on nodes"
    
    print(f"✅ Detected workflow type: {detected_type}")
    print(f"🔍 Detection method: {detection_method}")
    
    # Show which endpoint would be used
    expected_endpoint = f"/api/v1/automation/workflows/39/execute/stream" if detected_type == 'agent_based' else f"/api/v1/automation/workflows/39/execute"
    print(f"🎯 Expected endpoint: {expected_endpoint}")
    
    # Check if workflow has agent nodes to validate detection
    nodes = langflow_config.get('nodes', [])
    agent_nodes = [node for node in nodes if node.get('type') == 'agentnode' or node.get('data', {}).get('type') == 'AgentNode']
    
    print(f"🤖 Agent nodes found: {len(agent_nodes)}")
    for node in agent_nodes:
        print(f"   - {node.get('id', 'Unknown ID')}: {node.get('data', {}).get('node', {}).get('agent_name', 'No agent name')}")
    
    # Final validation
    if detected_type == 'agent_based' and len(agent_nodes) > 0:
        print("✅ SUCCESS: Workflow correctly detected as agent_based and should use streaming endpoint")
        return True
    elif detected_type == 'legacy' and len(agent_nodes) == 0:
        print("✅ SUCCESS: Workflow correctly detected as legacy and should use non-streaming endpoint") 
        return True
    else:
        print("❌ ISSUE: Workflow type detection may be incorrect")
        return False

if __name__ == "__main__":
    success = test_workflow_type_detection()
    if success:
        print("\n🎉 The fix should work! Workflow 39 will now call the correct endpoint.")
    else:
        print("\n⚠️  The fix may need additional work.")