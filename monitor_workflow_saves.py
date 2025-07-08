#!/usr/bin/env python3
"""
Monitor workflow saves to detect when nodes get wiped
"""
import time
import requests
import json
from datetime import datetime

def get_workflow(workflow_id):
    """Get workflow data"""
    try:
        response = requests.get(f'http://127.0.0.1:8000/api/v1/automation/workflows/{workflow_id}')
        if response.ok:
            return response.json()
    except Exception as e:
        print(f"Error fetching workflow: {e}")
    return None

def monitor_workflow(workflow_id, check_interval=5):
    """Monitor a workflow for changes"""
    print(f"Starting monitoring of workflow {workflow_id}")
    print(f"Checking every {check_interval} seconds...")
    print("-" * 60)
    
    last_state = None
    last_node_count = None
    
    while True:
        try:
            workflow = get_workflow(workflow_id)
            if workflow:
                config = workflow.get('langflow_config', {})
                nodes = config.get('nodes', [])
                edges = config.get('edges', [])
                node_count = len(nodes)
                edge_count = len(edges)
                
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Check for changes
                if last_node_count is not None and node_count != last_node_count:
                    print(f"\n🚨 [{timestamp}] NODE COUNT CHANGED!")
                    print(f"   Previous: {last_node_count} nodes")
                    print(f"   Current:  {node_count} nodes")
                    
                    if node_count == 0:
                        print("   ⚠️  CRITICAL: Workflow has NO NODES!")
                        print("   This will cause execution failures!")
                        
                        # Log the full config for debugging
                        print("\n   Full langflow_config:")
                        print(json.dumps(config, indent=4))
                    else:
                        print(f"   Node types: {[n.get('type') for n in nodes]}")
                    print("-" * 60)
                elif last_node_count is None:
                    # First check
                    print(f"[{timestamp}] Initial state: {node_count} nodes, {edge_count} edges")
                    if node_count > 0:
                        print(f"   Node types: {[n.get('type') for n in nodes]}")
                
                last_node_count = node_count
                last_state = workflow
                
            else:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Failed to fetch workflow")
            
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"Error during monitoring: {e}")
            time.sleep(check_interval)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        workflow_id = int(sys.argv[1])
    else:
        workflow_id = int(input("Enter workflow ID to monitor: "))
    
    monitor_workflow(workflow_id)