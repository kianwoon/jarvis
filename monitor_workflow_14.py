#!/usr/bin/env python3
"""
Monitor workflow 14 for data loss issues
"""
import requests
import time
import json
from datetime import datetime

def check_workflow():
    """Check workflow 14 status"""
    try:
        response = requests.get("http://localhost:8000/api/v1/automation/workflows/14")
        if response.status_code == 200:
            workflow = response.json()
            langflow_config = workflow.get('langflow_config', {})
            nodes = langflow_config.get('nodes', []) if isinstance(langflow_config, dict) else []
            edges = langflow_config.get('edges', []) if isinstance(langflow_config, dict) else []
            
            return {
                'name': workflow.get('name'),
                'nodes': len(nodes),
                'edges': len(edges),
                'updated_at': workflow.get('updated_at'),
                'status': 'ok' if nodes else 'no_nodes'
            }
        else:
            return {'status': f'error_{response.status_code}'}
    except Exception as e:
        return {'status': f'error: {e}'}

print("📊 Monitoring Workflow 14 - Press Ctrl+C to stop")
print("=" * 60)

last_status = None
while True:
    try:
        current_status = check_workflow()
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Only print if status changed
        if current_status != last_status:
            if current_status['status'] == 'ok':
                print(f"[{timestamp}] ✅ {current_status['name']} - Nodes: {current_status['nodes']}, Edges: {current_status['edges']}")
            elif current_status['status'] == 'no_nodes':
                print(f"[{timestamp}] ⚠️  WARNING: {current_status['name']} - NO NODES FOUND!")
            else:
                print(f"[{timestamp}] ❌ {current_status['status']}")
            
            if last_status and last_status.get('nodes', 0) > 0 and current_status.get('nodes', 0) == 0:
                print(f"[{timestamp}] 🚨 DATA LOSS DETECTED! Nodes went from {last_status['nodes']} to 0")
        
        last_status = current_status
        time.sleep(2)  # Check every 2 seconds
        
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")
        break