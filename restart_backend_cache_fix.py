#!/usr/bin/env python3
"""
Restart the backend to apply cache fixes for workflow saving issue
"""
import subprocess
import time
import os

print("🔄 Restarting backend to apply cache fixes...")

# Kill existing backend process
print("Stopping backend...")
subprocess.run(["pkill", "-f", "uvicorn.*8000"], capture_output=True)
time.sleep(2)

# Start backend
print("Starting backend...")
os.chdir("/Users/kianwoonwong/Downloads/jarvis")
subprocess.Popen([
    "uvicorn", "app.main:app", 
    "--host", "0.0.0.0", 
    "--port", "8000", 
    "--reload"
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("✅ Backend restarted! Waiting for it to be ready...")
time.sleep(5)

# Test if backend is up
import requests
try:
    response = requests.get("http://localhost:8000/api/v1/automation/workflows")
    if response.status_code == 200:
        print("✅ Backend is ready!")
        workflows = response.json()
        print(f"Found {len(workflows)} workflows")
        
        # Find workflow 14
        for workflow in workflows:
            if workflow.get('id') == 14:
                print(f"\n📋 Workflow 14: {workflow.get('name')}")
                langflow_config = workflow.get('langflow_config', {})
                if isinstance(langflow_config, dict):
                    nodes = langflow_config.get('nodes', [])
                    print(f"   Nodes: {len(nodes)}")
                    edges = langflow_config.get('edges', [])
                    print(f"   Edges: {len(edges)}")
                break
    else:
        print(f"❌ Backend returned status {response.status_code}")
except Exception as e:
    print(f"❌ Backend not ready: {e}")

print("\n🔧 Cache fixes applied:")
print("1. Added validation for langflow_config nodes")
print("2. Added direct database query fallback if cache is invalid")
print("3. Added delay after cache invalidation to prevent race conditions")
print("4. Added preloading after cache invalidation")
print("\n⚠️  Please test saving workflow 14 again!")