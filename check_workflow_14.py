import requests
import json

# Force clear the workflow cache
print("Clearing workflow cache...")
response = requests.post('http://127.0.0.1:8000/api/v1/automation/cache/invalidate')
print(f'Cache invalidation response: {response.status_code}')

# Get workflow 14 to check its data
print("\nFetching workflow 14...")
response = requests.get('http://127.0.0.1:8000/api/v1/automation/workflows/14')
if response.ok:
    workflow = response.json()
    print(f'Workflow name: {workflow.get("name")}')
    config = workflow.get('langflow_config', {})
    nodes = config.get('nodes', [])
    edges = config.get('edges', [])
    print(f'Nodes count: {len(nodes)}')
    print(f'Edges count: {len(edges)}')
    if nodes:
        print('Node types:', [n.get('type') for n in nodes])
        print('\nNodes details:')
        for node in nodes:
            print(f"  - {node.get('id')}: {node.get('type')} ({node.get('data', {}).get('label', 'no label')})")
    else:
        print('ERROR: No nodes found in the workflow!')
        print('Full config:', json.dumps(config, indent=2))
else:
    print(f'Failed to fetch workflow: {response.status_code}')