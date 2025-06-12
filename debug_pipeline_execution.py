#!/usr/bin/env python3
"""
Debug Pipeline Execution Script

This script demonstrates how to execute a pipeline with debug mode enabled
to see detailed agent inputs, outputs, and tool calls.

Usage:
    python debug_pipeline_execution.py <pipeline_id> <query> [--debug]
"""

import asyncio
import sys
import requests
import json
from typing import Optional

API_BASE_URL = "http://localhost:8000/api/v1"


def execute_pipeline(pipeline_id: int, query: str, debug_mode: bool = False):
    """Execute a pipeline with optional debug mode"""
    
    url = f"{API_BASE_URL}/pipelines/{pipeline_id}/execute"
    
    payload = {
        "query": query,
        "conversation_history": [],
        "trigger_type": "manual",
        "debug_mode": debug_mode,
        "additional_params": {}
    }
    
    print(f"\n{'='*80}")
    print(f"Executing Pipeline {pipeline_id}")
    print(f"Query: {query}")
    print(f"Debug Mode: {'ENABLED' if debug_mode else 'DISABLED'}")
    print(f"{'='*80}\n")
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        if debug_mode:
            print("\n[DEBUG MODE OUTPUT]")
            print("Check your API logs for detailed agent I/O information:")
            print("  - [PIPELINE DEBUG] - Pipeline execution flow")
            print("  - [AGENT I/O] - Agent inputs and outputs")
            print("  - [TOOL DETECTION] - Tool requests from agents")
            print("  - [TOOL CALL] - Actual tool executions")
            print("  - [TOOL RESULT] - Tool execution results")
            print("\nLook for these log markers in your terminal where the API is running.\n")
        
        print("\n[EXECUTION RESULT]")
        print(json.dumps(result, indent=2))
        
        # Extract key information
        if "result" in result:
            pipeline_result = result["result"]
            if "agent_outputs" in pipeline_result:
                print(f"\n[AGENT SUMMARY]")
                print(f"Total agents executed: {pipeline_result.get('total_agents', 0)}")
                print(f"Execution time: {pipeline_result.get('execution_time', 0):.2f}s")
                
                for idx, agent_output in enumerate(pipeline_result["agent_outputs"]):
                    print(f"\nAgent {idx+1}: {agent_output.get('agent', 'Unknown')}")
                    print(f"  - Output length: {len(agent_output.get('output', ''))} chars")
                    print(f"  - Execution time: {agent_output.get('execution_time', 0):.2f}s")
                    if agent_output.get('tools_used'):
                        print(f"  - Tools used: {agent_output['tools_used']}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error executing pipeline: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        sys.exit(1)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    
    pipeline_id = int(sys.argv[1])
    query = sys.argv[2]
    debug_mode = "--debug" in sys.argv
    
    execute_pipeline(pipeline_id, query, debug_mode)


if __name__ == "__main__":
    main()