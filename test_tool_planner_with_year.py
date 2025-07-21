#!/usr/bin/env python3
"""Test that tool planner includes system prompt with year information"""

import asyncio
import sys
import os
import json

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Capture stdout to see debug messages
import io
from contextlib import redirect_stdout

from app.langchain.intelligent_tool_planner import IntelligentToolPlanner
from app.core.llm_settings_cache import get_llm_settings, get_second_llm_full_config

async def test_tool_planning_with_year():
    """Test that tool planner properly includes year context when planning"""
    
    print("=== Testing Tool Planner with Year Context ===\n")
    
    # First verify second_llm has the year in system prompt
    settings = get_llm_settings()
    second_llm_config = get_second_llm_full_config(settings)
    print(f"Second LLM System Prompt: {second_llm_config.get('system_prompt', 'NOT FOUND')}\n")
    
    planner = IntelligentToolPlanner()
    
    # Test task that should use current year context
    test_task = "compare openai and anthropic, which is more advanced?"
    
    # Capture debug output during planning
    f = io.StringIO()
    
    try:
        # Run planning with output capture
        with redirect_stdout(f):
            plan = await planner.plan_tool_execution(test_task)
        
        captured_output = f.getvalue()
        
        # Check if system prompt was prepended
        if "[DEBUG make_llm_call] System prompt found and prepended" in captured_output:
            print("✓ System prompt was prepended during tool planning")
            
            # Extract the system prompt that was used
            for line in captured_output.split('\n'):
                if "System prompt found and prepended:" in line:
                    print(f"  {line.strip()}")
        else:
            print("✗ System prompt was NOT prepended during tool planning")
        
        # Display the planning results
        print(f"\nTask: {test_task}")
        print(f"\nPlanned tools:")
        for tool_plan in plan.tools:
            print(f"  - {tool_plan.tool_name}: {tool_plan.purpose}")
            print(f"    Parameters: {tool_plan.parameters}")
            
            # Check if google_search parameters contain year
            if tool_plan.tool_name == "google_search" and "query" in tool_plan.parameters:
                query = tool_plan.parameters["query"]
                print(f"    Query: '{query}'")
                
                # Check for problematic year concatenation
                if "2023" in query:
                    print("    ⚠️  WARNING: Query contains '2023' - system prompt may not be working!")
                elif "2025" in query:
                    print("    ✓ Query contains '2025' - system prompt is working!")
                else:
                    print("    ℹ️  Query doesn't contain explicit year")
        
        print(f"\nPlan reasoning: {plan.reasoning[:200]}...")
        
    except Exception as e:
        print(f"Error during planning: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_tool_planning_with_year())