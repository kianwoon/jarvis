#!/usr/bin/env python3
"""Test script to verify tool planner includes system prompt"""

import asyncio
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.langchain.intelligent_tool_planner import IntelligentToolPlanner

async def test_tool_planning():
    planner = IntelligentToolPlanner()
    
    # Test task that should trigger google_search
    test_task = "compare openai and anthropic, which is more advanced?"
    
    try:
        # Get available tools
        tools = planner.get_enhanced_tool_metadata()
        print(f"Available tools: {len(tools)}")
        
        # Create execution plan
        plan = await planner.plan_tool_execution(test_task)
        
        print(f"\nTask: {test_task}")
        print(f"Plan reasoning: {plan.reasoning}")
        print(f"\nPlanned tools:")
        for tool_plan in plan.tools:
            print(f"  - {tool_plan.tool_name}: {tool_plan.purpose}")
            print(f"    Parameters: {tool_plan.parameters}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_tool_planning())