#!/usr/bin/env python3
"""
Test script to verify the workflow execution system uses the fixed prompt generation.
"""

import sys
import os
import asyncio
import json
import logging

# Add project root to path
sys.path.insert(0, '/Users/kianwoonwong/Downloads/jarvis')

from app.automation.core.agent_workflow_executor import AgentWorkflowExecutor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_int_knowledge_base_workflow():
    """Test the Int Knowledge Base workflow with the prompt fix"""
    
    logger.info("=" * 60)
    logger.info("TESTING INT KNOWLEDGE BASE WORKFLOW EXECUTION")
    logger.info("=" * 60)
    
    # Create workflow executor
    executor = AgentWorkflowExecutor()
    
    # Get the workflow configuration from database (ID 39)
    try:
        from app.core.db import SessionLocal, AutomationWorkflow
        
        db = SessionLocal()
        try:
            workflow = db.query(AutomationWorkflow).filter(AutomationWorkflow.id == 39).first()
            if not workflow:
                print("❌ Workflow ID 39 (Int Knowledge Base) not found in database")
                return False
            
            print(f"✅ Found workflow: {workflow.name}")
            print(f"   Description: {workflow.description}")
            
            # Parse the langflow config
            workflow_config = workflow.langflow_config
            
            # Find the agent node to verify tools configuration
            agent_node = None
            for node in workflow_config.get("nodes", []):
                if node.get("type") == "agentnode":
                    agent_node = node
                    break
            
            if not agent_node:
                print("❌ No agent node found in workflow")
                return False
            
            agent_data = agent_node["data"]["node"]
            print(f"   Agent: {agent_data.get('agent_name')}")
            print(f"   Tools: {agent_data.get('tools', [])}")
            
            # Verify it has rag_knowledge_search tool
            if "rag_knowledge_search" not in agent_data.get("tools", []):
                print("❌ Workflow does not have rag_knowledge_search tool configured")
                return False
            
            print("✅ Workflow configuration is correct")
            
            # Test the system prompt building specifically
            print("\n🔍 Testing System Prompt Generation:")
            print("-" * 50)
            
            # Simulate agent node structure for testing
            test_agent_node = {
                "agent_name": agent_data.get("agent_name"),
                "tools": agent_data.get("tools", []),
                "custom_prompt": agent_data.get("custom_prompt", ""),
                "query": agent_data.get("query", ""),
                "agent_config": {
                    "role": "customer service",
                    "system_prompt": "You are a helpful assistant."
                }
            }
            
            # Use the fixed _build_system_prompt method
            system_prompt = executor._build_system_prompt(test_agent_node)
            
            print(f"Generated system prompt length: {len(system_prompt)}")
            print(f"Preview (first 300 chars): {system_prompt[:300]}...")
            
            # Check if it contains correct tools
            has_rag_tool = "rag_knowledge_search" in system_prompt
            has_email_tools = "find_email" in system_prompt or "read_email" in system_prompt
            
            print(f"\n📊 Analysis:")
            print(f"   Contains rag_knowledge_search: {has_rag_tool}")
            print(f"   Contains find_email/read_email: {has_email_tools}")
            
            if has_rag_tool and not has_email_tools:
                print("   ✅ SUCCESS: System prompt shows correct tools!")
                return True
            else:
                print("   ❌ FAILURE: System prompt still has wrong tools!")
                print(f"\nFull prompt:\n{system_prompt}")
                return False
                
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error testing workflow: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Starting Int Knowledge Base Workflow Test...")
    
    success = await test_int_knowledge_base_workflow()
    
    if success:
        print("\n🎯 SUMMARY: Int Knowledge Base workflow is now using correct tools!")
        return 0
    else:
        print("\n💥 SUMMARY: Test failed - workflow still has issues.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)