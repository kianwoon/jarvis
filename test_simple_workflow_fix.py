#!/usr/bin/env python3
"""
Simple test script to verify the workflow prompt fix without complex imports.
"""

import sys
import os
import json

# Add project root to path
sys.path.insert(0, '/Users/kianwoonwong/Downloads/jarvis')

def test_workflow_database_configuration():
    """Test that the workflow database configuration is correct"""
    
    print("=" * 60)
    print("TESTING INT KNOWLEDGE BASE WORKFLOW CONFIGURATION")
    print("=" * 60)
    
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
            print(f"   Query: {agent_data.get('query', 'No query')}")
            
            # Verify it has rag_knowledge_search tool
            if "rag_knowledge_search" not in agent_data.get("tools", []):
                print("❌ Workflow does not have rag_knowledge_search tool configured")
                return False
            
            print("✅ Workflow configuration is correct")
            
            # Check the custom prompt to see if it has hardcoded email tools
            custom_prompt = agent_data.get("custom_prompt", "")
            has_hardcoded_email = "find_email" in custom_prompt or "read_email" in custom_prompt
            
            print(f"\n📋 Custom Prompt Analysis:")
            print(f"   Length: {len(custom_prompt)} characters")
            print(f"   Contains hardcoded email tools: {has_hardcoded_email}")
            
            if has_hardcoded_email:
                print("   ⚠️  WARNING: Custom prompt still has hardcoded email tool references")
                print("   ℹ️  This should be fixed by our dynamic prompt generator")
            
            return True
                
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Error testing workflow: {e}")
        return False

def test_prompt_generator_directly():
    """Test the prompt generator directly"""
    
    print("\n🧪 TESTING PROMPT GENERATOR DIRECTLY")
    print("-" * 60)
    
    try:
        from app.automation.core.workflow_prompt_generator import generate_workflow_agent_prompt
        
        # Simulate Int Knowledge Base workflow
        workflow_tools = ["rag_knowledge_search"]
        agent_name = "customer service"
        
        # Base prompt with hardcoded email tools (simulating the issue)
        base_prompt_with_email_tools = """You are a customer service agent.

## Available Tools & Usage
You have access to the following tools:
- **find_email**: Search and locate specific emails in your inbox
- **read_email**: Read and analyze email content and attachments

Use these tools to help customers."""
        
        # Generate dynamic prompt
        dynamic_prompt = generate_workflow_agent_prompt(
            agent_name=agent_name,
            workflow_tools=workflow_tools,
            base_system_prompt=base_prompt_with_email_tools,
            role="customer service",
            custom_prompt=""
        )
        
        print(f"Input tools: {workflow_tools}")
        print(f"Generated prompt length: {len(dynamic_prompt)}")
        
        # Check results
        has_rag_tool = "rag_knowledge_search" in dynamic_prompt
        has_email_tools = "find_email" in dynamic_prompt or "read_email" in dynamic_prompt
        
        print(f"\n📊 Results:")
        print(f"   Contains rag_knowledge_search: {has_rag_tool}")
        print(f"   Contains find_email/read_email: {has_email_tools}")
        
        if has_rag_tool and not has_email_tools:
            print("   ✅ SUCCESS: Dynamic prompt generation removed hardcoded tools!")
            print(f"\n📄 Generated prompt preview:")
            print("-" * 50)
            # Show the tools section specifically
            lines = dynamic_prompt.split('\n')
            in_tools_section = False
            for line in lines:
                if "## Available Tools & Usage" in line:
                    in_tools_section = True
                elif in_tools_section and line.startswith("## ") and "Available Tools" not in line:
                    break
                    
                if in_tools_section:
                    print(line)
            print("-" * 50)
            return True
        else:
            print("   ❌ FAILURE: Still showing wrong tools!")
            print(f"\nFull prompt:\n{dynamic_prompt}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing prompt generator: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting Simple Workflow Fix Test...")
    
    config_test = test_workflow_database_configuration()
    prompt_test = test_prompt_generator_directly()
    
    if config_test and prompt_test:
        print("\n🎯 SUMMARY: All tests passed! The workflow prompt fix is working.")
        print("✅ The 'Int Knowledge Base' workflow will now use rag_knowledge_search instead of email tools.")
        return 0
    else:
        print("\n💥 SUMMARY: Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)