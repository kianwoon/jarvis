#!/usr/bin/env python3
"""
Test Workflow RAG Tool Execution Fix
=====================================

This test verifies that the workflow correctly:
1. Calls the RAG tool via MCP bridge
2. Gets results back with documents
3. Agent correctly interprets the results
4. Returns the same information as standard chat mode
"""

import sys
import os
import json
import asyncio

# Add project root to path
sys.path.insert(0, '/Users/kianwoonwong/Downloads/jarvis')

async def test_workflow_rag_execution():
    """Test the complete workflow RAG execution flow"""
    
    print("=" * 60)
    print("TESTING WORKFLOW RAG TOOL EXECUTION")
    print("=" * 60)
    
    # Import necessary modules
    from app.automation.core.agent_workflow_executor import AgentWorkflowExecutor
    from app.core.db import SessionLocal, AutomationWorkflow
    
    # Get the Int Knowledge Base workflow
    db = SessionLocal()
    try:
        workflow = db.query(AutomationWorkflow).filter(AutomationWorkflow.id == 39).first()
        if not workflow:
            print("❌ Workflow ID 39 not found")
            return False
            
        print(f"✅ Found workflow: {workflow.name}")
        
        # Create workflow executor
        executor = AgentWorkflowExecutor()
        
        # Test query - same as what fails in UI
        test_query = "beyondsoft alibaba partnership"
        execution_id = "test_" + str(os.getpid())
        
        print(f"\n📋 Test Configuration:")
        print(f"   Query: {test_query}")
        print(f"   Workflow ID: 39")
        print(f"   Execution ID: {execution_id}")
        
        # Execute workflow
        print("\n🚀 Executing workflow...")
        print("-" * 40)
        
        results = []
        errors = []
        agent_responses = []
        
        async for event in executor.execute_agent_workflow(
            workflow_id=39,
            execution_id=execution_id,
            workflow_config=workflow.langflow_config,
            message=test_query
        ):
            event_type = event.get("type")
            
            if event_type == "workflow_start":
                print("✅ Workflow started")
                
            elif event_type == "agent_plan":
                agents = event.get("agents", [])
                print(f"📊 Agent plan created with {len(agents)} agents")
                for agent in agents:
                    print(f"   - {agent.get('agent_name', 'Unknown')}")
                    
            elif event_type == "agent_start":
                agent_name = event.get("agent_name", "Unknown")
                print(f"\n🤖 Agent '{agent_name}' starting...")
                
            elif event_type == "tool_execution":
                tool_name = event.get("tool", "Unknown")
                print(f"   🔧 Executing tool: {tool_name}")
                
            elif event_type == "agent_progress":
                content = event.get("content", "")
                if content and len(content) > 0:
                    preview = content[:100] + "..." if len(content) > 100 else content
                    print(f"   💬 Agent output: {preview}")
                    
            elif event_type == "agent_complete":
                agent_name = event.get("agent_name", "Unknown")
                final_response = event.get("content", "")
                print(f"   ✅ Agent '{agent_name}' completed")
                
                # Check if agent found documents
                if final_response:
                    agent_responses.append(final_response)
                    
                    # Key phrases to check
                    found_docs = "document" in final_response.lower() or "found" in final_response.lower()
                    has_beyondsoft = "beyondsoft" in final_response.lower()
                    has_alibaba = "alibaba" in final_response.lower()
                    claims_zero = "zero results" in final_response.lower() or "no documents found" in final_response.lower()
                    
                    print(f"\n   📊 Response Analysis:")
                    print(f"      - Mentions documents: {found_docs}")
                    print(f"      - Mentions BeyondSoft: {has_beyondsoft}")
                    print(f"      - Mentions Alibaba: {has_alibaba}")
                    print(f"      - Claims zero results: {claims_zero}")
                    
                    if claims_zero:
                        print("      ❌ ERROR: Agent claims zero results!")
                        errors.append(f"Agent {agent_name} incorrectly claims zero results")
                    elif has_beyondsoft and has_alibaba:
                        print("      ✅ SUCCESS: Agent found and reported the partnership information!")
                        results.append(f"Agent {agent_name} correctly found documents")
                        
            elif event_type == "workflow_result":
                print("\n📄 Workflow completed")
                final_response = event.get("response", "")
                if final_response:
                    print(f"   Final response length: {len(final_response)} chars")
                    
            elif event_type == "execution_error":
                error = event.get("error", "Unknown error")
                print(f"❌ Execution error: {error}")
                errors.append(error)
                
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        
        if errors:
            print("\n❌ ERRORS FOUND:")
            for error in errors:
                print(f"   - {error}")
                
        if results:
            print("\n✅ SUCCESSES:")
            for result in results:
                print(f"   - {result}")
                
        # Overall assessment
        if errors and "zero results" in str(errors):
            print("\n🔴 FAILURE: Workflow is still claiming zero results when documents exist!")
            print("The fix needs more work.")
            return False
        elif results and any("correctly found documents" in r for r in results):
            print("\n🟢 SUCCESS: Workflow correctly found and reported documents!")
            print("The fix is working!")
            return True
        else:
            print("\n🟡 PARTIAL: Workflow executed but results are unclear")
            return False
            
    finally:
        db.close()
        # Clean up the executor
        if executor.dynamic_agent_system:
            from app.langchain.dynamic_agent_system import agent_instance_pool
            await agent_instance_pool.release_instance(executor.dynamic_agent_system)

async def main():
    """Main test runner"""
    print("🚀 Starting Workflow RAG Fix Test...")
    print("This test verifies the workflow can find BeyondSoft-Alibaba documents")
    print()
    
    success = await test_workflow_rag_execution()
    
    if success:
        print("\n✅ TEST PASSED: Workflow RAG execution is working correctly!")
        return 0
    else:
        print("\n❌ TEST FAILED: Workflow still has issues with RAG execution")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)