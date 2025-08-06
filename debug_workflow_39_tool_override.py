#!/usr/bin/env python3
"""
Debug script to test workflow 39 tool override mechanism
"""
import asyncio
import logging
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def debug_workflow_39():
    """Test the workflow 39 agent tool override"""
    
    # Import the workflow executor  
    from app.automation.core.agent_workflow_executor import AgentWorkflowExecutor
    from app.automation.core.workflow_state import WorkflowState
    
    # Create executor instance
    executor = AgentWorkflowExecutor()
    
    # Simulate the workflow 39 agent configuration
    agent_config = {
        "name": "customer service",
        "node": {
            "agent_name": "customer service", 
            "tools": ["rag_knowledge_search"]  # Override tools
        },
        "tools": ["rag_knowledge_search"],  # Also set at top level
        "agent_config": {}
    }
    
    query = "search internal knowledge base. partnership between beyondsoft and alibaba"
    workflow_id = 39
    execution_id = f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info("=== DEBUGGING WORKFLOW 39 TOOL OVERRIDE ===")
    logger.info(f"Agent config: {json.dumps(agent_config, indent=2)}")
    logger.info(f"Query: {query}")
    logger.info(f"Expected tools: ['rag_knowledge_search']")
    
    try:
        # Create minimal workflow state
        workflow_state = WorkflowState(
            workflow_id=workflow_id,
            execution_id=execution_id
        )
        
        logger.info("Executing agent with tool override...")
        
        # Initialize the dynamic agent system (normally done in main execution)
        from app.langchain.dynamic_agent_system import agent_instance_pool
        executor.dynamic_agent_system = await agent_instance_pool.get_or_create_instance()
        
        # Execute the agent
        result = await executor._execute_single_agent(
            agent=agent_config,
            prompt=query,
            workflow_id=workflow_id,
            execution_id=execution_id,
            workflow_state=workflow_state
        )
        
        logger.info("=== EXECUTION RESULT ===")
        logger.info(f"Response length: {len(result.get('response', ''))}")
        logger.info(f"Tools used: {result.get('tools_used', [])}")
        logger.info(f"Success: {result.get('success', False)}")
        
        if result.get('tools_used'):
            logger.info("✅ SUCCESS: Tools were used!")
            for tool in result.get('tools_used', []):
                logger.info(f"  - Tool: {tool.get('tool')}, Success: {tool.get('success')}")
        else:
            logger.error("❌ PROBLEM: No tools were used despite configuration!")
            
        logger.info(f"Response preview: {result.get('response', '')[:500]}...")
        
    except Exception as e:
        logger.error(f"❌ EXECUTION FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(debug_workflow_39())