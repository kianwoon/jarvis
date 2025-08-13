#!/usr/bin/env python3
"""
Agent Work Request Interface
=============================
This script is the ONLY way Claude can execute actual work in the system.
Claude can only read/analyze code and must delegate all execution to agents.

Usage:
    python request_agent_work.py --task "task_description" --agents "agent1,agent2" [options]

Available Agents:
    - Research Agent: Information gathering and analysis
    - Code Agent: Code generation and modification  
    - Data Agent: Data processing and analysis
    - Planning Agent: Task planning and coordination
    - QA Agent: Testing and validation
    - Documentation Agent: Documentation generation
    - Integration Agent: System integration tasks
    - Security Agent: Security analysis and validation

Example:
    python request_agent_work.py \
        --task "Update the user authentication system to use JWT tokens" \
        --agents "Planning Agent,Code Agent,QA Agent" \
        --priority high \
        --context "Current system uses session-based auth"
"""

import argparse
import asyncio
import json
import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# Add app directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Available agent types
AVAILABLE_AGENTS = [
    "Research Agent",
    "Code Agent", 
    "Data Agent",
    "Planning Agent",
    "QA Agent",
    "Documentation Agent",
    "Integration Agent",
    "Security Agent"
]

class AgentWorkRequest:
    """Interface for requesting work from the agent system"""
    
    def __init__(self):
        self.agent_system = None
        self.settings = None
        
    async def initialize(self):
        """Initialize the agent system connection"""
        try:
            # Import agent system
            from app.langchain.dynamic_agent_system import DynamicMultiAgentSystem
            from app.core.config import get_settings
            
            # Get settings
            self.settings = get_settings()
            
            # Initialize agent system
            self.agent_system = DynamicMultiAgentSystem(self.settings)
            logger.info("Agent system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent system: {e}")
            raise
    
    async def validate_agents(self, agent_names: List[str]) -> List[str]:
        """Validate that requested agents are available"""
        valid_agents = []
        invalid_agents = []
        
        for agent in agent_names:
            agent = agent.strip()
            if agent in AVAILABLE_AGENTS:
                valid_agents.append(agent)
            else:
                invalid_agents.append(agent)
        
        if invalid_agents:
            logger.warning(f"Invalid agents requested: {invalid_agents}")
            logger.info(f"Available agents: {AVAILABLE_AGENTS}")
        
        return valid_agents
    
    async def execute_task(
        self, 
        task: str,
        agents: List[str],
        context: Optional[str] = None,
        priority: str = "normal",
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        """Execute a task using the specified agents"""
        
        # Validate agents
        valid_agents = await self.validate_agents(agents)
        if not valid_agents:
            return {
                "status": "error",
                "message": "No valid agents specified",
                "available_agents": AVAILABLE_AGENTS
            }
        
        # Prepare the request
        request = {
            "query": task,
            "selected_agents": valid_agents,
            "context": context or "",
            "priority": priority,
            "max_iterations": max_iterations,
            "metadata": {
                "requested_by": "Claude",
                "timestamp": datetime.now().isoformat(),
                "execution_mode": "delegated"
            }
        }
        
        logger.info(f"Executing task with agents: {valid_agents}")
        logger.info(f"Task: {task}")
        
        try:
            # Execute through agent system
            result = await self.agent_system.process_request(
                query=request["query"],
                selected_agents=request["selected_agents"],
                context=request["context"],
                max_iterations=request["max_iterations"]
            )
            
            return {
                "status": "success",
                "result": result,
                "agents_used": valid_agents,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "agents_attempted": valid_agents
            }
    
    async def get_agent_capabilities(self) -> Dict[str, str]:
        """Get detailed capabilities of each agent"""
        capabilities = {
            "Research Agent": "Searches codebase, analyzes patterns, gathers information from multiple sources",
            "Code Agent": "Generates code, modifies existing files, implements features, fixes bugs",
            "Data Agent": "Processes data, performs analysis, handles database operations, manages collections",
            "Planning Agent": "Creates implementation plans, coordinates multi-agent workflows, designs architectures",
            "QA Agent": "Writes tests, validates implementations, checks for bugs, ensures quality",
            "Documentation Agent": "Generates documentation, creates API docs, writes user guides, updates README files",
            "Integration Agent": "Integrates external services, sets up APIs, configures connections, handles MCP tools",
            "Security Agent": "Analyzes security vulnerabilities, validates authentication, checks permissions, audits code"
        }
        return capabilities

async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Request work from the agent system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--task",
        required=False,
        help="Description of the task to execute"
    )
    
    parser.add_argument(
        "--agents",
        required=False,
        help="Comma-separated list of agents to use (e.g., 'Code Agent,QA Agent')"
    )
    
    parser.add_argument(
        "--context",
        help="Additional context for the task"
    )
    
    parser.add_argument(
        "--priority",
        choices=["low", "normal", "high", "critical"],
        default="normal",
        help="Task priority level"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum iterations for agent collaboration"
    )
    
    parser.add_argument(
        "--list-agents",
        action="store_true",
        help="List available agents and their capabilities"
    )
    
    parser.add_argument(
        "--output",
        choices=["json", "text"],
        default="text",
        help="Output format"
    )
    
    args = parser.parse_args()
    
    # Create request handler
    request_handler = AgentWorkRequest()
    
    # Initialize the system
    try:
        await request_handler.initialize()
    except Exception as e:
        print(f"ERROR: Failed to initialize agent system: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Handle list agents request
    if args.list_agents:
        capabilities = await request_handler.get_agent_capabilities()
        
        if args.output == "json":
            print(json.dumps(capabilities, indent=2))
        else:
            print("\n=== Available Agents ===\n")
            for agent, capability in capabilities.items():
                print(f"• {agent}")
                print(f"  {capability}\n")
        return
    
    # Check required args for task execution
    if not args.task or not args.agents:
        print("ERROR: --task and --agents are required for task execution", file=sys.stderr)
        print("Use --list-agents to see available agents", file=sys.stderr)
        sys.exit(1)
    
    # Parse agents
    agent_list = [a.strip() for a in args.agents.split(",")]
    
    # Execute the task
    result = await request_handler.execute_task(
        task=args.task,
        agents=agent_list,
        context=args.context,
        priority=args.priority,
        max_iterations=args.max_iterations
    )
    
    # Output results
    if args.output == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"\n=== Task Execution Result ===")
        print(f"Status: {result['status']}")
        
        if result['status'] == 'success':
            print(f"Agents Used: {', '.join(result['agents_used'])}")
            print(f"Timestamp: {result['timestamp']}")
            print(f"\nResult:")
            if isinstance(result['result'], dict):
                print(json.dumps(result['result'], indent=2, default=str))
            else:
                print(result['result'])
        else:
            print(f"Error: {result.get('message', 'Unknown error')}")
            if 'available_agents' in result:
                print(f"\nAvailable agents: {', '.join(result['available_agents'])}")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())