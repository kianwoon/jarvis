"""
Multi-Agent System using LangGraph
Implements router-based communication between specialized agents
"""

from typing import Dict, List, Any, Optional, TypedDict
from langgraph.graph import StateGraph, Graph
from langgraph.checkpoint.redis import RedisSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from app.core.langgraph_agents_cache import get_langgraph_agents, get_agent_by_role
from app.core.mcp_tools_cache import get_enabled_mcp_tools
from app.core.redis_client import get_redis_client_for_langgraph
import json
import uuid
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Initialize Redis saver for conversation persistence using pooled connection
try:
    redis_conn = get_redis_client_for_langgraph()
    if redis_conn:
        checkpointer = RedisSaver(redis_conn)
        logger.info("Multi-agent Redis checkpointer initialized with pooled connection")
    else:
        checkpointer = None
        logger.warning("Redis connection pool not available for multi-agent checkpointing")
except Exception as e:
    logger.error(f"Failed to initialize Redis checkpointer: {e}")
    checkpointer = None

class AgentState(TypedDict):
    """State shared between agents"""
    query: str
    conversation_id: str
    messages: List[BaseMessage]
    routing_decision: Dict[str, Any]
    agent_outputs: Dict[str, Any]
    tools_used: List[str]
    documents_retrieved: List[Dict[str, Any]]
    final_response: str
    metadata: Dict[str, Any]
    error: Optional[str]

class MultiAgentSystem:
    """Orchestrates communication between multiple specialized agents"""
    
    def __init__(self, conversation_id: Optional[str] = None):
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.agents = get_langgraph_agents()
        self.graph = self._build_graph()
        
    def _build_graph(self) -> Graph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes for each active agent
        for agent_name, agent_config in self.agents.items():
            if agent_config.get("is_active", True):
                workflow.add_node(agent_name, self._create_agent_node(agent_config))
        
        # Add router node (always first)
        workflow.add_node("router", self._router_node)
        
        # Add synthesizer node (always last)
        workflow.add_node("synthesizer", self._synthesizer_node)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Add conditional edges based on routing
        workflow.add_conditional_edges(
            "router",
            self._route_query,
            {
                "document_researcher": "document_researcher",
                "tool_executor": "tool_executor",
                "context_manager": "context_manager",
                "synthesizer": "synthesizer"
            }
        )
        
        # All specialized agents lead to synthesizer
        for agent_name in ["document_researcher", "tool_executor", "context_manager"]:
            if agent_name in self.agents:
                workflow.add_edge(agent_name, "synthesizer")
        
        # Set finish point
        workflow.set_finish_point("synthesizer")
        
        return workflow.compile(checkpointer=checkpointer)
    
    def _create_agent_node(self, agent_config: Dict[str, Any]):
        """Create a node function for an agent"""
        def agent_node(state: AgentState) -> AgentState:
            agent_name = agent_config["name"]
            system_prompt = agent_config["system_prompt"]
            
            # Prepare messages for the agent
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Query: {state['query']}\n\nContext: {json.dumps(state.get('agent_outputs', {}))}")
            ]
            
            # Generate response
            response = self._generate_agent_response(messages, agent_config)
            
            # Update state with agent's output
            if "agent_outputs" not in state:
                state["agent_outputs"] = {}
            state["agent_outputs"][agent_name] = response
            
            # If this agent uses tools, update tools_used
            if agent_config.get("tools"):
                state["tools_used"] = state.get("tools_used", []) + agent_config["tools"]
            
            return state
        
        return agent_node
    
    def _router_node(self, state: AgentState) -> AgentState:
        """Router agent that determines which agents to engage"""
        router_agent = get_agent_by_role("classifier")
        if not router_agent:
            # Fallback routing logic
            state["routing_decision"] = {
                "agents": ["document_researcher"],
                "reasoning": "No router agent configured, using default path"
            }
            return state
        
        # Use the classifier agent to determine routing
        messages = [
            SystemMessage(content=router_agent["system_prompt"]),
            HumanMessage(content=f"Query: {state['query']}")
        ]
        
        response = self._generate_agent_response(messages, router_agent)
        
        # Parse routing decision
        routing_decision = self._parse_routing_decision(response)
        state["routing_decision"] = routing_decision
        
        return state
    
    def _synthesizer_node(self, state: AgentState) -> AgentState:
        """Synthesizer agent that combines outputs from other agents"""
        synthesizer_agent = get_agent_by_role("synthesizer")
        if not synthesizer_agent:
            # Simple concatenation fallback
            outputs = state.get("agent_outputs", {})
            state["final_response"] = "\n\n".join([
                f"**{agent}**: {output}" 
                for agent, output in outputs.items()
            ])
            return state
        
        # Use synthesizer agent to create final response
        messages = [
            SystemMessage(content=synthesizer_agent["system_prompt"]),
            HumanMessage(content=f"""
Query: {state['query']}

Agent Outputs:
{json.dumps(state.get('agent_outputs', {}), indent=2)}

Documents Retrieved:
{json.dumps(state.get('documents_retrieved', []), indent=2)}

Please synthesize a comprehensive response.
""")
        ]
        
        response = self._generate_agent_response(messages, synthesizer_agent)
        state["final_response"] = response
        
        return state
    
    def _route_query(self, state: AgentState) -> str:
        """Determine next agent based on routing decision"""
        routing = state.get("routing_decision", {})
        agents = routing.get("agents", ["synthesizer"])
        
        # Return the first agent in the list (simplified routing)
        # In a more complex system, this could branch to multiple agents
        if agents and agents[0] in self.agents:
            return agents[0]
        return "synthesizer"
    
    def _parse_routing_decision(self, response: str) -> Dict[str, Any]:
        """Parse the router's response to determine which agents to engage"""
        # Simple parsing logic - can be enhanced with structured output
        decision = {
            "agents": [],
            "reasoning": response
        }
        
        # Look for agent names in the response
        response_lower = response.lower()
        if "document" in response_lower or "rag" in response_lower:
            decision["agents"].append("document_researcher")
        if "tool" in response_lower or "execute" in response_lower:
            decision["agents"].append("tool_executor")
        if "context" in response_lower or "history" in response_lower:
            decision["agents"].append("context_manager")
        
        # Default to document researcher if no specific agent identified
        if not decision["agents"]:
            decision["agents"] = ["document_researcher"]
        
        return decision
    
    def _generate_agent_response(self, messages: List[BaseMessage], agent_config: Dict[str, Any]) -> str:
        """Generate response for an agent using the LLM"""
        # Convert messages to prompt
        prompt = "\n\n".join([
            f"{msg.__class__.__name__}: {msg.content}" 
            for msg in messages
        ])
        
        # Use the inference module to generate response
        # This is a simplified version - you'd integrate with your actual LLM
        response = ""
        try:
            # TODO: Replace with actual LLM call using your inference module
            # For now, return a mock response
            response = f"Mock response from {agent_config['name']} for testing"
        except Exception as e:
            response = f"Error generating response: {str(e)}"
        
        return response
    
    async def process_query(self, query: str, stream: bool = True):
        """Process a query through the multi-agent system"""
        # Initialize state
        initial_state = AgentState(
            query=query,
            conversation_id=self.conversation_id,
            messages=[HumanMessage(content=query)],
            routing_decision={},
            agent_outputs={},
            tools_used=[],
            documents_retrieved=[],
            final_response="",
            metadata={
                "start_time": datetime.now().isoformat(),
                "mode": "multi_agent"
            },
            error=None
        )
        
        try:
            # Run the graph
            config = {"configurable": {"thread_id": self.conversation_id}}
            
            if stream:
                # Stream events as they happen
                async for event in self.graph.astream_events(initial_state, config):
                    yield {
                        "type": "agent_event",
                        "event": event
                    }
            else:
                # Run to completion
                final_state = await self.graph.ainvoke(initial_state, config)
                yield {
                    "type": "final_response",
                    "response": final_state["final_response"],
                    "metadata": final_state["metadata"],
                    "routing": final_state["routing_decision"],
                    "agent_outputs": final_state["agent_outputs"]
                }
                
        except Exception as e:
            yield {
                "type": "error",
                "error": str(e)
            }
    
    async def stream_events(
        self, 
        query: str, 
        selected_agents: Optional[List[str]] = None,
        max_iterations: int = 10
    ):
        """Stream events from multi-agent processing"""
        # Filter agents if specific ones are selected
        if selected_agents:
            # Store original agents and filter
            self.selected_agents = selected_agents
        
        async for event in self.process_query(query, stream=True):
            yield event

# Convenience function for API endpoint
async def process_multi_agent_query(
    question: str,
    conversation_id: Optional[str] = None,
    stream: bool = True
):
    """Process a query using the multi-agent system"""
    system = MultiAgentSystem(conversation_id)
    async for result in system.process_query(question, stream):
        yield result