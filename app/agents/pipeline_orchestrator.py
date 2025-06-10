"""
Enhanced Pipeline Orchestrator with Agent Awareness

Manages agent communication and context in pipelines
"""
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from app.agents.agent_contracts import AgentContract, AgentHandoff, DataType

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    """Orchestrates agent communication in pipelines"""
    
    def __init__(self, pipeline_config: Dict[str, Any]):
        self.pipeline_config = pipeline_config
        self.agent_contracts = {}
        self.execution_context = {}
        
    def register_agent_contract(self, contract: AgentContract):
        """Register an agent's contract"""
        self.agent_contracts[contract.agent_name] = contract
        
    def create_agent_context(self, agent_name: str, position: int, 
                           pipeline_agents: List[str]) -> Dict[str, Any]:
        """Create context for an agent including pipeline awareness"""
        
        # Identify upstream and downstream agents
        upstream_agents = pipeline_agents[:position] if position > 0 else []
        downstream_agents = pipeline_agents[position + 1:] if position < len(pipeline_agents) - 1 else []
        
        # Get downstream requirements
        downstream_requirements = []
        for downstream_agent in downstream_agents:
            if downstream_agent in self.agent_contracts:
                contract = self.agent_contracts[downstream_agent]
                downstream_requirements.append({
                    "agent": downstream_agent,
                    "role": contract.role,
                    "expects": contract.input_schema,
                    "required_data_types": [dt.value for dt in contract.capabilities[0].input_types]
                })
        
        # Build the context
        context = {
            "pipeline_info": {
                "total_agents": len(pipeline_agents),
                "current_position": position + 1,
                "is_first": position == 0,
                "is_last": position == len(pipeline_agents) - 1,
                "upstream_agents": upstream_agents,
                "downstream_agents": downstream_agents
            },
            "downstream_requirements": downstream_requirements,
            "execution_history": self.execution_context.get("history", []),
            "shared_context": self.execution_context.get("shared", {})
        }
        
        return context
        
    def create_agent_prompt(self, agent_name: str, base_prompt: str, 
                          context: Dict[str, Any], user_query: str) -> str:
        """Create an enhanced prompt with pipeline awareness"""
        
        prompt_parts = [base_prompt]
        
        # Add pipeline context
        if context["pipeline_info"]["is_first"]:
            prompt_parts.append("\nYou are the FIRST agent in this pipeline. Your role is to:")
            prompt_parts.append("1. Understand and process the user's initial query")
            prompt_parts.append("2. Gather necessary information using your tools")
            prompt_parts.append("3. Prepare structured output for downstream agents")
        elif context["pipeline_info"]["is_last"]:
            prompt_parts.append("\nYou are the FINAL agent in this pipeline. Your role is to:")
            prompt_parts.append("1. Process all information from previous agents")
            prompt_parts.append("2. Take the final action to complete the user's request")
            prompt_parts.append("3. Provide a clear summary of what was accomplished")
        else:
            prompt_parts.append(f"\nYou are agent {context['pipeline_info']['current_position']} of {context['pipeline_info']['total_agents']} in this pipeline.")
        
        # Add downstream requirements
        if context["downstream_requirements"]:
            prompt_parts.append("\nDOWNSTREAM AGENT REQUIREMENTS:")
            for req in context["downstream_requirements"]:
                prompt_parts.append(f"\nAgent: {req['agent']} ({req['role']})")
                prompt_parts.append(f"Expects data types: {', '.join(req['required_data_types'])}")
                if req['expects']:
                    prompt_parts.append(f"Expected format: {json.dumps(req['expects'], indent=2)}")
        
        # Add execution history
        if context["execution_history"]:
            prompt_parts.append("\nPREVIOUS AGENTS' OUTPUTS:")
            for hist in context["execution_history"]:
                prompt_parts.append(f"\n{hist['agent']}:")
                if isinstance(hist['output'], dict):
                    prompt_parts.append(json.dumps(hist['output'], indent=2))
                else:
                    prompt_parts.append(str(hist['output']))
        
        # Add structured output format
        prompt_parts.append("\nOUTPUT FORMAT:")
        prompt_parts.append("Structure your response to include:")
        prompt_parts.append("1. Tool calls (if needed): {\"tool\": \"tool_name\", \"parameters\": {...}}")
        prompt_parts.append("2. Analysis/findings: Clear summary of what you discovered")
        prompt_parts.append("3. Data for next agent: Structured data matching downstream requirements")
        prompt_parts.append("4. Instructions for next agent: Specific guidance if needed")
        
        # Add user query
        prompt_parts.append(f"\nUSER QUERY: {user_query}")
        
        return "\n".join(prompt_parts)
        
    def create_handoff(self, from_agent: str, to_agent: str, 
                      agent_output: str, tools_used: List[Dict]) -> AgentHandoff:
        """Create a structured handoff between agents"""
        
        # Parse agent output to extract structured data
        structured_data = self._parse_agent_output(agent_output, tools_used)
        
        # Determine data type based on content
        data_type = self._determine_data_type(structured_data)
        
        # Extract instructions for next agent if present
        instructions = structured_data.get("instructions_for_next_agent", 
                                         structured_data.get("next_steps", None))
        
        handoff = AgentHandoff(
            from_agent=from_agent,
            to_agent=to_agent,
            timestamp=datetime.utcnow().isoformat(),
            data_type=data_type,
            data=structured_data,
            metadata={
                "tools_used": [t["tool"] for t in tools_used] if tools_used else [],
                "execution_time": structured_data.get("execution_time", 0)
            },
            instructions=instructions
        )
        
        return handoff
        
    def _parse_agent_output(self, output: str, tools_used: List[Dict]) -> Dict[str, Any]:
        """Parse agent output to extract structured data"""
        
        structured_data = {}
        
        # Try to extract JSON blocks
        import re
        json_pattern = r'\{[^{}]*\}'
        json_matches = re.findall(json_pattern, output)
        
        for match in json_matches:
            try:
                data = json.loads(match)
                if "tool" not in data:  # It's structured data, not a tool call
                    structured_data.update(data)
            except:
                pass
        
        # Extract tool results
        if tools_used:
            structured_data["tool_results"] = []
            for tool in tools_used:
                if "result" in tool:
                    structured_data["tool_results"].append({
                        "tool": tool["tool"],
                        "result": tool["result"]
                    })
        
        # Extract email data if present
        email_pattern = r'Subject:\s*(.+?)(?:\n|$)'
        email_match = re.search(email_pattern, output)
        if email_match:
            structured_data["email_subject"] = email_match.group(1)
            
        # Extract key patterns
        if "email_id" in output or "ID:" in output:
            id_pattern = r'ID:\s*([a-zA-Z0-9]+)'
            id_match = re.search(id_pattern, output)
            if id_match:
                structured_data["email_id"] = id_match.group(1)
        
        # Add raw output as fallback
        structured_data["raw_output"] = output
        
        return structured_data
        
    def _determine_data_type(self, data: Dict[str, Any]) -> DataType:
        """Determine the data type based on content"""
        
        if "email_id" in data or "email_subject" in data:
            return DataType.EMAIL_CONTENT
        elif "analysis" in data or "findings" in data:
            return DataType.ANALYSIS_RESULT
        elif "tool_results" in data:
            return DataType.TOOL_RESULT
        elif "action_plan" in data or "next_steps" in data:
            return DataType.ACTION_PLAN
        else:
            return DataType.FREE_TEXT
            
    def update_execution_context(self, agent_name: str, handoff: AgentHandoff):
        """Update the shared execution context"""
        
        # Add to history
        if "history" not in self.execution_context:
            self.execution_context["history"] = []
            
        self.execution_context["history"].append({
            "agent": agent_name,
            "timestamp": handoff.timestamp,
            "output": handoff.data,
            "data_type": handoff.data_type.value
        })
        
        # Update shared context with key information
        if "shared" not in self.execution_context:
            self.execution_context["shared"] = {}
            
        # Store email-related data in shared context
        if handoff.data_type == DataType.EMAIL_CONTENT:
            self.execution_context["shared"]["current_email"] = handoff.data
        elif handoff.data_type == DataType.TOOL_RESULT:
            if "email_sent" in str(handoff.data):
                self.execution_context["shared"]["email_sent"] = True
                
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the pipeline execution"""
        
        return {
            "total_agents_executed": len(self.execution_context.get("history", [])),
            "data_flow": [
                {
                    "agent": h["agent"],
                    "data_type": h["data_type"],
                    "timestamp": h["timestamp"]
                }
                for h in self.execution_context.get("history", [])
            ],
            "shared_context": self.execution_context.get("shared", {}),
            "final_result": self.execution_context.get("history", [])[-1] if self.execution_context.get("history") else None
        }