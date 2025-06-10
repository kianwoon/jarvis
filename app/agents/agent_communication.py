"""
Agent Communication Enhancement System

Provides structured communication between agents in pipelines
"""
import json
import yaml
from typing import Dict, List, Any, Optional
from datetime import datetime

class AgentCommunicationProtocol:
    """Defines how agents communicate in pipelines"""
    
    @staticmethod
    def create_agent_instruction(agent_config: Dict[str, Any], 
                               pipeline_context: Dict[str, Any]) -> str:
        """Create clear instructions for an agent based on pipeline context"""
        
        instructions = []
        
        # Basic role
        instructions.append(f"You are: {agent_config.get('role', 'an agent')}")
        instructions.append(f"Your objective: {agent_config.get('description', 'Process information')}")
        
        # Pipeline position
        position = pipeline_context.get('position', {})
        if position.get('is_first'):
            instructions.append("\nAs the FIRST agent:")
            instructions.append("- Extract and understand the user's request")
            instructions.append("- Gather all necessary information")
            instructions.append("- Prepare data for downstream processing")
        elif position.get('is_last'):
            instructions.append("\nAs the FINAL agent:")
            instructions.append("- Review all previous agents' work")
            instructions.append("- Take the concluding action")
            instructions.append("- Summarize what was accomplished")
        else:
            instructions.append(f"\nAs agent {position.get('index')} in the pipeline:")
            instructions.append("- Process information from previous agents")
            instructions.append("- Add your specialized analysis")
            instructions.append("- Prepare output for the next agent")
        
        # Available tools
        if agent_config.get('tools'):
            instructions.append(f"\nYour available tools: {', '.join(agent_config['tools'])}")
            instructions.append("Use these tools to accomplish your objectives")
        
        # Input from previous agents
        previous_outputs = pipeline_context.get('previous_outputs', [])
        if previous_outputs:
            instructions.append("\nINPUT FROM PREVIOUS AGENTS:")
            for output in previous_outputs:
                instructions.append(f"\n{output['agent']}:")
                instructions.append(f"- Type: {output.get('type', 'unknown')}")
                instructions.append(f"- Summary: {output.get('summary', 'No summary')}")
                if output.get('key_data'):
                    instructions.append(f"- Key data: {json.dumps(output['key_data'], indent=2)}")
        
        # Expected output format
        next_agent = pipeline_context.get('next_agent')
        if next_agent:
            instructions.append(f"\nNEXT AGENT EXPECTATIONS:")
            instructions.append(f"- Agent: {next_agent['name']} ({next_agent.get('role', 'Unknown role')})")
            if next_agent.get('expected_input'):
                instructions.append(f"- Expects: {next_agent['expected_input']}")
            instructions.append("- Format your output to be easily consumed by this agent")
        
        # Output structure
        instructions.append("\nSTRUCTURE YOUR RESPONSE AS:")
        instructions.append("1. **Actions Taken**: List any tool calls and their results")
        instructions.append("2. **Findings**: Key information discovered")
        instructions.append("3. **Data Package**: Structured data for next agent")
        instructions.append("4. **Next Steps**: Clear instructions for the next agent")
        
        return "\n".join(instructions)
    
    @staticmethod
    def parse_agent_response(response: str) -> Dict[str, Any]:
        """Parse agent response into structured format"""
        
        parsed = {
            "raw_response": response,
            "actions": [],
            "findings": {},
            "data_package": {},
            "next_steps": "",
            "tool_calls": []
        }
        
        # Extract sections using markers
        sections = {
            "actions": ["**Actions Taken**", "**Findings**"],
            "findings": ["**Findings**", "**Data Package**"],
            "data_package": ["**Data Package**", "**Next Steps**"],
            "next_steps": ["**Next Steps**", None]
        }
        
        for section, (start_marker, end_marker) in sections.items():
            if start_marker in response:
                start_idx = response.find(start_marker) + len(start_marker)
                if end_marker and end_marker in response:
                    end_idx = response.find(end_marker)
                    content = response[start_idx:end_idx].strip()
                else:
                    content = response[start_idx:].strip()
                
                if section == "data_package":
                    # Try to parse as JSON
                    try:
                        # Look for JSON blocks
                        import re
                        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                        if json_match:
                            parsed[section] = json.loads(json_match.group(1))
                        else:
                            # Try direct JSON parsing
                            json_start = content.find('{')
                            json_end = content.rfind('}') + 1
                            if json_start >= 0 and json_end > json_start:
                                parsed[section] = json.loads(content[json_start:json_end])
                    except:
                        parsed[section] = {"raw_content": content}
                else:
                    parsed[section] = content
        
        # Extract tool calls
        import re
        tool_pattern = r'\{"tool":\s*"([^"]+)",\s*"parameters":\s*(\{[^}]+\})\}'
        tool_matches = re.findall(tool_pattern, response)
        for tool_name, params_str in tool_matches:
            try:
                params = json.loads(params_str)
                parsed["tool_calls"].append({
                    "tool": tool_name,
                    "parameters": params
                })
            except:
                pass
        
        return parsed

class PipelineContextManager:
    """Manages context flow through pipeline execution"""
    
    def __init__(self):
        self.context = {
            "execution_id": datetime.utcnow().isoformat(),
            "pipeline_state": {},
            "agent_outputs": [],
            "shared_memory": {},
            "tool_results": []
        }
    
    def add_agent_output(self, agent_name: str, output: Dict[str, Any]):
        """Add an agent's output to the context"""
        
        agent_output = {
            "agent": agent_name,
            "timestamp": datetime.utcnow().isoformat(),
            "output": output,
            "type": self._determine_output_type(output)
        }
        
        # Extract key data for easy access
        if "data_package" in output and isinstance(output["data_package"], dict):
            agent_output["key_data"] = output["data_package"]
        
        # Create summary
        agent_output["summary"] = self._create_summary(output)
        
        self.context["agent_outputs"].append(agent_output)
        
        # Update shared memory with important data
        self._update_shared_memory(agent_name, output)
    
    def get_context_for_agent(self, agent_name: str, position: int, 
                            total_agents: int, next_agent_info: Optional[Dict] = None) -> Dict[str, Any]:
        """Get relevant context for an agent"""
        
        return {
            "position": {
                "index": position + 1,
                "total": total_agents,
                "is_first": position == 0,
                "is_last": position == total_agents - 1
            },
            "previous_outputs": self.context["agent_outputs"],
            "shared_memory": self.context["shared_memory"],
            "next_agent": next_agent_info
        }
    
    def _determine_output_type(self, output: Dict[str, Any]) -> str:
        """Determine the type of output from an agent"""
        
        if "tool_calls" in output and output["tool_calls"]:
            tools_used = [tc["tool"] for tc in output["tool_calls"]]
            if any("email" in tool for tool in tools_used):
                return "email_operation"
            elif any("search" in tool for tool in tools_used):
                return "search_operation"
            else:
                return "tool_operation"
        elif "analysis" in str(output).lower():
            return "analysis"
        elif "decision" in str(output).lower():
            return "decision"
        else:
            return "information"
    
    def _create_summary(self, output: Dict[str, Any]) -> str:
        """Create a brief summary of agent output"""
        
        summary_parts = []
        
        if output.get("tool_calls"):
            tools = [tc["tool"] for tc in output["tool_calls"]]
            summary_parts.append(f"Used tools: {', '.join(tools)}")
        
        if output.get("findings"):
            summary_parts.append("Gathered findings")
        
        if output.get("data_package"):
            summary_parts.append("Prepared data package")
        
        if output.get("next_steps"):
            summary_parts.append("Provided next steps")
        
        return "; ".join(summary_parts) if summary_parts else "Processed information"
    
    def _update_shared_memory(self, agent_name: str, output: Dict[str, Any]):
        """Update shared memory with important data"""
        
        # Store email IDs
        if "email_id" in output.get("data_package", {}):
            self.context["shared_memory"]["current_email_id"] = output["data_package"]["email_id"]
        
        # Store email content
        if "email_content" in output.get("data_package", {}):
            self.context["shared_memory"]["email_content"] = output["data_package"]["email_content"]
        
        # Store decisions
        if "decision" in output.get("data_package", {}):
            if "decisions" not in self.context["shared_memory"]:
                self.context["shared_memory"]["decisions"] = []
            self.context["shared_memory"]["decisions"].append({
                "agent": agent_name,
                "decision": output["data_package"]["decision"]
            })

# Example pipeline templates
PIPELINE_TEMPLATES = {
    "email_response": {
        "agents": [
            {
                "name": "email_reader",
                "expected_output": {
                    "email_id": "string",
                    "subject": "string", 
                    "from": "string",
                    "body": "string",
                    "requires_response": "boolean"
                }
            },
            {
                "name": "email_responder",
                "expected_input": "Email content and analysis from previous agent",
                "expected_output": {
                    "email_sent": "boolean",
                    "message_id": "string"
                }
            }
        ]
    },
    "research_and_respond": {
        "agents": [
            {
                "name": "researcher",
                "expected_output": {
                    "findings": "array",
                    "sources": "array",
                    "summary": "string"
                }
            },
            {
                "name": "analyzer",
                "expected_input": "Research findings",
                "expected_output": {
                    "analysis": "string",
                    "recommendations": "array"
                }
            },
            {
                "name": "responder",
                "expected_input": "Analysis and recommendations",
                "expected_output": {
                    "response": "string",
                    "action_taken": "string"
                }
            }
        ]
    }
}