"""
Agent Contract System for Agentic Pipelines

Defines formal interfaces between agents to ensure proper communication
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum

class DataType(str, Enum):
    """Types of data agents can produce/consume"""
    EMAIL_CONTENT = "email_content"
    EMAIL_METADATA = "email_metadata"
    ANALYSIS_RESULT = "analysis_result"
    DECISION = "decision"
    ACTION_PLAN = "action_plan"
    TOOL_RESULT = "tool_result"
    DOCUMENT = "document"
    SUMMARY = "summary"
    QUERY_RESULT = "query_result"
    STRUCTURED_DATA = "structured_data"
    FREE_TEXT = "free_text"
    STRING = "string"
    OBJECT = "object"
    ARRAY = "array"
    BOOLEAN = "boolean"
    NUMBER = "number"

class AgentCapability(BaseModel):
    """Defines what an agent can do"""
    name: str
    description: str
    input_types: List[DataType] = Field(description="Data types this agent can process")
    output_types: List[DataType] = Field(description="Data types this agent produces")
    required_tools: List[str] = Field(default=[], description="Tools required for this capability")
    
class AgentContract(BaseModel):
    """Formal contract defining agent's interface"""
    agent_name: str
    role: str
    capabilities: List[AgentCapability]
    input_schema: Optional[Dict[str, Any]] = Field(default=None, description="JSON schema for expected inputs")
    output_schema: Optional[Dict[str, Any]] = Field(default=None, description="JSON schema for outputs")
    context_requirements: List[str] = Field(default=[], description="Required context fields")
    
class AgentHandoff(BaseModel):
    """Structured handoff between agents"""
    from_agent: str
    to_agent: str
    timestamp: str
    data_type: DataType
    data: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default={})
    instructions: Optional[str] = Field(description="Specific instructions for next agent")
    
# Example contracts
EMAIL_READER_CONTRACT = AgentContract(
    agent_name="email_reader",
    role="Read and extract email content",
    capabilities=[
        AgentCapability(
            name="read_emails",
            description="Search and read email content",
            input_types=[DataType.FREE_TEXT, DataType.QUERY_RESULT],
            output_types=[DataType.EMAIL_CONTENT, DataType.EMAIL_METADATA],
            required_tools=["search_emails", "read_email"]
        )
    ],
    output_schema={
        "type": "object",
        "properties": {
            "email_id": {"type": "string"},
            "subject": {"type": "string"},
            "from": {"type": "string"},
            "to": {"type": "array", "items": {"type": "string"}},
            "body": {"type": "string"},
            "timestamp": {"type": "string"},
            "requires_response": {"type": "boolean"},
            "key_points": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["email_id", "subject", "from", "body"]
    }
)

EMAIL_RESPONDER_CONTRACT = AgentContract(
    agent_name="email_responder",
    role="Compose and send email responses",
    capabilities=[
        AgentCapability(
            name="send_emails",
            description="Compose and send email responses",
            input_types=[DataType.EMAIL_CONTENT, DataType.ANALYSIS_RESULT],
            output_types=[DataType.TOOL_RESULT],
            required_tools=["gmail_send", "draft_email"]
        )
    ],
    input_schema={
        "type": "object",
        "properties": {
            "original_email": {"type": "object"},
            "response_directive": {"type": "string"},
            "key_points_to_address": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["original_email"]
    }
)


def create_agent_contract(
    name: str,
    description: str,
    instructions: str,
    capabilities: List[Any],
    expected_input: Dict[str, Any],
    output_format: Dict[str, Any],
    tools: List[str] = None,
    handoff_to: List[str] = None
) -> AgentContract:
    """Helper function to create an agent contract from parameters"""
    
    # Convert capability list to AgentCapability objects
    agent_capabilities = []
    if isinstance(capabilities, list):
        for cap in capabilities:
            if isinstance(cap, str):
                # Simple string capability
                agent_capabilities.append(
                    AgentCapability(
                        name=cap,
                        description=cap,
                        input_types=[DataType.FREE_TEXT],
                        output_types=[DataType.FREE_TEXT],
                        required_tools=tools or []
                    )
                )
            elif isinstance(cap, dict):
                # Dictionary with more details
                agent_capabilities.append(
                    AgentCapability(
                        name=cap.get("name", ""),
                        description=cap.get("description", ""),
                        input_types=cap.get("input_types", [DataType.FREE_TEXT]),
                        output_types=cap.get("output_types", [DataType.STRUCTURED_DATA]),
                        required_tools=cap.get("required_tools", tools or [])
                    )
                )
            elif isinstance(cap, AgentCapability):
                # Already an AgentCapability
                agent_capabilities.append(cap)
    
    # Create the contract
    contract = AgentContract(
        agent_name=name,
        role=description,
        capabilities=agent_capabilities,
        input_schema=expected_input,
        output_schema=output_format,
        context_requirements=handoff_to or []
    )
    
    # Store additional metadata as attributes
    setattr(contract, 'instructions', instructions)
    setattr(contract, 'tools', tools or [])
    setattr(contract, 'name', name)
    
    return contract