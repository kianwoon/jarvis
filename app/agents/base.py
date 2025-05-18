from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class AgentState(BaseModel):
    """State model for agents."""
    task_id: str
    status: str
    metadata: Dict[str, Any] = {}
    output: Optional[Any] = None
    error: Optional[str] = None

class AgentContext(BaseModel):
    """Context model for agent execution."""
    task_id: str
    input_data: Dict[str, Any]
    state: Optional[AgentState] = None
    metadata: Dict[str, Any] = {}

class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    async def execute(self, context: AgentContext) -> AgentState:
        """Execute the agent's main logic."""
        pass
    
    @abstractmethod
    async def validate(self, context: AgentContext) -> bool:
        """Validate the input context."""
        pass
    
    async def update_state(
        self,
        state: AgentState,
        status: str,
        output: Optional[Any] = None,
        error: Optional[str] = None
    ) -> AgentState:
        """Update the agent's state."""
        state.status = status
        if output is not None:
            state.output = output
        if error is not None:
            state.error = error
        return state 