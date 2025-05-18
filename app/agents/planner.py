from typing import Any, Dict, List
from app.agents.base import BaseAgent, AgentContext, AgentState
from app.llm.base import LLMConfig
from app.llm.qwen import QwenLLM

class PlannerAgent(BaseAgent):
    """Agent responsible for planning document structure."""
    
    def __init__(self):
        super().__init__(name="planner")
        self.llm = QwenLLM(
            LLMConfig(
                model_name="Qwen/Qwen-7B-Chat",
                temperature=0.7,
                max_tokens=1000
            )
        )
    
    async def validate(self, context: AgentContext) -> bool:
        """Validate the input context."""
        required_fields = ["topic", "doc_id"]
        return all(field in context.input_data for field in required_fields)
    
    async def execute(self, context: AgentContext) -> AgentState:
        """Execute the planning phase."""
        if not await self.validate(context):
            return await self.update_state(
                context.state or AgentState(task_id=context.task_id, status="created"),
                status="failed",
                error="Invalid input context"
            )
        
        try:
            # Generate document outline
            prompt = self._create_planning_prompt(context.input_data["topic"])
            response = await self.llm.generate(prompt)
            
            # Parse and structure the outline
            outline = self._parse_outline(response.text)
            
            return await self.update_state(
                context.state or AgentState(task_id=context.task_id, status="created"),
                status="completed",
                output=outline
            )
        except Exception as e:
            return await self.update_state(
                context.state or AgentState(task_id=context.task_id, status="created"),
                status="failed",
                error=str(e)
            )
    
    def _create_planning_prompt(self, topic: str) -> str:
        """Create the planning prompt for the LLM."""
        return f"""Create a detailed outline for a document about: {topic}

The outline should include:
1. Main sections
2. Subsections
3. Key points to cover in each section

Format the response as a structured outline with clear hierarchy."""

    def _parse_outline(self, llm_response: str) -> Dict[str, Any]:
        """Parse the LLM response into a structured outline."""
        # TODO: Implement outline parsing logic
        # This should convert the LLM's text response into a structured format
        return {
            "sections": [],
            "metadata": {
                "source": "planner_agent",
                "raw_response": llm_response
            }
        } 