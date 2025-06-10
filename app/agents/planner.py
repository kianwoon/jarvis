from typing import Any, Dict, List
from datetime import datetime
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
        import re
        
        sections = []
        current_section = None
        current_subsections = []
        
        # Split response into lines
        lines = llm_response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for main section (1., 2., etc. or I., II., etc.)
            main_section_match = re.match(r'^(\d+\.|[IVX]+\.)\s+(.+)', line)
            if main_section_match:
                # Save previous section if exists
                if current_section:
                    current_section['subsections'] = current_subsections
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    'title': main_section_match.group(2).strip(),
                    'number': main_section_match.group(1).strip('.'),
                    'subsections': []
                }
                current_subsections = []
                continue
            
            # Check for subsection (a., b., or - or •)
            subsection_match = re.match(r'^([a-z]\.|[-•])\s+(.+)', line)
            if subsection_match and current_section:
                current_subsections.append({
                    'title': subsection_match.group(2).strip(),
                    'marker': subsection_match.group(1)
                })
                continue
            
            # Check for key points (indented or starting with *)
            if (line.startswith('  ') or line.startswith('*')) and current_section:
                # Add as a key point to the last subsection or section
                point = line.strip(' *')
                if current_subsections:
                    if 'key_points' not in current_subsections[-1]:
                        current_subsections[-1]['key_points'] = []
                    current_subsections[-1]['key_points'].append(point)
                else:
                    if 'key_points' not in current_section:
                        current_section['key_points'] = []
                    current_section['key_points'].append(point)
        
        # Don't forget the last section
        if current_section:
            current_section['subsections'] = current_subsections
            sections.append(current_section)
        
        # Extract overall structure info
        outline = {
            "sections": sections,
            "section_count": len(sections),
            "total_subsections": sum(len(s.get('subsections', [])) for s in sections),
            "metadata": {
                "source": "planner_agent",
                "parsed_at": datetime.now().isoformat(),
                "raw_response": llm_response
            }
        }
        
        # Try to extract document type or focus area
        first_line = lines[0] if lines else ""
        if "outline" in first_line.lower():
            outline["document_type"] = "outline"
        elif "plan" in first_line.lower():
            outline["document_type"] = "plan"
        else:
            outline["document_type"] = "structured_response"
            
        return outline 