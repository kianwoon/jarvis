"""
Workflow Prompt Generator
========================

Dynamically generates agent system prompts with correct tool information
based on workflow configuration, eliminating hardcoded tool references.
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class WorkflowPromptGenerator:
    """Generates dynamic agent prompts based on workflow tool configuration"""
    
    def __init__(self):
        self._tool_descriptions = self._load_tool_descriptions()
    
    def _load_tool_descriptions(self) -> Dict[str, str]:
        """Load tool descriptions from MCP cache and known tools"""
        descriptions = {
            # RAG tool
            "rag_knowledge_search": "Search and retrieve information from the internal knowledge base using vector and keyword search",
            
            # Email tools
            "find_email": "Search and locate specific emails in your inbox",
            "read_email": "Read and analyze email content and attachments",
            "gmail_send": "Send emails via Gmail",
            "search_emails": "Search through emails with specific criteria",
            "draft_email": "Create email drafts",
            "delete_email": "Delete emails from inbox",
            "list_email_labels": "List available email labels/folders",
            "modify_email": "Modify email properties like labels or status",
            
            # Search tools
            "google_search": "Search the web using Google to find current information",
            "web_search": "Search the internet for relevant information",
            
            # Utility tools
            "get_datetime": "Get current date and time information",
            "file_operations": "Perform file system operations like read/write/delete",
            
            # Default fallback
            "unknown": "Execute specialized tasks (description not available)"
        }
        
        # Try to load from MCP tools cache
        try:
            from app.core.mcp_tools_cache import get_enabled_mcp_tools
            mcp_tools = get_enabled_mcp_tools()
            
            for tool_name, tool_info in mcp_tools.items():
                if tool_info.get("description"):
                    descriptions[tool_name] = tool_info["description"]
                    
            logger.info(f"Loaded {len(mcp_tools)} tool descriptions from MCP cache")
            
        except Exception as e:
            logger.warning(f"Could not load MCP tools cache: {e}")
        
        return descriptions
    
    def get_tool_description(self, tool_name: str) -> str:
        """Get description for a specific tool"""
        return self._tool_descriptions.get(tool_name, f"{tool_name}: Execute specialized tasks")
    
    def generate_tools_section(self, tools: List[str]) -> str:
        """Generate the Available Tools & Usage section based on workflow tools"""
        if not tools:
            return """## Available Tools & Usage
You have no specific tools available for this task. Use your knowledge and reasoning to provide the best response."""
        
        tools_section = "## Available Tools & Usage\nYou have access to the following tools to enhance your capabilities:\n"
        
        for tool in tools:
            description = self.get_tool_description(tool)
            tools_section += f"- **{tool}**: {description}\n"
        
        tools_section += """
### Tool Usage Guidelines:
- **Always consider using tools** when they can improve your response quality
- **Combine multiple tools** when tackling complex tasks
- **Explain your tool choices** when using them in your analysis
- **Leverage tools for real-time data** rather than relying on potentially outdated knowledge

### Important: RAG Knowledge Search Tool Usage and Response Interpretation
When using the **rag_knowledge_search** tool:

**Tool Usage:**
- **IMPORTANT**: Do NOT specify max_documents parameter unless you need a specific value different from the default
- The tool defaults to max_documents=8 from RAG configuration
- Only include max_documents in your tool call if you specifically need more or fewer documents
- Example: Use {"query": "your query", "include_content": true} instead of {"query": "your query", "max_documents": 5, "include_content": true}

**Response Interpretation:**
1. The tool returns a structured response with 'success' and 'result' fields
2. Check result.success to verify the search succeeded
3. Check result.documents_found for the number of documents retrieved
4. Access the actual documents in result.documents array
5. Each document contains 'title', 'content', 'score', and 'metadata'
6. **If result.documents_found > 0, documents WERE successfully retrieved**
7. **Never claim "zero results" or "no documents found" if result.documents_found > 0**
8. Always summarize the key information from the retrieved documents"""
        
        return tools_section
    
    def generate_dynamic_prompt(
        self, 
        base_prompt: str, 
        workflow_tools: List[str],
        role: str = "",
        custom_instructions: str = ""
    ) -> str:
        """
        Generate a complete system prompt with dynamic tool information
        
        Args:
            base_prompt: Base system prompt from agent or workflow
            workflow_tools: List of tools available in this workflow
            role: Agent role for context
            custom_instructions: Additional custom instructions
        """
        # Remove hardcoded tool references from base prompt
        cleaned_prompt = self._remove_hardcoded_tools_section(base_prompt)
        
        # Generate dynamic tools section
        dynamic_tools_section = self.generate_tools_section(workflow_tools)
        
        # Build complete prompt
        prompt_parts = []
        
        if cleaned_prompt:
            prompt_parts.append(cleaned_prompt)
        
        if custom_instructions:
            prompt_parts.append(f"## Additional Instructions\n{custom_instructions}")
        
        # Always add the dynamic tools section
        prompt_parts.append(dynamic_tools_section)
        
        # Add standard output requirements if not already present
        if "Output Requirements" not in "\n".join(prompt_parts):
            prompt_parts.append(self._get_standard_output_requirements())
        
        final_prompt = "\n\n".join(prompt_parts)
        
        logger.info(f"Generated dynamic prompt with {len(workflow_tools)} tools for role: {role}")
        logger.debug(f"Workflow tools: {workflow_tools}")
        
        return final_prompt
    
    def _remove_hardcoded_tools_section(self, prompt: str) -> str:
        """Remove hardcoded 'Available Tools & Usage' section from existing prompt"""
        lines = prompt.split('\n')
        cleaned_lines = []
        skip_section = False
        
        for line in lines:
            # Start skipping when we hit the hardcoded tools section
            if "## Available Tools & Usage" in line or "Available Tools & Usage" in line:
                skip_section = True
                continue
            
            # Stop skipping when we hit the next major section
            if skip_section and line.startswith("## ") and "Available Tools" not in line:
                skip_section = False
            
            # Stop skipping if we hit specific section markers
            if skip_section and any(marker in line for marker in ["## Output Requirements", "## Quality Standards", "## Success Metrics"]):
                skip_section = False
            
            if not skip_section:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def _get_standard_output_requirements(self) -> str:
        """Get standard output requirements section"""
        return """## Output Requirements
- Provide **comprehensive, actionable insights** relevant to your role
- Structure responses with **clear headings and bullet points**
- Include **specific recommendations** with rationale
- **Cite sources** when using tool-retrieved information
- Maintain **professional tone** appropriate for the context

## Quality Standards
- **Accuracy**: Verify information using available tools when possible
- **Relevance**: Focus on insights that advance the overall objective
- **Clarity**: Use clear, professional language without unnecessary jargon
- **Completeness**: Address all aspects of the question within your expertise"""


# Global instance
_prompt_generator = None

def get_workflow_prompt_generator() -> WorkflowPromptGenerator:
    """Get the global workflow prompt generator instance"""
    global _prompt_generator
    if _prompt_generator is None:
        _prompt_generator = WorkflowPromptGenerator()
    return _prompt_generator

def generate_workflow_agent_prompt(
    agent_name: str,
    workflow_tools: List[str], 
    base_system_prompt: str = "",
    role: str = "",
    custom_prompt: str = ""
) -> str:
    """
    Generate a dynamic agent prompt for workflow execution
    
    Args:
        agent_name: Name of the agent
        workflow_tools: List of tools available in the workflow
        base_system_prompt: Base system prompt from database
        role: Agent role
        custom_prompt: Custom workflow instructions
        
    Returns:
        Complete system prompt with dynamic tool information
    """
    generator = get_workflow_prompt_generator()
    
    # Use custom prompt if provided, otherwise fall back to base system prompt
    base_prompt = custom_prompt if custom_prompt else base_system_prompt
    
    return generator.generate_dynamic_prompt(
        base_prompt=base_prompt,
        workflow_tools=workflow_tools,
        role=role,
        custom_instructions=""
    )