"""
Simplified Multi-Agent System without LangGraph dependency
Implements router-based communication between specialized agents
"""

from typing import Dict, List, Any, Optional, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from app.core.langgraph_agents_cache import get_langgraph_agents, get_agent_by_role, get_active_agents
from app.core.mcp_tools_cache import get_enabled_mcp_tools
from app.core.llm_settings_cache import get_llm_settings
import json
import uuid
from datetime import datetime
import asyncio
import httpx
import hashlib
# Import rag_answer lazily to avoid circular import

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
    """Simplified orchestration for multiple specialized agents"""
    
    def __init__(self, conversation_id: Optional[str] = None):
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.agents = get_langgraph_agents()
        self.llm_settings = get_llm_settings()
        self.routing_cache = {}  # Simple in-memory cache for routing decisions
        
        # Agent avatars/emojis for better UI
        self.agent_avatars = {
            "router": "🧭",
            "document_researcher": "📚",
            "tool_executor": "🔧",
            "context_manager": "🧠",
            "sales_strategist": "💼",
            "technical_architect": "🏗️",
            "financial_analyst": "💰",
            "service_delivery_manager": "📋",
            "synthesizer": "🎯"
        }
    
    def _remove_thinking_tags(self, text: str) -> str:
        """Remove thinking tags from LLM response"""
        import re
        # Remove <think>...</think> tags and their content
        pattern = r'<think>.*?</think>'
        cleaned = re.sub(pattern, '', text, flags=re.DOTALL)
        # Also remove any standalone tags
        cleaned = re.sub(r'</?think>', '', cleaned)
        return cleaned.strip()
    
    def _clean_markdown_formatting(self, text: str) -> str:
        """Clean up excessive markdown formatting"""
        import re
        # Remove standalone "---" dividers
        text = re.sub(r'^---\s*$', '', text, flags=re.MULTILINE)
        # Remove "**" bold markers
        text = re.sub(r'\*\*', '', text)
        # Clean up multiple consecutive newlines (more than 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
        
    def _extract_query_context(self, query: str) -> Dict[str, Any]:
        """Extract key information from the query"""
        context = {
            "client": "Unknown",
            "requirement": "Unknown",
            "current_model": "Unknown",
            "proposed_model": "Unknown",
            "details": query
        }
        
        # Extract client name
        import re
        client_match = re.search(r'(?:client|customer)[,:\s]+([A-Z][A-Za-z0-9\s&]+?)(?:\s+request|\s+is|\s+wants|,|\.)', query, re.IGNORECASE)
        if client_match:
            context["client"] = client_match.group(1).strip()
        
        # Extract requirement (e.g., "3 x system engineers")
        req_match = re.search(r'(\d+\s*x?\s*(?:L1|L2|L3|system|network|database|security)?\s*(?:engineer|admin|analyst|developer|architect)s?)', query, re.IGNORECASE)
        if req_match:
            context["requirement"] = req_match.group(1).strip()
        
        # Identify current and proposed models
        if any(term in query.lower() for term in ["t&m", "time and material", "time & material"]):
            context["current_model"] = "Time & Material (T&M)"
        if any(term in query.lower() for term in ["managed service", "managed services", "msp"]):
            context["proposed_model"] = "Managed Services"
            
        return context
    
    async def _call_llm_stream(self, prompt: str, agent_name: str, temperature: float = 0.7, timeout: int = 30):
        """Call LLM and return only the final cleaned response as JSON event"""
        try:
            print(f"[DEBUG] *** {agent_name}: GENERATING RESPONSE ***")
            
            # Import LLM service directly to avoid HTTP recursion
            from app.llm.ollama import OllamaLLM
            from app.llm.base import LLMConfig
            import os
            
            # Use thinking mode settings if available
            model_config = self.llm_settings.get("thinking_mode", {})
            
            # Create LLM config
            config = LLMConfig(
                model_name=model_config.get("model", "qwen3:30b-a3b"),
                temperature=temperature,
                top_p=model_config.get("top_p", 0.9),
                max_tokens=model_config.get("max_tokens", 2000)
            )
            
            # Ollama base URL
            ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            print(f"[DEBUG] {agent_name}: Using model {config.model_name} at {ollama_url}")
            
            # Create LLM instance with base URL
            llm = OllamaLLM(config, base_url=ollama_url)
            
            # Accumulate complete response without streaming individual tokens
            response_text = ""
            print(f"[DEBUG] {agent_name}: Starting to generate complete response")
            
            async for response_chunk in llm.generate_stream(prompt):
                response_text += response_chunk.text
            
            print(f"[DEBUG] {agent_name}: Completed generation, response length = {len(response_text)}")
            
            # Clean the response and yield only one final event
            cleaned_content = self._remove_thinking_tags(response_text)
            cleaned_content = self._clean_markdown_formatting(cleaned_content)
            display_content = cleaned_content.strip()
            
            if display_content:  # Only yield if there's actual content
                print(f"[DEBUG] {agent_name}: Yielding final response event")
                event = {
                    "type": "agent_complete",
                    "agent": agent_name,
                    "content": display_content,
                    "avatar": self.agent_avatars.get(agent_name, "🤖")
                }
                yield event
            else:
                print(f"[DEBUG] {agent_name}: No content to yield after cleaning")
            
        except Exception as e:
            print(f"[ERROR] LLM call failed for {agent_name}: {str(e)}")
            yield {
                "type": "agent_complete",
                "agent": agent_name,
                "content": f"I encountered an error while processing your request: {str(e)}"
            }
        
    async def _router_agent(self, query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Hybrid LLM-based routing with keyword fallback"""
        print(f"[DEBUG] Starting routing for query: {query[:100]}...")
        
        # For now, use keyword-based routing (LLM routing has cache dependency issues)
        print("[DEBUG] Using keyword-based routing")
        return await self._keyword_router_agent(query)
    
    async def _llm_router_agent(self, query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """LLM-based intelligent routing"""
        # Create cache key for this routing decision
        cache_key = hashlib.md5(f"{query}_{len(conversation_history or [])}".encode()).hexdigest()
        
        # Check cache first
        if cache_key in self.routing_cache:
            print(f"[DEBUG] Using cached routing decision for query")
            return self.routing_cache[cache_key]
        
        # Get all active agents from cache and filter to only implemented ones
        active_agents = get_active_agents()
        
        # Only include agents that have actual implementations
        implemented_agents = {
            "document_researcher": active_agents.get("document_researcher"),
            "tool_executor": active_agents.get("tool_executor"),
            "context_manager": active_agents.get("context_manager"),
            "sales_strategist": active_agents.get("sales_strategist"),
            "technical_architect": active_agents.get("technical_architect"),
            "financial_analyst": active_agents.get("financial_analyst"),
            "service_delivery_manager": active_agents.get("service_delivery_manager")
        }
        
        # Remove None values (agents not in cache)
        implemented_agents = {k: v for k, v in implemented_agents.items() if v is not None}
        
        if not implemented_agents:
            raise Exception("No implemented agents available in cache")
        
        # Build agent descriptions for the LLM using only implemented agents
        agent_descriptions = {}
        for name, agent in implemented_agents.items():
            agent_descriptions[name] = {
                "role": agent.get("role", ""),
                "description": agent.get("description", ""),
                "tools": agent.get("tools", [])
            }
        
        # Add context from conversation history if available
        history_context = ""
        if conversation_history:
            recent_messages = conversation_history[-3:]  # Last 3 messages for context
            history_context = f"\nRecent conversation context:\n{json.dumps(recent_messages, indent=2)}\n"
        
        routing_prompt = f"""You are an intelligent agent router. Your task is to select the most appropriate agents to handle a user query.

Available agents:
{json.dumps(agent_descriptions, indent=2)}

{history_context}
User query: "{query}"

Instructions:
1. Select 1-4 most relevant agents that can best handle this query
2. Consider the query type, complexity, and required capabilities
3. Prioritize agents whose roles and tools match the query requirements
4. Provide clear reasoning for your selections

Respond ONLY in this JSON format:
{{
    "agents": ["agent_name1", "agent_name2"],
    "reasoning": "Brief explanation of why these agents were selected"
}}

Important: Only use agent names that exist in the available agents list above."""

        # Make a quick LLM call for routing with timeout
        try:
            async for chunk in self._call_llm_stream(routing_prompt, "router", temperature=0.3, timeout=8):
                if chunk.get("type") == "agent_complete":
                    response_text = chunk.get("content", "")
                    # Parse JSON response
                    try:
                        # Extract JSON from response
                        import re
                        json_match = re.search(r'\{[^}]*"agents"[^}]*\}', response_text, re.DOTALL)
                        if json_match:
                            routing_data = json.loads(json_match.group())
                            
                            # Validate agents exist and are implemented
                            valid_agents = []
                            for agent in routing_data.get("agents", []):
                                if agent in implemented_agents:
                                    valid_agents.append(agent)
                            
                            if valid_agents:
                                result = {
                                    "agents": valid_agents,
                                    "reasoning": routing_data.get("reasoning", "LLM-based routing")
                                }
                                # Cache the result
                                self.routing_cache[cache_key] = result
                                return result
                        
                        raise Exception("Invalid JSON format in LLM response")
                        
                    except json.JSONDecodeError as e:
                        raise Exception(f"Failed to parse LLM routing response: {e}")
                        
        except Exception as e:
            raise Exception(f"LLM routing call failed: {e}")
        
        raise Exception("No complete response received from LLM router")
    
    async def _keyword_router_agent(self, query: str) -> Dict[str, Any]:
        """Keyword-based routing fallback (original logic)"""
        routing = {
            "agents": [],
            "reasoning": ""
        }
        
        query_lower = query.lower()
        
        # For a proposal discussion, we want multiple perspectives
        if any(word in query_lower for word in ["proposal", "client", "discuss", "counter"]):
            routing["agents"] = ["sales_strategist", "technical_architect", "financial_analyst", "service_delivery_manager"]
            routing["reasoning"] = "Keyword-based routing: Strategic proposal discussion detected."
        else:
            # Original routing logic for other queries
            if any(word in query_lower for word in ["document", "file", "pdf", "information", "data"]):
                routing["agents"].append("document_researcher")
                routing["reasoning"] += "Document research needed. "
                
            if any(word in query_lower for word in ["calculate", "compute", "execute", "run", "tool"]):
                routing["agents"].append("tool_executor")
                routing["reasoning"] += "Tool execution may be required. "
                
            if any(word in query_lower for word in ["history", "context", "previous", "earlier", "remember"]):
                routing["agents"].append("context_manager")
                routing["reasoning"] += "Historical context referenced. "
        
        # Default if no specific routing
        if not routing["agents"]:
            routing["agents"] = ["document_researcher"]
            routing["reasoning"] = "Keyword-based fallback: Defaulting to document research."
            
        routing["reasoning"] = f"Keyword-based routing: {routing['reasoning']}"
        return routing
    
    async def _document_researcher_agent(self, state: AgentState) -> Dict[str, Any]:
        """Use RAG to research documents"""
        try:
            # Use the existing RAG system
            response = ""
            documents = []
            has_content = False
            
            print(f"[DEBUG] Document researcher processing query: {state['query']}")
            
            # Stream RAG response
            async for chunk in self._rag_stream(state["query"]):
                if isinstance(chunk, dict):
                    if "token" in chunk:
                        response += chunk["token"]
                        has_content = True
                    elif "answer" in chunk:
                        # Handle complete answer
                        response = chunk["answer"]
                        has_content = True
                    elif "source" in chunk and chunk["source"]:
                        documents.append({
                            "source": chunk["source"],
                            "timestamp": datetime.now().isoformat()
                        })
                    elif "error" in chunk:
                        # Handle error from RAG stream
                        return {
                            "response": f"RAG system error: {chunk['error']}. The system may need to have settings configured properly in the Settings page.",
                            "documents": []
                        }
            
            print(f"[DEBUG] Document researcher - Response length: {len(response)}, Has content: {has_content}")
            
            # If no response from RAG, provide a more helpful message
            if not response or not has_content:
                response = (
                    "No documents found in the knowledge base for your query. "
                    "This could mean:\n"
                    "1. No relevant documents have been uploaded yet\n"
                    "2. The search terms didn't match any indexed content\n"
                    "3. The vector database needs to be populated with relevant documents\n\n"
                    "For your query about OCBC bank and managed services, you may need to:\n"
                    "- Upload relevant proposal documents or service catalogs\n"
                    "- Upload documents about L1 system engineer roles and responsibilities\n"
                    "- Upload managed service frameworks or templates"
                )
            
            return {
                "response": response,
                "documents": documents
            }
        except Exception as e:
            import traceback
            print(f"[ERROR] Document researcher exception: {str(e)}")
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            return {
                "response": f"Error in document research: {str(e)}. Please ensure the vector database and embedding settings are properly configured.",
                "documents": []
            }
    
    async def _sales_strategist_agent(self, state: AgentState):
        """Sales perspective on the proposal - streaming version"""
        # Extract context from query
        context = self._extract_query_context(state["query"])
        
        prompt = f"""You are an experienced Sales Strategist specializing in IT services and managed service proposals.

Context:
- Client: {context['client']}
- Current Requirement: {context['requirement']}
- Current Model: {context['current_model']}
- Proposed Model: {context['proposed_model']}
- Details: {context['details']}

Provide strategic sales advice for converting this requirement from {context['current_model']} to {context['proposed_model']}.

Your response should include:
1. Value Proposition - Why managed services is better for this specific client
2. Key Selling Points - 3-4 compelling benefits specific to their industry
3. ROI Justification - Financial and operational benefits
4. Objection Handling - Anticipate and address 2-3 likely concerns
5. Next Steps - Concrete action items to move the deal forward

Be specific to {context['client']} and their industry. Use persuasive but professional language."""

        async for event in self._call_llm_stream(prompt, "sales_strategist", temperature=0.7, timeout=20):
            yield event
    
    async def _technical_architect_agent(self, state: AgentState):
        """Technical perspective on the proposal - streaming version"""
        context = self._extract_query_context(state["query"])
        
        prompt = f"""You are a Senior Technical Architect with expertise in IT infrastructure and managed services.

Context:
- Client: {context['client']}
- Requirement: {context['requirement']}
- Moving from: {context['current_model']} to {context['proposed_model']}

Provide a technical analysis of how managed services will better serve {context['client']}'s needs.

Your response should cover:
1. Technical Architecture - How the managed service will be structured
2. Technology Stack - Tools and platforms we'll bring to enhance their operations
3. Service Levels - Specific SLAs and KPIs appropriate for a financial institution
4. Security & Compliance - How we'll meet banking sector requirements
5. Integration Strategy - How we'll integrate with their existing systems
6. Automation Opportunities - Where we can add efficiency through automation

Focus on technical excellence and reliability that banks require. Be specific about technologies and metrics."""

        async for event in self._call_llm_stream(prompt, "technical_architect", temperature=0.6, timeout=20):
            yield event
    
    async def _financial_analyst_agent(self, state: AgentState):
        """Financial perspective on the proposal - streaming version"""
        context = self._extract_query_context(state["query"])
        
        prompt = f"""You are a Financial Analyst specializing in IT services cost modeling and ROI analysis.

Context:
- Client: {context['client']} (Banking sector)
- Current Requirement: {context['requirement']}
- Comparing: {context['current_model']} vs {context['proposed_model']}

Provide a detailed financial analysis comparing the total cost of ownership between hiring {context['requirement']} directly versus our managed service model.

Your analysis should include:
1. Direct Costs Comparison
   - Salaries, benefits, overhead for direct hires
   - Managed service fees structure
   
2. Hidden Costs Analysis
   - Recruitment, training, attrition costs
   - Management overhead
   - Technology and tool licenses
   
3. ROI Calculation
   - Year 1, Year 3, and Year 5 projections
   - Break-even analysis
   - Risk-adjusted returns
   
4. Value Drivers
   - Quantify productivity gains
   - Calculate cost of downtime reduction
   - Estimate value of scalability
   
5. Pricing Models
   - Recommend optimal pricing structure for {context['client']}
   - Show flexibility options

Use realistic market rates for Singapore/APAC region. Present numbers clearly with executive summary."""

        async for event in self._call_llm_stream(prompt, "financial_analyst", temperature=0.5, timeout=20):
            yield event
    
    async def _service_delivery_manager_agent(self, state: AgentState):
        """Service delivery perspective on the proposal - streaming version"""
        context = self._extract_query_context(state["query"])
        
        prompt = f"""You are a Service Delivery Manager with 15+ years experience in managing IT services for financial institutions.

Context:
- Client: {context['client']}
- Service Requirement: {context['requirement']}
- Service Model: Transitioning to {context['proposed_model']}

Design a comprehensive service delivery plan that ensures smooth operations and exceeds {context['client']}'s expectations.

Your plan should detail:
1. Governance Model
   - Account management structure
   - Meeting cadences and stakeholders
   - Escalation procedures
   
2. Transition Plan
   - Detailed week-by-week transition roadmap
   - Knowledge transfer approach
   - Risk mitigation during transition
   
3. Service Level Agreement (SLA)
   - Response and resolution times by priority
   - Availability guarantees
   - Penalty and bonus structures
   
4. Team Structure
   - Staffing model with roles and shifts
   - Skills matrix required
   - Training and certification plans
   
5. Quality Framework
   - Performance metrics and dashboards
   - Continuous improvement process
   - Customer satisfaction measurement
   
6. Innovation Roadmap
   - Value-add services timeline
   - Automation implementation plan
   - Technology refresh cycles

Remember {context['client']} is a major bank requiring highest service standards. Be specific and actionable."""

        async for event in self._call_llm_stream(prompt, "service_delivery_manager", temperature=0.6, timeout=20):
            yield event
    
    async def _tool_executor_agent(self, state: AgentState) -> Dict[str, Any]:
        """Execute tools based on query"""
        # For now, return a placeholder
        return {
            "response": "Tool execution capability is being developed.",
            "tools_used": []
        }
    
    async def _context_manager_agent(self, state: AgentState) -> Dict[str, Any]:
        """Manage conversation context"""
        # Simple context summary
        context_summary = f"Conversation ID: {state['conversation_id']}\n"
        context_summary += f"Message count: {len(state['messages'])}\n"
        
        return {
            "response": context_summary,
            "context": {
                "conversation_id": state["conversation_id"],
                "message_count": len(state["messages"])
            }
        }
    
    async def _synthesizer_agent(self, state: AgentState) -> str:
        """Synthesize final response from all agent outputs"""
        final_parts = []
        
        # Check if we have any meaningful responses
        has_meaningful_response = False
        for output in state["agent_outputs"].values():
            if output.get("response") and len(output["response"]) > 20:
                has_meaningful_response = True
                break
        
        # Skip the fallback message if we have specialized agents responding
        has_specialized_agents = any(
            agent in state["agent_outputs"] 
            for agent in ["sales_strategist", "technical_architect", "financial_analyst", "service_delivery_manager"]
        )
        
        if not has_meaningful_response and not has_specialized_agents:
            # No meaningful responses, provide a helpful default
            return (
                "I couldn't find specific information in the knowledge base to help with your query.\n\n"
                "Please try:\n"
                "• Uploading relevant documents to the knowledge base\n"
                "• Asking a more specific question\n"
                "• Using the standard chat mode for general inquiries"
            )
        
        # For specialized agents, create a well-formatted comprehensive response
        if has_specialized_agents:
            final_parts.append("Based on the multi-agent analysis, here's a comprehensive response:\n")
            
            # Add each specialized agent's response with proper formatting
            agent_order = ["sales_strategist", "technical_architect", "financial_analyst", "service_delivery_manager"]
            for agent_name in agent_order:
                if agent_name in state["agent_outputs"]:
                    output = state["agent_outputs"][agent_name]
                    if output.get("response"):
                        agent_title = agent_name.replace('_', ' ').title()
                        # Use markdown headers for better structure
                        final_parts.append(f"## {agent_title} Perspective\n")
                        final_parts.append(output["response"])
                        final_parts.append("")  # Add blank line for spacing
        else:
            # Standard routing with icons
            if state["routing_decision"] and state["routing_decision"].get("reasoning"):
                final_parts.append(f"📊 **Analysis**: {state['routing_decision']['reasoning']}\n")
            
            # Add agent responses
            for agent_name, output in state["agent_outputs"].items():
                if output.get("response"):
                    agent_title = agent_name.replace('_', ' ').title()
                    icon = "📄" if "document" in agent_name else "🔧" if "tool" in agent_name else "🧠"
                    final_parts.append(f"{icon} **{agent_title}**:\n")
                    final_parts.append(output['response'])
                    final_parts.append("")  # Add blank line for spacing
        
        # Add document sources if any
        all_docs = []
        for output in state["agent_outputs"].values():
            if output.get("documents"):
                all_docs.extend(output["documents"])
        
        if all_docs:
            final_parts.append("\n---\n")  # Horizontal rule
            final_parts.append("📚 **Sources**:")
            for doc in all_docs[:5]:  # Show up to 5 sources
                final_parts.append(f"- {doc.get('source', 'Unknown source')}")
        
        return "\n".join(final_parts) if final_parts else "No response generated."
    
    async def _rag_stream(self, query: str):
        """Stream RAG responses"""
        # Import lazily to avoid circular import
        from app.langchain.service import rag_answer
        import json
        # Use the existing rag_answer function
        try:
            for chunk in rag_answer(query, thinking=False, stream=True, conversation_id=self.conversation_id):
                # rag_answer yields JSON strings, we need to parse them
                if isinstance(chunk, str):
                    try:
                        # Remove trailing newline and parse JSON
                        chunk_data = json.loads(chunk.strip())
                        yield chunk_data
                    except json.JSONDecodeError:
                        # If it's not valid JSON, log and skip
                        print(f"[WARNING] Could not parse chunk as JSON: {chunk[:100]}...")
                        # Don't yield unparseable chunks
                        continue
                else:
                    # Already a dict, yield as-is
                    yield chunk
        except Exception as e:
            print(f"[ERROR] RAG stream error: {str(e)}")
            yield {"error": str(e)}
    
    async def stream_events(
        self, 
        query: str, 
        selected_agents: Optional[List[str]] = None,
        max_iterations: int = 10,
        conversation_history: Optional[List[Dict]] = None
    ):
        """Stream events from multi-agent processing"""
        try:
            # Initialize state
            state = AgentState(
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
            
            # Step 1: Router
            yield {
                "type": "agent_start",
                "agent": "router",
                "content": "Analyzing query...",
                "avatar": self.agent_avatars.get("router", "🧭")
            }
            
            routing = await self._router_agent(query, conversation_history)
            state["routing_decision"] = routing
            
            yield {
                "type": "agent_complete",
                "agent": "router",
                "content": routing["reasoning"],
                "routing": routing,
                "avatar": self.agent_avatars.get("router", "🧭")
            }
            
            # Step 2: Execute selected agents in parallel with streaming
            agents_to_run = selected_agents or routing["agents"]
            print(f"[DEBUG] Agents to run: {agents_to_run}")
            
            # Create async tasks for all agents
            agent_tasks = []
            for agent_name in agents_to_run:
                yield {
                    "type": "agent_start",
                    "agent": agent_name,
                    "content": f"Starting {agent_name}...",
                    "avatar": self.agent_avatars.get(agent_name, "🤖")
                }
                
                # Create async generator for each agent
                if agent_name == "sales_strategist":
                    print(f"[DEBUG] Adding streaming agent: {agent_name}")
                    agent_tasks.append((agent_name, self._sales_strategist_agent(state)))
                elif agent_name == "technical_architect":
                    print(f"[DEBUG] Adding streaming agent: {agent_name}")
                    agent_tasks.append((agent_name, self._technical_architect_agent(state)))
                elif agent_name == "financial_analyst":
                    print(f"[DEBUG] Adding streaming agent: {agent_name}")
                    agent_tasks.append((agent_name, self._financial_analyst_agent(state)))
                elif agent_name == "service_delivery_manager":
                    print(f"[DEBUG] Adding streaming agent: {agent_name}")
                    agent_tasks.append((agent_name, self._service_delivery_manager_agent(state)))
                elif agent_name == "document_researcher":
                    print(f"[DEBUG] Executing non-streaming agent: {agent_name}")
                    # Handle non-streaming agents
                    result = await self._document_researcher_agent(state)
                    state["agent_outputs"][agent_name] = result
                    yield {
                        "type": "agent_complete",
                        "agent": agent_name,
                        "content": result.get("response", ""),
                        "avatar": self.agent_avatars.get(agent_name, "🤖")
                    }
                elif agent_name == "tool_executor":
                    result = await self._tool_executor_agent(state)
                    state["agent_outputs"][agent_name] = result
                    yield {
                        "type": "agent_complete", 
                        "agent": agent_name,
                        "content": result.get("response", ""),
                        "avatar": self.agent_avatars.get(agent_name, "🤖")
                    }
                elif agent_name == "context_manager":
                    result = await self._context_manager_agent(state)
                    state["agent_outputs"][agent_name] = result
                    yield {
                        "type": "agent_complete",
                        "agent": agent_name, 
                        "content": result.get("response", ""),
                        "avatar": self.agent_avatars.get(agent_name, "🤖")
                    }
                else:
                    # Handle unimplemented agents
                    print(f"[WARNING] Agent '{agent_name}' not implemented, skipping")
                    yield {
                        "type": "agent_complete",
                        "agent": agent_name,
                        "content": f"Agent '{agent_name}' is not yet implemented. Please check the agent configuration.",
                        "avatar": self.agent_avatars.get(agent_name, "🤖")
                    }
            
            # Process streaming agents concurrently
            if agent_tasks:
                import asyncio
                
                # Process agent streams - moved inside merge_agent_streams to avoid unused warning
                
                # Run all agents concurrently and merge streams
                async def merge_agent_streams():
                    # Create queue for collecting events from all agents
                    import asyncio
                    from asyncio import Queue
                    
                    event_queue = Queue()
                    active_tasks = []
                    
                    async def agent_worker(name, gen):
                        try:
                            async for event in gen:
                                if not isinstance(event, dict):
                                    print(f"[ERROR] Agent {name} yielded non-dict: {type(event)} - {str(event)[:100]}")
                                    # Skip non-dict events
                                    continue
                                await event_queue.put(event)
                        except Exception as e:
                            await event_queue.put({
                                "type": "error",
                                "agent": name,
                                "error": str(e)
                            })
                        finally:
                            await event_queue.put({"type": "agent_done", "agent": name})
                    
                    # Start all agent tasks
                    for name, gen in agent_tasks:
                        task = asyncio.create_task(agent_worker(name, gen))
                        active_tasks.append(task)
                    
                    # Track completed agents
                    completed_agents = 0
                    total_agents = len(agent_tasks)
                    
                    # Process events from queue
                    while completed_agents < total_agents:
                        event = await event_queue.get()
                        
                        if event.get("type") == "agent_done":
                            completed_agents += 1
                        else:
                            yield event
                    
                    # Wait for all tasks to complete
                    await asyncio.gather(*active_tasks, return_exceptions=True)
                
                # Stream events from all agents
                async for event in merge_agent_streams():
                    if not isinstance(event, dict):
                        print(f"[ERROR] merge_agent_streams yielded non-dict: {type(event)} - {event}")
                        # Skip non-dict events
                        continue
                    
                    # Capture agent responses in state for synthesizer
                    if event.get("type") == "agent_complete" and event.get("agent") and event.get("content"):
                        agent_name = event["agent"]
                        if agent_name not in state["agent_outputs"]:
                            state["agent_outputs"][agent_name] = {}
                        state["agent_outputs"][agent_name]["response"] = event["content"]
                    
                    yield event
            
            # Step 3: Synthesize
            yield {
                "type": "agent_start",
                "agent": "synthesizer",
                "content": "Synthesizing final response...",
                "avatar": self.agent_avatars.get("synthesizer", "🎯")
            }
            
            final_response = await self._synthesizer_agent(state)
            state["final_response"] = final_response
            
            yield {
                "type": "final_response",
                "response": final_response,
                "metadata": state["metadata"],
                "routing": state["routing_decision"],
                "agent_outputs": state["agent_outputs"]
            }
            
        except Exception as e:
            print(f"[ERROR] Exception in stream_events: {str(e)}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            yield {
                "type": "error",
                "error": str(e)
            }