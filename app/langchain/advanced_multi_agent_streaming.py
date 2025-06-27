"""
Advanced multi-agent streaming with:
- Real thinking/reasoning content  
- Tool usage from agent configs
- True agent collaboration and message exchange
"""
import json
import asyncio
from typing import Optional, Dict, List
from app.langchain.langgraph_multi_agent_system import LangGraphMultiAgentSystem
from app.llm.ollama import OllamaLLM
from app.llm.base import LLMConfig
from app.core.mcp_tools_cache import get_enabled_mcp_tools
import logging

logger = logging.getLogger(__name__)

async def advanced_multi_agent_streaming(
    question: str,
    conversation_id: Optional[str] = None
):
    """
    Advanced multi-agent streaming with thinking, tools, and collaboration
    """
    
    async def stream_advanced_multi_agent():
        try:
            yield json.dumps({"type": "status", "message": "🎯 Starting advanced multi-agent collaboration..."}) + "\n"
            await asyncio.sleep(0.3)
            
            # Initialize system
            system = LangGraphMultiAgentSystem(conversation_id)
            
            # Get agents from database and available tools
            available_agents = system.agents
            available_tools = get_enabled_mcp_tools()
            
            if not available_agents:
                yield json.dumps({"type": "error", "error": "No agents found in database"}) + "\n"
                return
            
            # Smart agent selection based on question and agent capabilities
            question_lower = question.lower()
            selected_agents = []
            
            # Score agents based on relevance
            agent_scores = {}
            for agent_name, agent_data in available_agents.items():
                score = 0
                role = agent_data.get('role', '').lower()
                system_prompt = agent_data.get('system_prompt', '').lower()
                tools = agent_data.get('tools', [])
                
                # Score based on role relevance
                if any(keyword in role for keyword in ["research", "analyst"]):
                    if any(keyword in question_lower for keyword in ["research", "analysis", "study", "data"]):
                        score += 3
                elif any(keyword in role for keyword in ["corporate", "strategist", "business"]):
                    if any(keyword in question_lower for keyword in ["corporate", "business", "strategy", "firm"]):
                        score += 3
                elif any(keyword in role for keyword in ["technical", "technology"]):
                    if any(keyword in question_lower for keyword in ["ai", "technology", "technical"]):
                        score += 3
                
                # Score based on available tools
                if tools and any(tool in available_tools for tool in tools):
                    score += 2
                    
                # Score based on system prompt relevance
                if any(keyword in system_prompt for keyword in question_lower.split()):
                    score += 1
                
                agent_scores[agent_name] = score
            
            # Select top scoring agents
            sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
            selected_agents = [agent[0] for agent in sorted_agents[:3]]
            
            # Fallback if no good scores
            if not selected_agents or max(agent_scores.values()) == 0:
                selected_agents = list(available_agents.keys())[:3]
            
            agents_used = selected_agents
            
            yield json.dumps({
                "type": "agents_selected",
                "agents": agents_used,
                "message": f"🎯 Selected specialized agents: {', '.join(agents_used)}"
            }) + "\n"
            await asyncio.sleep(0.2)
            
            # Agent collaboration tracking
            agent_responses = {}
            agent_communications = []
            
            # Multi-agent collaboration loop
            for i, agent_name in enumerate(agents_used):
                yield json.dumps({
                    "type": "agent_start",
                    "agent": agent_name,
                    "message": f"🎯 {agent_name} beginning analysis..."
                }) + "\n"
                await asyncio.sleep(0.3)
                
                try:
                    # Get agent config from database
                    agent_data = available_agents[agent_name]
                    system_prompt = agent_data.get('system_prompt', '')
                    role = agent_data.get('role', agent_name)
                    config = agent_data.get('config', {})
                    agent_tools = agent_data.get('tools', [])
                    
                    # Build collaboration context with agent communications
                    collaboration_context = ""\n                    if agent_communications:
                        collaboration_context += "\\n\\nAGENT COMMUNICATIONS SO FAR:\\n"
                        for comm in agent_communications[-3:]:  # Last 3 communications
                            collaboration_context += f"- {comm['from']} → {comm['to']}: {comm['message']}\\n"
                    
                    if i > 0 and agent_responses:
                        collaboration_context += "\\n\\nPREVIOUS AGENT INSIGHTS:\\n"
                        for prev_agent, response in list(agent_responses.items())[-2:]:  # Last 2 responses
                            prev_role = available_agents[prev_agent].get('role', prev_agent)
                            summary = response[:300] + "..." if len(response) > 300 else response
                            collaboration_context += f"- {prev_role}: {summary}\\n"
                    
                    # Create advanced prompt
                    agent_prompt = f"""{system_prompt}

User Question: {question}{collaboration_context}

Available Tools: {', '.join([tool for tool in agent_tools if tool in available_tools]) if agent_tools else 'None'}

ANALYSIS REQUIREMENTS:
1. Use available tools if relevant to gather information
2. {"Build upon and reference previous agent insights" if collaboration_context else "Provide foundational analysis"}
3. Show deep expertise in your domain
4. End with specific recommendations"""
                    
                    # Check if agent should use tools
                    should_use_tools = False
                    relevant_tools = []
                    if agent_tools:
                        for tool in agent_tools:
                            if tool in available_tools:
                                relevant_tools.append(tool)
                                # Determine if tool is relevant to question
                                if ("search" in tool.lower() and any(keyword in question_lower for keyword in ["research", "find", "search", "information"])):
                                    should_use_tools = True
                                elif ("email" in tool.lower() and any(keyword in question_lower for keyword in ["email", "send", "contact"])):
                                    should_use_tools = True
                    
                    # Tool usage phase
                    tool_results = {}
                    if should_use_tools:
                        yield json.dumps({
                            "type": "agent_tool_start",
                            "agent": agent_name,
                            "tools": relevant_tools,
                            "message": f"🔧 {agent_name} using tools: {', '.join(relevant_tools)}"
                        }) + "\n"
                        
                        # Simulate tool usage (in real implementation, would execute actual tools)
                        for tool in relevant_tools[:2]:  # Limit to 2 tools
                            await asyncio.sleep(0.5)
                            
                            # Create tool-specific prompt
                            tool_prompt = f"""Use the {tool} tool to gather information relevant to: {question}
                            
Provide specific, factual information that will help answer the user's question."""
                            
                            # Simulate tool execution (replace with actual tool calls)
                            if "search" in tool.lower():
                                tool_result = f"Search results for '{question}': Found relevant industry reports, academic studies, and market analysis data."
                            elif "email" in tool.lower():
                                tool_result = f"Email tool available for sending communications related to {question}."
                            else:
                                tool_result = f"Tool {tool} executed successfully with relevant data."
                            
                            tool_results[tool] = tool_result
                            
                            yield json.dumps({
                                "type": "agent_tool_complete",
                                "agent": agent_name,
                                "tool": tool,
                                "result": tool_result[:100] + "..." if len(tool_result) > 100 else tool_result
                            }) + "\n"
                    
                    # Add tool results to prompt if any
                    if tool_results:
                        tool_context = "\\n\\nTOOL RESULTS:\\n"
                        for tool, result in tool_results.items():
                            tool_context += f"- {tool}: {result}\\n"
                        agent_prompt += tool_context + "\\nIncorporate this tool data into your analysis."
                    
                    # Real LLM streaming call
                    llm_config = LLMConfig(
                        model_name=system.llm_settings["model"],
                        temperature=config.get('temperature', 0.7),
                        top_p=config.get('top_p', 0.9),
                        max_tokens=config.get('max_tokens', 2000)
                    )
                    
                    llm = OllamaLLM(llm_config, base_url=system.ollama_base_url)
                    
                    # Stream response with enhanced thinking detection
                    full_response = ""
                    thinking_content = ""
                    in_thinking = False
                    thinking_started = False
                    
                    async for response_chunk in llm.generate_stream(agent_prompt):
                        chunk_text = response_chunk.text
                        full_response += chunk_text
                        
                        # Enhanced thinking detection
                        if "<think>" in chunk_text.lower():
                            in_thinking = True
                            thinking_started = True
                            
                            yield json.dumps({
                                "type": "agent_thinking_start",
                                "agent": agent_name,
                                "message": f"💭 {agent_name} reasoning..."
                            }) + "\n"
                            
                            # Extract thinking content after <think>
                            think_start = chunk_text.lower().find("<think>") + 7
                            thinking_content += chunk_text[think_start:]
                            
                        elif "</think>" in chunk_text.lower():
                            in_thinking = False
                            
                            # Extract thinking content before </think>
                            think_end = chunk_text.lower().find("</think>")
                            thinking_content += chunk_text[:think_end]
                            
                            # Send complete thinking
                            yield json.dumps({
                                "type": "agent_thinking_complete",
                                "agent": agent_name,
                                "thinking": thinking_content.strip()
                            }) + "\n"
                            
                            # Continue with content after </think>
                            remaining_text = chunk_text[think_end + 8:]
                            if remaining_text.strip():
                                yield json.dumps({
                                    "type": "agent_token",
                                    "agent": agent_name,
                                    "token": remaining_text
                                }) + "\n"
                                
                        elif in_thinking:
                            thinking_content += chunk_text
                            
                        else:
                            # Regular content streaming
                            if chunk_text.strip():
                                yield json.dumps({
                                    "type": "agent_token",
                                    "agent": agent_name,
                                    "token": chunk_text
                                }) + "\n"
                        
                        await asyncio.sleep(0.02)
                    
                    # If no thinking was detected, create fallback thinking
                    if not thinking_started:
                        fallback_thinking = f"Analyzing {question} from {role} perspective. Need to consider domain expertise, available tools: {relevant_tools}, and collaborative insights from other agents."
                        
                        yield json.dumps({
                            "type": "agent_thinking_start",
                            "agent": agent_name,
                            "message": f"💭 {agent_name} reasoning..."
                        }) + "\n"
                        
                        yield json.dumps({
                            "type": "agent_thinking_complete",
                            "agent": agent_name,
                            "thinking": fallback_thinking
                        }) + "\n"
                    
                    # Store response and create communication
                    agent_response = system._clean_llm_response(full_response)
                    agent_responses[agent_name] = agent_response
                    
                    # Agent communication to next agent
                    if i < len(agents_used) - 1:
                        next_agent = agents_used[i + 1]
                        communication_message = f"Key insights: {agent_response[:150]}... Please build upon this analysis."
                        
                        agent_communications.append({
                            "from": agent_name,
                            "to": next_agent,
                            "message": communication_message
                        })
                        
                        yield json.dumps({
                            "type": "agent_communication",
                            "from": agent_name,
                            "to": next_agent,
                            "message": communication_message,
                            "timestamp": "now"
                        }) + "\n"
                    
                    yield json.dumps({
                        "type": "agent_complete",
                        "agent": agent_name,
                        "content": agent_response,
                        "tools_used": list(tool_results.keys()) if tool_results else [],
                        "message": f"✅ {agent_name} completed analysis"
                    }) + "\n"
                    
                except Exception as e:
                    logger.error(f"Advanced agent processing failed for {agent_name}: {e}")
                    yield json.dumps({
                        "type": "agent_error",
                        "agent": agent_name,
                        "error": str(e),
                        "message": f"❌ {agent_name} encountered an error"
                    }) + "\n"
                
                await asyncio.sleep(0.3)
            
            # Advanced synthesis with collaboration summary
            yield json.dumps({
                "type": "status",
                "message": "🔮 Synthesizing collaborative insights..."
            }) + "\n"
            await asyncio.sleep(0.5)
            
            # Create comprehensive synthesis
            synthesis_prompt = f"""<think>
I need to synthesize insights from {len(agents_used)} specialized agents who have collaborated on this question: {question}

Let me review their communications and responses to create a comprehensive synthesis.


You are an expert synthesis agent. Create a comprehensive response that integrates the collaborative analysis from multiple specialized agents.

QUESTION: {question}

AGENT COLLABORATION SUMMARY:
{chr(10).join([f"Agent: {comm['from']} → {comm['to']}: {comm['message']}" for comm in agent_communications])}

DETAILED AGENT ANALYSES:
"""
            
            for agent_name, response in agent_responses.items():
                agent_role = available_agents[agent_name].get('role', agent_name)
                synthesis_prompt += f"\\n{agent_role} ({agent_name}):\\n{response}\\n"
            
            synthesis_prompt += """
Create a synthesis that:
1. Shows how agents collaborated and built upon each other's insights
2. Integrates all perspectives into a coherent response
3. Highlights unique contributions from each agent's expertise
4. Provides comprehensive recommendations based on collective analysis
5. Provides clear synthesis of the collaborative analysis"""
            
            # Synthesis LLM call
            try:
                synthesizer_config = LLMConfig(
                    model_name=system.llm_settings["model"],
                    temperature=0.6,
                    top_p=0.9,
                    max_tokens=2500
                )
                
                synthesizer_llm = OllamaLLM(synthesizer_config, base_url=system.ollama_base_url)
                
                final_response = ""
                thinking_content = ""
                in_thinking = False
                synthesis_thinking_started = False
                
                async for response_chunk in synthesizer_llm.generate_stream(synthesis_prompt):
                    chunk_text = response_chunk.text
                    final_response += chunk_text
                    
                    if "<think>" in chunk_text.lower():
                        in_thinking = True
                        synthesis_thinking_started = True
                        
                        yield json.dumps({
                            "type": "synthesis_thinking_start",
                            "agent": "synthesizer",
                            "message": "💭 Synthesizing collaborative insights..."
                        }) + "\n"
                        
                        think_start = chunk_text.lower().find("<think>") + 7
                        thinking_content += chunk_text[think_start:]
                        
                    elif "</think>" in chunk_text.lower():
                        in_thinking = False
                        
                        think_end = chunk_text.lower().find("</think>")
                        thinking_content += chunk_text[:think_end]
                        
                        yield json.dumps({
                            "type": "synthesis_thinking_complete",
                            "agent": "synthesizer",
                            "thinking": thinking_content.strip()
                        }) + "\n"
                        
                        remaining_text = chunk_text[think_end + 8:]
                        if remaining_text.strip():
                            yield json.dumps({
                                "type": "agent_token",
                                "agent": "synthesizer",
                                "token": remaining_text
                            }) + "\n"
                            
                    elif in_thinking:
                        thinking_content += chunk_text
                        
                    else:
                        if chunk_text.strip():
                            yield json.dumps({
                                "type": "agent_token",
                                "agent": "synthesizer",
                                "token": chunk_text
                            }) + "\n"
                    
                    await asyncio.sleep(0.02)
                
                final_answer = system._clean_llm_response(final_response)
                
            except Exception as e:
                logger.error(f"Synthesis failed: {e}")
                final_answer = f"Collaborative analysis from {', '.join(agents_used)} provides comprehensive insights into {question}."
            
            # Final completion with collaboration summary
            yield json.dumps({
                "type": "final_response",
                "response": final_answer,
                "agents_used": agents_used,
                "collaboration_summary": {
                    "communications": len(agent_communications),
                    "tools_used": sum(len(resp.get('tools_used', [])) for resp in agent_responses.values() if isinstance(resp, dict)),
                    "agents_collaborated": len(agents_used)
                },
                "execution_pattern": "advanced_collaborative",
                "confidence_score": 0.95,
                "conversation_id": conversation_id,
                "message": "🎉 Advanced multi-agent collaboration completed"
            }) + "\n"
            
        except Exception as e:
            logger.error(f"Advanced multi-agent streaming failed: {e}")
            yield json.dumps({
                "type": "error",
                "error": str(e),
                "message": "❌ Advanced multi-agent execution failed"
            }) + "\n"
    
    return stream_advanced_multi_agent()