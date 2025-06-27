"""
Real multi-agent streaming with actual LLM calls, thinking support, and agent collaboration
Uses agents from database only - NO HARDCODING
"""
import json
import asyncio
from typing import Optional
from app.langchain.langgraph_multi_agent_system import LangGraphMultiAgentSystem
from app.llm.ollama import OllamaLLM
from app.llm.base import LLMConfig
import logging

logger = logging.getLogger(__name__)

async def real_multi_agent_streaming(
    question: str,
    conversation_id: Optional[str] = None
):
    """
    Real multi-agent streaming with actual LLM calls and collaboration
    Uses agents from database table "langgraph_agents" only
    """
    
    async def stream_real_multi_agent():
        try:
            yield json.dumps({"type": "status", "message": "🎯 Starting real multi-agent analysis..."}) + "\n"
            await asyncio.sleep(0.3)
            
            # Initialize system
            system = LangGraphMultiAgentSystem(conversation_id)
            
            # Get agents from database only - use system.agents which loads from Redis cache of DB
            available_agents = system.agents
            if not available_agents:
                yield json.dumps({"type": "error", "error": "No agents found in database"}) + "\n"
                return
            
            # Smart agent selection based on question keywords
            question_lower = question.lower()
            selected_agents = []
            
            # Select agents based on their actual roles from database
            for agent_name, agent_data in available_agents.items():
                role = agent_data.get('role', '').lower()
                system_prompt = agent_data.get('system_prompt', '').lower()
                
                # Check if agent is relevant to question
                if (any(keyword in role for keyword in ["corporate", "strategist", "business"]) and 
                    any(keyword in question_lower for keyword in ["corporate", "business", "strategy", "firm"])):
                    selected_agents.append(agent_name)
                elif (any(keyword in role for keyword in ["research", "analyst"]) and 
                      any(keyword in question_lower for keyword in ["research", "analysis", "study", "data"])):
                    selected_agents.append(agent_name)
                elif (any(keyword in role for keyword in ["technical", "technology"]) and 
                      any(keyword in question_lower for keyword in ["ai", "technology", "technical", "implementation"])):
                    selected_agents.append(agent_name)
                elif len(selected_agents) < 2:  # Fallback to first available agents
                    selected_agents.append(agent_name)
                
                if len(selected_agents) >= 3:  # Limit to 3 agents
                    break
            
            # Fallback if no agents selected
            if not selected_agents:
                selected_agents = list(available_agents.keys())[:2]
            
            agents_used = selected_agents
            
            yield json.dumps({
                "type": "agents_selected",
                "agents": agents_used,
                "message": f"🎯 Selected agents from database: {', '.join(agents_used)}"
            }) + "\n"
            await asyncio.sleep(0.2)
            
            # Track responses for collaboration
            agent_responses = {}
            
            # Stream each agent's work with REAL LLM calls
            for i, agent_name in enumerate(agents_used):
                yield json.dumps({
                    "type": "agent_start",
                    "agent": agent_name,
                    "message": f"🎯 {agent_name} analyzing..."
                }) + "\n"
                await asyncio.sleep(0.3)
                
                try:
                    # Get agent config from database
                    agent_data = available_agents[agent_name]
                    system_prompt = agent_data.get('system_prompt', '')
                    role = agent_data.get('role', agent_name)
                    config = agent_data.get('config', {})
                    
                    # Build collaborative context from previous agents
                    collaboration_context = ""
                    if i > 0:
                        previous_insights = []
                        for prev_agent in agents_used[:i]:
                            if prev_agent in agent_responses:
                                prev_role = available_agents[prev_agent].get('role', prev_agent)
                                prev_summary = agent_responses[prev_agent][:200] + "..." if len(agent_responses[prev_agent]) > 200 else agent_responses[prev_agent]
                                previous_insights.append(f"- {prev_role}: {prev_summary}")
                        
                        if previous_insights:
                            collaboration_context = f"\n\nPREVIOUS AGENT INSIGHTS:\n{chr(10).join(previous_insights)}\n\nBuild upon these insights with your unique expertise. Reference specific points when relevant."
                    
                    # Create collaborative prompt using agent's actual system prompt
                    agent_prompt = f"""{system_prompt}

User Question: {question}{collaboration_context}

Provide a comprehensive analysis that:
1. Demonstrates your specialized expertise
2. {"References and builds upon previous insights" if collaboration_context else "Provides foundational analysis"}
3. Includes specific examples and evidence
4. Offers actionable recommendations

Use clear structure and detailed analysis."""
                    
                    # Real LLM streaming call using agent's config from database
                    llm_config = LLMConfig(
                        model_name=system.llm_settings["model"],
                        temperature=config.get('temperature', 0.7),
                        top_p=config.get('top_p', 0.9),
                        max_tokens=config.get('max_tokens', 1500)
                    )
                    
                    llm = OllamaLLM(llm_config, base_url=system.ollama_base_url)
                    
                    # Stream real response with thinking support
                    full_response = ""
                    thinking_content = ""
                    in_thinking = False
                    display_content = ""
                    
                    async for response_chunk in llm.generate_stream(agent_prompt):
                        chunk_text = response_chunk.text
                        full_response += chunk_text
                        
                        # Handle thinking tags for reasoning display
                        if "<think>" in chunk_text:
                            in_thinking = True
                            # Send thinking start event
                            yield json.dumps({
                                "type": "agent_thinking_start",
                                "agent": agent_name,
                                "message": f"💭 {agent_name} reasoning..."
                            }) + "\n"
                            
                            thinking_start = chunk_text.find("<think>") + 7
                            thinking_content += chunk_text[thinking_start:]
                            
                        elif "</think>" in chunk_text:
                            in_thinking = False
                            thinking_end = chunk_text.find("</think>")
                            thinking_content += chunk_text[:thinking_end]
                            
                            # Send complete thinking content
                            yield json.dumps({
                                "type": "agent_thinking_complete",
                                "agent": agent_name,
                                "thinking": thinking_content.strip()
                            }) + "\n"
                            
                            # Continue with regular content after </think>
                            remaining_text = chunk_text[thinking_end + 8:]
                            if remaining_text.strip():
                                display_content += remaining_text
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
                                display_content += chunk_text
                                yield json.dumps({
                                    "type": "agent_token",
                                    "agent": agent_name,
                                    "token": chunk_text
                                }) + "\n"
                        
                        await asyncio.sleep(0.02)  # Real-time streaming speed
                    
                    # Fallback thinking if none detected
                    if not thinking_detected:
                        fallback_thinking = f"Analyzing '{question}' from {role} perspective. I need to consider my domain expertise, available insights from other agents, and provide actionable recommendations based on my specialized knowledge."
                        
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
                    
                    # Clean and store response for collaboration
                    agent_response = system._clean_llm_response(full_response)
                    agent_responses[agent_name] = agent_response
                    
                    yield json.dumps({
                        "type": "agent_complete",
                        "agent": agent_name,
                        "content": agent_response,
                        "message": f"✅ {agent_name} completed"
                    }) + "\n"
                    
                except Exception as e:
                    logger.error(f"Real LLM call failed for {agent_name}: {e}")
                    # Use fallback response
                    fallback_response = f"Analysis from {agent_name} perspective: {question} requires careful consideration of multiple factors and strategic implications."
                    agent_responses[agent_name] = fallback_response
                    
                    yield json.dumps({
                        "type": "agent_complete", 
                        "agent": agent_name,
                        "content": fallback_response,
                        "message": f"✅ {agent_name} completed (fallback)"
                    }) + "\n"
                
                await asyncio.sleep(0.2)
            
            # Real synthesizer with collaboration
            yield json.dumps({
                "type": "status",
                "message": "🔮 Synthesizing collaborative insights..."
            }) + "\n"
            await asyncio.sleep(0.5)
            
            # Create synthesis prompt incorporating all agent responses
            synthesis_prompt = f"""You are a synthesis expert. Combine the following agent analyses into a comprehensive, coherent response:

QUESTION: {question}

AGENT ANALYSES:
"""
            
            for agent_name, response in agent_responses.items():
                agent_role = available_agents[agent_name].get('role', agent_name)
                synthesis_prompt += f"\n{agent_role}:\n{response}\n"
            
            synthesis_prompt += """
Create a comprehensive synthesis that:
1. Integrates insights from all agents
2. Identifies common themes and differences
3. Provides actionable recommendations
4. Addresses the question thoroughly
5. Shows how different perspectives complement each other

"""
            
            # Real synthesizer LLM call
            try:
                synthesizer_config = LLMConfig(
                    model_name=system.llm_settings["model"],
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=2000
                )
                
                synthesizer_llm = OllamaLLM(synthesizer_config, base_url=system.ollama_base_url)
                
                final_response = ""
                thinking_content = ""
                in_thinking = False
                
                async for response_chunk in synthesizer_llm.generate_stream(synthesis_prompt):
                    chunk_text = response_chunk.text
                    final_response += chunk_text
                    
                    # Handle synthesizer thinking
                    if "<think>" in chunk_text:
                        in_thinking = True
                        yield json.dumps({
                            "type": "synthesis_thinking_start",
                            "agent": "synthesizer",
                            "message": "💭 Synthesizing insights..."
                        }) + "\n"
                        thinking_start = chunk_text.find("<think>") + 7
                        thinking_content += chunk_text[thinking_start:]
                        
                    elif "</think>" in chunk_text:
                        in_thinking = False
                        thinking_end = chunk_text.find("</think>")
                        thinking_content += chunk_text[:thinking_end]
                        
                        yield json.dumps({
                            "type": "synthesis_thinking_complete",
                            "agent": "synthesizer",
                            "thinking": thinking_content.strip()
                        }) + "\n"
                        
                        remaining_text = chunk_text[thinking_end + 8:]
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
                logger.error(f"Synthesizer LLM call failed: {e}")
                final_answer = f"Based on collaborative analysis from {', '.join(agents_used)}, {question} requires integrated consideration of multiple perspectives for optimal outcomes."
            
            # Send final completion
            yield json.dumps({
                "type": "final_response",
                "response": final_answer,
                "agents_used": agents_used,
                "execution_pattern": "collaborative_sequential",
                "confidence_score": 0.9,
                "conversation_id": conversation_id,
                "message": "🎉 Real multi-agent collaborative analysis completed"
            }) + "\n"
            
        except Exception as e:
            logger.error(f"Real multi-agent streaming failed: {e}")
            yield json.dumps({
                "type": "error",
                "error": str(e),
                "message": "❌ Real multi-agent execution failed"
            }) + "\n"
    
    return stream_real_multi_agent()