"""
Agentic Pipeline Executor - Clean Implementation

A completely separate execution system for agentic pipelines that:
1. Uses ONLY pipeline_agents table configurations
2. Follows standard chat's clean tool execution pattern  
3. Has NO contamination from multi-agent systems
4. Handles sequential agent execution properly
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime

from app.core.pipeline_agents_cache import get_pipeline_agent_config
from app.core.mcp_tools_cache import get_enabled_mcp_tools
from app.core.llm_settings_cache import get_llm_settings
from app.llm.ollama import OllamaLLM
from app.llm.base import LLMConfig
from app.core.langfuse_integration import get_tracer

logger = logging.getLogger(__name__)

class AgenticPipelineExecutor:
    """Clean agentic pipeline executor with no multi-agent contamination"""
    
    def __init__(self, pipeline_id: int, execution_id: str, trace=None):
        self.pipeline_id = pipeline_id
        self.execution_id = execution_id
        self.trace = trace
        self.tracer = get_tracer()
        self.llm_settings = get_llm_settings()
        
    async def execute_sequential_pipeline(
        self, 
        query: str, 
        agents: List[Dict[str, Any]]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute agents sequentially using clean pipeline logic"""
        
        logger.info(f"🟢 [AGENTIC PIPELINE] CLEAN EXECUTOR STARTING - Pipeline {self.pipeline_id}")
        logger.info(f"🟢 [AGENTIC PIPELINE] NO multi-agent contamination - using pipeline_agents table ONLY")
        logger.info(f"🟢 [AGENTIC PIPELINE] Executing {len(agents)} agents: {[a.get('agent_name') for a in agents]}")
        
        previous_outputs = []
        total_agents = len(agents)
        
        for idx, agent_info in enumerate(agents):
            agent_name = agent_info.get("agent_name")
            
            logger.info(f"🟢 [AGENTIC PIPELINE] Starting agent {idx + 1}/{total_agents}: '{agent_name}'")
            logger.info(f"🟢 [AGENTIC PIPELINE] Agent info from pipeline: {agent_info}")
            
            # Emit agent start
            yield {
                "event": "agent_start",
                "data": {
                    "agent": agent_name,
                    "agent_index": idx,
                    "total_agents": total_agents,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Get clean agent config from pipeline_agents table
            logger.info(f"🟢 [AGENTIC PIPELINE] Looking up agent '{agent_name}' in pipeline_agents table for pipeline {self.pipeline_id}")
            agent_config = get_pipeline_agent_config(self.pipeline_id, agent_name)
            if not agent_config:
                logger.error(f"🟢 [AGENTIC PIPELINE] CRITICAL: Agent '{agent_name}' not found in pipeline_agents table for pipeline {self.pipeline_id}")
                yield {
                    "event": "error", 
                    "data": {"error": f"Agent '{agent_name}' not found in pipeline {self.pipeline_id}"}
                }
                return
            else:
                logger.info(f"🟢 [AGENTIC PIPELINE] Found agent config: {agent_config}")
                
            # Build agent query based on position
            if idx == 0:
                agent_query = query
                logger.info(f"🟢 [AGENTIC PIPELINE] Agent {agent_name} (first) - Query: {agent_query[:200]}...")
            else:
                # Sequential: pass previous agent's output as context
                previous_context = previous_outputs[-1]["response"] if previous_outputs else ""
                agent_query = f"Previous agent output:\n{previous_context}\n\nContinue with your task based on this context."
                logger.info(f"🟢 [AGENTIC PIPELINE] Agent {agent_name} (sequential) - Previous context: {len(previous_context)} chars")
                logger.info(f"🟢 [AGENTIC PIPELINE] Agent {agent_name} - Query: {agent_query[:200]}...")
            
            # Execute agent cleanly
            agent_response = ""
            tools_used = []
            
            async for result in self._execute_single_agent(agent_name, agent_config, agent_query):
                if result["type"] == "agent_complete":
                    agent_response = result["response"]
                    tools_used = result.get("tools_used", [])
                    break
                elif result["type"] == "agent_token":
                    # Forward streaming tokens
                    yield {
                        "event": "streaming",
                        "data": {
                            "agent": agent_name,
                            "content": result["token"]
                        }
                    }
                elif result["type"] == "error":
                    yield {"event": "error", "data": result}
                    return
            
            # Store output for next agent
            previous_outputs.append({
                "agent": agent_name,
                "response": agent_response,
                "tools_used": tools_used
            })
            
            # Emit agent completion
            yield {
                "event": "agent_complete",
                "data": {
                    "agent": agent_name,
                    "response": agent_response,
                    "agent_index": idx,
                    "total_agents": total_agents,
                    "tools_used": tools_used
                }
            }
        
        # Pipeline complete
        yield {
            "event": "pipeline_complete",
            "data": {
                "pipeline_id": self.pipeline_id,
                "execution_id": self.execution_id,
                "total_agents": total_agents,
                "final_output": previous_outputs[-1]["response"] if previous_outputs else ""
            }
        }
    
    async def _execute_single_agent(
        self, 
        agent_name: str, 
        agent_config: Dict[str, Any], 
        query: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute a single agent following standard chat pattern"""
        
        # Create agent span for Langfuse tracing
        agent_span = None
        if self.trace:
            try:
                from app.core.langfuse_integration import get_tracer
                tracer = get_tracer()
                if tracer.is_enabled():
                    agent_span = tracer.create_generation_with_usage(
                        trace=self.trace,
                        name=f"agent-{agent_name}",
                        model=self.llm_settings.get("model", "qwen3:30b-a3b"),
                        input_text=query,
                        metadata={
                            "agent_name": agent_name,
                            "pipeline_id": self.pipeline_id,
                            "execution_id": self.execution_id,
                            "enable_thinking": agent_config.get("enable_thinking", False),
                            "tools_available": agent_config.get("tools", [])
                        }
                    )
                    logger.info(f"🟢 [AGENTIC PIPELINE] Created Langfuse span for agent {agent_name}")
            except Exception as e:
                logger.warning(f"Failed to create agent span for {agent_name}: {e}")
        
        # Build clean prompt
        system_prompt = agent_config.get("system_prompt", "")
        available_tools = agent_config.get("tools", [])
        
        # Add tool specifications if agent has tools
        tool_specs = ""
        if available_tools:
            tool_specs = self._build_tool_specifications(available_tools)
        
        # Build prompt based on thinking mode
        enable_thinking = agent_config.get("enable_thinking", False)
        
        if enable_thinking:
            full_prompt = f"""{system_prompt}

USER QUERY: {query}{tool_specs}

Think through this step by step, then use the available tools to complete your task and provide your response."""
        else:
            full_prompt = f"""{system_prompt}

USER QUERY: {query}{tool_specs}

Use the available tools to complete your task, then provide your response."""
        
        # Create LLM instance using standard chat pattern
        enable_thinking = agent_config.get("enable_thinking", False)
        mode_config = self.llm_settings.get("thinking_mode" if enable_thinking else "non_thinking_mode", {})
        max_tokens = agent_config.get("max_tokens", self.llm_settings.get("max_tokens", 4000))
        
        config = LLMConfig(
            model_name=self.llm_settings.get("model"),
            temperature=float(agent_config.get("temperature", mode_config.get("temperature", 0.7))),
            top_p=float(mode_config.get("top_p", 1.0)),
            max_tokens=int(max_tokens)
        )
        
        import os
        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
        llm = OllamaLLM(config, base_url=ollama_url)
        
        # Stream LLM response
        response_text = ""
        chunk_count = 0
        try:
            logger.info(f"🟢 [AGENTIC PIPELINE] Starting LLM generation for agent {agent_name}")
            logger.info(f"🟢 [AGENTIC PIPELINE] LLM config: model={config.model_name}, max_tokens={config.max_tokens}, temperature={config.temperature}")
            logger.info(f"🟢 [AGENTIC PIPELINE] Thinking mode enabled: {enable_thinking}")
            logger.info(f"🟢 [AGENTIC PIPELINE] Prompt preview: {full_prompt[:300]}...")
            logger.info(f"🟢 [AGENTIC PIPELINE] Prompt length: {len(full_prompt)} chars")
            
            async for response_chunk in llm.generate_stream(full_prompt):
                chunk_count += 1
                response_text += response_chunk.text
                if response_chunk.text.strip():
                    yield {
                        "type": "agent_token",
                        "token": response_chunk.text
                    }
                    
                # Log progress every 50 chunks
                if chunk_count % 50 == 0:
                    logger.info(f"🟢 [AGENTIC PIPELINE] Agent {agent_name} - {chunk_count} chunks, {len(response_text)} chars so far")
                    
            logger.info(f"🟢 [AGENTIC PIPELINE] LLM generation complete for agent {agent_name} - {chunk_count} chunks, {len(response_text)} chars total")
            logger.info(f"🟢 [AGENTIC PIPELINE] Response ends with: ...{response_text[-200:] if len(response_text) > 200 else response_text}")
            logger.info(f"🟢 [AGENTIC PIPELINE] Full response preview: {response_text[:500]}...")
        except Exception as e:
            logger.error(f"🟢 [AGENTIC PIPELINE] LLM generation failed for agent {agent_name}: {e}")
            if agent_span:
                try:
                    from app.core.langfuse_integration import get_tracer
                    tracer = get_tracer()
                    tracer.end_span_with_result(agent_span, None, False, str(e))
                except Exception:
                    pass
            yield {"type": "error", "error": str(e)}
            return
        
        # Process tools using standard chat pattern
        logger.info(f"🟢 [AGENTIC PIPELINE] Processing tools for agent {agent_name}")
        from app.langchain.service import extract_and_execute_tool_calls
        tool_results = extract_and_execute_tool_calls(response_text, trace=self.trace)
        tools_used = []
        final_response = response_text
        
        logger.info(f"🟢 [AGENTIC PIPELINE] Tool extraction result for agent {agent_name}: {len(tool_results) if tool_results else 0} tools found")
        
        if tool_results and any(r.get('success') for r in tool_results):
            # Extract tool names for tracking
            tools_used = [r['tool'] for r in tool_results if r.get('success')]
            logger.info(f"🟢 [AGENTIC PIPELINE] Successful tools for agent {agent_name}: {tools_used}")
            
            # Build enhanced response with tool results
            tool_context = "\n\nTool Results:\n"
            for tr in tool_results:
                if tr.get('success'):
                    import json
                    tool_context += f"\n{tr['tool']}: {json.dumps(tr['result'], indent=2)}\n"
            
            # Create follow-up prompt to synthesize with tool results
            synthesis_prompt = f"""{system_prompt}

USER QUERY: {query}

{tool_context}

Based on these tool results, provide your final response."""
            
            # Generate final response with tool results
            logger.info(f"🟢 [AGENTIC PIPELINE] Generating synthesis response for agent {agent_name}")
            final_response = ""
            async for response_chunk in llm.generate_stream(synthesis_prompt):
                final_response += response_chunk.text
            logger.info(f"🟢 [AGENTIC PIPELINE] Synthesis complete for agent {agent_name} - Final response length: {len(final_response)} chars")
        else:
            logger.info(f"🟢 [AGENTIC PIPELINE] No tools executed for agent {agent_name}, using original response")
        
        # End Langfuse span
        if agent_span:
            try:
                from app.core.langfuse_integration import get_tracer
                tracer = get_tracer()
                tracer.end_span_with_result(
                    agent_span, 
                    {
                        "response": final_response[:500],  # Truncate for Langfuse
                        "tools_used": tools_used,
                        "response_length": len(final_response)
                    }, 
                    True
                )
                logger.info(f"🟢 [AGENTIC PIPELINE] Completed Langfuse span for agent {agent_name}")
            except Exception as e:
                logger.warning(f"Failed to end agent span for {agent_name}: {e}")
        
        logger.info(f"🟢 [AGENTIC PIPELINE] Agent {agent_name} execution complete - Final response length: {len(final_response)} chars")
        logger.info(f"🟢 [AGENTIC PIPELINE] Final response preview: {final_response[:200]}...")
        
        yield {
            "type": "agent_complete",
            "response": final_response,
            "tools_used": tools_used
        }
    
    def _build_tool_specifications(self, available_tools: List[str]) -> str:
        """Build tool specifications for agent prompt"""
        if not available_tools:
            return ""
        
        tools = get_enabled_mcp_tools()
        tool_specs = "\n\nAVAILABLE TOOLS:\n"
        
        for tool_name in available_tools:
            if tool_name in tools:
                tool_info = tools[tool_name]
                description = tool_info.get("description", f"{tool_name} tool")
                tool_specs += f"• {tool_name}: {description}\n"
        
        tool_specs += "\nTo use a tool, use this format:\n"
        tool_specs += '<tool>tool_name(param1="value1", param2="value2")</tool>\n'
        tool_specs += "\nUse the tools when needed to complete your task."
        
        return tool_specs