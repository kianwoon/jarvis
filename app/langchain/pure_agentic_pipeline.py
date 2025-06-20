"""
Pure Agentic Pipeline - Complete Rebuild

A completely clean implementation that:
1. Uses direct tool execution (no broken extract functions)
2. Passes clean results between agents (no thinking content)
3. Follows simple, proven patterns
4. No multi-agent contamination
"""

import asyncio
import json
import logging
import re
from typing import Dict, List, Any, Optional, AsyncGenerator, Tuple
from datetime import datetime

from app.core.pipeline_agents_cache import get_pipeline_agent_config
from app.core.mcp_tools_cache import get_enabled_mcp_tools
from app.core.llm_settings_cache import get_llm_settings
from app.llm.ollama import OllamaLLM
from app.llm.base import LLMConfig
from app.langchain.service import call_mcp_tool
from app.core.tool_error_handler import call_mcp_tool_with_retry, RetryConfig

logger = logging.getLogger(__name__)

class PureAgenticPipeline:
    """Pure agentic pipeline - simple, reliable, no contamination"""
    
    def __init__(self, pipeline_id: int, execution_id: str, trace=None):
        self.pipeline_id = pipeline_id
        self.execution_id = execution_id
        self.trace = trace
        self.llm_settings = get_llm_settings()
        
    async def execute_agents_sequentially(
        self, 
        query: str, 
        agents: List[Dict[str, Any]]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute agents sequentially with clean handoffs"""
        
        logger.info(f"🔵 [PURE PIPELINE] Starting execution - Pipeline {self.pipeline_id}")
        logger.info(f"🔵 [PURE PIPELINE] Agents: {[a.get('agent_name') for a in agents]}")
        
        clean_results = []  # Only clean results, no thinking content
        
        for idx, agent_info in enumerate(agents):
            agent_name = agent_info.get("agent_name")
            
            logger.info(f"🔵 [PURE PIPELINE] Executing agent {idx + 1}/{len(agents)}: {agent_name}")
            
            # Emit start event
            yield {
                "event": "agent_start",
                "data": {
                    "agent": agent_name,
                    "agent_index": idx,
                    "total_agents": len(agents),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Get agent config
            agent_config = get_pipeline_agent_config(self.pipeline_id, agent_name)
            if not agent_config:
                logger.error(f"🔵 [PURE PIPELINE] Agent {agent_name} not found")
                yield {
                    "event": "error",
                    "data": {"error": f"Agent {agent_name} not found"}
                }
                return
            
            # Build query for this agent
            if idx == 0:
                agent_query = query
            else:
                # Pass only clean results from previous agents
                previous_results = "\n\n".join([
                    f"Agent {result['agent']} result: {result['final_result']}"
                    for result in clean_results
                ])
                agent_query = f"Previous results:\n{previous_results}\n\nYour task: {agent_config.get('system_prompt', '')}"
            
            # Execute agent
            async for result in self._execute_clean_agent(agent_name, agent_config, agent_query):
                if result["type"] == "agent_complete":
                    # Store clean result only
                    clean_results.append({
                        "agent": agent_name,
                        "final_result": result["final_result"],
                        "tools_used": result["tools_used"]
                    })
                    
                    # Emit completion
                    yield {
                        "event": "agent_complete", 
                        "data": {
                            "agent": agent_name,
                            "response": result["final_result"],
                            "agent_index": idx,
                            "total_agents": len(agents),
                            "tools_used": result["tools_used"]
                        }
                    }
                    break
                elif result["type"] == "error":
                    logger.error(f"🔵 [PURE PIPELINE] Agent {agent_name} failed: {result['error']}")
                    yield {"event": "error", "data": result}
                    return
        
        # Pipeline complete
        yield {
            "event": "pipeline_complete",
            "data": {
                "pipeline_id": self.pipeline_id,
                "execution_id": self.execution_id,
                "total_agents": len(agents),
                "final_output": clean_results[-1]["final_result"] if clean_results else ""
            }
        }
    
    async def _execute_clean_agent(
        self, 
        agent_name: str, 
        agent_config: Dict[str, Any], 
        query: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute single agent with direct tool handling"""
        
        # Create Langfuse agent span and generation
        agent_span = None
        agent_generation = None
        if self.trace:
            try:
                from app.core.langfuse_integration import get_tracer
                tracer = get_tracer()
                if tracer.is_enabled():
                    # Create agent-level span to contain generation and tool spans
                    # Check if self.trace is actually a span (has .span method) or a trace
                    if hasattr(self.trace, 'span') and callable(getattr(self.trace, 'span')):
                        # self.trace is a span, create agent span as child
                        agent_span = self.trace.span(
                            name=f"agent-{agent_name}",
                            metadata={
                                "agent_name": agent_name,
                                "pipeline_id": self.pipeline_id,
                                "tools": agent_config.get("tools", [])
                            }
                        )
                    else:
                        # self.trace is a trace, use tracer method
                        agent_span = tracer.create_span(
                            self.trace,
                            name=f"agent-{agent_name}",
                            metadata={
                                "agent_name": agent_name,
                                "pipeline_id": self.pipeline_id,
                                "tools": agent_config.get("tools", [])
                            }
                        )
                    
                    # Create generation within the agent span
                    if agent_span:
                        agent_generation = tracer.create_generation_with_usage(
                            trace=self.trace,
                            name=f"{agent_name}-generation",
                            model=self.llm_settings.get("model", "qwen3:30b-a3b"),
                            input_text=query,
                            metadata={
                                "agent_name": agent_name,
                                "pipeline_id": self.pipeline_id
                            },
                            parent_span=agent_span
                        )
            except Exception as e:
                logger.warning(f"Failed to create span for {agent_name}: {e}")
        
        # Build prompt
        system_prompt = agent_config.get("system_prompt", "")
        available_tools = agent_config.get("tools", [])
        
        # Build tool specifications
        tool_specs = ""
        if available_tools:
            tool_specs = "\n\nAVAILABLE TOOLS:\n"
            tools = get_enabled_mcp_tools()
            for tool_name in available_tools:
                if tool_name in tools:
                    tool_info = tools[tool_name]
                    description = tool_info.get("description", f"{tool_name} tool")
                    tool_specs += f"• {tool_name}: {description}\n"
            
            tool_specs += "\nTo use tools, include them in your response like this:\n"
            tool_specs += "TOOL_CALL: tool_name(param1=value1, param2=value2)\n"
        
        full_prompt = f"""{system_prompt}

USER QUERY: {query}{tool_specs}

Complete your task step by step. If you need tools, include TOOL_CALL lines. Then provide your final result."""
        
        # Create LLM config
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
        
        # Generate response
        try:
            logger.info(f"🔵 [PURE PIPELINE] Generating response for {agent_name}")
            response_text = ""
            async for response_chunk in llm.generate_stream(full_prompt):
                response_text += response_chunk.text
            
            logger.info(f"🔵 [PURE PIPELINE] Generated {len(response_text)} chars for {agent_name}")
            
            # Execute tools directly
            tools_used = []
            tool_results = []
            
            # Extract tool calls with robust pattern
            tool_matches = self._extract_tool_calls(response_text)
            
            for tool_name, params_str in tool_matches:
                if tool_name in available_tools:
                    try:
                        # Parse parameters
                        params = self._parse_tool_params(params_str)
                        logger.info(f"🔵 [PURE PIPELINE] Executing {tool_name} with params: {params}")
                        
                        # Enhanced execution with retry and error handling
                        use_enhanced_error_handling = agent_config.get("enhanced_error_handling", True)
                        
                        # Create tool span as child of agent span
                        tool_span = None
                        if agent_span and self.trace:
                            try:
                                from app.core.langfuse_integration import get_tracer
                                tracer = get_tracer()
                                if tracer.is_enabled():
                                    # Use agent_span directly to create nested tool span
                                    tool_span = agent_span.span(
                                        name=f"tool-{tool_name}",
                                        input=params,
                                        metadata={
                                            "tool_name": tool_name,
                                            "agent": agent_name,
                                            "operation": "tool_execution",
                                            **params
                                        }
                                    )
                            except Exception as e:
                                logger.warning(f"Failed to create tool span for {tool_name}: {e}")
                        
                        if use_enhanced_error_handling:
                            # Use enhanced error handling with retry logic
                            retry_config = RetryConfig(
                                max_retries=agent_config.get("max_tool_retries", 3),
                                base_delay=agent_config.get("retry_base_delay", 1.0),
                                max_delay=agent_config.get("retry_max_delay", 60.0)
                            )
                            result = await call_mcp_tool_with_retry(tool_name, params, trace=tool_span or self.trace, retry_config=retry_config)
                        else:
                            # Use direct tool execution (legacy)
                            result = call_mcp_tool(tool_name, params, trace=tool_span or self.trace)
                        
                        # End tool span based on result
                        if tool_span:
                            try:
                                from app.core.langfuse_integration import get_tracer
                                tracer = get_tracer()
                                if isinstance(result, dict) and "error" in result:
                                    # Tool failed
                                    tracer.end_span_with_result(tool_span, None, False, result["error"])
                                else:
                                    # Tool succeeded
                                    tracer.end_span_with_result(tool_span, result, True)
                            except Exception as e:
                                logger.warning(f"Failed to end tool span for {tool_name}: {e}")
                        
                        # Check result and log appropriately
                        if isinstance(result, dict) and "error" in result:
                            error_msg = result["error"]
                            error_type = result.get("error_type", "unknown")
                            attempts = result.get("attempts", 1)
                            
                            if attempts > 1:
                                logger.error(f"🔵 [PURE PIPELINE] Tool {tool_name} failed after {attempts} attempts ({error_type}): {error_msg}")
                            else:
                                logger.error(f"🔵 [PURE PIPELINE] Tool {tool_name} failed ({error_type}): {error_msg}")
                                
                            tool_results.append(f"{tool_name}: Error ({error_type}) - {error_msg}")
                        else:
                            # Extract readable content from tool result
                            readable_result = self._extract_tool_content(result)
                            tool_results.append(f"{tool_name}: {readable_result}")
                            tools_used.append(tool_name)
                            logger.info(f"🔵 [PURE PIPELINE] Tool {tool_name} executed successfully")
                        
                    except Exception as e:
                        logger.error(f"🔵 [PURE PIPELINE] Tool {tool_name} execution exception: {e}")
                        
                        # End tool span with exception
                        if tool_span:
                            try:
                                from app.core.langfuse_integration import get_tracer
                                tracer = get_tracer()
                                tracer.end_span_with_result(tool_span, None, False, str(e))
                            except Exception:
                                pass
                        
                        tool_results.append(f"{tool_name}: Exception - {str(e)}")
            
            # If tools were used, generate synthesis response
            if tool_results:
                logger.info(f"🔵 [PURE PIPELINE] Generating synthesis for {agent_name} with {len(tool_results)} tool results")
                
                # Build tool context
                tool_context = "\n\n".join(tool_results)
                
                # Create synthesis prompt
                synthesis_prompt = f"""{system_prompt}

USER QUERY: {query}

TOOL RESULTS:
{tool_context}

Based on these tool results, provide your final response to complete the user's request. Be thorough and professional."""

                # Generate synthesis response
                synthesis_response = ""
                async for response_chunk in llm.generate_stream(synthesis_prompt):
                    synthesis_response += response_chunk.text
                
                logger.info(f"🔵 [PURE PIPELINE] Synthesis complete for {agent_name} - {len(synthesis_response)} chars")
                final_result = self._extract_text_after_thinking(synthesis_response)
            else:
                # No tools - extract content after thinking
                final_result = self._extract_text_after_thinking(response_text)
            
            # Complete Langfuse generation and span
            if agent_generation:
                try:
                    from app.core.langfuse_integration import get_tracer
                    tracer = get_tracer()
                    
                    # End the generation first
                    usage = tracer.estimate_token_usage(query, final_result)
                    agent_generation.end(
                        output=final_result,
                        usage_details=usage,
                        metadata={
                            "success": True,
                            "tools_used": tools_used,
                            "response_length": len(response_text),
                            "enhanced_error_handling": agent_config.get("enhanced_error_handling", True)
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to end generation for {agent_name}: {e}")
            
            if agent_span:
                try:
                    from app.core.langfuse_integration import get_tracer
                    tracer = get_tracer()
                    
                    # For thinking mode, show full response with <think> tags in Langfuse
                    # For non-thinking mode, show cleaned result
                    enable_thinking = agent_config.get("enable_thinking", False)
                    display_result = response_text if enable_thinking else final_result
                    
                    # Check if any tool failures occurred
                    tool_failures = [result for result in tool_results if "Error" in result or "Exception" in result]
                    retry_attempts_made = any("failed after" in result for result in tool_failures)
                    
                    tracer.end_span_with_result(
                        agent_span,
                        {
                            "full_response": display_result,
                            "final_result": final_result,  # Show complete result
                            "tools_used": tools_used,
                            "response_length": len(response_text),
                            "thinking_enabled": enable_thinking,
                            "enhanced_error_handling": agent_config.get("enhanced_error_handling", True),
                            "tool_failures_count": len(tool_failures),
                            "retry_attempts_made": retry_attempts_made,
                            "tool_results_summary": f"{len(tools_used)} successful, {len(tool_failures)} failed"
                        },
                        True
                    )
                except Exception:
                    pass
            
            logger.info(f"🔵 [PURE PIPELINE] Agent {agent_name} complete - Final result: {len(final_result)} chars")
            
            yield {
                "type": "agent_complete",
                "final_result": final_result,
                "tools_used": tools_used
            }
            
        except Exception as e:
            logger.error(f"🔵 [PURE PIPELINE] Agent {agent_name} execution failed: {e}")
            
            # End generation with error
            if agent_generation:
                try:
                    agent_generation.end(
                        output=f"Error: {str(e)}",
                        metadata={
                            "success": False,
                            "error": str(e)
                        }
                    )
                except Exception:
                    pass
            
            # End span with error
            if agent_span:
                try:
                    from app.core.langfuse_integration import get_tracer
                    tracer = get_tracer()
                    tracer.end_span_with_result(agent_span, None, False, str(e))
                except Exception:
                    pass
            
            yield {"type": "error", "error": str(e)}
    
    def _extract_tool_calls(self, text: str) -> List[Tuple[str, str]]:
        """Extract tool calls with proper parentheses handling"""
        tool_calls = []
        
        # Find all TOOL_CALL occurrences
        pattern = r'TOOL_CALL:\s*(\w+)\('
        matches = re.finditer(pattern, text)
        
        for match in matches:
            tool_name = match.group(1)
            start_pos = match.end() - 1  # Position of opening parenthesis
            
            # Find matching closing parenthesis
            paren_count = 0
            params_start = start_pos + 1
            params_end = params_start
            
            for i, char in enumerate(text[start_pos:], start_pos):
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                    if paren_count == 0:
                        params_end = i
                        break
            
            if paren_count == 0:  # Found matching closing parenthesis
                params_str = text[params_start:params_end]
                tool_calls.append((tool_name, params_str))
                logger.info(f"🔵 [PURE PIPELINE] Extracted tool call: {tool_name} with params: {params_str[:100]}...")
            else:
                logger.warning(f"🔵 [PURE PIPELINE] Unmatched parentheses for tool {tool_name}")
        
        return tool_calls
    
    def _parse_tool_params(self, params_str: str) -> Dict[str, Any]:
        """Parse tool parameters from string with proper quote handling"""
        params = {}
        
        # Enhanced parameter parsing that handles quoted strings properly
        # Pattern matches: param="quoted value with, commas" or param=simple_value
        param_pattern = r'(\w+)=(?:"([^"]*)"|\'([^\']*)\'|([^,]+?))\s*(?:,|$)'
        matches = re.findall(param_pattern, params_str)
        
        for match in matches:
            key = match[0]
            # Get the value from whichever group matched (double quote, single quote, or unquoted)
            value = match[1] or match[2] or match[3]
            value = value.strip()
            params[key] = value
        
        return params
    
    def _extract_tool_content(self, tool_result: Any) -> str:
        """Extract readable content from tool result"""
        try:
            if isinstance(tool_result, dict):
                # Handle MCP tool response format
                if 'content' in tool_result and isinstance(tool_result['content'], list):
                    content_parts = []
                    for item in tool_result['content']:
                        if isinstance(item, dict) and 'text' in item:
                            content_parts.append(item['text'])
                        elif isinstance(item, str):
                            content_parts.append(item)
                    return '\n'.join(content_parts) if content_parts else str(tool_result)
                else:
                    return str(tool_result)
            else:
                return str(tool_result)
        except Exception as e:
            logger.warning(f"Failed to extract tool content: {e}")
            return str(tool_result)
    
    def _extract_text_after_thinking(self, response_text: str) -> str:
        """Extract clean text content after thinking tags"""
        if '</think>' in response_text:
            parts = response_text.split('</think>')
            if len(parts) > 1:
                return parts[-1].strip()
        
        return response_text.strip()