"""
Agent Bridge for Langflow Integration
Connects Langflow workflows to your existing LangGraph agents
"""
import asyncio
import logging
import os
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from app.core.langgraph_agents_cache import get_langgraph_agents
from app.llm.ollama import OllamaLLM
from app.llm.base import LLMConfig
from app.core.langfuse_integration import get_tracer
from app.core.llm_settings_cache import get_second_llm_full_config

logger = logging.getLogger(__name__)

class AgentBridge:
    """Bridge between Langflow and LangGraph agents"""
    
    def __init__(self):
        self.tracer = get_tracer()
        # Use your existing Ollama URL pattern
        self.ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
    
    def get_available_agents(self) -> Dict[str, Any]:
        """Get all available agents for Langflow node configuration"""
        try:
            agents = get_langgraph_agents()
            logger.info(f"[AGENT BRIDGE] Retrieved {len(agents)} available agents")
            
            # Format for Langflow consumption
            formatted_agents = {}
            for agent_name, agent_info in agents.items():
                formatted_agents[agent_name] = {
                    "name": agent_name,
                    "role": agent_info.get("role", "No role specified"),
                    "system_prompt": agent_info.get("system_prompt", ""),
                    "tools": agent_info.get("tools", []),
                    "config": agent_info.get("config", {}),
                    "capabilities": agent_info.get("capabilities", {})
                }
            
            return formatted_agents
        except Exception as e:
            logger.error(f"[AGENT BRIDGE] Failed to get available agents: {e}")
            return {}
    
    def get_agent_config(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific agent"""
        agents = get_langgraph_agents()
        return agents.get(agent_name)
    
    def execute_agent_sync(
        self, 
        agent_name: str, 
        query: str, 
        context: Optional[str] = None,
        trace=None
    ) -> Dict[str, Any]:
        """Execute agent synchronously (for Langflow nodes)"""
        try:
            logger.info(f"[AGENT NODE DEBUG] Agent Bridge executing agent: {agent_name}")
            logger.info(f"[AGENT NODE DEBUG] Query length: {len(query) if query else 0}")
            logger.info(f"[AGENT NODE DEBUG] Context length: {len(context) if context else 0}")
            logger.info(f"[AGENT NODE DEBUG] Query preview: {query[:100]}...")
            
            # Get agent configuration
            agent_config = self.get_agent_config(agent_name)
            if not agent_config:
                logger.error(f"[AGENT NODE DEBUG] Agent {agent_name} not found in configuration")
                return {
                    "success": False,
                    "error": f"Agent {agent_name} not found",
                    "agent": agent_name,
                    "query": query
                }
            
            # Build prompt with system prompt and context
            system_prompt = agent_config.get("system_prompt", "")
            full_prompt = f"{system_prompt}\n\n"
            
            if context:
                full_prompt += f"Context: {context}\n\n"
            
            full_prompt += f"User Query: {query}"
            
            # Get model configuration (following your patterns)
            config = agent_config.get("config", {})
            
            # Use second_llm for lightweight agent bridge tasks
            second_llm_config = get_second_llm_full_config()
            model_name = config.get("model", second_llm_config.get("model", "qwen3:1.7b"))  # Default to second_llm
            
            # Create LLM instance (following your Ollama patterns)
            llm_config = LLMConfig(
                model_name=model_name,
                temperature=config.get("temperature", second_llm_config.get("temperature", 0.7)),
                top_p=config.get("top_p", second_llm_config.get("top_p", 0.9)),
                max_tokens=config.get("max_tokens", second_llm_config.get("max_tokens", 2000))
            )
            
            llm = OllamaLLM(llm_config, base_url=self.ollama_url)
            
            # Create LLM generation span if tracing enabled
            llm_generation_span = None
            if trace and self.tracer.is_enabled():
                try:
                    llm_generation_span = self.tracer.create_llm_generation_span(
                        trace,
                        model=model_name,
                        prompt=full_prompt,
                        operation=f"agent_execution_{agent_name}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to create LLM generation span: {e}")
            
            # Execute synchronously (convert async to sync) with timeout
            logger.info(f"[AGENT NODE DEBUG] Starting LLM generation for {agent_name} using model {model_name}")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Add timeout wrapper for agent LLM calls
                logger.info(f"[AGENT NODE DEBUG] Calling LLM with timeout of 45 seconds...")
                response = loop.run_until_complete(
                    asyncio.wait_for(
                        llm.generate(full_prompt), 
                        timeout=45.0  # 45 second timeout for agent execution
                    )
                )
                
                # End span with result
                if llm_generation_span:
                    try:
                        usage = self.tracer.estimate_token_usage(full_prompt, response.text)
                        llm_generation_span.end(
                            output=response.text,
                            usage=usage,
                            metadata={
                                "agent_name": agent_name,
                                "success": True
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Failed to end LLM generation span: {e}")
                
                logger.info(f"[AGENT NODE DEBUG] LLM generation completed for {agent_name}")
                logger.info(f"[AGENT NODE DEBUG] Response length: {len(response.text) if response.text else 0}")
                logger.debug(f"[AGENT NODE DEBUG] Response preview: {response.text[:200]}..." if response.text else "No response text")
                
                return {
                    "success": True,
                    "response": response.text,
                    "agent": agent_name,
                    "query": query,
                    "model": model_name,
                    "metadata": response.metadata
                }
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"[AGENT BRIDGE] Agent execution failed for {agent_name}: {e}")
            
            # End span with error
            if llm_generation_span:
                try:
                    llm_generation_span.end(
                        output=None,
                        metadata={
                            "agent_name": agent_name,
                            "success": False,
                            "error": str(e)
                        }
                    )
                except Exception:
                    pass
            
            return {
                "success": False,
                "error": str(e),
                "agent": agent_name,
                "query": query
            }
    
    async def execute_agent_async(
        self, 
        agent_name: str, 
        query: str, 
        context: Optional[str] = None,
        trace=None
    ) -> Dict[str, Any]:
        """Execute agent asynchronously"""
        logger.info(f"[AGENT NODE DEBUG] Agent Bridge async execution starting for: {agent_name}")
        
        loop = asyncio.get_event_loop()
        
        # Use ThreadPoolExecutor to run sync execution in async context
        with ThreadPoolExecutor() as executor:
            future = loop.run_in_executor(
                executor, 
                self.execute_agent_sync, 
                agent_name, 
                query, 
                context,
                trace
            )
            result = await future
            
            logger.info(f"[AGENT NODE DEBUG] Agent Bridge async execution completed for: {agent_name}")
            logger.info(f"[AGENT NODE DEBUG] Success: {result.get('success', False)}")
            
            return result
    
    def get_agent_capabilities(self, agent_name: str) -> Dict[str, Any]:
        """Get capabilities and tools for a specific agent"""
        agent_config = self.get_agent_config(agent_name)
        if not agent_config:
            return {}
        
        return {
            "tools": agent_config.get("tools", []),
            "capabilities": agent_config.get("capabilities", {}),
            "config": agent_config.get("config", {}),
            "role": agent_config.get("role", "")
        }

# Global instance for use in Langflow nodes
agent_bridge = AgentBridge()