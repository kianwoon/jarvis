"""
Pipeline Agents Cache

CRITICAL: This module handles ONLY agentic pipeline agents from pipeline_agents table.
It is completely separate from langgraph_agents and should NEVER mix the two.
"""

import logging
from typing import Dict, Any, Optional, List
from functools import lru_cache
from app.core.db import SessionLocal
from sqlalchemy import text

logger = logging.getLogger(__name__)

def get_pipeline_agent_config(pipeline_id: int, agent_name: str) -> Optional[Dict[str, Any]]:
    """
    Get pipeline agent configuration from pipeline_agents table ONLY.
    
    Args:
        pipeline_id: ID of the pipeline
        agent_name: Name of the agent
        
    Returns:
        Agent configuration dict or None if not found
    """
    db = SessionLocal()
    try:
        result = db.execute(text("""
            SELECT config FROM pipeline_agents 
            WHERE pipeline_id = :pipeline_id AND agent_name = :agent_name
        """), {"pipeline_id": pipeline_id, "agent_name": agent_name})
        row = result.fetchone()
        
        if row and row.config:
            logger.info(f"[PIPELINE CACHE] Found config for {agent_name} in pipeline {pipeline_id}: tools={row.config.get('tools', [])}")
            return row.config
        else:
            logger.error(f"[PIPELINE CACHE] No config found for {agent_name} in pipeline {pipeline_id}")
            return None
            
    except Exception as e:
        logger.error(f"[PIPELINE CACHE] Failed to get pipeline agent config: {e}")
        return None
    finally:
        db.close()

def get_pipeline_agent_tools(pipeline_id: int, agent_name: str) -> List[str]:
    """
    Get tools for a pipeline agent from pipeline_agents table ONLY.
    
    Args:
        pipeline_id: ID of the pipeline
        agent_name: Name of the agent
        
    Returns:
        List of tool names
    """
    config = get_pipeline_agent_config(pipeline_id, agent_name)
    if config and "tools" in config:
        tools = config["tools"]
        if isinstance(tools, list):
            logger.info(f"[PIPELINE CACHE] Agent {agent_name} in pipeline {pipeline_id} has tools: {tools}")
            return tools
        else:
            logger.warning(f"[PIPELINE CACHE] Tools for {agent_name} not in list format: {type(tools)}")
            return []
    else:
        logger.error(f"[PIPELINE CACHE] No tools found for {agent_name} in pipeline {pipeline_id}")
        return []

def get_all_pipeline_agents(pipeline_id: int) -> Dict[str, Dict[str, Any]]:
    """
    Get all agents for a specific pipeline.
    
    Args:
        pipeline_id: ID of the pipeline
        
    Returns:
        Dict mapping agent names to their configurations
    """
    db = SessionLocal()
    try:
        result = db.execute(text("""
            SELECT agent_name, config FROM pipeline_agents 
            WHERE pipeline_id = :pipeline_id
        """), {"pipeline_id": pipeline_id})
        
        agents = {}
        for row in result.fetchall():
            agents[row.agent_name] = row.config
            
        logger.info(f"[PIPELINE CACHE] Found {len(agents)} agents for pipeline {pipeline_id}")
        return agents
        
    except Exception as e:
        logger.error(f"[PIPELINE CACHE] Failed to get pipeline agents: {e}")
        return {}
    finally:
        db.close()

def validate_pipeline_agent_tools(pipeline_id: int, agent_name: str, tools_to_validate: List[str]) -> bool:
    """
    Validate that all tools are allowed for this pipeline agent.
    
    Args:
        pipeline_id: ID of the pipeline
        agent_name: Name of the agent
        tools_to_validate: List of tools to validate
        
    Returns:
        True if all tools are allowed, False otherwise
    """
    allowed_tools = get_pipeline_agent_tools(pipeline_id, agent_name)
    
    for tool in tools_to_validate:
        if tool not in allowed_tools:
            logger.error(f"[PIPELINE CACHE] SECURITY: Tool {tool} NOT allowed for {agent_name} in pipeline {pipeline_id}. Allowed: {allowed_tools}")
            return False
    
    logger.info(f"[PIPELINE CACHE] All tools validated for {agent_name} in pipeline {pipeline_id}")
    return True