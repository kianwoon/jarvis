import redis
import json
import os

REDIS_HOST = os.getenv("REDIS_HOST", "redis") 
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
LANGGRAPH_AGENTS_KEY = "langgraph_agents"

# Don't create connection at import time
r = None

def _get_redis_client():
    """Get Redis client with lazy initialization"""
    global r
    if r is None:
        try:
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            r.ping()
            print("Redis connected for LangGraph agents cache")
        except Exception as e:
            print(f"Redis connection failed for LangGraph agents: {e}")
            r = None
    return r

def get_langgraph_agents():
    """Get all LangGraph agents from cache"""
    # First try to get from database if Redis is not available
    redis_client = _get_redis_client()
    if not redis_client:
        # Fallback to loading from database directly
        try:
            from app.core.db import SessionLocal, LangGraphAgent
            db = SessionLocal()
            try:
                agents = db.query(LangGraphAgent).all()
                return {
                    agent.name: {
                        "id": agent.id,
                        "name": agent.name,
                        "role": agent.role,
                        "system_prompt": agent.system_prompt,
                        "tools": agent.tools,
                        "description": agent.description,
                        "is_active": agent.is_active,
                        "config": agent.config or {},
                        "created_at": agent.created_at.isoformat() if agent.created_at else None,
                        "updated_at": agent.updated_at.isoformat() if agent.updated_at else None
                    }
                    for agent in agents
                }
            finally:
                db.close()
        except Exception as e:
            print(f"Error loading agents from database: {e}")
            return {}
    
    try:
        agents_json = redis_client.get(LANGGRAPH_AGENTS_KEY)
        if agents_json:
            return json.loads(agents_json)
        # If not in cache, try to load from database
        return {}
    except Exception as e:
        print(f"Error getting LangGraph agents from cache: {e}")
        return {}

def get_agent_by_name(agent_name: str):
    """Get a specific agent by name (case-insensitive)"""
    agents = get_langgraph_agents()
    
    # First try exact match
    if agent_name in agents:
        return agents[agent_name]
    
    # Then try case-insensitive match
    agent_name_lower = agent_name.lower()
    for name, agent_data in agents.items():
        if name.lower() == agent_name_lower:
            return agent_data
    
    return None

def get_agent_by_role(role: str):
    """Get all agents with a specific role"""
    agents = get_langgraph_agents()
    return {name: agent for name, agent in agents.items() if agent.get("role") == role}

def get_active_agents():
    """Get all active agents"""
    agents = get_langgraph_agents()
    return {name: agent for name, agent in agents.items() if agent.get("is_active", True)}

def reload_cache_from_db():
    """Reload cache from database - called by API endpoint"""
    # This is handled by the API endpoint which has DB access
    pass