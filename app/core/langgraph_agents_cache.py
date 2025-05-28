import redis
import json
import os

REDIS_HOST = os.getenv("REDIS_HOST", "redis") 
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
LANGGRAPH_AGENTS_KEY = "langgraph_agents"

try:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    r.ping()
    print("Redis connected for LangGraph agents cache")
except Exception as e:
    print(f"Redis connection failed for LangGraph agents: {e}")
    r = None

def get_langgraph_agents():
    """Get all LangGraph agents from cache"""
    if not r:
        return {}
    
    try:
        agents_json = r.get(LANGGRAPH_AGENTS_KEY)
        if agents_json:
            return json.loads(agents_json)
        return {}
    except Exception as e:
        print(f"Error getting LangGraph agents from cache: {e}")
        return {}

def get_agent_by_name(agent_name: str):
    """Get a specific agent by name"""
    agents = get_langgraph_agents()
    return agents.get(agent_name)

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