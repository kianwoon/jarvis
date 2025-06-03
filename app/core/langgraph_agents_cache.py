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

def _load_agents_from_database():
    """Load agents directly from database"""
    try:
        from app.core.db import SessionLocal, LangGraphAgent
        db = SessionLocal()
        try:
            agents = db.query(LangGraphAgent).all()
            agents_dict = {
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
            print(f"[DEBUG] Loaded {len(agents_dict)} agents from database: {list(agents_dict.keys())}")
            return agents_dict
        finally:
            db.close()
    except Exception as e:
        print(f"[ERROR] Error loading agents from database: {e}")
        return {}

def _update_redis_cache(agents_dict):
    """Update Redis cache with agents data"""
    redis_client = _get_redis_client()
    if redis_client:
        try:
            redis_client.set(LANGGRAPH_AGENTS_KEY, json.dumps(agents_dict))
            print(f"[DEBUG] Updated Redis cache with {len(agents_dict)} agents")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to update Redis cache: {e}")
    return False

def get_langgraph_agents():
    """Get all LangGraph agents from cache with intelligent fallback"""
    redis_client = _get_redis_client()
    
    # Try Redis first if available
    if redis_client:
        try:
            agents_json = redis_client.get(LANGGRAPH_AGENTS_KEY)
            if agents_json:
                agents_dict = json.loads(agents_json)
                print(f"[DEBUG] Retrieved {len(agents_dict)} agents from Redis cache")
                return agents_dict
            else:
                print("[DEBUG] Redis cache is empty, loading from database and warming cache")
                # Cache miss - load from database and update cache
                agents_dict = _load_agents_from_database()
                if agents_dict:
                    _update_redis_cache(agents_dict)
                return agents_dict
        except Exception as e:
            print(f"[ERROR] Redis cache error: {e}, falling back to database")
            # Redis error - fall back to database
            return _load_agents_from_database()
    else:
        print("[DEBUG] Redis not available, loading directly from database")
        # Redis not available - load from database
        return _load_agents_from_database()

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
    print("[DEBUG] Forcing reload of agents cache from database")
    agents_dict = _load_agents_from_database()
    if agents_dict:
        _update_redis_cache(agents_dict)
        return agents_dict
    return {}

def validate_and_warm_cache():
    """Ensure Redis cache is properly populated with agents"""
    redis_client = _get_redis_client()
    if not redis_client:
        print("[DEBUG] Redis not available, cache warming skipped")
        return False
    
    try:
        # Check if cache exists and has data
        agents_json = redis_client.get(LANGGRAPH_AGENTS_KEY)
        if agents_json:
            agents_dict = json.loads(agents_json)
            if len(agents_dict) > 0:
                print(f"[DEBUG] Redis cache is warm with {len(agents_dict)} agents")
                return True
            else:
                print("[DEBUG] Redis cache exists but is empty")
        else:
            print("[DEBUG] Redis cache does not exist")
        
        # Cache is missing or empty - warm it up
        print("[DEBUG] Warming Redis cache from database")
        agents_dict = _load_agents_from_database()
        if agents_dict:
            _update_redis_cache(agents_dict)
            print(f"[DEBUG] Successfully warmed cache with {len(agents_dict)} agents")
            return True
        else:
            print("[ERROR] No agents found in database to cache")
            return False
            
    except Exception as e:
        print(f"[ERROR] Cache validation failed: {e}")
        return False

def get_cache_status():
    """Get detailed status of Redis cache"""
    redis_client = _get_redis_client()
    status = {
        "redis_available": redis_client is not None,
        "cache_exists": False,
        "agent_count": 0,
        "agents": []
    }
    
    if redis_client:
        try:
            agents_json = redis_client.get(LANGGRAPH_AGENTS_KEY)
            if agents_json:
                agents_dict = json.loads(agents_json)
                status["cache_exists"] = True
                status["agent_count"] = len(agents_dict)
                status["agents"] = list(agents_dict.keys())
        except Exception as e:
            status["error"] = str(e)
    
    return status