import redis
import json
import os
import time
from app.core.timeout_settings_cache import get_agent_response_cache_ttl

LANGGRAPH_AGENTS_KEY = "langgraph_agents"

# Don't create connection at import time
r = None

def _get_redis_client(max_retries: int = 3):
    """Get Redis client with robust connection pattern matching other working implementations"""
    global r
    if r is None:
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        redis_password = os.getenv("REDIS_PASSWORD", None)
        
        # Smart host detection - try multiple hosts in order of preference
        redis_host_env = os.getenv("REDIS_HOST", "redis")
        
        # Determine hosts to try based on environment
        if os.path.exists("/.dockerenv") or "CONTAINER" in os.environ:
            # Running in Docker - try Docker service name first
            hosts_to_try = [redis_host_env, "redis", "localhost", "127.0.0.1"]
        else:
            # Running locally - try localhost first
            hosts_to_try = ["localhost", "127.0.0.1", redis_host_env] if redis_host_env != "localhost" else ["localhost", "127.0.0.1"]
        
        for host in hosts_to_try:
            for attempt in range(max_retries):
                try:
                    r = redis.Redis(
                        host=host,
                        port=redis_port,
                        password=redis_password,
                        decode_responses=True,
                        socket_connect_timeout=2,  # Faster timeout for host testing
                        socket_timeout=2
                    )
                    r.ping()
                    print(f"✓ Redis connected for LangGraph agents cache (host={host}:{redis_port})")
                    return r
                except (redis.ConnectionError, redis.TimeoutError):
                    if attempt < max_retries - 1:
                        time.sleep(0.5)  # Short retry delay for host testing
                    # Continue to next host if all attempts fail
                    break
        
        print(f"Redis connection failed for LangGraph agents after trying hosts {hosts_to_try}")
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
            redis_client.set(LANGGRAPH_AGENTS_KEY, json.dumps(agents_dict), ex=get_agent_response_cache_ttl())
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

def analyze_agent_capabilities(agent_data: dict) -> dict:
    """Analyze agent capabilities based on role, system_prompt, and tools"""
    capabilities = {
        "primary_domain": "general",
        "skills": [],
        "expertise_areas": [],
        "tools_available": agent_data.get("tools", []),
        "interaction_style": "collaborative",
        "complexity_level": "intermediate"
    }
    
    role = agent_data.get("role", "").lower()
    system_prompt = agent_data.get("system_prompt", "").lower()
    combined_text = f"{role} {system_prompt}"
    
    # Analyze primary domain
    if any(keyword in combined_text for keyword in ["sales", "business", "commercial", "revenue"]):
        capabilities["primary_domain"] = "business"
        capabilities["skills"].extend(["negotiation", "strategy", "customer_relations"])
    elif any(keyword in combined_text for keyword in ["technical", "architect", "engineer", "code", "development"]):
        capabilities["primary_domain"] = "technical"
        capabilities["skills"].extend(["system_design", "architecture", "programming"])
    elif any(keyword in combined_text for keyword in ["financial", "finance", "cost", "roi", "budget"]):
        capabilities["primary_domain"] = "financial"
        capabilities["skills"].extend(["financial_analysis", "cost_modeling", "roi_calculation"])
    elif any(keyword in combined_text for keyword in ["research", "document", "knowledge", "analysis"]):
        capabilities["primary_domain"] = "research"
        capabilities["skills"].extend(["information_retrieval", "analysis", "synthesis"])
    elif any(keyword in combined_text for keyword in ["security", "compliance", "risk", "audit"]):
        capabilities["primary_domain"] = "security"
        capabilities["skills"].extend(["risk_assessment", "compliance", "security_analysis"])
    
    # Analyze expertise areas based on role keywords
    expertise_keywords = {
        "leadership": ["ceo", "cto", "cio", "director", "manager", "lead"],
        "strategy": ["strategic", "strategy", "planning", "vision"],
        "execution": ["executor", "implementation", "delivery", "operations"],
        "analysis": ["analyst", "analysis", "research", "investigation"],
        "communication": ["communication", "writing", "presentation", "reporting"],
        "coordination": ["coordinator", "orchestrator", "manager", "synthesizer"]
    }
    
    for expertise, keywords in expertise_keywords.items():
        if any(keyword in combined_text for keyword in keywords):
            capabilities["expertise_areas"].append(expertise)
    
    # Determine complexity level
    if any(keyword in combined_text for keyword in ["senior", "expert", "advanced", "architect"]):
        capabilities["complexity_level"] = "advanced"
    elif any(keyword in combined_text for keyword in ["junior", "basic", "simple"]):
        capabilities["complexity_level"] = "basic"
    
    return capabilities

def get_agents_with_capabilities():
    """Get all agents with their analyzed capabilities"""
    agents = get_langgraph_agents()
    agents_with_capabilities = {}
    
    for name, agent_data in agents.items():
        if agent_data.get("is_active", True):
            capabilities = analyze_agent_capabilities(agent_data)
            agent_with_capabilities = {
                **agent_data,
                "capabilities": capabilities
            }
            agents_with_capabilities[name] = agent_with_capabilities
    
    return agents_with_capabilities

def find_agents_by_domain(domain: str):
    """Find agents specializing in a specific domain"""
    agents = get_agents_with_capabilities()
    return {
        name: agent for name, agent in agents.items()
        if agent.get("capabilities", {}).get("primary_domain") == domain
    }

def find_agents_by_skill(skill: str):
    """Find agents with a specific skill"""
    agents = get_agents_with_capabilities()
    return {
        name: agent for name, agent in agents.items()
        if skill in agent.get("capabilities", {}).get("skills", [])
    }

def reload_cache_from_db():
    """Reload cache from database - called by API endpoint"""
    print("[DEBUG] Forcing reload of agents cache from database")
    agents_dict = _load_agents_from_database()
    if agents_dict:
        _update_redis_cache(agents_dict)
        return agents_dict
    return {}

def reload_langgraph_agents():
    """Alias for reload_cache_from_db for backward compatibility"""
    return reload_cache_from_db()

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