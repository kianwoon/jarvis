from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
from app.core.mcp_tools_cache import get_enabled_mcp_tools
import redis
import json
import os

router = APIRouter()

# Dependency to get DB session - use local function like working examples
def get_db():
    # Lazy import to avoid database connection at startup
    from app.core.db import SessionLocal
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Redis connection for caching - use robust pattern like other working implementations
LANGGRAPH_AGENTS_KEY = "langgraph_agents"

def _get_redis_connection():
    """Get Redis connection using smart host detection like the fixed langgraph_agents_cache"""
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
        try:
            client = redis.Redis(
                host=host,
                port=redis_port,
                password=redis_password,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2
            )
            client.ping()
            print(f"✓ Redis connected for LangGraph agents API (host={host}:{redis_port})")
            return client
        except (redis.ConnectionError, redis.TimeoutError):
            continue
    
    print(f"Warning: Redis not available for LangGraph agents cache after trying hosts {hosts_to_try}")
    return None

r = _get_redis_connection()

# Pydantic models
class LangGraphAgentBase(BaseModel):
    name: str
    role: str
    system_prompt: str
    tools: List[str] = []
    description: Optional[str] = None
    is_active: bool = True
    config: Optional[dict] = None

class LangGraphAgentCreate(LangGraphAgentBase):
    pass

class LangGraphAgentUpdate(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    system_prompt: Optional[str] = None
    tools: Optional[List[str]] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None
    config: Optional[dict] = None

class LangGraphAgent(LangGraphAgentBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


def update_redis_cache(db: Session):
    """Update Redis cache with all agents"""
    if not r:
        print("[WARNING] Redis not available, cannot update cache")
        return False
    
    try:
        # Lazy import to avoid database connection at startup
        from app.core.db import LangGraphAgent as LangGraphAgentDB
        
        agents = db.query(LangGraphAgentDB).all()
        agents_dict = {}
        
        for agent in agents:
            # Include config in cache data
            agent_config = agent.config or {}
            agents_dict[agent.name] = {
                "id": agent.id,
                "name": agent.name,
                "role": agent.role,
                "system_prompt": agent.system_prompt,
                "tools": agent.tools or [],
                "description": agent.description,
                "is_active": agent.is_active,
                "config": agent_config,
                "created_at": agent.created_at.isoformat() if agent.created_at else None,
                "updated_at": agent.updated_at.isoformat() if agent.updated_at else None
            }
        
        r.set(LANGGRAPH_AGENTS_KEY, json.dumps(agents_dict))
        print(f"[DEBUG] Updated Redis cache with {len(agents_dict)} agents: {list(agents_dict.keys())}")
        
        # Verify the cache was updated
        cached_data = r.get(LANGGRAPH_AGENTS_KEY)
        if cached_data:
            cached_agents = json.loads(cached_data)
            print(f"[DEBUG] Cache verification: {len(cached_agents)} agents now in cache")
        
        return True
    except Exception as e:
        print(f"[ERROR] Failed to update Redis cache: {e}")
        return False

@router.get("/tools/available")
async def get_available_tools():
    """Get all available tools from MCP cache"""
    try:
        enabled_tools = get_enabled_mcp_tools()
        
        # Transform to list format for frontend
        tools_list = []
        for tool_name, tool_info in enabled_tools.items():
            tools_list.append({
                "name": tool_name,
                "description": tool_info.get("description", ""),
                "enabled": tool_info.get("is_active", True)
            })
        
        return {"tools": tools_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch tools: {str(e)}")

@router.get("/agents", response_model=List[LangGraphAgent])
async def list_agents(db: Session = Depends(get_db)):
    """List all LangGraph agents"""
    # Lazy import to avoid database connection at startup
    from app.core.db import LangGraphAgent as LangGraphAgentDB
    
    agents = db.query(LangGraphAgentDB).all()
    return agents

@router.get("/agents/{agent_id}", response_model=LangGraphAgent)
async def get_agent(agent_id: int, db: Session = Depends(get_db)):
    """Get a specific LangGraph agent"""
    # Lazy import to avoid database connection at startup
    from app.core.db import LangGraphAgent as LangGraphAgentDB
    
    agent = db.query(LangGraphAgentDB).filter(LangGraphAgentDB.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@router.post("/agents", response_model=LangGraphAgent)
async def create_agent(agent: LangGraphAgentCreate, db: Session = Depends(get_db)):
    """Create a new LangGraph agent"""
    # Lazy import to avoid database connection at startup
    from app.core.db import LangGraphAgent as LangGraphAgentDB
    
    # Check if agent with same name exists
    existing = db.query(LangGraphAgentDB).filter(LangGraphAgentDB.name == agent.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Agent with this name already exists")
    
    # Validate tools exist
    available_tools = get_enabled_mcp_tools()
    for tool in agent.tools:
        if tool not in available_tools:
            raise HTTPException(status_code=400, detail=f"Tool '{tool}' not found in available tools")
    
    # Create agent
    db_agent = LangGraphAgentDB(**agent.model_dump())
    db.add(db_agent)
    db.commit()
    db.refresh(db_agent)
    
    # Update Redis cache
    update_redis_cache(db)
    
    return db_agent

@router.put("/agents/{agent_id}", response_model=LangGraphAgent)
async def update_agent(agent_id: int, agent_update: LangGraphAgentUpdate, db: Session = Depends(get_db)):
    """Update a LangGraph agent"""
    # Lazy import to avoid database connection at startup
    from app.core.db import LangGraphAgent as LangGraphAgentDB
    
    print(f"[DEBUG] Updating agent {agent_id}")
    print(f"[DEBUG] Update data: {agent_update.model_dump()}")
    
    db_agent = db.query(LangGraphAgentDB).filter(LangGraphAgentDB.id == agent_id).first()
    if not db_agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Check if new name conflicts
    if agent_update.name and agent_update.name != db_agent.name:
        existing = db.query(LangGraphAgentDB).filter(LangGraphAgentDB.name == agent_update.name).first()
        if existing:
            raise HTTPException(status_code=400, detail="Agent with this name already exists")
    
    # Validate tools if provided
    if agent_update.tools is not None:
        available_tools = get_enabled_mcp_tools()
        for tool in agent_update.tools:
            if tool not in available_tools:
                raise HTTPException(status_code=400, detail=f"Tool '{tool}' not found in available tools")
    
    # Update fields
    update_data = agent_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_agent, field, value)
    
    db_agent.updated_at = datetime.now()
    
    print(f"[DEBUG] Before commit - system_prompt: {db_agent.system_prompt[:50]}...")
    print(f"[DEBUG] Before commit - updated_at: {db_agent.updated_at}")
    
    db.commit()
    
    print(f"[DEBUG] After commit - committed successfully")
    
    db.refresh(db_agent)
    
    print(f"[DEBUG] After refresh - system_prompt: {db_agent.system_prompt[:50]}...")
    print(f"[DEBUG] After refresh - updated_at: {db_agent.updated_at}")
    
    # Update Redis cache
    update_redis_cache(db)
    
    return db_agent

@router.delete("/agents/{agent_id}")
async def delete_agent(agent_id: int, db: Session = Depends(get_db)):
    """Delete a LangGraph agent"""
    # Lazy import to avoid database connection at startup
    from app.core.db import LangGraphAgent as LangGraphAgentDB
    
    db_agent = db.query(LangGraphAgentDB).filter(LangGraphAgentDB.id == agent_id).first()
    if not db_agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    db.delete(db_agent)
    db.commit()
    
    # Update Redis cache
    update_redis_cache(db)
    
    return {"message": "Agent deleted successfully"}

@router.get("/agents/cache/status")
async def get_cache_status():
    """Get Redis cache status for agents"""
    try:
        from app.core.langgraph_agents_cache import get_cache_status
        status = get_cache_status()
        return {
            "status": "success",
            "cache_status": status,
            "timestamp": json.dumps(datetime.now().isoformat())
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": json.dumps(datetime.now().isoformat())
        }

@router.post("/agents/cache/reload")
async def reload_agents_cache(db: Session = Depends(get_db)):
    """Reload agents cache in Redis"""
    try:
        success = update_redis_cache(db)
        if success:
            return {"message": "Cache reloaded successfully", "status": "success"}
        else:
            return {"message": "Cache reload failed", "status": "error"}
    except Exception as e:
        return {"message": f"Cache reload failed: {str(e)}", "status": "error"}

@router.post("/agents/cache/warm")
async def warm_agents_cache():
    """Ensure Redis cache is warmed with agents"""
    try:
        from app.core.langgraph_agents_cache import validate_and_warm_cache
        success = validate_and_warm_cache()
        if success:
            return {"message": "Cache warmed successfully", "status": "success"}
        else:
            return {"message": "Cache warming failed", "status": "error"}
    except Exception as e:
        return {"message": f"Cache warming failed: {str(e)}", "status": "error"}

@router.post("/agents/migrate")
async def migrate_agents_table():
    """Add missing columns to langgraph_agents table"""
    try:
        # Check if is_active column exists
        from sqlalchemy import text, inspect
        from app.core.db import engine
        
        inspector = inspect(engine)
        columns = [col['name'] for col in inspector.get_columns('langgraph_agents')]
        
        if 'is_active' not in columns:
            # Add the column
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE langgraph_agents ADD COLUMN is_active BOOLEAN DEFAULT TRUE"))
                conn.commit()
            return {"message": "Successfully added is_active column"}
        else:
            return {"message": "is_active column already exists"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Migration failed: {str(e)}")