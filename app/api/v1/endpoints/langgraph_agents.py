from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
from app.core.db import get_db, LangGraphAgent as LangGraphAgentDB
from app.core.mcp_tools_cache import get_enabled_mcp_tools
import redis
import json
import os

router = APIRouter()

# Redis connection for caching
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
LANGGRAPH_AGENTS_KEY = "langgraph_agents"

try:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    r.ping()
except:
    print("Warning: Redis not available for LangGraph agents cache")
    r = None

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
        return
    
    agents = db.query(LangGraphAgentDB).all()
    agents_dict = {}
    
    for agent in agents:
        agents_dict[agent.name] = {
            "id": agent.id,
            "name": agent.name,
            "role": agent.role,
            "system_prompt": agent.system_prompt,
            "tools": agent.tools or [],
            "description": agent.description,
            "is_active": agent.is_active,
            "created_at": agent.created_at.isoformat() if agent.created_at else None,
            "updated_at": agent.updated_at.isoformat() if agent.updated_at else None
        }
    
    r.set(LANGGRAPH_AGENTS_KEY, json.dumps(agents_dict))
    print(f"Updated Redis cache with {len(agents_dict)} agents")

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
    agents = db.query(LangGraphAgentDB).all()
    return agents

@router.get("/agents/{agent_id}", response_model=LangGraphAgent)
async def get_agent(agent_id: int, db: Session = Depends(get_db)):
    """Get a specific LangGraph agent"""
    agent = db.query(LangGraphAgentDB).filter(LangGraphAgentDB.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@router.post("/agents", response_model=LangGraphAgent)
async def create_agent(agent: LangGraphAgentCreate, db: Session = Depends(get_db)):
    """Create a new LangGraph agent"""
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
    db.commit()
    db.refresh(db_agent)
    
    # Update Redis cache
    update_redis_cache(db)
    
    return db_agent

@router.delete("/agents/{agent_id}")
async def delete_agent(agent_id: int, db: Session = Depends(get_db)):
    """Delete a LangGraph agent"""
    db_agent = db.query(LangGraphAgentDB).filter(LangGraphAgentDB.id == agent_id).first()
    if not db_agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    db.delete(db_agent)
    db.commit()
    
    # Update Redis cache
    update_redis_cache(db)
    
    return {"message": "Agent deleted successfully"}

@router.post("/agents/cache/reload")
async def reload_agents_cache(db: Session = Depends(get_db)):
    """Reload agents cache in Redis"""
    update_redis_cache(db)
    return {"message": "Cache reloaded successfully"}

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