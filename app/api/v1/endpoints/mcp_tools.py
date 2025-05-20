from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.core.db import SessionLocal, MCPTool
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic models for API
class MCPToolBase(BaseModel):
    name: str
    description: Optional[str] = None
    endpoint: str
    method: str = "POST"
    parameters: Optional[dict] = None
    headers: Optional[dict] = None
    is_active: bool = True

class MCPToolCreate(MCPToolBase):
    pass

class MCPToolUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    parameters: Optional[dict] = None
    headers: Optional[dict] = None
    is_active: Optional[bool] = None

class MCPToolResponse(MCPToolBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

# CRUD operations
@router.post("/", response_model=MCPToolResponse)
def create_mcp_tool(tool: MCPToolCreate, db: Session = Depends(get_db)):
    db_tool = db.query(MCPTool).filter(MCPTool.name == tool.name).first()
    if db_tool:
        raise HTTPException(status_code=400, detail="Tool with this name already exists")
    
    db_tool = MCPTool(
        name=tool.name,
        description=tool.description,
        endpoint=tool.endpoint,
        method=tool.method,
        parameters=tool.parameters,
        headers=tool.headers,
        is_active=tool.is_active
    )
    db.add(db_tool)
    db.commit()
    db.refresh(db_tool)
    return db_tool

@router.get("/", response_model=List[MCPToolResponse])
def get_mcp_tools(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    tools = db.query(MCPTool).offset(skip).limit(limit).all()
    return tools

@router.get("/{tool_id}", response_model=MCPToolResponse)
def get_mcp_tool(tool_id: int, db: Session = Depends(get_db)):
    db_tool = db.query(MCPTool).filter(MCPTool.id == tool_id).first()
    if db_tool is None:
        raise HTTPException(status_code=404, detail="Tool not found")
    return db_tool

@router.put("/{tool_id}", response_model=MCPToolResponse)
def update_mcp_tool(tool_id: int, tool: MCPToolUpdate, db: Session = Depends(get_db)):
    db_tool = db.query(MCPTool).filter(MCPTool.id == tool_id).first()
    if db_tool is None:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    # Update attributes that are provided
    update_data = tool.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_tool, key, value)
    
    db.commit()
    db.refresh(db_tool)
    return db_tool

@router.delete("/{tool_id}", response_model=MCPToolResponse)
def delete_mcp_tool(tool_id: int, db: Session = Depends(get_db)):
    db_tool = db.query(MCPTool).filter(MCPTool.id == tool_id).first()
    if db_tool is None:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    db.delete(db_tool)
    db.commit()
    return db_tool 