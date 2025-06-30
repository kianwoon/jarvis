"""
API endpoints for agent templates and communication patterns
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
import json

from app.core.db import SessionLocal

router = APIRouter()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic models
class AgentTemplate(BaseModel):
    name: str
    description: Optional[str]
    capabilities: List[str] = []
    expected_input: Dict[str, Any] = {}
    output_format: Dict[str, Any] = {}
    default_instructions: Optional[str]

class CommunicationPattern(BaseModel):
    from_agent: str
    to_agent: str
    pattern_name: Optional[str]
    handoff_data: List[str] = []
    instructions_template: Optional[str]
    data_transformation: Dict[str, Any] = {}

class PipelineTemplate(BaseModel):
    name: str
    description: Optional[str]
    category: Optional[str]
    agent_sequence: List[Dict[str, Any]]
    default_config: Dict[str, Any] = {}
    is_active: bool = True

# Agent Template endpoints
@router.get("/")
def list_agent_templates(db: Session = Depends(get_db)):
    """List all agent templates"""
    query = text("SELECT * FROM agent_templates ORDER BY name")
    result = db.execute(query)
    templates = []
    for row in result:
        templates.append({
            "id": row.id,
            "name": row.name,
            "description": row.description,
            "capabilities": row.capabilities,
            "expected_input": row.expected_input,
            "output_format": row.output_format,
            "default_instructions": row.default_instructions
        })
    return templates

@router.get("/{name}")
def get_agent_template(name: str, db: Session = Depends(get_db)):
    """Get a specific agent template"""
    query = text("SELECT * FROM agent_templates WHERE name = :name")
    result = db.execute(query, {"name": name}).fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="Template not found")
    return {
        "id": result.id,
        "name": result.name,
        "description": result.description,
        "capabilities": result.capabilities,
        "expected_input": result.expected_input,
        "output_format": result.output_format,
        "default_instructions": result.default_instructions
    }

@router.post("/")
def create_agent_template(template: AgentTemplate, db: Session = Depends(get_db)):
    """Create a new agent template"""
    query = text("""
        INSERT INTO agent_templates 
        (name, description, capabilities, expected_input, output_format, default_instructions)
        VALUES (:name, :description, :capabilities, :expected_input, :output_format, :default_instructions)
        RETURNING id
    """)
    try:
        result = db.execute(query, {
            "name": template.name,
            "description": template.description,
            "capabilities": json.dumps(template.capabilities),
            "expected_input": json.dumps(template.expected_input),
            "output_format": json.dumps(template.output_format),
            "default_instructions": template.default_instructions
        })
        db.commit()
        template_id = result.fetchone()[0]
        return {"id": template_id, "message": "Template created successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

# Communication Pattern endpoints
@router.get("/communication-patterns")
def list_communication_patterns(db: Session = Depends(get_db)):
    """List all communication patterns"""
    query = text("SELECT * FROM agent_communication_patterns ORDER BY from_agent, to_agent")
    result = db.execute(query)
    patterns = []
    for row in result:
        patterns.append({
            "id": row.id,
            "from_agent": row.from_agent,
            "to_agent": row.to_agent,
            "pattern_name": row.pattern_name,
            "handoff_data": row.handoff_data,
            "instructions_template": row.instructions_template,
            "data_transformation": row.data_transformation
        })
    return patterns

@router.get("/communication-patterns/{from_agent}/{to_agent}")
def get_communication_pattern(from_agent: str, to_agent: str, db: Session = Depends(get_db)):
    """Get communication pattern between two agents"""
    query = text("""
        SELECT * FROM agent_communication_patterns 
        WHERE from_agent = :from_agent AND to_agent = :to_agent
    """)
    result = db.execute(query, {"from_agent": from_agent, "to_agent": to_agent}).fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="Communication pattern not found")
    return {
        "id": result.id,
        "from_agent": result.from_agent,
        "to_agent": result.to_agent,
        "pattern_name": result.pattern_name,
        "handoff_data": result.handoff_data,
        "instructions_template": result.instructions_template,
        "data_transformation": result.data_transformation
    }

@router.post("/communication-patterns")
def create_communication_pattern(pattern: CommunicationPattern, db: Session = Depends(get_db)):
    """Create a new communication pattern"""
    query = text("""
        INSERT INTO agent_communication_patterns 
        (from_agent, to_agent, pattern_name, handoff_data, instructions_template, data_transformation)
        VALUES (:from_agent, :to_agent, :pattern_name, :handoff_data, :instructions_template, :data_transformation)
        ON CONFLICT (from_agent, to_agent) 
        DO UPDATE SET 
            pattern_name = EXCLUDED.pattern_name,
            handoff_data = EXCLUDED.handoff_data,
            instructions_template = EXCLUDED.instructions_template,
            data_transformation = EXCLUDED.data_transformation
        RETURNING id
    """)
    try:
        result = db.execute(query, {
            "from_agent": pattern.from_agent,
            "to_agent": pattern.to_agent,
            "pattern_name": pattern.pattern_name,
            "handoff_data": json.dumps(pattern.handoff_data),
            "instructions_template": pattern.instructions_template,
            "data_transformation": json.dumps(pattern.data_transformation)
        })
        db.commit()
        pattern_id = result.fetchone()[0]
        return {"id": pattern_id, "message": "Communication pattern created/updated successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

# Pipeline Template endpoints
@router.get("/pipeline-templates")
def list_pipeline_templates(
    category: Optional[str] = None,
    is_active: Optional[bool] = True,
    db: Session = Depends(get_db)
):
    """List pipeline templates"""
    query = "SELECT * FROM pipeline_templates WHERE 1=1"
    params = {}
    
    if category:
        query += " AND category = :category"
        params["category"] = category
    
    if is_active is not None:
        query += " AND is_active = :is_active"
        params["is_active"] = is_active
    
    query += " ORDER BY name"
    
    result = db.execute(text(query), params)
    templates = []
    for row in result:
        templates.append({
            "id": row.id,
            "name": row.name,
            "description": row.description,
            "category": row.category,
            "agent_sequence": row.agent_sequence,
            "default_config": row.default_config,
            "is_active": row.is_active
        })
    return templates

@router.get("/pipeline-templates/{name}")
def get_pipeline_template(name: str, db: Session = Depends(get_db)):
    """Get a specific pipeline template"""
    query = text("SELECT * FROM pipeline_templates WHERE name = :name")
    result = db.execute(query, {"name": name}).fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="Pipeline template not found")
    return {
        "id": result.id,
        "name": result.name,
        "description": result.description,
        "category": result.category,
        "agent_sequence": result.agent_sequence,
        "default_config": result.default_config,
        "is_active": result.is_active
    }

@router.post("/pipeline-templates")
def create_pipeline_template(template: PipelineTemplate, db: Session = Depends(get_db)):
    """Create a new pipeline template"""
    query = text("""
        INSERT INTO pipeline_templates 
        (name, description, category, agent_sequence, default_config, is_active)
        VALUES (:name, :description, :category, :agent_sequence, :default_config, :is_active)
        RETURNING id
    """)
    try:
        result = db.execute(query, {
            "name": template.name,
            "description": template.description,
            "category": template.category,
            "agent_sequence": json.dumps(template.agent_sequence),
            "default_config": json.dumps(template.default_config),
            "is_active": template.is_active
        })
        db.commit()
        template_id = result.fetchone()[0]
        return {"id": template_id, "message": "Pipeline template created successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/pipeline-templates/{name}")
def update_pipeline_template(name: str, template: PipelineTemplate, db: Session = Depends(get_db)):
    """Update a pipeline template"""
    query = text("""
        UPDATE pipeline_templates 
        SET description = :description,
            category = :category,
            agent_sequence = :agent_sequence,
            default_config = :default_config,
            is_active = :is_active,
            updated_at = CURRENT_TIMESTAMP
        WHERE name = :name
        RETURNING id
    """)
    try:
        result = db.execute(query, {
            "name": name,
            "description": template.description,
            "category": template.category,
            "agent_sequence": json.dumps(template.agent_sequence),
            "default_config": json.dumps(template.default_config),
            "is_active": template.is_active
        })
        db.commit()
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Pipeline template not found")
        return {"message": "Pipeline template updated successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

