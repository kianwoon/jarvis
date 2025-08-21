"""
Meta-Task API Endpoints
Handles API requests for meta-task functionality
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import logging
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.services.meta_task.template_manager import MetaTaskTemplateManager
from app.services.meta_task.workflow_orchestrator import MetaTaskWorkflowOrchestrator
from app.services.meta_task.execution_engine import MetaTaskExecutionEngine

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic Models
class CreateTemplateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    template_type: str
    template_config: Dict[str, Any]
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    default_settings: Optional[Dict[str, Any]] = None

class UpdateTemplateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    template_type: Optional[str] = None
    template_config: Optional[Dict[str, Any]] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    default_settings: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None

class CreateWorkflowRequest(BaseModel):
    template_id: str
    name: str
    description: Optional[str] = None
    input_data: Dict[str, Any]

class ExecutePhaseRequest(BaseModel):
    phase_config: Dict[str, Any]
    input_data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None

class ExecuteWorkflowRequest(BaseModel):
    phases: List[Dict[str, Any]]
    input_data: Dict[str, Any]

# Initialize services
template_manager = MetaTaskTemplateManager()
workflow_orchestrator = MetaTaskWorkflowOrchestrator()
execution_engine = MetaTaskExecutionEngine()

# Template Endpoints
@router.get("/templates")
async def get_templates(active_only: bool = True):
    """Get all meta-task templates"""
    try:
        templates = await template_manager.get_templates(active_only=active_only)
        return {
            "templates": templates,
            "count": len(templates)
        }
    except Exception as e:
        logger.error(f"Error getting templates: {e}")
        raise HTTPException(status_code=500, detail="Failed to get templates")

@router.get("/templates/{template_id}")
async def get_template(template_id: str):
    """Get a specific template"""
    try:
        template = await template_manager.get_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        return template
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting template {template_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get template")

@router.post("/templates")
async def create_template(request: CreateTemplateRequest):
    """Create a new template"""
    try:
        # Validate template config
        validation = await template_manager.validate_template_config(request.template_config)
        if not validation['valid']:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid template config: {', '.join(validation['errors'])}"
            )
        
        template = await template_manager.create_template(request.dict())
        if not template:
            raise HTTPException(status_code=400, detail="Failed to create template")
        
        return template
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating template: {e}")
        raise HTTPException(status_code=500, detail="Failed to create template")

@router.put("/templates/{template_id}")
async def update_template(template_id: str, request: UpdateTemplateRequest):
    """Update an existing template"""
    try:
        updates = {k: v for k, v in request.dict().items() if v is not None}
        
        # Validate template config if provided
        if 'template_config' in updates:
            validation = await template_manager.validate_template_config(updates['template_config'])
            if not validation['valid']:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid template config: {', '.join(validation['errors'])}"
                )
        
        template = await template_manager.update_template(template_id, updates)
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        
        return template
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating template {template_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update template")

@router.delete("/templates/{template_id}")
async def delete_template(template_id: str):
    """Delete a template"""
    try:
        success = await template_manager.delete_template(template_id)
        if not success:
            raise HTTPException(status_code=404, detail="Template not found")
        
        return {"message": "Template deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting template {template_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete template")

# Workflow Endpoints
@router.get("/workflows")
async def get_workflows():
    """Get all workflows"""
    try:
        workflows = await workflow_orchestrator.get_workflows()
        return {
            "workflows": workflows,
            "count": len(workflows)
        }
    except Exception as e:
        logger.error(f"Error getting workflows: {e}")
        raise HTTPException(status_code=500, detail="Failed to get workflows")

@router.post("/workflows")
async def create_workflow(request: CreateWorkflowRequest):
    """Create a new workflow from a template"""
    try:
        workflow = await workflow_orchestrator.create_workflow(
            template_id=request.template_id,
            name=request.name,
            description=request.description,
            input_data=request.input_data
        )
        
        if not workflow:
            raise HTTPException(status_code=400, detail="Failed to create workflow")
        
        return workflow
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating workflow: {e}")
        raise HTTPException(status_code=500, detail="Failed to create workflow")

@router.get("/workflows/{workflow_id}/execute")
async def execute_workflow(workflow_id: str):
    """Execute a workflow and stream progress"""
    try:
        async def event_generator():
            async for event in workflow_orchestrator.execute_workflow(workflow_id):
                yield f"data: {json.dumps(event)}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
    except Exception as e:
        logger.error(f"Error executing workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to execute workflow")

# Execution Endpoints
@router.post("/execute/phase")
async def execute_phase(request: ExecutePhaseRequest):
    """Execute a single phase"""
    try:
        result = await execution_engine.execute_phase(
            phase_config=request.phase_config,
            input_data=request.input_data,
            context=request.context
        )
        
        return result
    except Exception as e:
        logger.error(f"Error executing phase: {e}")
        raise HTTPException(status_code=500, detail="Failed to execute phase")

@router.post("/execute/workflow")
async def execute_multi_phase_workflow(request: ExecuteWorkflowRequest):
    """Execute a multi-phase workflow and stream progress"""
    try:
        async def event_generator():
            async for event in execution_engine.execute_multi_phase_workflow(
                phases=request.phases,
                initial_input=request.input_data
            ):
                yield f"data: {json.dumps(event)}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
    except Exception as e:
        logger.error(f"Error executing multi-phase workflow: {e}")
        raise HTTPException(status_code=500, detail="Failed to execute workflow")

# Utility Endpoints
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "meta-task",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/stats")
async def get_stats():
    """Get meta-task system statistics"""
    try:
        # This would typically query the database for actual stats
        return {
            "total_templates": 2,  # Would be queried from DB
            "active_workflows": 0,
            "completed_workflows": 0,
            "total_executions": 0,
            "last_updated": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get stats")