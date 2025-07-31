"""
API endpoints for managing LLM prompts using settings table.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.core.db import get_db
from app.services.settings_prompt_service import get_prompt_service as get_settings_prompt_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/prompts", tags=["prompts"])

# Pydantic models for API
class PromptRequest(BaseModel):
    name: str
    description: Optional[str] = None
    prompt_template: str
    prompt_type: str
    parameters: Optional[Dict[str, Any]] = None

class PromptResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    prompt_type: str
    version: int
    is_active: bool
    parameters: Optional[Dict[str, Any]]
    created_at: str = "2024-01-01T00:00:00Z"
    updated_at: str = "2024-01-01T00:00:00Z"

class PromptUpdateRequest(BaseModel):
    description: Optional[str] = None
    prompt_template: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None

class PromptListResponse(BaseModel):
    prompts: List[PromptResponse]
    total: int

# Dependency to get prompt service
def get_prompt_service_instance():
    return get_settings_prompt_service()

@router.get("/", response_model=PromptListResponse)
async def list_prompts(
    prompt_type: Optional[str] = None,
    include_inactive: bool = False,
    prompt_service = Depends(get_prompt_service_instance)
):
    """List all available prompts"""
    try:
        prompts = prompt_service.list_prompts()
        
        # Convert to response format
        response_prompts = []
        for prompt in prompts:
            prompt_type = prompt.get("type", "custom")
            response_prompts.append(PromptResponse(
                id=prompt.get("id", prompt_type),
                name=prompt.get("name", prompt_type),
                description=prompt.get("description", ""),
                prompt_type=prompt_type,
                version=prompt.get("version", 1),
                is_active=True,
                parameters=prompt
            ))
        
        # Filter by type if specified
        if prompt_type:
            response_prompts = [p for p in response_prompts if p.prompt_type == prompt_type]
        
        return PromptListResponse(prompts=response_prompts, total=len(response_prompts))
        
    except Exception as e:
        logger.error(f"Error listing prompts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list prompts: {str(e)}")

@router.get("/{prompt_name}", response_model=PromptResponse)
async def get_prompt(
    prompt_name: str,
    prompt_service = Depends(get_prompt_service_instance)
):
    """Get a specific prompt by name from settings"""
    try:
        prompts = prompt_service.list_prompts()
        prompt = next((p for p in prompts if p["type"] == prompt_name), None)
        
        if not prompt:
            raise HTTPException(status_code=404, detail=f"Prompt '{prompt_name}' not found")
        
        return PromptResponse(
            id=prompt.get("id", prompt["type"]),
            name=prompt["name"],
            description=prompt.get("description", ""),
            prompt_type=prompt["type"],
            version=prompt.get("version", 1),
            is_active=True,
            parameters=prompt
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prompt {prompt_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get prompt: {str(e)}")

@router.post("/", response_model=PromptResponse)
async def create_prompt(
    request: PromptRequest,
    prompt_service = Depends(get_prompt_service_instance)
):
    """Create a new prompt"""
    try:
        # For now, just return a mock response
        # In the future, this would create a new prompt in the database
        return PromptResponse(
            id=request.name.lower().replace(" ", "_"),
            name=request.name,
            description=request.description,
            prompt_type=request.prompt_type,
            version=1,
            is_active=True,
            parameters=request.parameters or {}
        )
        
    except Exception as e:
        logger.error(f"Error creating prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create prompt: {str(e)}")

@router.put("/{prompt_name}", response_model=PromptResponse)
async def update_prompt(
    prompt_name: str,
    request: PromptUpdateRequest,
    prompt_service = Depends(get_prompt_service_instance)
):
    """Update an existing prompt"""
    try:
        success = prompt_service.update_prompt(prompt_name, request.prompt_template or "")
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Prompt '{prompt_name}' not found")
        
        # Return updated prompt
        prompts = prompt_service.list_prompts()
        prompt = next((p for p in prompts if p["name"] == prompt_name), None)
        
        return PromptResponse(
            id=prompt.get("id", prompt["name"]),
            name=prompt["name"],
            description=prompt.get("description", ""),
            prompt_type=prompt["type"],
            version=prompt.get("version", 1) + 1,
            is_active=prompt.get("is_active", True),
            parameters=prompt.get("parameters", {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating prompt {prompt_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update prompt: {str(e)}")

@router.get("/{prompt_name}/template")
async def get_prompt_template(
    prompt_name: str,
    prompt_service = Depends(get_prompt_service_instance)
):
    """Get the raw prompt template from settings"""
    try:
        template = prompt_service.get_prompt(prompt_name)
        return {"prompt_template": template}
        
    except Exception as e:
        logger.error(f"Error getting prompt template {prompt_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get prompt template: {str(e)}")

@router.post("/{prompt_name}/preview")
async def preview_prompt(
    prompt_name: str,
    variables: Dict[str, Any],
    prompt_service = Depends(get_prompt_service_instance)
):
    """Preview a prompt with variables substituted"""
    try:
        preview = prompt_service.get_prompt(prompt_name, variables)
        return {"preview": preview}
        
    except Exception as e:
        logger.error(f"Error previewing prompt {prompt_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to preview prompt: {str(e)}")

@router.get("/health")
async def health_check(
    prompt_service = Depends(get_prompt_service_instance)
):
    """Health check for prompt service"""
    try:
        prompts = prompt_service.list_prompts()
        return {
            "status": "healthy",
            "prompts_available": len(prompts),
            "source": "settings_table"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "source": "settings_table"
        }