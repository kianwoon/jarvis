"""
Radiating Settings API Endpoints

Provides endpoints for managing radiating system settings and prompts.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import logging

from app.core.radiating_settings_cache import (
    get_radiating_settings,
    set_radiating_settings,
    reload_radiating_settings,
    get_radiating_prompts,
    update_prompt,
    reload_radiating_prompts,
    get_prompt
)
from app.core.db import get_db_session
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/settings/radiating",
    tags=["radiating-settings"]
)

@router.get("/")
async def get_settings():
    """Get all radiating settings including prompts."""
    try:
        settings = get_radiating_settings()
        return {
            "status": "success",
            "settings": settings
        }
    except Exception as e:
        logger.error(f"Error getting radiating settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/")
async def update_settings(settings: Dict[str, Any]):
    """Update radiating settings."""
    try:
        set_radiating_settings(settings)
        return {
            "status": "success",
            "message": "Settings updated successfully"
        }
    except Exception as e:
        logger.error(f"Error updating radiating settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/prompts")
async def get_prompts():
    """Get all radiating prompts."""
    try:
        prompts = get_radiating_prompts()
        return {
            "status": "success",
            "prompts": prompts
        }
    except Exception as e:
        logger.error(f"Error getting radiating prompts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/prompts/{category}/{prompt_name}")
async def get_specific_prompt(category: str, prompt_name: str):
    """Get a specific prompt by category and name."""
    try:
        prompt = get_prompt(category, prompt_name)
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")
        
        return {
            "status": "success",
            "category": category,
            "prompt_name": prompt_name,
            "prompt": prompt
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prompt {category}/{prompt_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/prompts/{category}/{prompt_name}")
async def update_specific_prompt(
    category: str,
    prompt_name: str,
    request: Dict[str, str]
):
    """Update a specific prompt."""
    try:
        prompt_text = request.get("prompt")
        if not prompt_text:
            raise HTTPException(status_code=400, detail="Prompt text is required")
        
        update_prompt(category, prompt_name, prompt_text)
        
        return {
            "status": "success",
            "message": f"Prompt {category}/{prompt_name} updated successfully"
        }
    except Exception as e:
        logger.error(f"Error updating prompt {category}/{prompt_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reload")
async def reload_settings():
    """Reload radiating settings from database."""
    try:
        settings = reload_radiating_settings()
        return {
            "status": "success",
            "message": "Settings reloaded from database",
            "settings": settings
        }
    except Exception as e:
        logger.error(f"Error reloading radiating settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reload-prompts")
async def reload_prompts():
    """Force reload of radiating prompts from database."""
    try:
        settings = reload_radiating_prompts()
        prompts = get_radiating_prompts()
        
        return {
            "status": "success",
            "message": "Prompts reloaded from database",
            "prompts": prompts
        }
    except Exception as e:
        logger.error(f"Error reloading radiating prompts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/prompt-stats")
async def get_prompt_statistics():
    """Get statistics about radiating prompts."""
    try:
        prompts = get_radiating_prompts()
        
        stats = {
            "total_categories": len(prompts),
            "categories": {}
        }
        
        total_prompts = 0
        total_chars = 0
        
        for category, category_prompts in prompts.items():
            prompt_count = len(category_prompts)
            char_count = sum(len(p) for p in category_prompts.values())
            
            stats["categories"][category] = {
                "prompt_count": prompt_count,
                "total_characters": char_count,
                "average_length": char_count // prompt_count if prompt_count > 0 else 0,
                "prompts": list(category_prompts.keys())
            }
            
            total_prompts += prompt_count
            total_chars += char_count
        
        stats["total_prompts"] = total_prompts
        stats["total_characters"] = total_chars
        stats["average_prompt_length"] = total_chars // total_prompts if total_prompts > 0 else 0
        
        return {
            "status": "success",
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error getting prompt statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate-prompts")
async def validate_prompts():
    """Validate all prompts for proper formatting and variables."""
    try:
        prompts = get_radiating_prompts()
        validation_results = {}
        
        for category, category_prompts in prompts.items():
            validation_results[category] = {}
            
            for prompt_name, prompt_text in category_prompts.items():
                issues = []
                
                # Check for empty prompts
                if not prompt_text or not prompt_text.strip():
                    issues.append("Prompt is empty")
                
                # Check for unclosed braces
                open_braces = prompt_text.count('{')
                close_braces = prompt_text.count('}')
                if open_braces != close_braces:
                    issues.append(f"Mismatched braces: {open_braces} open, {close_braces} close")
                
                # Extract variables
                import re
                variables = re.findall(r'\{([^}]+)\}', prompt_text)
                
                validation_results[category][prompt_name] = {
                    "valid": len(issues) == 0,
                    "issues": issues,
                    "variables": variables,
                    "length": len(prompt_text)
                }
        
        # Calculate overall validity
        all_valid = all(
            prompt_info["valid"]
            for category_results in validation_results.values()
            for prompt_info in category_results.values()
        )
        
        return {
            "status": "success",
            "all_valid": all_valid,
            "validation_results": validation_results
        }
    except Exception as e:
        logger.error(f"Error validating prompts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reset-to-defaults")
async def reset_prompts_to_defaults():
    """Reset all prompts to their default values."""
    try:
        # This would typically load defaults from a configuration file
        # For now, we'll return a message indicating the action
        return {
            "status": "error",
            "message": "Reset to defaults requires implementation of default prompt storage"
        }
    except Exception as e:
        logger.error(f"Error resetting prompts to defaults: {e}")
        raise HTTPException(status_code=500, detail=str(e))