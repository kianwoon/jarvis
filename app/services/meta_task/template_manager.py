"""
Meta-Task Template Manager
Manages template definitions and configurations for different document types
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from app.core.db import MetaTaskTemplate, get_db_session

logger = logging.getLogger(__name__)

class MetaTaskTemplateManager:
    """Manages meta-task templates"""
    
    def __init__(self):
        self.logger = logger
    
    async def get_templates(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get all available templates"""
        try:
            with get_db_session() as db:
                query = db.query(MetaTaskTemplate)
                if active_only:
                    query = query.filter(MetaTaskTemplate.is_active == True)
                
                templates = query.all()
                return [self._template_to_dict(template) for template in templates]
        except Exception as e:
            self.logger.error(f"Error getting templates: {e}")
            return []
    
    async def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific template by ID"""
        try:
            with get_db_session() as db:
                template = db.query(MetaTaskTemplate).filter(
                    MetaTaskTemplate.id == template_id
                ).first()
                
                if template:
                    return self._template_to_dict(template)
                return None
        except Exception as e:
            self.logger.error(f"Error getting template {template_id}: {e}")
            return None
    
    async def create_template(self, template_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a new template"""
        try:
            with get_db_session() as db:
                template = MetaTaskTemplate(
                    name=template_data['name'],
                    description=template_data.get('description'),
                    template_type=template_data['template_type'],
                    template_config=template_data['template_config'],
                    input_schema=template_data.get('input_schema'),
                    output_schema=template_data.get('output_schema'),
                    default_settings=template_data.get('default_settings', {}),
                    is_active=template_data.get('is_active', True)
                )
                
                db.add(template)
                db.commit()
                db.refresh(template)
                
                self.logger.info(f"Created template: {template.name}")
                return self._template_to_dict(template)
        except Exception as e:
            self.logger.error(f"Error creating template: {e}")
            return None
    
    async def update_template(self, template_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update an existing template"""
        try:
            with get_db_session() as db:
                template = db.query(MetaTaskTemplate).filter(
                    MetaTaskTemplate.id == template_id
                ).first()
                
                if not template:
                    return None
                
                # Update fields
                for field, value in updates.items():
                    if hasattr(template, field):
                        setattr(template, field, value)
                
                db.commit()
                db.refresh(template)
                
                self.logger.info(f"Updated template: {template.name}")
                return self._template_to_dict(template)
        except Exception as e:
            self.logger.error(f"Error updating template {template_id}: {e}")
            return None
    
    async def delete_template(self, template_id: str) -> bool:
        """Delete a template"""
        try:
            with get_db_session() as db:
                template = db.query(MetaTaskTemplate).filter(
                    MetaTaskTemplate.id == template_id
                ).first()
                
                if not template:
                    return False
                
                db.delete(template)
                db.commit()
                
                self.logger.info(f"Deleted template: {template.name}")
                return True
        except Exception as e:
            self.logger.error(f"Error deleting template {template_id}: {e}")
            return False
    
    def _template_to_dict(self, template: MetaTaskTemplate) -> Dict[str, Any]:
        """Convert template model to dictionary"""
        return {
            'id': template.id,
            'name': template.name,
            'description': template.description,
            'template_type': template.template_type,
            'template_config': template.template_config,
            'input_schema': template.input_schema,
            'output_schema': template.output_schema,
            'default_settings': template.default_settings,
            'is_active': template.is_active,
            'created_at': template.created_at.isoformat() if template.created_at else None,
            'updated_at': template.updated_at.isoformat() if template.updated_at else None
        }
    
    async def validate_template_config(self, template_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate template configuration"""
        errors = []
        
        # Check required fields
        required_fields = ['phases']
        for field in required_fields:
            if field not in template_config:
                errors.append(f"Missing required field: {field}")
        
        # Validate phases
        if 'phases' in template_config:
            phases = template_config['phases']
            if not isinstance(phases, list) or len(phases) == 0:
                errors.append("Phases must be a non-empty list")
            else:
                for i, phase in enumerate(phases):
                    if not isinstance(phase, dict):
                        errors.append(f"Phase {i} must be a dictionary")
                        continue
                    
                    phase_required = ['name', 'type', 'description']
                    for field in phase_required:
                        if field not in phase:
                            errors.append(f"Phase {i} missing required field: {field}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }