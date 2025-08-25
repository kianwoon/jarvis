import json
from app.core.config import get_settings

config = get_settings()
REDIS_HOST = config.REDIS_HOST
REDIS_PORT = config.REDIS_PORT
REDIS_PASSWORD = config.REDIS_PASSWORD
NOTEBOOK_SOURCE_TEMPLATES_KEY = 'notebook_source_templates_cache'

# Don't create connection at import time
r = None

def _get_redis_client():
    """Get Redis client using the centralized redis_client"""
    global r
    if r is None:
        from app.core.redis_client import get_redis_client
        r = get_redis_client()
    return r

def get_notebook_source_templates():
    """
    Get notebook source templates with failsafe system.
    
    Failsafe hierarchy:
    1. Redis cache (fastest)
    2. Database reload (reliable)
    3. Emergency fallback settings (prevents total failure)
    """
    # Layer 1: Try Redis cache first
    redis_client = _get_redis_client()
    if redis_client:
        try:
            cached = redis_client.get(NOTEBOOK_SOURCE_TEMPLATES_KEY)
            if cached:
                settings = json.loads(cached)
                if _validate_source_templates(settings):
                    return settings
                else:
                    print(f"[NOTEBOOK_SOURCE_TEMPLATES] Cached templates invalid, falling back to database")
        except Exception as e:
            print(f"[NOTEBOOK_SOURCE_TEMPLATES] Redis error: {e}, falling back to database")
    
    # Layer 2: Try database reload
    try:
        return reload_notebook_source_templates()
    except Exception as e:
        print(f"[NOTEBOOK_SOURCE_TEMPLATES] Database reload failed: {e}, using emergency fallback")
        
        # Layer 3: Emergency fallback
        return _get_emergency_fallback_templates()

def reload_notebook_source_templates():
    """
    Force reload notebook source templates from database and update cache
    """
    try:
        # Lazy import to avoid database connection at startup
        from app.core.db import SessionLocal, Settings as SettingsModel
        
        db = SessionLocal()
        try:
            row = db.query(SettingsModel).filter(SettingsModel.category == 'notebook_source_templates').first()
            
            if row:
                settings = row.settings
                
                # Try to cache in Redis
                redis_client = _get_redis_client()
                if redis_client:
                    try:
                        from app.core.timeout_settings_cache import get_settings_cache_ttl
                        ttl = get_settings_cache_ttl()
                        redis_client.setex(NOTEBOOK_SOURCE_TEMPLATES_KEY, ttl, json.dumps(settings))
                    except Exception as e:
                        print(f"Failed to cache notebook source templates in Redis: {e}")
                
                return settings
            else:
                # No user-defined settings, use defaults
                return _get_emergency_fallback_templates()
        finally:
            db.close()
    except Exception as e:
        print(f"[NOTEBOOK_SOURCE_TEMPLATES] Failed to load notebook source templates from database: {e}")
        print(f"[NOTEBOOK_SOURCE_TEMPLATES] This is a critical failure - check database connectivity and settings table")
        raise RuntimeError(f"Database configuration retrieval failed: {e}")

def _validate_source_templates(templates):
    """
    Validate that templates contain required template keys
    """
    required_templates = [
        'base_source_intro',
        'mixed_sources_detail', 
        'memory_only_detail',
        'document_only_detail',
        'no_specific_detail',
        'synthesis_instruction',
        'memory_context_explanation',
        'comprehensive_answer_instruction'
    ]
    
    if not isinstance(templates, dict):
        return False
        
    for template_key in required_templates:
        if template_key not in templates:
            return False
        if not isinstance(templates[template_key], dict):
            return False
        if 'active' not in templates[template_key] or 'template' not in templates[template_key]:
            return False
            
    return True

def _get_emergency_fallback_templates():
    """
    Emergency fallback templates to prevent total failure
    """
    return {
        "base_source_intro": {
            "active": True,
            "template": "You have access to {total_sources} relevant information sources from this notebook",
            "description": "Base template for introducing available sources",
            "variables": ["total_sources"]
        },
        "mixed_sources_detail": {
            "active": True,
            "template": ", including {document_count} document{document_plural} and {memory_count} personal memor{memory_plural}. ",
            "description": "Template for when both documents and memories are present",
            "variables": ["document_count", "document_plural", "memory_count", "memory_plural"]
        },
        "memory_only_detail": {
            "active": True,
            "template": ", including {memory_count} personal memor{memory_plural}. ",
            "description": "Template for memory-only sources",
            "variables": ["memory_count", "memory_plural"]
        },
        "document_only_detail": {
            "active": True,
            "template": ", including {document_count} document{document_plural}. ",
            "description": "Template for document-only sources",
            "variables": ["document_count", "document_plural"]
        },
        "no_specific_detail": {
            "active": True,
            "template": ". ",
            "description": "Template fallback for general sources",
            "variables": []
        },
        "synthesis_instruction": {
            "active": True,
            "template": "When responding, synthesize information from ALL provided sources - both documents and memories contain valuable context. ",
            "description": "Instruction for synthesizing all source types",
            "variables": []
        },
        "memory_context_explanation": {
            "active": True,
            "template": "Memories typically contain personal experiences, recent developments, or contextual information that complements the formal document content. ",
            "description": "Explanation of what memories contain",
            "variables": []
        },
        "comprehensive_answer_instruction": {
            "active": True,
            "template": "Provide comprehensive answers that integrate insights from all available sources. If the context does not contain enough information, say so clearly.",
            "description": "Instruction for comprehensive responses",
            "variables": []
        }
    }

def apply_source_templates(total_sources, document_count, memory_count):
    """
    Apply source integration templates with dynamic values
    
    Args:
        total_sources (int): Total number of sources
        document_count (int): Number of document sources
        memory_count (int): Number of memory sources
        
    Returns:
        str: Formatted system prompt addition for source integration
    """
    templates = get_notebook_source_templates()
    
    # Build the source integration prompt
    prompt_parts = []
    
    # Base introduction with total sources
    base_template = templates.get('base_source_intro', {})
    if base_template.get('active', True):
        prompt_parts.append(base_template['template'].format(total_sources=total_sources))
    
    # Determine which detail template to use
    if document_count > 0 and memory_count > 0:
        # Both documents and memories
        detail_template = templates.get('mixed_sources_detail', {})
        if detail_template.get('active', True):
            prompt_parts.append(detail_template['template'].format(
                document_count=document_count,
                document_plural='s' if document_count != 1 else '',
                memory_count=memory_count,
                memory_plural='ies' if memory_count != 1 else 'y'
            ))
        
        # Add synthesis instruction for mixed sources
        synthesis_template = templates.get('synthesis_instruction', {})
        if synthesis_template.get('active', True):
            prompt_parts.append(synthesis_template['template'])
        
        # Add memory context explanation
        memory_explanation = templates.get('memory_context_explanation', {})
        if memory_explanation.get('active', True):
            prompt_parts.append(memory_explanation['template'])
            
    elif memory_count > 0:
        # Memory only
        detail_template = templates.get('memory_only_detail', {})
        if detail_template.get('active', True):
            prompt_parts.append(detail_template['template'].format(
                memory_count=memory_count,
                memory_plural='ies' if memory_count != 1 else 'y'
            ))
            
    elif document_count > 0:
        # Document only
        detail_template = templates.get('document_only_detail', {})
        if detail_template.get('active', True):
            prompt_parts.append(detail_template['template'].format(
                document_count=document_count,
                document_plural='s' if document_count != 1 else ''
            ))
    else:
        # No specific sources
        detail_template = templates.get('no_specific_detail', {})
        if detail_template.get('active', True):
            prompt_parts.append(detail_template['template'])
    
    # Add comprehensive answer instruction
    comprehensive_template = templates.get('comprehensive_answer_instruction', {})
    if comprehensive_template.get('active', True):
        prompt_parts.append(comprehensive_template['template'])
    
    return ''.join(prompt_parts)