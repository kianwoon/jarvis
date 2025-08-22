import json
import re
import logging
from typing import Dict, Any, List, Tuple, Optional
from app.core.config import get_settings

logger = logging.getLogger(__name__)

config = get_settings()
REDIS_HOST = config.REDIS_HOST
REDIS_PORT = config.REDIS_PORT
REDIS_PASSWORD = config.REDIS_PASSWORD
SYNTHESIS_PROMPTS_KEY = 'synthesis_prompts_cache'

# Don't create connection at import time
r = None

def _get_redis_client():
    """Get Redis client using the centralized redis_client"""
    global r
    if r is None:
        from app.core.redis_client import get_redis_client
        r = get_redis_client()
    return r

def get_synthesis_prompts():
    """Get synthesis prompt settings from cache or database"""
    # Try to get from Redis cache first
    redis_client = _get_redis_client()
    if redis_client:
        try:
            cached = redis_client.get(SYNTHESIS_PROMPTS_KEY)
            if cached:
                settings = json.loads(cached)
                logger.debug(f"Retrieved synthesis prompts from Redis cache with {len(settings)} categories")
                return settings
        except Exception as e:
            logger.warning(f"Redis error retrieving synthesis prompts: {e}, falling back to database")
    
    # If not cached or Redis failed, load from DB
    return reload_synthesis_prompts()

def set_synthesis_prompts(settings_dict):
    """Set synthesis prompt settings in cache and database"""
    if not settings_dict:
        logger.error("Cannot set empty synthesis prompt settings")
        return False
        
    try:
        # Store in database first
        from app.core.db import SessionLocal, Settings as SettingsModel
        
        db = SessionLocal()
        try:
            # Handle different synthesis categories
            categories = ['synthesis_prompts', 'formatting_templates', 'system_behaviors']
            
            for category in categories:
                if category in settings_dict:
                    # Update or create database entry for each category
                    row = db.query(SettingsModel).filter(SettingsModel.category == category).first()
                    if row:
                        row.settings = settings_dict[category]
                        logger.info(f"Updated existing {category} settings in database")
                    else:
                        new_settings = SettingsModel(category=category, settings=settings_dict[category])
                        db.add(new_settings)
                        logger.info(f"Created new {category} settings in database")
            
            db.commit()
            
            # Cache in Redis after successful database update
            redis_client = _get_redis_client()
            if redis_client:
                try:
                    from app.core.timeout_settings_cache import get_settings_cache_ttl
                    ttl = get_settings_cache_ttl()
                    redis_client.setex(SYNTHESIS_PROMPTS_KEY, ttl, json.dumps(settings_dict))
                    logger.debug(f"Cached synthesis prompts in Redis with TTL {ttl}s")
                except Exception as e:
                    logger.warning(f"Failed to cache synthesis prompts in Redis: {e}")
            
            return True
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Failed to set synthesis prompt settings: {e}")
        return False

def reload_synthesis_prompts():
    """Reload synthesis prompt settings from database and cache them"""
    try:
        # Lazy import to avoid database connection at startup
        from app.core.db import SessionLocal, Settings as SettingsModel
        
        db = SessionLocal()
        try:
            # Load all synthesis-related categories
            categories = ['synthesis_prompts', 'formatting_templates', 'system_behaviors']
            settings = {}
            
            for category in categories:
                row = db.query(SettingsModel).filter(SettingsModel.category == category).first()
                if row and row.settings:
                    settings[category] = row.settings
                    logger.debug(f"Loaded {category} from database")
                else:
                    # Load defaults if not found
                    settings[category] = get_default_synthesis_prompts().get(category, {})
                    logger.info(f"Using default {category} settings")
            
            logger.info(f"Loaded synthesis prompts from database with {len(settings)} categories")
            
            # Cache in Redis
            redis_client = _get_redis_client()
            if redis_client:
                try:
                    from app.core.timeout_settings_cache import get_settings_cache_ttl
                    ttl = get_settings_cache_ttl()
                    redis_client.setex(SYNTHESIS_PROMPTS_KEY, ttl, json.dumps(settings))
                    logger.debug(f"Cached synthesis prompts in Redis with TTL {ttl}s")
                except Exception as e:
                    logger.warning(f"Failed to cache synthesis prompts in Redis: {e}")
            
            return settings
                
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Failed to load synthesis prompts from database: {e}")
        raise

def clear_synthesis_prompts_cache():
    """Clear synthesis prompts from Redis cache to force reload from database"""
    redis_client = _get_redis_client()
    if redis_client:
        try:
            redis_client.delete(SYNTHESIS_PROMPTS_KEY)
            logger.info("Cleared synthesis prompts cache")
            return True
        except Exception as e:
            logger.warning(f"Failed to clear synthesis prompts cache: {e}")
            return False
    return False

def validate_synthesis_prompts(settings):
    """Validate synthesis prompt settings structure and required fields"""
    if not isinstance(settings, dict):
        return False, "Settings must be a dictionary"
    
    # Validate each category
    categories = {
        'synthesis_prompts': ['response_generation', 'context_integration', 'tool_synthesis'],
        'formatting_templates': ['markdown_format', 'code_blocks', 'lists_and_tables'],
        'system_behaviors': ['response_flags', 'error_handling', 'fallback_strategies']
    }
    
    for category, required_keys in categories.items():
        if category not in settings:
            return False, f"Missing required category: {category}"
        
        category_settings = settings[category]
        if not isinstance(category_settings, dict):
            return False, f"Category {category} must be a dictionary"
        
        # Check if required templates exist
        for key in required_keys:
            if key not in category_settings:
                return False, f"Missing required template: {category}.{key}"
            
            template = category_settings[key]
            if not isinstance(template, dict):
                return False, f"Template {category}.{key} must be a dictionary"
            
            # Validate template structure
            if 'content' not in template:
                return False, f"Template {category}.{key} missing content field"
            
            if 'variables' not in template:
                return False, f"Template {category}.{key} missing variables field"
            
            # Validate template content has valid variable syntax
            is_valid, error = validate_template_syntax(template['content'], template.get('variables', []))
            if not is_valid:
                return False, f"Template {category}.{key} syntax error: {error}"
    
    return True, "Settings validation passed"

def validate_template_syntax(content: str, variables: List[str]) -> Tuple[bool, str]:
    """Validate template syntax and variable usage"""
    if not isinstance(content, str):
        return False, "Template content must be a string"
    
    if not isinstance(variables, list):
        return False, "Template variables must be a list"
    
    # Find all variables in template using regex
    template_vars = re.findall(r'\{([^}]+)\}', content)
    
    # Check for undefined variables
    undefined_vars = set(template_vars) - set(variables)
    if undefined_vars:
        return False, f"Undefined variables in template: {', '.join(undefined_vars)}"
    
    # Check for unused variables (warning, not error)
    unused_vars = set(variables) - set(template_vars)
    if unused_vars:
        logger.warning(f"Unused variables in template: {', '.join(unused_vars)}")
    
    # Validate variable naming
    invalid_vars = [var for var in template_vars if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', var)]
    if invalid_vars:
        return False, f"Invalid variable names: {', '.join(invalid_vars)}"
    
    return True, "Template syntax is valid"

def render_template(template_content: str, variables: Dict[str, Any]) -> str:
    """Render template with variable substitution"""
    try:
        rendered = template_content
        for var_name, var_value in variables.items():
            placeholder = f"{{{var_name}}}"
            rendered = rendered.replace(placeholder, str(var_value))
        return rendered
    except Exception as e:
        logger.error(f"Error rendering template: {e}")
        return template_content

def get_template_variables(template_content: str) -> List[str]:
    """Extract variable names from template content"""
    return re.findall(r'\{([^}]+)\}', template_content)

def get_default_synthesis_prompts():
    """Get default synthesis prompt templates"""
    return {
        "synthesis_prompts": {
            "response_generation": {
                "content": "Based on the {tool_context} and user question '{question}', generate a comprehensive response that addresses {key_points}. Ensure the response is {response_style} and includes relevant details.",
                "description": "Core template for generating responses from tool outputs",
                "variables": ["tool_context", "question", "key_points", "response_style"],
                "active": True,
                "version": "1.0",
                "metadata": {
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                    "author": "system"
                }
            },
            "context_integration": {
                "content": "Integrate the following context: {context} with the current conversation. The user's enhanced question is: {enhanced_question}. Provide a response that connects the context to their specific needs.",
                "description": "Template for integrating additional context into responses",
                "variables": ["context", "enhanced_question"],
                "active": True,
                "version": "1.0",
                "metadata": {
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                    "author": "system"
                }
            },
            "tool_synthesis": {
                "content": "You need to provide a final answer based on the tool results below.\n\n{tool_context}\n\nBased on these search results, provide a comprehensive answer to the user's question: \"{question}\".\n\nCRITICAL: Preserve ALL specific details from the search results including:\n- Exact version numbers (e.g., \"4.1\" in \"Claude Opus 4.1\")\n- Precise model names and technical specifications\n- Specific dates, timestamps, and timeline information\n- Accurate numerical data, measurements, and statistics\n- Proper names of people, companies, products, and technologies\n- Direct quotes and exact terminology used in sources\n\nFormat the information clearly and include the most relevant findings while maintaining complete accuracy of all specific details.",
                "description": "Template for synthesizing tool results into final answers",
                "variables": ["tool_context", "question"],
                "active": True,
                "version": "1.0",
                "metadata": {
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                    "author": "system"
                }
            },
            "enhanced_tool_synthesis": {
                "content": "{response_text}\n\n{tool_context}\n\nPlease provide a complete answer using the tool results above.\n\nCRITICAL: Preserve ALL specific details from the tool results including:\n- Exact version numbers, model names, and technical specifications\n- Specific dates, timestamps, and timeline information\n- Accurate numerical data, measurements, and statistics\n- Proper names of people, companies, products, and technologies\n- Direct quotes and exact terminology used in sources\n- Precise technical details without generalization or simplification\n\nMaintain complete accuracy of all specific information while formatting clearly.",
                "description": "Template for enhanced synthesis combining existing response with tool results",
                "variables": ["response_text", "tool_context"],
                "active": True,
                "version": "1.0",
                "metadata": {
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                    "author": "system"
                }
            },
            "knowledge_base_synthesis": {
                "content": "📚 CURRENT FACTUAL DATA (Knowledge Base - PRIMARY SOURCE):\n{rag_context}\n\nSYNTHESIS INSTRUCTIONS: Use the above information as your primary source. Answer based on this factual data.\n\nCRITICAL: Preserve ALL specific details from the knowledge base including:\n- Exact version numbers, model names, and technical specifications\n- Specific dates, timestamps, and timeline information\n- Accurate numerical data, measurements, and statistics\n- Proper names of people, companies, products, and technologies\n- Direct quotes and exact terminology used in sources\n- Precise technical details without generalization or simplification\n\nMaintain complete accuracy of all specific information.",
                "description": "Template for synthesizing knowledge base content with preservation instructions",
                "variables": ["rag_context"],
                "active": True,
                "version": "1.0",
                "metadata": {
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                    "author": "system"
                }
            }
        },
        "formatting_templates": {
            "markdown_format": {
                "content": "Format the following content as markdown:\n\n{content}\n\nEnsure proper heading structure, bullet points, and code formatting where appropriate. Use {emphasis_style} for important points.",
                "description": "Template for markdown formatting",
                "variables": ["content", "emphasis_style"],
                "active": True,
                "version": "1.0",
                "metadata": {
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                    "author": "system"
                }
            },
            "code_blocks": {
                "content": "Present the code in the following format:\n\n```{language}\n{code}\n```\n\nInclude {additional_info} and provide {explanation_level} explanation.",
                "description": "Template for code block formatting",
                "variables": ["language", "code", "additional_info", "explanation_level"],
                "active": True,
                "version": "1.0",
                "metadata": {
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                    "author": "system"
                }
            },
            "lists_and_tables": {
                "content": "Organize the information as follows:\n\n{list_format}\n\n{data}\n\nUse {table_style} for tabular data and {list_style} for list items.",
                "description": "Template for lists and tables",
                "variables": ["list_format", "data", "table_style", "list_style"],
                "active": True,
                "version": "1.0",
                "metadata": {
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                    "author": "system"
                }
            }
        },
        "system_behaviors": {
            "response_flags": {
                "content": "Response configuration: verbose={verbose}, include_examples={include_examples}, technical_level={technical_level}, follow_up_questions={follow_up_questions}",
                "description": "System flags for response behavior",
                "variables": ["verbose", "include_examples", "technical_level", "follow_up_questions"],
                "active": True,
                "version": "1.0",
                "metadata": {
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                    "author": "system"
                }
            },
            "error_handling": {
                "content": "Error encountered: {error_type}. Context: {error_context}. Fallback action: {fallback_action}. User message: {user_message}",
                "description": "Template for error handling responses",
                "variables": ["error_type", "error_context", "fallback_action", "user_message"],
                "active": True,
                "version": "1.0",
                "metadata": {
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                    "author": "system"
                }
            },
            "fallback_strategies": {
                "content": "Primary approach failed: {primary_failure}. Using fallback strategy: {fallback_strategy}. Alternative approach: {alternative_approach}. Success probability: {success_probability}",
                "description": "Template for fallback strategy communication",
                "variables": ["primary_failure", "fallback_strategy", "alternative_approach", "success_probability"],
                "active": True,
                "version": "1.0",
                "metadata": {
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                    "author": "system"
                }
            }
        }
    }

def get_synthesis_setting(category: str, template_name: str, field: str = None, default=None):
    """Get a specific synthesis setting value"""
    try:
        settings = get_synthesis_prompts()
        template = settings.get(category, {}).get(template_name, {})
        
        if field:
            return template.get(field, default)
        else:
            return template if template else default
            
    except Exception as e:
        logger.error(f"Failed to get synthesis setting {category}.{template_name}.{field}: {e}")
        if default is not None:
            return default
        raise

def update_synthesis_setting(category: str, template_name: str, field: str, value: Any):
    """Update a specific synthesis setting value"""
    try:
        settings = get_synthesis_prompts()
        
        if category not in settings:
            settings[category] = {}
        if template_name not in settings[category]:
            settings[category][template_name] = {}
            
        settings[category][template_name][field] = value
        
        # Update metadata
        if 'metadata' not in settings[category][template_name]:
            settings[category][template_name]['metadata'] = {}
        
        import datetime
        settings[category][template_name]['metadata']['updated_at'] = datetime.datetime.utcnow().isoformat() + 'Z'
        
        # Validate if we're updating content or variables
        if field in ['content', 'variables']:
            template = settings[category][template_name]
            is_valid, error = validate_template_syntax(
                template.get('content', ''), 
                template.get('variables', [])
            )
            if not is_valid:
                logger.error(f"Invalid template after update: {error}")
                return False
        
        return set_synthesis_prompts(settings)
        
    except Exception as e:
        logger.error(f"Failed to update synthesis setting {category}.{template_name}.{field}: {e}")
        return False

def create_template(category: str, template_name: str, template_data: Dict[str, Any]):
    """Create a new template"""
    try:
        settings = get_synthesis_prompts()
        
        if category not in settings:
            settings[category] = {}
            
        if template_name in settings[category]:
            return False, f"Template {template_name} already exists in category {category}"
        
        # Validate template data
        required_fields = ['content', 'variables', 'description']
        for field in required_fields:
            if field not in template_data:
                return False, f"Missing required field: {field}"
        
        # Validate template syntax
        is_valid, error = validate_template_syntax(
            template_data['content'], 
            template_data['variables']
        )
        if not is_valid:
            return False, f"Template syntax error: {error}"
        
        # Add metadata
        import datetime
        template_data['metadata'] = {
            'created_at': datetime.datetime.utcnow().isoformat() + 'Z',
            'updated_at': datetime.datetime.utcnow().isoformat() + 'Z',
            'author': 'user',
            'version': '1.0'
        }
        template_data['active'] = template_data.get('active', True)
        
        settings[category][template_name] = template_data
        
        success = set_synthesis_prompts(settings)
        return success, "Template created successfully" if success else "Failed to save template"
        
    except Exception as e:
        logger.error(f"Failed to create template {category}.{template_name}: {e}")
        return False, str(e)

def delete_template(category: str, template_name: str):
    """Delete a template"""
    try:
        settings = get_synthesis_prompts()
        
        if category not in settings or template_name not in settings[category]:
            return False, f"Template {template_name} not found in category {category}"
        
        del settings[category][template_name]
        
        success = set_synthesis_prompts(settings)
        return success, "Template deleted successfully" if success else "Failed to delete template"
        
    except Exception as e:
        logger.error(f"Failed to delete template {category}.{template_name}: {e}")
        return False, str(e)

def reset_to_defaults(category: str = None):
    """Reset templates to default values"""
    try:
        defaults = get_default_synthesis_prompts()
        
        if category:
            # Reset specific category
            if category not in defaults:
                return False, f"Unknown category: {category}"
            
            settings = get_synthesis_prompts()
            settings[category] = defaults[category]
            success = set_synthesis_prompts(settings)
            return success, f"Category {category} reset to defaults" if success else "Failed to reset category"
        else:
            # Reset all categories
            success = set_synthesis_prompts(defaults)
            return success, "All templates reset to defaults" if success else "Failed to reset templates"
            
    except Exception as e:
        logger.error(f"Failed to reset templates: {e}")
        return False, str(e)

def get_template_preview(category: str, template_name: str, sample_variables: Dict[str, Any] = None):
    """Get a preview of rendered template with sample data"""
    try:
        template = get_synthesis_setting(category, template_name)
        if not template:
            return None, f"Template {template_name} not found in category {category}"
        
        content = template.get('content', '')
        variables = template.get('variables', [])
        
        # Use provided sample variables or generate defaults
        if sample_variables is None:
            sample_variables = {}
            for var in variables:
                sample_variables[var] = f"[{var}_sample]"
        
        rendered = render_template(content, sample_variables)
        
        return {
            'original_content': content,
            'rendered_content': rendered,
            'variables_used': variables,
            'sample_variables': sample_variables,
            'template_metadata': template.get('metadata', {})
        }, None
        
    except Exception as e:
        logger.error(f"Failed to preview template {category}.{template_name}: {e}")
        return None, str(e)

def get_formatted_synthesis_template(category: str, template_name: str, variables: Dict[str, str] = None) -> str:
    """Get a synthesis template with variables substituted
    
    Args:
        category: Template category (synthesis_prompts, formatting_templates, system_behaviors)
        template_name: Name of the template
        variables: Dictionary of variables to substitute
        
    Returns:
        Formatted template string with variables substituted
    """
    try:
        # Get the template
        template = get_synthesis_setting(category, template_name)
        if not template:
            logger.warning(f"Template {category}.{template_name} not found, using fallback")
            # Return a basic fallback template
            if category == "synthesis_prompts" and template_name == "tool_synthesis":
                return """You need to provide a final answer based on the tool results below.

{tool_context}

Based on these search results, provide a comprehensive answer to the user's question: "{question}".

CRITICAL: Preserve ALL specific details from the search results including:
- Exact version numbers (e.g., "4.1" in "Claude Opus 4.1")
- Precise model names and technical specifications
- Specific dates, timestamps, and timeline information
- Accurate numerical data, measurements, and statistics
- Proper names of people, companies, products, and technologies
- Direct quotes and exact terminology used in sources

Format the information clearly and include the most relevant findings while maintaining complete accuracy of all specific details."""
            else:
                return "Template not found: {category}.{template_name}"
        
        # Get template content
        content = template.get('content', '')
        if not content:
            logger.warning(f"Template {category}.{template_name} has no content")
            return f"Template {category}.{template_name} has no content"
        
        # Substitute variables if provided
        if variables:
            formatted_content = content
            for var_name, var_value in variables.items():
                placeholder = "{" + var_name + "}"
                formatted_content = formatted_content.replace(placeholder, str(var_value))
            return formatted_content
        else:
            return content
            
    except Exception as e:
        logger.error(f"Failed to get formatted template {category}.{template_name}: {e}")
        # Return emergency fallback
        return f"Error loading template {category}.{template_name}: {str(e)}"