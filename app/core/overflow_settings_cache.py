"""
Overflow Settings Cache
Manages cached overflow configuration from database for performance.
"""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from app.core.db import get_db_session, Settings
from app.schemas.overflow import OverflowConfig

logger = logging.getLogger(__name__)

# Global cache for overflow settings
_overflow_settings_cache: Optional[Dict[str, Any]] = None
_cache_timestamp: Optional[datetime] = None
_cache_ttl_minutes = 5  # Cache for 5 minutes

def get_overflow_settings() -> Dict[str, Any]:
    """
    Get overflow settings from cache or database.
    
    Returns:
        Dictionary of overflow configuration
    """
    global _overflow_settings_cache, _cache_timestamp
    
    # Check if cache is valid
    if (_overflow_settings_cache is not None and 
        _cache_timestamp is not None and
        datetime.utcnow() - _cache_timestamp < timedelta(minutes=_cache_ttl_minutes)):
        return _overflow_settings_cache
    
    # Load from database
    return reload_overflow_settings()

def reload_overflow_settings() -> Dict[str, Any]:
    """
    Force reload overflow settings from database and update cache.
    
    Returns:
        Dictionary of overflow configuration
    """
    global _overflow_settings_cache, _cache_timestamp
    
    try:
        with get_db_session() as db:
            # Query overflow settings
            settings_row = db.query(Settings).filter(Settings.category == "overflow").first()
            
            if settings_row and settings_row.settings:
                settings = settings_row.settings
                logger.info("Loaded overflow settings from database")
            else:
                # Create default settings if not exists
                default_config = OverflowConfig()
                settings = default_config.dict()
                
                # Save to database
                if not settings_row:
                    settings_row = Settings(
                        category="overflow",
                        settings=settings
                    )
                    db.add(settings_row)
                else:
                    settings_row.settings = settings
                
                db.commit()
                logger.info("Created default overflow settings in database")
            
            # Update cache
            _overflow_settings_cache = settings
            _cache_timestamp = datetime.utcnow()
            
            return settings
            
    except Exception as e:
        logger.error(f"Error loading overflow settings: {str(e)}")
        
        # Return default settings if database fails
        default_config = OverflowConfig()
        settings = default_config.dict()
        
        # Still update cache to avoid repeated DB failures
        _overflow_settings_cache = settings
        _cache_timestamp = datetime.utcnow()
        
        return settings

def update_overflow_settings(updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update overflow settings in database and refresh cache.
    
    Args:
        updates: Dictionary of settings to update
        
    Returns:
        Updated settings dictionary
    """
    global _overflow_settings_cache, _cache_timestamp
    
    try:
        with get_db_session() as db:
            # Get existing settings
            settings_row = db.query(Settings).filter(Settings.category == "overflow").first()
            
            if settings_row:
                # Merge updates with existing settings
                current_settings = settings_row.settings or {}
                updated_settings = {**current_settings, **updates}
                
                # Validate with schema
                config = OverflowConfig(**updated_settings)
                settings_row.settings = config.dict()
            else:
                # Create new settings with updates
                config = OverflowConfig(**updates)
                settings_row = Settings(
                    category="overflow",
                    settings=config.dict()
                )
                db.add(settings_row)
            
            db.commit()
            
            # Update cache
            _overflow_settings_cache = settings_row.settings
            _cache_timestamp = datetime.utcnow()
            
            logger.info("Updated overflow settings in database and cache")
            return _overflow_settings_cache
            
    except Exception as e:
        logger.error(f"Error updating overflow settings: {str(e)}")
        raise

def invalidate_overflow_cache():
    """
    Invalidate the overflow settings cache, forcing reload on next access.
    """
    global _overflow_settings_cache, _cache_timestamp
    
    _overflow_settings_cache = None
    _cache_timestamp = None
    logger.info("Invalidated overflow settings cache")

def get_overflow_config() -> OverflowConfig:
    """
    Get overflow configuration as a Pydantic model.
    
    Returns:
        OverflowConfig model instance
    """
    settings = get_overflow_settings()
    return OverflowConfig(**settings)