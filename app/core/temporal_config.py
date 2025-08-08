"""
Temporal Context Configuration

This module provides configuration utilities for the temporal context management system.
It demonstrates how to configure timezone, business hours, and temporal detection settings
without hardcoding values.

All configurations can be overridden via environment variables for deployment flexibility.
"""

import logging
import os
from typing import Dict, List, Any, Optional
from datetime import time

logger = logging.getLogger(__name__)

class TemporalConfig:
    """Configuration manager for temporal context settings"""
    
    @staticmethod
    def get_timezone_config() -> str:
        """
        Get timezone configuration with fallback chain
        
        Priority:
        1. SYSTEM_TIMEZONE environment variable
        2. Application settings
        3. Default: Asia/Singapore (as per requirements)
        """
        timezone = os.environ.get("SYSTEM_TIMEZONE")
        if timezone:
            logger.info(f"Using timezone from environment: {timezone}")
            return timezone
        
        # Could add more fallback sources here (database, config file, etc.)
        
        default_timezone = "Asia/Singapore"
        logger.info(f"Using default timezone: {default_timezone}")
        return default_timezone
    
    @staticmethod
    def get_business_hours_config() -> Dict[str, Any]:
        """
        Get business hours configuration from environment variables
        
        Environment variables:
        - BUSINESS_HOURS_START: Start hour (0-23), default: 9
        - BUSINESS_HOURS_END: End hour (0-23), default: 18
        - BUSINESS_WORK_DAYS: Comma-separated weekdays (1=Mon, 7=Sun), default: 1,2,3,4,5
        - BUSINESS_HOURS_TIMEZONE: Timezone for business hours, default: system timezone
        """
        try:
            config = {
                'start_hour': int(os.environ.get("BUSINESS_HOURS_START", "9")),
                'end_hour': int(os.environ.get("BUSINESS_HOURS_END", "18")),
                'work_days': [int(day) for day in os.environ.get("BUSINESS_WORK_DAYS", "1,2,3,4,5").split(',')],
                'timezone': os.environ.get("BUSINESS_HOURS_TIMEZONE") or TemporalConfig.get_timezone_config()
            }
            
            # Validate configuration
            if not (0 <= config['start_hour'] <= 23):
                raise ValueError(f"Invalid start_hour: {config['start_hour']}")
            if not (0 <= config['end_hour'] <= 23):
                raise ValueError(f"Invalid end_hour: {config['end_hour']}")
            if config['start_hour'] >= config['end_hour']:
                raise ValueError("start_hour must be less than end_hour")
            if not all(1 <= day <= 7 for day in config['work_days']):
                raise ValueError(f"Invalid work_days: {config['work_days']}")
            
            logger.info(f"Business hours configured: {config['start_hour']}:00-{config['end_hour']}:00, days: {config['work_days']}")
            return config
            
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid business hours configuration: {e}")
            # Return safe defaults
            return {
                'start_hour': 9,
                'end_hour': 18,
                'work_days': [1, 2, 3, 4, 5],
                'timezone': TemporalConfig.get_timezone_config()
            }
    
    @staticmethod
    def get_time_keywords_config() -> List[str]:
        """
        Get time-related keywords for query detection
        
        Environment variable:
        - TIME_KEYWORDS: Comma-separated list of keywords
        """
        keywords_env = os.environ.get("TIME_KEYWORDS")
        if keywords_env:
            keywords = [k.strip() for k in keywords_env.split(',') if k.strip()]
            logger.info(f"Using custom time keywords from environment: {len(keywords)} keywords")
            return keywords
        
        # Comprehensive default set
        default_keywords = [
            # Direct time queries
            'today', 'tomorrow', 'yesterday', 'now', 'current time',
            'what time', 'what date', 'time is', 'date is',
            
            # Scheduling and calendar
            'schedule', 'calendar', 'appointment', 'meeting', 'deadline',
            'due date', 'reminder', 'event',
            
            # Temporal references
            'recent', 'latest', 'current', 'when', 'time', 'date',
            'clock', 'hour', 'minute', 'second',
            
            # Time periods
            'morning', 'afternoon', 'evening', 'night',
            'business hours', 'working hours', 'office hours',
            
            # Relative time
            'this week', 'next week', 'last week',
            'this month', 'next month', 'last month',
            'this year', 'next year', 'last year',
            
            # Time-related questions
            'how long', 'duration', 'since when', 'until when',
            'how old', 'age', 'how much time'
        ]
        
        logger.info(f"Using default time keywords: {len(default_keywords)} keywords")
        return default_keywords
    
    @staticmethod
    def get_context_refresh_config() -> Dict[str, int]:
        """
        Get context refresh intervals in minutes
        
        Environment variables:
        - CONTEXT_REFRESH_CONVERSATION: Minutes for conversation context, default: 30
        - CONTEXT_REFRESH_SYSTEM_PROMPT: Minutes for system prompt context, default: 60
        - CONTEXT_REFRESH_TOOLS: Minutes for tool suggestions, default: 15
        """
        try:
            config = {
                'conversation_minutes': int(os.environ.get("CONTEXT_REFRESH_CONVERSATION", "30")),
                'system_prompt_minutes': int(os.environ.get("CONTEXT_REFRESH_SYSTEM_PROMPT", "60")),
                'tool_suggestions_minutes': int(os.environ.get("CONTEXT_REFRESH_TOOLS", "15"))
            }
            
            # Validate positive values
            for key, value in config.items():
                if value <= 0:
                    raise ValueError(f"Invalid {key}: {value} (must be positive)")
            
            logger.info(f"Context refresh intervals: {config}")
            return config
            
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid context refresh configuration: {e}")
            return {
                'conversation_minutes': 30,
                'system_prompt_minutes': 60,
                'tool_suggestions_minutes': 15
            }
    
    @staticmethod
    def get_temporal_detection_config() -> Dict[str, float]:
        """
        Get temporal detection confidence thresholds
        
        Environment variables:
        - TEMPORAL_DETECTION_THRESHOLD: Minimum confidence for temporal detection, default: 0.5
        - TEMPORAL_AUTO_SUGGEST_THRESHOLD: Minimum confidence for auto-suggesting tools, default: 0.6
        - TEMPORAL_CONTEXT_THRESHOLD: Minimum confidence for context decisions, default: 0.6
        """
        try:
            config = {
                'detection_threshold': float(os.environ.get("TEMPORAL_DETECTION_THRESHOLD", "0.5")),
                'auto_suggest_threshold': float(os.environ.get("TEMPORAL_AUTO_SUGGEST_THRESHOLD", "0.6")),
                'context_threshold': float(os.environ.get("TEMPORAL_CONTEXT_THRESHOLD", "0.6"))
            }
            
            # Validate threshold ranges (0.0 to 1.0)
            for key, value in config.items():
                if not (0.0 <= value <= 1.0):
                    raise ValueError(f"Invalid {key}: {value} (must be between 0.0 and 1.0)")
            
            logger.info(f"Temporal detection thresholds: {config}")
            return config
            
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid temporal detection configuration: {e}")
            return {
                'detection_threshold': 0.5,
                'auto_suggest_threshold': 0.6,
                'context_threshold': 0.6
            }
    
    @staticmethod
    def get_system_prompt_config() -> Dict[str, bool]:
        """
        Get system prompt enhancement settings
        
        Environment variables:
        - ENABLE_TEMPORAL_SYSTEM_PROMPT: Enable automatic time context in system prompts, default: true
        - ENABLE_BUSINESS_CONTEXT: Include business hours context, default: true
        - ENABLE_TEMPORAL_GUIDANCE: Include temporal reasoning guidelines, default: true
        """
        config = {
            'enable_temporal_system_prompt': os.environ.get("ENABLE_TEMPORAL_SYSTEM_PROMPT", "true").lower() == "true",
            'enable_business_context': os.environ.get("ENABLE_BUSINESS_CONTEXT", "true").lower() == "true",
            'enable_temporal_guidance': os.environ.get("ENABLE_TEMPORAL_GUIDANCE", "true").lower() == "true"
        }
        
        logger.info(f"System prompt configuration: {config}")
        return config
    
    @staticmethod
    def validate_configuration() -> Dict[str, Any]:
        """
        Validate the complete temporal configuration and return a summary
        
        Returns:
            Dictionary with validation results and configuration summary
        """
        validation_results = {
            'status': 'valid',
            'errors': [],
            'warnings': [],
            'configuration': {}
        }
        
        try:
            # Test timezone configuration
            timezone = TemporalConfig.get_timezone_config()
            try:
                from zoneinfo import ZoneInfo
                ZoneInfo(timezone)  # Validate timezone
                validation_results['configuration']['timezone'] = timezone
            except ImportError:
                try:
                    from backports.zoneinfo import ZoneInfo
                    ZoneInfo(timezone)
                    validation_results['configuration']['timezone'] = timezone
                except Exception as e:
                    validation_results['errors'].append(f"Invalid timezone '{timezone}': {e}")
            except Exception as e:
                validation_results['errors'].append(f"Invalid timezone '{timezone}': {e}")
            
            # Test business hours configuration
            business_hours = TemporalConfig.get_business_hours_config()
            validation_results['configuration']['business_hours'] = business_hours
            
            # Test other configurations
            validation_results['configuration']['time_keywords_count'] = len(TemporalConfig.get_time_keywords_config())
            validation_results['configuration']['context_refresh'] = TemporalConfig.get_context_refresh_config()
            validation_results['configuration']['detection_thresholds'] = TemporalConfig.get_temporal_detection_config()
            validation_results['configuration']['system_prompt'] = TemporalConfig.get_system_prompt_config()
            
            # Check for potential issues
            if business_hours['start_hour'] >= 12 and business_hours['end_hour'] <= 13:
                validation_results['warnings'].append("Business hours might not include lunch time")
            
            if len(business_hours['work_days']) < 5:
                validation_results['warnings'].append("Less than 5 work days configured")
            
            if validation_results['errors']:
                validation_results['status'] = 'invalid'
            elif validation_results['warnings']:
                validation_results['status'] = 'valid_with_warnings'
            
        except Exception as e:
            validation_results['status'] = 'error'
            validation_results['errors'].append(f"Configuration validation failed: {e}")
        
        logger.info(f"Temporal configuration validation: {validation_results['status']}")
        if validation_results['errors']:
            logger.error(f"Configuration errors: {validation_results['errors']}")
        if validation_results['warnings']:
            logger.warning(f"Configuration warnings: {validation_results['warnings']}")
        
        return validation_results

# Utility functions for easy access
def get_current_temporal_config() -> Dict[str, Any]:
    """Get the complete current temporal configuration"""
    return {
        'timezone': TemporalConfig.get_timezone_config(),
        'business_hours': TemporalConfig.get_business_hours_config(),
        'time_keywords': TemporalConfig.get_time_keywords_config(),
        'context_refresh': TemporalConfig.get_context_refresh_config(),
        'detection_thresholds': TemporalConfig.get_temporal_detection_config(),
        'system_prompt': TemporalConfig.get_system_prompt_config()
    }

def print_temporal_config_summary():
    """Print a human-readable summary of the temporal configuration"""
    config = get_current_temporal_config()
    validation = TemporalConfig.validate_configuration()
    
    print("=== Temporal Context Configuration Summary ===")
    print(f"Status: {validation['status'].upper()}")
    print(f"Timezone: {config['timezone']}")
    print(f"Business Hours: {config['business_hours']['start_hour']}:00 - {config['business_hours']['end_hour']}:00")
    print(f"Work Days: {config['business_hours']['work_days']} (1=Mon, 7=Sun)")
    print(f"Time Keywords: {len(config['time_keywords'])} keywords configured")
    print(f"Detection Thresholds: {config['detection_thresholds']}")
    print(f"System Prompt Enhancements: {config['system_prompt']}")
    
    if validation['errors']:
        print(f"\n❌ ERRORS: {len(validation['errors'])}")
        for error in validation['errors']:
            print(f"  - {error}")
    
    if validation['warnings']:
        print(f"\n⚠️  WARNINGS: {len(validation['warnings'])}")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    print("\n=== Environment Variables for Configuration ===")
    print("# Timezone")
    print("export SYSTEM_TIMEZONE='Asia/Singapore'")
    print("\n# Business Hours")
    print("export BUSINESS_HOURS_START=9")
    print("export BUSINESS_HOURS_END=18")
    print("export BUSINESS_WORK_DAYS='1,2,3,4,5'")
    print("\n# Detection Thresholds")
    print("export TEMPORAL_DETECTION_THRESHOLD=0.5")
    print("export TEMPORAL_AUTO_SUGGEST_THRESHOLD=0.6")
    print("\n# System Prompt Features")
    print("export ENABLE_TEMPORAL_SYSTEM_PROMPT=true")
    print("export ENABLE_BUSINESS_CONTEXT=true")
    print("\n==============================================")

if __name__ == "__main__":
    # When run directly, print configuration summary
    print_temporal_config_summary()