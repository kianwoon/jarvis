"""
Configuration for large generation detection and processing
"""

import os
from typing import Dict, List, Any
from dataclasses import dataclass, field

@dataclass
class LargeGenerationConfig:
    """Configuration for large generation detection and chunked processing"""
    
    # Detection thresholds
    strong_number_threshold: int = 30  # Numbers >= this always trigger chunking
    medium_number_threshold: int = 20  # Numbers >= this + keywords trigger chunking
    small_number_threshold: int = 20   # Numbers < this never trigger chunking
    
    # Scoring thresholds
    min_score_for_keywords: int = 3    # Minimum score to trigger on keywords alone
    min_score_for_medium_numbers: int = 2  # Minimum score for medium numbers
    
    # Estimation parameters
    score_multiplier: int = 15         # Items per score point when no number given
    default_comprehensive_items: int = 30  # Default for "comprehensive" requests
    min_estimated_items: int = 10      # Minimum estimated items
    
    # Classification threshold
    min_items_for_chunking: int = 20   # Minimum items to trigger chunked processing
    
    # Keywords that indicate large output
    large_output_indicators: List[str] = field(default_factory=lambda: [
        "generate", "create", "list", "write", "develop", "design", "build",
        "comprehensive", "detailed", "complete", "full", "extensive", "thorough",
        "step by step", "step-by-step", "all", "many", "multiple", "various",
        "questions", "examples", "ideas", "recommendations", "strategies", "options",
        "points", "items", "factors", "aspects", "benefits", "advantages", "features"
    ])
    
    # Patterns that suggest large output (regex patterns)
    large_patterns: List[str] = field(default_factory=lambda: [
        r'\b(\d+)\s+(questions|examples|items|points|ideas|strategies|options|factors|aspects|benefits|features)',
        r'(comprehensive|detailed|complete|full|extensive|thorough)\s+(list|guide|analysis|overview|breakdown)',
        r'(all|many|multiple|various)\s+(ways|methods|approaches|techniques|strategies|options)',
        r'generate.*\b(\d+)',
        r'create.*\b(\d+)',
        r'list.*\b(\d+)'
    ])
    
    # Keywords that strongly suggest comprehensive content
    comprehensive_keywords: List[str] = field(default_factory=lambda: [
        "comprehensive", "detailed", "all", "many"
    ])
    
    # Pattern score weight (how much to boost score for pattern matches)
    pattern_score_weight: int = 2
    
    # Confidence calculation parameters
    max_score_for_confidence: float = 5.0    # Score that gives 100% base confidence
    max_number_for_confidence: float = 100.0 # Number that gives 100% number confidence
    
    # Chunked processing parameters
    default_chunk_size: int = 15        # Default items per chunk
    max_target_count: int = 500         # Maximum items to generate
    estimated_seconds_per_chunk: int = 45  # For time estimation
    
    # Memory management
    redis_conversation_ttl: int = 3600  # 1 hour
    max_redis_messages: int = 50        # Messages to keep in Redis
    max_memory_messages: int = 20       # Messages to keep in memory
    conversation_history_display: int = 10  # Messages to show in history

def get_large_generation_config() -> LargeGenerationConfig:
    """Get large generation configuration with environment variable overrides"""
    
    config = LargeGenerationConfig()
    
    # Allow environment variable overrides
    env_overrides = {
        'LARGE_GEN_STRONG_THRESHOLD': 'strong_number_threshold',
        'LARGE_GEN_MEDIUM_THRESHOLD': 'medium_number_threshold', 
        'LARGE_GEN_SMALL_THRESHOLD': 'small_number_threshold',
        'LARGE_GEN_MIN_SCORE_KEYWORDS': 'min_score_for_keywords',
        'LARGE_GEN_MIN_SCORE_MEDIUM': 'min_score_for_medium_numbers',
        'LARGE_GEN_SCORE_MULTIPLIER': 'score_multiplier',
        'LARGE_GEN_DEFAULT_COMPREHENSIVE': 'default_comprehensive_items',
        'LARGE_GEN_MIN_ESTIMATED': 'min_estimated_items',
        'LARGE_GEN_MIN_CHUNKING': 'min_items_for_chunking',
        'LARGE_GEN_PATTERN_WEIGHT': 'pattern_score_weight',
        'LARGE_GEN_MAX_SCORE_CONF': 'max_score_for_confidence',
        'LARGE_GEN_MAX_NUMBER_CONF': 'max_number_for_confidence',
        'LARGE_GEN_DEFAULT_CHUNK_SIZE': 'default_chunk_size',
        'LARGE_GEN_MAX_TARGET': 'max_target_count',
        'LARGE_GEN_SECONDS_PER_CHUNK': 'estimated_seconds_per_chunk',
        'REDIS_CONVERSATION_TTL': 'redis_conversation_ttl',
        'REDIS_MAX_MESSAGES': 'max_redis_messages',
        'MEMORY_MAX_MESSAGES': 'max_memory_messages',
        'CONVERSATION_HISTORY_DISPLAY': 'conversation_history_display'
    }
    
    for env_var, attr_name in env_overrides.items():
        env_value = os.getenv(env_var)
        if env_value is not None:
            try:
                # Convert to appropriate type
                current_value = getattr(config, attr_name)
                if isinstance(current_value, int):
                    setattr(config, attr_name, int(env_value))
                elif isinstance(current_value, float):
                    setattr(config, attr_name, float(env_value))
                else:
                    setattr(config, attr_name, env_value)
            except (ValueError, TypeError) as e:
                print(f"[WARNING] Invalid value for {env_var}: {env_value}, using default")
    
    return config

# Global config instance
_config = None

def get_config():
    """Get the global configuration instance from database/cache"""
    from app.core.large_generation_settings_cache import get_large_generation_settings
    return get_large_generation_settings()

def reload_config():
    """Reload configuration from database and update cache"""
    from app.core.large_generation_settings_cache import reload_large_generation_settings
    return reload_large_generation_settings()