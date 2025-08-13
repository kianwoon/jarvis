"""
Utility functions for large generation configuration
"""

from typing import Any, Dict

class LargeGenerationConfigAccessor:
    """Helper class to access nested configuration values with dot notation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get nested configuration value using dot notation (e.g., 'detection_thresholds.strong_number_threshold')"""
        try:
            keys = path.split('.')
            value = self.config
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    # Detection thresholds
    @property
    def strong_number_threshold(self) -> int:
        return self.get('detection_thresholds.strong_number_threshold', 30)
    
    @property
    def medium_number_threshold(self) -> int:
        return self.get('detection_thresholds.medium_number_threshold', 20)
    
    @property
    def small_number_threshold(self) -> int:
        return self.get('detection_thresholds.small_number_threshold', 20)
    
    @property
    def min_items_for_chunking(self) -> int:
        return self.get('detection_thresholds.min_items_for_chunking', 20)
    
    # Scoring parameters
    @property
    def min_score_for_keywords(self) -> int:
        return self.get('scoring_parameters.min_score_for_keywords', 3)
    
    @property
    def min_score_for_medium_numbers(self) -> int:
        return self.get('scoring_parameters.min_score_for_medium_numbers', 2)
    
    @property
    def score_multiplier(self) -> int:
        return self.get('scoring_parameters.score_multiplier', 15)
    
    @property
    def default_comprehensive_items(self) -> int:
        return self.get('scoring_parameters.default_comprehensive_items', 30)
    
    @property
    def min_estimated_items(self) -> int:
        return self.get('scoring_parameters.min_estimated_items', 10)
    
    @property
    def pattern_score_weight(self) -> int:
        return self.get('scoring_parameters.pattern_score_weight', 2)
    
    # Confidence calculation
    @property
    def max_score_for_confidence(self) -> float:
        return self.get('confidence_calculation.max_score_for_confidence', 5.0)
    
    @property
    def max_number_for_confidence(self) -> float:
        return self.get('confidence_calculation.max_number_for_confidence', 100.0)
    
    # Processing parameters
    @property
    def default_chunk_size(self) -> int:
        return self.get('processing_parameters.default_chunk_size', 15)
    
    @property
    def max_target_count(self) -> int:
        return self.get('processing_parameters.max_target_count', 500)
    
    @property
    def estimated_seconds_per_chunk(self) -> int:
        return self.get('processing_parameters.estimated_seconds_per_chunk', 45)
    
    # Memory management
    @property
    def redis_conversation_ttl(self) -> int:
        """
        Get conversation TTL from centralized timeout settings.
        This ensures single source of truth for timeout configuration.
        """
        from app.core.timeout_settings_cache import get_timeout_value
        # Use centralized timeout settings instead of large_generation settings
        return get_timeout_value("session_cache", "conversation_cache_ttl", 86400)
    
    @property
    def max_redis_messages(self) -> int:
        return self.get('memory_management.max_redis_messages', 50)
    
    @property
    def max_memory_messages(self) -> int:
        return self.get('memory_management.max_memory_messages', 20)
    
    @property
    def conversation_history_display(self) -> int:
        return self.get('memory_management.conversation_history_display', 10)
    
    # Keywords and patterns
    @property
    def large_output_indicators(self) -> list:
        return self.get('keywords_and_patterns.large_output_indicators', [
            "generate", "create", "list", "write", "develop", "design", "build",
            "comprehensive", "detailed", "complete", "full", "extensive", "thorough",
            "step by step", "step-by-step", "all", "many", "multiple", "various",
            "questions", "examples", "ideas", "recommendations", "strategies", "options",
            "points", "items", "factors", "aspects", "benefits", "advantages", "features"
        ])
    
    @property
    def comprehensive_keywords(self) -> list:
        return self.get('keywords_and_patterns.comprehensive_keywords', [
            "comprehensive", "detailed", "all", "many"
        ])
    
    @property
    def large_patterns(self) -> list:
        return self.get('keywords_and_patterns.large_patterns', [
            r'\b(\d+)\s+(questions|examples|items|points|ideas|strategies|options|factors|aspects|benefits|features)',
            r'(comprehensive|detailed|complete|full|extensive|thorough)\s+(list|guide|analysis|overview|breakdown)',
            r'(all|many|multiple|various)\s+(ways|methods|approaches|techniques|strategies|options)',
            r'generate.*\b(\d+)',
            r'create.*\b(\d+)',
            r'list.*\b(\d+)'
        ])

def get_config_accessor():
    """Get a configuration accessor instance"""
    from app.core.large_generation_settings_cache import get_large_generation_settings
    config_dict = get_large_generation_settings()
    return LargeGenerationConfigAccessor(config_dict)