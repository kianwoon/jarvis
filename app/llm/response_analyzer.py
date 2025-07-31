"""
Dynamic model response analyzer to detect thinking vs non-thinking behavior.
This module analyzes actual model responses to determine behavior patterns,
making the system model-agnostic and future-proof.
"""

import re
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

@dataclass
class ModelBehaviorProfile:
    """Profile of detected model behavior"""
    model_name: str
    is_thinking_model: bool
    confidence: float
    sample_response: str
    detected_at: datetime
    response_count: int

class ResponseAnalyzer:
    """Analyzes model responses to detect thinking vs non-thinking behavior"""
    
    def __init__(self):
        self._behavior_cache: Dict[str, ModelBehaviorProfile] = {}
        self._cache_ttl = timedelta(hours=1)  # Cache behavior for 1 hour
    
    def detect_thinking_behavior(self, response_text: str, model_name: str) -> Tuple[bool, float]:
        """
        Detect if a model response indicates thinking behavior.
        
        Args:
            response_text: The actual response from the model
            model_name: Name of the model for caching
            
        Returns:
            Tuple of (is_thinking_model, confidence_score)
        """
        # Check cache first
        cached_profile = self._get_cached_behavior(model_name)
        if cached_profile and cached_profile.confidence > 0.8:
            return cached_profile.is_thinking_model, cached_profile.confidence
        
        # Analyze response for thinking patterns
        thinking_indicators = self._analyze_thinking_patterns(response_text)
        is_thinking = thinking_indicators['has_thinking_tags']
        confidence = thinking_indicators['confidence']
        
        # Update cache
        self._update_behavior_cache(model_name, is_thinking, confidence, response_text)
        
        return is_thinking, confidence
    
    def _analyze_thinking_patterns(self, response_text: str) -> Dict:
        """Analyze response text for thinking behavior indicators"""
        
        indicators = {
            'has_thinking_tags': False,
            'confidence': 0.0,
            'patterns_found': []
        }
        
        if not response_text:
            return indicators
        
        # Primary indicator: <think> tags (must be very early in response)
        think_pattern = r'<think>'
        first_200_chars = response_text[:200]  # Only check beginning of response
        if re.search(think_pattern, first_200_chars, re.IGNORECASE):
            indicators['has_thinking_tags'] = True
            indicators['confidence'] = 0.95
            indicators['patterns_found'].append('explicit_think_tags')
            return indicators
        
        # Secondary indicators for thinking behavior
        thinking_patterns = [
            (r'Let me think', 0.7, 'explicit_thinking_phrase'),
            (r'I need to consider', 0.6, 'consideration_phrase'),
            (r'Let me analyze', 0.6, 'analysis_phrase'),
            (r'First, I should', 0.5, 'step_by_step_thinking'),
            (r'My reasoning is', 0.5, 'reasoning_explanation'),
        ]
        
        max_confidence = 0.0
        found_patterns = []
        
        for pattern, conf, name in thinking_patterns:
            if re.search(pattern, response_text, re.IGNORECASE):
                max_confidence = max(max_confidence, conf)
                found_patterns.append(name)
        
        # Non-thinking indicators (reduce confidence)
        non_thinking_patterns = [
            (r'^[A-Z][a-z]', 0.2, 'direct_answer_start'),  # Starts directly with answer
            (r'^\w+\s+is\s+', 0.3, 'definition_start'),     # "Something is..."
        ]
        
        for pattern, reduction, name in non_thinking_patterns:
            if re.search(pattern, response_text.strip()[:50], re.IGNORECASE):
                max_confidence = max(0.0, max_confidence - reduction)
                found_patterns.append(f'non_thinking_{name}')
        
        # If no patterns found, it's likely a non-thinking model
        if max_confidence == 0.0:
            indicators['confidence'] = 0.0
            indicators['has_thinking_tags'] = False
        else:
            indicators['confidence'] = max_confidence
            indicators['has_thinking_tags'] = max_confidence > 0.6
        
        indicators['patterns_found'] = found_patterns
        
        return indicators
    
    def _get_cached_behavior(self, model_name: str) -> Optional[ModelBehaviorProfile]:
        """Get cached behavior profile if still valid"""
        if model_name not in self._behavior_cache:
            return None
        
        profile = self._behavior_cache[model_name]
        if datetime.now() - profile.detected_at > self._cache_ttl:
            # Cache expired
            del self._behavior_cache[model_name]
            return None
        
        return profile
    
    def _update_behavior_cache(self, model_name: str, is_thinking: bool, confidence: float, sample_response: str):
        """Update behavior cache with new detection"""
        
        existing = self._behavior_cache.get(model_name)
        if existing:
            # Update existing profile with weighted average
            response_count = existing.response_count + 1
            weighted_confidence = (existing.confidence * existing.response_count + confidence) / response_count
            
            self._behavior_cache[model_name] = ModelBehaviorProfile(
                model_name=model_name,
                is_thinking_model=weighted_confidence > 0.6,
                confidence=weighted_confidence,
                sample_response=sample_response[:200],  # Keep sample short
                detected_at=datetime.now(),
                response_count=response_count
            )
        else:
            # Create new profile
            self._behavior_cache[model_name] = ModelBehaviorProfile(
                model_name=model_name,
                is_thinking_model=is_thinking,
                confidence=confidence,
                sample_response=sample_response[:200],
                detected_at=datetime.now(),
                response_count=1
            )
    
    def get_model_profile(self, model_name: str) -> Optional[ModelBehaviorProfile]:
        """Get the complete behavior profile for a model"""
        return self._get_cached_behavior(model_name)
    
    def clear_cache(self, model_name: Optional[str] = None):
        """Clear behavior cache for specific model or all models"""
        if model_name:
            self._behavior_cache.pop(model_name, None)
        else:
            self._behavior_cache.clear()
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about cached model behaviors"""
        stats = {
            'total_models': len(self._behavior_cache),
            'thinking_models': 0,
            'non_thinking_models': 0,
            'models': {}
        }
        
        for model_name, profile in self._behavior_cache.items():
            if profile.is_thinking_model:
                stats['thinking_models'] += 1
            else:
                stats['non_thinking_models'] += 1
            
            stats['models'][model_name] = {
                'is_thinking': profile.is_thinking_model,
                'confidence': profile.confidence,
                'response_count': profile.response_count,
                'detected_at': profile.detected_at.isoformat()
            }
        
        return stats

# Global instance for easy access
response_analyzer = ResponseAnalyzer()

def detect_model_thinking_behavior(response_text: str, model_name: str) -> Tuple[bool, float]:
    """
    Convenience function to detect model thinking behavior.
    
    Args:
        response_text: The actual response from the model
        model_name: Name of the model for caching
        
    Returns:
        Tuple of (is_thinking_model, confidence_score)
    """
    return response_analyzer.detect_thinking_behavior(response_text, model_name)

def get_model_behavior_profile(model_name: str) -> Optional[ModelBehaviorProfile]:
    """Get cached behavior profile for a model"""
    return response_analyzer.get_model_profile(model_name)

def clear_behavior_cache(model_name: Optional[str] = None):
    """Clear behavior cache"""
    response_analyzer.clear_cache(model_name)