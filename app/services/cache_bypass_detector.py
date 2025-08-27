"""
Cache Bypass Detection Service

This service provides programmatic detection of user queries that require bypassing
cached context and forcing fresh retrieval from data sources.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class CacheBypassDetector:
    """
    Detects when user queries require bypassing cached context based on keyword patterns
    and semantic indicators that suggest need for fresh data retrieval.
    """
    
    def __init__(self):
        self.logger = logger
        
        # Core bypass patterns - combinations that strongly indicate need for fresh data
        self.bypass_patterns = [
            # "all" + "sources" + "again" patterns
            r'\ball\s+.*?sources\s+again\b',
            r'\bfrom\s+all\s+sources\s+again\b',
            r'\ball\s+.*?from\s+.*?sources\s+again\b',
            
            # "query/find/get/search" + "all" + "again" patterns  
            r'\b(?:query|find|get|search)\s+all\s+.*?again\b',
            r'\b(?:query|find|get|search)\s+.*?all\s+.*?again\b',
            
            # "all" + data scope + "again" patterns
            r'\ball\s+(?:projects|data|documents|files|information|content)\s+again\b',
            r'\ball\s+.*?(?:projects|data|documents|files|information|content)\s+.*?again\b',
            
            # "retrieve/pull/fetch" + "all" + "again" patterns
            r'\b(?:retrieve|pull|fetch)\s+all\s+.*?again\b',
            r'\b(?:retrieve|pull|fetch)\s+.*?all\s+.*?again\b',
            
            # Direct fresh data requests
            r'\bfresh\s+(?:data|search|query|retrieval)\b',
            r'\bnew\s+(?:search|query|retrieval)\b',
            r'\brefresh\s+(?:data|search|query|results)\b',
            
            # "start over" type patterns
            r'\bstart\s+over\b',
            r'\bfrom\s+scratch\b',
            r'\bignore\s+(?:previous|cached|stored)\b',
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.bypass_patterns]
        
        # Keywords that boost bypass likelihood when combined
        self.bypass_keywords = {
            'scope_indicators': ['all', 'everything', 'complete', 'entire', 'full'],
            'source_indicators': ['sources', 'databases', 'repositories', 'data'],
            'refresh_indicators': ['again', 'fresh', 'new', 'refresh', 'reload', 'restart'],
            'action_indicators': ['query', 'find', 'get', 'search', 'retrieve', 'pull', 'fetch']
        }
    
    def should_bypass_cache(self, message: str, conversation_id: str = None) -> Dict[str, any]:
        """
        Determine if cache should be bypassed based on message content.
        
        Args:
            message: User query message
            conversation_id: Optional conversation ID for logging
            
        Returns:
            Dict containing:
            - should_bypass: bool - Whether to bypass cache
            - confidence: float - Confidence score (0.0 to 1.0)  
            - reason: str - Human-readable reason for decision
            - matched_patterns: List[str] - Patterns that matched
        """
        if not message or not message.strip():
            return {
                'should_bypass': False,
                'confidence': 0.0,
                'reason': 'Empty message',
                'matched_patterns': []
            }
            
        message_clean = message.strip().lower()
        matched_patterns = []
        
        # Check direct pattern matches
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.search(message_clean):
                matched_patterns.append(self.bypass_patterns[i])
        
        # If we have direct pattern matches, high confidence bypass
        if matched_patterns:
            confidence = min(0.95, 0.7 + (len(matched_patterns) * 0.1))
            reason = f"Direct pattern match: {matched_patterns[0]}"
            
            self.logger.info(f"[CACHE_BYPASS] Detected bypass needed for conversation {conversation_id}: "
                           f"matched {len(matched_patterns)} patterns, confidence: {confidence:.2f}")
            
            return {
                'should_bypass': True,
                'confidence': confidence,
                'reason': reason,
                'matched_patterns': matched_patterns
            }
        
        # Check keyword combination scoring
        keyword_score = self._calculate_keyword_score(message_clean)
        
        # Bypass if keyword score is high enough
        if keyword_score >= 0.6:
            reason = f"High keyword combination score: {keyword_score:.2f}"
            
            self.logger.info(f"[CACHE_BYPASS] Detected bypass needed for conversation {conversation_id}: "
                           f"keyword score: {keyword_score:.2f}")
            
            return {
                'should_bypass': True,
                'confidence': keyword_score,
                'reason': reason,
                'matched_patterns': []
            }
        
        # No bypass needed
        self.logger.debug(f"[CACHE_BYPASS] No bypass needed for conversation {conversation_id}: "
                         f"keyword score: {keyword_score:.2f}, no direct patterns")
        
        return {
            'should_bypass': False,
            'confidence': 1.0 - keyword_score,
            'reason': f"Low bypass indicators (score: {keyword_score:.2f})",
            'matched_patterns': []
        }
    
    def _calculate_keyword_score(self, message: str) -> float:
        """Calculate bypass probability based on keyword combinations."""
        words = message.split()
        word_set = set(words)
        
        # Count keywords in each category
        category_scores = {}
        for category, keywords in self.bypass_keywords.items():
            matches = len([w for w in keywords if any(kw in word for word in word_set for kw in [w])])
            category_scores[category] = min(1.0, matches / len(keywords))
        
        # Special combination scoring
        has_scope = category_scores.get('scope_indicators', 0) > 0
        has_refresh = category_scores.get('refresh_indicators', 0) > 0
        has_action = category_scores.get('action_indicators', 0) > 0
        has_source = category_scores.get('source_indicators', 0) > 0
        
        # High score for scope + refresh + action combinations
        if has_scope and has_refresh and has_action:
            return 0.8
        
        # Medium-high score for scope + refresh
        if has_scope and has_refresh:
            return 0.7
        
        # Medium score for action + refresh + source
        if has_action and has_refresh and has_source:
            return 0.65
        
        # Calculate weighted average of category scores
        weights = {
            'scope_indicators': 0.3,
            'refresh_indicators': 0.4,  # Most important
            'action_indicators': 0.2,
            'source_indicators': 0.1
        }
        
        total_score = sum(score * weights.get(cat, 0.1) 
                         for cat, score in category_scores.items())
        
        return min(1.0, total_score)
    
    def analyze_bypass_patterns(self, messages: List[str]) -> Dict[str, any]:
        """
        Analyze multiple messages to identify bypass patterns for debugging.
        
        Args:
            messages: List of messages to analyze
            
        Returns:
            Analysis results with pattern statistics
        """
        results = []
        pattern_counts = {}
        
        for msg in messages:
            result = self.should_bypass_cache(msg)
            results.append({
                'message': msg[:100] + '...' if len(msg) > 100 else msg,
                'should_bypass': result['should_bypass'],
                'confidence': result['confidence'],
                'reason': result['reason']
            })
            
            # Count pattern matches
            for pattern in result['matched_patterns']:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        bypass_count = sum(1 for r in results if r['should_bypass'])
        
        return {
            'total_messages': len(messages),
            'bypass_recommended': bypass_count,
            'bypass_rate': bypass_count / len(messages) if messages else 0,
            'pattern_matches': pattern_counts,
            'results': results
        }

# Global instance
cache_bypass_detector = CacheBypassDetector()