"""
Conflict Prevention Engine
Pre-emptive conflict detection and prevention for conversation cache
"""

import json
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import redis.asyncio as redis
from app.core.redis_client import get_async_redis_client
import logging
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)

class ConflictPreventionEngine:
    """
    Advanced conflict prevention system that checks for conflicts before adding messages to cache.
    Features:
    - Pre-emptive conflict checking
    - Pattern-based prevention
    - Intelligent TTL adjustment
    - Historical pattern learning
    """
    
    def __init__(self):
        self.redis_client = None
        self.base_ttl = 86400  # 24 hours default
        
        # Conflict pattern definitions with volatility scores
        self.conflict_patterns = {
            'existence': {
                'patterns': [
                    {
                        'positive': [r'\b(exists?|available|released?|launched?|announced?|is out)\b'],
                        'negative': [r'\b(does not exist|doesn\'t exist|not exist|no.*available|persistent myth)\b']
                    }
                ],
                'volatility': 0.8,  # High volatility - new releases happen frequently
                'ttl_multiplier': 0.3  # Reduce TTL by 70% for existence claims
            },
            'version': {
                'patterns': [
                    {
                        'current': [r'\b(latest|current|newest).*version.*(\d+(?:\.\d+)*)\b'],
                        'outdated': [r'\b(old|previous|earlier).*version.*(\d+(?:\.\d+)*)\b']
                    }
                ],
                'volatility': 0.9,  # Very high volatility - versions change often
                'ttl_multiplier': 0.2  # Reduce TTL by 80% for version info
            },
            'temporal': {
                'patterns': [
                    {
                        'recent': [r'\b(today|yesterday|this week|recently|just)\b'],
                        'old': [r'\b(last year|months ago|previously|historically)\b']
                    }
                ],
                'volatility': 1.0,  # Maximum volatility - time-sensitive
                'ttl_multiplier': 0.1  # Reduce TTL by 90% for temporal claims
            },
            'statistics': {
                'patterns': [
                    {
                        'numeric': [r'\b(\d+(?:\.\d+)?)\s*(?:%|percent|million|billion|users|downloads)\b']
                    }
                ],
                'volatility': 0.6,  # Moderate volatility - stats change periodically
                'ttl_multiplier': 0.5  # Reduce TTL by 50% for statistics
            },
            'availability': {
                'patterns': [
                    {
                        'available': [r'\b(open.?source|freely available|public|free to use)\b'],
                        'restricted': [r'\b(proprietary|closed.?source|private|paid|licensed)\b']
                    }
                ],
                'volatility': 0.7,  # Moderate-high volatility
                'ttl_multiplier': 0.4  # Reduce TTL by 60%
            }
        }
        
        # Historical conflict tracking
        self.conflict_history_key = "conflict_prevention:history"
        self.pattern_frequency_key = "conflict_prevention:patterns"
        
    async def _get_redis(self) -> redis.Redis:
        """Get Redis client"""
        if not self.redis_client:
            self.redis_client = await get_async_redis_client()
        return self.redis_client
    
    async def check_for_conflicts(
        self, 
        new_content: str, 
        conversation_id: str,
        role: str = "assistant",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Check if new content would conflict with existing conversation history
        
        Returns:
            Dict with conflict analysis and recommendations
        """
        try:
            redis_client = await self._get_redis()
            
            # Get existing conversation history
            key = f"conversation:{conversation_id}:messages"
            existing_messages = await redis_client.lrange(key, -20, -1)  # Check last 20 messages
            
            if not existing_messages:
                return {
                    'has_conflicts': False,
                    'conflicts': [],
                    'recommended_ttl': self.base_ttl,
                    'volatility_score': 0.0,
                    'should_add': True
                }
            
            # Analyze new content for conflict patterns
            new_content_lower = new_content.lower()
            detected_conflicts = []
            max_volatility = 0.0
            
            # Check each existing message for conflicts
            for msg_bytes in existing_messages:
                try:
                    existing_msg = json.loads(msg_bytes)
                    existing_content = existing_msg.get('content', '').lower()
                    existing_role = existing_msg.get('role', '')
                    
                    # Skip checking against user messages if we're adding an assistant message
                    if role == "assistant" and existing_role == "user":
                        continue
                    
                    # Check for conflicts using pattern matching
                    for conflict_type, config in self.conflict_patterns.items():
                        conflict = self._detect_pattern_conflict(
                            new_content_lower, 
                            existing_content, 
                            config['patterns']
                        )
                        
                        if conflict:
                            detected_conflicts.append({
                                'type': conflict_type,
                                'severity': self._calculate_severity(conflict_type, existing_msg),
                                'existing_message': existing_content[:200],
                                'timestamp': existing_msg.get('timestamp'),
                                'pattern_matched': conflict
                            })
                            max_volatility = max(max_volatility, config['volatility'])
                            
                except Exception as e:
                    logger.warning(f"[PREVENTION] Failed to parse message during conflict check: {e}")
                    continue
            
            # Record conflict patterns for learning
            if detected_conflicts:
                await self._record_conflict_patterns(conversation_id, detected_conflicts)
            
            # Calculate recommended TTL based on volatility
            recommended_ttl = self._calculate_dynamic_ttl(max_volatility, detected_conflicts)
            
            # Predict future conflicts based on historical patterns
            prediction = await self._predict_future_conflicts(new_content_lower, conversation_id)
            
            # Determine if message should be added
            should_add = self._should_add_message(detected_conflicts, prediction)
            
            return {
                'has_conflicts': len(detected_conflicts) > 0,
                'conflicts': detected_conflicts,
                'recommended_ttl': recommended_ttl,
                'volatility_score': max_volatility,
                'should_add': should_add,
                'prediction': prediction,
                'resolution_strategy': self._get_resolution_strategy(detected_conflicts)
            }
            
        except Exception as e:
            logger.error(f"[PREVENTION] Error checking for conflicts: {e}")
            return {
                'has_conflicts': False,
                'conflicts': [],
                'recommended_ttl': self.base_ttl,
                'volatility_score': 0.0,
                'should_add': True
            }
    
    def _detect_pattern_conflict(
        self, 
        new_content: str, 
        existing_content: str, 
        patterns: List[Dict[str, List[str]]]
    ) -> Optional[str]:
        """Detect if content patterns conflict"""
        for pattern_set in patterns:
            # Handle different pattern types
            if 'positive' in pattern_set and 'negative' in pattern_set:
                # Check for existence conflicts
                new_positive = any(re.search(p, new_content) for p in pattern_set['positive'])
                new_negative = any(re.search(p, new_content) for p in pattern_set['negative'])
                existing_positive = any(re.search(p, existing_content) for p in pattern_set['positive'])
                existing_negative = any(re.search(p, existing_content) for p in pattern_set['negative'])
                
                if (new_positive and existing_negative) or (new_negative and existing_positive):
                    return 'existence_conflict'
                    
            elif 'current' in pattern_set and 'outdated' in pattern_set:
                # Check for version conflicts
                new_current = any(re.search(p, new_content) for p in pattern_set['current'])
                existing_current = any(re.search(p, existing_content) for p in pattern_set['current'])
                
                if new_current and existing_current:
                    # Extract version numbers and compare
                    new_version = self._extract_version(new_content)
                    existing_version = self._extract_version(existing_content)
                    if new_version and existing_version and new_version != existing_version:
                        return 'version_conflict'
                        
            elif 'available' in pattern_set and 'restricted' in pattern_set:
                # Check for availability conflicts
                new_available = any(re.search(p, new_content) for p in pattern_set['available'])
                new_restricted = any(re.search(p, new_content) for p in pattern_set['restricted'])
                existing_available = any(re.search(p, existing_content) for p in pattern_set['available'])
                existing_restricted = any(re.search(p, existing_content) for p in pattern_set['restricted'])
                
                if (new_available and existing_restricted) or (new_restricted and existing_available):
                    return 'availability_conflict'
                    
            elif 'numeric' in pattern_set:
                # Check for statistical conflicts
                new_stats = re.findall(pattern_set['numeric'][0], new_content)
                existing_stats = re.findall(pattern_set['numeric'][0], existing_content)
                
                if new_stats and existing_stats:
                    # Check if discussing same metric but different values
                    if self._are_stats_conflicting(new_content, existing_content, new_stats, existing_stats):
                        return 'statistical_conflict'
        
        return None
    
    def _extract_version(self, content: str) -> Optional[str]:
        """Extract version number from content"""
        version_pattern = r'\b(\d+(?:\.\d+)*)\b'
        matches = re.findall(version_pattern, content)
        return matches[0] if matches else None
    
    def _are_stats_conflicting(
        self, 
        new_content: str, 
        existing_content: str, 
        new_stats: List[str], 
        existing_stats: List[str]
    ) -> bool:
        """Check if statistics are conflicting"""
        # Simple heuristic: if both discuss similar metrics but values differ significantly
        metric_keywords = ['users', 'downloads', 'revenue', 'percent', 'growth', 'rate']
        
        for keyword in metric_keywords:
            if keyword in new_content and keyword in existing_content:
                # Compare numerical values
                try:
                    new_val = float(new_stats[0].replace(',', ''))
                    existing_val = float(existing_stats[0].replace(',', ''))
                    # Consider conflicting if difference is more than 20%
                    if abs(new_val - existing_val) / max(new_val, existing_val) > 0.2:
                        return True
                except:
                    pass
        
        return False
    
    def _calculate_severity(self, conflict_type: str, existing_msg: Dict[str, Any]) -> str:
        """Calculate conflict severity based on type and message age"""
        # Get message age
        try:
            msg_time = datetime.fromisoformat(existing_msg.get('timestamp', datetime.now().isoformat()))
            age_hours = (datetime.now() - msg_time).total_seconds() / 3600
        except:
            age_hours = 24
        
        # Severity based on conflict type and age
        if conflict_type in ['existence', 'version']:
            if age_hours < 1:
                return 'critical'
            elif age_hours < 12:
                return 'high'
            else:
                return 'medium'
        elif conflict_type in ['temporal', 'statistics']:
            if age_hours < 6:
                return 'high'
            else:
                return 'medium'
        else:
            return 'low'
    
    def _calculate_dynamic_ttl(self, volatility: float, conflicts: List[Dict[str, Any]]) -> int:
        """Calculate dynamic TTL based on content volatility and conflicts"""
        base_ttl = self.base_ttl
        
        # Apply volatility-based reduction
        if volatility > 0:
            # Find minimum TTL multiplier from detected conflict types
            min_multiplier = 1.0
            for conflict in conflicts:
                conflict_type = conflict['type']
                for pattern_type, config in self.conflict_patterns.items():
                    if conflict_type.startswith(pattern_type):
                        min_multiplier = min(min_multiplier, config['ttl_multiplier'])
            
            # Apply multiplier
            adjusted_ttl = int(base_ttl * min_multiplier)
            
            # Ensure minimum TTL of 1 hour for highly volatile content
            return max(adjusted_ttl, 3600)
        
        return base_ttl
    
    async def _record_conflict_patterns(self, conversation_id: str, conflicts: List[Dict[str, Any]]) -> None:
        """Record conflict patterns for future prediction"""
        try:
            redis_client = await self._get_redis()
            
            # Record to conflict history
            history_entry = {
                'conversation_id': conversation_id,
                'timestamp': datetime.now().isoformat(),
                'conflicts': conflicts
            }
            
            await redis_client.lpush(self.conflict_history_key, json.dumps(history_entry))
            await redis_client.ltrim(self.conflict_history_key, 0, 999)  # Keep last 1000 entries
            
            # Update pattern frequency
            for conflict in conflicts:
                pattern_key = f"{self.pattern_frequency_key}:{conflict['type']}"
                await redis_client.hincrby(pattern_key, conversation_id, 1)
                await redis_client.expire(pattern_key, 604800)  # 7 days TTL
                
        except Exception as e:
            logger.error(f"[PREVENTION] Failed to record conflict patterns: {e}")
    
    async def _predict_future_conflicts(self, content: str, conversation_id: str) -> Dict[str, Any]:
        """Predict likelihood of future conflicts based on historical patterns"""
        try:
            redis_client = await self._get_redis()
            
            # Get historical conflict frequency for this conversation
            total_conflicts = 0
            pattern_frequencies = {}
            
            for conflict_type in self.conflict_patterns.keys():
                pattern_key = f"{self.pattern_frequency_key}:{conflict_type}"
                frequency = await redis_client.hget(pattern_key, conversation_id)
                if frequency:
                    freq_count = int(frequency)
                    total_conflicts += freq_count
                    pattern_frequencies[conflict_type] = freq_count
            
            # Calculate prediction score
            if total_conflicts > 0:
                # High frequency = high likelihood of future conflicts
                prediction_score = min(total_conflicts / 10, 1.0)  # Normalize to 0-1
                
                # Identify most common conflict type
                most_common = max(pattern_frequencies.items(), key=lambda x: x[1]) if pattern_frequencies else None
                
                return {
                    'likelihood': prediction_score,
                    'total_historical_conflicts': total_conflicts,
                    'most_common_type': most_common[0] if most_common else None,
                    'recommended_action': 'monitor_closely' if prediction_score > 0.5 else 'normal'
                }
            
            return {
                'likelihood': 0.0,
                'total_historical_conflicts': 0,
                'most_common_type': None,
                'recommended_action': 'normal'
            }
            
        except Exception as e:
            logger.error(f"[PREVENTION] Failed to predict conflicts: {e}")
            return {
                'likelihood': 0.0,
                'total_historical_conflicts': 0,
                'most_common_type': None,
                'recommended_action': 'normal'
            }
    
    def _should_add_message(self, conflicts: List[Dict[str, Any]], prediction: Dict[str, Any]) -> bool:
        """Determine if message should be added based on conflicts and predictions"""
        # Count critical and high severity conflicts
        critical_count = sum(1 for c in conflicts if c.get('severity') == 'critical')
        high_count = sum(1 for c in conflicts if c.get('severity') == 'high')
        
        # Don't add if multiple critical conflicts
        if critical_count > 1:
            logger.warning(f"[PREVENTION] Blocking message due to {critical_count} critical conflicts")
            return False
        
        # Don't add if high prediction score and existing conflicts
        if prediction.get('likelihood', 0) > 0.7 and (critical_count > 0 or high_count > 2):
            logger.warning(f"[PREVENTION] Blocking message due to high conflict prediction ({prediction['likelihood']:.2f})")
            return False
        
        # Otherwise allow with appropriate TTL adjustment
        return True
    
    def _get_resolution_strategy(self, conflicts: List[Dict[str, Any]]) -> str:
        """Get recommended resolution strategy based on conflicts"""
        if not conflicts:
            return "add_normally"
        
        severities = [c.get('severity', 'low') for c in conflicts]
        
        if 'critical' in severities:
            return "require_verification"
        elif severities.count('high') >= 2:
            return "add_with_warning"
        else:
            return "add_with_reduced_ttl"
    
    async def cleanup_conflict_history(self, days_to_keep: int = 7) -> int:
        """Clean up old conflict history entries"""
        try:
            redis_client = await self._get_redis()
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            history = await redis_client.lrange(self.conflict_history_key, 0, -1)
            
            entries_removed = 0
            entries_to_keep = []
            
            for entry_bytes in history:
                try:
                    entry = json.loads(entry_bytes)
                    entry_time = datetime.fromisoformat(entry['timestamp'])
                    if entry_time > cutoff_date:
                        entries_to_keep.append(entry_bytes)
                    else:
                        entries_removed += 1
                except:
                    continue
            
            if entries_removed > 0:
                # Replace with filtered entries
                await redis_client.delete(self.conflict_history_key)
                if entries_to_keep:
                    await redis_client.rpush(self.conflict_history_key, *entries_to_keep)
                
                logger.info(f"[PREVENTION] Cleaned up {entries_removed} old conflict history entries")
            
            return entries_removed
            
        except Exception as e:
            logger.error(f"[PREVENTION] Failed to cleanup conflict history: {e}")
            return 0

# Global instance
conflict_prevention_engine = ConflictPreventionEngine()