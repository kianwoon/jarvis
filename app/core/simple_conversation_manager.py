"""
Simple Conversation History Manager
Manages conversation history for standard chat using Redis
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import redis.asyncio as redis
from app.core.redis_client import get_async_redis_client
from app.core.conflict_prevention_engine import conflict_prevention_engine
import logging

logger = logging.getLogger(__name__)

class SimpleConversationManager:
    """Simple manager for storing and retrieving conversation history"""
    
    def __init__(self):
        self.redis_client = None
        self.ttl = 3600  # 1 hour TTL for conversations
        
    async def _get_redis(self) -> redis.Redis:
        """Get Redis client"""
        if not self.redis_client:
            self.redis_client = await get_async_redis_client()
        return self.redis_client
    
    async def add_message(self, conversation_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a message to conversation history with conflict prevention and dynamic TTL"""
        try:
            # CONFLICT PREVENTION: Check for conflicts before adding
            conflict_check = await conflict_prevention_engine.check_for_conflicts(
                new_content=content,
                conversation_id=conversation_id,
                role=role,
                metadata=metadata
            )
            
            # Log conflict analysis
            if conflict_check['has_conflicts']:
                logger.warning(f"[PREVENTION] Detected {len(conflict_check['conflicts'])} conflicts for {role} message")
                for conflict in conflict_check['conflicts']:
                    logger.info(f"[PREVENTION] Conflict: {conflict['type']} (severity: {conflict['severity']})")
            
            # Check if message should be added
            if not conflict_check['should_add']:
                logger.warning(f"[PREVENTION] Message blocked due to conflicts. Strategy: {conflict_check.get('resolution_strategy')}")
                # Store blocked message metadata for analysis
                await self._record_blocked_message(conversation_id, role, content, conflict_check)
                return
            
            redis_client = await self._get_redis()
            
            now = datetime.now()
            
            # Calculate dynamic TTL based on volatility
            dynamic_ttl = conflict_check['recommended_ttl']
            
            message = {
                "role": role,
                "content": content,
                "timestamp": now.isoformat(),
                "metadata": metadata or {},
                "freshness_score": 1.0,  # New messages start with max freshness
                "source_type": metadata.get("source_type", "conversation") if metadata else "conversation",
                "created_at": now.timestamp(),
                "volatility_score": conflict_check['volatility_score'],
                "ttl": dynamic_ttl,
                "conflict_check": {
                    "performed": True,
                    "conflicts_found": len(conflict_check.get('conflicts', [])),
                    "resolution_strategy": conflict_check.get('resolution_strategy', 'normal')
                }
            }
            
            # Add warning metadata if conflicts exist but message is allowed
            if conflict_check['has_conflicts']:
                message['metadata']['has_conflicts'] = True
                message['metadata']['conflict_types'] = [c['type'] for c in conflict_check['conflicts']]
                message['metadata']['conflict_warning'] = f"Added with {len(conflict_check['conflicts'])} known conflicts"
            
            # Store in Redis list
            key = f"conversation:{conversation_id}:messages"
            await redis_client.rpush(key, json.dumps(message))
            
            # Set dynamic TTL on the conversation
            await redis_client.expire(key, dynamic_ttl)
            
            logger.info(f"Added {role} message to conversation {conversation_id} with TTL={dynamic_ttl}s (volatility={conflict_check['volatility_score']:.2f})")
            
        except Exception as e:
            logger.error(f"Failed to add message to conversation {conversation_id}: {e}")
            # Don't fail the request if Redis fails
    
    async def get_conversation_history(self, conversation_id: str, limit: int = 6, max_age_hours: float = 24.0, include_superseded: bool = False) -> List[Dict[str, Any]]:
        """Get recent conversation history with freshness scoring, age filtering, and conflict resolution awareness"""
        try:
            redis_client = await self._get_redis()
            
            key = f"conversation:{conversation_id}:messages"
            
            # Get last N messages
            messages = await redis_client.lrange(key, -limit, -1)
            
            # Parse messages and apply freshness scoring
            history = []
            now = datetime.now()
            cutoff_time = now.timestamp() - (max_age_hours * 3600)  # Convert hours to seconds
            
            for msg in messages:
                try:
                    message = json.loads(msg)
                    
                    # Add backward compatibility for old messages without freshness data
                    if "created_at" not in message:
                        try:
                            # Try to parse timestamp
                            msg_time = datetime.fromisoformat(message.get("timestamp", now.isoformat()))
                            message["created_at"] = msg_time.timestamp()
                        except:
                            message["created_at"] = now.timestamp()
                    
                    # Skip messages older than max_age_hours
                    if message["created_at"] < cutoff_time:
                        logger.debug(f"Filtering out message older than {max_age_hours} hours")
                        continue
                    
                    # Skip superseded messages unless explicitly requested
                    if not include_superseded and message.get('metadata', {}).get('superseded', False):
                        logger.debug(f"Filtering out superseded message from conflict resolution")
                        continue
                    
                    # Calculate age-based freshness score (1.0 = fresh, 0.0 = max age)
                    age_hours = (now.timestamp() - message["created_at"]) / 3600
                    age_factor = max(0.0, 1.0 - (age_hours / max_age_hours))
                    
                    # Apply freshness decay
                    message["freshness_score"] = age_factor
                    message["age_hours"] = age_hours
                    
                    # Add source type if missing (backward compatibility)
                    if "source_type" not in message:
                        message["source_type"] = "conversation"
                    
                    history.append(message)
                except Exception as e:
                    logger.warning(f"Failed to parse message: {e}")
                    continue
                    
            logger.info(f"Retrieved {len(history)} fresh messages for conversation {conversation_id}")
            return history
            
        except Exception as e:
            logger.error(f"Failed to get conversation history for {conversation_id}: {e}")
            return []
    
    async def clear_conversation(self, conversation_id: str) -> None:
        """Clear conversation history"""
        try:
            redis_client = await self._get_redis()
            key = f"conversation:{conversation_id}:messages"
            await redis_client.delete(key)
            logger.info(f"Cleared conversation {conversation_id}")
        except Exception as e:
            logger.error(f"Failed to clear conversation {conversation_id}: {e}")
    
    async def mark_message_superseded(self, conversation_id: str, message_index: int, conflict_metadata: dict) -> bool:
        """Mark a specific message as superseded due to conflict resolution"""
        try:
            redis_client = await self._get_redis()
            key = f"conversation:{conversation_id}:messages"
            
            # Get all messages
            messages = await redis_client.lrange(key, 0, -1)
            if message_index >= len(messages):
                logger.warning(f"Message index {message_index} out of range for conversation {conversation_id}")
                return False
            
            # Parse and update the specific message
            message = json.loads(messages[message_index])
            message['metadata'] = message.get('metadata', {})
            message['metadata'].update({
                'superseded': True,
                'superseded_at': datetime.now().isoformat(),
                'conflict_resolution': conflict_metadata,
                'freshness_score': 0.0  # Mark as completely stale
            })
            
            # Update the message in Redis
            await redis_client.lset(key, message_index, json.dumps(message))
            
            logger.info(f"Marked message {message_index} as superseded in conversation {conversation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to mark message superseded in conversation {conversation_id}: {e}")
            return False
    
    async def remove_superseded_messages(self, conversation_id: str) -> int:
        """Remove all messages marked as superseded from conversation"""
        try:
            redis_client = await self._get_redis()
            key = f"conversation:{conversation_id}:messages"
            
            # Get all messages
            messages = await redis_client.lrange(key, 0, -1)
            if not messages:
                return 0
            
            # Filter out superseded messages
            valid_messages = []
            removed_count = 0
            
            for msg in messages:
                try:
                    message = json.loads(msg)
                    if not message.get('metadata', {}).get('superseded', False):
                        valid_messages.append(msg)
                    else:
                        removed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to parse message during cleanup: {e}")
                    # Keep unparseable messages to avoid data loss
                    valid_messages.append(msg)
            
            if removed_count > 0:
                # Clear and rebuild conversation with valid messages only
                await redis_client.delete(key)
                if valid_messages:
                    await redis_client.rpush(key, *valid_messages)
                    await redis_client.expire(key, self.ttl)
                
                logger.info(f"Removed {removed_count} superseded messages from conversation {conversation_id}")
            
            return removed_count
            
        except Exception as e:
            logger.error(f"Failed to remove superseded messages from conversation {conversation_id}: {e}")
            return 0
    
    async def _record_blocked_message(self, conversation_id: str, role: str, content: str, conflict_analysis: dict) -> None:
        """Record blocked messages for analysis and learning"""
        try:
            redis_client = await self._get_redis()
            blocked_key = f"conversation:{conversation_id}:blocked"
            
            blocked_entry = {
                'timestamp': datetime.now().isoformat(),
                'role': role,
                'content': content[:500],  # Store first 500 chars
                'conflicts': conflict_analysis.get('conflicts', []),
                'prediction': conflict_analysis.get('prediction', {}),
                'resolution_strategy': conflict_analysis.get('resolution_strategy')
            }
            
            # Store blocked message log (keep last 20 entries)
            await redis_client.lpush(blocked_key, json.dumps(blocked_entry))
            await redis_client.ltrim(blocked_key, 0, 19)
            await redis_client.expire(blocked_key, self.ttl * 2)  # Longer TTL for analysis
            
            logger.info(f"[PREVENTION] Recorded blocked message for conversation {conversation_id}")
            
        except Exception as e:
            logger.error(f"[PREVENTION] Failed to record blocked message: {e}")
    
    async def add_conflict_resolution_log(self, conversation_id: str, conflict_details: dict) -> None:
        """Add a conflict resolution log entry to track cache cleanup actions"""
        try:
            redis_client = await self._get_redis()
            log_key = f"conversation:{conversation_id}:conflicts"
            
            conflict_entry = {
                'timestamp': datetime.now().isoformat(),
                'conflict_type': conflict_details.get('type', 'unknown'),
                'resolution_action': conflict_details.get('action', 'cache_cleanup'),
                'affected_messages': conflict_details.get('affected_count', 0),
                'superseded_by': conflict_details.get('superseded_by', 'unknown')
            }
            
            # Store conflict resolution log (keep last 10 entries)
            await redis_client.lpush(log_key, json.dumps(conflict_entry))
            await redis_client.ltrim(log_key, 0, 9)  # Keep only 10 most recent
            await redis_client.expire(log_key, self.ttl * 2)  # Longer TTL for logs
            
            logger.info(f"Added conflict resolution log for conversation {conversation_id}")
            
        except Exception as e:
            logger.error(f"Failed to add conflict resolution log for conversation {conversation_id}: {e}")
    
    def filter_contradictory_history(self, history: List[Dict[str, Any]], tool_context: str) -> List[Dict[str, Any]]:
        """Filter out conversation history messages that contradict fresh tool/search results"""
        if not history or not tool_context:
            return history
        
        import re
        
        tool_lower = tool_context.lower()
        filtered_history = []
        
        # Define contradiction patterns - entity + positive/negative action combinations
        contradiction_patterns = [
            # OpenAI patterns - enhanced for ChatGPT-5 and GPT model detection
            {
                'entity': [
                    'openai', 'open ai', 'openai inc', 'openai limited',
                    'chatgpt', 'chat gpt', 'chatgpt-5', 'chatgpt 5', 'gpt-5', 'gpt 5',
                    'gpt5', 'chatgpt5'
                ],
                'positive_actions': [
                    'released', 'launches', 'introduces', 'announces', 'open-source', 'open source', 
                    'gpt-oss', 'made available', 'published', 'shared', 'provides', 'offers',
                    'open-sourced', 'opensourced', 'made public', 'freely available', 'is out',
                    'now available', 'has launched', 'has released', 'exists', 'available'
                ],
                'negative_phrases': [
                    'has not released', 'no.*open-source', 'no.*open source', 'not released any', 
                    'does not have.*open', 'no open.*source', 'never released', 'hasn\'t released',
                    'no public.*model', 'not made.*available', 'not open.*source', 'closed.*source',
                    'proprietary.*only', 'not.*open.*source', 'does not exist', 'doesn\'t exist',
                    'not exist', 'persistent myth', 'is.*myth', 'not.*real', 'not.*available',
                    'no.*official.*release', 'no.*credible.*evidence', 'latest.*model.*is'
                ],
                'topics': [
                    'models', 'gpt', 'language model', 'ai models', 'gpt-4', 'gpt-3', 'gpt-5', 
                    'chatgpt-5', 'chatgpt 5', 'gpt 5', 'neural network', 'chatgpt', 'chat model'
                ]
            },
            # Meta/Facebook patterns
            {
                'entity': ['meta', 'facebook', 'meta ai'],
                'positive_actions': [
                    'released', 'launches', 'introduces', 'announces', 'open-source', 'open source', 
                    'made available', 'published', 'shared', 'provides', 'offers', 'open-sourced'
                ],
                'negative_phrases': [
                    'has not released', 'no.*open-source', 'no.*open source', 'not released any', 
                    'does not have.*open', 'never released', 'hasn\'t released', 'not open.*source'
                ],
                'topics': ['models', 'llama', 'language model', 'ai models', 'llama-2', 'llama-3']
            },
            # Google patterns
            {
                'entity': ['google', 'alphabet', 'deepmind'],
                'positive_actions': [
                    'released', 'launches', 'introduces', 'announces', 'open-source', 'open source', 
                    'made available', 'published', 'shared', 'provides', 'offers', 'open-sourced'
                ],
                'negative_phrases': [
                    'has not released', 'no.*open-source', 'no.*open source', 'not released any', 
                    'does not have.*open', 'never released', 'hasn\'t released', 'not open.*source'
                ],
                'topics': ['models', 'gemini', 'bard', 'language model', 'ai models', 'palm', 'bert']
            },
            # Anthropic patterns
            {
                'entity': ['anthropic'],
                'positive_actions': [
                    'released', 'launches', 'introduces', 'announces', 'open-source', 'open source', 
                    'made available', 'published', 'shared', 'provides', 'offers', 'open-sourced'
                ],
                'negative_phrases': [
                    'has not released', 'no.*open-source', 'no.*open source', 'not released any', 
                    'does not have.*open', 'never released', 'hasn\'t released', 'not open.*source'
                ],
                'topics': ['models', 'claude', 'language model', 'ai models']
            },
            # Microsoft patterns
            {
                'entity': ['microsoft'],
                'positive_actions': [
                    'released', 'launches', 'introduces', 'announces', 'open-source', 'open source', 
                    'made available', 'published', 'shared', 'provides', 'offers', 'open-sourced'
                ],
                'negative_phrases': [
                    'has not released', 'no.*open-source', 'no.*open source', 'not released any', 
                    'does not have.*open', 'never released', 'hasn\'t released', 'not open.*source'
                ],
                'topics': ['models', 'phi', 'language model', 'ai models', 'bing']
            },
            # Generic high-tech company patterns for broader coverage
            {
                'entity': ['apple', 'amazon', 'nvidia', 'intel', 'huawei', 'baidu'],
                'positive_actions': [
                    'released', 'launches', 'introduces', 'announces', 'open-source', 'open source', 
                    'made available', 'published', 'shared', 'provides', 'offers', 'open-sourced'
                ],
                'negative_phrases': [
                    'has not released', 'no.*open-source', 'no.*open source', 'not released any', 
                    'does not have.*open', 'never released', 'hasn\'t released', 'not open.*source'
                ],
                'topics': ['models', 'language model', 'ai models', 'machine learning', 'neural network']
            }
        ]
        
        # Check if tool context contains positive information about an entity
        has_positive_info = False
        detected_entity = None
        detected_topic = None
        
        for pattern in contradiction_patterns:
            # Check if tool context mentions entity + positive actions + topic
            entity_found = any(entity in tool_lower for entity in pattern['entity'])
            action_found = any(action in tool_lower for action in pattern['positive_actions'])
            topic_found = any(topic in tool_lower for topic in pattern['topics'])
            
            if entity_found and action_found and topic_found:
                has_positive_info = True
                detected_entity = pattern['entity'][0]
                detected_topic = next((topic for topic in pattern['topics'] if topic in tool_lower), pattern['topics'][0])
                logger.info(f"[FILTER] Detected positive information about {detected_entity} and {detected_topic} in tool context")
                break
        
        if not has_positive_info:
            logger.info("[FILTER] No positive information detected in tool context, returning original history")
            return history
            
        # Filter out contradictory messages from conversation history
        messages_removed = 0
        for msg in history:
            content = msg.get('content', '').lower()
            role = msg.get('role', '')
            
            should_filter = False
            
            # Check if this message contradicts the positive information
            for pattern in contradiction_patterns:
                if pattern['entity'][0] == detected_entity:
                    entity_in_msg = any(entity in content for entity in pattern['entity'])
                    negative_phrase_found = any(re.search(phrase, content) for phrase in pattern['negative_phrases'])
                    topic_in_msg = detected_topic in content
                    
                    if entity_in_msg and negative_phrase_found and topic_in_msg:
                        should_filter = True
                        messages_removed += 1
                        logger.info(f"[FILTER] Removing contradictory {role} message: '{msg.get('content', '')[:100]}...'")
                        break
            
            if not should_filter:
                filtered_history.append(msg)
        
        logger.info(f"[FILTER] Filtered conversation history: {messages_removed} contradictory messages removed, {len(filtered_history)} messages remain")
        return filtered_history

    def format_history_for_prompt(self, history: List[Dict[str, Any]], current_question: str) -> str:
        """Format conversation history for inclusion in LLM prompt"""
        if not history:
            return current_question
            
        # Build conversation context
        formatted_history = []
        for msg in history:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            if role == 'user':
                formatted_history.append(f"User: {content}")
            elif role == 'assistant':
                formatted_history.append(f"Assistant: {content}")
        
        # Combine history with current question
        conversation_context = "\n\n".join(formatted_history)
        
        return f"""Previous conversation:
{conversation_context}

Current question: {current_question}

Please answer the current question while considering the context of our previous conversation."""

    async def get_conflict_statistics(self, conversation_id: str) -> Dict[str, Any]:
        """Get conflict prevention statistics for a conversation"""
        try:
            redis_client = await self._get_redis()
            
            # Get blocked messages count
            blocked_key = f"conversation:{conversation_id}:blocked"
            blocked_messages = await redis_client.llen(blocked_key)
            
            # Get conflict resolution logs
            log_key = f"conversation:{conversation_id}:conflicts"
            conflict_logs = await redis_client.llen(log_key)
            
            # Get messages with conflicts
            key = f"conversation:{conversation_id}:messages"
            messages = await redis_client.lrange(key, 0, -1)
            
            messages_with_conflicts = 0
            total_conflicts = 0
            conflict_types = {}
            
            for msg in messages:
                try:
                    message = json.loads(msg)
                    if message.get('metadata', {}).get('has_conflicts'):
                        messages_with_conflicts += 1
                        types = message['metadata'].get('conflict_types', [])
                        for ct in types:
                            conflict_types[ct] = conflict_types.get(ct, 0) + 1
                            total_conflicts += 1
                except:
                    continue
            
            return {
                'blocked_messages': blocked_messages,
                'messages_with_conflicts': messages_with_conflicts,
                'total_conflicts': total_conflicts,
                'conflict_types': conflict_types,
                'resolution_logs': conflict_logs,
                'prevention_active': True
            }
            
        except Exception as e:
            logger.error(f"Failed to get conflict statistics: {e}")
            return {}

# Global instance
conversation_manager = SimpleConversationManager()