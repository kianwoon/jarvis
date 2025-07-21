"""
Simple Conversation History Manager
Manages conversation history for standard chat using Redis
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import redis.asyncio as redis
from app.core.redis_client import get_async_redis_client
import logging

logger = logging.getLogger(__name__)

class SimpleConversationManager:
    """Simple manager for storing and retrieving conversation history"""
    
    def __init__(self):
        self.redis_client = None
        self.ttl = 86400  # 24 hours default TTL for conversations
        
    async def _get_redis(self) -> redis.Redis:
        """Get Redis client"""
        if not self.redis_client:
            self.redis_client = await get_async_redis_client()
        return self.redis_client
    
    async def add_message(self, conversation_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a message to conversation history"""
        try:
            redis_client = await self._get_redis()
            
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            # Store in Redis list
            key = f"conversation:{conversation_id}:messages"
            await redis_client.rpush(key, json.dumps(message))
            
            # Set TTL on the conversation
            await redis_client.expire(key, self.ttl)
            
            logger.info(f"Added {role} message to conversation {conversation_id}")
            
        except Exception as e:
            logger.error(f"Failed to add message to conversation {conversation_id}: {e}")
            # Don't fail the request if Redis fails
    
    async def get_conversation_history(self, conversation_id: str, limit: int = 6) -> List[Dict[str, Any]]:
        """Get recent conversation history (default: last 3 exchanges = 6 messages)"""
        try:
            redis_client = await self._get_redis()
            
            key = f"conversation:{conversation_id}:messages"
            
            # Get last N messages
            messages = await redis_client.lrange(key, -limit, -1)
            
            # Parse messages
            history = []
            for msg in messages:
                try:
                    history.append(json.loads(msg))
                except:
                    continue
                    
            logger.info(f"Retrieved {len(history)} messages for conversation {conversation_id}")
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

# Global instance
conversation_manager = SimpleConversationManager()