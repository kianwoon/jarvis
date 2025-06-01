"""
Enhanced Conversation Service - Drop-in Replacement
Seamlessly integrates with existing code while providing advanced memory capabilities
"""

import asyncio
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

from app.memory.conversation_retrieval_system import ConversationContextProvider
from app.memory.conversation_memory_manager import ConversationMemoryManager
from app.memory.context_assembly_engine import ContextStrategy


class EnhancedConversationService:
    """
    Enhanced conversation service that can replace existing conversation handling
    Provides backward compatibility while adding advanced capabilities
    """
    
    def __init__(self):
        self.context_provider = ConversationContextProvider()
        self.memory_manager = ConversationMemoryManager()
        
        # Legacy compatibility mode
        self.compatibility_mode = True
        self.fallback_to_legacy = True
        
        # Cache for frequently accessed conversations
        self._context_cache = {}
        self._cache_ttl = 300  # 5 minutes
    
    # ==========================================
    # DROP-IN REPLACEMENTS FOR EXISTING FUNCTIONS
    # ==========================================
    
    def get_conversation_history(self, conversation_id: str) -> str:
        """
        SYNCHRONOUS drop-in replacement for existing get_conversation_history()
        Maintains exact same interface while providing enhanced capabilities
        """
        if not conversation_id:
            return ""
        
        try:
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self._get_enhanced_history(conversation_id, "")
                )
                return result
            finally:
                loop.close()
                
        except Exception as e:
            print(f"[ENHANCED_CONV_SERVICE] Error in sync get_conversation_history: {e}")
            if self.fallback_to_legacy:
                return self._legacy_fallback(conversation_id)
            return ""
    
    def store_conversation_message(self, conversation_id: str, role: str, content: str):
        """
        SYNCHRONOUS drop-in replacement for existing store_conversation_message()
        """
        if not conversation_id:
            return
        
        try:
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    self._store_enhanced_message(conversation_id, role, content)
                )
            finally:
                loop.close()
                
        except Exception as e:
            print(f"[ENHANCED_CONV_SERVICE] Error in sync store_conversation_message: {e}")
            if self.fallback_to_legacy:
                self._legacy_store_fallback(conversation_id, role, content)
    
    # ==========================================
    # ENHANCED ASYNC METHODS 
    # ==========================================
    
    async def get_conversation_history_async(
        self, 
        conversation_id: str, 
        current_query: str = "",
        context_type: str = "standard_chat",
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Enhanced async method with full capabilities
        """
        return await self._get_enhanced_history(
            conversation_id, current_query, context_type, max_tokens
        )
    
    async def store_conversation_message_async(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Enhanced async storage with metadata support
        """
        await self._store_enhanced_message(conversation_id, role, content, metadata)
    
    async def get_contextual_history(
        self,
        conversation_id: str,
        current_query: str,
        strategy: ContextStrategy = ContextStrategy.ADAPTIVE,
        include_metadata: bool = False
    ) -> str:
        """
        Get conversation history optimized for the current query
        Uses intelligent context assembly
        """
        try:
            context, metadata = await self.context_provider.retriever.get_contextual_conversation_history(
                conversation_id=conversation_id,
                current_query=current_query,
                strategy=strategy
            )
            
            if include_metadata:
                return json.dumps({
                    "context": context,
                    "metadata": metadata
                })
            else:
                return context
                
        except Exception as e:
            print(f"[ENHANCED_CONV_SERVICE] Error getting contextual history: {e}")
            return ""
    
    async def search_conversation(
        self,
        conversation_id: str,
        query: str,
        limit: int = 10,
        include_context: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search within conversation history using vector similarity
        """
        try:
            results = await self.context_provider.search_conversation(
                conversation_id, query, limit
            )
            
            if include_context:
                # Add surrounding context for each result
                enhanced_results = []
                for result in results:
                    message = result["message"]
                    
                    # Get context around this message
                    context_window = await self._get_message_context_window(
                        conversation_id, message.message_id
                    )
                    
                    enhanced_result = {
                        **result,
                        "context_window": context_window,
                        "message_content": message.content,
                        "timestamp": message.timestamp.isoformat(),
                        "role": message.role
                    }
                    enhanced_results.append(enhanced_result)
                
                return enhanced_results
            else:
                return results
                
        except Exception as e:
            print(f"[ENHANCED_CONV_SERVICE] Error searching conversation: {e}")
            return []
    
    async def get_conversation_summary(
        self,
        conversation_id: str,
        summary_type: str = "key_points"
    ) -> Dict[str, Any]:
        """
        Generate intelligent conversation summary
        """
        try:
            return await self.context_provider.retriever.get_conversation_summary(
                conversation_id, summary_type
            )
        except Exception as e:
            print(f"[ENHANCED_CONV_SERVICE] Error generating summary: {e}")
            return {"summary": "", "error": str(e)}
    
    async def export_conversation(
        self,
        conversation_id: str,
        format: str = "json",
        include_metadata: bool = True
    ) -> str:
        """
        Export full conversation in various formats
        """
        try:
            return await self.memory_manager.export_conversation(conversation_id, format)
        except Exception as e:
            print(f"[ENHANCED_CONV_SERVICE] Error exporting conversation: {e}")
            return ""
    
    # ==========================================
    # INTEGRATION WITH EXISTING RAG SYSTEM
    # ==========================================
    
    async def get_rag_optimized_context(
        self,
        conversation_id: str,
        current_query: str,
        max_tokens: int = 6000
    ) -> str:
        """
        Get conversation context optimized for RAG queries
        Leaves room for RAG content while preserving conversation flow
        """
        try:
            context, metadata = await self.context_provider.retriever.get_contextual_conversation_history(
                conversation_id=conversation_id,
                current_query=current_query,
                context_type="rag_enhanced",
                strategy=ContextStrategy.RELEVANCE_FOCUS
            )
            
            return context
            
        except Exception as e:
            print(f"[ENHANCED_CONV_SERVICE] Error getting RAG context: {e}")
            return ""
    
    async def get_multi_agent_context(
        self,
        conversation_id: str,
        current_query: str,
        agent_name: Optional[str] = None
    ) -> str:
        """
        Get conversation context optimized for multi-agent systems
        Conservative token usage for agent coordination
        """
        try:
            # Use thread-aware strategy for multi-agent coordination
            context, metadata = await self.context_provider.retriever.get_contextual_conversation_history(
                conversation_id=conversation_id,
                current_query=current_query,
                context_type="multi_agent",
                strategy=ContextStrategy.THREAD_AWARE
            )
            
            return context
            
        except Exception as e:
            print(f"[ENHANCED_CONV_SERVICE] Error getting multi-agent context: {e}")
            return ""
    
    # ==========================================
    # INTERNAL METHODS
    # ==========================================
    
    async def _get_enhanced_history(
        self,
        conversation_id: str,
        current_query: str = "",
        context_type: str = "standard_chat",
        max_tokens: Optional[int] = None
    ) -> str:
        """Internal method for getting enhanced conversation history"""
        
        # Check cache first
        cache_key = f"{conversation_id}_{hash(current_query)}_{context_type}"
        if cache_key in self._context_cache:
            cached_data = self._context_cache[cache_key]
            if (datetime.now() - cached_data["timestamp"]).seconds < self._cache_ttl:
                return cached_data["context"]
        
        try:
            # Get context using enhanced system
            context, metadata = await self.context_provider.retriever.get_contextual_conversation_history(
                conversation_id=conversation_id,
                current_query=current_query,
                context_type=context_type
            )
            
            # Cache result
            self._context_cache[cache_key] = {
                "context": context,
                "metadata": metadata,
                "timestamp": datetime.now()
            }
            
            # Clean old cache entries
            await self._clean_cache()
            
            return context
            
        except Exception as e:
            print(f"[ENHANCED_CONV_SERVICE] Enhanced history failed: {e}")
            return ""
    
    async def _store_enhanced_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Internal method for storing messages with enhancements"""
        try:
            # Store in enhanced system
            await self.context_provider.store_conversation_message(
                conversation_id, role, content, metadata
            )
            
            # Clear relevant cache entries
            keys_to_remove = [key for key in self._context_cache.keys() if conversation_id in key]
            for key in keys_to_remove:
                del self._context_cache[key]
                
        except Exception as e:
            print(f"[ENHANCED_CONV_SERVICE] Enhanced storage failed: {e}")
            raise
    
    async def _get_message_context_window(
        self,
        conversation_id: str,
        message_id: str,
        window_size: int = 3
    ) -> List[Dict[str, Any]]:
        """Get messages before and after a specific message"""
        try:
            # Get all messages and find the target
            all_messages = await self.memory_manager.get_full_conversation(conversation_id)
            
            target_index = None
            for i, msg in enumerate(all_messages):
                if msg.message_id == message_id:
                    target_index = i
                    break
            
            if target_index is None:
                return []
            
            # Get window around target message
            start_index = max(0, target_index - window_size)
            end_index = min(len(all_messages), target_index + window_size + 1)
            
            window_messages = all_messages[start_index:end_index]
            
            return [
                {
                    "message_id": msg.message_id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "is_target": msg.message_id == message_id
                }
                for msg in window_messages
            ]
            
        except Exception as e:
            print(f"[ENHANCED_CONV_SERVICE] Error getting context window: {e}")
            return []
    
    async def _clean_cache(self):
        """Clean old cache entries"""
        try:
            current_time = datetime.now()
            keys_to_remove = []
            
            for key, data in self._context_cache.items():
                if (current_time - data["timestamp"]).seconds > self._cache_ttl:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._context_cache[key]
                
        except Exception as e:
            print(f"[ENHANCED_CONV_SERVICE] Cache cleanup error: {e}")
    
    def _legacy_fallback(self, conversation_id: str) -> str:
        """Fallback to legacy conversation handling"""
        try:
            # Import the legacy function
            from app.langchain.service import get_conversation_history
            return get_conversation_history(conversation_id)
        except Exception as e:
            print(f"[ENHANCED_CONV_SERVICE] Legacy fallback failed: {e}")
            return ""
    
    def _legacy_store_fallback(self, conversation_id: str, role: str, content: str):
        """Fallback to legacy storage"""
        try:
            from app.langchain.service import store_conversation_message
            store_conversation_message(conversation_id, role, content)
        except Exception as e:
            print(f"[ENHANCED_CONV_SERVICE] Legacy store fallback failed: {e}")


# Global instance for easy integration
enhanced_conversation_service = EnhancedConversationService()


# ==========================================
# COMPATIBILITY FUNCTIONS FOR EXISTING CODE
# ==========================================

def get_conversation_history_enhanced(conversation_id: str, current_query: str = "") -> str:
    """
    Enhanced version of get_conversation_history with query-aware context
    Can be used as a drop-in replacement in existing code
    """
    return enhanced_conversation_service.get_conversation_history(conversation_id)


def store_conversation_message_enhanced(conversation_id: str, role: str, content: str):
    """
    Enhanced version of store_conversation_message
    Can be used as a drop-in replacement in existing code
    """
    enhanced_conversation_service.store_conversation_message(conversation_id, role, content)


async def get_conversation_context_for_rag(conversation_id: str, current_query: str) -> str:
    """
    Get conversation context optimized for RAG operations
    Use this in RAG functions to get better conversation context
    """
    return await enhanced_conversation_service.get_rag_optimized_context(
        conversation_id, current_query
    )


async def get_conversation_context_for_agents(conversation_id: str, current_query: str) -> str:
    """
    Get conversation context optimized for multi-agent systems
    Use this in multi-agent functions for better context management
    """
    return await enhanced_conversation_service.get_multi_agent_context(
        conversation_id, current_query
    )