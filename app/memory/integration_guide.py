"""
Integration Guide for Enhanced Conversation Memory System
Provides step-by-step integration with existing codebase
"""

from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime

from app.memory.enhanced_conversation_service import enhanced_conversation_service


class ConversationMemoryIntegrator:
    """
    Helper class to integrate enhanced conversation memory with existing codebase
    Provides migration utilities and compatibility checks
    """
    
    def __init__(self):
        self.enhanced_service = enhanced_conversation_service
        self.migration_log = []
        
    # ==========================================
    # STEP-BY-STEP INTEGRATION METHODS
    # ==========================================
    
    def step1_verify_dependencies(self) -> Dict[str, bool]:
        """
        Step 1: Verify all dependencies are available
        Run this first to ensure system is ready
        """
        dependencies = {
            "redis_available": False,
            "vector_store_available": False,
            "embeddings_available": False,
            "async_support": False
        }
        
        try:
            # Check Redis
            from app.core.redis_client import get_redis_client
            redis_client = get_redis_client()
            if redis_client:
                redis_client.ping()
                dependencies["redis_available"] = True
        except Exception as e:
            print(f"Redis check failed: {e}")
        
        try:
            # Check vector store
            from app.core.vector_db_settings_cache import get_vector_db_settings
            vector_cfg = get_vector_db_settings()
            if vector_cfg and "milvus" in vector_cfg:
                dependencies["vector_store_available"] = True
        except Exception as e:
            print(f"Vector store check failed: {e}")
        
        try:
            # Check embeddings
            from app.core.embedding_settings_cache import get_embedding_settings
            embedding_cfg = get_embedding_settings()
            if embedding_cfg:
                dependencies["embeddings_available"] = True
        except Exception as e:
            print(f"Embeddings check failed: {e}")
        
        try:
            # Check async support
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.close()
            dependencies["async_support"] = True
        except Exception as e:
            print(f"Async support check failed: {e}")
        
        return dependencies
    
    def step2_create_backup_compatibility_layer(self) -> str:
        """
        Step 2: Create a backup compatibility layer
        Returns the code to add to existing service.py
        """
        
        backup_code = '''
# ==========================================
# ENHANCED CONVERSATION MEMORY BACKUP LAYER
# Add this to your existing service.py file
# ==========================================

# Keep original functions as backup
_original_get_conversation_history = get_conversation_history
_original_store_conversation_message = store_conversation_message

def get_conversation_history_with_fallback(conversation_id: str, current_query: str = "") -> str:
    """Enhanced conversation history with fallback to original"""
    try:
        # Try enhanced version first
        from app.memory.enhanced_conversation_service import get_conversation_history_enhanced
        return get_conversation_history_enhanced(conversation_id, current_query)
    except Exception as e:
        print(f"[FALLBACK] Enhanced conversation history failed: {e}")
        # Fallback to original
        return _original_get_conversation_history(conversation_id)

def store_conversation_message_with_fallback(conversation_id: str, role: str, content: str):
    """Enhanced message storage with fallback to original"""
    try:
        # Try enhanced version first
        from app.memory.enhanced_conversation_service import store_conversation_message_enhanced
        store_conversation_message_enhanced(conversation_id, role, content)
    except Exception as e:
        print(f"[FALLBACK] Enhanced conversation storage failed: {e}")
        # Fallback to original
        _original_store_conversation_message(conversation_id, role, content)

# Replace original functions with fallback versions
get_conversation_history = get_conversation_history_with_fallback
store_conversation_message = store_conversation_message_with_fallback
'''
        
        return backup_code
    
    def step3_migrate_existing_conversations(self, conversation_ids: List[str]) -> Dict[str, Any]:
        """
        Step 3: Migrate existing conversation data to enhanced system
        """
        migration_results = {
            "migrated_conversations": 0,
            "failed_conversations": 0,
            "total_messages": 0,
            "errors": []
        }
        
        try:
            # Import original conversation cache
            from app.langchain.service import _conversation_cache
            
            for conversation_id in conversation_ids:
                try:
                    if conversation_id in _conversation_cache:
                        messages = _conversation_cache[conversation_id]
                        
                        # Migrate each message
                        for msg in messages:
                            try:
                                # Convert to enhanced format
                                role = msg.get("role", "user")
                                content = msg.get("content", "")
                                timestamp_str = msg.get("timestamp")
                                
                                if timestamp_str:
                                    # Store with timestamp preservation
                                    metadata = {"migrated_from_cache": True, "original_timestamp": timestamp_str}
                                else:
                                    metadata = {"migrated_from_cache": True}
                                
                                # Use sync version for migration
                                self.enhanced_service.store_conversation_message(
                                    conversation_id, role, content
                                )
                                
                                migration_results["total_messages"] += 1
                                
                            except Exception as e:
                                error_msg = f"Failed to migrate message in {conversation_id}: {e}"
                                migration_results["errors"].append(error_msg)
                        
                        migration_results["migrated_conversations"] += 1
                        
                except Exception as e:
                    migration_results["failed_conversations"] += 1
                    error_msg = f"Failed to migrate conversation {conversation_id}: {e}"
                    migration_results["errors"].append(error_msg)
            
        except Exception as e:
            migration_results["errors"].append(f"Migration setup failed: {e}")
        
        return migration_results
    
    def step4_gradual_rollout_strategy(self) -> Dict[str, Any]:
        """
        Step 4: Provide gradual rollout strategy
        Returns configuration for phased deployment
        """
        
        rollout_strategy = {
            "phase_1": {
                "description": "Enable for new conversations only",
                "percentage": 0,  # 0% of existing conversations
                "features": ["enhanced_storage", "basic_retrieval"],
                "fallback_enabled": True,
                "monitoring_level": "high"
            },
            "phase_2": {
                "description": "Enable for 25% of conversations",
                "percentage": 25,
                "features": ["enhanced_storage", "basic_retrieval", "context_assembly"],
                "fallback_enabled": True,
                "monitoring_level": "medium"
            },
            "phase_3": {
                "description": "Enable for 75% of conversations",
                "percentage": 75,
                "features": ["full_enhanced_memory", "vector_search", "intelligent_context"],
                "fallback_enabled": True,
                "monitoring_level": "medium"
            },
            "phase_4": {
                "description": "Full rollout with all features",
                "percentage": 100,
                "features": ["all_features"],
                "fallback_enabled": False,
                "monitoring_level": "low"
            }
        }
        
        return rollout_strategy
    
    # ==========================================
    # TESTING AND VALIDATION METHODS
    # ==========================================
    
    async def test_enhanced_system(self, test_conversation_id: str = "test_conv_123") -> Dict[str, Any]:
        """
        Test the enhanced conversation system end-to-end
        """
        test_results = {
            "storage_test": False,
            "retrieval_test": False,
            "search_test": False,
            "context_assembly_test": False,
            "errors": []
        }
        
        try:
            # Test 1: Storage
            await self.enhanced_service.store_conversation_message_async(
                test_conversation_id, "user", "This is a test message for storage"
            )
            await self.enhanced_service.store_conversation_message_async(
                test_conversation_id, "assistant", "This is a test response from assistant"
            )
            test_results["storage_test"] = True
            
            # Test 2: Retrieval
            history = await self.enhanced_service.get_conversation_history_async(
                test_conversation_id, "test query"
            )
            if "test message" in history:
                test_results["retrieval_test"] = True
            
            # Test 3: Search
            search_results = await self.enhanced_service.search_conversation(
                test_conversation_id, "test", limit=5
            )
            if search_results:
                test_results["search_test"] = True
            
            # Test 4: Context Assembly
            contextual_history = await self.enhanced_service.get_contextual_history(
                test_conversation_id, "What did we discuss about testing?"
            )
            if contextual_history:
                test_results["context_assembly_test"] = True
            
        except Exception as e:
            test_results["errors"].append(f"Test failed: {e}")
        
        return test_results
    
    def validate_integration(self) -> Dict[str, Any]:
        """
        Validate that integration is working correctly
        """
        validation_results = {
            "dependencies_ok": False,
            "services_available": False,
            "compatibility_maintained": False,
            "performance_acceptable": False,
            "recommendations": []
        }
        
        # Check dependencies
        deps = self.step1_verify_dependencies()
        if all(deps.values()):
            validation_results["dependencies_ok"] = True
        else:
            missing_deps = [k for k, v in deps.items() if not v]
            validation_results["recommendations"].append(f"Missing dependencies: {missing_deps}")
        
        # Check services
        try:
            # Test sync methods
            test_history = self.enhanced_service.get_conversation_history("test_id")
            validation_results["services_available"] = True
        except Exception as e:
            validation_results["recommendations"].append(f"Service test failed: {e}")
        
        # Check compatibility
        try:
            from app.memory.enhanced_conversation_service import get_conversation_history_enhanced
            validation_results["compatibility_maintained"] = True
        except Exception as e:
            validation_results["recommendations"].append(f"Compatibility check failed: {e}")
        
        return validation_results
    
    # ==========================================
    # SPECIFIC INTEGRATION FOR EXISTING FUNCTIONS
    # ==========================================
    
    def get_rag_integration_code(self) -> str:
        """
        Get code to integrate with existing RAG system
        """
        
        integration_code = '''
# ==========================================
# RAG INTEGRATION WITH ENHANCED CONVERSATION MEMORY
# Add this to your rag_answer function in service.py
# ==========================================

async def enhanced_rag_answer(question: str, thinking: bool = False, stream: bool = False, 
                            conversation_id: str = None, use_langgraph: bool = True):
    """Enhanced RAG with better conversation context management"""
    
    # Get enhanced conversation context
    if conversation_id:
        try:
            from app.memory.enhanced_conversation_service import get_conversation_context_for_rag
            enhanced_context = await get_conversation_context_for_rag(conversation_id, question)
            
            # Use enhanced context in your RAG pipeline
            if enhanced_context:
                # Your existing RAG logic here, but with enhanced_context instead of basic history
                print(f"[RAG_ENHANCED] Using enhanced context: {len(enhanced_context)} chars")
            
        except Exception as e:
            print(f"[RAG_ENHANCED] Fallback to original context: {e}")
            # Fallback to original get_conversation_history
            enhanced_context = get_conversation_history(conversation_id)
    
    # Continue with your existing RAG logic...
    # (rest of your rag_answer function)
'''
        
        return integration_code
    
    def get_multi_agent_integration_code(self) -> str:
        """
        Get code to integrate with multi-agent system
        """
        
        integration_code = '''
# ==========================================
# MULTI-AGENT INTEGRATION WITH ENHANCED CONVERSATION MEMORY
# Add this to your multi-agent system
# ==========================================

async def enhanced_multi_agent_stream_events(self, query: str, conversation_history: Optional[List[Dict]] = None):
    """Enhanced multi-agent with better conversation context"""
    
    # Get enhanced conversation context optimized for multi-agent use
    if self.conversation_id:
        try:
            from app.memory.enhanced_conversation_service import get_conversation_context_for_agents
            enhanced_context = await get_conversation_context_for_agents(self.conversation_id, query)
            
            # Use enhanced context instead of basic conversation_history
            if enhanced_context:
                # Parse enhanced context into structured format if needed
                context_lines = enhanced_context.split('\\n\\n')
                structured_history = []
                
                for line in context_lines:
                    if line.startswith('User:'):
                        structured_history.append({"role": "user", "content": line[5:].strip()})
                    elif line.startswith('Assistant:'):
                        structured_history.append({"role": "assistant", "content": line[10:].strip()})
                
                # Use structured_history in your multi-agent system
                conversation_history = structured_history
            
        except Exception as e:
            print(f"[MULTI_AGENT_ENHANCED] Fallback to original history: {e}")
            # Continue with original conversation_history
    
    # Continue with your existing multi-agent logic...
'''
        
        return integration_code


# ==========================================
# UTILITY FUNCTIONS FOR EASY INTEGRATION
# ==========================================

def quick_integration_check() -> bool:
    """
    Quick check if enhanced conversation system is ready to use
    Returns True if system is ready, False if fallback should be used
    """
    try:
        integrator = ConversationMemoryIntegrator()
        deps = integrator.step1_verify_dependencies()
        
        # Check minimum requirements
        minimum_deps = ["redis_available", "async_support"]
        return all(deps.get(dep, False) for dep in minimum_deps)
        
    except Exception as e:
        print(f"[INTEGRATION_CHECK] Failed: {e}")
        return False


def get_enhanced_conversation_history_safe(conversation_id: str, current_query: str = "") -> str:
    """
    Safe wrapper that automatically falls back to original implementation
    Use this as a direct replacement for get_conversation_history()
    """
    try:
        if quick_integration_check():
            from app.memory.enhanced_conversation_service import enhanced_conversation_service
            return enhanced_conversation_service.get_conversation_history(conversation_id)
        else:
            # Fallback to original
            from app.langchain.service import get_conversation_history
            return get_conversation_history(conversation_id)
            
    except Exception as e:
        print(f"[SAFE_CONV_HISTORY] Error: {e}")
        # Ultimate fallback
        try:
            from app.langchain.service import get_conversation_history
            return get_conversation_history(conversation_id)
        except:
            return ""


def store_conversation_message_safe(conversation_id: str, role: str, content: str):
    """
    Safe wrapper that automatically falls back to original implementation
    Use this as a direct replacement for store_conversation_message()
    """
    try:
        if quick_integration_check():
            from app.memory.enhanced_conversation_service import enhanced_conversation_service
            enhanced_conversation_service.store_conversation_message(conversation_id, role, content)
        else:
            # Fallback to original
            from app.langchain.service import store_conversation_message
            store_conversation_message(conversation_id, role, content)
            
    except Exception as e:
        print(f"[SAFE_CONV_STORE] Error: {e}")
        # Ultimate fallback
        try:
            from app.langchain.service import store_conversation_message
            store_conversation_message(conversation_id, role, content)
        except:
            pass  # Silent fail for storage


# Create global integrator instance
conversation_integrator = ConversationMemoryIntegrator()