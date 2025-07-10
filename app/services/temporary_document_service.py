"""
High-level temporary document service that orchestrates document processing 
and integrates existing temporary document management with in-memory RAG.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path

from app.core.temp_document_manager import TempDocumentManager
from app.langchain.in_memory_rag_service import get_in_memory_rag_service, InMemoryRAGService
from app.core.in_memory_rag_settings import get_in_memory_rag_settings
from app.document_handlers.base import ExtractedChunk

logger = logging.getLogger(__name__)

class TemporaryDocumentService:
    """
    High-level service for managing temporary documents with in-memory RAG capabilities.
    Orchestrates between existing temp document system and new in-memory RAG service.
    """
    
    def __init__(self):
        self.temp_doc_manager = TempDocumentManager()
        self.config = get_in_memory_rag_settings()
    
    async def upload_and_process_document(
        self,
        file_content: bytes,
        filename: str,
        conversation_id: str,
        ttl_hours: int = 2,
        auto_include: bool = True,
        enable_in_memory_rag: bool = True
    ) -> Dict[str, Any]:
        """
        Upload and process a document for both existing temp system and in-memory RAG.
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            conversation_id: Associated conversation ID
            ttl_hours: Time-to-live in hours
            auto_include: Whether to automatically include in chat context
            enable_in_memory_rag: Whether to also add to in-memory RAG service
            
        Returns:
            Processing result with temp_doc_id and metadata
        """
        try:
            logger.info(f"Processing document {filename} for conversation {conversation_id}")
            
            # Use existing temp document manager for initial processing
            result = await self.temp_doc_manager.upload_and_process_document(
                file_content=file_content,
                filename=filename,
                conversation_id=conversation_id,
                ttl_hours=ttl_hours,
                auto_include=auto_include
            )
            
            if not result.get('success'):
                return result
            
            # If in-memory RAG is enabled, also add to in-memory service
            if enable_in_memory_rag and auto_include:
                try:
                    in_memory_success = await self._add_to_in_memory_rag(
                        conversation_id, 
                        result['temp_doc_id'], 
                        filename
                    )
                    
                    # Add in-memory RAG status to result
                    result['metadata']['in_memory_rag_enabled'] = in_memory_success
                    
                    if in_memory_success:
                        logger.info(f"Successfully added {filename} to in-memory RAG")
                    else:
                        logger.warning(f"Failed to add {filename} to in-memory RAG")
                        
                except Exception as e:
                    logger.error(f"In-memory RAG processing failed for {filename}: {e}")
                    result['metadata']['in_memory_rag_enabled'] = False
                    result['metadata']['in_memory_rag_error'] = str(e)
            else:
                result['metadata']['in_memory_rag_enabled'] = False
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process document {filename}: {e}")
            return {
                'success': False,
                'error': str(e),
                'temp_doc_id': None
            }
    
    async def _add_to_in_memory_rag(
        self, 
        conversation_id: str, 
        temp_doc_id: str, 
        filename: str
    ) -> bool:
        """Add document to in-memory RAG service."""
        try:
            # Get in-memory RAG service for conversation
            rag_service = await get_in_memory_rag_service(conversation_id)
            
            # Get document chunks from existing temp document system
            chunks = await self._get_document_chunks(temp_doc_id)
            
            if not chunks:
                logger.warning(f"No chunks found for temp document {temp_doc_id}")
                return False
            
            # Convert chunks to format expected by in-memory RAG
            documents = []
            for chunk in chunks:
                doc = {
                    'content': chunk.content,
                    'metadata': {
                        **chunk.metadata,
                        'temp_doc_id': temp_doc_id,
                        'conversation_id': conversation_id,
                        'filename': filename,
                        'added_to_in_memory_rag': datetime.now().isoformat()
                    }
                }
                documents.append(doc)
            
            # Add to in-memory RAG service
            success = await rag_service.add_documents(documents)
            
            if success:
                logger.info(f"Added {len(documents)} chunks to in-memory RAG for {filename}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to add document to in-memory RAG: {e}")
            return False
    
    async def _get_document_chunks(self, temp_doc_id: str) -> List[ExtractedChunk]:
        """Get document chunks from existing temp document system."""
        try:
            # Query the temp document to get its content
            # This is a simplified approach - in practice, you might want to 
            # store chunks separately or extract them differently
            
            from app.services.temp_document_indexer import TempDocumentIndexer
            indexer = TempDocumentIndexer()
            
            # Get the LlamaIndex document
            temp_index = indexer.get_temp_index(temp_doc_id)
            
            if not temp_index:
                logger.warning(f"No temp index found for {temp_doc_id}")
                return []
            
            # Extract chunks from the index
            # This is a workaround since LlamaIndex doesn't expose documents directly
            # In practice, you might want to store chunks during initial processing
            
            # For now, we'll return empty list and let the existing system handle it
            # The real implementation would extract document content and re-chunk it
            logger.warning("Document chunk extraction not fully implemented - using fallback")
            return []
            
        except Exception as e:
            logger.error(f"Failed to get document chunks for {temp_doc_id}: {e}")
            return []
    
    async def query_documents(
        self,
        conversation_id: str,
        query: str,
        use_in_memory_rag: bool = True,
        fallback_to_temp_docs: bool = True,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Query documents using in-memory RAG service with fallback to existing system.
        
        Args:
            conversation_id: Conversation ID
            query: Query string
            use_in_memory_rag: Whether to use in-memory RAG first
            fallback_to_temp_docs: Whether to fallback to existing temp doc system
            top_k: Number of top results to return
            
        Returns:
            Query results with sources and metadata
        """
        try:
            results = {
                'query': query,
                'sources': [],
                'total_chunks': 0,
                'processing_time_ms': 0,
                'source_type': None,
                'conversation_id': conversation_id
            }
            
            # Try in-memory RAG first if enabled
            if use_in_memory_rag:
                try:
                    rag_service = await get_in_memory_rag_service(conversation_id)
                    rag_response = await rag_service.query(query, top_k)
                    
                    if rag_response.results:
                        # Convert in-memory RAG results to standard format
                        sources = []
                        for result in rag_response.results:
                            sources.append({
                                'content': result.chunk.content,
                                'metadata': result.chunk.metadata,
                                'score': result.score,
                                'rank': result.rank,
                                'chunk_id': result.chunk.chunk_id
                            })
                        
                        results.update({
                            'sources': sources,
                            'total_chunks': rag_response.total_chunks,
                            'processing_time_ms': rag_response.processing_time_ms,
                            'source_type': 'in_memory_rag',
                            'rag_stats': rag_service.get_stats()
                        })
                        
                        logger.info(f"In-memory RAG returned {len(sources)} results for query")
                        return results
                    
                except Exception as e:
                    logger.warning(f"In-memory RAG query failed: {e}")
            
            # Fallback to existing temp document system
            if fallback_to_temp_docs:
                try:
                    temp_results = await self.temp_doc_manager.query_conversation_documents(
                        conversation_id=conversation_id,
                        query=query,
                        include_all=False,
                        top_k=top_k
                    )
                    
                    if temp_results:
                        results.update({
                            'sources': temp_results,
                            'total_chunks': len(temp_results),
                            'source_type': 'temp_documents',
                            'processing_time_ms': 0  # Not tracked in existing system
                        })
                        
                        logger.info(f"Temp documents returned {len(temp_results)} results for query")
                        return results
                    
                except Exception as e:
                    logger.warning(f"Temp documents query failed: {e}")
            
            # No results from either system
            results['source_type'] = 'none'
            logger.info(f"No results found for query in conversation {conversation_id}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to query documents: {e}")
            return {
                'query': query,
                'sources': [],
                'error': str(e),
                'conversation_id': conversation_id
            }
    
    async def get_conversation_documents(
        self, 
        conversation_id: str,
        include_in_memory_stats: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get all temporary documents for a conversation with optional in-memory RAG stats.
        
        Args:
            conversation_id: Conversation ID
            include_in_memory_stats: Whether to include in-memory RAG statistics
            
        Returns:
            List of document metadata with status information
        """
        try:
            # Get documents from existing temp document manager
            documents = await self.temp_doc_manager.get_conversation_documents(conversation_id)
            
            # Add in-memory RAG stats if requested
            if include_in_memory_stats:
                try:
                    rag_service = await get_in_memory_rag_service(conversation_id)
                    rag_stats = rag_service.get_stats()
                    
                    # Add in-memory RAG info to each document
                    for doc in documents:
                        doc['in_memory_rag_stats'] = {
                            'total_chunks': rag_stats['document_count'],
                            'vector_count': rag_stats['vector_count'],
                            'last_access': rag_stats['last_access']
                        }
                        
                except Exception as e:
                    logger.warning(f"Failed to get in-memory RAG stats: {e}")
                    for doc in documents:
                        doc['in_memory_rag_stats'] = {'error': str(e)}
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to get conversation documents: {e}")
            return []
    
    async def update_document_preferences(
        self,
        temp_doc_id: str,
        preferences: Dict[str, Any],
        conversation_id: Optional[str] = None
    ) -> bool:
        """
        Update document preferences including in-memory RAG inclusion.
        
        Args:
            temp_doc_id: Temporary document ID
            preferences: User preferences
            conversation_id: Conversation ID (required if changing in-memory RAG status)
            
        Returns:
            Success status
        """
        try:
            # Update existing temp document preferences
            success = await self.temp_doc_manager.update_document_preferences(
                temp_doc_id, preferences
            )
            
            # Handle in-memory RAG preference changes
            if 'enable_in_memory_rag' in preferences and conversation_id:
                try:
                    if preferences['enable_in_memory_rag']:
                        # Add to in-memory RAG if not already there
                        await self._add_to_in_memory_rag(
                            conversation_id, 
                            temp_doc_id, 
                            preferences.get('filename', 'unknown')
                        )
                    else:
                        # Remove from in-memory RAG (would need implementation)
                        logger.info(f"In-memory RAG removal not yet implemented for {temp_doc_id}")
                        
                except Exception as e:
                    logger.warning(f"Failed to update in-memory RAG preferences: {e}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update document preferences: {e}")
            return False
    
    async def delete_document(
        self, 
        temp_doc_id: str, 
        conversation_id: Optional[str] = None
    ) -> bool:
        """
        Delete a temporary document from both existing system and in-memory RAG.
        
        Args:
            temp_doc_id: Temporary document ID
            conversation_id: Conversation ID (optional, for in-memory RAG cleanup)
            
        Returns:
            Success status
        """
        try:
            # Delete from existing temp document system
            success = await self.temp_doc_manager.delete_document(temp_doc_id)
            
            # Clean up from in-memory RAG if conversation_id provided
            if conversation_id:
                try:
                    rag_service = await get_in_memory_rag_service(conversation_id)
                    # Note: This would require implementing selective document removal
                    # from the in-memory RAG service, which is complex with FAISS
                    # For now, we'll just log this limitation
                    logger.info(f"In-memory RAG selective removal not implemented for {temp_doc_id}")
                    
                except Exception as e:
                    logger.warning(f"Failed to clean up in-memory RAG for {temp_doc_id}: {e}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete document {temp_doc_id}: {e}")
            return False
    
    async def cleanup_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        Clean up all temporary documents and in-memory RAG for a conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Cleanup results
        """
        try:
            # Clean up existing temp documents
            temp_docs_cleaned = await self.temp_doc_manager.cleanup_conversation_documents(
                conversation_id
            )
            
            # Clean up in-memory RAG
            in_memory_rag_cleaned = False
            try:
                rag_service = await get_in_memory_rag_service(conversation_id)
                await rag_service.clear()
                in_memory_rag_cleaned = True
                
                # Remove from global registry
                from app.langchain.in_memory_rag_service import _rag_services
                if conversation_id in _rag_services:
                    del _rag_services[conversation_id]
                    
            except Exception as e:
                logger.warning(f"Failed to clean up in-memory RAG: {e}")
            
            return {
                'conversation_id': conversation_id,
                'temp_documents_cleaned': temp_docs_cleaned,
                'in_memory_rag_cleaned': in_memory_rag_cleaned,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to cleanup conversation {conversation_id}: {e}")
            return {
                'conversation_id': conversation_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_service_stats(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for temporary document services.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Service statistics
        """
        try:
            stats = {
                'conversation_id': conversation_id,
                'timestamp': datetime.now().isoformat()
            }
            
            # Get temp document stats
            try:
                temp_docs = await self.temp_doc_manager.get_conversation_documents(conversation_id)
                stats['temp_documents'] = {
                    'count': len(temp_docs),
                    'active_count': len([doc for doc in temp_docs if doc.get('is_included', False)]),
                    'documents': temp_docs
                }
            except Exception as e:
                stats['temp_documents'] = {'error': str(e)}
            
            # Get in-memory RAG stats
            try:
                rag_service = await get_in_memory_rag_service(conversation_id)
                stats['in_memory_rag'] = rag_service.get_stats()
            except Exception as e:
                stats['in_memory_rag'] = {'error': str(e)}
            
            # Get configuration
            stats['config'] = {
                'vector_store_type': self.config.vector_store_type.value,
                'embedding_model': self.config.embedding_model_name,
                'similarity_threshold': self.config.similarity_threshold,
                'max_documents': self.config.max_documents_per_conversation,
                'default_ttl_hours': self.config.default_ttl_hours
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get service stats: {e}")
            return {
                'conversation_id': conversation_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Global service instance
_service_instance: Optional[TemporaryDocumentService] = None

def get_temporary_document_service() -> TemporaryDocumentService:
    """Get the global temporary document service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = TemporaryDocumentService()
    return _service_instance