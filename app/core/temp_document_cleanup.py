"""
Temporary document cleanup and TTL management system.
Handles automatic cleanup of expired temporary documents and Redis maintenance.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import schedule
import time
from threading import Thread

from app.core.redis_base import RedisCache
from app.services.temp_document_indexer import TempDocumentIndexer
from app.core.temp_document_manager import TempDocumentManager

logger = logging.getLogger(__name__)

class TempDocumentCleanupService:
    """
    Service for managing temporary document cleanup and TTL enforcement.
    Runs as a background service to automatically clean up expired documents.
    """
    
    def __init__(self):
        self.redis_cache = RedisCache(key_prefix="cleanup_")
        self.indexer = TempDocumentIndexer()
        self.manager = TempDocumentManager()
        self.is_running = False
        self.cleanup_thread = None
        
    async def cleanup_expired_documents(self) -> Dict[str, Any]:
        """
        Clean up all expired temporary documents across all conversations.
        
        Returns:
            Cleanup statistics
        """
        try:
            logger.info("Starting temporary document cleanup process")
            start_time = datetime.now()
            
            # Get cleanup statistics
            stats = {
                'cleanup_started': start_time.isoformat(),
                'documents_cleaned': 0,
                'conversations_affected': 0,
                'errors': [],
                'cleanup_duration_seconds': 0
            }
            
            # Scan for expired document metadata
            expired_docs = []
            conversations_with_expired = set()
            
            async for key in self.redis_cache.scan_iter(match="temp_doc_metadata:*"):
                try:
                    metadata = await self.redis_cache.get(key)
                    if metadata and 'expiry_timestamp' in metadata:
                        expiry_time = datetime.fromisoformat(metadata['expiry_timestamp'])
                        if datetime.now() > expiry_time:
                            expired_docs.append(metadata)
                            conversations_with_expired.add(metadata['conversation_id'])
                            
                except Exception as e:
                    logger.warning(f"Failed to check expiry for {key}: {e}")
                    stats['errors'].append(f"Check expiry {key}: {str(e)}")
            
            logger.info(f"Found {len(expired_docs)} expired documents across {len(conversations_with_expired)} conversations")
            
            # Clean up expired documents
            for doc_metadata in expired_docs:
                try:
                    temp_doc_id = doc_metadata['temp_doc_id']
                    success = await self.manager.delete_document(temp_doc_id)
                    
                    if success:
                        stats['documents_cleaned'] += 1
                        logger.debug(f"Cleaned up expired document: {temp_doc_id}")
                    else:
                        stats['errors'].append(f"Failed to delete {temp_doc_id}")
                        
                except Exception as e:
                    logger.error(f"Failed to cleanup document {doc_metadata.get('temp_doc_id', 'unknown')}: {e}")
                    stats['errors'].append(f"Cleanup {doc_metadata.get('temp_doc_id', 'unknown')}: {str(e)}")
            
            stats['conversations_affected'] = len(conversations_with_expired)
            
            # Clean up orphaned Redis keys
            await self._cleanup_orphaned_keys()
            
            # Update statistics
            end_time = datetime.now()
            stats['cleanup_completed'] = end_time.isoformat()
            stats['cleanup_duration_seconds'] = (end_time - start_time).total_seconds()
            
            # Store cleanup statistics in Redis (7 day TTL)
            await self.redis_cache.setex(
                f"cleanup_stats:{start_time.strftime('%Y%m%d_%H%M%S')}",
                7 * 24 * 3600,  # 7 days
                stats
            )
            
            logger.info(f"Cleanup completed: {stats['documents_cleaned']} documents cleaned, "
                       f"{len(stats['errors'])} errors, {stats['cleanup_duration_seconds']:.2f}s duration")
            
            return stats
            
        except Exception as e:
            logger.error(f"Cleanup process failed: {e}")
            return {
                'error': str(e),
                'cleanup_started': datetime.now().isoformat(),
                'documents_cleaned': 0,
                'conversations_affected': 0
            }
    
    async def cleanup_conversation_documents(self, conversation_id: str) -> int:
        """
        Clean up all temporary documents for a specific conversation.
        
        Args:
            conversation_id: Conversation ID to clean up
            
        Returns:
            Number of documents cleaned up
        """
        try:
            return await self.manager.cleanup_conversation_documents(conversation_id)
        except Exception as e:
            logger.error(f"Failed to cleanup conversation {conversation_id}: {e}")
            return 0
    
    async def extend_document_ttl(self, temp_doc_id: str, additional_hours: int) -> bool:
        """
        Extend the TTL of a temporary document.
        
        Args:
            temp_doc_id: Temporary document ID
            additional_hours: Hours to add to current TTL
            
        Returns:
            Success status
        """
        try:
            # Get current metadata
            metadata = await self.redis_cache.get(f"metadata:{temp_doc_id}")
            if not metadata:
                return False
            
            # Calculate new expiry time
            current_expiry = datetime.fromisoformat(metadata['expiry_timestamp'])
            new_expiry = current_expiry + timedelta(hours=additional_hours)
            
            # Update metadata
            metadata['expiry_timestamp'] = new_expiry.isoformat()
            
            # Update TTL in Redis
            collection_name = metadata['collection_name']
            new_ttl_seconds = int((new_expiry - datetime.now()).total_seconds())
            
            if new_ttl_seconds > 0:
                # Update index TTL
                await self.redis_cache.expire(f"llamaindex_temp:{collection_name}", new_ttl_seconds)
                
                # Update metadata TTL
                await self.redis_cache.setex(f"metadata:{temp_doc_id}", new_ttl_seconds, metadata)
                
                logger.info(f"Extended TTL for {temp_doc_id} by {additional_hours} hours")
                return True
            else:
                logger.warning(f"Cannot extend TTL for {temp_doc_id}: would expire in the past")
                return False
                
        except Exception as e:
            logger.error(f"Failed to extend TTL for {temp_doc_id}: {e}")
            return False
    
    async def get_cleanup_statistics(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get cleanup statistics for the past N days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of cleanup statistics
        """
        try:
            stats = []
            
            # Scan for cleanup statistics
            async for key in self.redis_cache.scan_iter(match="cleanup_stats:*"):
                try:
                    stat_data = await self.redis_cache.get(key)
                    if stat_data:
                        # Check if within time range
                        cleanup_time = datetime.fromisoformat(stat_data['cleanup_started'])
                        if cleanup_time > datetime.now() - timedelta(days=days):
                            stats.append(stat_data)
                except Exception as e:
                    logger.warning(f"Failed to read cleanup stats {key}: {e}")
            
            # Sort by cleanup time
            stats.sort(key=lambda x: x['cleanup_started'], reverse=True)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get cleanup statistics: {e}")
            return []
    
    async def get_document_health_status(self) -> Dict[str, Any]:
        """
        Get health status of temporary document system.
        
        Returns:
            System health information
        """
        try:
            # Count active documents
            total_docs = 0
            expired_docs = 0
            conversations_with_docs = set()
            
            async for key in self.redis_cache.scan_iter(match="temp_doc_metadata:*"):
                try:
                    metadata = await self.redis_cache.get(key)
                    if metadata:
                        total_docs += 1
                        conversations_with_docs.add(metadata['conversation_id'])
                        
                        # Check if expired
                        expiry_time = datetime.fromisoformat(metadata['expiry_timestamp'])
                        if datetime.now() > expiry_time:
                            expired_docs += 1
                            
                except Exception:
                    pass
            
            # Get recent cleanup stats
            recent_cleanups = await self.get_cleanup_statistics(days=1)
            last_cleanup = recent_cleanups[0] if recent_cleanups else None
            
            return {
                'timestamp': datetime.now().isoformat(),
                'total_active_documents': total_docs,
                'expired_documents_pending_cleanup': expired_docs,
                'conversations_with_documents': len(conversations_with_docs),
                'last_cleanup': last_cleanup,
                'cleanup_service_running': self.is_running,
                'system_status': 'healthy' if expired_docs < 10 else 'needs_cleanup'
            }
            
        except Exception as e:
            logger.error(f"Failed to get health status: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'system_status': 'error'
            }
    
    async def _cleanup_orphaned_keys(self):
        """Clean up orphaned Redis keys that no longer have associated documents."""
        try:
            orphaned_count = 0
            
            # Look for conversation document sets with no documents
            async for key in self.redis_cache.scan_iter(match="conv_temp_docs:*"):
                try:
                    doc_ids = await self.redis_cache.smembers(key)
                    if doc_ids:
                        # Check if any of the referenced documents still exist
                        existing_docs = 0
                        for doc_id in doc_ids:
                            if await self.redis_cache.exists(f"metadata:{doc_id}"):
                                existing_docs += 1
                        
                        # If no documents exist, clean up the conversation set
                        if existing_docs == 0:
                            await self.redis_cache.delete(key)
                            orphaned_count += 1
                            
                except Exception as e:
                    logger.warning(f"Failed to check orphaned key {key}: {e}")
            
            if orphaned_count > 0:
                logger.info(f"Cleaned up {orphaned_count} orphaned conversation document sets")
                
        except Exception as e:
            logger.warning(f"Failed to cleanup orphaned keys: {e}")
    
    def start_background_cleanup(self, interval_hours: int = 1):
        """
        Start background cleanup service.
        
        Args:
            interval_hours: Cleanup interval in hours
        """
        if self.is_running:
            logger.warning("Cleanup service is already running")
            return
        
        self.is_running = True
        
        def cleanup_scheduler():
            """Run cleanup on schedule"""
            schedule.every(interval_hours).hours.do(self._run_cleanup_sync)
            
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        self.cleanup_thread = Thread(target=cleanup_scheduler, daemon=True)
        self.cleanup_thread.start()
        
        logger.info(f"Started background cleanup service with {interval_hours}h interval")
    
    def stop_background_cleanup(self):
        """Stop background cleanup service."""
        self.is_running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        
        logger.info("Stopped background cleanup service")
    
    def _run_cleanup_sync(self):
        """Synchronous wrapper for async cleanup method."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.cleanup_expired_documents())
            loop.close()
        except Exception as e:
            logger.error(f"Background cleanup failed: {e}")

# Global cleanup service instance
_cleanup_service = None

def get_cleanup_service() -> TempDocumentCleanupService:
    """Get global cleanup service instance."""
    global _cleanup_service
    if _cleanup_service is None:
        _cleanup_service = TempDocumentCleanupService()
    return _cleanup_service

async def manual_cleanup() -> Dict[str, Any]:
    """Manually trigger cleanup and return results."""
    service = get_cleanup_service()
    return await service.cleanup_expired_documents()

async def start_cleanup_service(interval_hours: int = 2):
    """Start the background cleanup service."""
    service = get_cleanup_service()
    service.start_background_cleanup(interval_hours)

def stop_cleanup_service():
    """Stop the background cleanup service."""
    global _cleanup_service
    if _cleanup_service:
        _cleanup_service.stop_background_cleanup()