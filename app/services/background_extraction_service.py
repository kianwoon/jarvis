"""
Background Extraction Service
Handles extraction tasks asynchronously to prevent timeout issues.
"""

import asyncio
import json
import logging
import time
from typing import List, Optional, Dict, Any
from datetime import datetime

from app.core.redis_client import get_redis_client
from app.services.notebook_rag_service import NotebookRAGService

logger = logging.getLogger(__name__)

class BackgroundExtractionService:
    """Service for handling extraction tasks in the background"""
    
    def __init__(self):
        self.redis_client = None
        self.extraction_service = NotebookRAGService()
        self.queue_key = "background_extraction_queue"
        self.result_key_prefix = "extraction_result"
        
    async def _get_redis_client(self):
        """Get Redis client instance"""
        if not self.redis_client:
            self.redis_client = await get_redis_client()
        return self.redis_client
    
    async def queue_extraction_task(
        self, 
        task_id: str, 
        notebook_id: str, 
        sources: List[Any],
        query: str,
        conversation_id: str = None
    ) -> str:
        """
        Queue an extraction task for background processing.
        
        Args:
            task_id: Unique identifier for this extraction task
            notebook_id: Notebook ID
            sources: List of sources to extract from
            query: Original query for context
            conversation_id: Optional conversation ID
            
        Returns:
            task_id for tracking the extraction progress
        """
        try:
            redis_client = await self._get_redis_client()
            
            # Prepare task data
            task_data = {
                "task_id": task_id,
                "notebook_id": notebook_id,
                "query": query,
                "conversation_id": conversation_id,
                "sources_count": len(sources),
                "queued_at": datetime.now().isoformat(),
                "status": "queued",
                # Store simplified source data to avoid memory issues
                "sources": [
                    {
                        "document_id": getattr(source, 'document_id', ''),
                        "content": getattr(source, 'content', '')[:2000],  # Limit content size
                        "source_type": getattr(source, 'source_type', 'document'),
                        "score": getattr(source, 'score', 0.0)
                    }
                    for source in sources
                ]
            }
            
            # Add task to queue
            await redis_client.lpush(self.queue_key, json.dumps(task_data))
            
            # Set initial status
            status_key = f"{self.result_key_prefix}:{task_id}:status"
            await redis_client.setex(status_key, 3600, json.dumps({
                "status": "queued",
                "queued_at": task_data["queued_at"],
                "sources_count": len(sources)
            }))
            
            logger.info(f"[BACKGROUND_EXTRACTION] Queued extraction task {task_id} with {len(sources)} sources")
            return task_id
            
        except Exception as e:
            logger.error(f"[BACKGROUND_EXTRACTION] Failed to queue extraction task: {str(e)}")
            raise
    
    async def get_extraction_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of an extraction task"""
        try:
            redis_client = await self._get_redis_client()
            status_key = f"{self.result_key_prefix}:{task_id}:status"
            
            status_data = await redis_client.get(status_key)
            if not status_data:
                return {"status": "not_found", "error": "Task not found or expired"}
            
            return json.loads(status_data)
            
        except Exception as e:
            logger.error(f"[BACKGROUND_EXTRACTION] Failed to get extraction status: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def get_extraction_results(self, task_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get extraction results if available"""
        try:
            redis_client = await self._get_redis_client()
            result_key = f"{self.result_key_prefix}:{task_id}:results"
            
            results_data = await redis_client.get(result_key)
            if not results_data:
                return None
            
            return json.loads(results_data)
            
        except Exception as e:
            logger.error(f"[BACKGROUND_EXTRACTION] Failed to get extraction results: {str(e)}")
            return None
    
    async def process_extraction_queue(self, max_tasks: int = 1):
        """
        Process extraction tasks from the queue.
        This should be called by a background worker.
        """
        try:
            redis_client = await self._get_redis_client()
            
            # Process tasks one by one to avoid overwhelming the system
            for _ in range(max_tasks):
                # Get next task from queue
                task_data = await redis_client.brpop(self.queue_key, timeout=1)
                if not task_data:
                    break  # No more tasks
                
                task_json = task_data[1]  # Redis returns (key, value) tuple
                task = json.loads(task_json)
                
                await self._process_single_extraction_task(task)
                
        except Exception as e:
            logger.error(f"[BACKGROUND_EXTRACTION] Error processing extraction queue: {str(e)}")
    
    async def _process_single_extraction_task(self, task: Dict[str, Any]):
        """Process a single extraction task"""
        task_id = task["task_id"]
        
        try:
            redis_client = await self._get_redis_client()
            status_key = f"{self.result_key_prefix}:{task_id}:status"
            result_key = f"{self.result_key_prefix}:{task_id}:results"
            
            # Update status to processing
            await redis_client.setex(status_key, 3600, json.dumps({
                "status": "processing",
                "started_at": datetime.now().isoformat(),
                "sources_count": task["sources_count"]
            }))
            
            logger.info(f"[BACKGROUND_EXTRACTION] Processing task {task_id} with {task['sources_count']} sources")
            
            # Convert task sources back to source objects for processing
            # This is a simplified approach - in production you might want to
            # reconstruct full source objects from the database
            sources = []
            for source_data in task["sources"]:
                # Create mock source object with essential data
                source = type('MockSource', (), source_data)
                sources.append(source)
            
            # Perform the actual extraction
            start_time = time.time()
            extracted_projects = await self.extraction_service.extract_project_data(sources)
            processing_time = time.time() - start_time
            
            # Convert results to JSON-serializable format
            results = []
            if extracted_projects:
                results = [
                    {
                        "name": project.name,
                        "company": project.company,
                        "year": project.year,
                        "description": project.description,
                        "confidence_score": project.confidence_score
                    }
                    for project in extracted_projects
                ]
            
            # Store results
            await redis_client.setex(result_key, 3600, json.dumps(results))
            
            # Update final status
            await redis_client.setex(status_key, 3600, json.dumps({
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "processing_time": processing_time,
                "results_count": len(results),
                "sources_count": task["sources_count"]
            }))
            
            logger.info(f"[BACKGROUND_EXTRACTION] Completed task {task_id}: {len(results)} projects in {processing_time:.1f}s")
            
        except Exception as e:
            logger.error(f"[BACKGROUND_EXTRACTION] Failed to process task {task_id}: {str(e)}")
            
            # Update status to failed
            try:
                redis_client = await self._get_redis_client()
                status_key = f"{self.result_key_prefix}:{task_id}:status"
                await redis_client.setex(status_key, 3600, json.dumps({
                    "status": "failed",
                    "failed_at": datetime.now().isoformat(),
                    "error": str(e)
                }))
            except:
                pass  # Don't let status update failures crash the worker

# Global instance
background_extraction_service = BackgroundExtractionService()