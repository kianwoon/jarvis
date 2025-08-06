"""
Continuity Manager for Context-Limit-Transcending System
Manages state continuity and seamless execution across task chunks
"""

import json
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime
import uuid
import logging

from app.agents.task_decomposer import TaskChunk
from app.core.llm_settings_cache import get_llm_settings
from app.llm.ollama import OllamaLLM
from app.llm.base import LLMConfig
import os

logger = logging.getLogger(__name__)


class ChunkResult:
    """Represents the result of executing a single chunk"""
    
    def __init__(self, chunk: TaskChunk):
        self.chunk = chunk
        self.generated_items: List[str] = []
        self.execution_time: float = 0.0
        self.status: str = "pending"  # pending, executing, completed, failed
        self.error_message: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
    
    def mark_started(self):
        """Mark chunk execution as started"""
        self.status = "executing"
        self.start_time = datetime.now()
    
    def mark_completed(self, items: List[str], metadata: Dict[str, Any] = None):
        """Mark chunk execution as completed"""
        self.status = "completed"
        self.end_time = datetime.now()
        self.generated_items = items
        if metadata:
            self.metadata.update(metadata)
        
        if self.start_time:
            self.execution_time = (self.end_time - self.start_time).total_seconds()
    
    def mark_failed(self, error: str):
        """Mark chunk execution as failed"""
        self.status = "failed"
        self.end_time = datetime.now()
        self.error_message = error
        
        if self.start_time:
            self.execution_time = (self.end_time - self.start_time).total_seconds()


class ContinuityManager:
    """
    Manages the execution of chunked tasks while maintaining state continuity
    and providing progress tracking with recovery capabilities
    """
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.chunk_results: Dict[str, ChunkResult] = {}
        self.progress_tracker: Dict[str, Any] = {}
        self.llm_settings = get_llm_settings()
        self.continuation_overlap = 3  # Number of previous items to include for context
        
        logger.info(f"[CONTINUITY] Initialized session {self.session_id}")
    
    async def execute_chunked_task(
        self, 
        chunks: List[TaskChunk],
        agent_name: str = "continuation_agent"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute a list of task chunks while maintaining continuity
        
        Args:
            chunks: List of TaskChunk objects to execute
            agent_name: Name of the agent to use for execution
            
        Yields:
            Progress events and results
        """
        logger.info(f"[CONTINUITY] Starting execution of {len(chunks)} chunks")
        
        # Initialize progress tracking
        self.progress_tracker = {
            "total_chunks": len(chunks),
            "completed_chunks": 0,
            "failed_chunks": 0,
            "total_items_generated": 0,
            "estimated_completion_time": None,
            "start_time": datetime.now().isoformat()
        }
        
        # Initialize chunk results
        for chunk in chunks:
            self.chunk_results[chunk.chunk_id] = ChunkResult(chunk)
        
        yield {
            "type": "task_started",
            "session_id": self.session_id,
            "total_chunks": len(chunks),
            "estimated_duration": len(chunks) * 30  # rough estimate in seconds
        }
        
        # Execute chunks sequentially to maintain continuity
        all_generated_items = []
        
        for i, chunk in enumerate(chunks):
            try:
                # Build continuation context from previous results
                continuation_context = self._build_continuation_context(chunk, all_generated_items)
                
                # Update chunk with continuation context
                chunk.continuation_context = continuation_context
                
                # Execute chunk
                chunk_result = self.chunk_results[chunk.chunk_id]
                chunk_result.mark_started()
                
                yield {
                    "type": "chunk_started",
                    "chunk_number": chunk.chunk_number,
                    "total_chunks": chunk.total_chunks,
                    "chunk_id": chunk.chunk_id,
                    "items_range": f"{chunk.start_index}-{chunk.end_index}"
                }
                
                # Execute the chunk using specialized agent
                async for event in self._execute_single_chunk(chunk, agent_name):
                    if event.get("type") == "chunk_completed":
                        # Extract generated items from the response
                        generated_items = self._extract_items_from_response(
                            event.get("content", ""), 
                            chunk
                        )
                        
                        # Mark chunk as completed
                        chunk_result.mark_completed(
                            generated_items,
                            {"response_length": len(event.get("content", ""))}
                        )
                        
                        # Add to overall results
                        all_generated_items.extend(generated_items)
                        
                        # Update progress
                        self.progress_tracker["completed_chunks"] += 1
                        self.progress_tracker["total_items_generated"] = len(all_generated_items)
                        
                        # Calculate estimated completion time
                        if i > 0:  # After first chunk
                            avg_time_per_chunk = sum(
                                result.execution_time for result in self.chunk_results.values() 
                                if result.status == "completed"
                            ) / self.progress_tracker["completed_chunks"]
                            
                            remaining_chunks = len(chunks) - (i + 1)
                            estimated_remaining_seconds = remaining_chunks * avg_time_per_chunk
                            
                            self.progress_tracker["estimated_completion_time"] = estimated_remaining_seconds
                        
                        yield {
                            "type": "chunk_completed",
                            "chunk_number": chunk.chunk_number,
                            "chunk_id": chunk.chunk_id,
                            "items_generated": len(generated_items),
                            "cumulative_items": len(all_generated_items),
                            "execution_time": chunk_result.execution_time,
                            "progress_percentage": ((i + 1) / len(chunks)) * 100,
                            "estimated_remaining_seconds": self.progress_tracker.get("estimated_completion_time"),
                            "partial_results": generated_items[-3:] if generated_items else []  # Show last 3 items
                        }
                        
                    elif event.get("type") == "chunk_error":
                        chunk_result.mark_failed(event.get("error", "Unknown error"))
                        self.progress_tracker["failed_chunks"] += 1
                        
                        yield {
                            "type": "chunk_failed",
                            "chunk_number": chunk.chunk_number,
                            "chunk_id": chunk.chunk_id,
                            "error": event.get("error")
                        }
                        
                        # Continue with next chunk even if one fails
                        continue
                    
                    # Forward other events
                    yield event
                
            except Exception as e:
                logger.error(f"[CONTINUITY] Chunk {chunk.chunk_id} failed: {e}")
                chunk_result = self.chunk_results[chunk.chunk_id]
                chunk_result.mark_failed(str(e))
                self.progress_tracker["failed_chunks"] += 1
                
                yield {
                    "type": "chunk_failed",
                    "chunk_number": chunk.chunk_number,
                    "chunk_id": chunk.chunk_id,
                    "error": str(e)
                }
        
        # Final completion event
        self.progress_tracker["end_time"] = datetime.now().isoformat()
        total_time = (
            datetime.fromisoformat(self.progress_tracker["end_time"]) - 
            datetime.fromisoformat(self.progress_tracker["start_time"])
        ).total_seconds()
        
        yield {
            "type": "task_completed",
            "session_id": self.session_id,
            "total_items_generated": len(all_generated_items),
            "completed_chunks": self.progress_tracker["completed_chunks"],
            "failed_chunks": self.progress_tracker["failed_chunks"],
            "total_execution_time": total_time,
            "final_results": all_generated_items,
            "summary": self._generate_execution_summary()
        }
    
    def _build_continuation_context(self, chunk: TaskChunk, previous_items: List[str]) -> str:
        """Build context for continuation from previous chunks"""
        if not previous_items or chunk.chunk_number == 1:
            # First chunk - no continuation context needed
            return f"Generate items {chunk.start_index} to {chunk.end_index}:"
        
        # Get last few items for context
        context_items = previous_items[-self.continuation_overlap:] if len(previous_items) >= self.continuation_overlap else previous_items
        
        # Find the last item number to determine continuation point
        last_item_number = chunk.start_index - 1
        
        context = f"""Previous items for context (maintain the same style and pattern):
        
{chr(10).join(f"{last_item_number - len(context_items) + i + 1}. {item.strip()}" for i, item in enumerate(context_items))}

Now continue generating items {chunk.start_index} to {chunk.end_index} following the same pattern:"""
        
        return context
    
    async def _execute_single_chunk(
        self, 
        chunk: TaskChunk, 
        agent_name: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute a single chunk using the specified agent"""
        
        # Build the prompt for this chunk
        prompt = self._build_chunk_prompt(chunk)
        
        try:
            # Use the existing LLM infrastructure
            model_config = self.llm_settings.get("thinking_mode", {})
            
            config = LLMConfig(
                model_name=model_config.get("model", "qwen3:30b-a3b"),
                temperature=0.7,
                top_p=model_config.get("top_p", 0.9),
                max_tokens=model_config.get("max_tokens", 3000)  # Use configured value or fallback to 3000
            )
            
            ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            llm = OllamaLLM(config, base_url=ollama_url)
            
            response_text = ""
            
            # Use timeout for each chunk
            async with asyncio.timeout(120):  # 2 minute timeout per chunk
                async for response_chunk in llm.generate_stream(prompt):
                    response_text += response_chunk.text
            
            yield {
                "type": "chunk_completed",
                "content": response_text,
                "agent": agent_name,
                "chunk_id": chunk.chunk_id
            }
            
        except asyncio.TimeoutError:
            yield {
                "type": "chunk_error",
                "error": f"Chunk {chunk.chunk_id} timed out after 120 seconds",
                "chunk_id": chunk.chunk_id
            }
        except Exception as e:
            yield {
                "type": "chunk_error",
                "error": str(e),
                "chunk_id": chunk.chunk_id
            }
    
    def _build_chunk_prompt(self, chunk: TaskChunk) -> str:
        """Build the prompt for executing a specific chunk"""
        
        task_analysis = chunk.metadata.get("task_analysis", {})
        requires_numbering = task_analysis.get("requires_numbering", True)
        content_type = task_analysis.get("content_type", "unknown")
        
        base_prompt = f"""You are a specialized continuation agent for large content generation tasks.

TASK: {chunk.task_description}

CHUNK DETAILS:
- Generate items {chunk.start_index} to {chunk.end_index} ({chunk.chunk_size} items)
- Chunk {chunk.chunk_number} of {chunk.total_chunks}
- Content type: {content_type}

{chunk.continuation_context}

REQUIREMENTS:
- {"Maintain consistent numbering starting from " + str(chunk.start_index) if requires_numbering else "Generate unnumbered items"}
- Keep the same style, format, and quality as any previous items
- Generate exactly {chunk.chunk_size} items
- Ensure each item is complete and well-formed

Generate the items now:"""

        return base_prompt
    
    def _extract_items_from_response(self, response: str, chunk: TaskChunk) -> List[str]:
        """Extract individual items from the LLM response"""
        if not response:
            return []
        
        lines = response.strip().split('\n')
        items = []
        
        # Try to extract numbered items first
        import re
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Match numbered items (1. item, 1) item, etc.)
            numbered_match = re.match(r'^\d+[\.\)]\s*(.+)', line)
            if numbered_match:
                items.append(numbered_match.group(1).strip())
            elif line and not line.startswith('#') and len(line) > 5:
                # Non-numbered item (if it looks substantial)
                items.append(line)
        
        # If we didn't extract enough items, try a different approach
        if len(items) < chunk.chunk_size // 2:  # If we got less than half expected
            # Split by double newlines and filter
            potential_items = [item.strip() for item in response.split('\n\n') if item.strip()]
            if potential_items:
                items = potential_items
        
        logger.info(f"[CONTINUITY] Extracted {len(items)} items from chunk {chunk.chunk_id}")
        return items[:chunk.chunk_size]  # Ensure we don't exceed expected count
    
    def _generate_execution_summary(self) -> Dict[str, Any]:
        """Generate a summary of the execution"""
        completed_results = [r for r in self.chunk_results.values() if r.status == "completed"]
        failed_results = [r for r in self.chunk_results.values() if r.status == "failed"]
        
        summary = {
            "total_chunks_executed": len(self.chunk_results),
            "successful_chunks": len(completed_results),
            "failed_chunks": len(failed_results),
            "success_rate": len(completed_results) / len(self.chunk_results) if self.chunk_results else 0,
            "average_execution_time": sum(r.execution_time for r in completed_results) / len(completed_results) if completed_results else 0,
            "total_items_generated": sum(len(r.generated_items) for r in completed_results),
            "errors": [r.error_message for r in failed_results if r.error_message]
        }
        
        return summary
    
    def get_progress_status(self) -> Dict[str, Any]:
        """Get current progress status"""
        return {
            "session_id": self.session_id,
            "progress": self.progress_tracker,
            "chunk_statuses": {
                chunk_id: {
                    "status": result.status,
                    "items_generated": len(result.generated_items),
                    "execution_time": result.execution_time,
                    "error": result.error_message
                }
                for chunk_id, result in self.chunk_results.items()
            }
        }
    
    def get_all_results(self) -> List[str]:
        """Get all generated items in order"""
        all_items = []
        
        # Sort chunks by chunk number to maintain order
        sorted_results = sorted(
            self.chunk_results.values(),
            key=lambda r: r.chunk.chunk_number
        )
        
        for result in sorted_results:
            if result.status == "completed":
                all_items.extend(result.generated_items)
        
        return all_items