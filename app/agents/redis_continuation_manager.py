"""
Redis-based Continuity Manager for Context-Limit-Transcending System
Uses Redis for fast session state management and progress tracking
"""

import json
import asyncio
import redis.asyncio as redis
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime, timedelta
import uuid
import logging

from app.agents.task_decomposer import TaskChunk
from app.core.llm_settings_cache import get_llm_settings
from app.llm.ollama import OllamaLLM
from app.llm.base import LLMConfig
import os

logger = logging.getLogger(__name__)


class RedisChunkResult:
    """Redis-serializable chunk result"""
    
    def __init__(self, chunk: TaskChunk):
        self.chunk_id = chunk.chunk_id
        self.chunk_number = chunk.chunk_number
        self.generated_items: List[str] = []
        self.execution_time: float = 0.0
        self.status: str = "pending"
        self.error_message: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
        self.start_time: Optional[str] = None
        self.end_time: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to Redis-storable dictionary"""
        return {
            "chunk_id": self.chunk_id,
            "chunk_number": self.chunk_number,
            "generated_items": self.generated_items,
            "execution_time": self.execution_time,
            "status": self.status,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "start_time": self.start_time,
            "end_time": self.end_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], chunk: TaskChunk):
        """Create from Redis dictionary"""
        instance = cls(chunk)
        instance.chunk_id = data.get("chunk_id", chunk.chunk_id)
        instance.chunk_number = data.get("chunk_number", chunk.chunk_number)
        instance.generated_items = data.get("generated_items", [])
        instance.execution_time = data.get("execution_time", 0.0)
        instance.status = data.get("status", "pending")
        instance.error_message = data.get("error_message")
        instance.metadata = data.get("metadata", {})
        instance.start_time = data.get("start_time")
        instance.end_time = data.get("end_time")
        return instance
    
    def mark_started(self):
        """Mark chunk execution as started"""
        self.status = "executing"
        self.start_time = datetime.now().isoformat()
    
    def mark_completed(self, items: List[str], metadata: Dict[str, Any] = None):
        """Mark chunk execution as completed"""
        self.status = "completed"
        self.end_time = datetime.now().isoformat()
        self.generated_items = items
        if metadata:
            self.metadata.update(metadata)
        
        if self.start_time:
            start_dt = datetime.fromisoformat(self.start_time)
            end_dt = datetime.fromisoformat(self.end_time)
            self.execution_time = (end_dt - start_dt).total_seconds()
    
    def mark_failed(self, error: str):
        """Mark chunk execution as failed"""
        self.status = "failed"
        self.end_time = datetime.now().isoformat()
        self.error_message = error
        
        if self.start_time:
            start_dt = datetime.fromisoformat(self.start_time)
            end_dt = datetime.fromisoformat(self.end_time)
            self.execution_time = (end_dt - start_dt).total_seconds()


class RedisContinuityManager:
    """
    Redis-based continuity manager for fast session state management
    and progress tracking across chunked tasks
    """
    
    def __init__(self, session_id: Optional[str] = None, redis_url: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        
        # Use environment-aware Redis URL
        if redis_url:
            self.redis_url = redis_url
        else:
            import os
            redis_host = os.getenv("REDIS_HOST", "redis")
            redis_port = int(os.getenv("REDIS_PORT", 6379))
            
            # Docker environment detection
            if redis_host == "localhost" and os.path.exists("/.dockerenv"):
                redis_host = "redis"
            
            self.redis_url = f"redis://{redis_host}:{redis_port}"
        
        self.redis_client: Optional[redis.Redis] = None
        self.llm_settings = get_llm_settings()
        self.continuation_overlap = 3
        
        # Redis key patterns
        self.session_key = f"chunked_session:{self.session_id}"
        self.progress_key = f"chunked_progress:{self.session_id}"
        self.chunks_key = f"chunked_results:{self.session_id}"
        self.context_key = f"chunked_context:{self.session_id}"
        
        # TTL settings from centralized configuration
        from app.core.timeout_settings_cache import get_timeout_value
        self.session_ttl = get_timeout_value("session_cache", "conversation_cache_ttl", 86400)
        
        logger.info(f"[REDIS_CONTINUITY] Initialized session {self.session_id}")
        logger.info(f"[REDIS_CONTINUITY] Using Redis URL: {self.redis_url}")
    
    async def _get_redis_client(self) -> redis.Redis:
        """Get or create Redis client"""
        if self.redis_client is None:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        return self.redis_client
    
    async def _close_redis_client(self):
        """Close Redis client"""
        if self.redis_client:
            await self.redis_client.aclose()
            self.redis_client = None
    
    async def execute_chunked_task(
        self, 
        chunks: List[TaskChunk],
        agent_name: str = "continuation_agent"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute chunked task with Redis state management
        """
        try:
            redis_client = await self._get_redis_client()
            
            logger.info(f"[REDIS_CONTINUITY] Starting execution of {len(chunks)} chunks")
            
            # Initialize session in Redis
            session_data = {
                "session_id": self.session_id,
                "total_chunks": len(chunks),
                "status": "started",
                "start_time": datetime.now().isoformat(),
                "agent_name": agent_name
            }
            
            await redis_client.hset(self.session_key, mapping={
                k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) 
                for k, v in session_data.items()
            })
            await redis_client.expire(self.session_key, self.session_ttl)
            
            # Initialize progress tracking
            progress_data = {
                "total_chunks": len(chunks),
                "completed_chunks": 0,
                "failed_chunks": 0,
                "total_items_generated": 0,
                "current_chunk": 0,
                "estimated_completion_time": None,
                "start_time": datetime.now().isoformat()
            }
            
            await redis_client.hset(self.progress_key, mapping={
                k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) 
                for k, v in progress_data.items()
            })
            await redis_client.expire(self.progress_key, self.session_ttl)
            
            yield {
                "type": "task_started",
                "session_id": self.session_id,
                "total_chunks": len(chunks),
                "redis_keys": {
                    "session": self.session_key,
                    "progress": self.progress_key,
                    "chunks": self.chunks_key
                }
            }
            
            # Execute chunks sequentially
            all_generated_items = []
            
            logger.info(f"[REDIS_CONTINUITY] Processing {len(chunks)} chunks SEQUENTIALLY (NOT PARALLEL)")
            
            # CRITICAL: Process chunks one at a time to prevent duplicates
            for i, chunk in enumerate(chunks):
                logger.info(f"[REDIS_CONTINUITY] === STARTING CHUNK {chunk.chunk_number} OF {len(chunks)} ===")
                try:
                    # Update current chunk in progress
                    await redis_client.hset(self.progress_key, "current_chunk", str(i + 1))
                    
                    # Build continuation context from current session + Redis
                    continuation_context = self._build_continuation_context_from_memory(
                        chunk, all_generated_items, redis_client
                    )
                    chunk.continuation_context = continuation_context
                    
                    logger.info(f"[REDIS_CONTINUITY] Chunk {chunk.chunk_number} context: {continuation_context[:200]}...")
                    
                    # Wait for any previous chunks to complete (ensure sequential processing)
                    if chunk.chunk_number > 1:
                        # Check if previous chunk is actually completed
                        prev_chunk_completed = False
                        max_wait = 30  # 30 seconds max wait
                        wait_count = 0
                        
                        while not prev_chunk_completed and wait_count < max_wait:
                            completed_chunks = await redis_client.hget(self.progress_key, "completed_chunks")
                            if completed_chunks and int(completed_chunks) >= chunk.chunk_number - 1:
                                prev_chunk_completed = True
                            else:
                                logger.info(f"[REDIS_CONTINUITY] Waiting for chunk {chunk.chunk_number - 1} to complete...")
                                await asyncio.sleep(1)
                                wait_count += 1
                        
                        if not prev_chunk_completed:
                            logger.warning(f"[REDIS_CONTINUITY] Timeout waiting for previous chunk, proceeding anyway")
                    
                    # Store current chunk context in Redis
                    await redis_client.hset(
                        self.context_key, 
                        f"chunk_{chunk.chunk_number}_context",
                        continuation_context
                    )
                    await redis_client.expire(self.context_key, self.session_ttl)
                    
                    # Create chunk result and store in Redis
                    chunk_result = RedisChunkResult(chunk)
                    chunk_result.mark_started()
                    
                    await redis_client.hset(
                        self.chunks_key,
                        chunk.chunk_id,
                        json.dumps(chunk_result.to_dict())
                    )
                    await redis_client.expire(self.chunks_key, self.session_ttl)
                    
                    yield {
                        "type": "chunk_started",
                        "chunk_number": chunk.chunk_number,
                        "total_chunks": chunk.total_chunks,
                        "chunk_id": chunk.chunk_id,
                        "items_range": f"{chunk.start_index}-{chunk.end_index}",
                        "session_id": self.session_id
                    }
                    
                    # Execute the chunk
                    async for event in self._execute_single_chunk(chunk, agent_name):
                        if event.get("type") == "chunk_completed":
                            # Extract generated items
                            generated_items = self._extract_items_from_response(
                                event.get("content", ""), 
                                chunk
                            )
                            
                            # Update chunk result
                            chunk_result.mark_completed(
                                generated_items,
                                {"response_length": len(event.get("content", ""))}
                            )
                            
                            # Store updated result in Redis
                            await redis_client.hset(
                                self.chunks_key,
                                chunk.chunk_id,
                                json.dumps(chunk_result.to_dict())
                            )
                            
                            # Add to overall results and store in Redis
                            all_generated_items.extend(generated_items)
                            await redis_client.hset(
                                self.session_key,
                                "all_items",
                                json.dumps(all_generated_items)
                            )
                            
                            logger.info(f"[REDIS_CONTINUITY] Chunk {chunk.chunk_number} extracted {len(generated_items)} items (expected {chunk.chunk_size})")
                            logger.info(f"[REDIS_CONTINUITY] Total items stored in Redis: {len(all_generated_items)}")
                            logger.info(f"[REDIS_CONTINUITY] Latest items: {generated_items[:2] if generated_items else 'None'}")
                            
                            # Check for over-extraction
                            if len(generated_items) > chunk.chunk_size:
                                logger.warning(f"[REDIS_CONTINUITY] WARNING: Extracted {len(generated_items)} items but chunk size is {chunk.chunk_size}")
                            
                            logger.info(f"[REDIS_CONTINUITY] === COMPLETED CHUNK {chunk.chunk_number} ===")
                            
                            # Update progress in Redis
                            progress_updates = {
                                "completed_chunks": str(i + 1),
                                "total_items_generated": str(len(all_generated_items))
                            }
                            
                            # Calculate estimated completion time
                            if i > 0:
                                # Get execution times from Redis
                                chunk_results_raw = await redis_client.hgetall(self.chunks_key)
                                completed_times = []
                                
                                for chunk_data_json in chunk_results_raw.values():
                                    chunk_data = json.loads(chunk_data_json)
                                    if chunk_data.get("status") == "completed":
                                        completed_times.append(chunk_data.get("execution_time", 0))
                                
                                if completed_times:
                                    avg_time = sum(completed_times) / len(completed_times)
                                    remaining_chunks = len(chunks) - (i + 1)
                                    estimated_remaining = remaining_chunks * avg_time
                                    progress_updates["estimated_completion_time"] = str(estimated_remaining)
                            
                            await redis_client.hset(self.progress_key, mapping=progress_updates)
                            
                            yield {
                                "type": "chunk_completed",
                                "chunk_number": chunk.chunk_number,
                                "chunk_id": chunk.chunk_id,
                                "content": event.get("content", ""),  # ADD THE ACTUAL CONTENT!
                                "items_generated": len(generated_items),
                                "cumulative_items": len(all_generated_items),
                                "execution_time": chunk_result.execution_time,
                                "progress_percentage": ((i + 1) / len(chunks)) * 100,
                                "estimated_remaining_seconds": progress_updates.get("estimated_completion_time"),
                                "partial_results": generated_items[-3:] if generated_items else [],
                                "session_id": self.session_id
                            }
                            
                        elif event.get("type") == "chunk_error":
                            chunk_result.mark_failed(event.get("error", "Unknown error"))
                            
                            # Update Redis
                            await redis_client.hset(
                                self.chunks_key,
                                chunk.chunk_id,
                                json.dumps(chunk_result.to_dict())
                            )
                            await redis_client.hincrby(self.progress_key, "failed_chunks", 1)
                            
                            yield {
                                "type": "chunk_failed",
                                "chunk_number": chunk.chunk_number,
                                "chunk_id": chunk.chunk_id,
                                "error": event.get("error"),
                                "session_id": self.session_id
                            }
                        
                        # Forward other events
                        yield event
                
                except Exception as e:
                    logger.error(f"[REDIS_CONTINUITY] Chunk {chunk.chunk_id} failed: {e}")
                    
                    # Update Redis with failure
                    chunk_result = RedisChunkResult(chunk)
                    chunk_result.mark_failed(str(e))
                    await redis_client.hset(
                        self.chunks_key,
                        chunk.chunk_id,
                        json.dumps(chunk_result.to_dict())
                    )
                    await redis_client.hincrby(self.progress_key, "failed_chunks", 1)
                    
                    yield {
                        "type": "chunk_failed",
                        "chunk_number": chunk.chunk_number,
                        "chunk_id": chunk.chunk_id,
                        "error": str(e),
                        "session_id": self.session_id
                    }
            
            # Final completion
            end_time = datetime.now().isoformat()
            await redis_client.hset(self.session_key, "end_time", end_time)
            await redis_client.hset(self.session_key, "status", "completed")
            await redis_client.hset(self.progress_key, "end_time", end_time)
            
            # Generate execution summary
            summary = await self._generate_execution_summary_from_redis(redis_client)
            
            # Ensure we don't exceed the original target count
            # Calculate the actual target from chunk end indices
            original_target = max(chunk.end_index for chunk in chunks) if chunks else len(all_generated_items)
            if len(all_generated_items) > original_target:
                logger.warning(f"[REDIS_CONTINUITY] Over-generated: {len(all_generated_items)} items vs target {original_target}, trimming")
                all_generated_items = all_generated_items[:original_target]
            
            yield {
                "type": "task_completed", 
                "session_id": self.session_id,
                "total_items_generated": len(all_generated_items),
                "final_results": all_generated_items,
                "summary": summary,
                "redis_session_retained_for": f"{self.session_ttl} seconds"
            }
            
        finally:
            # Keep Redis connection open for potential progress queries
            pass
    
    async def _build_continuation_context_from_redis(
        self, 
        chunk: TaskChunk, 
        redis_client: redis.Redis
    ) -> str:
        """Build continuation context from Redis-stored previous results"""
        
        if chunk.chunk_number == 1:
            return f"Generate items {chunk.start_index} to {chunk.end_index} (first chunk - start fresh):"
        
        try:
            # Get all previous items from Redis
            all_items_json = await redis_client.hget(self.session_key, "all_items")
            
            if all_items_json:
                all_items = json.loads(all_items_json)
                
                # Get last few items for context
                context_items = all_items[-self.continuation_overlap:] if len(all_items) >= self.continuation_overlap else all_items
                
                last_item_number = chunk.start_index - 1
                
                context = f"""Previous items for context (DO NOT REPEAT THESE):

{chr(10).join(f"{last_item_number - len(context_items) + i + 1}. {item.strip()}" for i, item in enumerate(context_items))}

IMPORTANT: 
- Continue from item {chunk.start_index} to {chunk.end_index}
- DO NOT repeat any of the above questions
- Generate NEW, UNIQUE questions that are different from all previous ones
- Follow the same professional interview style

Generate items {chunk.start_index} to {chunk.end_index}:"""
                
                return context
            
        except Exception as e:
            logger.error(f"[REDIS_CONTINUITY] Failed to build context from Redis: {e}")
        
        # Fallback - at least provide basic continuation guidance
        return f"""Continue generating items {chunk.start_index} to {chunk.end_index}.

IMPORTANT: 
- This is chunk {chunk.chunk_number} of {chunk.total_chunks}
- Generate UNIQUE questions different from any previous items
- Do NOT repeat questions from earlier chunks
- Maintain professional interview style"""
    
    def _build_continuation_context_from_memory(
        self, 
        chunk: TaskChunk, 
        all_generated_items: List[str],
        redis_client: redis.Redis
    ) -> str:
        """Build continuation context from in-memory items for better reliability"""
        
        if chunk.chunk_number == 1:
            return f"Generate items {chunk.start_index} to {chunk.end_index} (first chunk - start fresh):"
        
        if not all_generated_items:
            logger.warning(f"[REDIS_CONTINUITY] No previous items in memory for chunk {chunk.chunk_number}")
            return f"""Continue generating items {chunk.start_index} to {chunk.end_index}.

IMPORTANT: 
- This is chunk {chunk.chunk_number} of {chunk.total_chunks}
- Generate UNIQUE questions different from any previous items
- Do NOT repeat questions from earlier chunks
- Maintain professional interview style"""
        
        # Use more items from memory for better duplicate prevention
        context_overlap = min(8, len(all_generated_items))  # Show up to 8 previous items
        context_items = all_generated_items[-context_overlap:] if len(all_generated_items) >= context_overlap else all_generated_items
        
        context = f"""Previous items for context (DO NOT REPEAT THESE):

{chr(10).join(f"{len(all_generated_items) - len(context_items) + i + 1}. {item.strip()}" for i, item in enumerate(context_items))}

IMPORTANT: 
- Continue from item {chunk.start_index} to {chunk.end_index}
- DO NOT repeat any of the above questions
- DO NOT create similar questions with slightly different wording
- Generate NEW, UNIQUE questions that are different from ALL {len(all_generated_items)} previous ones
- Each question must cover a DIFFERENT aspect or topic area
- Avoid overlapping concepts (e.g., if "align AI with strategy" exists, don't ask about "strategic alignment")
- Follow the same professional interview style

Generate items {chunk.start_index} to {chunk.end_index}:"""
        
        logger.info(f"[REDIS_CONTINUITY] Using {len(context_items)} previous items for context in chunk {chunk.chunk_number}")
        return context
    
    async def _execute_single_chunk(
        self, 
        chunk: TaskChunk, 
        agent_name: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute a single chunk (same as before but with Redis awareness)"""
        
        prompt = self._build_chunk_prompt(chunk)
        
        try:
            # Use thinking_mode for better reasoning in chunked generation
            model_config = self.llm_settings.get("thinking_mode", {})
            if not model_config:
                # Fallback to non_thinking_mode if thinking_mode not available
                model_config = self.llm_settings.get("non_thinking_mode", {})
            
            config = LLMConfig(
                model_name=model_config.get("model", "qwen3:30b-a3b"),
                temperature=0.6,  # Slightly higher for creative content generation
                top_p=model_config.get("top_p", 0.9),
                max_tokens=3500
            )
            
            # Get Ollama URL from settings or environment
            ollama_url = None
            
            # Try to get from model_config first
            if model_config.get('model_server'):
                ollama_url = model_config['model_server']
            
            # Fall back to environment variable
            if not ollama_url:
                ollama_url = os.environ.get("OLLAMA_BASE_URL")
            
            # Use default based on environment
            if not ollama_url:
                in_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER')
                ollama_url = "http://host.docker.internal:11434" if in_docker else "http://host.docker.internal:11434"
            llm = OllamaLLM(config, base_url=ollama_url)
            
            response_text = ""
            
            async with asyncio.timeout(120):
                async for response_chunk in llm.generate_stream(prompt):
                    response_text += response_chunk.text
            
            # Apply dynamic response processing for thinking tag removal
            cleaned_response = response_text
            try:
                from app.llm.response_analyzer import detect_model_thinking_behavior
                
                # Get model name from the LLM config - need to find it in the scope
                # This is a bit tricky since we need the model name, let me get it from settings
                from app.core.llm_settings_cache import get_llm_settings, get_second_llm_full_config
                settings = get_llm_settings()
                model_config = get_second_llm_full_config(settings)
                model_name = model_config.get('model', 'unknown')
                
                # Detect model behavior and process response accordingly
                is_thinking, confidence = detect_model_thinking_behavior(response_text, model_name)
                
                if is_thinking and confidence > 0.8:
                    # Remove thinking tags from response
                    import re
                    cleaned_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL | re.IGNORECASE).strip()
                    print(f"[CONTINUATION] Removed thinking tags from {model_name} response (confidence: {confidence:.2f})")
                else:
                    # Non-thinking model or low confidence - use original cleaning method
                    cleaned_response = self._clean_chunk_response(response_text)
                    print(f"[CONTINUATION] Using original cleaning for {model_name} (thinking: {is_thinking}, confidence: {confidence:.2f})")
                
            except Exception as e:
                print(f"[CONTINUATION] Dynamic detection failed: {e}, using fallback cleaning")
                cleaned_response = self._clean_chunk_response(response_text)
            
            yield {
                "type": "chunk_completed", 
                "content": cleaned_response,
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
        """Build prompt for chunk execution"""
        task_analysis = chunk.metadata.get("task_analysis", {})
        requires_numbering = task_analysis.get("requires_numbering", True)
        content_type = task_analysis.get("content_type", "unknown")
        
        prompt = f"""You are a specialized interview question generator. Think through your approach first.

TASK: {chunk.task_description}

GENERATE: Items {chunk.start_index} to {chunk.end_index} ({chunk.chunk_size} items)

{chunk.continuation_context}

REQUIREMENTS:
- {"Start numbering from " + str(chunk.start_index) if requires_numbering else "Do not use numbers"}
- Think about what areas haven't been covered yet
- Ensure each question is distinct and covers different competencies
- Avoid semantic overlap with previous questions

<think>
First, let me review the previous questions to understand what areas have been covered. Then I'll brainstorm new areas and competencies that should be tested for an AI Director role at a bank. I need to ensure each question is unique and covers different aspects like technical skills, leadership, strategy, compliance, ethics, implementation, team management, etc.
</think>

Generate the {chunk.chunk_size} questions now:"""
        
        return prompt
    
    def _clean_chunk_response(self, response: str) -> str:
        """Clean chunk response by removing only excessive reasoning sections"""
        import re
        
        cleaned = response
        
        # Remove reasoning sections
        cleaned = re.sub(r'Reasoning:\s*.*?(?=\d+\.|\n\n|\Z)', '', cleaned, flags=re.DOTALL)
        
        # Remove excessive repetition of the same reasoning text
        lines = cleaned.split('\n')
        unique_lines = []
        seen_content = set()
        
        for line in lines:
            line_clean = line.strip()
            # Skip empty lines and repetitive reasoning
            if line_clean and not any(phrase in line_clean.lower() for phrase in [
                'okay, i need to generate',
                'let me start by understanding',
                'i should brainstorm',
                'let me check',
                'i need to make sure'
            ]):
                # Only add if we haven't seen very similar content
                line_key = line_clean[:50].lower()  # First 50 chars as key
                if line_key not in seen_content:
                    seen_content.add(line_key)
                    unique_lines.append(line)
        
        # Rejoin and clean up excessive whitespace
        cleaned = '\n'.join(unique_lines)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Max 2 consecutive newlines
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _extract_items_from_response(self, response: str, chunk: TaskChunk) -> List[str]:
        """Extract items from response with improved question detection"""
        if not response:
            return []
        
        items = []
        import re
        
        # Strategy 1: Extract numbered items (questions/statements)
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Match numbered questions/items: "1. Question", "1) Question", etc.
            numbered_match = re.match(r'^\d+[\.\)]\s*(.+)', line)
            if numbered_match:
                question = numbered_match.group(1).strip()
                if len(question) > 10:  # Ensure it's a substantial question
                    items.append(question)
        
        # Strategy 2: If not enough numbered items, look for question patterns
        if len(items) < chunk.chunk_size // 2:
            question_patterns = [
                r'How do you [^?]+\?',
                r'What [^?]+\?',
                r'Can you [^?]+\?',
                r'Describe [^?]+\?',
                r'Explain [^?]+\?',
                r'Tell me about [^?]+\?'
            ]
            
            for pattern in question_patterns:
                if len(items) >= chunk.chunk_size:  # Stop if we have enough
                    break
                matches = re.findall(pattern, response, re.IGNORECASE)
                for match in matches:
                    if len(items) >= chunk.chunk_size:  # Stop if we have enough
                        break
                    if match.strip() and len(match.strip()) > 10 and match.strip() not in items:
                        items.append(match.strip())
        
        # Strategy 3: Split by common separators if still not enough
        if len(items) < chunk.chunk_size // 2:
            # Try splitting by double newlines or question marks
            potential_items = []
            
            # Split by question marks and look for complete questions
            question_chunks = response.split('?')
            for chunk_text in question_chunks:
                chunk_text = chunk_text.strip()
                if len(chunk_text) > 15 and any(word in chunk_text.lower() for word in ['how', 'what', 'can', 'describe', 'explain']):
                    potential_items.append(chunk_text + '?')
            
            # Add unique items
            for item in potential_items:
                if len(items) >= chunk.chunk_size:  # Stop if we have enough
                    break
                if item not in items and len(items) < chunk.chunk_size:
                    items.append(item)
        
        # Ensure we don't exceed the requested chunk size
        items = items[:chunk.chunk_size]
        
        logger.info(f"[REDIS_CONTINUITY] Extracted {len(items)} items from chunk {chunk.chunk_id}")
        logger.info(f"[REDIS_CONTINUITY] Sample items: {items[:3] if items else 'None'}")
        
        return items
    
    async def _generate_execution_summary_from_redis(self, redis_client: redis.Redis) -> Dict[str, Any]:
        """Generate execution summary from Redis data"""
        try:
            # Get all chunk results
            chunk_results_raw = await redis_client.hgetall(self.chunks_key)
            
            completed_count = 0
            failed_count = 0
            total_execution_time = 0
            total_items = 0
            
            for chunk_data_json in chunk_results_raw.values():
                chunk_data = json.loads(chunk_data_json)
                
                if chunk_data.get("status") == "completed":
                    completed_count += 1
                    total_execution_time += chunk_data.get("execution_time", 0)
                    total_items += len(chunk_data.get("generated_items", []))
                elif chunk_data.get("status") == "failed":
                    failed_count += 1
            
            total_chunks = len(chunk_results_raw)
            
            return {
                "total_chunks_executed": total_chunks,
                "successful_chunks": completed_count,
                "failed_chunks": failed_count,
                "success_rate": completed_count / total_chunks if total_chunks > 0 else 0,
                "average_execution_time": total_execution_time / completed_count if completed_count > 0 else 0,
                "total_items_generated": total_items,
                "session_id": self.session_id,
                "redis_keys_used": [self.session_key, self.progress_key, self.chunks_key, self.context_key]
            }
            
        except Exception as e:
            logger.error(f"[REDIS_CONTINUITY] Failed to generate summary: {e}")
            return {"error": str(e)}
    
    async def get_progress_status(self) -> Dict[str, Any]:
        """Get current progress from Redis"""
        try:
            redis_client = await self._get_redis_client()
            
            # Get progress data
            progress_data = await redis_client.hgetall(self.progress_key)
            session_data = await redis_client.hgetall(self.session_key)
            
            # Parse JSON values
            for key, value in progress_data.items():
                try:
                    progress_data[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    pass  # Keep as string
            
            return {
                "session_id": self.session_id,
                "progress": progress_data,
                "session_info": session_data,
                "redis_available": True
            }
            
        except Exception as e:
            logger.error(f"[REDIS_CONTINUITY] Failed to get progress: {e}")
            return {
                "session_id": self.session_id,
                "error": str(e),
                "redis_available": False
            }
    
    async def cleanup_session(self):
        """Clean up Redis keys for this session"""
        try:
            redis_client = await self._get_redis_client()
            
            keys_to_delete = [
                self.session_key,
                self.progress_key,
                self.chunks_key,
                self.context_key
            ]
            
            deleted = await redis_client.delete(*keys_to_delete)
            logger.info(f"[REDIS_CONTINUITY] Cleaned up {deleted} Redis keys for session {self.session_id}")
            
        except Exception as e:
            logger.error(f"[REDIS_CONTINUITY] Failed to cleanup session: {e}")
        finally:
            await self._close_redis_client()