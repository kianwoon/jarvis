"""
Task Decomposer for Context-Limit-Transcending System
Breaks large generation tasks into manageable chunks that stay within context limits
"""

import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib
import math

from app.core.llm_settings_cache import get_llm_settings
from app.llm.ollama import OllamaLLM
from app.llm.base import LLMConfig
import os


@dataclass
class TaskChunk:
    """Represents a single chunk of a larger task"""
    chunk_id: str
    chunk_number: int
    total_chunks: int
    task_description: str
    start_index: int
    end_index: int
    chunk_size: int
    continuation_context: str = ""
    previous_items: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.previous_items is None:
            self.previous_items = []
        if self.metadata is None:
            self.metadata = {}


class TaskDecomposer:
    """
    Intelligently decomposes large generation tasks into manageable chunks
    that respect context limits while maintaining continuity
    """
    
    def __init__(self):
        self.max_context_size = 3500  # Conservative limit for token safety
        self.min_chunk_size = 5      # Minimum items per chunk
        self.max_chunk_size = 20     # Maximum items per chunk for quality
        self.overlap_size = 3        # Number of previous items to include for context
        self.llm_settings = get_llm_settings()
        
    async def decompose_large_task(
        self, 
        query: str, 
        target_count: int = 100,
        chunk_size: Optional[int] = None
    ) -> List[TaskChunk]:
        """
        Break a large generation task into manageable chunks
        
        Args:
            query: The original task description
            target_count: Total number of items to generate
            chunk_size: Optional fixed chunk size, otherwise auto-calculated
            
        Returns:
            List of TaskChunk objects ready for execution
        """
        print(f"[DECOMPOSER] Starting task decomposition for {target_count} items")
        
        # Calculate optimal chunk size if not provided
        if chunk_size is None:
            chunk_size = self._calculate_optimal_chunk_size(query, target_count)
        
        # Ensure chunk size is within bounds
        chunk_size = max(self.min_chunk_size, min(self.max_chunk_size, chunk_size))
        
        # Calculate number of chunks needed
        total_chunks = math.ceil(target_count / chunk_size)
        
        print(f"[DECOMPOSER] Using chunk size: {chunk_size}, total chunks: {total_chunks}")
        
        # Analyze the task to understand generation pattern
        task_analysis = await self._analyze_task_pattern(query, target_count)
        
        # Create chunks
        chunks = []
        for i in range(total_chunks):
            start_index = i * chunk_size + 1
            end_index = min((i + 1) * chunk_size, target_count)
            actual_chunk_size = end_index - start_index + 1
            
            # Generate unique chunk ID
            chunk_id = self._generate_chunk_id(query, i, start_index, end_index)
            
            chunk = TaskChunk(
                chunk_id=chunk_id,
                chunk_number=i + 1,
                total_chunks=total_chunks,
                task_description=query,
                start_index=start_index,
                end_index=end_index,
                chunk_size=actual_chunk_size,
                metadata={
                    "task_analysis": task_analysis,
                    "estimated_tokens": self._estimate_chunk_tokens(query, actual_chunk_size),
                    "created_at": datetime.now().isoformat()
                }
            )
            
            chunks.append(chunk)
        
        print(f"[DECOMPOSER] Created {len(chunks)} chunks successfully")
        return chunks
    
    def _calculate_optimal_chunk_size(self, query: str, target_count: int) -> int:
        """Calculate optimal chunk size based on task complexity and target count"""
        
        print(f"[DECOMPOSER] Calculating chunk size for target_count={target_count}")
        
        # Base chunk size calculation
        base_chunk_size = 15  # Conservative default
        
        # Adjust based on task complexity
        query_complexity = self._assess_query_complexity(query)
        
        if query_complexity == "simple":
            base_chunk_size = 20  # Can handle more simple items
        elif query_complexity == "complex":
            base_chunk_size = 10  # Fewer complex items per chunk
        
        # Adjust based on total target count - IMPROVED LOGIC FOR BETTER UX
        if target_count <= 20:
            chunk_size = min(target_count, 10)  # Small tasks don't need chunking
        elif target_count <= 30:
            # For 21-30 items, use 2 chunks to avoid too many small chunks
            chunk_size = max(10, target_count // 2)  # Aim for roughly 2 chunks
        elif target_count <= 50:
            # For 31-50 items, aim for 2-3 chunks maximum for better UX
            if target_count <= 40:
                chunk_size = max(15, target_count // 2)  # 2 chunks for 31-40 items
            else:
                chunk_size = max(17, target_count // 3)  # 3 chunks for 41-50 items
        else:
            chunk_size = base_chunk_size
        
        print(f"[DECOMPOSER] Selected chunk_size={chunk_size} for target_count={target_count}")
        return chunk_size
    
    def _assess_query_complexity(self, query: str) -> str:
        """Assess the complexity of the generation task"""
        query_lower = query.lower()
        
        # Complex indicators
        complex_keywords = [
            "detailed", "comprehensive", "analysis", "explanation", 
            "research", "strategy", "plan", "framework", "methodology"
        ]
        
        # Simple indicators  
        simple_keywords = [
            "list", "name", "title", "short", "brief", "quick",
            "simple", "basic", "example"
        ]
        
        complex_score = sum(1 for keyword in complex_keywords if keyword in query_lower)
        simple_score = sum(1 for keyword in simple_keywords if keyword in query_lower)
        
        if complex_score > simple_score:
            return "complex"
        elif simple_score > complex_score:
            return "simple"
        else:
            return "medium"
    
    async def _analyze_task_pattern(self, query: str, target_count: int) -> Dict[str, Any]:
        """Analyze the task to understand the pattern and requirements"""
        
        analysis_prompt = f"""Analyze this generation task and provide insights for chunked execution:

Task: {query}
Target Count: {target_count}

Provide analysis in JSON format:
{{
    "content_type": "questionnaire|list|document|code|data",
    "requires_numbering": true/false,
    "requires_consistency": true/false,
    "difficulty_level": "simple|medium|complex",
    "key_requirements": ["requirement1", "requirement2"],
    "continuation_strategy": "numbered_sequence|thematic_continuation|independent_items"
}}

Focus on understanding what type of content needs to be generated and how to maintain consistency across chunks."""

        try:
            response = await self._call_llm(analysis_prompt, temperature=0.3, max_tokens=500)
            
            # Parse JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                print(f"[DECOMPOSER] Task analysis: {analysis}")
                return analysis
        except Exception as e:
            print(f"[DECOMPOSER] Task analysis failed: {e}")
        
        # Fallback analysis
        return {
            "content_type": "unknown",
            "requires_numbering": "question" in query.lower() or "list" in query.lower(),
            "requires_consistency": True,
            "difficulty_level": "medium",
            "key_requirements": ["maintain_consistency", "proper_numbering"],
            "continuation_strategy": "numbered_sequence"
        }
    
    def _estimate_chunk_tokens(self, query: str, chunk_size: int) -> int:
        """Estimate token count for a chunk"""
        # Rough estimation: 
        # - Query: ~100 tokens
        # - Context from previous items: ~200 tokens  
        # - Generated items: chunk_size * average_tokens_per_item
        # - Instructions and formatting: ~100 tokens
        
        avg_tokens_per_item = 50  # Conservative estimate
        
        estimated_tokens = (
            100 +  # Query
            200 +  # Context
            (chunk_size * avg_tokens_per_item) +  # Generated content
            100    # Instructions
        )
        
        return estimated_tokens
    
    def _generate_chunk_id(self, query: str, chunk_number: int, start_index: int, end_index: int) -> str:
        """Generate a unique ID for the chunk"""
        content = f"{query}_{chunk_number}_{start_index}_{end_index}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    async def _call_llm(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Call LLM for task analysis"""
        try:
            model_config = self.llm_settings.get("thinking_mode", {})
            
            config = LLMConfig(
                model_name=model_config.get("model", "qwen3:30b-a3b"),
                temperature=temperature,
                top_p=model_config.get("top_p", 0.9),
                max_tokens=max_tokens
            )
            
            ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            llm = OllamaLLM(config, base_url=ollama_url)
            
            response_text = ""
            async for response_chunk in llm.generate_stream(prompt):
                response_text += response_chunk.text
                
            return response_text
            
        except Exception as e:
            print(f"[DECOMPOSER] LLM call failed: {e}")
            return ""
    
    def create_continuation_context(self, chunk: TaskChunk, previous_results: List[str]) -> str:
        """Create context for continuation between chunks"""
        if not previous_results:
            return ""
        
        # Take last few items for context
        context_items = previous_results[-self.overlap_size:] if len(previous_results) >= self.overlap_size else previous_results
        
        context = f"""Previous items for context (continue the pattern):
{chr(10).join(f"{i+chunk.start_index-len(context_items)}. {item}" for i, item in enumerate(context_items))}

Continue from item {chunk.start_index}..."""
        
        return context
    
    def validate_chunk_feasibility(self, chunk: TaskChunk) -> bool:
        """Validate that a chunk is feasible within context limits"""
        estimated_tokens = chunk.metadata.get("estimated_tokens", 0)
        
        if estimated_tokens > self.max_context_size:
            print(f"[DECOMPOSER] Warning: Chunk {chunk.chunk_id} may exceed context limit ({estimated_tokens} tokens)")
            return False
        
        return True