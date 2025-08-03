"""
Dynamic Chunk Sizing Service

Optimizes chunk sizes based on model context limits to maximize efficiency
while maintaining quality for knowledge graph extraction.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from app.core.knowledge_graph_settings_cache import get_knowledge_graph_settings
from app.document_handlers.base import ExtractedChunk

logger = logging.getLogger(__name__)

class DynamicChunkSizer:
    """Dynamically sizes chunks based on model capabilities"""
    
    # Model context window mappings (in tokens, approximate)
    MODEL_CONTEXT_LIMITS = {
        # OpenAI models
        'gpt-4': 8192,
        'gpt-4-32k': 32768,
        'gpt-4-turbo': 128000,
        'gpt-3.5-turbo': 4096,
        'gpt-3.5-turbo-16k': 16384,
        
        # Claude models
        'claude-3-sonnet': 200000,
        'claude-3-opus': 200000,
        'claude-3-haiku': 200000,
        'claude-2': 100000,
        'claude-instant': 100000,
        
        # Qwen models (based on model name patterns) - FALLBACK ONLY
        'qwen3:1.7b': 32768,
        'qwen3:7b': 32768,
        'qwen3:14b': 32768,
        'qwen3:30b': 40960,  # Conservative fallback - database should have actual capacity
        'qwen3:30b-a3b-instruct-2507-q4_k_m': 40960,  # Database should override this
        'qwen3:72b': 32768,
        
        # Llama models
        'llama2:7b': 4096,
        'llama2:13b': 4096,
        'llama2:70b': 4096,
        'llama3:8b': 8192,
        'llama3:70b': 8192,
        
        # Mistral models
        'mistral:7b': 8192,
        'mixtral:8x7b': 32768,
        
        # Generic defaults by size
        'small': 4096,    # < 3B parameters
        'medium': 8192,   # 3B - 15B parameters  
        'large': 32768,   # 15B - 100B parameters
        'xlarge': 100000, # > 100B parameters
    }
    
    def __init__(self):
        self.kg_settings = get_knowledge_graph_settings()
        self.model_name = self._get_current_model()
        self.context_limit = self._determine_context_limit()
        self.optimal_chunk_size = self._calculate_optimal_chunk_size()
        
        logger.info(f"🧠 Dynamic Chunk Sizer initialized:")
        logger.info(f"   Model: {self.model_name}")
        logger.info(f"   Context limit: {self.context_limit:,} tokens")
        logger.info(f"   Optimal chunk size: {self.optimal_chunk_size:,} characters")
    
    def _get_current_model(self) -> str:
        """Get the current model from knowledge graph settings"""
        # Check the model_config section first (most likely location)
        model_config = self.kg_settings.get('model_config', {})
        if 'model' in model_config:
            model = model_config['model']
            logger.debug(f"🎯 Found model in model_config: {model}")
            return str(model).lower()
        
        # Check direct model field in settings
        if 'model' in self.kg_settings:
            model = self.kg_settings['model']
            logger.debug(f"🎯 Found model in direct settings: {model}")
            return str(model).lower()
        
        # Check for legacy locations
        legacy_locations = [
            'llm_model',
            'extraction_model', 
            'kg_model'
        ]
        
        for location in legacy_locations:
            if location in self.kg_settings:
                model = self.kg_settings[location]
                logger.debug(f"🎯 Found model in legacy location {location}: {model}")
                return str(model).lower()
        
        # Log available keys for debugging
        logger.warning(f"⚠️  Model not found in knowledge graph settings. Available keys: {list(self.kg_settings.keys())}")
        if model_config:
            logger.warning(f"⚠️  model_config keys: {list(model_config.keys())}")
        
        return 'unknown'
    
    def _determine_context_limit(self) -> int:
        """Determine context limit for the current model - prioritize database settings"""
        
        # PRIORITY 1: Check database/Redis settings for context_length
        model_config = self.kg_settings.get('model_config', {})
        if 'context_length' in model_config:
            context_length = model_config['context_length']
            logger.info(f"✅ Using context_length from database: {context_length:,} tokens")
            return int(context_length)
        
        # Check direct settings too
        if 'context_length' in self.kg_settings:
            context_length = self.kg_settings['context_length']
            logger.info(f"✅ Using context_length from settings: {context_length:,} tokens")
            return int(context_length)
        
        # PRIORITY 2: Fallback to hardcoded model mappings only if no database setting
        logger.warning(f"⚠️  No context_length in database settings, using hardcoded fallback for {self.model_name}")
        model_name = self.model_name.lower()
        
        # Try exact match first
        if model_name in self.MODEL_CONTEXT_LIMITS:
            return self.MODEL_CONTEXT_LIMITS[model_name]
        
        # Try pattern matching for common model formats
        for pattern, limit in self.MODEL_CONTEXT_LIMITS.items():
            if pattern in model_name:
                return limit
        
        # Conservative default if no match found
        logger.warning(f"⚠️  No context limit found for model {model_name}, using conservative default: 4096")
        return 4096
    
    def _calculate_optimal_chunk_size(self) -> int:
        """Calculate optimal chunk size in characters"""
        # Convert tokens to approximate characters (1 token ≈ 4 characters average)
        chars_per_token = 4
        
        # Reserve space for system prompt, response, and overhead
        prompt_overhead = 1000  # tokens for system prompt + response
        safety_margin = 0.7     # Use 70% of available context
        
        available_tokens = (self.context_limit - prompt_overhead) * safety_margin
        optimal_chars = int(available_tokens * chars_per_token)
        
        # Apply reasonable bounds
        min_chunk_size = 1000    # Minimum 1KB chunks
        max_chunk_size = 200000  # Maximum 200KB chunks
        
        return max(min_chunk_size, min(optimal_chars, max_chunk_size))
    
    def should_use_large_chunks(self) -> bool:
        """Determine if large chunks should be used for this model"""
        return self.context_limit >= 32768  # 32K+ context models can handle large chunks
    
    def get_chunk_configuration(self) -> Dict[str, Any]:
        """Get recommended chunk configuration"""
        if self.should_use_large_chunks():
            # Large context models: use big chunks, minimal splitting
            # But allow small documents to be processed too
            return {
                'max_chunk_size': self.optimal_chunk_size,
                'min_chunk_size': 100,  # Allow small documents for large context models
                'chunk_overlap': 500,
                'processing_strategy': 'large_context',
                'combine_small_chunks': True,
                'max_chunks_per_call': 1,  # Process one large chunk at a time
                'enable_document_level_processing': True
            }
        else:
            # Small context models: use traditional chunking
            return {
                'max_chunk_size': min(self.optimal_chunk_size, 3000),
                'min_chunk_size': 500,
                'chunk_overlap': 200,
                'processing_strategy': 'traditional',
                'combine_small_chunks': False,
                'max_chunks_per_call': 1,  # Still serial processing
                'enable_document_level_processing': False
            }
    
    def optimize_chunks(self, chunks: List[ExtractedChunk]) -> List[ExtractedChunk]:
        """Optimize chunk sizes based on model capabilities"""
        config = self.get_chunk_configuration()
        
        if config['processing_strategy'] == 'large_context':
            return self._create_large_chunks(chunks, config)
        else:
            return self._resize_traditional_chunks(chunks, config)
    
    def _create_large_chunks(self, chunks: List[ExtractedChunk], config: Dict[str, Any]) -> List[ExtractedChunk]:
        """Combine small chunks into large ones for high-context models"""
        if not chunks:
            return chunks
        
        max_size = config['max_chunk_size']
        overlap = config['chunk_overlap']
        optimized_chunks = []
        
        current_content = ""
        current_metadata = chunks[0].metadata.copy()
        chunk_sources = []
        
        for i, chunk in enumerate(chunks):
            chunk_content = chunk.content.strip()
            
            # Check if adding this chunk would exceed the limit
            potential_size = len(current_content) + len(chunk_content) + overlap
            
            if potential_size > max_size and current_content:
                # Create the large chunk
                optimized_chunk = ExtractedChunk(
                    content=current_content.strip(),
                    metadata={
                        **current_metadata,
                        'chunk_id': f"large_chunk_{len(optimized_chunks)}",
                        'combined_from': chunk_sources,
                        'optimization': 'large_context',
                        'original_chunk_count': len(chunk_sources)
                    },
                    quality_score=sum(c.quality_score for c in chunks[max(0, i-len(chunk_sources)):i]) / len(chunk_sources) if chunk_sources else 1.0
                )
                optimized_chunks.append(optimized_chunk)
                
                # Start new chunk with overlap
                if overlap > 0 and len(current_content) > overlap:
                    current_content = current_content[-overlap:] + "\n\n" + chunk_content
                else:
                    current_content = chunk_content
                
                chunk_sources = [chunk.chunk_id]
                current_metadata = chunk.metadata.copy()
            else:
                # Add to current chunk
                if current_content:
                    current_content += "\n\n" + chunk_content
                else:
                    current_content = chunk_content
                chunk_sources.append(chunk.chunk_id)
        
        # Add final chunk if there's remaining content
        if current_content.strip():
            optimized_chunk = ExtractedChunk(
                content=current_content.strip(),
                metadata={
                    **current_metadata,
                    'chunk_id': f"large_chunk_{len(optimized_chunks)}",
                    'combined_from': chunk_sources,
                    'optimization': 'large_context',
                    'original_chunk_count': len(chunk_sources)
                },
                quality_score=sum(c.quality_score for c in chunks[-len(chunk_sources):]) / len(chunk_sources) if chunk_sources else 1.0
            )
            optimized_chunks.append(optimized_chunk)
        
        logger.info(f"🔄 Large chunk optimization: {len(chunks)} → {len(optimized_chunks)} chunks")
        logger.info(f"   Average chunk size: {sum(len(c.content) for c in optimized_chunks) // len(optimized_chunks):,} chars")
        
        return optimized_chunks
    
    def _resize_traditional_chunks(self, chunks: List[ExtractedChunk], config: Dict[str, Any]) -> List[ExtractedChunk]:
        """Resize chunks for traditional models with smaller context windows"""
        max_size = config['max_chunk_size']
        min_size = config['min_chunk_size']
        optimized_chunks = []
        
        for chunk in chunks:
            content_length = len(chunk.content.strip())
            
            if content_length < min_size:
                # Too small - combine with next chunk if possible
                chunk.metadata['optimization'] = 'too_small'
                optimized_chunks.append(chunk)
                
            elif content_length > max_size:
                # Too large - split into smaller chunks
                split_chunks = self._split_large_chunk_smart(chunk, max_size)
                optimized_chunks.extend(split_chunks)
                
            else:
                # Good size - keep as is
                chunk.metadata['optimization'] = 'optimal'
                optimized_chunks.append(chunk)
        
        logger.info(f"🔄 Traditional chunk optimization: {len(chunks)} → {len(optimized_chunks)} chunks")
        return optimized_chunks
    
    def _split_large_chunk_smart(self, chunk: ExtractedChunk, max_size: int) -> List[ExtractedChunk]:
        """Smart splitting that respects sentence boundaries"""
        content = chunk.content
        sentences = self._split_into_sentences(content)
        
        sub_chunks = []
        current_chunk_text = ""
        current_sentences = []
        
        for sentence in sentences:
            potential_length = len(current_chunk_text) + len(sentence) + 1
            
            if potential_length > max_size and current_chunk_text:
                # Create sub-chunk
                sub_chunk = ExtractedChunk(
                    content=current_chunk_text.strip(),
                    metadata={
                        **chunk.metadata,
                        'parent_chunk_id': chunk.chunk_id,
                        'sub_chunk_index': len(sub_chunks),
                        'sentences': current_sentences,
                        'optimization': 'split_large'
                    },
                    quality_score=chunk.quality_score
                )
                sub_chunks.append(sub_chunk)
                
                # Reset for next chunk
                current_chunk_text = sentence
                current_sentences = [sentence]
            else:
                current_chunk_text += (" " + sentence if current_chunk_text else sentence)
                current_sentences.append(sentence)
        
        # Add final chunk if there's remaining content
        if current_chunk_text.strip():
            sub_chunk = ExtractedChunk(
                content=current_chunk_text.strip(),
                metadata={
                    **chunk.metadata,
                    'parent_chunk_id': chunk.chunk_id,
                    'sub_chunk_index': len(sub_chunks),
                    'sentences': current_sentences,
                    'optimization': 'split_large'
                },
                quality_score=chunk.quality_score
            )
            sub_chunks.append(sub_chunk)
        
        return sub_chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for better processing"""
        import re
        
        # Simple sentence splitting - can be enhanced with more sophisticated methods
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        return sentences

# Singleton instance
_chunk_sizer: Optional[DynamicChunkSizer] = None

def get_dynamic_chunk_sizer() -> DynamicChunkSizer:
    """Get or create dynamic chunk sizer singleton"""
    global _chunk_sizer
    if _chunk_sizer is None:
        _chunk_sizer = DynamicChunkSizer()
    return _chunk_sizer

def get_optimal_chunk_config() -> Dict[str, Any]:
    """Get optimal chunk configuration for current model"""
    sizer = get_dynamic_chunk_sizer()
    return sizer.get_chunk_configuration()

def optimize_chunks_for_model(chunks: List[ExtractedChunk]) -> List[ExtractedChunk]:
    """Optimize chunks for the current model's capabilities"""
    sizer = get_dynamic_chunk_sizer()
    return sizer.optimize_chunks(chunks)