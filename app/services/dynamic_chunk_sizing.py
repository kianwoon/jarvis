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
    """Dynamically sizes chunks based on model capabilities - NO HARDCODING"""
    
    # NO HARDCODED CONTEXT LIMITS - EVERYTHING MUST COME FROM DATABASE/CONFIG
    # This is only used as last resort when no context_length is configured
    ABSOLUTE_MINIMUM_CONTEXT = 4096  # Absolute minimum for any model
    
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
        """Determine context limit DYNAMICALLY from database/config - NO HARDCODING"""
        
        # ARCHITECTURAL PRINCIPLE: Always use configured context_length from database
        # This ensures we adapt to ANY model without code changes
        
        # PRIORITY 1: Check model_config section for context_length
        model_config = self.kg_settings.get('model_config', {})
        if 'context_length' in model_config:
            context_length = int(model_config['context_length'])
            logger.info(f"✅ Using DYNAMIC context_length from model_config: {context_length:,} tokens")
            return context_length
        
        # PRIORITY 2: Check direct settings for context_length
        if 'context_length' in self.kg_settings:
            context_length = int(self.kg_settings['context_length'])
            logger.info(f"✅ Using DYNAMIC context_length from settings: {context_length:,} tokens")
            return context_length
        
        # PRIORITY 3: Try to get from LLM settings if available
        try:
            from app.core.llm_settings_cache import get_llm_settings
            llm_settings = get_llm_settings()
            
            # Check for model-specific configuration
            model_configs = llm_settings.get('model_configs', {})
            if self.model_name in model_configs:
                model_specific = model_configs[self.model_name]
                if 'context_length' in model_specific:
                    context_length = int(model_specific['context_length'])
                    logger.info(f"✅ Using DYNAMIC context_length from LLM model config: {context_length:,} tokens")
                    return context_length
            
            # Check global LLM settings
            if 'context_length' in llm_settings:
                context_length = int(llm_settings['context_length'])
                logger.info(f"✅ Using DYNAMIC context_length from global LLM settings: {context_length:,} tokens")
                return context_length
        except Exception as e:
            logger.debug(f"Could not load LLM settings for context length: {e}")
        
        # CRITICAL: If no context_length is configured, this is a configuration error
        logger.error(f"❌ NO CONTEXT_LENGTH CONFIGURED for model {self.model_name}!")
        logger.error(f"   Available settings keys: {list(self.kg_settings.keys())}")
        if model_config:
            logger.error(f"   model_config keys: {list(model_config.keys())}")
        
        # Use absolute minimum as emergency fallback
        logger.warning(f"⚠️  Using ABSOLUTE MINIMUM context: {self.ABSOLUTE_MINIMUM_CONTEXT} tokens")
        logger.warning(f"   CONFIGURE context_length in settings to use model's full capacity!")
        return self.ABSOLUTE_MINIMUM_CONTEXT
    
    def _calculate_optimal_chunk_size(self) -> int:
        """Calculate optimal chunk size DYNAMICALLY based on actual model context"""
        # DYNAMIC CALCULATION - scales with ANY model context size
        
        # Token to character conversion (approximate)
        chars_per_token = 4  # 1 token ≈ 4 characters on average
        
        # Dynamic overhead calculation based on context size
        # Smaller models need more conservative overhead
        # Larger models can use more of their context
        if self.context_limit >= 200000:  # 200k+ tokens
            prompt_overhead_ratio = 0.05  # Only 5% overhead for large models
            safety_margin = 0.85  # Use 85% of context
        elif self.context_limit >= 100000:  # 100k-200k tokens
            prompt_overhead_ratio = 0.08  # 8% overhead
            safety_margin = 0.80  # Use 80% of context
        elif self.context_limit >= 50000:  # 50k-100k tokens
            prompt_overhead_ratio = 0.10  # 10% overhead
            safety_margin = 0.75  # Use 75% of context
        else:  # < 50k tokens
            prompt_overhead_ratio = 0.15  # 15% overhead for small models
            safety_margin = 0.70  # Use 70% of context
        
        # Calculate available tokens dynamically
        prompt_overhead = int(self.context_limit * prompt_overhead_ratio)
        available_tokens = (self.context_limit - prompt_overhead) * safety_margin
        optimal_chars = int(available_tokens * chars_per_token)
        
        # DYNAMIC BOUNDS - scale with model capacity
        # Minimum chunk size: 0.5% of context or 1KB, whichever is larger
        min_chunk_size = max(1000, int(self.context_limit * chars_per_token * 0.005))
        
        # Maximum chunk size: Just use the calculated optimal size
        # NO ARTIFICIAL LIMITS - let the model use its full capacity
        max_chunk_size = optimal_chars
        
        logger.info(f"📊 Dynamic chunk calculation:")
        logger.info(f"   Model context: {self.context_limit:,} tokens")
        logger.info(f"   Overhead ratio: {prompt_overhead_ratio:.1%}")
        logger.info(f"   Safety margin: {safety_margin:.1%}")
        logger.info(f"   Available tokens: {available_tokens:,.0f}")
        logger.info(f"   Optimal chunk size: {optimal_chars:,} chars")
        logger.info(f"   Min chunk: {min_chunk_size:,} chars")
        logger.info(f"   Max chunk: {max_chunk_size:,} chars")
        
        return max(min_chunk_size, optimal_chars)
    
    def should_use_large_chunks(self) -> bool:
        """Determine if large chunks should be used - DYNAMIC based on context"""
        # DYNAMIC THRESHOLD: Models with 32K+ context can handle large chunks
        # This scales automatically with any model
        return self.context_limit >= 32768
    
    def get_chunk_configuration(self) -> Dict[str, Any]:
        """Get chunk configuration - FULLY DYNAMIC based on model context"""
        
        # DYNAMIC STRATEGY SELECTION based on context size
        if self.context_limit >= 200000:  # 200k+ tokens
            strategy = 'full_context_utilization'
            target_chunks = 1  # Try to fit entire document in one chunk
            max_consolidation = 1000  # Unlimited consolidation
            preserve_granularity = False
            min_chunk_size = 10000  # 10KB minimum for very large models
        elif self.context_limit >= 100000:  # 100k-200k tokens
            strategy = 'aggressive_consolidation'
            target_chunks = 2  # Aim for 2 large chunks
            max_consolidation = 50
            preserve_granularity = False
            min_chunk_size = 5000  # 5KB minimum
        elif self.context_limit >= 50000:  # 50k-100k tokens
            strategy = 'balanced_large_context'
            target_chunks = 3  # Aim for 3-4 chunks
            max_consolidation = 20
            preserve_granularity = True
            min_chunk_size = 3000  # 3KB minimum
        elif self.context_limit >= 32768:  # 32k-50k tokens
            strategy = 'moderate_consolidation'
            target_chunks = 5  # Aim for 5-6 chunks
            max_consolidation = 10
            preserve_granularity = True
            min_chunk_size = 2000  # 2KB minimum
        else:  # < 32k tokens
            strategy = 'traditional'
            target_chunks = 10  # Traditional smaller chunks
            max_consolidation = 2
            preserve_granularity = True
            min_chunk_size = 1000  # 1KB minimum
        
        # DYNAMIC OVERLAP based on chunk size
        chunk_overlap = min(1000, int(self.optimal_chunk_size * 0.1))  # 10% overlap, max 1000 chars
        
        config = {
            'max_chunk_size': self.optimal_chunk_size,  # Use calculated optimal size
            'min_chunk_size': min_chunk_size,
            'chunk_overlap': chunk_overlap,
            'processing_strategy': strategy,
            'combine_small_chunks': self.context_limit >= 32768,  # Combine for 32k+ models
            'max_chunks_per_call': 1,  # Always process one chunk at a time
            'enable_document_level_processing': self.context_limit >= 100000,  # 100k+ can do full docs
            'preserve_granularity': preserve_granularity,
            'target_chunks_per_document': target_chunks,
            'max_consolidation_ratio': max_consolidation,
            # Additional dynamic parameters
            'context_utilization_target': 0.8 if self.context_limit >= 100000 else 0.7,
            'enable_intelligent_splitting': True,
            'respect_natural_boundaries': True,
            'dynamic_overlap_adjustment': True
        }
        
        logger.info(f"🎯 Dynamic chunk configuration for {self.context_limit:,} token model:")
        logger.info(f"   Strategy: {strategy}")
        logger.info(f"   Max chunk: {config['max_chunk_size']:,} chars")
        logger.info(f"   Target chunks per doc: {target_chunks}")
        logger.info(f"   Max consolidation ratio: {max_consolidation}:1")
        
        return config
    
    def optimize_chunks(self, chunks: List[ExtractedChunk]) -> List[ExtractedChunk]:
        """Optimize chunk sizes based on model capabilities - FULLY DYNAMIC"""
        config = self.get_chunk_configuration()
        strategy = config['processing_strategy']
        
        logger.info(f"🔄 Optimizing {len(chunks)} chunks using '{strategy}' strategy")
        
        # DYNAMIC STRATEGY ROUTING
        if strategy == 'full_context_utilization':
            # For 200k+ models: Use entire context capacity
            return self._create_full_context_chunks(chunks, config)
        elif strategy == 'aggressive_consolidation':
            # For 100k-200k models: Aggressive but not full consolidation
            return self._create_aggressive_chunks(chunks, config)
        elif strategy in ['balanced_large_context', 'moderate_consolidation']:
            # For 32k-100k models: Balanced approach
            return self._create_balanced_chunks(chunks, config)
        else:  # 'traditional'
            # For <32k models: Traditional chunking
            return self._resize_traditional_chunks(chunks, config)
    
    def _create_full_context_chunks(self, chunks: List[ExtractedChunk], config: Dict[str, Any]) -> List[ExtractedChunk]:
        """Create minimal chunks for 256k+ context models - USE FULL CAPACITY"""
        if not chunks:
            return chunks
        
        max_size = config['max_chunk_size']
        overlap = config['chunk_overlap']
        
        # Calculate total content size
        total_content_size = sum(len(chunk.content) for chunk in chunks)
        
        logger.info(f"🚀 FULL CONTEXT UTILIZATION MODE:")
        logger.info(f"   Model context: {self.context_limit:,} tokens (~{self.context_limit * 4:,} chars)")
        logger.info(f"   Document size: {total_content_size:,} chars")
        logger.info(f"   Max chunk size: {max_size:,} chars")
        
        # If the entire document fits in one chunk, DO IT
        if total_content_size <= max_size:
            logger.info(f"✅ ENTIRE DOCUMENT FITS IN ONE CHUNK! Using full context capacity.")
            # Combine ALL chunks into one massive chunk
            combined_content = "\n\n".join(chunk.content.strip() for chunk in chunks)
            
            mega_chunk = ExtractedChunk(
                content=combined_content,
                metadata={
                    **chunks[0].metadata,
                    'chunk_id': 'full_document_chunk',
                    'combined_from': [c.chunk_id for c in chunks],
                    'optimization': 'full_context_utilization',
                    'original_chunk_count': len(chunks),
                    'utilization_ratio': f"{(total_content_size / (self.context_limit * 4)):.1%}"
                },
                quality_score=sum(c.quality_score for c in chunks) / len(chunks) if chunks else 1.0
            )
            
            logger.info(f"✅ Created 1 MEGA CHUNK using {(total_content_size / (self.context_limit * 4)):.1%} of model capacity")
            return [mega_chunk]
        
        # If document is larger than max chunk size, create minimal chunks
        optimized_chunks = []
        current_content = ""
        chunk_sources = []
        
        for chunk in chunks:
            chunk_content = chunk.content.strip()
            potential_size = len(current_content) + len(chunk_content) + overlap
            
            if potential_size > max_size and current_content:
                # Create a massive chunk
                optimized_chunk = ExtractedChunk(
                    content=current_content.strip(),
                    metadata={
                        **chunk.metadata,
                        'chunk_id': f"full_context_chunk_{len(optimized_chunks)}",
                        'combined_from': chunk_sources,
                        'optimization': 'full_context_utilization',
                        'original_chunk_count': len(chunk_sources)
                    },
                    quality_score=sum(c.quality_score for c in chunks[:len(chunk_sources)]) / len(chunk_sources) if chunk_sources else 1.0
                )
                optimized_chunks.append(optimized_chunk)
                
                # Start new chunk with overlap
                current_content = current_content[-overlap:] + "\n\n" + chunk_content if overlap > 0 else chunk_content
                chunk_sources = [chunk.chunk_id]
            else:
                # Add to current chunk
                current_content += ("\n\n" + chunk_content) if current_content else chunk_content
                chunk_sources.append(chunk.chunk_id)
        
        # Add final chunk
        if current_content.strip():
            optimized_chunk = ExtractedChunk(
                content=current_content.strip(),
                metadata={
                    **chunks[-1].metadata,
                    'chunk_id': f"full_context_chunk_{len(optimized_chunks)}",
                    'combined_from': chunk_sources,
                    'optimization': 'full_context_utilization',
                    'original_chunk_count': len(chunk_sources)
                },
                quality_score=sum(c.quality_score for c in chunks[-len(chunk_sources):]) / len(chunk_sources) if chunk_sources else 1.0
            )
            optimized_chunks.append(optimized_chunk)
        
        consolidation_ratio = len(chunks) / len(optimized_chunks) if optimized_chunks else 1
        logger.info(f"✅ Full context optimization: {len(chunks)} → {len(optimized_chunks)} chunks (ratio: {consolidation_ratio:.1f}:1)")
        logger.info(f"   Average chunk size: {sum(len(c.content) for c in optimized_chunks) // len(optimized_chunks):,} chars")
        
        return optimized_chunks
    
    def _create_aggressive_chunks(self, chunks: List[ExtractedChunk], config: Dict[str, Any]) -> List[ExtractedChunk]:
        """Create aggressively consolidated chunks for 100k-200k context models"""
        if not chunks:
            return chunks
        
        max_size = config['max_chunk_size']
        target_chunks = config['target_chunks_per_document']
        
        # Calculate total content size
        total_content_size = sum(len(chunk.content) for chunk in chunks)
        
        logger.info(f"🚀 AGGRESSIVE CONSOLIDATION MODE:")
        logger.info(f"   Model context: {self.context_limit:,} tokens (~{self.context_limit * 4:,} chars)")
        logger.info(f"   Document size: {total_content_size:,} chars")
        logger.info(f"   Target chunks: {target_chunks}")
        
        # If entire document fits in max_size, return as single chunk
        if total_content_size <= max_size:
            combined_content = "\n\n".join(chunk.content.strip() for chunk in chunks)
            
            mega_chunk = ExtractedChunk(
                content=combined_content,
                metadata={
                    **chunks[0].metadata,
                    'chunk_id': 'aggressive_full_document',
                    'combined_from': [c.chunk_id for c in chunks],
                    'optimization': 'aggressive_consolidation',
                    'original_chunk_count': len(chunks),
                    'utilization_ratio': f"{(total_content_size / (self.context_limit * 4)):.1%}"
                },
                quality_score=sum(c.quality_score for c in chunks) / len(chunks) if chunks else 1.0
            )
            
            logger.info(f"✅ Created 1 LARGE CHUNK using {(total_content_size / (self.context_limit * 4)):.1%} of model capacity")
            return [mega_chunk]
        
        # Otherwise, create target number of chunks
        target_chunk_size = total_content_size // target_chunks
        working_max_size = min(max_size, max(target_chunk_size, config['min_chunk_size']))
        
        return self._consolidate_to_target_size(chunks, working_max_size, config)
    
    def _create_balanced_chunks(self, chunks: List[ExtractedChunk], config: Dict[str, Any]) -> List[ExtractedChunk]:
        """Create balanced chunks that maintain granularity while utilizing large context - FIXED consolidation"""
        if not chunks:
            return chunks
        
        max_size = config['max_chunk_size']
        min_size = config['min_chunk_size']
        overlap = config['chunk_overlap']
        target_chunk_count = config.get('target_chunks_per_document', 5)
        max_consolidation_ratio = config.get('max_consolidation_ratio', 5)
        
        # Calculate total content size
        total_content_size = sum(len(chunk.content) for chunk in chunks)
        
        # CRITICAL FIX: Don't over-consolidate if we already have reasonable-sized chunks
        if len(chunks) <= target_chunk_count:
            logger.info(f"🎯 Already have {len(chunks)} chunks (target: {target_chunk_count}), minimal consolidation")
            return self._minimal_consolidation(chunks, config)
        
        # Calculate optimal chunk size to reach target count
        optimal_size_for_target = total_content_size // target_chunk_count
        working_max_size = min(max_size, max(optimal_size_for_target, min_size))
        
        logger.info(f"🔄 Balanced chunking: {len(chunks)} → target ~{target_chunk_count} chunks")
        logger.info(f"   Total content: {total_content_size:,} chars, working max size: {working_max_size:,}")
        
        optimized_chunks = []
        current_content = ""
        current_metadata = chunks[0].metadata.copy()
        chunk_sources = []
        chunks_consolidated = 0
        
        for i, chunk in enumerate(chunks):
            chunk_content = chunk.content.strip()
            
            # Check if adding this chunk would exceed the working limit
            potential_size = len(current_content) + len(chunk_content) + overlap
            
            # ANTI-CONSOLIDATION CHECK: Don't consolidate too many chunks
            would_exceed_consolidation = chunks_consolidated >= max_consolidation_ratio
            
            if (potential_size > working_max_size or would_exceed_consolidation) and current_content:
                # Create the balanced chunk
                optimized_chunk = ExtractedChunk(
                    content=current_content.strip(),
                    metadata={
                        **current_metadata,
                        'chunk_id': f"balanced_chunk_{len(optimized_chunks)}",
                        'combined_from': chunk_sources,
                        'optimization': 'balanced_large_context',
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
                chunks_consolidated = 1
            else:
                # Add to current chunk
                if current_content:
                    current_content += "\n\n" + chunk_content
                else:
                    current_content = chunk_content
                chunk_sources.append(chunk.chunk_id)
                chunks_consolidated += 1
        
        # Add final chunk if there's remaining content
        if current_content.strip():
            optimized_chunk = ExtractedChunk(
                content=current_content.strip(),
                metadata={
                    **current_metadata,
                    'chunk_id': f"balanced_chunk_{len(optimized_chunks)}",
                    'combined_from': chunk_sources,
                    'optimization': 'balanced_large_context',
                    'original_chunk_count': len(chunk_sources)
                },
                quality_score=sum(c.quality_score for c in chunks[-len(chunk_sources):]) / len(chunk_sources) if chunk_sources else 1.0
            )
            optimized_chunks.append(optimized_chunk)
        
        actual_ratio = len(chunks) / len(optimized_chunks) if optimized_chunks else 1
        
        logger.info(f"✅ Balanced chunk optimization: {len(chunks)} → {len(optimized_chunks)} chunks (ratio: {actual_ratio:.1f}:1)")
        logger.info(f"   Average chunk size: {sum(len(c.content) for c in optimized_chunks) // len(optimized_chunks):,} chars")
        
        return optimized_chunks
    
    def _minimal_consolidation(self, chunks: List[ExtractedChunk], config: Dict[str, Any]) -> List[ExtractedChunk]:
        """Minimal consolidation for chunks that are already reasonable size"""
        min_size = config['min_chunk_size']
        max_size = config['max_chunk_size']
        
        optimized_chunks = []
        i = 0
        
        while i < len(chunks):
            chunk = chunks[i]
            content_length = len(chunk.content.strip())
            
            if content_length < min_size and i + 1 < len(chunks):
                # Only combine if current chunk is too small
                next_chunk = chunks[i + 1]
                combined_content = chunk.content + "\n\n" + next_chunk.content
                
                if len(combined_content) <= max_size:
                    # Combine these two chunks
                    combined_chunk = ExtractedChunk(
                        content=combined_content,
                        metadata={
                            **chunk.metadata,
                            'chunk_id': f"minimal_combined_{len(optimized_chunks)}",
                            'combined_from': [chunk.chunk_id, next_chunk.chunk_id],
                            'optimization': 'minimal_consolidation',
                            'original_chunk_count': 2
                        },
                        quality_score=(chunk.quality_score + next_chunk.quality_score) / 2
                    )
                    optimized_chunks.append(combined_chunk)
                    i += 2  # Skip next chunk as it's been combined
                else:
                    # Can't combine, keep as is
                    chunk.metadata['optimization'] = 'kept_as_is'
                    optimized_chunks.append(chunk)
                    i += 1
            else:
                # Chunk is good size, keep as is
                chunk.metadata['optimization'] = 'kept_as_is'
                optimized_chunks.append(chunk)
                i += 1
        
        logger.info(f"🔧 Minimal consolidation: {len(chunks)} → {len(optimized_chunks)} chunks")
        return optimized_chunks
    
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
    
    def _consolidate_to_target_size(self, chunks: List[ExtractedChunk], target_size: int, config: Dict[str, Any]) -> List[ExtractedChunk]:
        """Consolidate chunks to reach target size - DYNAMIC helper method"""
        optimized_chunks = []
        current_content = ""
        chunk_sources = []
        overlap = config.get('chunk_overlap', 0)
        
        for chunk in chunks:
            chunk_content = chunk.content.strip()
            potential_size = len(current_content) + len(chunk_content) + (overlap if current_content else 0)
            
            if potential_size > target_size and current_content:
                # Create consolidated chunk
                optimized_chunk = ExtractedChunk(
                    content=current_content.strip(),
                    metadata={
                        **chunk.metadata,
                        'chunk_id': f"consolidated_chunk_{len(optimized_chunks)}",
                        'combined_from': chunk_sources,
                        'optimization': 'dynamic_consolidation',
                        'original_chunk_count': len(chunk_sources)
                    },
                    quality_score=sum(c.quality_score for c in chunks[:len(chunk_sources)]) / len(chunk_sources) if chunk_sources else 1.0
                )
                optimized_chunks.append(optimized_chunk)
                
                # Start new chunk with overlap if configured
                if overlap > 0 and len(current_content) > overlap:
                    current_content = current_content[-overlap:] + "\n\n" + chunk_content
                else:
                    current_content = chunk_content
                
                chunk_sources = [chunk.chunk_id]
            else:
                # Add to current chunk
                if current_content:
                    current_content += "\n\n" + chunk_content
                else:
                    current_content = chunk_content
                chunk_sources.append(chunk.chunk_id)
        
        # Add final chunk
        if current_content.strip():
            optimized_chunk = ExtractedChunk(
                content=current_content.strip(),
                metadata={
                    **chunks[-1].metadata,
                    'chunk_id': f"consolidated_chunk_{len(optimized_chunks)}",
                    'combined_from': chunk_sources,
                    'optimization': 'dynamic_consolidation',
                    'original_chunk_count': len(chunk_sources)
                },
                quality_score=sum(c.quality_score for c in chunks[-len(chunk_sources):]) / len(chunk_sources) if chunk_sources else 1.0
            )
            optimized_chunks.append(optimized_chunk)
        
        return optimized_chunks

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