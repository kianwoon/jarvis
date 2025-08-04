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
        """Get the current model from LLM settings (primary) or knowledge graph settings (fallback)"""
        # PRIORITY 1: Check LLM settings for main_llm model (most reliable source)
        try:
            from app.core.llm_settings_cache import get_llm_settings
            llm_settings = get_llm_settings()
            
            # Check main_llm configuration (primary location)
            main_llm = llm_settings.get('main_llm', {})
            if 'model' in main_llm:
                model = main_llm['model']
                logger.debug(f"🎯 Found model in main_llm config: {model}")
                return str(model).lower()
            
            # Check knowledge_graph configuration in LLM settings
            kg_llm = llm_settings.get('knowledge_graph', {})
            if 'model' in kg_llm:
                model = kg_llm['model']
                logger.debug(f"🎯 Found model in knowledge_graph LLM config: {model}")
                return str(model).lower()
                
        except Exception as e:
            logger.debug(f"Could not load LLM settings for model: {e}")
        
        # PRIORITY 2: Check the model_config section in KG settings
        model_config = self.kg_settings.get('model_config', {})
        if 'model' in model_config:
            model = model_config['model']
            logger.debug(f"🎯 Found model in KG model_config: {model}")
            return str(model).lower()
        
        # PRIORITY 3: Check direct model field in KG settings
        if 'model' in self.kg_settings:
            model = self.kg_settings['model']
            logger.debug(f"🎯 Found model in direct KG settings: {model}")
            return str(model).lower()
        
        # PRIORITY 4: Check for legacy locations
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
        logger.warning(f"⚠️  Model not found in any configuration. KG settings keys: {list(self.kg_settings.keys())}")
        if model_config:
            logger.warning(f"⚠️  model_config keys: {list(model_config.keys())}")
        
        return 'unknown'
    
    def _determine_context_limit(self) -> int:
        """Determine context limit DYNAMICALLY from database/config - NO HARDCODING"""
        
        # ARCHITECTURAL PRINCIPLE: Always use configured context_length from database
        # This ensures we adapt to ANY model without code changes
        
        # PRIORITY 1: Check LLM settings for main_llm context_length (most reliable source)
        try:
            from app.core.llm_settings_cache import get_llm_settings
            llm_settings = get_llm_settings()
            
            # Check main_llm configuration (primary location for 256k models)
            main_llm = llm_settings.get('main_llm', {})
            if 'context_length' in main_llm:
                context_length = int(main_llm['context_length'])
                logger.info(f"✅ Using DYNAMIC context_length from main_llm config: {context_length:,} tokens")
                return context_length
            
            # Check knowledge_graph configuration in LLM settings
            kg_llm = llm_settings.get('knowledge_graph', {})
            if 'context_length' in kg_llm:
                context_length = int(kg_llm['context_length'])
                logger.info(f"✅ Using DYNAMIC context_length from knowledge_graph LLM config: {context_length:,} tokens")
                return context_length
            
            # Check for model-specific configuration
            model_configs = llm_settings.get('model_configs', {})
            if self.model_name in model_configs:
                model_specific = model_configs[self.model_name]
                if 'context_length' in model_specific:
                    context_length = int(model_specific['context_length'])
                    logger.info(f"✅ Using DYNAMIC context_length from LLM model-specific config: {context_length:,} tokens")
                    return context_length
            
            # Check global LLM settings fallback
            if 'context_length' in llm_settings:
                context_length = int(llm_settings['context_length'])
                logger.info(f"✅ Using DYNAMIC context_length from global LLM settings: {context_length:,} tokens")
                return context_length
                
        except Exception as e:
            logger.debug(f"Could not load LLM settings for context length: {e}")
        
        # PRIORITY 2: Check KG model_config section for context_length
        model_config = self.kg_settings.get('model_config', {})
        if 'context_length' in model_config:
            context_length = int(model_config['context_length'])
            logger.info(f"✅ Using DYNAMIC context_length from KG model_config: {context_length:,} tokens")
            return context_length
        
        # PRIORITY 3: Check direct KG settings for context_length
        if 'context_length' in self.kg_settings:
            context_length = int(self.kg_settings['context_length'])
            logger.info(f"✅ Using DYNAMIC context_length from KG direct settings: {context_length:,} tokens")
            return context_length
        
        # CRITICAL: If no context_length is configured, this is a configuration error
        logger.error(f"❌ NO CONTEXT_LENGTH CONFIGURED for model {self.model_name}!")
        logger.error(f"   Available KG settings keys: {list(self.kg_settings.keys())}")
        if model_config:
            logger.error(f"   model_config keys: {list(model_config.keys())}")
        
        # Use absolute minimum as emergency fallback
        logger.warning(f"⚠️  Using ABSOLUTE MINIMUM context: {self.ABSOLUTE_MINIMUM_CONTEXT} tokens")
        logger.warning(f"   CONFIGURE context_length in settings to use model's full capacity!")
        return self.ABSOLUTE_MINIMUM_CONTEXT
    
    def _calculate_optimal_chunk_size(self) -> int:
        """Calculate optimal chunk size using 50% MODEL UTILIZATION RULE"""
        # **50% MODEL UTILIZATION RULE**: Use exactly 50% of model context length as target chunk size
        # This is simple, predictable, and ensures we use the model's capacity efficiently
        
        # Token to character conversion (approximate)
        chars_per_token = 4  # 1 token ≈ 4 characters on average
        
        # **CORE PRINCIPLE**: 50% of model context = target chunk size
        target_utilization = 0.5  # Use exactly 50% of model context
        target_tokens = int(self.context_limit * target_utilization)
        target_chunk_size = int(target_tokens * chars_per_token)
        
        # Minimum chunk size: 1KB (safety floor)
        min_chunk_size = 1000
        
        # Use 50% rule as the optimal size
        optimal_chars = max(min_chunk_size, target_chunk_size)
        
        logger.info(f"📊 50% Model Utilization Calculation:")
        logger.info(f"   Model context: {self.context_limit:,} tokens")
        logger.info(f"   Target utilization: {target_utilization:.0%}")
        logger.info(f"   Target tokens: {target_tokens:,} tokens")
        logger.info(f"   Target chunk size: {optimal_chars:,} chars")
        logger.info(f"   Min chunk size: {min_chunk_size:,} chars")
        
        return optimal_chars
    
    def should_use_large_chunks(self) -> bool:
        """Always use 50% model utilization - no complex logic needed"""
        return True
    
    def get_chunk_configuration(self, document_type: str = 'general', processing_purpose: str = 'knowledge_graph') -> Dict[str, Any]:
        """Get chunk configuration using 50% MODEL UTILIZATION RULE - SIMPLE AND PREDICTABLE
        
        **CORE LOGIC**:
        - Use 50% of model context as target chunk size
        - If document fits in 50% capacity → Single chunk
        - If document exceeds 50% → Split into multiple 50% chunks
        - No complex business logic, no document type overrides
        
        Args:
            document_type: Type of document (preserved for compatibility but not used)
            processing_purpose: Purpose of processing (preserved for compatibility but not used)
        """
        
        # **50% MODEL UTILIZATION RULE** - Simple and consistent
        target_chunk_size = self.optimal_chunk_size  # Already calculated as 50% of model context
        min_chunk_size = 1000  # 1KB minimum
        chunk_overlap = int(target_chunk_size * 0.1)  # 10% overlap
        
        # **SIMPLE STRATEGY**: No complex logic, just use 50% chunks
        strategy = 'fifty_percent_utilization'
        
        config = {
            'max_chunk_size': target_chunk_size,
            'min_chunk_size': min_chunk_size,
            'chunk_overlap': chunk_overlap,
            'processing_strategy': strategy,
            'combine_small_chunks': True,  # Always combine small chunks
            'max_chunks_per_call': 1,  # Process one chunk at a time
            'enable_document_level_processing': True,  # Always try single chunk first
            'preserve_granularity': False,  # Prioritize efficiency over granularity
            'target_chunks_per_document': 1,  # Always aim for single chunk if possible
            'max_consolidation_ratio': 1000,  # Unlimited consolidation
            # 50% utilization parameters
            'context_utilization_target': 0.5,  # Exactly 50%
            'enable_intelligent_splitting': True,
            'respect_natural_boundaries': True,
            'dynamic_overlap_adjustment': False  # Keep overlap fixed at 10%
        }
        
        logger.info(f"🎯 50% Model Utilization Configuration:")
        logger.info(f"   Model context: {self.context_limit:,} tokens")
        logger.info(f"   Target chunk size: {target_chunk_size:,} chars (50% utilization)")
        logger.info(f"   Min chunk size: {min_chunk_size:,} chars")
        logger.info(f"   Chunk overlap: {chunk_overlap:,} chars (10%)")
        logger.info(f"   Strategy: {strategy}")
        
        return config
    
    def optimize_chunks(self, chunks: List[ExtractedChunk], document_type: str = 'general', processing_purpose: str = 'knowledge_graph') -> List[ExtractedChunk]:
        """Optimize chunks using 50% MODEL UTILIZATION RULE - SIMPLE DECISION LOGIC"""
        config = self.get_chunk_configuration(document_type, processing_purpose)
        target_chunk_size = config['max_chunk_size']
        
        # Calculate total document size
        total_content_size = sum(len(chunk.content) for chunk in chunks)
        
        logger.info(f"🔄 50% Model Utilization Optimization:")
        logger.info(f"   Input chunks: {len(chunks)}")
        logger.info(f"   Total document size: {total_content_size:,} chars")
        logger.info(f"   Target chunk size: {target_chunk_size:,} chars (50% of {self.context_limit:,} tokens)")
        
        # **SIMPLE DECISION LOGIC**:
        # If document fits in 50% capacity → Single chunk
        # If document exceeds 50% → Split into multiple 50% chunks
        
        if total_content_size <= target_chunk_size:
            logger.info(f"✅ SINGLE CHUNK: Document fits in 50% model capacity")
            return self._create_single_mega_chunk(chunks, config)
        else:
            logger.info(f"✅ MULTIPLE CHUNKS: Document exceeds 50% capacity, splitting into {target_chunk_size:,} char chunks")
            return self._create_fifty_percent_chunks(chunks, config)
    
    def _create_single_mega_chunk(self, chunks: List[ExtractedChunk], config: Dict[str, Any]) -> List[ExtractedChunk]:
        """Create single chunk when document fits in 50% model capacity"""
        if not chunks:
            return chunks
        
        # Combine ALL chunks into one chunk
        combined_content = "\n\n".join(chunk.content.strip() for chunk in chunks)
        total_content_size = len(combined_content)
        
        mega_chunk = ExtractedChunk(
            content=combined_content,
            metadata={
                **chunks[0].metadata,
                'chunk_id': 'single_document_chunk',
                'combined_from': [c.chunk_id for c in chunks],
                'optimization': 'fifty_percent_single_chunk',
                'original_chunk_count': len(chunks),
                'utilization_ratio': f"{(total_content_size / (self.context_limit * 4)):.1%}"
            },
            quality_score=sum(c.quality_score for c in chunks) / len(chunks) if chunks else 1.0
        )
        
        logger.info(f"✅ Created 1 SINGLE CHUNK using {(total_content_size / (self.context_limit * 4)):.1%} of model capacity")
        return [mega_chunk]
    
    def _create_fifty_percent_chunks(self, chunks: List[ExtractedChunk], config: Dict[str, Any]) -> List[ExtractedChunk]:
        """Create multiple chunks when document exceeds 50% model capacity"""
        if not chunks:
            return chunks
        
        target_chunk_size = config['max_chunk_size']
        overlap = config['chunk_overlap']
        
        # First, combine all input chunks into one text
        all_content = "\n\n".join(chunk.content.strip() for chunk in chunks)
        
        # If total content is smaller than target, something went wrong in the decision logic
        if len(all_content) <= target_chunk_size:
            logger.warning("Content smaller than target but routed to splitting - using single chunk")
            return self._create_single_mega_chunk(chunks, config)
        
        # Split the content into target-sized chunks
        optimized_chunks = []
        start_pos = 0
        chunk_index = 0
        
        while start_pos < len(all_content):
            # Calculate end position for this chunk
            end_pos = start_pos + target_chunk_size
            
            # If this is not the last chunk and we have overlap, adjust for overlap
            if end_pos < len(all_content) and overlap > 0:
                # Find a good breaking point near the target size
                # Look for sentence ending within last 10% of chunk
                search_start = max(start_pos + int(target_chunk_size * 0.9), start_pos + 1000)
                search_end = min(end_pos, len(all_content))
                
                # Look for sentence boundaries
                for i in range(search_end - 1, search_start - 1, -1):
                    if all_content[i:i+1] in '.!?':
                        end_pos = i + 1
                        break
            
            # Extract chunk content
            chunk_content = all_content[start_pos:end_pos].strip()
            
            if chunk_content:  # Only create non-empty chunks
                optimized_chunk = ExtractedChunk(
                    content=chunk_content,
                    metadata={
                        **chunks[0].metadata,
                        'chunk_id': f"fifty_percent_chunk_{chunk_index}",
                        'combined_from': [c.chunk_id for c in chunks],
                        'optimization': 'fifty_percent_utilization',
                        'original_chunk_count': len(chunks),
                        'split_index': chunk_index
                    },
                    quality_score=sum(c.quality_score for c in chunks) / len(chunks) if chunks else 1.0
                )
                optimized_chunks.append(optimized_chunk)
                chunk_index += 1
            
            # Move to next chunk with overlap
            if end_pos >= len(all_content):
                break
            
            # Apply overlap for next chunk
            start_pos = max(start_pos + 1, end_pos - overlap)
        
        logger.info(f"✅ 50% utilization splitting: {len(all_content):,} chars → {len(optimized_chunks)} chunks")
        if optimized_chunks:
            avg_size = sum(len(c.content) for c in optimized_chunks) // len(optimized_chunks)
            logger.info(f"   Average chunk size: {avg_size:,} chars")
        
        return optimized_chunks
    
    # Removed complex chunking methods - using simple 50% utilization strategy only
    

# Singleton instance
_chunk_sizer: Optional[DynamicChunkSizer] = None

def get_dynamic_chunk_sizer() -> DynamicChunkSizer:
    """Get or create dynamic chunk sizer singleton"""
    global _chunk_sizer
    if _chunk_sizer is None:
        _chunk_sizer = DynamicChunkSizer()
    return _chunk_sizer

def get_optimal_chunk_config(document_type: str = 'general', processing_purpose: str = 'knowledge_graph') -> Dict[str, Any]:
    """Get optimal chunk configuration for current model, document type, and processing purpose"""
    sizer = get_dynamic_chunk_sizer()
    return sizer.get_chunk_configuration(document_type, processing_purpose)

def optimize_chunks_for_model(chunks: List[ExtractedChunk], document_type: str = 'general', processing_purpose: str = 'knowledge_graph') -> List[ExtractedChunk]:
    """Optimize chunks for the current model's capabilities, document type, and processing purpose"""
    sizer = get_dynamic_chunk_sizer()
    return sizer.optimize_chunks(chunks, document_type, processing_purpose)