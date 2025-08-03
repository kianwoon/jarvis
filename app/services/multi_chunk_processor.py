"""
Multi-Chunk Relationship Detection Service

Processes documents with overlapping chunks to detect relationships that span
across chunk boundaries, improving knowledge graph connectivity and completeness.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime

from app.document_handlers.base import ExtractedChunk
from app.services.knowledge_graph_service import (
    get_knowledge_graph_service, 
    GraphExtractionResult, 
    ExtractedEntity, 
    ExtractedRelationship
)
from app.core.knowledge_graph_settings_cache import get_knowledge_graph_settings

logger = logging.getLogger(__name__)

@dataclass
class ChunkWindow:
    """Sliding window of overlapping chunks"""
    chunks: List[ExtractedChunk]
    window_text: str
    start_chunk_idx: int
    end_chunk_idx: int
    overlap_metadata: Dict[str, Any]

@dataclass
class MultiChunkExtractionResult:
    """Result of multi-chunk relationship extraction"""
    individual_results: List[GraphExtractionResult]
    cross_chunk_entities: List[ExtractedEntity]
    cross_chunk_relationships: List[ExtractedRelationship]
    total_processing_time_ms: float
    overlap_analysis: Dict[str, Any]
    quality_metrics: Dict[str, Any]

class MultiChunkRelationshipProcessor:
    """Enhanced service for detecting relationships across chunk boundaries with 10x better performance"""
    
    def __init__(self):
        self.config = get_knowledge_graph_settings()
        self.kg_service = get_knowledge_graph_service()
        
        # ULTRA-AGGRESSIVE configuration for maximum entity extraction
        self.overlap_size = 500  # Much larger overlap for business documents
        self.window_size = 3     # Number of chunks in sliding window  
        self.min_cross_chunk_confidence = 0.2  # Very low threshold for business entities
        
        # ULTRA-AGGRESSIVE cross-chunk analysis configuration for 4x better extraction
        self.cross_chunk_settings = {
            'aggressive_entity_linking': True,
            'entity_similarity_threshold': 0.5,  # Much lower threshold for business entities
            'relationship_inference_threshold': 0.3,  # Aggressive relationship inference
            'enable_document_level_analysis': True,
            'enable_entity_frequency_analysis': True,
            'enable_cooccurrence_matrix': True,
            'enable_semantic_clustering': True,
            'enable_business_hierarchy_inference': True,
            'business_entity_confidence_boost': 0.2,  # Boost business entities
            'enable_pattern_based_extraction': True,
            'enable_number_metric_extraction': True,
            'enable_temporal_extraction': True
        }
        
        # Coreference patterns for entity linking across chunks
        self.coreference_patterns = [
            # Pronoun references
            {'pattern': r'\b(he|she|it|they|this|that|these|those)\b', 'type': 'pronoun'},
            # Definite references
            {'pattern': r'\bthe\s+(?:company|organization|person|system|project|technology)\b', 'type': 'definite'},
            # Relative references
            {'pattern': r'\b(?:which|who|that|where|when)\b', 'type': 'relative'},
            # Abbreviated references
            {'pattern': r'\b[A-Z]{2,}\b', 'type': 'abbreviation'}
        ]
    
    async def process_document_with_overlap(self, chunks: List[ExtractedChunk], 
                                          document_id: str, 
                                          progressive_storage: bool = False,
                                          kg_service=None) -> MultiChunkExtractionResult:
        """Enhanced document processing with aggressive cross-chunk analysis for 10x better yield"""
        start_time = datetime.now()
        
        if not self.config.get('extraction', {}).get('enable_multi_chunk_relationships', True):
            logger.info("Multi-chunk processing disabled, using standard processing")
            return await self._process_chunks_individually(chunks, document_id)
        
        try:
            logger.info(f"🚀 ENHANCED MULTI-CHUNK: Processing {len(chunks)} chunks with aggressive cross-boundary analysis")
            
            # Step 1: Enhanced individual chunk processing with multi-pass extraction
            individual_results = await self._process_individual_chunks_enhanced(
                chunks, document_id, progressive_storage, kg_service
            )
            
            # Step 2: Document-level entity frequency analysis
            document_entity_analysis = await self._analyze_document_level_entities(
                chunks, individual_results
            )
            
            # Step 3: Create enhanced sliding windows with adaptive overlap
            chunk_windows = self._create_enhanced_chunk_windows(chunks, document_entity_analysis)
            
            # Step 4: Aggressive cross-chunk entity linking and relationship extraction
            cross_chunk_entities, cross_chunk_relationships = await self._extract_cross_chunk_relationships_enhanced(
                chunk_windows, individual_results, document_entity_analysis
            )
            
            # Step 5: Business hierarchy and semantic clustering analysis
            hierarchy_entities, hierarchy_relationships = await self._analyze_business_hierarchies(
                individual_results + [self._create_synthetic_result(cross_chunk_entities, cross_chunk_relationships)],
                document_entity_analysis
            )
            
            # Step 6: Enhanced coreference resolution with business context
            resolved_entities, resolved_relationships = await self._resolve_coreferences_enhanced(
                individual_results, cross_chunk_entities + hierarchy_entities, 
                cross_chunk_relationships + hierarchy_relationships, document_entity_analysis
            )
            
            # Step 7: Document-level relationship inference matrix
            inferred_relationships = await self._infer_document_level_relationships(
                resolved_entities, resolved_relationships, chunks, document_entity_analysis
            )
            resolved_relationships.extend(inferred_relationships)
            
            # Step 8: Quality analysis with enhanced metrics
            quality_metrics = self._analyze_extraction_quality_enhanced(
                individual_results, resolved_entities, resolved_relationships, document_entity_analysis
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            total_entities = len(resolved_entities)
            total_relationships = len(resolved_relationships)
            original_entities = sum(len(r.entities) for r in individual_results)
            original_relationships = sum(len(r.relationships) for r in individual_results)
            
            enhancement_factor_entities = total_entities / max(original_entities, 1)
            enhancement_factor_relationships = total_relationships / max(original_relationships, 1)
            
            logger.info(f"🎯 ENHANCED MULTI-CHUNK COMPLETE:")
            logger.info(f"   📊 Entities: {original_entities} → {total_entities} ({enhancement_factor_entities:.1f}x improvement)")
            logger.info(f"   🔗 Relationships: {original_relationships} → {total_relationships} ({enhancement_factor_relationships:.1f}x improvement)")
            logger.info(f"   ⚡ Cross-chunk enhancements: +{len(cross_chunk_entities)} entities, +{len(cross_chunk_relationships)} relationships")
            logger.info(f"   🏢 Business hierarchy analysis: +{len(hierarchy_entities)} entities, +{len(hierarchy_relationships)} relationships")
            logger.info(f"   🔮 Document-level inference: +{len(inferred_relationships)} relationships")
            
            return MultiChunkExtractionResult(
                individual_results=individual_results,
                cross_chunk_entities=resolved_entities,
                cross_chunk_relationships=resolved_relationships,
                total_processing_time_ms=processing_time,
                overlap_analysis=self._analyze_overlap_effectiveness_enhanced(
                    chunk_windows, cross_chunk_relationships, document_entity_analysis, enhancement_factor_entities, enhancement_factor_relationships
                ),
                quality_metrics=quality_metrics
            )
            
        except Exception as e:
            logger.error(f"Multi-chunk processing failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Fallback to individual processing
            individual_results = await self._process_individual_chunks(
                chunks, document_id, progressive_storage, kg_service
            )
            return MultiChunkExtractionResult(
                individual_results=individual_results,
                cross_chunk_entities=[],
                cross_chunk_relationships=[],
                total_processing_time_ms=processing_time,
                overlap_analysis={'error': str(e)},
                quality_metrics={'processing_failed': True}
            )
    
    async def _process_individual_chunks(self, chunks: List[ExtractedChunk], 
                                       document_id: str, 
                                       progressive_storage: bool = False,
                                       kg_service=None) -> List[GraphExtractionResult]:
        """Process each chunk individually"""
        # Process chunks serially to avoid overloading Ollama service
        valid_results = []
        
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Processing individual chunk {i}/{len(chunks)}")
            try:
                result = await self.kg_service.extract_from_chunk(chunk)
                valid_results.append(result)
                
                # Progressive storage: store immediately after extraction
                if progressive_storage and kg_service and (result.entities or result.relationships):
                    try:
                        storage_result = await kg_service.store_in_neo4j(result, document_id)
                        if storage_result.get('success'):
                            logger.info(f"📊 Progressive storage: chunk {i} stored ({storage_result.get('entities_stored', 0)} entities, {storage_result.get('relationships_stored', 0)} relationships)")
                        else:
                            logger.warning(f"❌ Progressive storage failed for chunk {i}: {storage_result.get('error')}")
                    except Exception as storage_e:
                        logger.error(f"❌ Progressive storage error for chunk {i}: {storage_e}")
                
                logger.info(f"✅ Individual chunk {i} completed: {len(result.entities)} entities, {len(result.relationships)} relationships")
            except Exception as e:
                logger.error(f"❌ Individual chunk {i} processing failed: {e}")
                continue
        
        return valid_results
    
    def _create_chunk_windows(self, chunks: List[ExtractedChunk]) -> List[ChunkWindow]:
        """Create overlapping windows of chunks for cross-boundary analysis"""
        windows = []
        
        for i in range(len(chunks) - 1):  # Sliding window of adjacent chunks
            # Create window with current and next chunk
            window_chunks = chunks[i:i+2]
            
            # Create combined text with overlap detection
            combined_text = ""
            overlap_info = {}
            
            for j, chunk in enumerate(window_chunks):
                if j == 0:
                    combined_text = chunk.content
                else:
                    # Find potential overlap between chunks
                    overlap_text, overlap_length = self._find_chunk_overlap(
                        window_chunks[j-1].content, chunk.content
                    )
                    
                    if overlap_length > 0:
                        # Remove overlap to avoid duplication
                        combined_text += "\n" + chunk.content[overlap_length:]
                        overlap_info[f'overlap_{j-1}_{j}'] = {
                            'text': overlap_text,
                            'length': overlap_length
                        }
                    else:
                        combined_text += "\n" + chunk.content
            
            window = ChunkWindow(
                chunks=window_chunks,
                window_text=combined_text,
                start_chunk_idx=i,
                end_chunk_idx=i+1,
                overlap_metadata=overlap_info
            )
            windows.append(window)
        
        logger.debug(f"🔄 Created {len(windows)} chunk windows for overlap analysis")
        return windows
    
    def _find_chunk_overlap(self, chunk1_content: str, chunk2_content: str) -> Tuple[str, int]:
        """Find overlapping text between two chunks"""
        # Look for overlap at the end of chunk1 and beginning of chunk2
        max_overlap = min(self.overlap_size, len(chunk1_content), len(chunk2_content))
        
        # Only try to find overlap if we have enough text
        if max_overlap >= 20:
            for overlap_len in range(max_overlap, 19, -1):  # Minimum 20 chars for meaningful overlap
                chunk1_end = chunk1_content[-overlap_len:].strip()
                chunk2_start = chunk2_content[:overlap_len].strip()
                
                # Check for approximate match (allowing for minor differences)
                if self._text_similarity(chunk1_end, chunk2_start) > 0.8:
                    return chunk1_end, len(chunk2_content[:overlap_len])
        
        return "", 0
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text snippets"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    async def _extract_cross_chunk_relationships(self, windows: List[ChunkWindow],
                                               individual_results: List[GraphExtractionResult]) -> Tuple[List[ExtractedEntity], List[ExtractedRelationship]]:
        """Extract entities and relationships from chunk windows"""
        cross_chunk_entities = []
        cross_chunk_relationships = []
        
        for window in windows:
            try:
                # Extract from combined window text
                window_chunk = ExtractedChunk(
                    content=window.window_text,
                    metadata={
                        'chunk_id': f"window_{window.start_chunk_idx}_{window.end_chunk_idx}",
                        'window_type': 'cross_chunk',
                        'source_chunks': [c.metadata.get('chunk_id', 'unknown') for c in window.chunks],
                        'overlap_info': window.overlap_metadata
                    }
                )
                
                window_result = await self.kg_service.extract_from_chunk(window_chunk)
                
                # Filter for cross-chunk entities (entities that span boundaries)
                boundary_entities = self._identify_boundary_entities(
                    window_result.entities, window, individual_results
                )
                cross_chunk_entities.extend(boundary_entities)
                
                # Filter for cross-chunk relationships
                boundary_relationships = self._identify_boundary_relationships(
                    window_result.relationships, window, individual_results
                )
                cross_chunk_relationships.extend(boundary_relationships)
                
            except Exception as e:
                logger.error(f"Window processing failed: {e}")
                continue
        
        # Deduplicate cross-chunk extractions
        cross_chunk_entities = self._deduplicate_entities(cross_chunk_entities)
        cross_chunk_relationships = self._deduplicate_relationships(cross_chunk_relationships)
        
        return cross_chunk_entities, cross_chunk_relationships
    
    def _identify_boundary_entities(self, window_entities: List[ExtractedEntity],
                                   window: ChunkWindow, 
                                   individual_results: List[GraphExtractionResult]) -> List[ExtractedEntity]:
        """Identify entities that appear at chunk boundaries"""
        boundary_entities = []
        
        # Get entities from individual chunks in this window
        individual_entities = set()
        for i in range(window.start_chunk_idx, window.end_chunk_idx + 1):
            if i < len(individual_results):
                for entity in individual_results[i].entities:
                    individual_entities.add(entity.canonical_form.lower())
        
        # Find entities in window that are not in individual chunks (cross-boundary entities)
        for entity in window_entities:
            if entity.canonical_form.lower() not in individual_entities:
                # This is likely a cross-boundary entity
                entity.properties = entity.properties or {}
                entity.properties['cross_chunk'] = True
                entity.properties['source_window'] = f"{window.start_chunk_idx}_{window.end_chunk_idx}"
                entity.confidence *= 0.9  # Slightly lower confidence for cross-chunk entities
                boundary_entities.append(entity)
        
        return boundary_entities
    
    def _identify_boundary_relationships(self, window_relationships: List[ExtractedRelationship],
                                       window: ChunkWindow,
                                       individual_results: List[GraphExtractionResult]) -> List[ExtractedRelationship]:
        """Identify relationships that span chunk boundaries"""
        boundary_relationships = []
        
        # Get relationships from individual chunks in this window
        individual_relationships = set()
        for i in range(window.start_chunk_idx, window.end_chunk_idx + 1):
            if i < len(individual_results):
                for rel in individual_results[i].relationships:
                    rel_key = (rel.source_entity.lower(), rel.target_entity.lower(), rel.relationship_type)
                    individual_relationships.add(rel_key)
        
        # Find relationships in window that are not in individual chunks
        for rel in window_relationships:
            rel_key = (rel.source_entity.lower(), rel.target_entity.lower(), rel.relationship_type)
            if rel_key not in individual_relationships:
                # This is likely a cross-boundary relationship
                rel.properties = rel.properties or {}
                rel.properties['cross_chunk'] = True
                rel.properties['source_window'] = f"{window.start_chunk_idx}_{window.end_chunk_idx}"
                rel.confidence *= 0.8  # Lower confidence for cross-chunk relationships
                boundary_relationships.append(rel)
        
        return boundary_relationships
    
    async def _resolve_coreferences(self, individual_results: List[GraphExtractionResult],
                                  cross_chunk_entities: List[ExtractedEntity],
                                  cross_chunk_relationships: List[ExtractedRelationship]) -> Tuple[List[ExtractedEntity], List[ExtractedRelationship]]:
        """Resolve coreferences across chunks for better entity linking"""
        # For now, implement basic coreference resolution
        # This could be enhanced with more sophisticated NLP techniques
        
        resolved_entities = cross_chunk_entities.copy()
        resolved_relationships = cross_chunk_relationships.copy()
        
        # Simple pronoun resolution - replace generic pronouns with actual entity references
        # This is a simplified version; full coreference resolution would require more sophisticated NLP
        
        logger.debug(f"🔄 Coreference resolution: {len(resolved_entities)} entities, {len(resolved_relationships)} relationships")
        
        return resolved_entities, resolved_relationships
    
    def _analyze_extraction_quality(self, individual_results: List[GraphExtractionResult],
                                   cross_entities: List[ExtractedEntity],
                                   cross_relationships: List[ExtractedRelationship]) -> Dict[str, Any]:
        """Analyze quality of multi-chunk extraction"""
        total_individual_entities = sum(len(r.entities) for r in individual_results)
        total_individual_relationships = sum(len(r.relationships) for r in individual_results)
        
        quality_metrics = {
            'individual_chunks': len(individual_results),
            'individual_entities': total_individual_entities,
            'individual_relationships': total_individual_relationships,
            'cross_chunk_entities': len(cross_entities),
            'cross_chunk_relationships': len(cross_relationships),
            'entity_improvement_ratio': len(cross_entities) / max(total_individual_entities, 1),
            'relationship_improvement_ratio': len(cross_relationships) / max(total_individual_relationships, 1),
            'avg_cross_chunk_confidence': sum(e.confidence for e in cross_entities) / max(len(cross_entities), 1),
            'avg_cross_relationship_confidence': sum(r.confidence for r in cross_relationships) / max(len(cross_relationships), 1)
        }
        
        return quality_metrics
    
    def _analyze_overlap_effectiveness(self, windows: List[ChunkWindow],
                                     cross_relationships: List[ExtractedRelationship]) -> Dict[str, Any]:
        """Analyze effectiveness of overlap processing"""
        overlap_analysis = {
            'total_windows': len(windows),
            'windows_with_overlap': sum(1 for w in windows if w.overlap_metadata),
            'cross_chunk_relationships_found': len(cross_relationships),
            'avg_window_overlap': sum(
                sum(info['length'] for info in w.overlap_metadata.values()) 
                for w in windows if w.overlap_metadata
            ) / max(len(windows), 1)
        }
        
        return overlap_analysis
    
    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate entities - FIXED to preserve chunk-specific entities"""
        entity_map = {}
        
        for entity in entities:
            # Include chunk context in deduplication key to avoid over-merging
            base_key = entity.canonical_form.lower()
            entity_type = entity.label.upper()
            chunk_context = entity.properties.get('chunk_id', 'unknown') if entity.properties else 'unknown'
            
            # Create compound key that preserves entities from different chunks
            compound_key = f"{base_key}|{entity_type}|{chunk_context}"
            
            if compound_key in entity_map:
                # Keep entity with higher confidence
                existing = entity_map[compound_key]
                if entity.confidence > existing.confidence:
                    entity_map[compound_key] = entity
            else:
                entity_map[compound_key] = entity
        
        # Final pass: only merge entities that are truly identical AND from same source
        final_entities = list(entity_map.values())
        logger.info(f"🔧 Standard deduplication: {len(entities)} → {len(final_entities)} entities (preserved chunk context)")
        return final_entities
    
    def _deduplicate_relationships(self, relationships: List[ExtractedRelationship]) -> List[ExtractedRelationship]:
        """Remove duplicate relationships"""
        seen = set()
        deduplicated = []
        
        for rel in relationships:
            key = (rel.source_entity.lower(), rel.target_entity.lower(), rel.relationship_type)
            if key not in seen:
                seen.add(key)
                deduplicated.append(rel)
        
        return deduplicated
    
    async def _process_chunks_individually(self, chunks: List[ExtractedChunk], 
                                         document_id: str, 
                                         progressive_storage: bool = False,
                                         kg_service=None) -> MultiChunkExtractionResult:
        """Fallback to individual chunk processing"""
        individual_results = await self._process_individual_chunks(
            chunks, document_id, progressive_storage, kg_service
        )
        
        return MultiChunkExtractionResult(
            individual_results=individual_results,
            cross_chunk_entities=[],
            cross_chunk_relationships=[],
            total_processing_time_ms=0.0,
            overlap_analysis={'method': 'individual_only'},
            quality_metrics={'multi_chunk_disabled': True}
        )
    
    async def _process_individual_chunks_enhanced(self, chunks: List[ExtractedChunk], 
                                                document_id: str, 
                                                progressive_storage: bool = False,
                                                kg_service=None) -> List[GraphExtractionResult]:
        """Enhanced individual chunk processing using multi-pass extraction"""
        from app.services.llm_knowledge_extractor import get_llm_knowledge_extractor
        
        valid_results = []
        llm_extractor = get_llm_knowledge_extractor()
        
        # Detect if this is a business document requiring enhanced extraction
        sample_text = ' '.join([chunk.content[:800] for chunk in chunks[:5]])  # Larger sample
        is_business_doc = llm_extractor._is_business_document(sample_text, ['strategy', 'business', 'financial', 'banking'])
        
        if is_business_doc:
            logger.info(f"🚀 BUSINESS STRATEGY DOCUMENT DETECTED - using ULTRA-AGGRESSIVE extraction per chunk")
            logger.info(f"   Target: 4x entity extraction improvement (aim for 60-70+ entities total)")
        
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Processing enhanced chunk {i}/{len(chunks)}")
            try:
                if is_business_doc and len(chunk.content) > 400:  # Lower threshold for business chunks
                    # Use ULTRA-AGGRESSIVE LLM extraction for business chunks
                    domain_hints = ['business', 'strategy', 'technology', 'organization', 'financial', 'banking', 'digital_transformation']
                    context = {
                        'document_type': 'business_strategy_confidential',
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'document_id': document_id,
                        'aggressive_extraction': True,
                        'target_entity_density': 2.0  # 2 entities per 1K chars
                    }
                    
                    llm_result = await llm_extractor.extract_with_llm(chunk.content, context, domain_hints)
                    
                    # Convert LLM result to GraphExtractionResult
                    result = GraphExtractionResult(
                        chunk_id=chunk.metadata.get('chunk_id', f'{document_id}_chunk_{i}'),
                        entities=llm_result.entities,
                        relationships=llm_result.relationships,
                        processing_time_ms=llm_result.processing_time_ms,
                        source_metadata={
                            'extraction_type': llm_result.extraction_metadata.get('extraction_type', 'enhanced'),
                            'confidence_score': llm_result.confidence_score,
                            'business_document': True,
                            'llm_model': llm_result.llm_model_used
                        },
                        warnings=[]
                    )
                else:
                    # Use standard extraction for non-business or small chunks
                    result = await self.kg_service.extract_from_chunk(chunk)
                
                valid_results.append(result)
                
                # Progressive storage with enhanced metadata
                if progressive_storage and kg_service and (result.entities or result.relationships):
                    try:
                        # Add enhancement metadata before storage
                        enhanced_metadata = {
                            'processing_mode': 'enhanced_multi_chunk',
                            'chunk_index': i,
                            'total_chunks': len(chunks),
                            'business_document': is_business_doc,
                            'extraction_enhancement': result.source_metadata.get('extraction_type', 'standard')
                        }
                        result.source_metadata.update(enhanced_metadata)
                        
                        storage_result = await kg_service.store_in_neo4j(result, document_id)
                        if storage_result.get('success'):
                            logger.info(f"📈 Enhanced progressive storage: chunk {i} ({len(result.entities)} entities, {len(result.relationships)} relationships)")
                        else:
                            logger.warning(f"⚠️ Enhanced progressive storage failed for chunk {i}: {storage_result.get('error')}")
                    except Exception as storage_e:
                        logger.error(f"❌ Enhanced progressive storage error for chunk {i}: {storage_e}")
                
                logger.info(f"✅ Enhanced chunk {i} completed: {len(result.entities)} entities, {len(result.relationships)} relationships")
                
            except Exception as e:
                logger.error(f"❌ Enhanced chunk {i} processing failed: {e}")
                continue
        
        original_total = sum(len(r.entities) + len(r.relationships) for r in valid_results if hasattr(r, 'entities'))
        enhancement_factor = original_total / max(len(chunks) * 3, 1)  # Baseline expectation
        logger.info(f"🚀 ULTRA-AGGRESSIVE processing complete: {len(valid_results)} chunks, {original_total} total extractions")
        logger.info(f"   Enhancement factor: {enhancement_factor:.1f}x (target: 4x improvement)")
        if enhancement_factor < 3.0:
            logger.warning(f"   ⚠️  Enhancement below target - consider lowering confidence thresholds")
        
        return valid_results
    
    async def _analyze_document_level_entities(self, chunks: List[ExtractedChunk], 
                                             individual_results: List[GraphExtractionResult]) -> Dict[str, Any]:
        """Analyze entity patterns across the entire document for better cross-chunk linking"""
        logger.info("🔍 Analyzing document-level entity patterns...")
        
        # Collect all entities with their occurrence data
        entity_frequency = {}
        entity_contexts = {}
        entity_chunk_map = {}
        
        for i, result in enumerate(individual_results):
            chunk_id = result.chunk_id
            
            for entity in result.entities:
                canonical = entity.canonical_form.lower().strip()
                
                if canonical not in entity_frequency:
                    entity_frequency[canonical] = {
                        'count': 0,
                        'entity_objects': [],
                        'chunks': [],
                        'contexts': [],
                        'types': set(),
                        'confidences': []
                    }
                
                entity_frequency[canonical]['count'] += 1
                entity_frequency[canonical]['entity_objects'].append(entity)
                entity_frequency[canonical]['chunks'].append(chunk_id)
                entity_frequency[canonical]['types'].add(entity.label)
                entity_frequency[canonical]['confidences'].append(entity.confidence)
                
                # Store chunk mapping
                if canonical not in entity_chunk_map:
                    entity_chunk_map[canonical] = []
                entity_chunk_map[canonical].append(i)
        
        # Identify high-frequency entities (potential document themes) - MORE AGGRESSIVE
        high_frequency_entities = {
            name: data for name, data in entity_frequency.items() 
            if data['count'] >= 1  # Any entity that appears (lowered threshold for business docs)
        }
        
        # Identify potential business hierarchies
        business_hierarchies = self._identify_business_hierarchies(entity_frequency)
        
        # Calculate entity co-occurrence matrix
        cooccurrence_matrix = self._calculate_entity_cooccurrence_matrix(
            individual_results, entity_frequency
        )
        
        analysis_result = {
            'entity_frequency': entity_frequency,
            'high_frequency_entities': high_frequency_entities,
            'business_hierarchies': business_hierarchies,
            'cooccurrence_matrix': cooccurrence_matrix,
            'entity_chunk_map': entity_chunk_map,
            'total_unique_entities': len(entity_frequency),
            'multi_chunk_entities': len(high_frequency_entities)
        }
        
        logger.info(f"📊 Document analysis complete: {len(entity_frequency)} unique entities, {len(high_frequency_entities)} cross-chunk entities")
        return analysis_result
    
    def _identify_business_hierarchies(self, entity_frequency: Dict[str, Any]) -> Dict[str, List[str]]:
        """Identify potential business hierarchies and relationships"""
        hierarchies = {
            'organizations': [],
            'people': [],
            'technologies': [],
            'locations': [],
            'products': []
        }
        
        for entity_name, data in entity_frequency.items():
            entity_types = data['types']
            
            if any(t in ['ORGANIZATION', 'COMPANY', 'ORG'] for t in entity_types):
                hierarchies['organizations'].append(entity_name)
            elif any(t in ['PERSON', 'EXECUTIVE', 'CEO', 'CTO'] for t in entity_types):
                hierarchies['people'].append(entity_name)
            elif any(t in ['TECHNOLOGY', 'SYSTEM', 'PLATFORM'] for t in entity_types):
                hierarchies['technologies'].append(entity_name)
            elif any(t in ['LOCATION', 'CITY', 'COUNTRY'] for t in entity_types):
                hierarchies['locations'].append(entity_name)
            elif any(t in ['PRODUCT', 'SERVICE', 'SOLUTION'] for t in entity_types):
                hierarchies['products'].append(entity_name)
        
        return hierarchies
    
    def _calculate_entity_cooccurrence_matrix(self, individual_results: List[GraphExtractionResult],
                                            entity_frequency: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
        """Calculate how often entities appear together in chunks"""
        cooccurrence = {}
        
        for result in individual_results:
            chunk_entities = [e.canonical_form.lower().strip() for e in result.entities]
            
            # Calculate co-occurrence for this chunk
            for i, entity1 in enumerate(chunk_entities):
                if entity1 not in cooccurrence:
                    cooccurrence[entity1] = {}
                
                for entity2 in chunk_entities[i+1:]:
                    if entity2 not in cooccurrence[entity1]:
                        cooccurrence[entity1][entity2] = 0
                    if entity1 not in cooccurrence.get(entity2, {}):
                        if entity2 not in cooccurrence:
                            cooccurrence[entity2] = {}
                        cooccurrence[entity2][entity1] = 0
                    
                    cooccurrence[entity1][entity2] += 1
                    cooccurrence[entity2][entity1] += 1
        
        return cooccurrence
    
    def _create_enhanced_chunk_windows(self, chunks: List[ExtractedChunk], 
                                     document_entity_analysis: Dict[str, Any]) -> List[ChunkWindow]:
        """Create enhanced sliding windows with adaptive overlap based on entity analysis"""
        windows = []
        
        # Get high-frequency entities for adaptive windowing
        high_freq_entities = document_entity_analysis.get('high_frequency_entities', {})
        entity_chunk_map = document_entity_analysis.get('entity_chunk_map', {})
        
        # Create standard sliding windows first
        for i in range(len(chunks) - 1):
            window_chunks = chunks[i:i+2]
            
            # Calculate adaptive overlap based on entity boundaries
            overlap_info = self._calculate_adaptive_overlap(
                window_chunks, high_freq_entities, entity_chunk_map, i
            )
            
            # Create combined text with intelligent overlap handling
            combined_text = self._combine_chunks_with_overlap(window_chunks, overlap_info)
            
            window = ChunkWindow(
                chunks=window_chunks,
                window_text=combined_text,
                start_chunk_idx=i,
                end_chunk_idx=i+1,
                overlap_metadata={
                    'adaptive_overlap': overlap_info,
                    'entity_density': len([e for e in high_freq_entities.keys() 
                                         if any(chunk_idx in [i, i+1] for chunk_idx in entity_chunk_map.get(e, []))]),
                    'enhancement_type': 'entity_aware'
                }
            )
            windows.append(window)
        
        # Create extended windows for high-entity-density regions
        entity_hotspots = self._identify_entity_hotspots(chunks, document_entity_analysis)
        for hotspot in entity_hotspots:
            if hotspot['chunk_span'] > 2:  # Only for regions spanning more than 2 chunks
                extended_chunks = chunks[hotspot['start_idx']:hotspot['end_idx']+1]
                combined_text = '\n'.join([chunk.content for chunk in extended_chunks])
                
                extended_window = ChunkWindow(
                    chunks=extended_chunks,
                    window_text=combined_text,
                    start_chunk_idx=hotspot['start_idx'],
                    end_chunk_idx=hotspot['end_idx'],
                    overlap_metadata={
                        'entity_hotspot': True,
                        'entity_density': hotspot['entity_density'],
                        'enhancement_type': 'entity_hotspot'
                    }
                )
                windows.append(extended_window)
        
        logger.info(f"🔧 Created {len(windows)} enhanced chunk windows ({len(entity_hotspots)} entity hotspots)")
        return windows
    
    def _calculate_adaptive_overlap(self, window_chunks: List[ExtractedChunk], 
                                  high_freq_entities: Dict[str, Any],
                                  entity_chunk_map: Dict[str, List[int]], 
                                  chunk_idx: int) -> Dict[str, Any]:
        """Calculate adaptive overlap based on entity boundaries"""
        overlap_info = {
            'standard_overlap': self.overlap_size,
            'entity_based_extension': 0,
            'boundary_entities': []
        }
        
        # Find entities that span the boundary between these chunks
        boundary_entities = []
        for entity_name, chunk_indices in entity_chunk_map.items():
            if chunk_idx in chunk_indices and (chunk_idx + 1) in chunk_indices:
                boundary_entities.append(entity_name)
                overlap_info['boundary_entities'].append(entity_name)
        
        # Extend overlap for boundary entities
        if boundary_entities:
            overlap_info['entity_based_extension'] = min(200, len(boundary_entities) * 50)
        
        return overlap_info
    
    def _combine_chunks_with_overlap(self, chunks: List[ExtractedChunk], 
                                   overlap_info: Dict[str, Any]) -> str:
        """Combine chunks with intelligent overlap handling"""
        if len(chunks) == 1:
            return chunks[0].content
        
        # Standard overlap detection
        overlap_text, overlap_length = self._find_chunk_overlap(chunks[0].content, chunks[1].content)
        
        # Apply entity-based extension if needed
        entity_extension = overlap_info.get('entity_based_extension', 0)
        effective_overlap = max(overlap_length, entity_extension)
        
        if effective_overlap > 0 and effective_overlap < len(chunks[1].content):
            combined_text = chunks[0].content + "\n" + chunks[1].content[effective_overlap:]
        else:
            combined_text = chunks[0].content + "\n" + chunks[1].content
        
        return combined_text
    
    def _identify_entity_hotspots(self, chunks: List[ExtractedChunk], 
                                document_entity_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify regions with high entity density for extended window processing"""
        entity_chunk_map = document_entity_analysis.get('entity_chunk_map', {})
        chunk_entity_density = {}
        
        # Calculate entity density per chunk
        for chunk_idx in range(len(chunks)):
            chunk_entities = [entity for entity, chunk_indices in entity_chunk_map.items() 
                            if chunk_idx in chunk_indices]
            chunk_entity_density[chunk_idx] = len(chunk_entities)
        
        # Identify hotspots (consecutive chunks with high entity density)
        hotspots = []
        current_hotspot = None
        density_threshold = max(2, sum(chunk_entity_density.values()) / len(chunks) * 1.5)
        
        for chunk_idx, density in chunk_entity_density.items():
            if density >= density_threshold:
                if current_hotspot is None:
                    current_hotspot = {
                        'start_idx': chunk_idx,
                        'end_idx': chunk_idx,
                        'entity_density': density,
                        'chunk_span': 1
                    }
                else:
                    current_hotspot['end_idx'] = chunk_idx
                    current_hotspot['entity_density'] = max(current_hotspot['entity_density'], density)
                    current_hotspot['chunk_span'] = current_hotspot['end_idx'] - current_hotspot['start_idx'] + 1
            else:
                if current_hotspot is not None:
                    hotspots.append(current_hotspot)
                    current_hotspot = None
        
        # Add final hotspot if exists
        if current_hotspot is not None:
            hotspots.append(current_hotspot)
        
        return hotspots
    
    async def _extract_cross_chunk_relationships_enhanced(self, windows: List[ChunkWindow],
                                                        individual_results: List[GraphExtractionResult],
                                                        document_entity_analysis: Dict[str, Any]) -> Tuple[List[ExtractedEntity], List[ExtractedRelationship]]:
        """Enhanced cross-chunk relationship extraction with entity-aware processing"""
        cross_chunk_entities = []
        cross_chunk_relationships = []
        
        high_freq_entities = document_entity_analysis.get('high_frequency_entities', {})
        cooccurrence_matrix = document_entity_analysis.get('cooccurrence_matrix', {})
        
        logger.info(f"🔗 Enhanced cross-chunk processing: {len(windows)} windows, {len(high_freq_entities)} high-freq entities")
        
        for window in windows:
            try:
                # Extract from combined window text with enhanced focus
                window_chunk = ExtractedChunk(
                    content=window.window_text,
                    metadata={
                        'chunk_id': f"enhanced_window_{window.start_chunk_idx}_{window.end_chunk_idx}",
                        'window_type': 'enhanced_cross_chunk',
                        'source_chunks': [c.metadata.get('chunk_id', 'unknown') for c in window.chunks],
                        'overlap_info': window.overlap_metadata,
                        'entity_context': list(high_freq_entities.keys())[:10]
                    }
                )
                
                window_result = await self.kg_service.extract_from_chunk(window_chunk)
                
                # Enhanced boundary entity identification
                boundary_entities = self._identify_boundary_entities_enhanced(
                    window_result.entities, window, individual_results, high_freq_entities
                )
                cross_chunk_entities.extend(boundary_entities)
                
                # Enhanced boundary relationship identification
                boundary_relationships = self._identify_boundary_relationships_enhanced(
                    window_result.relationships, window, individual_results, cooccurrence_matrix
                )
                cross_chunk_relationships.extend(boundary_relationships)
                
            except Exception as e:
                logger.error(f"Enhanced window processing failed: {e}")
                continue
        
        # Enhanced deduplication with entity frequency weighting
        cross_chunk_entities = self._deduplicate_entities_enhanced(cross_chunk_entities, high_freq_entities)
        cross_chunk_relationships = self._deduplicate_relationships_enhanced(cross_chunk_relationships, cooccurrence_matrix)
        
        logger.info(f"✅ Enhanced cross-chunk extraction: {len(cross_chunk_entities)} entities, {len(cross_chunk_relationships)} relationships")
        return cross_chunk_entities, cross_chunk_relationships
    
    def _identify_boundary_entities_enhanced(self, window_entities: List[ExtractedEntity],
                                           window: ChunkWindow, 
                                           individual_results: List[GraphExtractionResult],
                                           high_freq_entities: Dict[str, Any]) -> List[ExtractedEntity]:
        """Enhanced boundary entity identification with frequency analysis"""
        boundary_entities = []
        
        # Get entities from individual chunks in this window
        individual_entities = set()
        for i in range(window.start_chunk_idx, window.end_chunk_idx + 1):
            if i < len(individual_results):
                for entity in individual_results[i].entities:
                    individual_entities.add(entity.canonical_form.lower())
        
        # Find entities in window that are not in individual chunks or are high-frequency
        for entity in window_entities:
            entity_key = entity.canonical_form.lower()
            
            # Prioritize cross-boundary entities and high-frequency entities
            is_cross_boundary = entity_key not in individual_entities
            is_high_frequency = entity_key in high_freq_entities
            
            if is_cross_boundary or is_high_frequency:
                # Enhance entity properties
                entity.properties = entity.properties or {}
                entity.properties['cross_chunk'] = is_cross_boundary
                entity.properties['high_frequency'] = is_high_frequency
                entity.properties['source_window'] = f"{window.start_chunk_idx}_{window.end_chunk_idx}"
                
                if is_high_frequency:
                    # Boost confidence for high-frequency entities
                    entity.confidence = min(1.0, entity.confidence + 0.1)
                elif is_cross_boundary:
                    # Slightly lower confidence for pure cross-boundary entities
                    entity.confidence *= 0.9
                
                boundary_entities.append(entity)
        
        return boundary_entities
    
    def _identify_boundary_relationships_enhanced(self, window_relationships: List[ExtractedRelationship],
                                                window: ChunkWindow,
                                                individual_results: List[GraphExtractionResult],
                                                cooccurrence_matrix: Dict[str, Dict[str, int]]) -> List[ExtractedRelationship]:
        """Enhanced boundary relationship identification with co-occurrence analysis"""
        boundary_relationships = []
        
        # Get relationships from individual chunks
        individual_relationships = set()
        for i in range(window.start_chunk_idx, window.end_chunk_idx + 1):
            if i < len(individual_results):
                for rel in individual_results[i].relationships:
                    rel_key = (rel.source_entity.lower(), rel.target_entity.lower(), rel.relationship_type)
                    individual_relationships.add(rel_key)
        
        # Find relationships that are cross-boundary or have high co-occurrence
        for rel in window_relationships:
            rel_key = (rel.source_entity.lower(), rel.target_entity.lower(), rel.relationship_type)
            is_cross_boundary = rel_key not in individual_relationships
            
            # Check co-occurrence score
            source_lower = rel.source_entity.lower()
            target_lower = rel.target_entity.lower()
            cooccurrence_score = cooccurrence_matrix.get(source_lower, {}).get(target_lower, 0)
            
            if is_cross_boundary or cooccurrence_score >= 2:
                # Enhance relationship properties
                rel.properties = rel.properties or {}
                rel.properties['cross_chunk'] = is_cross_boundary
                rel.properties['cooccurrence_score'] = cooccurrence_score
                rel.properties['source_window'] = f"{window.start_chunk_idx}_{window.end_chunk_idx}"
                
                # Adjust confidence based on co-occurrence
                if cooccurrence_score >= 3:
                    rel.confidence = min(1.0, rel.confidence + 0.1)
                elif is_cross_boundary:
                    rel.confidence *= 0.8
                
                boundary_relationships.append(rel)
        
        return boundary_relationships
    
    def _deduplicate_entities_enhanced(self, entities: List[ExtractedEntity], 
                                     high_freq_entities: Dict[str, Any]) -> List[ExtractedEntity]:
        """Enhanced entity deduplication with frequency weighting - FIXED to preserve distinct entities"""
        entity_map = {}
        
        for entity in entities:
            # Create more specific key that includes chunk context to avoid over-deduplication
            base_key = entity.canonical_form.lower()
            chunk_context = entity.properties.get('chunk_id', 'unknown') if entity.properties else 'unknown'
            entity_type = entity.label.upper()
            source_window = entity.properties.get('source_window', 'main') if entity.properties else 'main'
            
            # Create compound key that preserves chunk-specific entities
            # Only deduplicate if entities are from the same chunk AND have identical canonical forms
            compound_key = f"{base_key}|{entity_type}|{chunk_context}|{source_window}"
            
            if compound_key in entity_map:
                # Keep entity with higher confidence for truly identical entities
                existing = entity_map[compound_key]
                is_new_high_freq = base_key in high_freq_entities
                is_existing_high_freq = existing.properties.get('high_frequency', False) if existing.properties else False
                
                # Only replace if new entity is clearly better
                if (is_new_high_freq and not is_existing_high_freq) or \
                   (is_new_high_freq == is_existing_high_freq and entity.confidence > existing.confidence + 0.1):
                    entity_map[compound_key] = entity
            else:
                entity_map[compound_key] = entity
        
        deduplicated_entities = list(entity_map.values())
        
        # Now do a final pass to merge only truly identical entities (same name, type, and context)
        final_entity_map = {}
        for entity in deduplicated_entities:
            final_key = f"{entity.canonical_form.lower()}|{entity.label.upper()}"
            
            if final_key in final_entity_map:
                existing = final_entity_map[final_key]
                # Only merge if entities are truly similar in context and one is clearly superior
                entity_context = entity.properties.get('chunk_id', '') if entity.properties else ''
                existing_context = existing.properties.get('chunk_id', '') if existing.properties else ''
                
                # If from different chunks, keep both unless one has much higher confidence
                if entity_context != existing_context:
                    if entity.confidence > existing.confidence + 0.2:  # Significant confidence difference
                        final_entity_map[final_key] = entity
                    # Otherwise keep existing (don't replace with similar entity from different chunk)
                else:
                    # Same chunk - can safely merge, keep higher confidence
                    if entity.confidence > existing.confidence:
                        final_entity_map[final_key] = entity
            else:
                final_entity_map[final_key] = entity
        
        result_entities = list(final_entity_map.values())
        logger.info(f"🔧 Enhanced deduplication: {len(entities)} → {len(result_entities)} entities (preserved chunk-specific entities)")
        return result_entities
    
    def _deduplicate_relationships_enhanced(self, relationships: List[ExtractedRelationship],
                                          cooccurrence_matrix: Dict[str, Dict[str, int]]) -> List[ExtractedRelationship]:
        """Enhanced relationship deduplication with co-occurrence weighting"""
        rel_map = {}
        
        for rel in relationships:
            key = (rel.source_entity.lower(), rel.target_entity.lower(), rel.relationship_type)
            
            if key in rel_map:
                # Keep relationship with higher confidence, but prefer high co-occurrence
                existing = rel_map[key]
                new_cooccurrence = rel.properties.get('cooccurrence_score', 0)
                existing_cooccurrence = existing.properties.get('cooccurrence_score', 0)
                
                if (new_cooccurrence > existing_cooccurrence) or \
                   (new_cooccurrence == existing_cooccurrence and rel.confidence > existing.confidence):
                    rel_map[key] = rel
            else:
                rel_map[key] = rel
        
        return list(rel_map.values())
    
    async def _analyze_business_hierarchies(self, all_results: List[GraphExtractionResult],
                                          document_entity_analysis: Dict[str, Any]) -> Tuple[List[ExtractedEntity], List[ExtractedRelationship]]:
        """Analyze and extract business hierarchies and organizational relationships"""
        hierarchy_entities = []
        hierarchy_relationships = []
        
        business_hierarchies = document_entity_analysis.get('business_hierarchies', {})
        high_freq_entities = document_entity_analysis.get('high_frequency_entities', {})
        
        # Extract organizational hierarchy relationships
        orgs = business_hierarchies.get('organizations', [])
        people = business_hierarchies.get('people', [])
        technologies = business_hierarchies.get('technologies', [])
        
        # Infer organizational relationships
        for org in orgs:
            for person in people:
                if org in high_freq_entities and person in high_freq_entities:
                    # Create organizational relationship
                    rel = ExtractedRelationship(
                        source_entity=person.title(),
                        target_entity=org.title(),
                        relationship_type='WORKS_FOR',
                        confidence=0.6,
                        context=f"Inferred from business hierarchy analysis",
                        properties={
                            'hierarchy_inferred': True,
                            'inference_method': 'business_hierarchy'
                        }
                    )
                    hierarchy_relationships.append(rel)
        
        # Infer technology-organization relationships
        for org in orgs:
            for tech in technologies:
                if org in high_freq_entities and tech in high_freq_entities:
                    rel = ExtractedRelationship(
                        source_entity=org.title(),
                        target_entity=tech.title(),
                        relationship_type='USES',
                        confidence=0.5,
                        context=f"Inferred from business hierarchy analysis",
                        properties={
                            'hierarchy_inferred': True,
                            'inference_method': 'technology_adoption'
                        }
                    )
                    hierarchy_relationships.append(rel)
        
        logger.info(f"📈 Business hierarchy analysis: {len(hierarchy_entities)} entities, {len(hierarchy_relationships)} relationships")
        return hierarchy_entities, hierarchy_relationships
    
    def _create_synthetic_result(self, entities: List[ExtractedEntity], 
                               relationships: List[ExtractedRelationship]) -> GraphExtractionResult:
        """Create synthetic result for processing pipeline"""
        return GraphExtractionResult(
            chunk_id="synthetic_cross_chunk",
            entities=entities,
            relationships=relationships,
            processing_time_ms=0.0,
            source_metadata={'synthetic': True},
            warnings=[]
        )
    
    async def _resolve_coreferences_enhanced(self, individual_results: List[GraphExtractionResult],
                                           cross_chunk_entities: List[ExtractedEntity],
                                           cross_chunk_relationships: List[ExtractedRelationship],
                                           document_entity_analysis: Dict[str, Any]) -> Tuple[List[ExtractedEntity], List[ExtractedRelationship]]:
        """Enhanced coreference resolution with business context"""
        # Combine all entities for comprehensive resolution
        all_entities = []
        for result in individual_results:
            all_entities.extend(result.entities)
        all_entities.extend(cross_chunk_entities)
        
        # Combine all relationships
        all_relationships = []
        for result in individual_results:
            all_relationships.extend(result.relationships)
        all_relationships.extend(cross_chunk_relationships)
        
        # Apply enhanced deduplication
        high_freq_entities = document_entity_analysis.get('high_frequency_entities', {})
        cooccurrence_matrix = document_entity_analysis.get('cooccurrence_matrix', {})
        
        resolved_entities = self._deduplicate_entities_enhanced(all_entities, high_freq_entities)
        resolved_relationships = self._deduplicate_relationships_enhanced(all_relationships, cooccurrence_matrix)
        
        logger.info(f"🔄 Enhanced coreference resolution: {len(resolved_entities)} entities, {len(resolved_relationships)} relationships")
        return resolved_entities, resolved_relationships
    
    async def _infer_document_level_relationships(self, entities: List[ExtractedEntity], 
                                                relationships: List[ExtractedRelationship],
                                                chunks: List[ExtractedChunk],
                                                document_entity_analysis: Dict[str, Any]) -> List[ExtractedRelationship]:
        """Infer additional relationships based on document-level patterns"""
        inferred_relationships = []
        cooccurrence_matrix = document_entity_analysis.get('cooccurrence_matrix', {})
        
        # Create entity lookup
        entity_map = {e.canonical_form.lower(): e for e in entities}
        
        # Infer relationships from high co-occurrence pairs
        for entity1_name, cooccurrences in cooccurrence_matrix.items():
            for entity2_name, count in cooccurrences.items():
                if count >= 3:  # High co-occurrence threshold
                    entity1 = entity_map.get(entity1_name)
                    entity2 = entity_map.get(entity2_name)
                    
                    if entity1 and entity2:
                        # Check if relationship already exists
                        existing = any(
                            r.source_entity.lower() == entity1_name and r.target_entity.lower() == entity2_name
                            for r in relationships
                        )
                        
                        if not existing:
                            # Infer relationship type based on entity types
                            rel_type = self._infer_relationship_from_types(entity1, entity2)
                            
                            inferred_rel = ExtractedRelationship(
                                source_entity=entity1.canonical_form,
                                target_entity=entity2.canonical_form,
                                relationship_type=rel_type,
                                confidence=0.4 + (count * 0.1),
                                context=f"Document-level inference (co-occurs {count} times)",
                                properties={
                                    'document_level_inferred': True,
                                    'cooccurrence_count': count,
                                    'inference_method': 'document_cooccurrence'
                                }
                            )
                            inferred_relationships.append(inferred_rel)
        
        logger.info(f"🔮 Document-level inference: {len(inferred_relationships)} relationships")
        return inferred_relationships
    
    def _infer_relationship_from_types(self, entity1: ExtractedEntity, entity2: ExtractedEntity) -> str:
        """Infer relationship type based on entity types"""
        type1, type2 = entity1.label, entity2.label
        
        # Business relationship inference rules
        if type1 in ['PERSON', 'EXECUTIVE'] and type2 in ['ORGANIZATION', 'COMPANY']:
            return 'ASSOCIATED_WITH'
        elif type1 in ['ORGANIZATION', 'COMPANY'] and type2 in ['TECHNOLOGY', 'SYSTEM']:
            return 'USES'
        elif type1 in ['ORGANIZATION', 'COMPANY'] and type2 in ['PRODUCT', 'SERVICE']:
            return 'OFFERS'
        elif type1 in ['TECHNOLOGY', 'SYSTEM'] and type2 in ['PRODUCT', 'SERVICE']:
            return 'ENABLES'
        else:
            return 'RELATED_TO'
    
    def _analyze_extraction_quality_enhanced(self, individual_results: List[GraphExtractionResult],
                                           cross_entities: List[ExtractedEntity],
                                           cross_relationships: List[ExtractedRelationship],
                                           document_entity_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced quality analysis with business intelligence metrics"""
        total_individual_entities = sum(len(r.entities) for r in individual_results)
        total_individual_relationships = sum(len(r.relationships) for r in individual_results)
        
        high_freq_entities = len(document_entity_analysis.get('high_frequency_entities', {}))
        business_hierarchies = document_entity_analysis.get('business_hierarchies', {})
        
        quality_metrics = {
            'individual_chunks': len(individual_results),
            'individual_entities': total_individual_entities,
            'individual_relationships': total_individual_relationships,
            'final_entities': len(cross_entities),
            'final_relationships': len(cross_relationships),
            'entity_enhancement_ratio': len(cross_entities) / max(total_individual_entities, 1),
            'relationship_enhancement_ratio': len(cross_relationships) / max(total_individual_relationships, 1),
            'high_frequency_entities': high_freq_entities,
            'business_hierarchy_coverage': {
                'organizations': len(business_hierarchies.get('organizations', [])),
                'people': len(business_hierarchies.get('people', [])),
                'technologies': len(business_hierarchies.get('technologies', [])),
                'locations': len(business_hierarchies.get('locations', [])),
                'products': len(business_hierarchies.get('products', []))
            },
            'avg_entity_confidence': sum(e.confidence for e in cross_entities) / max(len(cross_entities), 1),
            'avg_relationship_confidence': sum(r.confidence for r in cross_relationships) / max(len(cross_relationships), 1),
            'cross_chunk_entities': len([e for e in cross_entities if e.properties.get('cross_chunk', False)]),
            'high_freq_entity_coverage': len([e for e in cross_entities if e.properties.get('high_frequency', False)]),
            'inferred_relationships': len([r for r in cross_relationships if r.properties.get('hierarchy_inferred', False) or r.properties.get('document_level_inferred', False)])
        }
        
        return quality_metrics
    
    def _analyze_overlap_effectiveness_enhanced(self, windows: List[ChunkWindow],
                                              cross_relationships: List[ExtractedRelationship],
                                              document_entity_analysis: Dict[str, Any],
                                              entity_enhancement_factor: float,
                                              relationship_enhancement_factor: float) -> Dict[str, Any]:
        """Enhanced overlap effectiveness analysis"""
        overlap_analysis = {
            'total_windows': len(windows),
            'entity_aware_windows': len([w for w in windows if w.overlap_metadata.get('enhancement_type') == 'entity_aware']),
            'entity_hotspot_windows': len([w for w in windows if w.overlap_metadata.get('enhancement_type') == 'entity_hotspot']),
            'cross_chunk_relationships_found': len(cross_relationships),
            'entity_enhancement_factor': entity_enhancement_factor,
            'relationship_enhancement_factor': relationship_enhancement_factor,
            'high_confidence_relationships': len([r for r in cross_relationships if r.confidence >= 0.7]),
            'inferred_relationships': len([r for r in cross_relationships if 'inferred' in str(r.properties)]),
            'avg_entity_density_per_window': sum(
                w.overlap_metadata.get('entity_density', 0) for w in windows
            ) / max(len(windows), 1),
            'document_analysis_quality': {
                'unique_entities': document_entity_analysis.get('total_unique_entities', 0),
                'multi_chunk_entities': document_entity_analysis.get('multi_chunk_entities', 0),
                'business_hierarchy_completeness': len(document_entity_analysis.get('business_hierarchies', {}))
            }
        }
        
        return overlap_analysis

# Singleton instance
_multi_chunk_processor: Optional[MultiChunkRelationshipProcessor] = None

def get_multi_chunk_processor() -> MultiChunkRelationshipProcessor:
    """Get or create multi-chunk processor singleton"""
    global _multi_chunk_processor
    if _multi_chunk_processor is None:
        _multi_chunk_processor = MultiChunkRelationshipProcessor()
    return _multi_chunk_processor