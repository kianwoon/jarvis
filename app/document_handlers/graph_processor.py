"""
Graph Document Processor

Extends the existing document handler interface to process documents specifically
for knowledge graph extraction and ingestion into Neo4j.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from app.document_handlers.base import DocumentHandler, ExtractedChunk, ExtractionPreview
from app.services.knowledge_graph_service import get_knowledge_graph_service
from app.services.knowledge_graph_types import (
    GraphExtractionResult,
    ExtractedEntity,
    ExtractedRelationship
)
from app.core.knowledge_graph_settings_cache import get_knowledge_graph_settings
from app.services.dynamic_chunk_sizing import get_dynamic_chunk_sizer, optimize_chunks_for_model

logger = logging.getLogger(__name__)

@dataclass
class GraphProcessingResult:
    """Result of graph processing for a document"""
    document_id: str
    total_chunks: int
    processed_chunks: int
    total_entities: int
    total_relationships: int
    processing_time_ms: float
    success: bool
    errors: List[str] = None
    graph_data: List[GraphExtractionResult] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.graph_data is None:
            self.graph_data = []

@dataclass 
class GraphChunk(ExtractedChunk):
    """Extended chunk with graph-specific metadata"""
    entities: List[ExtractedEntity] = None
    relationships: List[ExtractedRelationship] = None
    graph_confidence: float = 0.0
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.relationships is None:
            self.relationships = []
        # Update quality score based on graph content
        if self.entities or self.relationships:
            self.graph_confidence = min(1.0, (len(self.entities) * 0.1 + len(self.relationships) * 0.2))
            self.quality_score = max(self.quality_score, self.graph_confidence)

class GraphDocumentProcessor:
    """
    Specialized document processor for knowledge graph extraction.
    Works with existing document handlers to add graph processing capabilities.
    """
    
    def __init__(self):
        self.kg_service = get_knowledge_graph_service()
        self.config = get_knowledge_graph_settings()
        self.chunk_sizer = get_dynamic_chunk_sizer()
        
        # Get dynamic chunk configuration based on current model
        chunk_config = self.chunk_sizer.get_chunk_configuration()
        
        # Graph processing settings with dynamic chunk sizing
        self.min_chunk_length = chunk_config.get('min_chunk_size', 100)
        self.max_chunk_length = chunk_config.get('max_chunk_size', 2000)
        self.chunk_overlap = chunk_config.get('chunk_overlap', 200)
        self.processing_strategy = chunk_config.get('processing_strategy', 'traditional')
        self.entity_threshold = self.config.get('extraction', {}).get('min_entity_confidence', 0.7)
        self.relationship_threshold = self.config.get('extraction', {}).get('min_relationship_confidence', 0.6)
        
        logger.info(f"🧠 Graph processor initialized with dynamic chunk sizing:")
        logger.info(f"   Strategy: {self.processing_strategy}")
        logger.info(f"   Chunk size: {self.min_chunk_length:,} - {self.max_chunk_length:,} characters")
        logger.info(f"   Model context: {self.chunk_sizer.context_limit:,} tokens")
    
    async def process_document_for_graph(
        self, 
        chunks: List[ExtractedChunk], 
        document_id: str,
        store_in_neo4j: bool = True,
        progressive_storage: bool = True
    ) -> GraphProcessingResult:
        """
        Process document chunks for knowledge graph extraction with multi-chunk overlap detection.
        
        Args:
            chunks: List of extracted chunks from document handlers
            document_id: Unique identifier for the document
            store_in_neo4j: Whether to store results in Neo4j database
            
        Returns:
            GraphProcessingResult with extraction statistics and data
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing {len(chunks)} chunks for knowledge graph extraction (doc: {document_id})")
            
            # Check if multi-chunk processing is enabled
            enable_multi_chunk = self.config.get('extraction', {}).get('enable_multi_chunk_relationships', True)
            
            if enable_multi_chunk and len(chunks) > 1:
                # Use multi-chunk processor for better cross-boundary relationship detection
                return await self._process_with_multi_chunk_overlap(
                    chunks, document_id, store_in_neo4j, progressive_storage
                )
            else:
                # Use standard single-chunk processing
                return await self._process_chunks_individually(
                    chunks, document_id, store_in_neo4j, progressive_storage
                )
                
        except Exception as e:
            logger.error(f"Graph processing failed for document {document_id}: {e}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return GraphProcessingResult(
                document_id=document_id,
                total_chunks=len(chunks),
                processed_chunks=0,
                total_entities=0,
                total_relationships=0,
                processing_time_ms=processing_time,
                success=False,
                errors=[f"Processing failed: {str(e)}"]
            )
    
    async def _process_with_multi_chunk_overlap(self, chunks: List[ExtractedChunk], 
                                              document_id: str, store_in_neo4j: bool, 
                                              progressive_storage: bool = True) -> GraphProcessingResult:
        """Process chunks with overlap detection for cross-boundary relationships"""
        start_time = datetime.now()
        
        try:
            from app.services.multi_chunk_processor import get_multi_chunk_processor
            multi_chunk_processor = get_multi_chunk_processor()
            
            # Filter chunks suitable for graph processing
            suitable_chunks = self._filter_chunks_for_graph_processing(chunks)
            
            if not suitable_chunks:
                logger.warning("No suitable chunks found for graph processing")
                return self._create_empty_result(document_id, len(chunks))
            
            # Process with multi-chunk overlap detection
            kg_service = self.kg_service if progressive_storage else None
            multi_result = await multi_chunk_processor.process_document_with_overlap(
                suitable_chunks, document_id, progressive_storage, kg_service
            )
            
            # Combine all results for final calculation
            all_extraction_results = multi_result.individual_results.copy()
            
            # Add cross-chunk results for calculation
            if multi_result.cross_chunk_entities or multi_result.cross_chunk_relationships:
                cross_chunk_result = GraphExtractionResult(
                    chunk_id=f"{document_id}_cross_chunk",
                    entities=multi_result.cross_chunk_entities,
                    relationships=multi_result.cross_chunk_relationships,
                    processing_time_ms=multi_result.total_processing_time_ms * 0.3,
                    source_metadata={'type': 'cross_chunk_synthesis'},
                    warnings=[]
                )
                all_extraction_results.append(cross_chunk_result)
            
            # Store in Neo4j if requested (skip individual results if progressive storage was used)
            storage_results = []
            if store_in_neo4j and not progressive_storage:
                # Only store if progressive storage wasn't used (to avoid duplicates)
                for result in multi_result.individual_results:
                    if result.entities or result.relationships:
                        storage_result = await self.kg_service.store_in_neo4j(result, document_id)
                        storage_results.append(storage_result)
            
            # Always store cross-chunk results (these are new relationships)
            cross_chunk_stored = 0
            if store_in_neo4j and (multi_result.cross_chunk_entities or multi_result.cross_chunk_relationships):
                cross_chunk_result = GraphExtractionResult(
                    chunk_id=f"{document_id}_cross_chunk",
                    entities=multi_result.cross_chunk_entities,
                    relationships=multi_result.cross_chunk_relationships,
                    processing_time_ms=multi_result.total_processing_time_ms * 0.3,
                    source_metadata={'type': 'cross_chunk_synthesis'},
                    warnings=[]
                )
                storage_result = await self.kg_service.store_in_neo4j(cross_chunk_result, document_id)
                storage_results.append(storage_result)
                if storage_result.get('success'):
                    cross_chunk_stored = storage_result.get('entities_stored', 0) + storage_result.get('relationships_stored', 0)
                    logger.info(f"🔗 Cross-chunk analysis stored: {cross_chunk_stored} new connections")
            
            # Calculate totals
            total_entities = sum(len(result.entities) for result in all_extraction_results)
            total_relationships = sum(len(result.relationships) for result in all_extraction_results)
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Check for errors
            errors = []
            for result in all_extraction_results:
                errors.extend(result.warnings)
            
            success = len(errors) == 0 and total_entities > 0
            
            # Run final anti-silo analysis if progressive storage was used
            anti_silo_results = {}
            if progressive_storage and store_in_neo4j:
                try:
                    logger.info("🌍 Running final anti-silo analysis after progressive storage...")
                    anti_silo_results = await self.kg_service.run_global_anti_silo_analysis()
                    if anti_silo_results.get('success'):
                        logger.info(f"✅ Anti-silo complete: {anti_silo_results.get('connections_made', 0)} new connections")
                    else:
                        logger.warning(f"⚠️ Anti-silo analysis failed: {anti_silo_results.get('error')}")
                except Exception as e:
                    logger.error(f"❌ Anti-silo analysis error: {e}")
                    anti_silo_results = {'error': str(e)}
            
            storage_mode = "progressive" if progressive_storage else "batch"
            logger.info(f"🔄 MULTI-CHUNK COMPLETE ({storage_mode}): {total_entities} entities, {total_relationships} relationships (+{cross_chunk_stored} cross-chunk)")
            
            return GraphProcessingResult(
                document_id=document_id,
                total_chunks=len(chunks),
                processed_chunks=len(suitable_chunks),
                total_entities=total_entities,
                total_relationships=total_relationships,
                processing_time_ms=processing_time,
                success=success,
                errors=errors,
                graph_data=all_extraction_results
            )
            
        except Exception as e:
            logger.error(f"Multi-chunk processing failed: {e}")
            # Fallback to individual processing
            return await self._process_chunks_individually(
                chunks, document_id, store_in_neo4j, progressive_storage
            )
    
    async def _process_chunks_individually(self, chunks: List[ExtractedChunk], 
                                         document_id: str, store_in_neo4j: bool, 
                                         progressive_storage: bool = True) -> GraphProcessingResult:
        """Standard individual chunk processing"""
        start_time = datetime.now()
        
        try:
            # Filter chunks suitable for graph processing
            suitable_chunks = self._filter_chunks_for_graph_processing(chunks)
            logger.info(f"Filtered to {len(suitable_chunks)} suitable chunks for graph extraction")
            
            # Process chunks serially with optional progressive storage
            extraction_results = []
            
            for i, chunk in enumerate(suitable_chunks, 1):
                logger.info(f"Processing chunk {i}/{len(suitable_chunks)} for knowledge graph extraction")
                try:
                    result = await self.kg_service.extract_from_chunk(chunk)
                    extraction_results.append(result)
                    
                    # Progressive storage: store immediately after extraction
                    if progressive_storage and store_in_neo4j and (result.entities or result.relationships):
                        try:
                            storage_result = await self.kg_service.store_in_neo4j(result, document_id)
                            if storage_result.get('success'):
                                logger.info(f"📊 Progressive storage: chunk {i} stored ({storage_result.get('entities_stored', 0)} entities, {storage_result.get('relationships_stored', 0)} relationships)")
                            else:
                                logger.warning(f"❌ Progressive storage failed for chunk {i}: {storage_result.get('error')}")
                        except Exception as storage_e:
                            logger.error(f"❌ Progressive storage error for chunk {i}: {storage_e}")
                    
                    logger.info(f"✅ Chunk {i} completed: {len(result.entities)} entities, {len(result.relationships)} relationships")
                except Exception as e:
                    logger.error(f"❌ Chunk {i} extraction failed: {e}")
                    continue
            
            # Collect statistics
            total_entities = sum(len(result.entities) for result in extraction_results)
            total_relationships = sum(len(result.relationships) for result in extraction_results)
            
            logger.info(f"Extracted {total_entities} entities and {total_relationships} relationships")
            
            # Store in Neo4j if requested (only in batch mode - progressive mode already stored)
            storage_errors = []
            if store_in_neo4j and extraction_results and not progressive_storage:
                logger.info(f"Storing graph data in Neo4j for document {document_id} (batch mode)")
                
                for result in extraction_results:
                    if result.entities or result.relationships:
                        storage_result = await self.kg_service.store_in_neo4j(result, document_id)
                        if not storage_result.get('success'):
                            error_msg = f"Failed to store chunk {result.chunk_id}: {storage_result.get('error')}"
                            storage_errors.append(error_msg)
                            logger.error(error_msg)
            elif progressive_storage:
                logger.info(f"Individual chunks stored progressively during extraction")
            
            # Run final anti-silo analysis if progressive storage was used
            anti_silo_results = {}
            if progressive_storage and store_in_neo4j and extraction_results:
                try:
                    logger.info("🌍 Running final anti-silo analysis after progressive storage...")
                    anti_silo_results = await self.kg_service.run_global_anti_silo_analysis()
                    if anti_silo_results.get('success'):
                        logger.info(f"✅ Anti-silo complete: {anti_silo_results.get('connections_made', 0)} new connections")
                    else:
                        logger.warning(f"⚠️ Anti-silo analysis failed: {anti_silo_results.get('error')}")
                except Exception as e:
                    logger.error(f"❌ Anti-silo analysis error: {e}")
                    anti_silo_results = {'error': str(e)}
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create result
            result = GraphProcessingResult(
                document_id=document_id,
                total_chunks=len(chunks),
                processed_chunks=len(extraction_results),
                total_entities=total_entities,
                total_relationships=total_relationships,
                processing_time_ms=processing_time,
                success=len(storage_errors) == 0,
                errors=storage_errors,
                graph_data=extraction_results
            )
            
            storage_mode = "progressive" if progressive_storage else "batch"
            logger.info(f"Graph processing completed for {document_id} ({storage_mode}): {total_entities} entities, {total_relationships} relationships")
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Graph processing failed for document {document_id}: {e}")
            
            return GraphProcessingResult(
                document_id=document_id,
                total_chunks=len(chunks),
                processed_chunks=0,
                total_entities=0,
                total_relationships=0,
                processing_time_ms=processing_time,
                success=False,
                errors=[str(e)]
            )
    
    def _filter_chunks_for_graph_processing(self, chunks: List[ExtractedChunk]) -> List[ExtractedChunk]:
        """Filter and optimize chunks for graph processing based on model capabilities"""
        logger.info(f"🔧 Optimizing {len(chunks)} chunks for model: {self.chunk_sizer.model_name}")
        
        # Use dynamic chunk sizing to optimize chunks for the current model
        optimized_chunks = optimize_chunks_for_model(chunks)
        
        # Apply additional filtering based on content quality
        suitable_chunks = []
        logger.debug(f"🔍 Chunk filtering: min_chunk_length={self.min_chunk_length}, max_chunk_length={self.max_chunk_length}")
        
        for chunk in optimized_chunks:
            content_length = len(chunk.content.strip())
            logger.debug(f"🔍 Checking chunk: {content_length} chars")
            
            # Check minimum length requirement  
            if content_length < self.min_chunk_length:
                logger.debug(f"⏭️  Skipping short chunk: {content_length} chars < {self.min_chunk_length}")
                continue
            
            # Check maximum length (should be handled by optimizer, but safety check)
            if content_length > self.max_chunk_length * 1.1:  # 10% tolerance
                logger.warning(f"⚠️  Chunk too large after optimization: {content_length} chars > {self.max_chunk_length}")
                # Split if still too large
                split_chunks = self._split_large_chunk(chunk)
                suitable_chunks.extend(split_chunks)
            else:
                suitable_chunks.append(chunk)
        
        logger.info(f"✅ Chunk optimization complete: {len(chunks)} → {len(suitable_chunks)} chunks")
        if suitable_chunks:
            avg_size = sum(len(c.content) for c in suitable_chunks) // len(suitable_chunks)
            logger.info(f"   Average chunk size: {avg_size:,} characters")
            logger.info(f"   Size range: {min(len(c.content) for c in suitable_chunks):,} - {max(len(c.content) for c in suitable_chunks):,} characters")
        
        return suitable_chunks
    
    def _create_empty_result(self, document_id: str, total_chunks: int) -> GraphProcessingResult:
        """Create an empty result when no suitable chunks are found"""
        return GraphProcessingResult(
            document_id=document_id,
            total_chunks=total_chunks,
            processed_chunks=0,
            total_entities=0,
            total_relationships=0,
            processing_time_ms=0.0,
            success=True,
            errors=[],
            graph_data=[]
        )
    
    def _split_large_chunk(self, chunk: ExtractedChunk) -> List[ExtractedChunk]:
        """Split a large chunk into smaller, more manageable pieces"""
        content = chunk.content
        sentences = self._split_into_sentences(content)
        
        sub_chunks = []
        current_chunk_text = ""
        current_sentences = []
        
        for sentence in sentences:
            # Check if adding this sentence would exceed the max length
            potential_length = len(current_chunk_text) + len(sentence) + 1
            
            if potential_length > self.max_chunk_length and current_chunk_text:
                # Create sub-chunk
                sub_chunk = ExtractedChunk(
                    content=current_chunk_text.strip(),
                    metadata={
                        **chunk.metadata,
                        'parent_chunk_id': chunk.metadata.get('chunk_id'),
                        'sub_chunk_index': len(sub_chunks),
                        'sentences': current_sentences
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
                    'parent_chunk_id': chunk.metadata.get('chunk_id'),
                    'sub_chunk_index': len(sub_chunks),
                    'sentences': current_sentences
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
    
    async def preview_graph_extraction(
        self, 
        chunks: List[ExtractedChunk], 
        max_chunks: int = 3
    ) -> Dict[str, Any]:
        """
        Generate a preview of what entities and relationships would be extracted.
        Useful for showing users what the graph processing would produce.
        """
        try:
            # Take a sample of chunks for preview
            sample_chunks = chunks[:max_chunks]
            
            preview_extractions = []
            for chunk in sample_chunks:
                extraction = await self.kg_service.extract_from_chunk(chunk)
                preview_extractions.append(extraction)
            
            # Aggregate preview data
            all_entities = []
            all_relationships = []
            
            for extraction in preview_extractions:
                all_entities.extend(extraction.entities)
                all_relationships.extend(extraction.relationships)
            
            # Get unique entity types and relationship types
            entity_types = list(set(entity.label for entity in all_entities))
            relationship_types = list(set(rel.relationship_type for rel in all_relationships))
            
            # Sample entities and relationships for preview
            sample_entities = all_entities[:10]  # Show first 10 entities
            sample_relationships = all_relationships[:5]  # Show first 5 relationships
            
            return {
                'total_entities_found': len(all_entities),
                'total_relationships_found': len(all_relationships),
                'entity_types': entity_types,
                'relationship_types': relationship_types,
                'sample_entities': [
                    {
                        'text': entity.text,
                        'type': entity.label,
                        'confidence': entity.confidence
                    }
                    for entity in sample_entities
                ],
                'sample_relationships': [
                    {
                        'source': rel.source_entity,
                        'target': rel.target_entity,
                        'type': rel.relationship_type,
                        'confidence': rel.confidence,
                        'context': rel.context[:100] + "..." if len(rel.context) > 100 else rel.context
                    }
                    for rel in sample_relationships
                ],
                'processing_time_ms': sum(extraction.processing_time_ms for extraction in preview_extractions)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate graph extraction preview: {e}")
            return {
                'error': str(e),
                'total_entities_found': 0,
                'total_relationships_found': 0,
                'entity_types': [],
                'relationship_types': [],
                'sample_entities': [],
                'sample_relationships': []
            }
    
    def enhance_chunks_with_graph_data(
        self, 
        chunks: List[ExtractedChunk], 
        extraction_results: List[GraphExtractionResult]
    ) -> List[GraphChunk]:
        """
        Enhance regular chunks with graph extraction data to create GraphChunks.
        """
        enhanced_chunks = []
        
        # Create mapping of chunk_id to extraction result
        extraction_map = {}
        for result in extraction_results:
            extraction_map[result.chunk_id] = result
        
        for chunk in chunks:
            chunk_id = chunk.metadata.get('chunk_id', 'unknown')
            extraction = extraction_map.get(chunk_id)
            
            if extraction:
                graph_chunk = GraphChunk(
                    content=chunk.content,
                    metadata=chunk.metadata,
                    quality_score=chunk.quality_score,
                    entities=extraction.entities,
                    relationships=extraction.relationships
                )
            else:
                # No extraction data available
                graph_chunk = GraphChunk(
                    content=chunk.content,
                    metadata=chunk.metadata,
                    quality_score=chunk.quality_score
                )
            
            enhanced_chunks.append(graph_chunk)
        
        return enhanced_chunks

# Singleton instance
_graph_processor: Optional[GraphDocumentProcessor] = None

def get_graph_document_processor() -> GraphDocumentProcessor:
    """Get or create graph document processor singleton"""
    global _graph_processor
    if _graph_processor is None:
        _graph_processor = GraphDocumentProcessor()
    return _graph_processor