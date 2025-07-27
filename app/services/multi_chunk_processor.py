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
    """Service for detecting relationships across chunk boundaries"""
    
    def __init__(self):
        self.config = get_knowledge_graph_settings()
        self.kg_service = get_knowledge_graph_service()
        
        # Configuration for multi-chunk processing
        self.overlap_size = 200  # Characters of overlap between chunks
        self.window_size = 3     # Number of chunks in sliding window
        self.min_cross_chunk_confidence = 0.6
        
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
                                          document_id: str) -> MultiChunkExtractionResult:
        """Process document chunks with overlap to detect cross-chunk relationships"""
        start_time = datetime.now()
        
        if not self.config.get('extraction', {}).get('enable_multi_chunk_relationships', True):
            logger.info("Multi-chunk processing disabled, using standard processing")
            return await self._process_chunks_individually(chunks, document_id)
        
        try:
            logger.info(f"🔄 MULTI-CHUNK: Processing {len(chunks)} chunks with overlap detection")
            
            # Step 1: Process individual chunks
            individual_results = await self._process_individual_chunks(chunks, document_id)
            
            # Step 2: Create sliding windows with overlap
            chunk_windows = self._create_chunk_windows(chunks)
            
            # Step 3: Extract cross-chunk entities and relationships
            cross_chunk_entities, cross_chunk_relationships = await self._extract_cross_chunk_relationships(
                chunk_windows, individual_results
            )
            
            # Step 4: Perform coreference resolution
            resolved_entities, resolved_relationships = await self._resolve_coreferences(
                individual_results, cross_chunk_entities, cross_chunk_relationships
            )
            
            # Step 5: Quality analysis
            quality_metrics = self._analyze_extraction_quality(
                individual_results, resolved_entities, resolved_relationships
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(f"🔄 MULTI-CHUNK COMPLETE: +{len(cross_chunk_entities)} entities, +{len(cross_chunk_relationships)} relationships from overlap")
            
            return MultiChunkExtractionResult(
                individual_results=individual_results,
                cross_chunk_entities=resolved_entities,
                cross_chunk_relationships=resolved_relationships,
                total_processing_time_ms=processing_time,
                overlap_analysis=self._analyze_overlap_effectiveness(chunk_windows, cross_chunk_relationships),
                quality_metrics=quality_metrics
            )
            
        except Exception as e:
            logger.error(f"Multi-chunk processing failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Fallback to individual processing
            individual_results = await self._process_individual_chunks(chunks, document_id)
            return MultiChunkExtractionResult(
                individual_results=individual_results,
                cross_chunk_entities=[],
                cross_chunk_relationships=[],
                total_processing_time_ms=processing_time,
                overlap_analysis={'error': str(e)},
                quality_metrics={'processing_failed': True}
            )
    
    async def _process_individual_chunks(self, chunks: List[ExtractedChunk], 
                                       document_id: str) -> List[GraphExtractionResult]:
        """Process each chunk individually"""
        individual_results = []
        
        # Process chunks in parallel for efficiency
        tasks = []
        for chunk in chunks:
            task = self.kg_service.extract_from_chunk(chunk)
            tasks.append(task)
        
        individual_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log errors
        valid_results = []
        for i, result in enumerate(individual_results):
            if isinstance(result, Exception):
                logger.error(f"Chunk {i} processing failed: {result}")
            else:
                valid_results.append(result)
        
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
        
        for overlap_len in range(max_overlap, 20, -1):  # Minimum 20 chars for meaningful overlap
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
        """Remove duplicate entities"""
        seen = set()
        deduplicated = []
        
        for entity in entities:
            key = (entity.canonical_form.lower(), entity.label)
            if key not in seen:
                seen.add(key)
                deduplicated.append(entity)
        
        return deduplicated
    
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
                                         document_id: str) -> MultiChunkExtractionResult:
        """Fallback to individual chunk processing"""
        individual_results = await self._process_individual_chunks(chunks, document_id)
        
        return MultiChunkExtractionResult(
            individual_results=individual_results,
            cross_chunk_entities=[],
            cross_chunk_relationships=[],
            total_processing_time_ms=0.0,
            overlap_analysis={'method': 'individual_only'},
            quality_metrics={'multi_chunk_disabled': True}
        )

# Singleton instance
_multi_chunk_processor: Optional[MultiChunkRelationshipProcessor] = None

def get_multi_chunk_processor() -> MultiChunkRelationshipProcessor:
    """Get or create multi-chunk processor singleton"""
    global _multi_chunk_processor
    if _multi_chunk_processor is None:
        _multi_chunk_processor = MultiChunkRelationshipProcessor()
    return _multi_chunk_processor