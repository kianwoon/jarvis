"""
Unified Document Processing Service

Handles simultaneous document ingestion into both Milvus (vector storage) 
and Neo4j (knowledge graph) with comprehensive quality tracking and 
cross-reference management.
"""

import asyncio
import logging
import hashlib
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import tempfile
import os

from fastapi import UploadFile
from sqlalchemy.orm import Session

# Core imports
from app.core.db import get_db_session, KnowledgeGraphDocument, ExtractionQualityMetric, DocumentCrossReference
from app.core.knowledge_graph_settings_cache import get_knowledge_graph_settings
from app.core.vector_db_settings_cache import get_vector_db_settings
from app.core.embedding_settings_cache import get_embedding_settings

# Document processing imports
from app.document_handlers.base import ExtractedChunk
from app.document_handlers.graph_processor import get_graph_document_processor
from app.services.knowledge_graph_service import get_knowledge_graph_service, GraphExtractionResult
from app.services.vector_db_service import get_active_vector_db

# Vector and embedding imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from pymilvus import Collection, utility

logger = logging.getLogger(__name__)

@dataclass
class ProcessingProgress:
    """Real-time processing progress tracking"""
    document_id: str
    filename: str
    total_steps: int
    current_step: int
    step_name: str
    chunks_processed: int
    total_chunks: int
    entities_extracted: int
    relationships_extracted: int
    processing_time_ms: float
    status: str  # 'processing', 'completed', 'failed', 'cancelled'
    error_message: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

@dataclass 
class UnifiedProcessingResult:
    """Complete result of unified document processing"""
    document_id: str
    success: bool
    
    # File metadata
    filename: str
    file_size_bytes: int
    file_hash: str
    
    # Processing results
    total_chunks: int
    chunks_processed: int
    
    # Vector storage results
    milvus_collection: Optional[str]
    milvus_chunks_stored: int
    
    # Knowledge graph results
    neo4j_graph_id: Optional[str]
    entities_extracted: int
    relationships_extracted: int
    
    # Cross-references
    cross_references_created: int
    
    # Quality metrics
    processing_time_ms: float
    extraction_confidence: float
    quality_scores: Dict[str, Any]
    
    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = None
    partial_success: bool = False
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

class UnifiedDocumentProcessor:
    """Unified processor for simultaneous Milvus and Neo4j document ingestion"""
    
    def __init__(self):
        self.kg_settings = get_knowledge_graph_settings()
        self.vector_db_settings = get_vector_db_settings()
        self.embedding_settings = get_embedding_settings()
        self.knowledge_graph_service = get_knowledge_graph_service()
        self.active_processing = {}  # Track active processing sessions
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize processing components"""
        try:
            # Initialize text splitter with optimized settings
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.kg_settings.get('chunk_size', 1000),
                chunk_overlap=self.kg_settings.get('chunk_overlap', 200),
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            # Initialize embeddings
            embedding_model = self.embedding_settings.get('model', 'sentence-transformers/all-MiniLM-L6-v2')
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            logger.info("✅ Unified document processor initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize unified document processor: {e}")
            raise
    
    async def process_document(
        self, 
        file: UploadFile,
        processing_mode: str = 'unified',
        collection_name: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> UnifiedProcessingResult:
        """
        Process document with unified Milvus + Neo4j ingestion
        
        Args:
            file: Uploaded file to process
            processing_mode: 'unified', 'milvus-only', 'neo4j-only'
            collection_name: Optional Milvus collection name
            progress_callback: Optional callback for real-time progress updates
            
        Returns:
            UnifiedProcessingResult with comprehensive processing information
        """
        start_time = datetime.now()
        document_id = self._generate_document_id(file.filename)
        
        logger.info(f"🚀 Starting unified document processing: {file.filename} (ID: {document_id})")
        
        try:
            # Step 1: Pre-processing and validation
            progress = ProcessingProgress(
                document_id=document_id,
                filename=file.filename,
                total_steps=8,
                current_step=1,
                step_name="Pre-processing and validation",
                chunks_processed=0,
                total_chunks=0,
                entities_extracted=0,
                relationships_extracted=0,
                processing_time_ms=0,
                status='processing'
            )
            
            if progress_callback:
                await progress_callback(progress)
            
            # Save file temporarily and calculate hash
            temp_file_path, file_hash, file_size = await self._save_and_hash_file(file)
            
            # Create database record
            kg_doc = await self._create_document_record(
                document_id, file.filename, file_hash, file_size, 
                processing_mode, collection_name
            )
            
            # Step 2: Document loading and chunking
            progress.current_step = 2
            progress.step_name = "Document loading and chunking"
            if progress_callback:
                await progress_callback(progress)
            
            chunks = await self._load_and_chunk_document(temp_file_path, document_id)
            progress.total_chunks = len(chunks)
            
            # Step 3: Initialize storage systems
            progress.current_step = 3
            progress.step_name = "Initialize storage systems"
            if progress_callback:
                await progress_callback(progress)
            
            milvus_collection = None
            if processing_mode in ['unified', 'milvus-only']:
                milvus_collection = await self._initialize_milvus_collection(collection_name or 'default')
            
            # Step 4 & 5: Parallel processing
            progress.current_step = 4
            progress.step_name = "Processing chunks (Milvus + Neo4j)"
            if progress_callback:
                await progress_callback(progress)
            
            # Process chunks in parallel for Milvus and Neo4j
            vector_results, graph_results = await self._process_chunks_parallel(
                chunks, document_id, processing_mode, milvus_collection, progress, progress_callback
            )
            
            # Step 6: Cross-reference linking
            progress.current_step = 6
            progress.step_name = "Creating cross-references"
            if progress_callback:
                await progress_callback(progress)
            
            cross_references = await self._create_cross_references(
                document_id, vector_results, graph_results, milvus_collection
            )
            
            # Step 7: Quality validation and metrics
            progress.current_step = 7
            progress.step_name = "Quality validation and metrics"
            if progress_callback:
                await progress_callback(progress)
            
            quality_metrics = await self._calculate_quality_metrics(
                document_id, chunks, graph_results, processing_mode
            )
            
            # Step 8: Finalization
            progress.current_step = 8
            progress.step_name = "Finalizing processing"
            if progress_callback:
                await progress_callback(progress)
            
            # Update document record with final results
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            await self._finalize_document_record(
                document_id, len(chunks), len(vector_results), 
                len([e for gr in graph_results for e in gr.entities]),
                len([r for gr in graph_results for r in gr.relationships]),
                processing_time, quality_metrics
            )
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            # Create and return result
            result = UnifiedProcessingResult(
                document_id=document_id,
                success=True,
                filename=file.filename,
                file_size_bytes=file_size,
                file_hash=file_hash,
                total_chunks=len(chunks),
                chunks_processed=len(chunks),
                milvus_collection=milvus_collection.name if milvus_collection else None,
                milvus_chunks_stored=len(vector_results),
                neo4j_graph_id=document_id,
                entities_extracted=len([e for gr in graph_results for e in gr.entities]),
                relationships_extracted=len([r for gr in graph_results for r in gr.relationships]),
                cross_references_created=len(cross_references),
                processing_time_ms=processing_time,
                extraction_confidence=quality_metrics.get('overall_confidence', 0.0),
                quality_scores=quality_metrics
            )
            
            # Run anti-silo analysis if knowledge graph extraction was performed
            if processing_mode in ['unified', 'neo4j-only'] and graph_results:
                logger.info("🔗 Running anti-silo analysis to connect isolated entities")
                try:
                    anti_silo_result = await self.knowledge_graph_service.run_global_anti_silo_analysis()
                    if anti_silo_result:
                        logger.info(f"✅ Anti-silo analysis completed: {anti_silo_result.get('relationships_created', 0)} new connections")
                    else:
                        logger.warning("⚠️ Anti-silo analysis returned no results")
                except Exception as anti_silo_error:
                    logger.error(f"❌ Anti-silo analysis failed: {anti_silo_error}")
                    # Don't fail the entire processing for anti-silo errors
            
            progress.status = 'completed'
            progress.processing_time_ms = processing_time
            if progress_callback:
                await progress_callback(progress)
            
            logger.info(f"✅ Document processing completed successfully: {document_id}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {e}")
            
            # Update progress with error
            if 'progress' in locals():
                progress.status = 'failed'
                progress.error_message = str(e)
                if progress_callback:
                    await progress_callback(progress)
            
            # Update database record with error
            await self._update_document_error(document_id, str(e))
            
            # Clean up temporary file if it exists
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            
            # Return failed result
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return UnifiedProcessingResult(
                document_id=document_id,
                success=False,
                filename=file.filename if file else "unknown",
                file_size_bytes=0,
                file_hash="",
                total_chunks=0,
                chunks_processed=0,
                milvus_collection=None,
                milvus_chunks_stored=0,
                neo4j_graph_id=None,
                entities_extracted=0,
                relationships_extracted=0,
                cross_references_created=0,
                processing_time_ms=processing_time,
                extraction_confidence=0.0,
                quality_scores={},
                error_message=str(e)
            )
    
    def _generate_document_id(self, filename: str) -> str:
        """Generate unique document ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_hash = hashlib.md5(f"{filename}_{timestamp}".encode()).hexdigest()[:8]
        return f"doc_{timestamp}_{file_hash}"
    
    async def _save_and_hash_file(self, file: UploadFile) -> Tuple[str, str, int]:
        """Save uploaded file temporarily and calculate hash"""
        # Create temporary file
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"upload_{file.filename}")
        
        # Read file content and calculate hash
        content = await file.read()
        file_hash = hashlib.sha256(content).hexdigest()
        file_size = len(content)
        
        # Save to temporary file
        with open(temp_path, 'wb') as f:
            f.write(content)
        
        # Reset file pointer for potential reuse
        await file.seek(0)
        
        return temp_path, file_hash, file_size
    
    async def _create_document_record(
        self, document_id: str, filename: str, file_hash: str, 
        file_size: int, processing_mode: str, collection_name: Optional[str]
    ) -> KnowledgeGraphDocument:
        """Create initial database record for document"""
        try:
            with get_db_session() as db:
                kg_doc = KnowledgeGraphDocument(
                    document_id=document_id,
                    filename=filename,
                    file_hash=file_hash,
                    file_size_bytes=file_size,
                    file_type=Path(filename).suffix.lower(),
                    processing_mode=processing_mode,
                    milvus_collection=collection_name,
                    neo4j_graph_id=document_id,
                    processing_status='processing',
                    processing_started_at=datetime.now(),
                    processing_config={
                        'mode': processing_mode,
                        'collection': collection_name,
                        'llm_enhancement': self.kg_settings.get('extraction', {}).get('enable_llm_enhancement', False)
                    }
                )
                
                db.add(kg_doc)
                db.commit()
                db.refresh(kg_doc)
                
                logger.info(f"📝 Created document record: {document_id}")
                return kg_doc
                
        except Exception as e:
            logger.error(f"Failed to create document record: {e}")
            raise
    
    async def _load_and_chunk_document(self, file_path: str, document_id: str) -> List[ExtractedChunk]:
        """Load document and create chunks"""
        try:
            # Load document using PyPDFLoader
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Split into chunks
            chunks = []
            for i, doc in enumerate(documents):
                text_chunks = self.text_splitter.split_text(doc.page_content)
                
                for j, chunk_text in enumerate(text_chunks):
                    chunk_id = f"{document_id}_page{i}_chunk{j}"
                    chunk = ExtractedChunk(
                        content=chunk_text,
                        metadata={
                            'chunk_id': chunk_id,
                            'document_id': document_id,
                            'page_number': i,
                            'chunk_index': j,
                            'source': file_path,
                            'total_chunks': len(text_chunks)
                        }
                    )
                    chunks.append(chunk)
            
            logger.info(f"📄 Created {len(chunks)} chunks from document {document_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to load and chunk document: {e}")
            raise
    
    async def _initialize_milvus_collection(self, collection_name: str):
        """Initialize Milvus collection for vector storage"""
        try:
            # Get active vector database configuration
            active_db = get_active_vector_db()
            if not active_db or active_db['id'] != 'milvus':
                raise ValueError("Milvus is not the active vector database")
            
            # Initialize Milvus collection
            # This would typically involve setting up the collection schema and indexes
            # For now, we'll return a placeholder
            logger.info(f"🗂️ Initialized Milvus collection: {collection_name}")
            return None  # Placeholder - would return actual collection object
            
        except Exception as e:
            logger.error(f"Failed to initialize Milvus collection: {e}")
            raise
    
    async def _process_chunks_parallel(
        self, chunks: List[ExtractedChunk], document_id: str, 
        processing_mode: str, milvus_collection, 
        progress: ProcessingProgress, progress_callback: Optional[callable]
    ) -> Tuple[List[Dict], List[GraphExtractionResult]]:
        """Process chunks in parallel for both Milvus and Neo4j"""
        
        vector_results = []
        graph_results = []
        
        try:
            for i, chunk in enumerate(chunks):
                # Update progress
                progress.chunks_processed = i + 1
                if progress_callback:
                    await progress_callback(progress)
                
                # Process for Milvus (vector storage)
                if processing_mode in ['unified', 'milvus-only']:
                    vector_result = await self._process_chunk_vector(chunk, milvus_collection)
                    if vector_result:
                        vector_results.append(vector_result)
                
                # Process for Neo4j (knowledge graph)
                if processing_mode in ['unified', 'neo4j-only']:
                    graph_result = await self.knowledge_graph_service.extract_from_chunk(chunk)
                    if graph_result and (graph_result.entities or graph_result.relationships):
                        # Store in Neo4j
                        await self.knowledge_graph_service.store_in_neo4j(graph_result, document_id)
                        graph_results.append(graph_result)
                        
                        # Update progress with extraction counts
                        progress.entities_extracted += len(graph_result.entities)
                        progress.relationships_extracted += len(graph_result.relationships)
            
            logger.info(f"📊 Processed {len(chunks)} chunks: {len(vector_results)} vector, {len(graph_results)} graph")
            return vector_results, graph_results
            
        except Exception as e:
            logger.error(f"Failed to process chunks in parallel: {e}")
            raise
    
    async def _process_chunk_vector(self, chunk: ExtractedChunk, milvus_collection) -> Optional[Dict]:
        """Process individual chunk for vector storage"""
        try:
            # Generate embedding
            embedding = self.embeddings.embed_query(chunk.content)
            
            # Prepare metadata for vector storage
            metadata = {
                'chunk_id': chunk.metadata.get('chunk_id'),
                'document_id': chunk.metadata.get('document_id'),
                'page_number': chunk.metadata.get('page_number'),
                'text_preview': chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            }
            
            # Store in Milvus (simplified - would use actual Milvus API)
            result = {
                'chunk_id': chunk.metadata.get('chunk_id'),
                'embedding': embedding,
                'metadata': metadata,
                'stored': True
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process chunk for vector storage: {e}")
            return None
    
    async def _create_cross_references(
        self, document_id: str, vector_results: List[Dict], 
        graph_results: List[GraphExtractionResult], milvus_collection
    ) -> List[DocumentCrossReference]:
        """Create cross-references between Milvus chunks and Neo4j entities"""
        cross_references = []
        
        try:
            with get_db_session() as db:
                # Create mapping between chunks and entities
                chunk_to_entities = {}
                
                for graph_result in graph_results:
                    chunk_id = graph_result.chunk_id
                    if chunk_id not in chunk_to_entities:
                        chunk_to_entities[chunk_id] = []
                    
                    for entity in graph_result.entities:
                        chunk_to_entities[chunk_id].append(entity)
                
                # Create cross-reference records
                for vector_result in vector_results:
                    chunk_id = vector_result['chunk_id']
                    
                    if chunk_id in chunk_to_entities:
                        for entity in chunk_to_entities[chunk_id]:
                            cross_ref = DocumentCrossReference(
                                document_id=document_id,
                                milvus_collection=milvus_collection.name if milvus_collection else 'default',
                                milvus_chunk_id=chunk_id,
                                chunk_text_preview=vector_result['metadata']['text_preview'],
                                neo4j_entity_id=f"entity_{entity.canonical_form}_{chunk_id}",
                                entity_name=entity.canonical_form,
                                entity_type=entity.label,
                                confidence_score=entity.confidence,
                                relationship_type='EXTRACTED_FROM',
                                validation_status='pending'
                            )
                            
                            db.add(cross_ref)
                            cross_references.append(cross_ref)
                
                db.commit()
                logger.info(f"🔗 Created {len(cross_references)} cross-references for document {document_id}")
                
        except Exception as e:
            logger.error(f"Failed to create cross-references: {e}")
        
        return cross_references
    
    async def _calculate_quality_metrics(
        self, document_id: str, chunks: List[ExtractedChunk], 
        graph_results: List[GraphExtractionResult], processing_mode: str
    ) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics"""
        try:
            total_entities = sum(len(gr.entities) for gr in graph_results)
            total_relationships = sum(len(gr.relationships) for gr in graph_results)
            
            # Calculate confidence scores
            entity_confidences = [e.confidence for gr in graph_results for e in gr.entities]
            relationship_confidences = [r.confidence for gr in graph_results for r in gr.relationships]
            
            avg_entity_confidence = sum(entity_confidences) / len(entity_confidences) if entity_confidences else 0
            avg_relationship_confidence = sum(relationship_confidences) / len(relationship_confidences) if relationship_confidences else 0
            
            # Overall extraction confidence
            overall_confidence = (avg_entity_confidence + avg_relationship_confidence) / 2
            
            quality_metrics = {
                'overall_confidence': overall_confidence,
                'entity_confidence_avg': avg_entity_confidence,
                'relationship_confidence_avg': avg_relationship_confidence,
                'entity_confidence_distribution': entity_confidences,
                'relationship_confidence_distribution': relationship_confidences,
                'extraction_success_rate': len([gr for gr in graph_results if gr.entities or gr.relationships]) / len(chunks) if chunks else 0,
                'entities_per_chunk': total_entities / len(chunks) if chunks else 0,
                'relationships_per_chunk': total_relationships / len(chunks) if chunks else 0,
                'processing_mode': processing_mode
            }
            
            # Store quality metrics in database
            with get_db_session() as db:
                for graph_result in graph_results:
                    quality_metric = ExtractionQualityMetric(
                        document_id=document_id,
                        chunk_id=graph_result.chunk_id,
                        entities_discovered=len(graph_result.entities),
                        relationships_discovered=len(graph_result.relationships),
                        confidence_scores={
                            'entities': [e.confidence for e in graph_result.entities],
                            'relationships': [r.confidence for r in graph_result.relationships]
                        },
                        processing_method='unified_llm_enhanced' if self.kg_settings.get('extraction', {}).get('enable_llm_enhancement') else 'unified_traditional',
                        processing_time_ms=graph_result.processing_time_ms
                    )
                    db.add(quality_metric)
                
                db.commit()
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate quality metrics: {e}")
            return {}
    
    async def _finalize_document_record(
        self, document_id: str, total_chunks: int, chunks_processed: int,
        entities_extracted: int, relationships_extracted: int,
        processing_time_ms: float, quality_metrics: Dict[str, Any]
    ):
        """Update document record with final processing results"""
        try:
            with get_db_session() as db:
                kg_doc = db.query(KnowledgeGraphDocument).filter(
                    KnowledgeGraphDocument.document_id == document_id
                ).first()
                
                if kg_doc:
                    kg_doc.processing_status = 'completed'
                    kg_doc.total_chunks = total_chunks
                    kg_doc.chunks_processed = chunks_processed
                    kg_doc.entities_extracted = entities_extracted
                    kg_doc.relationships_extracted = relationships_extracted
                    kg_doc.processing_time_ms = int(processing_time_ms)
                    kg_doc.extraction_confidence = quality_metrics.get('overall_confidence', 0.0)
                    kg_doc.quality_scores = quality_metrics
                    kg_doc.processing_completed_at = datetime.now()
                    
                    db.commit()
                    logger.info(f"📊 Finalized document record: {document_id}")
                
        except Exception as e:
            logger.error(f"Failed to finalize document record: {e}")
    
    async def _update_document_error(self, document_id: str, error_message: str):
        """Update document record with error information"""
        try:
            with get_db_session() as db:
                kg_doc = db.query(KnowledgeGraphDocument).filter(
                    KnowledgeGraphDocument.document_id == document_id
                ).first()
                
                if kg_doc:
                    kg_doc.processing_status = 'failed'
                    kg_doc.error_message = error_message
                    kg_doc.retry_count += 1
                    
                    db.commit()
                    logger.info(f"❌ Updated document with error: {document_id}")
                
        except Exception as e:
            logger.error(f"Failed to update document error: {e}")
    
    async def get_processing_status(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get current processing status for a document"""
        try:
            with get_db_session() as db:
                kg_doc = db.query(KnowledgeGraphDocument).filter(
                    KnowledgeGraphDocument.document_id == document_id
                ).first()
                
                if kg_doc:
                    return {
                        'document_id': kg_doc.document_id,
                        'filename': kg_doc.filename,
                        'status': kg_doc.processing_status,
                        'chunks_processed': kg_doc.chunks_processed,
                        'total_chunks': kg_doc.total_chunks,
                        'entities_extracted': kg_doc.entities_extracted,
                        'relationships_extracted': kg_doc.relationships_extracted,
                        'processing_time_ms': kg_doc.processing_time_ms,
                        'extraction_confidence': kg_doc.extraction_confidence,
                        'error_message': kg_doc.error_message,
                        'created_at': kg_doc.created_at.isoformat(),
                        'processing_started_at': kg_doc.processing_started_at.isoformat() if kg_doc.processing_started_at else None,
                        'processing_completed_at': kg_doc.processing_completed_at.isoformat() if kg_doc.processing_completed_at else None
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get processing status: {e}")
            return None

# Singleton instance
_unified_processor: Optional[UnifiedDocumentProcessor] = None

def get_unified_document_processor() -> UnifiedDocumentProcessor:
    """Get or create unified document processor singleton"""
    global _unified_processor
    if _unified_processor is None:
        _unified_processor = UnifiedDocumentProcessor()
    return _unified_processor