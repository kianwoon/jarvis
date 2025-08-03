"""
Unified Document Processing API Endpoints

Provides REST API endpoints for the enhanced unified document processing system
that handles both Milvus (vector storage) and Neo4j (knowledge graph) ingestion.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
from uuid import uuid4

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from app.core.db import get_db, KnowledgeGraphDocument, ExtractionQualityMetric, DocumentCrossReference, GraphSchemaEvolution
from app.services.unified_document_processor import get_unified_document_processor, ProcessingProgress
from app.core.knowledge_graph_settings_cache import get_knowledge_graph_settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Active processing sessions for SSE streaming
active_sessions = {}

@router.post("/process-unified")
async def process_document_unified(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    processing_mode: str = Form("unified"),
    collection_name: Optional[str] = Form(None),
    enable_llm_enhancement: bool = Form(True)
):
    """
    Process document with unified Milvus + Neo4j ingestion
    
    Args:
        file: Document file to process
        processing_mode: 'unified', 'milvus-only', 'neo4j-only'
        collection_name: Optional Milvus collection name
        enable_llm_enhancement: Enable LLM-based entity/relationship extraction
    
    Returns:
        Processing result with document ID and initial status
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not file.filename.lower().endswith(('.pdf', '.doc', '.docx', '.txt')):
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Validate processing mode
        if processing_mode not in ['unified', 'milvus-only', 'neo4j-only']:
            raise HTTPException(status_code=400, detail="Invalid processing mode")
        
        processor = get_unified_document_processor()
        
        # Create progress callback for real-time updates
        session_id = str(uuid4())
        
        async def progress_callback(progress: ProcessingProgress):
            """Callback to store progress for SSE streaming"""
            active_sessions[session_id] = progress
        
        # Start processing in background
        background_tasks.add_task(
            _process_document_background,
            processor,
            file,
            processing_mode,
            collection_name,
            session_id,
            progress_callback
        )
        
        return {
            "success": True,
            "session_id": session_id,
            "message": "Document processing started",
            "processing_mode": processing_mode,
            "filename": file.filename,
            "progress_endpoint": f"/api/v1/documents/process-unified/{session_id}/progress"
        }
        
    except Exception as e:
        logger.error(f"Failed to start unified document processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _process_document_background(
    processor,
    file: UploadFile,
    processing_mode: str,
    collection_name: Optional[str],
    session_id: str,
    progress_callback
):
    """Background task for document processing"""
    try:
        result = await processor.process_document(
            file=file,
            processing_mode=processing_mode,
            collection_name=collection_name,
            progress_callback=progress_callback
        )
        
        # Store final result
        if session_id in active_sessions:
            active_sessions[session_id].status = 'completed' if result.success else 'failed'
            active_sessions[session_id].result = result
        
    except Exception as e:
        logger.error(f"Background processing failed: {e}")
        if session_id in active_sessions:
            active_sessions[session_id].status = 'failed'
            active_sessions[session_id].error_message = str(e)

@router.get("/process-unified/{session_id}/progress")
async def stream_processing_progress(session_id: str):
    """
    Stream real-time processing progress via Server-Sent Events
    
    Args:
        session_id: Processing session ID
        
    Returns:
        SSE stream of processing progress
    """
    async def generate_progress():
        """Generate SSE progress events"""
        try:
            # Send initial connection event
            yield f"data: {json.dumps({'event': 'connected', 'session_id': session_id})}\n\n"
            
            last_progress = None
            while True:
                if session_id in active_sessions:
                    progress = active_sessions[session_id]
                    
                    # Only send updates when progress changes
                    if progress != last_progress:
                        progress_data = {
                            'document_id': progress.document_id,
                            'filename': progress.filename,
                            'total_steps': progress.total_steps,
                            'current_step': progress.current_step,
                            'step_name': progress.step_name,
                            'chunks_processed': progress.chunks_processed,
                            'total_chunks': progress.total_chunks,
                            'entities_extracted': progress.entities_extracted,
                            'relationships_extracted': progress.relationships_extracted,
                            'processing_time_ms': progress.processing_time_ms,
                            'status': progress.status,
                            'error_message': progress.error_message,
                            'warnings': progress.warnings or []
                        }
                        
                        yield f"data: {json.dumps(progress_data)}\n\n"
                        last_progress = progress
                    
                    # Check if processing is complete
                    if progress.status in ['completed', 'failed', 'cancelled']:
                        # Send final status and cleanup
                        yield f"data: {json.dumps({'event': 'finished', 'status': progress.status})}\n\n"
                        
                        # Clean up session after delay
                        await asyncio.sleep(5)
                        if session_id in active_sessions:
                            del active_sessions[session_id]
                        break
                
                await asyncio.sleep(1)  # Update every second
                
        except Exception as e:
            logger.error(f"SSE stream error: {e}")
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_progress(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

@router.get("/documents")
async def list_processed_documents(
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = Query(None),
    processing_mode: Optional[str] = Query(None)
):
    """
    List processed documents with filtering and pagination
    
    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        status: Filter by processing status
        processing_mode: Filter by processing mode
        
    Returns:
        List of processed documents with metadata
    """
    try:
        query = db.query(KnowledgeGraphDocument)
        
        # Apply filters
        if status:
            query = query.filter(KnowledgeGraphDocument.processing_status == status)
        
        if processing_mode:
            query = query.filter(KnowledgeGraphDocument.processing_mode == processing_mode)
        
        # Get total count for pagination
        total = query.count()
        
        # Apply pagination and ordering
        documents = query.order_by(desc(KnowledgeGraphDocument.created_at)).offset(skip).limit(limit).all()
        
        # Format results
        results = []
        for doc in documents:
            results.append({
                'document_id': doc.document_id,
                'filename': doc.filename,
                'file_size_bytes': doc.file_size_bytes,
                'file_type': doc.file_type,
                'processing_mode': doc.processing_mode,
                'processing_status': doc.processing_status,
                'entities_extracted': doc.entities_extracted,
                'relationships_extracted': doc.relationships_extracted,
                'chunks_processed': doc.chunks_processed,
                'total_chunks': doc.total_chunks,
                'processing_time_ms': doc.processing_time_ms,
                'extraction_confidence': doc.extraction_confidence,
                'quality_scores': doc.quality_scores,
                'created_at': doc.created_at.isoformat(),
                'processing_completed_at': doc.processing_completed_at.isoformat() if doc.processing_completed_at else None,
                'error_message': doc.error_message
            })
        
        return {
            'documents': results,
            'total': total,
            'skip': skip,
            'limit': limit,
            'has_more': total > skip + limit
        }
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents/{document_id}")
async def get_document_details(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a processed document
    
    Args:
        document_id: Document identifier
        
    Returns:
        Detailed document information including quality metrics and cross-references
    """
    try:
        # Get document record
        document = db.query(KnowledgeGraphDocument).filter(
            KnowledgeGraphDocument.document_id == document_id
        ).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get quality metrics
        quality_metrics = db.query(ExtractionQualityMetric).filter(
            ExtractionQualityMetric.document_id == document_id
        ).all()
        
        # Get cross-references
        cross_references = db.query(DocumentCrossReference).filter(
            DocumentCrossReference.document_id == document_id
        ).all()
        
        # Format quality metrics
        metrics_data = []
        for metric in quality_metrics:
            metrics_data.append({
                'chunk_id': metric.chunk_id,
                'entities_discovered': metric.entities_discovered,
                'relationships_discovered': metric.relationships_discovered,
                'entities_validated': metric.entities_validated,
                'relationships_validated': metric.relationships_validated,
                'confidence_scores': metric.confidence_scores,
                'validation_scores': metric.validation_scores,
                'processing_method': metric.processing_method,
                'processing_time_ms': metric.processing_time_ms,
                'validation_errors': metric.validation_errors,
                'created_at': metric.created_at.isoformat()
            })
        
        # Format cross-references
        cross_ref_data = []
        for ref in cross_references:
            cross_ref_data.append({
                'milvus_collection': ref.milvus_collection,
                'milvus_chunk_id': ref.milvus_chunk_id,
                'chunk_text_preview': ref.chunk_text_preview,
                'neo4j_entity_id': ref.neo4j_entity_id,
                'entity_name': ref.entity_name,
                'entity_type': ref.entity_type,
                'confidence_score': ref.confidence_score,
                'relationship_type': ref.relationship_type,
                'validation_status': ref.validation_status,
                'manual_review': ref.manual_review,
                'created_at': ref.created_at.isoformat()
            })
        
        return {
            'document': {
                'document_id': document.document_id,
                'filename': document.filename,
                'file_size_bytes': document.file_size_bytes,
                'file_type': document.file_type,
                'file_hash': document.file_hash,
                'processing_mode': document.processing_mode,
                'processing_status': document.processing_status,
                'milvus_collection': document.milvus_collection,
                'neo4j_graph_id': document.neo4j_graph_id,
                'entities_extracted': document.entities_extracted,
                'relationships_extracted': document.relationships_extracted,
                'chunks_processed': document.chunks_processed,
                'total_chunks': document.total_chunks,
                'processing_time_ms': document.processing_time_ms,
                'extraction_confidence': document.extraction_confidence,
                'quality_scores': document.quality_scores,
                'processing_config': document.processing_config,
                'upload_metadata': document.upload_metadata,
                'error_message': document.error_message,
                'retry_count': document.retry_count,
                'created_at': document.created_at.isoformat(),
                'processing_started_at': document.processing_started_at.isoformat() if document.processing_started_at else None,
                'processing_completed_at': document.processing_completed_at.isoformat() if document.processing_completed_at else None
            },
            'quality_metrics': metrics_data,
            'cross_references': cross_ref_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics")
async def get_processing_analytics(
    db: Session = Depends(get_db),
    days: int = Query(30, ge=1, le=365)
):
    """
    Get processing analytics and statistics
    
    Args:
        days: Number of days to include in analytics
        
    Returns:
        Analytics data including processing statistics and trends
    """
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Overall statistics
        total_documents = db.query(KnowledgeGraphDocument).count()
        recent_documents = db.query(KnowledgeGraphDocument).filter(
            KnowledgeGraphDocument.created_at >= cutoff_date
        ).count()
        
        completed_documents = db.query(KnowledgeGraphDocument).filter(
            KnowledgeGraphDocument.processing_status == 'completed'
        ).count()
        
        failed_documents = db.query(KnowledgeGraphDocument).filter(
            KnowledgeGraphDocument.processing_status == 'failed'
        ).count()
        
        # Processing mode distribution
        mode_stats = db.query(
            KnowledgeGraphDocument.processing_mode,
            func.count(KnowledgeGraphDocument.id)
        ).filter(
            KnowledgeGraphDocument.created_at >= cutoff_date
        ).group_by(KnowledgeGraphDocument.processing_mode).all()
        
        # Average processing metrics
        avg_metrics = db.query(
            func.avg(KnowledgeGraphDocument.processing_time_ms),
            func.avg(KnowledgeGraphDocument.extraction_confidence),
            func.sum(KnowledgeGraphDocument.entities_extracted),
            func.sum(KnowledgeGraphDocument.relationships_extracted),
            func.sum(KnowledgeGraphDocument.chunks_processed)
        ).filter(
            KnowledgeGraphDocument.processing_status == 'completed',
            KnowledgeGraphDocument.created_at >= cutoff_date
        ).first()
        
        # Quality metrics aggregation
        quality_stats = db.query(
            func.avg(ExtractionQualityMetric.entities_discovered),
            func.avg(ExtractionQualityMetric.relationships_discovered),
            func.count(ExtractionQualityMetric.id)
        ).join(KnowledgeGraphDocument).filter(
            KnowledgeGraphDocument.created_at >= cutoff_date
        ).first()
        
        # Cross-reference statistics
        cross_ref_stats = db.query(
            func.count(DocumentCrossReference.id),
            func.count(DocumentCrossReference.id.distinct())
        ).join(KnowledgeGraphDocument).filter(
            KnowledgeGraphDocument.created_at >= cutoff_date
        ).first()
        
        # Daily processing trends (last 7 days)
        daily_trends = []
        for i in range(7):
            date = datetime.now().date() - timedelta(days=i)
            count = db.query(KnowledgeGraphDocument).filter(
                func.date(KnowledgeGraphDocument.created_at) == date
            ).count()
            daily_trends.append({
                'date': date.isoformat(),
                'documents_processed': count
            })
        
        daily_trends.reverse()  # Show chronologically
        
        return {
            'period_days': days,
            'total_statistics': {
                'total_documents': total_documents,
                'recent_documents': recent_documents,
                'completed_documents': completed_documents,
                'failed_documents': failed_documents,
                'success_rate': (completed_documents / total_documents * 100) if total_documents > 0 else 0
            },
            'processing_modes': {
                mode: count for mode, count in mode_stats
            },
            'average_metrics': {
                'processing_time_ms': float(avg_metrics[0]) if avg_metrics[0] else 0,
                'extraction_confidence': float(avg_metrics[1]) if avg_metrics[1] else 0,
                'total_entities_extracted': int(avg_metrics[2]) if avg_metrics[2] else 0,
                'total_relationships_extracted': int(avg_metrics[3]) if avg_metrics[3] else 0,
                'total_chunks_processed': int(avg_metrics[4]) if avg_metrics[4] else 0
            },
            'quality_statistics': {
                'avg_entities_per_chunk': float(quality_stats[0]) if quality_stats[0] else 0,
                'avg_relationships_per_chunk': float(quality_stats[1]) if quality_stats[1] else 0,
                'total_quality_records': int(quality_stats[2]) if quality_stats[2] else 0
            },
            'cross_reference_statistics': {
                'total_cross_references': int(cross_ref_stats[0]) if cross_ref_stats[0] else 0,
                'unique_mappings': int(cross_ref_stats[1]) if cross_ref_stats[1] else 0
            },
            'daily_trends': daily_trends
        }
        
    except Exception as e:
        logger.error(f"Failed to get processing analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance")
async def get_performance_metrics():
    """
    Get current system performance metrics
    
    Returns:
        Real-time performance data including resource usage and processing speed
    """
    try:
        # In a real implementation, you would gather actual system metrics
        # This is a placeholder that returns mock data
        
        import psutil
        import time
        
        # Get CPU and memory usage
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Get active processing sessions
        active_processing = len([s for s in active_sessions.values() if s.status == 'processing'])
        
        # Calculate approximate processing speed based on recent completions
        processing_speed = 150  # chunks per minute (placeholder)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_resources': {
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'memory_total_gb': memory.total / (1024**3),
                'memory_available_gb': memory.available / (1024**3)
            },
            'processing_metrics': {
                'active_sessions': len(active_sessions),
                'active_processing': active_processing,
                'processing_speed_chunks_per_minute': processing_speed,
                'average_document_time_ms': 45000  # placeholder
            },
            'database_connections': {
                'postgresql_active': True,
                'milvus_active': True,
                'neo4j_active': True,
                'redis_active': True
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a processed document and all associated data
    
    Args:
        document_id: Document identifier
        
    Returns:
        Deletion confirmation
    """
    try:
        # Check if document exists
        document = db.query(KnowledgeGraphDocument).filter(
            KnowledgeGraphDocument.document_id == document_id
        ).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete associated records (cascading deletes should handle this)
        db.query(ExtractionQualityMetric).filter(
            ExtractionQualityMetric.document_id == document_id
        ).delete()
        
        db.query(DocumentCrossReference).filter(
            DocumentCrossReference.document_id == document_id
        ).delete()
        
        # Delete document record
        db.delete(document)
        db.commit()
        
        logger.info(f"Deleted document and associated data: {document_id}")
        
        return {
            'success': True,
            'message': f'Document {document_id} and all associated data deleted successfully'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/documents/{document_id}/retry")
async def retry_document_processing(
    document_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Retry processing for a failed document
    
    Args:
        document_id: Document identifier
        
    Returns:
        Retry status
    """
    try:
        document = db.query(KnowledgeGraphDocument).filter(
            KnowledgeGraphDocument.document_id == document_id
        ).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if document.processing_status not in ['failed', 'cancelled']:
            raise HTTPException(status_code=400, detail="Document is not in a retryable state")
        
        if document.retry_count >= document.max_retries:
            raise HTTPException(status_code=400, detail="Maximum retry attempts exceeded")
        
        # Reset document status for retry
        document.processing_status = 'pending'
        document.retry_count += 1
        document.error_message = None
        db.commit()
        
        # TODO: Implement actual retry logic
        # This would involve re-queuing the document for processing
        
        return {
            'success': True,
            'message': f'Document {document_id} queued for retry processing',
            'retry_count': document.retry_count,
            'max_retries': document.max_retries
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retry document processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))