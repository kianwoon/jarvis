"""
IDC (Intelligent Document Comparison) API Endpoints

Provides REST API for IDC functionality following Jarvis patterns:
- Uses FastAPI with dependency injection
- PostgreSQL database integration
- Redis for progress tracking
- Ollama model integration
- No hardcoded values - uses settings classes
"""

import asyncio
import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import uuid4
import os

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.db import get_db, get_db_session, IDCReferenceDocument, IDCValidationSession, IDCUnitValidationResult, IDCTemplate, Settings
from app.core.config import get_settings
from app.core.llm_settings_cache import get_llm_settings
from app.core.idc_settings_cache import (
    get_idc_settings,
    set_idc_settings,
    reload_idc_settings,
    get_extraction_config,
    get_validation_config
)
from app.services.idc_extraction_service import IDCExtractionService, ExtractionMode, GranularExtractionConfig
from app.services.idc_validation_service import IDCValidationService, ValidationConfig
from app.services.idc_reference_manager import IDCReferenceManager

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic models for API requests/responses
class ReferenceDocumentUpload(BaseModel):
    name: str = Field(..., description="Name of the reference document")
    document_type: str = Field(..., description="Type of document (contract, exam, resume, etc.)")
    category: Optional[str] = Field(None, description="Document category")
    extraction_model: Optional[str] = Field(None, description="Ollama model to use for extraction")
    recommended_modes: Optional[List[str]] = Field(None, description="Recommended extraction modes")

class ReferenceDocumentUpdate(BaseModel):
    name: Optional[str] = Field(None, description="Updated name of the reference document")
    document_type: Optional[str] = Field(None, description="Updated document type")
    category: Optional[str] = Field(None, description="Updated document category")
    extraction_model: Optional[str] = Field(None, description="Updated extraction model")
    content: Optional[str] = Field(None, description="Updated markdown content")
    re_extract: bool = Field(False, description="Whether to re-extract content with new model")

class ValidationRequest(BaseModel):
    reference_id: str = Field(..., description="Reference document ID")
    extraction_mode: str = Field(..., description="Extraction mode (sentence/paragraph/qa_pairs/section)")
    validation_model: Optional[str] = Field(None, description="Ollama model for validation")
    max_context_usage: float = Field(0.35, description="Maximum context usage (0.0-1.0)")
    preserve_context: bool = Field(True, description="Whether to preserve context in extraction")
    quality_threshold: float = Field(0.8, description="Quality threshold for human review")

class ConfigurationUpdate(BaseModel):
    # Extraction settings
    extraction_model: Optional[str] = None
    extraction_system_prompt: Optional[str] = None
    extraction_temperature: Optional[float] = None
    extraction_max_tokens: Optional[int] = None
    extraction_max_context_usage: Optional[float] = None
    extraction_confidence_threshold: Optional[float] = None
    enable_chunking: Optional[bool] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    
    # Validation settings
    validation_model: Optional[str] = None
    validation_system_prompt: Optional[str] = None
    validation_temperature: Optional[float] = None
    validation_max_tokens: Optional[int] = None
    max_context_usage: Optional[float] = None
    quality_threshold: Optional[float] = None
    enable_structured_output: Optional[bool] = None
    
    # Comparison settings
    comparison_algorithm: Optional[str] = None
    similarity_threshold: Optional[float] = None
    fuzzy_threshold: Optional[float] = None
    ignore_case: Optional[bool] = None
    ignore_whitespace: Optional[bool] = None
    enable_fuzzy_matching: Optional[bool] = None

class TemplateCreate(BaseModel):
    name: str
    description: str
    template_type: str
    reference_document_id: str
    default_extraction_mode: str
    validation_config: Dict[str, Any]

# Initialize services
extraction_service = IDCExtractionService()
validation_service = IDCValidationService()
reference_manager = IDCReferenceManager()

@router.post("/reference/upload")
async def upload_reference_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Reference document file"),
    name: str = Form(..., description="Document name"),
    document_type: str = Form(..., description="Document type"),
    category: Optional[str] = Form(None, description="Document category"),
    extraction_model: Optional[str] = Form(None, description="Ollama model for extraction"),
    recommended_modes: str = Form("paragraph,sentence", description="Comma-separated extraction modes"),
    db: Session = Depends(get_db)
):
    """
    Upload reference document and extract to structured markdown
    
    Returns:
        - document_id: Unique identifier
        - extraction_preview: First 500 chars of extracted markdown
        - recommended_extraction_modes: Suggested modes for this document type
        - extraction_confidence: Quality score of extraction
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Read file content
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty file provided")
        
        # Parse recommended modes
        modes_list = [mode.strip() for mode in recommended_modes.split(",") if mode.strip()]
        if not modes_list:
            modes_list = ["paragraph", "sentence"]
        
        # Process document
        result = await reference_manager.upload_reference_document(
            file_content=file_content,
            name=name,
            document_type=document_type,
            category=category,
            extraction_model=extraction_model,
            recommended_modes=modes_list,
            created_by=None  # TODO: Add user authentication
        )
        
        logger.info(f"Reference document uploaded successfully: {result.document_id}")
        
        return {
            "status": "success",
            "document_id": result.document_id,
            "name": name,
            "document_type": document_type,
            "category": category,
            "original_filename": file.filename,
            "file_size_bytes": len(file_content),
            "extraction_model": extraction_model,
            "extraction_preview": result.extraction_preview,
            "recommended_extraction_modes": result.recommended_extraction_modes,
            "extraction_confidence": result.extraction_confidence,
            "processing_time_ms": result.processing_time_ms,
            "model_used": result.model_used,
            "is_active": True,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Reference document upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/references")
async def get_reference_documents(
    document_type: Optional[str] = None,
    category: Optional[str] = None,
    active_only: bool = True,
    db: Session = Depends(get_db)
):
    """
    Get all reference documents with optional filtering
    
    Returns list of reference documents with metadata
    """
    try:
        documents = await reference_manager.get_all_reference_documents(
            document_type=document_type,
            category=category,
            active_only=active_only
        )
        
        return {
            "status": "success",
            "documents": [
                {
                    "document_id": doc.document_id,
                    "name": doc.name,
                    "document_type": doc.document_type,
                    "category": doc.category,
                    "original_filename": doc.original_filename if hasattr(doc, 'original_filename') else None,
                    "file_size_bytes": doc.file_size_bytes,
                    "extraction_model": doc.extraction_model if hasattr(doc, 'extraction_model') else None,
                    "extraction_confidence": doc.extraction_confidence if hasattr(doc, 'extraction_confidence') else None,
                    "recommended_extraction_modes": doc.recommended_extraction_modes,
                    "is_active": doc.is_active,
                    "created_at": doc.created_at.isoformat(),
                    "updated_at": doc.updated_at.isoformat()
                }
                for doc in documents
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get reference documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get references: {str(e)}")

@router.get("/references/{reference_id}/content")
async def get_reference_document_content(
    reference_id: str,
    db: Session = Depends(get_db)
):
    """
    Get extracted markdown content of a reference document for preview
    
    Returns:
        - document_id: The reference document ID
        - name: Document name
        - document_type: Type of document
        - content: Extracted markdown content
        - content_preview: First 500 characters for quick preview
        - content_stats: Statistics about the content
    """
    try:
        # Get reference document info
        reference_doc = await reference_manager.get_reference_document(reference_id)
        if not reference_doc:
            raise HTTPException(status_code=404, detail="Reference document not found")
        
        # Get the extracted markdown content
        content = await reference_manager.get_reference_content(reference_id)
        if not content:
            raise HTTPException(status_code=404, detail="Reference document content not found")
        
        # Generate content statistics
        lines = content.split('\n')
        words = len(content.split())
        characters = len(content)
        
        # Count markdown elements
        heading_count = len([line for line in lines if line.strip().startswith('#')])
        code_block_count = content.count('```')
        list_item_count = len([line for line in lines if line.strip().startswith(('-', '*', '+'))])
        
        return {
            "status": "success",
            "document_id": reference_doc.document_id,
            "name": reference_doc.name,
            "document_type": reference_doc.document_type,
            "category": reference_doc.category,
            "content": content,
            "content_preview": content[:500] + "..." if len(content) > 500 else content,
            "content_stats": {
                "total_characters": characters,
                "total_words": words,
                "total_lines": len(lines),
                "heading_count": heading_count,
                "code_block_count": code_block_count // 2,  # Divide by 2 since each block has opening and closing
                "list_item_count": list_item_count,
                "estimated_reading_time_minutes": max(1, words // 200)  # Average reading speed
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get reference document content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get content: {str(e)}")

@router.get("/validate/sessions")
async def get_validation_sessions(
    reference_document_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    Get validation sessions with optional filtering
    
    Returns list of validation sessions with metadata
    """
    try:
        query = db.query(IDCValidationSession)
        
        # Filter by reference document if provided
        if reference_document_id:
            # Find the reference document database ID
            ref_doc = db.query(IDCReferenceDocument).filter(
                IDCReferenceDocument.document_id == reference_document_id
            ).first()
            if ref_doc:
                query = query.filter(IDCValidationSession.reference_document_id == ref_doc.id)
        
        # Filter by status if provided
        if status:
            query = query.filter(IDCValidationSession.status == status)
        
        # Order by creation time (newest first) and apply pagination
        sessions = query.order_by(IDCValidationSession.created_at.desc()).offset(offset).limit(limit).all()
        
        sessions_data = []
        for session in sessions:
            # Get reference document info
            ref_doc = db.query(IDCReferenceDocument).filter(
                IDCReferenceDocument.id == session.reference_document_id
            ).first()
            
            session_data = {
                "id": session.id,
                "session_id": session.session_id,
                "reference_document_id": ref_doc.document_id if ref_doc else None,
                "input_filename": session.input_filename,
                "status": session.status,
                "overall_score": session.overall_score,
                "confidence_score": session.confidence_score,
                "created_at": session.created_at.isoformat(),
                "completed_at": session.processing_end_time.isoformat() if session.processing_end_time else None,
                "extraction_model": session.validation_model,  # Using validation_model as extraction model for display
                "validation_model": session.validation_model,
                "extraction_mode": session.extraction_mode,
                "validation_method": "granular_validation"  # Default method name
            }
            sessions_data.append(session_data)
        
        return {
            "status": "success",
            "sessions": sessions_data
        }
        
    except Exception as e:
        logger.error(f"Failed to get validation sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get sessions: {str(e)}")

@router.post("/validate/granular")
async def validate_with_granular_extraction(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Input document to validate"),
    reference_id: str = Form(..., description="Reference document ID"),
    extraction_mode: str = Form(..., description="Extraction mode"),
    validation_model: Optional[str] = Form(None, description="Validation model"),
    max_context_usage: float = Form(0.35, description="Max context usage"),
    preserve_context: bool = Form(True, description="Preserve context"),
    quality_threshold: float = Form(0.8, description="Quality threshold"),
    db: Session = Depends(get_db)
):
    """
    Validate document using user-selected granular extraction
    
    Process:
    1. Extract content based on user-selected mode
    2. Systematically validate each unit against reference
    3. Track progress in real-time
    4. Aggregate results with detailed breakdown
    
    Returns:
        - session_id: Validation session identifier
        - extraction_summary: How many units extracted
        - estimated_processing_time: Based on unit count
        - progress_endpoint: WebSocket URL for real-time updates
    """
    try:
        # Validate inputs
        if extraction_mode not in ["sentence", "paragraph", "qa_pairs", "section"]:
            raise HTTPException(status_code=400, detail="Invalid extraction mode")
        
        if not 0.1 <= max_context_usage <= 0.8:
            raise HTTPException(status_code=400, detail="Context usage must be between 0.1 and 0.8")
        
        # Get reference document
        reference_doc = await reference_manager.get_reference_document(reference_id)
        if not reference_doc:
            raise HTTPException(status_code=404, detail="Reference document not found")
        
        # Read input file
        input_content = await file.read()
        input_text = input_content.decode('utf-8', errors='ignore')
        
        if not input_text.strip():
            raise HTTPException(status_code=400, detail="Empty input document")
        
        # Generate session ID
        session_id = str(uuid4())
        
        # Create extraction configuration
        extraction_config = GranularExtractionConfig(
            mode=ExtractionMode(extraction_mode),
            preserve_context=preserve_context,
            extraction_model=validation_model
        )
        
        # Create validation configuration
        validation_config = ValidationConfig(
            validation_model=validation_model,
            max_context_usage=max_context_usage,
            quality_threshold=quality_threshold
        )
        
        # Extract units from input document
        extraction_result = await extraction_service.extract_by_mode(
            document_content=input_text,
            extraction_config=extraction_config
        )
        
        logger.info(f"Extracted {len(extraction_result.extracted_units)} units for validation session {session_id}")
        
        # Prepare serializable extraction config
        extraction_config_dict = {
            "mode": extraction_config.mode.value,  # Convert enum to string
            "max_unit_size": extraction_config.max_unit_size,
            "overlap_size": extraction_config.overlap_size,
            "preserve_context": extraction_config.preserve_context,
            "quality_threshold": extraction_config.quality_threshold,
            "extraction_model": extraction_config.extraction_model
        }
        
        # Prepare serializable extracted units
        extracted_units_dict = []
        for unit in extraction_result.extracted_units:
            unit_dict = {
                "index": unit.index,
                "type": unit.type,
                "content": unit.content,
                "context_before": unit.context_before,
                "context_after": unit.context_after,
                "metadata": unit.metadata
            }
            extracted_units_dict.append(unit_dict)
        
        # Get reference document database ID
        with get_db_session() as db_session:
            ref_doc_db = db_session.query(IDCReferenceDocument).filter(
                IDCReferenceDocument.document_id == reference_doc.document_id
            ).first()
            
            if not ref_doc_db:
                raise HTTPException(status_code=404, detail="Reference document not found in database")
        
            validation_session = IDCValidationSession(
                session_id=session_id,
                reference_document_id=ref_doc_db.id,  # Use database ID, not document_id
                input_filename=file.filename,
                input_file_hash=reference_manager._calculate_file_hash(input_content),
                input_file_size_bytes=len(input_content),
                extraction_mode=extraction_mode,
                extraction_config=extraction_config_dict,
                validation_model=validation_model or validation_service.default_model,
                max_context_usage=max_context_usage,
                total_units_extracted=len(extraction_result.extracted_units),
                extracted_units=extracted_units_dict,
                status="initialized",
                processing_start_time=datetime.utcnow()
            )
            
            db_session.add(validation_session)
            db_session.commit()
        
        # Start background validation
        background_tasks.add_task(
            _run_systematic_validation,
            extraction_result.extracted_units,
            reference_doc.extracted_markdown,
            session_id,
            validation_config
        )
        
        # Estimate processing time (rough estimate: 3-5 seconds per unit)
        estimated_time_seconds = len(extraction_result.extracted_units) * 4
        
        return {
            "status": "success",
            "session_id": session_id,
            "extraction_summary": {
                "total_units": len(extraction_result.extracted_units),
                "extraction_mode": extraction_mode,
                "average_unit_size": extraction_result.total_tokens_used / len(extraction_result.extracted_units) if extraction_result.extracted_units else 0
            },
            "estimated_processing_time_seconds": estimated_time_seconds,
            "progress_endpoint": f"/api/v1/idc/validate/{session_id}/progress",
            "websocket_endpoint": f"ws://localhost:8000/api/v1/idc/validate/{session_id}/stream"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Granular validation initiation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

async def _run_systematic_validation(
    extracted_units: List,
    reference_document: str,
    session_id: str,
    validation_config: ValidationConfig
):
    """Background task to run systematic validation"""
    try:
        logger.info(f"Starting systematic validation for session {session_id}")
        
        result = await validation_service.validate_all_units(
            extracted_units=extracted_units,
            reference_document=reference_document,
            session_id=session_id,
            validation_config=validation_config
        )
        
        logger.info(f"Systematic validation completed for session {session_id}")
        
    except Exception as e:
        logger.error(f"Systematic validation failed for session {session_id}: {str(e)}")
        
        # Update session with error
        try:
            with get_db_session() as db:
                session = db.query(IDCValidationSession).filter(
                    IDCValidationSession.session_id == session_id
                ).first()
                if session:
                    session.status = "failed"
                    session.error_message = str(e)
                    session.processing_end_time = datetime.utcnow()
                    db.commit()
        except Exception as db_error:
            logger.error(f"Failed to update session error: {db_error}")

@router.get("/validate/{session_id}/progress")
async def get_granular_validation_progress(
    session_id: str,
    db: Session = Depends(get_db)
):
    """
    Get real-time progress of granular validation
    
    Returns:
        - units_completed: Number of units processed
        - total_units: Total units to process
        - current_unit_index: Currently processing unit
        - percentage_complete: Overall progress
        - average_time_per_unit: Processing speed
        - estimated_completion: When validation will finish
        - context_usage_stats: Average and max context usage
    """
    try:
        # Get progress from Redis cache
        progress = await validation_service.get_validation_progress(session_id)
        
        if progress:
            return {
                "status": "success",
                "progress": progress
            }
        
        # Fallback: get from database
        session = db.query(IDCValidationSession).filter(
            IDCValidationSession.session_id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Validation session not found")
        
        return {
            "status": "success",
            "progress": {
                "session_id": session_id,
                "total_units": session.total_units_extracted,
                "completed_units": session.units_processed,
                "progress_percentage": (session.units_processed / session.total_units_extracted * 100) if session.total_units_extracted > 0 else 0,
                "status": session.status,
                "processing_start_time": session.processing_start_time.isoformat() if session.processing_start_time else None,
                "processing_end_time": session.processing_end_time.isoformat() if session.processing_end_time else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get validation progress: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get progress: {str(e)}")

@router.get("/validate/{session_id}/results/detailed")
async def get_detailed_validation_results(
    session_id: str,
    include_unit_details: bool = True,
    include_failed_units: bool = True,
    db: Session = Depends(get_db)
):
    """
    Get comprehensive validation results with unit-by-unit breakdown
    
    Returns:
        - overall_results: Aggregated scores and summary
        - unit_results: Individual validation results for each unit
        - processing_stats: Context usage, timing, performance metrics
        - quality_indicators: Confidence scores, coverage analysis
        - failed_units: Units that couldn't be processed (if any)
    """
    try:
        # Get session from database
        session = db.query(IDCValidationSession).filter(
            IDCValidationSession.session_id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Validation session not found")
        
        results = {
            "status": "success",
            "session_id": session_id,
            "overall_results": {
                "overall_score": session.overall_score,
                "confidence_score": session.confidence_score,
                "completeness_score": session.completeness_score,
                "total_units": session.total_units_extracted,
                "units_processed": session.units_processed,
                "units_failed": session.units_failed,
                "processing_status": session.status
            },
            "processing_stats": {
                "total_processing_time_ms": session.total_processing_time_ms,
                "average_context_usage": session.average_context_usage,
                "max_context_usage": session.max_context_usage_recorded,
                "extraction_mode": session.extraction_mode,
                "validation_model": session.validation_model
            }
        }
        
        if include_unit_details:
            # Get individual unit results
            unit_results = db.query(IDCUnitValidationResult).filter(
                IDCUnitValidationResult.session_id == session_id
            ).order_by(IDCUnitValidationResult.unit_index).all()
            
            results["unit_results"] = [
                {
                    "unit_index": unit.unit_index,
                    "unit_type": unit.unit_type,
                    "unit_content": unit.unit_content[:200] + "..." if len(unit.unit_content) > 200 else unit.unit_content,
                    "validation_score": unit.validation_score,
                    "confidence_score": unit.confidence_score,
                    "validation_feedback": unit.validation_feedback,
                    "matched_reference_sections": unit.matched_reference_sections,
                    "context_usage_percentage": unit.context_usage_percentage,
                    "processing_time_ms": unit.processing_time_ms,
                    "requires_human_review": unit.requires_human_review,
                    "quality_flags": unit.quality_flags
                }
                for unit in unit_results
            ]
        
        if include_failed_units and session.failed_units:
            results["failed_units"] = session.failed_units
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get detailed results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get results: {str(e)}")


@router.get("/templates")
async def get_idc_templates(
    template_type: Optional[str] = None,
    reference_document_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get available validation templates
    """
    try:
        templates = await reference_manager.get_templates(
            template_type=template_type,
            reference_document_id=reference_document_id
        )
        
        return {
            "status": "success",
            "templates": templates
        }
        
    except Exception as e:
        logger.error(f"Failed to get templates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get templates: {str(e)}")

@router.post("/templates")
async def create_idc_template(
    template: TemplateCreate,
    db: Session = Depends(get_db)
):
    """
    Create new validation template
    """
    try:
        template_id = await reference_manager.create_template(
            name=template.name,
            description=template.description,
            template_type=template.template_type,
            reference_document_id=template.reference_document_id,
            default_extraction_mode=template.default_extraction_mode,
            validation_config=template.validation_config,
            created_by=None  # TODO: Add user authentication
        )
        
        return {
            "status": "success",
            "template_id": template_id,
            "message": "Template created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create template: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create template: {str(e)}")

@router.websocket("/validate/{session_id}/stream")
async def stream_granular_validation_progress(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time granular validation progress
    
    Streams:
        - Unit completion notifications
        - Processing speed updates
        - Context usage monitoring
        - Error notifications
        - Quality indicators
    """
    await websocket.accept()
    
    try:
        while True:
            # Get current progress
            progress = await validation_service.get_validation_progress(session_id)
            
            if progress:
                # Send progress update
                await websocket.send_json({
                    "type": "progress",
                    "data": progress
                })
                
                # Check if validation is complete
                if progress.get("status") in ["completed", "failed"]:
                    await websocket.send_json({
                        "type": "completion",
                        "data": {
                            "status": progress.get("status"),
                            "session_id": session_id
                        }
                    })
                    break
            
            # Wait before next update
            await asyncio.sleep(2)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {str(e)}")
        await websocket.send_json({
            "type": "error",
            "data": {"message": str(e)}
        })
    finally:
        await websocket.close()

@router.get("/configuration")
async def get_idc_configuration(db: Session = Depends(get_db)):
    """
    Get IDC configuration settings from database/cache
    Following Jarvis patterns - uses settings cache with fallback
    """
    try:
        # Get settings from cache/database
        idc_settings = get_idc_settings()
        
        # Get available Ollama models for the configuration panel
        # Use the same graceful handling pattern as the Ollama models endpoint
        available_models = []
        try:
            import httpx
            
            # Get Ollama URL using the same comprehensive pattern as LLM router
            def get_ollama_url():
                """Get Ollama URL from settings cache, environment, or defaults"""
                try:
                    # Try to get from IDC extraction settings first
                    extraction_config = get_extraction_config(idc_settings)
                    if extraction_config.get('model_server'):
                        return extraction_config['model_server']
                    
                    # Try to get from LLM settings as fallback
                    from app.core.llm_settings_cache import get_llm_settings
                    settings = get_llm_settings()
                    model_config = settings.get('model_config', {})
                    if model_config.get('model_server'):
                        return model_config['model_server']
                    
                    # Check main_llm config as fallback
                    main_llm = settings.get('main_llm', {})
                    if main_llm.get('model_server'):
                        return main_llm['model_server']
                except Exception:
                    pass  # Fall back to environment/defaults if settings unavailable
                
                # Use environment variable if set
                import os
                if os.environ.get("OLLAMA_BASE_URL"):
                    return os.environ.get("OLLAMA_BASE_URL")
                
                # Use appropriate default based on environment
                # Check if we're running inside Docker
                in_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER')
                return "http://host.docker.internal:11434" if in_docker else "http://host.docker.internal:11434"
            
            ollama_base_url = get_ollama_url()
            
            if ollama_base_url:
                async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                    resp = await client.get(f"{ollama_base_url}/api/tags")
                    resp.raise_for_status()
                    data = resp.json()
                    
                    # Extract detailed model information (same logic as Settings page)
                    for model in data.get('models', []):
                        model_name = model.get('name', '')
                        
                        # Convert size from bytes to human-readable format
                        size_bytes = model.get('size', 0)
                        if size_bytes > 1024**3:
                            size = f"{size_bytes / (1024**3):.1f} GB"
                        elif size_bytes > 1024**2:
                            size = f"{size_bytes / (1024**2):.0f} MB"
                        else:
                            size = f"{size_bytes / 1024:.0f} KB"
                        
                        # Parse modified time
                        modified_at = model.get('modified_at', '')
                        if modified_at:
                            try:
                                from datetime import datetime
                                import pytz
                                dt = datetime.fromisoformat(modified_at.replace('Z', '+00:00'))
                                now = datetime.now(pytz.UTC)
                                diff = now - dt
                                
                                if diff.days == 0:
                                    if diff.seconds < 3600:
                                        modified = f"{diff.seconds // 60} minutes ago"
                                    else:
                                        modified = f"{diff.seconds // 3600} hours ago"
                                elif diff.days == 1:
                                    modified = "Yesterday"
                                elif diff.days < 7:
                                    modified = f"{diff.days} days ago"
                                elif diff.days < 30:
                                    modified = f"{diff.days // 7} weeks ago"
                                else:
                                    modified = f"{diff.days // 30} months ago"
                            except:
                                modified = modified_at
                        else:
                            modified = "Unknown"
                        
                        # Fetch context length from model details
                        context_length = "Unknown"
                        try:
                            show_resp = await client.post(
                                f"{ollama_base_url}/api/show",
                                json={"name": model_name}
                            )
                            if show_resp.status_code == 200:
                                show_data = show_resp.json()
                                model_info = show_data.get('model_info', {})
                                
                                # Look for context length in various possible fields
                                for key, value in model_info.items():
                                    if 'context_length' in key.lower():
                                        context_length = f"{value:,}"
                                        break
                                
                                # If not found in model_info, check details
                                if context_length == "Unknown":
                                    details = show_data.get('details', {})
                                    if 'context_length' in details:
                                        context_length = f"{details['context_length']:,}"
                        except Exception as e:
                            logger.debug(f"Failed to fetch context length for {model_name}: {e}")
                        
                        available_models.append({
                            "name": model_name,
                            "id": model.get('digest', '')[:12],  # First 12 chars of digest as ID
                            "size": size,
                            "modified": modified,
                            "context_length": context_length
                        })
        except httpx.ConnectError:
            # Ollama is not reachable - provide fallback models without logging warnings
            logger.debug(f"Ollama service not available at {ollama_base_url}")
            available_models = [
                {"name": "qwen3:30b-a3b-q4_K_M", "id": "fallback-01", "size": "18 GB", "modified": "N/A", "context_length": "32,768"},
                {"name": "llama3.1:8b", "id": "fallback-02", "size": "4.7 GB", "modified": "N/A", "context_length": "128,000"},
                {"name": "llama3.1:70b", "id": "fallback-03", "size": "40 GB", "modified": "N/A", "context_length": "128,000"},
                {"name": "qwen2.5:32b", "id": "fallback-04", "size": "19 GB", "modified": "N/A", "context_length": "32,768"},
                {"name": "deepseek-r1:8b", "id": "fallback-05", "size": "4.9 GB", "modified": "N/A", "context_length": "65,536"}
            ]
        except httpx.TimeoutException:
            # Ollama is slow to respond - provide fallback models
            logger.debug(f"Timeout connecting to Ollama at {ollama_base_url}")
            available_models = [
                {"name": "qwen3:30b-a3b-q4_K_M", "id": "fallback-01", "size": "18 GB", "modified": "N/A", "context_length": "32,768"},
                {"name": "llama3.1:8b", "id": "fallback-02", "size": "4.7 GB", "modified": "N/A", "context_length": "128,000"},
                {"name": "llama3.1:70b", "id": "fallback-03", "size": "40 GB", "modified": "N/A", "context_length": "128,000"},
                {"name": "qwen2.5:32b", "id": "fallback-04", "size": "19 GB", "modified": "N/A", "context_length": "32,768"},
                {"name": "deepseek-r1:8b", "id": "fallback-05", "size": "4.9 GB", "modified": "N/A", "context_length": "65,536"}
            ]
        except Exception as e:
            # Other connection or parsing errors - provide fallback models
            logger.debug(f"Failed to get Ollama models: {str(e)}")
            available_models = [
                {"name": "qwen3:30b-a3b-q4_K_M", "id": "fallback-01", "size": "18 GB", "modified": "N/A", "context_length": "32,768"},
                {"name": "llama3.1:8b", "id": "fallback-02", "size": "4.7 GB", "modified": "N/A", "context_length": "128,000"},
                {"name": "llama3.1:70b", "id": "fallback-03", "size": "40 GB", "modified": "N/A", "context_length": "128,000"},
                {"name": "qwen2.5:32b", "id": "fallback-04", "size": "19 GB", "modified": "N/A", "context_length": "32,768"},
                {"name": "deepseek-r1:8b", "id": "fallback-05", "size": "4.9 GB", "modified": "N/A", "context_length": "65,536"}
            ]
        
        # Provide defaults for missing settings
        extraction_settings = idc_settings.get("extraction", {})
        validation_settings = idc_settings.get("validation", {})
        comparison_settings = idc_settings.get("comparison", {})
        
        # Ensure all required fields are present with defaults
        if not extraction_settings:
            extraction_settings = {
                "model": "qwen3:30b-a3b-q4_K_M",
                "system_prompt": "Extract and structure the content from this document in clear, well-organized markdown format.",
                "max_tokens": 4000,
                "temperature": 0.1,
                "top_p": 0.9,
                "top_k": 40,
                "enable_chunking": True,
                "chunk_size": 4000,
                "chunk_overlap": 200
            }
        
        if not validation_settings:
            validation_settings = {
                "model": "qwen3:30b-a3b-q4_K_M",
                "system_prompt": "Compare the extracted content with the reference document and provide detailed validation feedback.",
                "max_tokens": 4000,
                "temperature": 0.1,
                "top_p": 0.9,
                "max_context_usage": 0.35,
                "confidence_threshold": 0.9,
                "enable_structured_output": True
            }
        
        if not comparison_settings:
            comparison_settings = {
                "algorithm": "semantic",
                "similarity_threshold": 0.85,
                "ignore_whitespace": True,
                "ignore_case": False,
                "enable_fuzzy_matching": True,
                "fuzzy_threshold": 0.8
            }
        
        return {
            "status": "success",
            "configuration": {
                "extraction": extraction_settings,
                "validation": validation_settings,
                "comparison": comparison_settings
            },
            "available_models": available_models
        }
        
    except Exception as e:
        logger.error(f"Failed to get IDC configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get configuration: {str(e)}")

@router.post("/configuration")
async def update_idc_configuration(
    config: ConfigurationUpdate,
    db: Session = Depends(get_db)
):
    """
    Update IDC configuration settings
    Following Jarvis patterns - saves to database and updates cache
    """
    try:
        # Get current settings
        current_settings = get_idc_settings()
        
        # Update extraction settings
        if config.extraction_model is not None:
            # Handle case where frontend might send model ID instead of name
            # If it looks like an ID (12 char hex), try to find the actual model name
            if len(config.extraction_model) == 12 and all(c in '0123456789abcdef' for c in config.extraction_model.lower()):
                logger.warning(f"Received model ID instead of name: {config.extraction_model}. This should be fixed in frontend.")
                # Keep the ID for now but log the issue
            current_settings["extraction"]["model"] = config.extraction_model
        if config.extraction_system_prompt is not None:
            current_settings["extraction"]["system_prompt"] = config.extraction_system_prompt
        if config.extraction_temperature is not None:
            current_settings["extraction"]["temperature"] = config.extraction_temperature
        if config.extraction_max_tokens is not None:
            current_settings["extraction"]["max_tokens"] = config.extraction_max_tokens
        if config.extraction_max_context_usage is not None:
            current_settings["extraction"]["max_context_usage"] = config.extraction_max_context_usage
        if config.extraction_confidence_threshold is not None:
            current_settings["extraction"]["confidence_threshold"] = config.extraction_confidence_threshold
        if config.enable_chunking is not None:
            current_settings["extraction"]["enable_chunking"] = config.enable_chunking
        if config.chunk_size is not None:
            current_settings["extraction"]["chunk_size"] = config.chunk_size
        if config.chunk_overlap is not None:
            current_settings["extraction"]["chunk_overlap"] = config.chunk_overlap
        
        # Update validation settings
        if config.validation_model is not None:
            # Handle case where frontend might send model ID instead of name
            # If it looks like an ID (12 char hex), try to find the actual model name
            if len(config.validation_model) == 12 and all(c in '0123456789abcdef' for c in config.validation_model.lower()):
                logger.warning(f"Received model ID instead of name: {config.validation_model}. This should be fixed in frontend.")
                # Keep the ID for now but log the issue
            current_settings["validation"]["model"] = config.validation_model
        if config.validation_system_prompt is not None:
            current_settings["validation"]["system_prompt"] = config.validation_system_prompt
        if config.validation_temperature is not None:
            current_settings["validation"]["temperature"] = config.validation_temperature
        if config.validation_max_tokens is not None:
            current_settings["validation"]["max_tokens"] = config.validation_max_tokens
        if config.max_context_usage is not None:
            current_settings["validation"]["max_context_usage"] = config.max_context_usage
        if config.quality_threshold is not None:
            current_settings["validation"]["confidence_threshold"] = config.quality_threshold
        if config.enable_structured_output is not None:
            current_settings["validation"]["enable_structured_output"] = config.enable_structured_output
        
        # Update comparison settings
        if config.comparison_algorithm is not None:
            current_settings["comparison"]["algorithm"] = config.comparison_algorithm
        if config.similarity_threshold is not None:
            current_settings["comparison"]["similarity_threshold"] = config.similarity_threshold
        if config.fuzzy_threshold is not None:
            current_settings["comparison"]["fuzzy_threshold"] = config.fuzzy_threshold
        if config.ignore_case is not None:
            current_settings["comparison"]["ignore_case"] = config.ignore_case
        if config.ignore_whitespace is not None:
            current_settings["comparison"]["ignore_whitespace"] = config.ignore_whitespace
        if config.enable_fuzzy_matching is not None:
            current_settings["comparison"]["enable_fuzzy_matching"] = config.enable_fuzzy_matching
        
        # Save to database
        settings_row = db.query(Settings).filter(Settings.category == 'idc').first()
        if settings_row:
            settings_row.settings = current_settings
        else:
            settings_row = Settings(category='idc', settings=current_settings)
            db.add(settings_row)
        
        db.commit()
        
        # Update cache
        set_idc_settings(current_settings)
        
        # Reload settings to ensure consistency
        reload_idc_settings()
        
        return {
            "status": "success",
            "message": "Configuration updated successfully",
            "configuration": current_settings
        }
        
    except Exception as e:
        logger.error(f"Failed to update IDC configuration: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")

@router.post("/configuration/reload")
async def reload_idc_configuration():
    """
    Reload IDC configuration from database, clearing cache
    """
    try:
        # Reload from database
        settings = reload_idc_settings()
        
        return {
            "status": "success",
            "message": "Configuration reloaded from database",
            "configuration": settings
        }
        
    except Exception as e:
        logger.error(f"Failed to reload IDC configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reload configuration: {str(e)}")

@router.delete("/reference/{reference_id}")
async def delete_reference_document(
    reference_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a reference document (soft delete - marks as inactive)
    """
    try:
        success = await reference_manager.delete_reference_document(reference_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Reference document not found")
        
        return {
            "status": "success",
            "message": "Reference document deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete reference document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete reference: {str(e)}")

@router.put("/reference/{reference_id}")
async def update_reference_document(
    reference_id: str,
    update_data: ReferenceDocumentUpdate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Update a reference document
    
    Supports:
    - Updating metadata (name, document type, category)
    - Changing extraction model
    - Re-extracting content with new model
    """
    try:
        # Check if document exists
        reference_doc = await reference_manager.get_reference_document(reference_id)
        if not reference_doc:
            raise HTTPException(status_code=404, detail="Reference document not found")
        
        # Prepare updates dict
        updates = {}
        if update_data.name is not None:
            updates['name'] = update_data.name
        if update_data.document_type is not None:
            updates['document_type'] = update_data.document_type
        if update_data.category is not None:
            updates['category'] = update_data.category
        if update_data.extraction_model is not None:
            updates['extraction_model'] = update_data.extraction_model
        if update_data.content is not None:
            updates['extracted_markdown'] = update_data.content
        
        # If re-extraction is requested, handle it
        if update_data.re_extract and update_data.extraction_model:
            # Re-extract content with new model (ignores manual content edits)
            success = await reference_manager.update_reference_document_with_reextraction(
                document_id=reference_id,
                updates=updates,
                new_extraction_model=update_data.extraction_model
            )
        else:
            # Simple metadata and/or content update
            success = await reference_manager.update_reference_document(
                document_id=reference_id,
                updates=updates
            )
        
        if not success:
            raise HTTPException(status_code=404, detail="Reference document not found or update failed")
        
        # Get updated document info
        updated_doc = await reference_manager.get_reference_document(reference_id)
        
        return {
            "status": "success",
            "message": "Reference document updated successfully",
            "document": {
                "document_id": updated_doc.document_id,
                "name": updated_doc.name,
                "document_type": updated_doc.document_type,
                "category": updated_doc.category,
                "extraction_model": getattr(updated_doc, 'extraction_model', None),
                "updated_at": updated_doc.updated_at.isoformat()
            } if updated_doc else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update reference document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update reference: {str(e)}")