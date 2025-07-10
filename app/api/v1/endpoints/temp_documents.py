"""
API endpoints for temporary document operations and in-memory RAG management.
Provides RESTful interface for chat interfaces to manage temporary documents.
"""

import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

from app.services.temporary_document_service import get_temporary_document_service
from app.core.in_memory_rag_settings import get_in_memory_rag_settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic models for request/response validation
class DocumentUploadRequest(BaseModel):
    conversation_id: str = Field(..., description="Conversation ID for document association")
    ttl_hours: int = Field(default=2, ge=1, le=24, description="Time-to-live in hours")
    auto_include: bool = Field(default=True, description="Auto-include in chat context")
    enable_in_memory_rag: bool = Field(default=True, description="Enable in-memory RAG processing")

class DocumentUploadResponse(BaseModel):
    success: bool
    temp_doc_id: Optional[str] = None
    filename: str
    message: str
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class DocumentQueryRequest(BaseModel):
    conversation_id: str = Field(..., description="Conversation ID")
    query: str = Field(..., min_length=1, description="Query string")
    use_in_memory_rag: bool = Field(default=True, description="Use in-memory RAG first")
    fallback_to_temp_docs: bool = Field(default=True, description="Fallback to temp docs if needed")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")

class DocumentQueryResponse(BaseModel):
    query: str
    sources: List[Dict[str, Any]]
    total_chunks: int
    processing_time_ms: float
    source_type: Optional[str] = None
    conversation_id: str
    error: Optional[str] = None

class DocumentPreferencesRequest(BaseModel):
    is_included: Optional[bool] = None
    enable_in_memory_rag: Optional[bool] = None
    ttl_hours: Optional[int] = Field(None, ge=1, le=24)

class DocumentListResponse(BaseModel):
    conversation_id: str
    documents: List[Dict[str, Any]]
    total_count: int
    active_count: int
    in_memory_rag_enabled: bool

class ServiceStatsResponse(BaseModel):
    conversation_id: str
    temp_documents: Dict[str, Any]
    in_memory_rag: Dict[str, Any]
    config: Dict[str, Any]
    timestamp: str

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_temp_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    conversation_id: str = Form(...),
    ttl_hours: int = Form(default=2),
    auto_include: bool = Form(default=True),
    enable_in_memory_rag: bool = Form(default=True)
):
    """
    Upload a temporary document for conversation-scoped processing.
    
    - **file**: Document file (PDF, DOCX, XLSX, PPTX, TXT)
    - **conversation_id**: Conversation ID for document association
    - **ttl_hours**: Time-to-live in hours (1-24)
    - **auto_include**: Auto-include in chat context
    - **enable_in_memory_rag**: Enable in-memory RAG processing
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check file extension
        allowed_extensions = {'.pdf', '.docx', '.xlsx', '.pptx', '.txt'}
        file_ext = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
        if f'.{file_ext}' not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Read file content
        file_content = await file.read()
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Check file size (50MB limit)
        max_size_mb = 50
        file_size_mb = len(file_content) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large: {file_size_mb:.1f}MB (max: {max_size_mb}MB)"
            )
        
        # Get service and process document
        service = get_temporary_document_service()
        result = await service.upload_and_process_document(
            file_content=file_content,
            filename=file.filename,
            conversation_id=conversation_id,
            ttl_hours=ttl_hours,
            auto_include=auto_include,
            enable_in_memory_rag=enable_in_memory_rag
        )
        
        if result['success']:
            return DocumentUploadResponse(
                success=True,
                temp_doc_id=result['temp_doc_id'],
                filename=file.filename,
                message=f"Successfully processed document '{file.filename}'",
                metadata=result.get('metadata', {})
            )
        else:
            return DocumentUploadResponse(
                success=False,
                temp_doc_id=result.get('temp_doc_id'),
                filename=file.filename,
                message="Document processing failed",
                error=result.get('error', 'Unknown error')
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/query", response_model=DocumentQueryResponse)
async def query_temp_documents(request: DocumentQueryRequest):
    """
    Query temporary documents using in-memory RAG with fallback options.
    
    - **conversation_id**: Conversation ID
    - **query**: Query string
    - **use_in_memory_rag**: Use in-memory RAG first
    - **fallback_to_temp_docs**: Fallback to temp docs if needed
    - **top_k**: Number of results to return
    """
    try:
        service = get_temporary_document_service()
        result = await service.query_documents(
            conversation_id=request.conversation_id,
            query=request.query,
            use_in_memory_rag=request.use_in_memory_rag,
            fallback_to_temp_docs=request.fallback_to_temp_docs,
            top_k=request.top_k
        )
        
        return DocumentQueryResponse(**result)
    
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@router.get("/list/{conversation_id}", response_model=DocumentListResponse)
async def list_temp_documents(conversation_id: str, include_stats: bool = True):
    """
    Get all temporary documents for a conversation.
    
    - **conversation_id**: Conversation ID
    - **include_stats**: Include in-memory RAG statistics
    """
    try:
        service = get_temporary_document_service()
        documents = await service.get_conversation_documents(
            conversation_id, 
            include_in_memory_stats=include_stats
        )
        
        active_count = len([doc for doc in documents if doc.get('is_included', False)])
        
        # Check if any document has in-memory RAG enabled
        in_memory_rag_enabled = any(
            doc.get('metadata', {}).get('in_memory_rag_enabled', False) 
            for doc in documents
        )
        
        return DocumentListResponse(
            conversation_id=conversation_id,
            documents=documents,
            total_count=len(documents),
            active_count=active_count,
            in_memory_rag_enabled=in_memory_rag_enabled
        )
    
    except Exception as e:
        logger.error(f"List documents failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@router.put("/preferences/{temp_doc_id}")
async def update_document_preferences(
    temp_doc_id: str,
    request: DocumentPreferencesRequest,
    conversation_id: Optional[str] = None
):
    """
    Update preferences for a temporary document.
    
    - **temp_doc_id**: Temporary document ID
    - **request**: Preferences to update
    - **conversation_id**: Required if changing in-memory RAG settings
    """
    try:
        # Convert request to dict, excluding None values
        preferences = {
            k: v for k, v in request.dict().items() 
            if v is not None
        }
        
        if not preferences:
            raise HTTPException(status_code=400, detail="No preferences provided")
        
        # Check if conversation_id is required
        if 'enable_in_memory_rag' in preferences and not conversation_id:
            raise HTTPException(
                status_code=400, 
                detail="conversation_id required when changing in-memory RAG settings"
            )
        
        service = get_temporary_document_service()
        success = await service.update_document_preferences(
            temp_doc_id=temp_doc_id,
            preferences=preferences,
            conversation_id=conversation_id
        )
        
        if success:
            return {"success": True, "message": "Preferences updated successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update preferences failed: {e}")
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")

@router.delete("/delete/{temp_doc_id}")
async def delete_temp_document(
    temp_doc_id: str,
    conversation_id: Optional[str] = None
):
    """
    Delete a temporary document.
    
    - **temp_doc_id**: Temporary document ID
    - **conversation_id**: Optional conversation ID for in-memory RAG cleanup
    """
    try:
        service = get_temporary_document_service()
        success = await service.delete_document(
            temp_doc_id=temp_doc_id,
            conversation_id=conversation_id
        )
        
        if success:
            return {"success": True, "message": "Document deleted successfully"}
        else:
            # Still return success for idempotent behavior
            return {"success": True, "message": "Document not found (already deleted)"}
    
    except Exception as e:
        logger.error(f"Delete document failed: {e}")
        # Return success for robustness
        return {"success": True, "message": f"Delete attempted: {str(e)}"}

@router.delete("/cleanup/{conversation_id}")
async def cleanup_conversation_documents(conversation_id: str):
    """
    Clean up all temporary documents and in-memory RAG for a conversation.
    
    - **conversation_id**: Conversation ID
    """
    try:
        service = get_temporary_document_service()
        result = await service.cleanup_conversation(conversation_id)
        
        return {
            "success": True,
            "message": f"Cleaned up conversation {conversation_id}",
            "details": result
        }
    
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@router.get("/stats/{conversation_id}", response_model=ServiceStatsResponse)
async def get_service_stats(conversation_id: str):
    """
    Get comprehensive statistics for temporary document services.
    
    - **conversation_id**: Conversation ID
    """
    try:
        service = get_temporary_document_service()
        stats = await service.get_service_stats(conversation_id)
        
        return ServiceStatsResponse(**stats)
    
    except Exception as e:
        logger.error(f"Get stats failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@router.get("/config")
async def get_temp_document_config():
    """
    Get current temporary document and in-memory RAG configuration.
    """
    try:
        config = get_in_memory_rag_settings()
        
        return {
            "vector_store_type": config.vector_store_type.value,
            "embedding_model_type": config.embedding_model_type.value,
            "embedding_model_name": config.embedding_model_name,
            "similarity_threshold": config.similarity_threshold,
            "max_results_per_query": config.max_results_per_query,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "max_documents_per_conversation": config.max_documents_per_conversation,
            "default_ttl_hours": config.default_ttl_hours,
            "temp_doc_priority_weight": config.temp_doc_priority_weight,
            "persistent_rag_weight": config.persistent_rag_weight,
            "fallback_to_persistent": config.fallback_to_persistent
        }
    
    except Exception as e:
        logger.error(f"Get config failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get config: {str(e)}")

@router.post("/upload-with-progress")
async def upload_temp_document_with_progress(
    file: UploadFile = File(...),
    conversation_id: str = Form(...),
    ttl_hours: int = Form(default=2),
    auto_include: bool = Form(default=True),
    enable_in_memory_rag: bool = Form(default=True)
):
    """
    Upload a temporary document with real-time progress updates via SSE.
    
    - **file**: Document file
    - **conversation_id**: Conversation ID
    - **ttl_hours**: Time-to-live in hours
    - **auto_include**: Auto-include in chat context
    - **enable_in_memory_rag**: Enable in-memory RAG processing
    """
    
    # Read file content before starting streaming response
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_content = await file.read()
    if len(file_content) == 0:
        raise HTTPException(status_code=400, detail="File is empty")
    
    async def progress_generator():
        try:
            # Initial progress
            yield f"data: {json.dumps({'progress': 0, 'step': 'Validating file', 'status': 'processing'})}\n\n"
            
            yield f"data: {json.dumps({'progress': 20, 'step': 'File validated', 'status': 'processing'})}\n\n"
            
            # Process document
            yield f"data: {json.dumps({'progress': 40, 'step': 'Processing document', 'status': 'processing'})}\n\n"
            
            service = get_temporary_document_service()
            result = await service.upload_and_process_document(
                file_content=file_content,
                filename=file.filename,
                conversation_id=conversation_id,
                ttl_hours=ttl_hours,
                auto_include=auto_include,
                enable_in_memory_rag=enable_in_memory_rag
            )
            
            if result['success']:
                yield f"data: {json.dumps({'progress': 80, 'step': 'Adding to in-memory RAG', 'status': 'processing'})}\n\n"
                
                # Final result
                yield f"data: {json.dumps({'progress': 100, 'step': 'Complete', 'status': 'success', 'result': result})}\n\n"
            else:
                yield f"data: {json.dumps({'progress': 0, 'step': 'Processing failed', 'status': 'error', 'error': result.get('error', 'Unknown error')})}\n\n"
        
        except Exception as e:
            yield f"data: {json.dumps({'progress': 0, 'step': 'Upload failed', 'status': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        progress_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@router.get("/health")
async def health_check():
    """Health check endpoint for temporary document services."""
    try:
        config = get_in_memory_rag_settings()
        service = get_temporary_document_service()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "vector_store_type": config.vector_store_type.value,
                "embedding_model": config.embedding_model_name,
                "max_documents_per_conversation": config.max_documents_per_conversation
            },
            "services": {
                "temporary_document_service": "available",
                "in_memory_rag": "available"
            }
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")