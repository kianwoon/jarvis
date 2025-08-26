"""
FastAPI router for Notebook management endpoints.
Provides CRUD operations and RAG functionality for notebooks.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path, UploadFile, File, Request, Body
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, DatabaseError
from sqlalchemy import and_, func, text
from typing import List, Optional, Dict, Any
import logging
import uuid
import os
import hashlib
import json
import re
from datetime import datetime

from app.core.db import get_db, KnowledgeGraphDocument
from app.models.notebook_models import (
    NotebookCreateRequest, NotebookUpdateRequest, NotebookResponse, NotebookDetailResponse,
    NotebookListResponse, NotebookDocumentAddRequest, NotebookDocumentBulkRequest,
    NotebookRAGRequest, NotebookRAGResponse, NotebookConversationRequest,
    NotebookDocumentBulkResponse, NotebookStatsResponse, NotebookOperationResponse,
    NotebookError, NotebookValidationError, NotebookDocumentResponse, 
    NotebookConversationResponse, DocumentDeleteRequest, DocumentDeleteResponse,
    DocumentUsageInfo, DocumentDeletionSummary, NotebookChatRequest, NotebookChatResponse,
    # Memory models
    MemoryCreateRequest, MemoryUpdateRequest, MemoryResponse, MemoryListResponse,
    # Chunk editing models
    ChunkUpdateRequest, ChunkResponse, ChunkListResponse, BulkChunkReEmbedRequest,
    ChunkOperationResponse, BulkChunkOperationResponse
)
from pydantic import BaseModel
from app.services.notebook_service import NotebookService
from app.services.notebook_rag_service import NotebookRAGService
from app.services.hierarchical_notebook_rag_service import get_hierarchical_notebook_rag_service
from app.services.document_admin_service import DocumentAdminService
from app.services.chunk_management_service import ChunkManagementService
from app.services.ai_task_planner import ai_task_planner, TaskExecutionPlan
from app.services.ai_verification_service import ai_verification_service
from app.core.llm_settings_cache import get_llm_settings, get_main_llm_full_config
from app.core.notebook_llm_settings_cache import get_notebook_llm_full_config
from app.core.notebook_source_templates_cache import apply_source_templates
from app.llm.ollama import OllamaLLM
from app.llm.base import LLMConfig

# Upload response model
class NotebookUploadResponse(BaseModel):
    status: str
    document_id: str
    filename: str
    file_id: str
    total_chunks: int
    unique_chunks: int
    duplicates_filtered: int
    collection: str
    pages_processed: int
    message: str

# Document processing imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from app.core.embedding_settings_cache import get_embedding_settings
from app.core.vector_db_settings_cache import get_vector_db_settings
from app.utils.metadata_extractor import MetadataExtractor
from app.rag.bm25_processor import BM25Processor
from utils.deduplication import hash_text, get_existing_hashes, get_existing_doc_ids, filter_new_chunks
from app.core.document_classifier import get_document_classifier
from app.core.collection_registry_cache import get_collection_config

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()

def _should_use_intelligent_planning(query: str, intent_analysis: dict, total_available: int) -> bool:
    """
    Detect when to use intelligent AI planning pipeline.
    
    Uses query analysis to determine if the request requires intelligent
    planning, execution, and verification for complete, accurate results.
    
    Args:
        query: User query string
        intent_analysis: Results from query intent analysis
        total_available: Total items available in notebook
        
    Returns:
        bool: True if intelligent pipeline should be used
    """
    query_lower = query.lower()
    
    # Enumeration query indicators - comprehensive listing requests
    enumeration_indicators = [
        'list all', 'show all', 'enumerate', 'table format', 'in table', 
        'as table', 'table form', 'complete list', 'full list', 'every',
        'comprehensive', 'overview of all', 'summary of all'
    ]
    
    # Complex analytical indicators - requiring structured analysis
    analytical_indicators = [
        'analyze', 'compare', 'categorize', 'organize', 'group by',
        'break down', 'summarize by', 'pattern', 'trend', 'relationship',
        'correlation', 'distribution', 'statistics'
    ]
    
    # Quality-critical indicators - high precision requirements
    quality_indicators = [
        'count', 'total', 'exactly', 'precise', 'accurate', 'complete',
        'comprehensive', 'thorough', 'detailed', 'exhaustive'
    ]
    
    # Check for enumeration queries
    has_enumeration = any(indicator in query_lower for indicator in enumeration_indicators)
    
    # Check for analytical complexity
    has_analysis = any(indicator in query_lower for indicator in analytical_indicators)
    
    # Check for quality requirements
    needs_precision = any(indicator in query_lower for indicator in quality_indicators)
    
    # Check intent analysis results
    wants_comprehensive = intent_analysis.get('wants_comprehensive', False)
    query_type = intent_analysis.get('query_type', 'filtered')
    confidence = intent_analysis.get('confidence', 0.8)
    
    # Use intelligent pipeline if:
    # 1. Explicit enumeration request
    # 2. High-complexity analytical query
    # 3. Quality-critical requirements
    # 4. Intent analysis indicates comprehensive need
    # 5. Large dataset with structured requirements
    
    should_use = (
        has_enumeration or
        has_analysis or 
        needs_precision or
        wants_comprehensive or
        (query_type == 'comprehensive' and confidence > 0.7) or
        (total_available > 50 and (has_enumeration or wants_comprehensive))
    )
    
    if should_use:
        logger.info(f"[AI_PIPELINE] Intelligent planning triggered: enumeration={has_enumeration}, "
                   f"analysis={has_analysis}, precision={needs_precision}, comprehensive={wants_comprehensive}")
    
    return should_use

async def _analyze_query_intent(query: str) -> dict:
    """
    Analyze query intent using AI-powered QueryIntentAnalyzer for intelligent understanding.
    
    Provides comprehensive intent analysis including:
    - Query type detection (comprehensive, filtered, specific)
    - Quantity intent analysis (all, limited, few, single)
    - Confidence scoring and semantic understanding
    - Context-aware categorization
    
    Args:
        query: The user query string
        
    Returns:
        dict: Query analysis results with intent classification and confidence
    """
    try:
        from app.services.query_intent_analyzer import analyze_query_intent
        
        # Use AI-powered intent analysis with notebook LLM config
        notebook_llm_config = get_notebook_llm_full_config()
        intent_result = await analyze_query_intent(query, llm_config=notebook_llm_config)
        
        # Transform to expected format while preserving AI insights
        return {
            "wants_comprehensive": intent_result.get('quantity_intent') == 'all' or intent_result.get('scope') == 'comprehensive',
            "confidence": intent_result.get('confidence', 0.8),
            "query_type": intent_result.get('scope', 'filtered'),
            "quantity_intent": intent_result.get('quantity_intent', 'limited'),
            "user_type": intent_result.get('user_type', 'casual'),
            "completeness_preference": intent_result.get('completeness_preference', 'balanced'),
            "context": intent_result.get('context', {}),
            "reasoning": intent_result.get('reasoning', 'AI semantic analysis'),
            "ai_powered": True
        }
        
    except Exception as e:
        logger.warning(f"AI intent analysis failed, using semantic fallback: {str(e)}")
        # Intelligent fallback based on query characteristics
        query_lower = query.lower()
        
        # Semantic indicators for comprehensive queries
        comprehensive_indicators = (
            any(word in query_lower for word in ['all', 'every', 'complete', 'comprehensive', 'overview', 'summary']) or
            len(query.split()) > 8 or  # Complex queries often want comprehensive results
            any(phrase in query_lower for phrase in ['give me', 'show me', 'list all', 'find everything'])
        )
        
        return {
            "wants_comprehensive": comprehensive_indicators,
            "confidence": 0.7,  # Moderate confidence for semantic fallback
            "query_type": "comprehensive" if comprehensive_indicators else "filtered",
            "quantity_intent": "all" if comprehensive_indicators else "limited",
            "fallback_reason": str(e),
            "ai_powered": False
        }

async def _optimize_max_sources_for_intent(query: str, current_max_sources: int, notebook_id: str, rag_service) -> int:
    """
    Intelligently optimize max_sources based on AI intent analysis and dynamic content counting.
    
    Uses QueryIntentAnalyzer for sophisticated optimization strategies based on:
    - AI-powered query intent and scope analysis
    - Actual available content in the notebook
    - User preferences for comprehensive vs targeted results
    - Dynamic content-aware calculations (no hardcoded limits)
    
    Args:
        query: The user query string
        current_max_sources: Current max_sources value
        notebook_id: Notebook ID for content counting
        rag_service: RAG service instance for dynamic counting
        
    Returns:
        Intelligently optimized max_sources value based on intent and available content
    """
    try:
        # Get AI-powered intent analysis first
        intent_analysis = await _analyze_query_intent(query)
        wants_comprehensive = intent_analysis.get("wants_comprehensive", False)
        quantity_intent = intent_analysis.get("quantity_intent", "limited")
        confidence = intent_analysis.get("confidence", 0.5)
        
        # Get actual content count for this notebook
        try:
            total_available = await rag_service.get_actual_content_count(notebook_id)
            
            # Check if counting returned 0 but we should be conservative
            if total_available == 0:
                # Check for comprehensive queries or listing requests that likely have content
                listing_keywords = ["list", "all", "show", "find", "get", "projects", "documents", "memories", "everything"]
                has_listing_intent = any(keyword in query.lower() for keyword in listing_keywords)
                
                if wants_comprehensive or quantity_intent == "all" or has_listing_intent:
                    logger.warning(f"[ADAPTIVE_RETRIEVAL] Content count returned 0 for query '{query}' that suggests content exists - using conservative fallback")
                    total_available = max(current_max_sources * 3, 200)  # Conservative estimate for comprehensive queries
                else:
                    logger.warning(f"[ADAPTIVE_RETRIEVAL] Content count returned 0 for specific query '{query}' - using modest fallback")
                    total_available = max(current_max_sources, 50)  # Modest fallback for specific queries
                
        except Exception as count_error:
            logger.warning(f"Could not get content count, using fallback: {count_error}")
            # For comprehensive queries, be more conservative when counting fails
            if wants_comprehensive:
                total_available = max(current_max_sources * 3, 200)  # Conservative estimate for comprehensive
            else:
                total_available = max(current_max_sources * 2, 100)  # Standard fallback
        user_type = intent_analysis.get("user_type", "casual")
        
        logger.info(f"[ADAPTIVE_RETRIEVAL] Intent analysis - comprehensive: {wants_comprehensive}, quantity: {quantity_intent}, confidence: {confidence:.2f}, user_type: {user_type}")
        
        # Calculate optimized limit based on intent and available content
        if wants_comprehensive and quantity_intent == "all":
            # User wants everything - provide all available content up to reasonable limits
            if total_available <= 100:
                optimized_limit = total_available  # Give them everything
            elif total_available <= 500:
                optimized_limit = min(total_available, int(total_available * 0.9))  # 90% of available
            else:
                optimized_limit = min(500, int(total_available * 0.7))  # Cap at 500 but use 70% of available
            
            logger.info(f"[ADAPTIVE_RETRIEVAL] Comprehensive query detected: providing {optimized_limit} sources from {total_available} available")
            
        elif quantity_intent == "limited" or user_type == "researcher":
            # Moderate coverage - balance comprehensiveness with manageability
            if total_available <= 50:
                optimized_limit = total_available  # Use everything if limited content
            elif total_available <= 200:
                optimized_limit = min(int(total_available * 0.5), 100)  # 50% up to 100
            else:
                optimized_limit = min(150, int(total_available * 0.3))  # 30% up to 150
                
        elif quantity_intent in ["few", "single"]:
            # Targeted results - focus on relevance
            optimized_limit = min(current_max_sources, 25)  # Keep it focused
            
        else:
            # Default case - moderate scaling based on available content
            if total_available <= 30:
                optimized_limit = total_available
            else:
                base_multiplier = 2 if confidence > 0.8 else 1.5
                optimized_limit = min(int(current_max_sources * base_multiplier), int(total_available * 0.4))
        
        # Apply confidence-based adjustments
        if confidence < 0.6:
            # Lower confidence - be more conservative
            optimized_limit = min(optimized_limit, int(optimized_limit * 0.8))
        elif confidence > 0.9:
            # High confidence - can be more aggressive
            optimized_limit = min(optimized_limit * 1.2, total_available)
        
        # Ensure minimum viable limit
        optimized_limit = max(optimized_limit, 5)
        
        # Log the optimization decision
        if optimized_limit != current_max_sources:
            logger.info(f"[ADAPTIVE_RETRIEVAL] Optimized max_sources from {current_max_sources} to {optimized_limit} (total available: {total_available})")
            logger.info(f"[ADAPTIVE_RETRIEVAL] Reasoning: {intent_analysis.get('reasoning', 'AI optimization')}")
        
        return int(optimized_limit)
        
    except Exception as e:
        logger.error(f"[ADAPTIVE_RETRIEVAL] Optimization failed: {str(e)}")
        # Fallback to modest scaling based on current value
        return min(current_max_sources * 2, 100)

# Initialize services
notebook_service = NotebookService()
notebook_rag_service = NotebookRAGService()
chunk_management_service = ChunkManagementService()

# HTTP embedding function for document processing
import requests

class HTTPEmbeddingFunction:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        if not self.endpoint:
            raise ValueError("Embedding endpoint must be provided - no hardcoding allowed")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            normalized_text = text.lower().strip()
            payload = {"texts": [normalized_text]}
            
            try:
                resp = requests.post(
                    self.endpoint, 
                    json=payload,
                    timeout=30
                )
                resp.raise_for_status()
                embedding_data = resp.json()["embeddings"][0]
                embeddings.append(embedding_data)
                
            except requests.exceptions.Timeout as e:
                logger.error(f"Embedding service timeout: {str(e)}")
                raise HTTPException(status_code=504, detail=f"Embedding service timeout: {str(e)}")
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Embedding service connection error: {str(e)}")
                raise HTTPException(status_code=503, detail=f"Embedding service unavailable: {str(e)}")
            except Exception as e:
                logger.error(f"Embedding error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")
                
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

# Milvus utilities
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

def ensure_milvus_collection(collection_name: str, vector_dim: int, uri: str, token: str):
    """Ensure Milvus collection exists with proper schema"""
    connections.connect(uri=uri, token=token)
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="page", dtype=DataType.INT64),
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="uploaded_at", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="author", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="hash", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="bm25_tokens", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="bm25_term_count", dtype=DataType.INT64),
            FieldSchema(name="bm25_unique_terms", dtype=DataType.INT64),
            FieldSchema(name="bm25_top_terms", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="creation_date", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="last_modified_date", dtype=DataType.VARCHAR, max_length=100),
        ]
        schema = CollectionSchema(fields, description="Knowledge base with metadata, deduplication support")
        collection = Collection(collection_name, schema)
    
    # Create index for vector field if not exists
    has_index = False
    for idx in collection.indexes:
        if idx.field_name == "vector":
            has_index = True
            break
    if not has_index:
        collection.create_index(
            field_name="vector",
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 1024}
            }
        )
    return collection

@router.post("/", response_model=NotebookResponse, status_code=201)
async def create_notebook(
    request: NotebookCreateRequest,
    db: Session = Depends(get_db)
):
    """
    Create a new notebook.
    
    Args:
        request: Notebook creation parameters
        db: Database session
        
    Returns:
        Created notebook details
        
    Raises:
        HTTPException: If creation fails
    """
    try:
        logger.info(f"Creating notebook: {request.name}")
        
        notebook = await notebook_service.create_notebook(
            db=db,
            name=request.name,
            description=request.description,
            user_id=request.user_id,
            source_filter=request.source_filter,
            metadata=request.metadata
        )
        
        logger.info(f"Successfully created notebook {notebook.id}")
        return notebook
        
    except ValueError as e:
        logger.error(f"Validation error creating notebook: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except IntegrityError as e:
        logger.error(f"Database integrity error creating notebook: {str(e)}")
        raise HTTPException(status_code=409, detail="Notebook with this name may already exist")
    except DatabaseError as e:
        logger.error(f"Database error creating notebook: {str(e)}")
        raise HTTPException(status_code=500, detail="Database operation failed")
    except Exception as e:
        logger.error(f"Unexpected error creating notebook: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/", response_model=NotebookListResponse)
async def list_notebooks(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    search: Optional[str] = Query(None, description="Search in name or description"),
    db: Session = Depends(get_db)
):
    """
    List notebooks with pagination and filtering.
    
    Args:
        page: Page number (1-based)
        page_size: Number of items per page
        user_id: Filter by user ID
        search: Search query for name/description
        db: Database session
        
    Returns:
        Paginated list of notebooks
    """
    try:
        logger.info(f"Listing notebooks: page={page}, size={page_size}, user_id={user_id}")
        
        result = await notebook_service.list_notebooks(
            db=db,
            page=page,
            page_size=page_size,
            user_id=user_id,
            search=search
        )
        
        return result
        
    except ValueError as e:
        logger.error(f"Validation error listing notebooks: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except DatabaseError as e:
        logger.error(f"Database error listing notebooks: {str(e)}")
        raise HTTPException(status_code=500, detail="Database operation failed")
    except Exception as e:
        logger.error(f"Unexpected error listing notebooks: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{notebook_id}", response_model=NotebookDetailResponse)
async def get_notebook(
    notebook_id: str = Path(..., description="Notebook ID"),
    db: Session = Depends(get_db)
):
    """
    Get detailed notebook information including documents and conversations.
    
    Args:
        notebook_id: Notebook ID
        db: Database session
        
    Returns:
        Detailed notebook information
        
    Raises:
        HTTPException: If notebook not found
    """
    try:
        logger.info(f"Getting notebook details: {notebook_id}")
        
        notebook = await notebook_service.get_notebook_detail(db=db, notebook_id=notebook_id)
        
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")
        
        return notebook
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error getting notebook: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except DatabaseError as e:
        logger.error(f"Database error getting notebook: {str(e)}")
        raise HTTPException(status_code=500, detail="Database operation failed")
    except Exception as e:
        logger.error(f"Unexpected error getting notebook: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.put("/{notebook_id}", response_model=NotebookResponse)
async def update_notebook(
    notebook_id: str = Path(..., description="Notebook ID"),
    request: NotebookUpdateRequest = None,
    db: Session = Depends(get_db)
):
    """
    Update notebook information.
    
    Args:
        notebook_id: Notebook ID
        request: Update parameters
        db: Database session
        
    Returns:
        Updated notebook details
        
    Raises:
        HTTPException: If notebook not found or update fails
    """
    try:
        logger.info(f"Updating notebook: {notebook_id}")
        
        if not request:
            raise HTTPException(status_code=400, detail="Update request body is required")
        
        notebook = await notebook_service.update_notebook(
            db=db,
            notebook_id=notebook_id,
            name=request.name,
            description=request.description,
            user_id=request.user_id,
            source_filter=request.source_filter,
            metadata=request.metadata
        )
        
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")
        
        logger.info(f"Successfully updated notebook {notebook_id}")
        return notebook
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error updating notebook: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except IntegrityError as e:
        logger.error(f"Database integrity error updating notebook: {str(e)}")
        raise HTTPException(status_code=409, detail="Update would violate constraints")
    except DatabaseError as e:
        logger.error(f"Database error updating notebook: {str(e)}")
        raise HTTPException(status_code=500, detail="Database operation failed")
    except Exception as e:
        logger.error(f"Unexpected error updating notebook: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/{notebook_id}", response_model=NotebookOperationResponse)
async def delete_notebook(
    notebook_id: str = Path(..., description="Notebook ID"),
    db: Session = Depends(get_db)
):
    """
    Delete a notebook and all associated data.
    
    Args:
        notebook_id: Notebook ID
        db: Database session
        
    Returns:
        Operation result
        
    Raises:
        HTTPException: If notebook not found or deletion fails
    """
    try:
        logger.info(f"Deleting notebook: {notebook_id}")
        
        success = await notebook_service.delete_notebook(db=db, notebook_id=notebook_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Notebook not found")
        
        logger.info(f"Successfully deleted notebook {notebook_id}")
        return NotebookOperationResponse(
            success=True,
            message=f"Notebook {notebook_id} deleted successfully"
        )
        
    except HTTPException:
        raise
    except DatabaseError as e:
        logger.error(f"Database error deleting notebook: {str(e)}")
        raise HTTPException(status_code=500, detail="Database operation failed")
    except Exception as e:
        logger.error(f"Unexpected error deleting notebook: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{notebook_id}/documents", response_model=NotebookDocumentResponse, status_code=201)
async def add_document_to_notebook(
    notebook_id: str = Path(..., description="Notebook ID"),
    request: NotebookDocumentAddRequest = None,
    db: Session = Depends(get_db)
):
    """
    Add a document to a notebook.
    
    Args:
        notebook_id: Notebook ID
        request: Document addition parameters
        db: Database session
        
    Returns:
        Added document details
        
    Raises:
        HTTPException: If notebook not found or addition fails
    """
    try:
        logger.info(f"Adding document {request.document_id} to notebook {notebook_id}")
        
        if not request:
            raise HTTPException(status_code=400, detail="Document request body is required")
        
        document = await notebook_service.add_document_to_notebook(
            db=db,
            notebook_id=notebook_id,
            document_id=request.document_id,
            document_name=request.document_name,
            document_type=request.document_type,
            milvus_collection=request.milvus_collection,
            metadata=request.metadata
        )
        
        if not document:
            raise HTTPException(status_code=404, detail="Notebook not found")
        
        logger.info(f"Successfully added document to notebook {notebook_id}")
        return document
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error adding document: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except IntegrityError as e:
        logger.error(f"Database integrity error adding document: {str(e)}")
        raise HTTPException(status_code=409, detail="Document may already exist in notebook")
    except DatabaseError as e:
        logger.error(f"Database error adding document: {str(e)}")
        raise HTTPException(status_code=500, detail="Database operation failed")
    except Exception as e:
        logger.error(f"Unexpected error adding document: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/{notebook_id}/documents/{document_id}", response_model=NotebookOperationResponse)
async def remove_document_from_notebook(
    notebook_id: str = Path(..., description="Notebook ID"),
    document_id: str = Path(..., description="Document ID"),
    db: Session = Depends(get_db)
):
    """
    Remove a document from a notebook.
    
    Args:
        notebook_id: Notebook ID
        document_id: Document ID to remove
        db: Database session
        
    Returns:
        Operation result
        
    Raises:
        HTTPException: If notebook or document not found
    """
    try:
        logger.info(f"Removing document {document_id} from notebook {notebook_id}")
        
        success = await notebook_service.remove_document_from_notebook(
            db=db,
            notebook_id=notebook_id,
            document_id=document_id
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Notebook or document not found")
        
        logger.info(f"Successfully removed document from notebook {notebook_id}")
        return NotebookOperationResponse(
            success=True,
            message=f"Document {document_id} removed from notebook"
        )
        
    except HTTPException:
        raise
    except DatabaseError as e:
        logger.error(f"Database error removing document: {str(e)}")
        raise HTTPException(status_code=500, detail="Database operation failed")
    except Exception as e:
        logger.error(f"Unexpected error removing document: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{notebook_id}/query", response_model=NotebookRAGResponse)
async def query_notebook(
    notebook_id: str = Path(..., description="Notebook ID"),
    request: NotebookRAGRequest = None,
    db: Session = Depends(get_db)
):
    """
    Query notebook documents using RAG.
    
    Args:
        notebook_id: Notebook ID
        request: Query parameters
        db: Database session
        
    Returns:
        RAG query results
        
    Raises:
        HTTPException: If notebook not found or query fails
    """
    try:
        logger.info(f"Querying notebook {notebook_id} with query: {request.query[:50]}...")
        
        if not request:
            raise HTTPException(status_code=400, detail="Query request body is required")
        
        # Verify notebook exists
        notebook_exists = await notebook_service.notebook_exists(db=db, notebook_id=notebook_id)
        if not notebook_exists:
            raise HTTPException(status_code=404, detail="Notebook not found")
        
        result = await notebook_rag_service.query_notebook(
            db=db,
            notebook_id=notebook_id,
            query=request.query,
            top_k=request.top_k,
            include_metadata=request.include_metadata,
            collection_filter=request.collection_filter
        )
        
        logger.info(f"Successfully queried notebook {notebook_id}, found {len(result.sources)} sources")
        return result
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error querying notebook: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error querying notebook: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{notebook_id}/query-optimized", response_model=NotebookRAGResponse)
async def notebook_rag_query_optimized(
    notebook_id: str = Path(..., description="Notebook ID"),
    request: NotebookRAGRequest = Body(..., description="RAG query request"),
    db: Session = Depends(get_db)
):
    """
    Enhanced RAG query with hierarchical retrieval and context optimization.
    Uses Google NotebookLM-like strategies to manage context window efficiently.
    """
    try:
        logger.info(f"Optimized RAG query for notebook {notebook_id}: '{request.query[:100]}...'")
        
        # Verify notebook exists
        notebook_exists = await notebook_service.notebook_exists(db=db, notebook_id=notebook_id)
        if not notebook_exists:
            raise HTTPException(status_code=404, detail="Notebook not found")
        
        # Use hierarchical RAG service
        hierarchical_rag_service = get_hierarchical_notebook_rag_service()
        result = await hierarchical_rag_service.query_with_context_optimization(
            notebook_id=notebook_id,
            query=request.query,
            top_k=request.top_k,
            include_metadata=request.include_metadata,
            collection_filter=request.collection_filter,
            db=db
        )

        logger.info(f"Optimized RAG query completed: {len(result.sources)} sources, strategy: {result.metadata.get('strategy', 'unknown')}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during optimized notebook RAG query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{notebook_id}/context-stats")
async def get_notebook_context_stats(
    notebook_id: str = Path(..., description="Notebook ID"),
    query: str = Query(..., description="Sample query to analyze"),
    db: Session = Depends(get_db)
):
    """
    Get context usage statistics for notebook RAG queries.
    Helps understand token usage and optimization opportunities.
    """
    try:
        # Verify notebook exists
        notebook_exists = await notebook_service.notebook_exists(db=db, notebook_id=notebook_id)
        if not notebook_exists:
            raise HTTPException(status_code=404, detail="Notebook not found")
        
        # Get context statistics
        hierarchical_rag_service = get_hierarchical_notebook_rag_service()
        stats = await hierarchical_rag_service.get_context_stats(notebook_id, query)
        
        return {
            "notebook_id": notebook_id,
            "query_sample": query[:100] + "..." if len(query) > 100 else query,
            "context_analysis": stats,
            "recommendations": {
                "use_hierarchical": stats.get("optimization_needed", False),
                "recommended_max_chunks": stats.get("recommended_max_chunks", 10),
                "current_efficiency": "low" if stats.get("would_exceed_budget", False) else "good"
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting context stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{notebook_id}/conversations", response_model=NotebookConversationResponse, status_code=201)
async def start_notebook_conversation(
    notebook_id: str = Path(..., description="Notebook ID"),
    request: NotebookConversationRequest = None,
    db: Session = Depends(get_db)
):
    """
    Start a new conversation with the notebook.
    
    Args:
        notebook_id: Notebook ID
        request: Conversation parameters
        db: Database session
        
    Returns:
        Conversation details
        
    Raises:
        HTTPException: If notebook not found or conversation creation fails
    """
    try:
        logger.info(f"Starting conversation {request.conversation_id} for notebook {notebook_id}")
        
        if not request:
            raise HTTPException(status_code=400, detail="Conversation request body is required")
        
        conversation = await notebook_service.start_conversation(
            db=db,
            notebook_id=notebook_id,
            conversation_id=request.conversation_id
        )
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Notebook not found")
        
        logger.info(f"Successfully started conversation for notebook {notebook_id}")
        return conversation
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error starting conversation: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except IntegrityError as e:
        logger.error(f"Database integrity error starting conversation: {str(e)}")
        raise HTTPException(status_code=409, detail="Conversation may already exist")
    except DatabaseError as e:
        logger.error(f"Database error starting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Database operation failed")
    except Exception as e:
        logger.error(f"Unexpected error starting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{notebook_id}/stats", response_model=NotebookStatsResponse)
async def get_notebook_stats(
    notebook_id: str = Path(..., description="Notebook ID"),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive statistics for a notebook.
    
    Args:
        notebook_id: Notebook ID
        db: Database session
        
    Returns:
        Notebook statistics
        
    Raises:
        HTTPException: If notebook not found
    """
    try:
        logger.info(f"Getting stats for notebook {notebook_id}")
        
        stats = await notebook_service.get_notebook_stats(db=db, notebook_id=notebook_id)
        
        if not stats:
            raise HTTPException(status_code=404, detail="Notebook not found")
        
        return stats
        
    except HTTPException:
        raise
    except DatabaseError as e:
        logger.error(f"Database error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Database operation failed")
    except Exception as e:
        logger.error(f"Unexpected error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{notebook_id}/upload", response_model=NotebookUploadResponse, status_code=201)
async def upload_file_to_notebook(
    notebook_id: str = Path(..., description="Notebook ID"),
    file: UploadFile = File(..., description="File to upload"),
    db: Session = Depends(get_db)
):
    """
    Upload a file directly to a notebook.
    
    All notebook documents are automatically stored in the 'notebooks' collection for
    consistent organization and retrieval.
    
    This endpoint accepts file upload, processes it (extracts text, creates embeddings),
    adds it to the vector store, and then links it to the specified notebook.
    
    Args:
        notebook_id: Notebook ID to add the document to
        file: File to upload (PDF, TXT, etc.)
        db: Database session
        
    Returns:
        Upload result with document information, collection set to 'notebooks'
        
    Raises:
        HTTPException: If notebook not found, notebooks collection missing, or processing fails
    """
    try:
        logger.info(f"Uploading file {file.filename} to notebook {notebook_id}")
        
        # 1. Verify notebook exists
        notebook_exists = await notebook_service.notebook_exists(db=db, notebook_id=notebook_id)
        if not notebook_exists:
            raise HTTPException(status_code=404, detail="Notebook not found")
        
        # 2. Save file temporarily
        temp_path = f"/tmp/{file.filename}"
        try:
            with open(temp_path, "wb") as f:
                f.write(await file.read())
            logger.info(f"📁 Saved {file.filename} to {temp_path}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to save file: {str(e)}")
        
        # 3. Load document based on file type
        try:
            if file.filename.lower().endswith('.pdf'):
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                logger.info(f"✅ PDF loaded: {len(docs)} pages from {file.filename}")
            else:
                # For non-PDF files, read as text
                with open(temp_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                from langchain.schema import Document
                docs = [Document(page_content=content, metadata={"source": file.filename, "page": 0})]
                logger.info(f"✅ Text file loaded: {file.filename}")
                
        except Exception as e:
            logger.error(f"❌ Error loading file: {e}")
            os.remove(temp_path) if os.path.exists(temp_path) else None
            raise HTTPException(status_code=400, detail=f"Failed to load file: {str(e)}")
        
        # 4. Text splitting
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
            length_function=len,
            is_separator_regex=False
        )
        
        chunks = splitter.split_documents(docs)
        total_chunks = len(chunks)
        logger.info(f"✅ Created {total_chunks} chunks")
        
        # 5. Generate file ID and metadata
        with open(temp_path, "rb") as f:
            file_content = f.read()
            file_id = hashlib.sha256(file_content).hexdigest()[:12]
            file_size_bytes = len(file_content)
        
        logger.info(f"🔑 File ID: {file_id}")
        
        # Initialize BM25 processor
        bm25_processor = BM25Processor()
        
        # Extract file metadata
        file_metadata = MetadataExtractor.extract_metadata(temp_path, file.filename)
        
        # 6. Enhanced metadata setting
        for i, chunk in enumerate(chunks):
            if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                chunk.metadata = {}
            
            original_page = chunk.metadata.get('page', 0)
            
            chunk.metadata.update({
                'source': file.filename.lower(),
                'page': original_page,
                'doc_type': 'pdf' if file.filename.lower().endswith('.pdf') else 'text',
                'uploaded_at': datetime.now().isoformat(),
                'section': f"chunk_{i}",
                'author': '',
                'chunk_index': i,
                'file_id': file_id,
                'creation_date': file_metadata['creation_date'],
                'last_modified_date': file_metadata['last_modified_date']
            })
            
            # Add hash and doc_id
            chunk_hash = hash_text(chunk.page_content)
            chunk.metadata['hash'] = chunk_hash
            chunk.metadata['doc_id'] = f"{file_id}_p{original_page}_c{i}"
            
            # Add BM25 preprocessing
            bm25_metadata = bm25_processor.prepare_document_for_bm25(chunk.page_content, chunk.metadata)
            chunk.metadata.update(bm25_metadata)
        
        # 7. Determine target collection - notebooks always use 'notebooks' collection
        target_collection = "notebooks"
        logger.info(f"📚 Using dedicated notebooks collection: {target_collection}")
        
        # Validate that the notebooks collection exists
        collection_info = get_collection_config(target_collection)
        if not collection_info:
            raise HTTPException(
                status_code=500, 
                detail=f"Notebooks collection '{target_collection}' not found. Please create it first."
            )
        
        # Add collection info to chunk metadata
        for chunk in chunks:
            chunk.metadata['collection_name'] = target_collection
        
        # 8. Vector DB storage
        from app.utils.vector_db_migration import migrate_vector_db_settings
        vector_db_cfg = migrate_vector_db_settings(get_vector_db_settings())
        
        # Find active Milvus database
        milvus_db = None
        for db_config in vector_db_cfg.get("databases", []):
            if db_config.get("id") == "milvus" and db_config.get("enabled"):
                milvus_db = db_config
                break
        
        if not milvus_db:
            os.remove(temp_path) if os.path.exists(temp_path) else None
            raise HTTPException(status_code=500, detail="No active Milvus configuration found")
        
        logger.info("🔄 Using Milvus vector database")
        milvus_cfg = milvus_db.get("config", {})
        uri = milvus_cfg.get("MILVUS_URI")
        token = milvus_cfg.get("MILVUS_TOKEN")
        collection = target_collection
        vector_dim = int(milvus_cfg.get("dimension", 2560))
        
        # Ensure collection exists
        collection_obj = ensure_milvus_collection(collection, vector_dim=vector_dim, uri=uri, token=token)
        
        # Get embedding configuration
        embedding_cfg = get_embedding_settings()
        embedding_model = embedding_cfg.get('embedding_model')
        embedding_endpoint = embedding_cfg.get('embedding_endpoint')
        
        if embedding_endpoint:
            embeddings = HTTPEmbeddingFunction(embedding_endpoint)
            logger.info("✅ Using HTTP embedding endpoint")
        else:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
            logger.info(f"✅ Using HuggingFace embeddings: {embedding_model}")
        
        # Enhanced deduplication check
        collection_obj.load()
        existing_hashes = get_existing_hashes(collection_obj)
        existing_doc_ids = get_existing_doc_ids(collection_obj)
        
        logger.info(f"📊 Deduplication analysis:")
        logger.info(f"   - Total chunks to process: {len(chunks)}")
        logger.info(f"   - Existing hashes in DB: {len(existing_hashes)}")
        logger.info(f"   - Existing doc_ids in DB: {len(existing_doc_ids)}")
        
        # Filter duplicates
        unique_chunks = []
        duplicate_count = 0
        
        for chunk in chunks:
            chunk_hash = chunk.metadata.get('hash')
            doc_id = chunk.metadata.get('doc_id')
            
            is_duplicate = (chunk_hash in existing_hashes or doc_id in existing_doc_ids)
            
            if is_duplicate:
                duplicate_count += 1
            else:
                unique_chunks.append(chunk)
        
        logger.info(f"📊 After deduplication:")
        logger.info(f"   - Unique chunks to insert: {len(unique_chunks)}")
        logger.info(f"   - Duplicates filtered: {duplicate_count}")
        
        if not unique_chunks:
            # All chunks are duplicates, but we can still link existing document to notebook
            logger.info(f"📋 Document already processed, checking if linked to notebook...")
            os.remove(temp_path) if os.path.exists(temp_path) else None
            
            # Try to add existing document to notebook
            try:
                result = await notebook_service.add_document_to_notebook(
                    db=db,
                    notebook_id=notebook_id,
                    document_id=file_id,
                    document_name=file.filename,
                    document_type=file.filename.split('.')[-1].lower() if '.' in file.filename else 'unknown',
                    milvus_collection=target_collection,
                    metadata={
                        'file_size_bytes': file_size_bytes,
                        'total_chunks': total_chunks,
                        'unique_chunks': 0,
                        'processing_status': 'already_exists'
                    }
                )
                
                if result:
                    logger.info(f"✅ Linked existing document {file_id} to notebook {notebook_id}")
                    return NotebookUploadResponse(
                        status="success",
                        document_id=file_id,
                        filename=file.filename,
                        file_id=file_id,
                        total_chunks=total_chunks,
                        unique_chunks=0,
                        duplicates_filtered=total_chunks,
                        collection=target_collection,
                        pages_processed=total_chunks,  # Approximate
                        message=f"Document already exists and has been linked to notebook"
                    )
                else:
                    raise HTTPException(status_code=409, detail="Document already exists in this notebook")
                    
            except Exception as link_error:
                logger.error(f"❌ Error linking existing document: {link_error}")
                if "already exist" in str(link_error).lower():
                    raise HTTPException(status_code=409, detail="Document already exists in this notebook")
                else:
                    raise HTTPException(status_code=500, detail=f"Failed to link document: {str(link_error)}")
        
        # Generate embeddings and insert into Milvus
        unique_ids = [str(uuid.uuid4()) for _ in unique_chunks]
        unique_texts = [chunk.page_content for chunk in unique_chunks]
        
        logger.info(f"🔄 Generating embeddings for {len(unique_texts)} chunks...")
        try:
            embeddings_list = embeddings.embed_documents(unique_texts)
            logger.info(f"✅ Generated {len(embeddings_list)} embeddings")
        except Exception as e:
            logger.error(f"❌ Error generating embeddings: {e}")
            os.remove(temp_path) if os.path.exists(temp_path) else None
            raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {str(e)}")
        
        # Prepare data for insertion
        data = [
            unique_ids,
            embeddings_list,
            unique_texts,
            [chunk.metadata.get('source', '') for chunk in unique_chunks],
            [chunk.metadata.get('page', 0) for chunk in unique_chunks],
            [chunk.metadata.get('doc_type', 'text') for chunk in unique_chunks],
            [chunk.metadata.get('uploaded_at', '') for chunk in unique_chunks],
            [chunk.metadata.get('section', '') for chunk in unique_chunks],
            [chunk.metadata.get('author', '') for chunk in unique_chunks],
            [chunk.metadata.get('hash', '') for chunk in unique_chunks],
            [chunk.metadata.get('doc_id', '') for chunk in unique_chunks],
            [chunk.metadata.get('bm25_tokens', '') for chunk in unique_chunks],
            [chunk.metadata.get('bm25_term_count', 0) for chunk in unique_chunks],
            [chunk.metadata.get('bm25_unique_terms', 0) for chunk in unique_chunks],
            [chunk.metadata.get('bm25_top_terms', '') for chunk in unique_chunks],
            [chunk.metadata.get('creation_date', '') for chunk in unique_chunks],
            [chunk.metadata.get('last_modified_date', '') for chunk in unique_chunks],
        ]
        
        logger.info(f"🔄 Inserting {len(unique_chunks)} chunks into Milvus collection '{collection}'...")
        try:
            insert_result = collection_obj.insert(data)
            collection_obj.flush()
            logger.info(f"✅ Successfully inserted {len(unique_chunks)} chunks")
        except Exception as e:
            logger.error(f"❌ Error inserting into Milvus: {e}")
            os.remove(temp_path) if os.path.exists(temp_path) else None
            raise HTTPException(status_code=500, detail=f"Failed to insert into Milvus: {str(e)}")
        
        # 9. Create document record in database (handle duplicates)
        try:
            document = KnowledgeGraphDocument(
                document_id=file_id,
                filename=file.filename,
                file_hash=hashlib.sha256(file_content).hexdigest(),
                file_size_bytes=file_size_bytes,
                file_type=file.filename.split('.')[-1].lower() if '.' in file.filename else 'unknown',
                milvus_collection=collection,
                processing_status='completed',
                chunks_processed=len(unique_chunks),
                total_chunks=len(chunks),
                processing_completed_at=datetime.now()
            )
            
            db.add(document)
            db.commit()
            db.refresh(document)
            logger.info(f"✅ Created new document record: {file_id}")
            
        except Exception as e:
            db.rollback()
            if "duplicate key value violates unique constraint" in str(e):
                # Document already exists, get the existing one
                logger.info(f"📋 Document {file_id} already exists, using existing record")
                document = db.query(KnowledgeGraphDocument).filter(
                    KnowledgeGraphDocument.document_id == file_id
                ).first()
                if not document:
                    raise HTTPException(status_code=500, detail="Document exists but could not retrieve it")
            else:
                logger.error(f"❌ Failed to create document record: {e}")
                raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        
        # 10. Add document to notebook
        try:
            await notebook_service.add_document_to_notebook(
                db=db,
                notebook_id=notebook_id,
                document_id=file_id,
                document_name=file.filename,
                document_type=file.filename.split('.')[-1].lower() if '.' in file.filename else 'unknown',
                milvus_collection=collection,
                metadata={
                    'file_size_bytes': file_size_bytes,
                    'total_chunks': len(chunks),
                    'unique_chunks': len(unique_chunks),
                    'processing_status': 'completed'
                }
            )
            logger.info(f"✅ Added document to notebook {notebook_id}")
        except Exception as e:
            logger.error(f"❌ Error adding document to notebook: {e}")
            # Document is in vector store but not linked to notebook
            raise HTTPException(
                status_code=500, 
                detail=f"Document processed but failed to link to notebook: {str(e)}"
            )
        
        # Cleanup temporary file
        try:
            os.remove(temp_path)
            logger.info(f"🧹 Cleaned up temporary file: {temp_path}")
        except Exception as e:
            logger.warning(f"⚠️  Could not remove temp file: {e}")
        
        return NotebookUploadResponse(
            status="success",
            document_id=file_id,
            filename=file.filename,
            file_id=file_id,
            total_chunks=len(chunks),
            unique_chunks=len(unique_chunks),
            duplicates_filtered=duplicate_count,
            collection=collection,
            pages_processed=len(docs),
            message=f"Successfully uploaded {file.filename} to notebook {notebook_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error uploading file to notebook: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Document Admin Endpoints

@router.get("/documents/{document_id}/usage", response_model=DocumentUsageInfo)
async def get_document_usage_info(
    document_id: str = Path(..., description="Document ID to analyze"),
    db: Session = Depends(get_db)
):
    """
    Get usage information for a document before deletion.
    Shows which notebooks use it and what the impact of deletion will be.
    """
    try:
        logger.info(f"Getting usage info for document: {document_id}")
        
        admin_service = DocumentAdminService()
        usage_info = await admin_service.get_document_usage_info(db, document_id)
        
        if 'error' in usage_info:
            if usage_info['error'] == 'Document not found':
                raise HTTPException(status_code=404, detail="Document not found")
            else:
                raise HTTPException(status_code=500, detail=usage_info['error'])
        
        return DocumentUsageInfo(**usage_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document usage info: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/documents/permanent", response_model=DocumentDeleteResponse)
async def delete_documents_permanently(
    request: DocumentDeleteRequest,
    db: Session = Depends(get_db)
):
    """
    Permanently delete documents from all systems (Milvus, database, notebooks).
    This operation cannot be undone!
    """
    try:
        logger.info(f"Permanent deletion requested for {len(request.document_ids)} documents")
        
        # Additional safety check
        if not request.confirm_permanent_deletion:
            raise HTTPException(
                status_code=400, 
                detail="Must confirm permanent deletion by setting confirm_permanent_deletion=true"
            )
        
        admin_service = DocumentAdminService()
        
        if len(request.document_ids) == 1:
            # Single document deletion
            result = await admin_service.delete_document_permanently(
                db, 
                request.document_ids[0],
                request.remove_from_notebooks
            )
            
            response_data = {
                'success': result.get('success', False),
                'message': f"Document {'successfully' if result.get('success') else 'failed to be'} deleted permanently",
                'total_requested': 1,
                'successful_deletions': 1 if result.get('success') else 0,
                'failed_deletions': 0 if result.get('success') else 1,
                'deletion_details': [DocumentDeletionSummary(**result)],
                'overall_errors': result.get('errors', [])
            }
        else:
            # Bulk document deletion
            result = await admin_service.bulk_delete_documents(
                db,
                request.document_ids,
                request.remove_from_notebooks
            )
            
            response_data = {
                'success': result.get('success', False),
                'message': f"Bulk deletion completed: {result.get('successful_deletions', 0)} successful, {result.get('failed_deletions', 0)} failed",
                'total_requested': result.get('total_requested', 0),
                'successful_deletions': result.get('successful_deletions', 0),
                'failed_deletions': result.get('failed_deletions', 0),
                'deletion_details': [DocumentDeletionSummary(**detail) for detail in result.get('deletion_details', [])],
                'overall_errors': result.get('overall_errors', [])
            }
        
        logger.info(f"Permanent deletion completed: {response_data['successful_deletions']} successful, {response_data['failed_deletions']} failed")
        return DocumentDeleteResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in permanent document deletion: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/documents/{document_id}/permanent", response_model=DocumentDeleteResponse)
async def delete_single_document_permanently(
    document_id: str = Path(..., description="Document ID to delete permanently"),
    remove_from_notebooks: bool = Query(True, description="Remove from all notebooks"),
    confirm: bool = Query(..., description="Confirmation that deletion is permanent"),
    db: Session = Depends(get_db)
):
    """
    Permanently delete a single document from all systems.
    Convenience endpoint for single document deletion.
    """
    try:
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="Must confirm permanent deletion with confirm=true query parameter"
            )
        
        # Convert to DocumentDeleteRequest format
        request = DocumentDeleteRequest(
            document_ids=[document_id],
            remove_from_notebooks=remove_from_notebooks,
            confirm_permanent_deletion=confirm
        )
        
        # Use the bulk deletion endpoint logic
        return await delete_documents_permanently(request, db)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in single document permanent deletion: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# System-wide Documents Management Endpoint

@router.get("/system/documents", response_model=Dict[str, Any])
async def get_all_system_documents(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Page size"),
    search: Optional[str] = Query(None, description="Search in filename or document_id"),
    file_type: Optional[str] = Query(None, description="Filter by file type"),
    status: Optional[str] = Query(None, description="Filter by processing status"),
    collection: Optional[str] = Query(None, description="Filter by Milvus collection"),
    sort_by: str = Query("created_at", description="Sort field: created_at, filename, file_size_bytes"),
    sort_order: str = Query("desc", description="Sort order: asc, desc"),
    db: Session = Depends(get_db)
):
    """
    Get ALL documents across the entire system with notebook relationships.
    This is the system-wide admin view for document management.
    
    Returns documents with:
    - Document metadata
    - List of notebooks using each document
    - Processing status and collection information
    - File size, type, and creation timestamps
    
    Args:
        page: Page number (1-based)
        page_size: Number of items per page (max 200 for admin)
        search: Search query for filename or document_id
        file_type: Filter by file type (pdf, txt, etc.)
        status: Filter by processing status
        collection: Filter by Milvus collection
        sort_by: Field to sort by
        sort_order: Sort direction (asc/desc)
        db: Database session
        
    Returns:
        Comprehensive system document list with notebook relationships
    """
    try:
        logger.info(f"System documents query: page={page}, size={page_size}, search={search}")
        
        # Build document query with filters using raw SQL for consistency
        where_conditions = []
        params = {"offset": (page - 1) * page_size, "limit": page_size}
        
        if search:
            where_conditions.append("(filename ILIKE :search OR document_id ILIKE :search)")
            params["search"] = f"%{search}%"
            
        if file_type:
            where_conditions.append("file_type = :file_type")
            params["file_type"] = file_type
            
        if status:
            where_conditions.append("processing_status = :status")
            params["status"] = status
            
        if collection:
            where_conditions.append("milvus_collection = :collection")
            params["collection"] = collection
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        # Determine sort field and order
        valid_sort_fields = {"created_at", "filename", "file_size_bytes"}
        sort_field = sort_by if sort_by in valid_sort_fields else "created_at"
        sort_direction = "ASC" if sort_order.lower() == "asc" else "DESC"
        
        # Get total count
        count_query = text(f"""
            SELECT COUNT(*) FROM knowledge_graph_documents 
            WHERE {where_clause}
        """)
        total_count = db.execute(count_query, params).scalar() or 0
        
        # Get documents with pagination
        documents_query = text(f"""
            SELECT * FROM knowledge_graph_documents
            WHERE {where_clause}
            ORDER BY {sort_field} {sort_direction}
            LIMIT :limit OFFSET :offset
        """)
        documents = db.execute(documents_query, params).fetchall()
        
        # Get document IDs for notebook relationship lookup
        document_ids = [doc.document_id for doc in documents]
        
        # Get notebook details efficiently with a join query
        doc_to_notebooks = {}
        if document_ids:
            notebook_query = text("""
                SELECT 
                    nd.document_id,
                    nd.notebook_id,
                    nd.added_at,
                    n.name as notebook_name
                FROM notebook_documents nd
                JOIN notebooks n ON nd.notebook_id = n.id
                WHERE nd.document_id = ANY(:document_ids)
                ORDER BY nd.document_id, nd.added_at DESC
            """)
            
            notebook_results = db.execute(notebook_query, {"document_ids": document_ids}).fetchall()
            
            for row in notebook_results:
                if row.document_id not in doc_to_notebooks:
                    doc_to_notebooks[row.document_id] = []
                
                doc_to_notebooks[row.document_id].append({
                    "id": row.notebook_id,
                    "name": row.notebook_name,
                    "added_at": row.added_at.isoformat() if row.added_at else None
                })
        
        # Build comprehensive response
        system_documents = []
        for doc in documents:
            notebooks_using = doc_to_notebooks.get(doc.document_id, [])
            
            system_documents.append({
                "document_id": doc.document_id,
                "filename": doc.filename,
                "file_type": doc.file_type,
                "file_size_bytes": doc.file_size_bytes or 0,
                "processing_status": doc.processing_status,
                "milvus_collection": doc.milvus_collection,
                "created_at": doc.created_at.isoformat() if doc.created_at else "",
                "updated_at": doc.updated_at.isoformat() if doc.updated_at else "",
                "chunks_processed": getattr(doc, 'chunks_processed', 0),
                "total_chunks": getattr(doc, 'total_chunks', 0),
                "file_hash": getattr(doc, 'file_hash', ''),
                
                # Notebook relationship information
                "notebook_count": len(notebooks_using),
                "notebooks_using": notebooks_using,
                "is_orphaned": len(notebooks_using) == 0,
                
                # Admin metadata
                "processing_completed_at": doc.processing_completed_at.isoformat() if getattr(doc, 'processing_completed_at', None) else None,
                "can_be_deleted": True,  # All system documents can be deleted by admin
            })
        
        # Get summary statistics
        stats_query = text("""
            SELECT 
                COUNT(*) as total_documents,
                COUNT(DISTINCT file_type) as unique_file_types,
                COUNT(DISTINCT milvus_collection) as unique_collections,
                COALESCE(SUM(file_size_bytes), 0) as total_size_bytes,
                COUNT(CASE WHEN processing_status = 'completed' THEN 1 END) as completed_documents,
                COUNT(CASE WHEN processing_status = 'failed' THEN 1 END) as failed_documents
            FROM knowledge_graph_documents
        """)
        stats_result = db.execute(stats_query).fetchone()
        
        # Count orphaned documents (not in any notebook)
        orphaned_query = text("""
            SELECT COUNT(*)
            FROM knowledge_graph_documents kgd
            LEFT JOIN notebook_documents nd ON kgd.document_id = nd.document_id
            WHERE nd.document_id IS NULL
        """)
        orphaned_count = db.execute(orphaned_query).scalar() or 0
        
        response_data = {
            "documents": system_documents,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_count": total_count,
                "total_pages": (total_count + page_size - 1) // page_size,
                "has_next": page * page_size < total_count,
                "has_prev": page > 1
            },
            "summary_stats": {
                "total_documents": stats_result.total_documents or 0,
                "unique_file_types": stats_result.unique_file_types or 0,
                "unique_collections": stats_result.unique_collections or 0,
                "total_size_bytes": stats_result.total_size_bytes or 0,
                "completed_documents": stats_result.completed_documents or 0,
                "failed_documents": stats_result.failed_documents or 0,
                "orphaned_documents": orphaned_count
            },
            "filters_applied": {
                "search": search,
                "file_type": file_type,
                "status": status,
                "collection": collection,
                "sort_by": sort_by,
                "sort_order": sort_order
            }
        }
        
        logger.info(f"System documents query completed: {len(system_documents)} documents returned")
        return response_data
        
    except Exception as e:
        logger.error(f"Error getting system documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{notebook_id}/chat-debug")
async def notebook_chat_debug(
    request: Request,
    notebook_id: str = Path(..., description="Notebook ID"),
    db: Session = Depends(get_db)
):
    """Debug endpoint to see raw request body"""
    try:
        body = await request.body()
        body_str = body.decode('utf-8')
        logger.info(f"🔍 Raw request body: {body_str}")
        
        import json
        json_data = json.loads(body_str)
        logger.info(f"📋 Parsed JSON: {json_data}")
        
        # Try to validate manually
        chat_request = NotebookChatRequest(**json_data)
        logger.info(f"✅ Validation successful: {chat_request}")
        
        return {
            "status": "validation_success", 
            "parsed": json_data,
            "validated": {
                "message": chat_request.message,
                "conversation_id": chat_request.conversation_id,
                "include_context": chat_request.include_context,
                "max_sources": chat_request.max_sources
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "validation_failed", "error": str(e), "traceback": traceback.format_exc()}

@router.post("/test-chat-validation")
async def test_chat_validation(request: NotebookChatRequest):
    """Simple test endpoint to validate NotebookChatRequest"""
    return {
        "status": "validation_success",
        "message": request.message,
        "conversation_id": request.conversation_id,
        "include_context": request.include_context,
        "max_sources": request.max_sources
    }


@router.post("/{notebook_id}/chat")
async def notebook_chat(
    request: NotebookChatRequest,
    notebook_id: str = Path(..., description="Notebook ID"),
    db: Session = Depends(get_db)
):
    """
    Chat with notebook using RAG-powered AI assistant.
    
    Provides streaming responses based on notebook documents and context.
    Uses NotebookRAGService to query relevant content and generates
    conversational responses using the configured LLM.
    
    Args:
        notebook_id: Notebook ID to chat with
        request: Chat request parameters
        db: Database session
        
    Returns:
        StreamingResponse: Real-time chat response
        
    Raises:
        HTTPException: If notebook not found or chat fails
    """
    # Intelligently analyze query intent and optimize max_sources
    original_max_sources = request.max_sources
    optimized_max_sources = await _optimize_max_sources_for_intent(request.message, request.max_sources, notebook_id, notebook_rag_service)
    
    # Update the request if optimization was applied
    if optimized_max_sources != original_max_sources:
        request.max_sources = optimized_max_sources
        logger.info(f"[INTENT_OPTIMIZATION] Intelligent analysis adjusted max_sources from {original_max_sources} to {optimized_max_sources}")
    
    logger.info(f"📨 Notebook chat request received for {notebook_id}")
    logger.info(f"🔍 Request details: message='{request.message}', conversation_id='{request.conversation_id}', include_context={request.include_context}, max_sources={request.max_sources}")
    
    async def stream_chat_response():
        try:
            logger.info(f"Starting notebook chat for {notebook_id}: {request.message[:50]}...")
            logger.info(f"Chat request details: message='{request.message}', conversation_id='{request.conversation_id}', include_context={request.include_context}, max_sources={request.max_sources}")
            
            if not request or not request.message:
                yield json.dumps({
                    "error": "Chat request with message is required",
                    "notebook_id": notebook_id
                }) + "\n"
                return
            
            # Verify notebook exists
            notebook_exists = await notebook_service.notebook_exists(db=db, notebook_id=notebook_id)
            if not notebook_exists:
                yield json.dumps({
                    "error": "Notebook not found",
                    "notebook_id": notebook_id
                }) + "\n"
                return
            
            # Start by yielding initial status
            yield json.dumps({
                "status": "searching",
                "message": "Searching notebook documents...",
                "notebook_id": notebook_id,
                "conversation_id": request.conversation_id
            }) + "\n"
            
            # Query relevant documents using RAG service
            rag_response = None
            # DEBUG: Context inclusion check
            if request.include_context:
                try:
                    # Check if progressive loading would be beneficial for large datasets
                    intent_analysis = await notebook_rag_service._analyze_query_intent(request.message)
                    total_available = await notebook_rag_service.get_notebook_total_items(notebook_id)
                    
                    # === INTELLIGENT AI PIPELINE: Understand → Plan → Execute → Verify → Respond ===
                    
                    # Step 1: Understand - Detect when to use intelligent planning
                    should_use_intelligent_pipeline = _should_use_intelligent_planning(request.message, intent_analysis, total_available)
                    execution_plan = None
                    verification_result = None
                    
                    if should_use_intelligent_pipeline:
                        logger.info(f"[AI_PIPELINE] Activating intelligent pipeline for complex query")
                        
                        yield json.dumps({
                            "status": "ai_planning",
                            "message": "Creating intelligent execution plan...",
                            "notebook_id": notebook_id,
                            "conversation_id": request.conversation_id
                        }) + "\n"
                        
                        # Step 2: Plan - Create AI task plan
                        try:
                            execution_plan = await ai_task_planner.understand_and_plan(request.message)
                            logger.info(f"[AI_PIPELINE] Task plan created - Strategies: {len(execution_plan.retrieval_strategies)}, "
                                       f"Expected: {execution_plan.data_requirements.expected_count}")
                            logger.info(f"[AI_PIPELINE] Plan details: {execution_plan.data_requirements.entities}, "
                                       f"Format: {execution_plan.presentation.format}")
                            
                            yield json.dumps({
                                "status": "ai_executing",
                                "message": f"Executing intelligent plan ({len(execution_plan.retrieval_strategies)} strategies)...",
                                "notebook_id": notebook_id,
                                "conversation_id": request.conversation_id,
                                "plan_metadata": {
                                    "entities": execution_plan.data_requirements.entities,
                                    "completeness": execution_plan.data_requirements.completeness,
                                    "format": execution_plan.presentation.format
                                }
                            }) + "\n"
                            
                        except Exception as e:
                            logger.warning(f"[AI_PIPELINE] Task planning failed, falling back to regular RAG: {str(e)}")
                            execution_plan = None
                    
                    should_use_progressive = await notebook_rag_service.should_use_progressive_loading(
                        query=request.message,
                        intent_analysis=intent_analysis,
                        total_available=total_available,
                        max_sources=request.max_sources
                    )
                    
                    if should_use_progressive:
                        logger.info(f"[REGULAR_CHAT_PROGRESSIVE] Redirecting to progressive loading due to large dataset: "
                                   f"available={total_available}, requested={request.max_sources}")
                        
                        # Stream a message about switching to progressive mode
                        yield json.dumps({
                            "status": "switching_to_progressive",
                            "message": f"Large dataset detected ({total_available} items). Switching to progressive loading for better performance...",
                            "notebook_id": notebook_id,
                            "conversation_id": request.conversation_id,
                            "progressive_loading_recommended": True
                        }) + "\n"
                        
                        # Use progressive streaming instead
                        async for progress_chunk in stream_progressive_notebook_chat(
                            notebook_id=notebook_id,
                            query=request.message,
                            intent_analysis=intent_analysis,
                            rag_service=notebook_rag_service
                        ):
                            yield progress_chunk
                        
                        # End the regular chat stream here since progressive handling is complete
                        return
                    
                    # Step 3: Execute - Use intelligent plan or fall back to standard RAG
                    if execution_plan:
                        logger.info(f"[AI_PIPELINE] Executing intelligent plan with {len(execution_plan.retrieval_strategies)} strategies")
                        rag_response = await notebook_rag_service.execute_intelligent_plan(
                            db=db,
                            notebook_id=notebook_id,
                            plan=execution_plan,
                            include_metadata=True
                        )
                        
                        # Step 4: Verify - Check completeness and trigger self-correction if needed
                        yield json.dumps({
                            "status": "ai_verifying",
                            "message": "Verifying result completeness...",
                            "notebook_id": notebook_id,
                            "conversation_id": request.conversation_id
                        }) + "\n"
                        
                        verification_result = await ai_verification_service.verify_completeness(
                            results=rag_response,
                            plan=execution_plan,
                            notebook_id=notebook_id
                        )
                        
                        logger.info(f"[AI_PIPELINE] Verification - Confidence: {verification_result.confidence:.2f}, "
                                   f"Completeness: {verification_result.completeness_score:.2f}, "
                                   f"Needs correction: {verification_result.needs_correction}")
                        
                        # Self-correction if verification indicates incompleteness
                        if verification_result.needs_correction and verification_result.correction_strategies:
                            logger.info(f"[AI_PIPELINE] Triggering self-correction with {len(verification_result.correction_strategies)} strategies")
                            
                            yield json.dumps({
                                "status": "ai_correcting",
                                "message": f"Self-correcting to improve completeness (confidence: {verification_result.confidence:.1%})...",
                                "notebook_id": notebook_id,
                                "conversation_id": request.conversation_id,
                                "verification_metadata": {
                                    "original_confidence": verification_result.confidence,
                                    "correction_strategies_count": len(verification_result.correction_strategies)
                                }
                            }) + "\n"
                            
                            # Execute correction strategies
                            correction_response = await notebook_rag_service.execute_correction_strategies(
                                db=db,
                                notebook_id=notebook_id,
                                original_response=rag_response,
                                correction_strategies=verification_result.correction_strategies
                            )
                            
                            if correction_response and len(correction_response.sources) > len(rag_response.sources):
                                logger.info(f"[AI_PIPELINE] Self-correction successful: {len(rag_response.sources)} → {len(correction_response.sources)} sources")
                                rag_response = correction_response
                        
                    else:
                        # Continue with regular RAG processing for smaller datasets
                        logger.info(f"[REGULAR_CHAT] Using standard retrieval: available={total_available}, requested={request.max_sources}")
                        
                        # Use standard retrieval path (comprehensive mode removed)
                        # Check if hierarchical RAG would be beneficial
                        hierarchical_rag_service = get_hierarchical_notebook_rag_service()
                        context_stats = await hierarchical_rag_service.get_context_stats(notebook_id, request.message)
                        
                        # Use hierarchical service if optimization is needed or for complex queries
                        if context_stats.get("optimization_needed", False) or request.max_sources > 15:
                            logger.info(f"Using hierarchical RAG for context optimization (max_sources={request.max_sources})")
                            rag_response = await hierarchical_rag_service.query_with_context_optimization(
                                notebook_id=notebook_id,
                                query=request.message,
                                top_k=request.max_sources,
                                include_metadata=True,
                                db=db
                            )
                            
                            # Log optimization results
                            if rag_response.metadata:
                                token_stats = rag_response.metadata.get('token_budget', {})
                                logger.info(f"Hierarchical RAG used {token_stats.get('retrieval_used', 0)} tokens "
                                          f"({token_stats.get('utilization_percent', 0)}% of budget)")
                        else:
                            # Use adaptive RAG service with intelligent enumeration
                            rag_response = await notebook_rag_service.query_notebook_adaptive(
                                db=db,
                                notebook_id=notebook_id,
                                query=request.message,
                                max_sources=request.max_sources,
                                include_metadata=True
                            )
                    
                    # RAG Query Execution completed
                    
                    # Build comprehensive status with AI pipeline metadata
                    status_data = {
                        "status": "context_found",
                        "sources_count": len(rag_response.sources),
                        "collections_searched": rag_response.collections_searched,
                        "notebook_id": notebook_id,
                        "conversation_id": request.conversation_id
                    }
                    
                    # Add AI pipeline metadata if available
                    if execution_plan:
                        status_data["ai_pipeline_metadata"] = {
                            "used_intelligent_pipeline": True,
                            "strategies_executed": len(execution_plan.retrieval_strategies),
                            "data_completeness": execution_plan.data_requirements.completeness,
                            "expected_format": execution_plan.presentation.format
                        }
                        
                        if verification_result:
                            status_data["ai_pipeline_metadata"]["verification"] = {
                                "confidence": verification_result.confidence,
                                "completeness_score": verification_result.completeness_score,
                                "used_self_correction": verification_result.needs_correction
                            }
                    
                    yield json.dumps(status_data) + "\n"
                    
                except Exception as e:
                    logger.warning(f"RAG query failed for notebook {notebook_id}: {str(e)}")
                    yield json.dumps({
                        "status": "context_warning",
                        "message": "Could not retrieve full context, proceeding with general response",
                        "notebook_id": notebook_id,
                        "conversation_id": request.conversation_id
                    }) + "\n"
            
            # Get notebook LLM configuration
            llm_config_full = get_notebook_llm_full_config()
            if not llm_config_full:
                yield json.dumps({
                    "error": "Notebook LLM configuration not available",
                    "notebook_id": notebook_id,
                    "conversation_id": request.conversation_id
                }) + "\n"
                return
            
            # Extract the notebook_llm config from the nested structure
            llm_config = llm_config_full.get('notebook_llm', {})
            if not llm_config:
                yield json.dumps({
                    "error": "Notebook LLM configuration not properly structured in database",
                    "notebook_id": notebook_id,
                    "conversation_id": request.conversation_id
                }) + "\n"
                return
            
            yield json.dumps({
                "status": "generating",
                "message": "Generating response...",
                "notebook_id": notebook_id,
                "conversation_id": request.conversation_id
            }) + "\n"
            
            # Initialize LLM with database configuration - NO HARDCODED FALLBACKS
            llm_config_obj = LLMConfig(
                model_name=llm_config['model'],
                temperature=float(llm_config['temperature']),
                top_p=float(llm_config['top_p']),
                max_tokens=int(llm_config['max_tokens'])
            )
            llm = OllamaLLM(llm_config_obj)
            
            # Build context-aware prompt using configurable system prompt from database
            base_system_prompt = llm_config['system_prompt']
            system_prompt = base_system_prompt + " "
            context_info = ""
            
            if rag_response and rag_response.sources:
                # Count different source types for better prompt customization
                document_count = sum(1 for src in rag_response.sources if src.source_type != "memory")
                memory_count = sum(1 for src in rag_response.sources if src.source_type == "memory")
                
                # Apply configurable source integration templates
                source_integration_prompt = apply_source_templates(
                    total_sources=len(rag_response.sources),
                    document_count=document_count,
                    memory_count=memory_count
                )
                system_prompt += source_integration_prompt
                
                # === INTELLIGENT AI ENHANCEMENTS: Add structured AI planning instructions ===
                
                # Step 5: Enhance LLM with AI planning context
                if execution_plan:
                    # Add AI task plan context to system prompt
                    ai_enhancement_prompt = f"\n\nINTELLIGENT AI CONTEXT:\n"
                    ai_enhancement_prompt += f"This query has been processed through intelligent AI planning with the following understanding:\n"
                    ai_enhancement_prompt += f"- Required entities: {', '.join(execution_plan.data_requirements.entities)}\n"
                    ai_enhancement_prompt += f"- Required attributes: {', '.join(execution_plan.data_requirements.attributes)}\n"
                    ai_enhancement_prompt += f"- Completeness requirement: {execution_plan.data_requirements.completeness}\n"
                    ai_enhancement_prompt += f"- Expected format: {execution_plan.presentation.format}\n"
                    
                    if execution_plan.data_requirements.expected_count:
                        ai_enhancement_prompt += f"- Expected count: {execution_plan.data_requirements.expected_count}\n"
                    
                    # Add format-specific instructions
                    if execution_plan.presentation.format == "table":
                        ai_enhancement_prompt += f"\nFORMAT REQUIREMENT: Present results in a well-formatted table with columns: {', '.join(execution_plan.presentation.fields_to_show)}\n"
                        
                        if execution_plan.presentation.sorting:
                            sort_field = execution_plan.presentation.sorting.get('field', 'default')
                            sort_order = execution_plan.presentation.sorting.get('order', 'asc')
                            ai_enhancement_prompt += f"Sort by {sort_field} in {sort_order} order.\n"
                            
                    elif execution_plan.presentation.format == "list":
                        ai_enhancement_prompt += f"\nFORMAT REQUIREMENT: Present results as a structured list with clear organization.\n"
                    
                    # Add verification context if available
                    if verification_result:
                        ai_enhancement_prompt += f"\nQUALITY ASSURANCE:\n"
                        ai_enhancement_prompt += f"- Confidence level: {verification_result.confidence:.1%}\n"
                        ai_enhancement_prompt += f"- Completeness score: {verification_result.completeness_score:.1%}\n"
                        
                        if verification_result.needs_correction:
                            ai_enhancement_prompt += f"- Self-correction was applied to improve completeness\n"
                        
                        if verification_result.quality_issues:
                            ai_enhancement_prompt += f"- Note potential gaps: {'; '.join(verification_result.quality_issues[:2])}\n"
                    
                    ai_enhancement_prompt += f"\nPlease provide a response that meets these intelligent requirements and leverages the comprehensive analysis performed.\n"
                    
                    system_prompt += ai_enhancement_prompt
                    logger.info(f"[AI_PIPELINE] Enhanced LLM prompt with intelligent planning context ({len(ai_enhancement_prompt)} chars)")
                
                # === COMPREHENSIVE ENUMERATION ENHANCEMENT ===
                
                # Detect comprehensive enumeration queries that require complete results
                query_lower = request.message.lower()
                is_comprehensive_enumeration = (
                    ("list" in query_lower and "all" in query_lower) or
                    ("show" in query_lower and "all" in query_lower) or
                    ("enumerate" in query_lower) or
                    (("table" in query_lower or "format" in query_lower) and 
                     ("projects" in query_lower or "project" in query_lower or "work" in query_lower))
                )
                
                if is_comprehensive_enumeration:
                    source_count = len(rag_response.sources) if rag_response else 0
                    system_prompt += f"""

ENUMERATION TASK:
Process all {source_count} sources and list every relevant item mentioned.
Present the findings in the requested format (table, list, etc.).
"""
                    
                    logger.info(f"[COMPLETENESS] Enhanced prompt for comprehensive enumeration - {source_count} sources")
                
                context_info = "\n\n--- RELEVANT CONTEXT FROM NOTEBOOK ---\n"
                context_info += "The following sources contain information relevant to your question. "
                context_info += "Please integrate information from ALL sources in your response:\n"
                
                for i, source in enumerate(rag_response.sources, 1):
                    # Format source name with type prefix for clarity
                    source_name = source.document_name or 'Unknown'
                    source_type_prefix = "Personal Memory" if source.source_type == "memory" else "Document"
                    
                    # DEBUG: Context Building Process
                    
                    context_info += f"\n[Source {i} - {source_type_prefix}: {source_name}]:\n"
                    context_info += source.content[:1000] + ("..." if len(source.content) > 1000 else "")
                    context_info += "\n"
                
                context_info += "--- END CONTEXT ---\n\n"
                
                # Enhanced prompt with structured project data for enumeration queries
                if hasattr(rag_response, 'extracted_projects') and rag_response.extracted_projects:
                    # Detect if this is an enumeration/listing query that could benefit from structured data
                    query_lower = request.message.lower()
                    enumeration_indicators = ['list all', 'show all', 'enumerate', 'table format', 'in table', 'as table', 'table form']
                    project_indicators = ['project', 'projects', 'work', 'experience']
                    
                    is_enumeration_query = (
                        any(indicator in query_lower for indicator in enumeration_indicators) and 
                        any(indicator in query_lower for indicator in project_indicators)
                    )
                    
                    if is_enumeration_query:
                        project_count = len(rag_response.extracted_projects)
                        logger.info(f"[STRUCTURED_RESPONSE] Injecting {project_count} structured projects into prompt for enumeration query")
                        
                        # Add structured project data to context
                        context_info += "--- STRUCTURED PROJECT DATA ---\n"
                        context_info += f"IMPORTANT: Here are ALL {project_count} projects extracted from the notebook content. "
                        context_info += "You MUST include every single project listed below, even if some fields show 'N/A':\n\n"
                        
                        for i, project in enumerate(rag_response.extracted_projects, 1):
                            context_info += f"Project {i}:\n"
                            context_info += f"  Name: {project.name}\n"
                            context_info += f"  Company: {project.company}\n"
                            context_info += f"  Year: {project.year}\n"
                            context_info += f"  Description: {project.description[:200]}{'...' if len(project.description) > 200 else ''}\n\n"
                        
                        context_info += "--- END STRUCTURED DATA ---\n\n"
                        
                        # Add explicit formatting instructions to system prompt
                        if any(table_word in query_lower for table_word in ['table', 'format']):
                            system_prompt += f" CRITICAL INSTRUCTION: The user has requested table format. You must create a table that includes ALL {project_count} projects listed above. Do not exclude any projects due to missing information - use 'N/A' for missing fields. Ensure your table shows exactly {project_count} rows of projects."
                        else:
                            system_prompt += f" CRITICAL INSTRUCTION: You must mention ALL {project_count} projects listed above. Do not exclude any projects due to missing information. Include every project even if some details are marked as 'N/A'."
                
                # DEBUG: Final Context Analysis
                
                # DEBUG: Check if memory content is in final context
            else:
                system_prompt += "No specific context was found in this notebook for this question. Provide a helpful general response and suggest ways the user might find relevant information."
            
            # Build the full prompt
            full_prompt = f"{system_prompt}\n\n{context_info}User Question: {request.message}\n\nResponse:"
            
            # DEBUG: LLM Prompt Construction
            logger.debug(f"[PROMPT_DEBUG] Full prompt length: {len(full_prompt)} chars")
            logger.debug(f"[PROMPT_DEBUG] Prompt preview: {full_prompt[:300]}...")
            
            # DEBUG: Verify memory context reaches LLM
            if "Memory" in full_prompt:
                logger.debug(f"[PROMPT_DEBUG] Memory content confirmed in LLM prompt")
            else:
                logger.debug(f"[PROMPT_DEBUG] WARNING: No memory content found in LLM prompt")
            
            # Generate streaming response
            collected_response = ""
            async for response_chunk in llm.generate_stream(full_prompt):
                if response_chunk.text.strip():
                    collected_response += response_chunk.text
                    
                    # Stream the response chunk
                    yield json.dumps({
                        "chunk": response_chunk.text,
                        "notebook_id": notebook_id,
                        "conversation_id": request.conversation_id
                    }) + "\n"
            
            # Send final response with complete answer and sources
            final_response = {
                "answer": collected_response,
                "sources": [
                    {
                        "document_id": source.document_id,
                        "document_name": source.document_name,
                        "content": source.content[:500] + ("..." if len(source.content) > 500 else ""),
                        "score": source.score,
                        "collection": source.collection
                    }
                    for source in (rag_response.sources if rag_response else [])
                ],
                "notebook_id": notebook_id,
                "conversation_id": request.conversation_id,
                "status": "complete"
            }
            
            # === INTELLIGENT AI PIPELINE: Include comprehensive metadata in response ===
            
            # Add AI pipeline metadata to final response for transparency and debugging
            if execution_plan:
                final_response["ai_pipeline_metadata"] = {
                    "used_intelligent_pipeline": True,
                    "pipeline_version": "1.0",
                    "task_plan": {
                        "entities": execution_plan.data_requirements.entities,
                        "attributes": execution_plan.data_requirements.attributes,
                        "completeness_requirement": execution_plan.data_requirements.completeness,
                        "expected_count": execution_plan.data_requirements.expected_count,
                        "format": execution_plan.presentation.format,
                        "fields_shown": execution_plan.presentation.fields_to_show,
                        "sorting": execution_plan.presentation.sorting,
                        "strategies_executed": len(execution_plan.retrieval_strategies)
                    }
                }
                
                # Add verification results if available
                if verification_result:
                    final_response["ai_pipeline_metadata"]["verification"] = {
                        "confidence": verification_result.confidence,
                        "completeness_score": verification_result.completeness_score,
                        "needs_correction": verification_result.needs_correction,
                        "result_count": verification_result.result_count,
                        "unique_sources": verification_result.unique_sources,
                        "diversity_score": verification_result.diversity_score,
                        "expected_vs_actual": verification_result.expected_vs_actual,
                        "used_self_correction": verification_result.needs_correction,
                        "quality_issues": verification_result.quality_issues[:3] if verification_result.quality_issues else [],
                        "reasoning": verification_result.reasoning[:200] + "..." if len(verification_result.reasoning) > 200 else verification_result.reasoning
                    }
                
                logger.info(f"[AI_PIPELINE] Complete intelligent pipeline executed successfully - "
                           f"Plan: {len(execution_plan.retrieval_strategies)} strategies, "
                           f"Verification: {verification_result.confidence:.1%} confidence" if verification_result else "No verification")
            else:
                # Mark as using traditional RAG approach
                final_response["ai_pipeline_metadata"] = {
                    "used_intelligent_pipeline": False,
                    "pipeline_version": "traditional_rag",
                    "fallback_reason": "Query did not meet intelligent pipeline criteria"
                }
            
            # Include structured project data in response if available
            if rag_response and hasattr(rag_response, 'extracted_projects') and rag_response.extracted_projects:
                final_response["extracted_projects"] = [
                    {
                        "name": project.name,
                        "company": project.company,
                        "year": project.year,
                        "description": project.description,
                        "confidence_score": project.confidence_score
                    }
                    for project in rag_response.extracted_projects
                ]
                final_response["extracted_projects_count"] = len(rag_response.extracted_projects)
            
            # DEBUG: Final Response Analysis
            logger.debug(f"[RESPONSE_DEBUG] Generated response length: {len(collected_response)} chars")
            logger.debug(f"[RESPONSE_DEBUG] Response preview: {collected_response[:200]}...")
            logger.debug(f"[RESPONSE_DEBUG] Final response includes {len(final_response['sources'])} sources")
            
            # DEBUG: Check if response mentions employment/memory content
            if any(keyword in collected_response.lower() for keyword in ['employment', 'job', 'work', 'career', 'company']):
                logger.debug(f"[RESPONSE_DEBUG] Response contains employment-related content")
            else:
                logger.debug(f"[RESPONSE_DEBUG] Response does NOT contain employment-related content")
            
            # DEBUG: Validate structured project data usage
            if rag_response and hasattr(rag_response, 'extracted_projects') and rag_response.extracted_projects:
                extracted_count = len(rag_response.extracted_projects)
                query_lower = request.message.lower()
                
                # Count project mentions in response for validation
                project_mentions = collected_response.lower().count('project')
                table_format_requested = any(table_word in query_lower for table_word in ['table', 'format'])
                
                logger.info(f"[PROJECT_VALIDATION] Query: '{request.message[:50]}...'")
                logger.info(f"[PROJECT_VALIDATION] Extracted projects: {extracted_count}")
                logger.info(f"[PROJECT_VALIDATION] Project mentions in response: {project_mentions}")
                logger.info(f"[PROJECT_VALIDATION] Table format requested: {table_format_requested}")
                
                # Warn if significantly fewer projects mentioned than extracted
                if table_format_requested and project_mentions < extracted_count * 0.8:
                    logger.warning(f"[PROJECT_VALIDATION] Potential project loss: {extracted_count} extracted but only {project_mentions} mentions in response")
                    
            else:
                logger.debug(f"[PROJECT_VALIDATION] No extracted projects available for validation")
            
            yield json.dumps(final_response) + "\n"
            
            logger.info(f"Successfully completed notebook chat for {notebook_id}")
            
        except Exception as e:
            logger.error(f"Notebook chat error for {notebook_id}: {str(e)}")
            yield json.dumps({
                "error": f"Chat error: {str(e)}",
                "notebook_id": notebook_id,
                "conversation_id": request.conversation_id if request else None,
                "status": "error"
            }) + "\n"
    
    return StreamingResponse(stream_chat_response(), media_type="application/json")


async def stream_progressive_notebook_chat(
    notebook_id: str,
    query: str,
    intent_analysis: dict,
    rag_service: NotebookRAGService
):
    """
    Handle progressive streaming for large result sets.
    
    Streams initial results quickly, then continues with background loading
    and provides progress updates to frontend.
    
    Args:
        notebook_id: Notebook ID
        query: User query string
        intent_analysis: Query intent analysis results
        rag_service: RAG service instance
        
    Yields:
        JSON chunks with progressive results and progress updates
    """
    try:
        logger.info(f"[PROGRESSIVE_CHAT] Starting progressive streaming for notebook {notebook_id}")
        logger.info(f"[PROGRESSIVE_CHAT] Query: '{query[:100]}...', comprehensive: {intent_analysis.get('wants_comprehensive', False)}")
        
        # Get total available content for progress tracking
        total_available = await rag_service.get_notebook_total_items(notebook_id)
        max_sources = intent_analysis.get("estimated_sources_needed", 500)  # From intent analysis
        
        # Determine if progressive loading is beneficial
        should_use_progressive = await rag_service.should_use_progressive_loading(
            query=query,
            intent_analysis=intent_analysis,
            total_available=total_available,
            max_sources=max_sources
        )
        
        if not should_use_progressive:
            logger.info(f"[PROGRESSIVE_CHAT] Progressive loading not beneficial, using standard retrieval")
            # Fall back to standard adaptive retrieval
            from app.core.db import SessionLocal
            db = SessionLocal()
            try:
                # Use standard adaptive retrieval
                response = await rag_service.query_notebook_adaptive(
                    db=db,
                    notebook_id=notebook_id,
                    query=query,
                    max_sources=max_sources,
                    include_metadata=True
                )
                
                # Send as single complete response
                yield json.dumps({
                    "stage": "complete",
                    "sources": [
                        {
                            "document_id": source.document_id,
                            "document_name": source.document_name,
                            "content": source.content[:1000] + ("..." if len(source.content) > 1000 else ""),
                            "score": source.score,
                            "collection": source.collection,
                            "source_type": source.source_type
                        }
                        for source in response.sources
                    ],
                    "total_sources": response.total_sources,
                    "progress_percent": 100.0,
                    "more_available": False,
                    "progressive_loading_used": False,
                    "notebook_id": notebook_id
                }) + "\n"
            finally:
                db.close()
            return
        
        # Use multi-stage retrieval for large datasets
        stage_count = 0
        total_retrieved = 0
        all_sources = []
        
        logger.info(f"[PROGRESSIVE_CHAT] Using progressive loading: max_sources={max_sources}, available={total_available}")
        
        # Stream each stage of retrieval
        async for stage_response in rag_service.multi_stage_retrieval(
            notebook_id=notebook_id,
            query=query,
            intent_analysis=intent_analysis,
            total_available=total_available,
            max_sources=max_sources
        ):
            stage_count += 1
            stage_sources = stage_response.sources
            all_sources.extend(stage_sources)
            total_retrieved += len(stage_sources)
            
            # Get stage metadata
            stage_metadata = stage_response.metadata.get("multi_stage", {})
            is_initial_stage = stage_metadata.get("stage") == "initial"
            progress_percent = stage_metadata.get("progress_percent", 0)
            more_available = stage_metadata.get("more_available", False)
            
            logger.info(f"[PROGRESSIVE_CHAT] Stage {stage_count}: {len(stage_sources)} sources, "
                       f"total: {total_retrieved}, progress: {progress_percent}%")
            
            # Stream stage results
            stage_data = {
                "stage": "initial" if is_initial_stage else "progressive",
                "stage_number": stage_count,
                "sources": [
                    {
                        "document_id": source.document_id,
                        "document_name": source.document_name,
                        "content": source.content[:1000] + ("..." if len(source.content) > 1000 else ""),
                        "score": source.score,
                        "collection": source.collection,
                        "source_type": source.source_type
                    }
                    for source in stage_sources
                ],
                "batch_size": len(stage_sources),
                "total_retrieved_so_far": total_retrieved,
                "progress_percent": progress_percent,
                "more_available": more_available,
                "progressive_loading_used": True,
                "notebook_id": notebook_id
            }
            
            # Add timing information for initial stage
            if is_initial_stage:
                stage_data["initial_response_time"] = "2-3 seconds"
                stage_data["message"] = "Quick initial results - more loading in background"
            
            yield json.dumps(stage_data) + "\n"
            
            # Add small delay between stages for better UX
            if not is_initial_stage:
                import asyncio
                await asyncio.sleep(0.1)
        
        # Send completion signal
        yield json.dumps({
            "stage": "complete",
            "total_stages": stage_count,
            "final_source_count": total_retrieved,
            "progress_percent": 100.0,
            "more_available": False,
            "progressive_loading_used": True,
            "notebook_id": notebook_id,
            "message": f"Progressive loading complete: retrieved {total_retrieved} sources in {stage_count} stages"
        }) + "\n"
        
        logger.info(f"[PROGRESSIVE_CHAT] Completed progressive streaming: {total_retrieved} sources in {stage_count} stages")
        
    except Exception as e:
        logger.error(f"[PROGRESSIVE_CHAT] Error in progressive streaming: {str(e)}")
        yield json.dumps({
            "stage": "error",
            "error": str(e),
            "progressive_loading_used": True,
            "notebook_id": notebook_id
        }) + "\n"


@router.post("/{notebook_id}/chat-progressive")
async def notebook_chat_progressive(
    request: NotebookChatRequest,
    notebook_id: str = Path(..., description="Notebook ID"),
    db: Session = Depends(get_db)
):
    """
    Progressive notebook chat with multi-stage retrieval for large datasets.
    
    This endpoint provides:
    - Quick initial results (50-100 items) within 2-3 seconds
    - Progressive loading of additional results in background
    - Real-time progress updates
    - Optimized performance for very large notebooks (500+ items)
    
    Args:
        request: Chat request with message and configuration
        notebook_id: Target notebook ID
        db: Database session
        
    Returns:
        Streaming response with progressive results
    """
    # Initialize services
    notebook_service = NotebookService(db)
    notebook_rag_service = NotebookRAGService()
    
    # Apply intelligent intent-based optimization
    original_max_sources = request.max_sources
    optimized_max_sources = await _optimize_max_sources_for_intent(
        request.message, request.max_sources, notebook_id, notebook_rag_service
    )
    
    if optimized_max_sources != original_max_sources:
        request.max_sources = optimized_max_sources
        logger.info(f"[PROGRESSIVE_INTENT] Optimized max_sources from {original_max_sources} to {optimized_max_sources}")
    
    async def progressive_chat_stream():
        try:
            logger.info(f"[PROGRESSIVE_ENDPOINT] Starting progressive chat for {notebook_id}: {request.message[:50]}...")
            
            if not request or not request.message:
                yield json.dumps({
                    "error": "Chat request with message is required",
                    "notebook_id": notebook_id
                }) + "\n"
                return
            
            # Verify notebook exists
            notebook_exists = await notebook_service.notebook_exists(db=db, notebook_id=notebook_id)
            if not notebook_exists:
                yield json.dumps({
                    "error": "Notebook not found",
                    "notebook_id": notebook_id
                }) + "\n"
                return
            
            # Analyze query intent for progressive loading decision
            intent_analysis = await notebook_rag_service._analyze_query_intent(request.message)
            intent_analysis["estimated_sources_needed"] = request.max_sources
            
            logger.info(f"[PROGRESSIVE_ENDPOINT] Intent analysis: comprehensive={intent_analysis.get('wants_comprehensive', False)}, "
                       f"quantity={intent_analysis.get('quantity_intent', 'limited')}")
            
            # Send initial status
            yield json.dumps({
                "status": "analyzing",
                "message": "Analyzing query and determining retrieval strategy...",
                "notebook_id": notebook_id,
                "conversation_id": request.conversation_id,
                "intent_analysis": {
                    "wants_comprehensive": intent_analysis.get("wants_comprehensive", False),
                    "quantity_intent": intent_analysis.get("quantity_intent", "limited"),
                    "confidence": intent_analysis.get("confidence", 0.5)
                }
            }) + "\n"
            
            if request.include_context:
                # Stream progressive retrieval results
                yield json.dumps({
                    "status": "retrieving",
                    "message": "Starting progressive document retrieval...",
                    "notebook_id": notebook_id,
                    "conversation_id": request.conversation_id
                }) + "\n"
                
                # Use progressive streaming function
                async for progress_chunk in stream_progressive_notebook_chat(
                    notebook_id=notebook_id,
                    query=request.message,
                    intent_analysis=intent_analysis,
                    rag_service=notebook_rag_service
                ):
                    yield progress_chunk
            
            # After retrieval is complete, send final status
            yield json.dumps({
                "status": "complete",
                "message": "Progressive retrieval complete",
                "notebook_id": notebook_id,
                "conversation_id": request.conversation_id
            }) + "\n"
            
            logger.info(f"[PROGRESSIVE_ENDPOINT] Successfully completed progressive chat for {notebook_id}")
            
        except Exception as e:
            logger.error(f"[PROGRESSIVE_ENDPOINT] Progressive chat error for {notebook_id}: {str(e)}")
            yield json.dumps({
                "error": f"Progressive chat error: {str(e)}",
                "notebook_id": notebook_id,
                "conversation_id": request.conversation_id if request else None,
                "status": "error"
            }) + "\n"
    
    return StreamingResponse(progressive_chat_stream(), media_type="application/json")


# Memory Management Endpoints

@router.post("/{notebook_id}/memories", response_model=MemoryResponse, status_code=201)
async def create_memory(
    notebook_id: str = Path(..., description="Notebook ID"),
    request: MemoryCreateRequest = ...,
    db: Session = Depends(get_db)
):
    """
    Create a new memory for a notebook.
    
    Args:
        notebook_id: Target notebook ID
        request: Memory creation parameters
        db: Database session
        
    Returns:
        Created memory details
        
    Raises:
        HTTPException: If creation fails
    """
    try:
        logger.info(f"Creating memory '{request.name}' for notebook {notebook_id}")
        
        memory = await notebook_service.create_memory(
            db=db,
            notebook_id=notebook_id,
            name=request.name,
            content=request.content,
            description=request.description,
            metadata=request.metadata
        )
        
        logger.info(f"Successfully created memory {memory.memory_id}")
        return memory
        
    except ValueError as e:
        logger.error(f"Validation error creating memory: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating memory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create memory: {str(e)}")

@router.get("/{notebook_id}/memories", response_model=MemoryListResponse)
async def get_memories(
    notebook_id: str = Path(..., description="Notebook ID"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    db: Session = Depends(get_db)
):
    """
    Get memories for a notebook with pagination.
    
    Args:
        notebook_id: Target notebook ID
        page: Page number (1-based)
        page_size: Number of memories per page
        db: Database session
        
    Returns:
        List of memories with pagination info
        
    Raises:
        HTTPException: If retrieval fails
    """
    try:
        logger.info(f"Getting memories for notebook {notebook_id} (page {page})")
        
        memories = await notebook_service.get_memories(
            db=db,
            notebook_id=notebook_id,
            page=page,
            page_size=page_size
        )
        
        return memories
        
    except ValueError as e:
        logger.error(f"Validation error getting memories: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting memories: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get memories: {str(e)}")

@router.get("/{notebook_id}/memories/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    notebook_id: str = Path(..., description="Notebook ID"),
    memory_id: str = Path(..., description="Memory ID"),
    db: Session = Depends(get_db)
):
    """
    Get a specific memory by ID.
    
    Args:
        notebook_id: Target notebook ID
        memory_id: Memory ID
        db: Database session
        
    Returns:
        Memory details
        
    Raises:
        HTTPException: If memory not found
    """
    try:
        logger.info(f"Getting memory {memory_id} for notebook {notebook_id}")
        
        memory = await notebook_service.get_memory(
            db=db,
            notebook_id=notebook_id,
            memory_id=memory_id
        )
        
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")
        
        return memory
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting memory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get memory: {str(e)}")

@router.put("/{notebook_id}/memories/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    notebook_id: str = Path(..., description="Notebook ID"),
    memory_id: str = Path(..., description="Memory ID"),
    request: MemoryUpdateRequest = ...,
    db: Session = Depends(get_db)
):
    """
    Update a memory.
    
    Args:
        notebook_id: Target notebook ID
        memory_id: Memory ID
        request: Memory update parameters
        db: Database session
        
    Returns:
        Updated memory details
        
    Raises:
        HTTPException: If update fails
    """
    try:
        logger.info(f"Updating memory {memory_id} for notebook {notebook_id}")
        
        memory = await notebook_service.update_memory(
            db=db,
            notebook_id=notebook_id,
            memory_id=memory_id,
            name=request.name,
            description=request.description,
            content=request.content,
            metadata=request.metadata
        )
        
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")
        
        logger.info(f"Successfully updated memory {memory_id}")
        return memory
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error updating memory: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating memory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update memory: {str(e)}")

@router.delete("/{notebook_id}/memories/{memory_id}")
async def delete_memory(
    notebook_id: str = Path(..., description="Notebook ID"),
    memory_id: str = Path(..., description="Memory ID"),
    db: Session = Depends(get_db)
):
    """
    Delete a memory and all its associated data.
    
    Args:
        notebook_id: Target notebook ID
        memory_id: Memory ID
        db: Database session
        
    Returns:
        Success confirmation
        
    Raises:
        HTTPException: If deletion fails
    """
    try:
        logger.info(f"Deleting memory {memory_id} for notebook {notebook_id}")
        
        deleted = await notebook_service.delete_memory(
            db=db,
            notebook_id=notebook_id,
            memory_id=memory_id
        )
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Memory not found")
        
        logger.info(f"Successfully deleted memory {memory_id}")
        return {"message": "Memory deleted successfully", "deleted": True}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting memory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete memory: {str(e)}")

# Chunk Management Endpoints

@router.get("/chunks/{collection_name}/{document_id}", response_model=ChunkListResponse)
async def get_chunks_for_document(
    collection_name: str = Path(..., description="Milvus collection name"),
    document_id: str = Path(..., description="Document or memory ID"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    db: Session = Depends(get_db)
):
    """
    Get all chunks for a specific document or memory.
    
    Args:
        collection_name: Milvus collection name
        document_id: Document or memory ID
        page: Page number (1-based)
        page_size: Number of chunks per page
        db: Database session
        
    Returns:
        List of chunks with pagination info
        
    Raises:
        HTTPException: If retrieval fails
    """
    try:
        logger.info(f"Getting chunks for document {document_id} from collection {collection_name}")
        
        chunks = await chunk_management_service.get_chunks_for_document(
            db=db,
            collection_name=collection_name,
            document_id=document_id,
            page=page,
            page_size=page_size
        )
        
        return chunks
        
    except ValueError as e:
        logger.error(f"Validation error getting chunks: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting chunks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get chunks: {str(e)}")

@router.get("/chunks/{collection_name}/chunk/{chunk_id}", response_model=ChunkResponse)
async def get_chunk_by_id(
    collection_name: str = Path(..., description="Milvus collection name"),
    chunk_id: str = Path(..., description="Chunk ID"),
    db: Session = Depends(get_db)
):
    """
    Get a specific chunk by ID.
    
    Args:
        collection_name: Milvus collection name
        chunk_id: Chunk ID to retrieve
        db: Database session
        
    Returns:
        Chunk details with edit history
        
    Raises:
        HTTPException: If chunk not found
    """
    try:
        logger.info(f"Getting chunk {chunk_id} from collection {collection_name}")
        
        chunk = await chunk_management_service.get_chunk_by_id(
            db=db,
            collection_name=collection_name,
            chunk_id=chunk_id
        )
        
        return chunk
        
    except ValueError as e:
        logger.error(f"Validation error getting chunk: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting chunk: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get chunk: {str(e)}")

@router.put("/chunks/{collection_name}/chunk/{chunk_id}", response_model=ChunkOperationResponse)
async def update_chunk(
    collection_name: str = Path(..., description="Milvus collection name"),
    chunk_id: str = Path(..., description="Chunk ID"),
    request: ChunkUpdateRequest = ...,
    user_id: Optional[str] = Query(None, description="User ID making the edit"),
    db: Session = Depends(get_db)
):
    """
    Update a chunk's content and optionally re-embed it.
    
    Args:
        collection_name: Milvus collection name
        chunk_id: Chunk ID to update
        request: Chunk update parameters
        user_id: User making the edit
        db: Database session
        
    Returns:
        Operation result
        
    Raises:
        HTTPException: If update fails
    """
    try:
        logger.info(f"Updating chunk {chunk_id} in collection {collection_name}")
        
        result = await chunk_management_service.update_chunk(
            db=db,
            collection_name=collection_name,
            chunk_id=chunk_id,
            new_content=request.content,
            re_embed=request.re_embed,
            user_id=user_id,
            edit_metadata=request.metadata
        )
        
        return result
        
    except ValueError as e:
        logger.error(f"Validation error updating chunk: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating chunk: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update chunk: {str(e)}")

@router.post("/chunks/{collection_name}/bulk-re-embed", response_model=BulkChunkOperationResponse)
async def bulk_re_embed_chunks(
    collection_name: str = Path(..., description="Milvus collection name"),
    request: BulkChunkReEmbedRequest = ...,
    user_id: Optional[str] = Query(None, description="User ID performing the operation"),
    db: Session = Depends(get_db)
):
    """
    Re-embed multiple chunks in bulk.
    
    Args:
        collection_name: Milvus collection name
        request: Bulk re-embed parameters
        user_id: User performing the operation
        db: Database session
        
    Returns:
        Bulk operation results
        
    Raises:
        HTTPException: If operation fails
    """
    try:
        logger.info(f"Bulk re-embedding {len(request.chunk_ids)} chunks in collection {collection_name}")
        
        result = await chunk_management_service.bulk_re_embed_chunks(
            db=db,
            collection_name=collection_name,
            chunk_ids=request.chunk_ids,
            user_id=user_id
        )
        
        return result
        
    except ValueError as e:
        logger.error(f"Validation error bulk re-embedding: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error bulk re-embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to bulk re-embed: {str(e)}")

# Error handlers are handled at the app level, not router level
# The individual endpoints already have proper try/catch blocks with HTTPException