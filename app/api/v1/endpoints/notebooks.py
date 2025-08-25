"""
FastAPI router for Notebook management endpoints.
Provides CRUD operations and RAG functionality for notebooks.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path, UploadFile, File, Request
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
from datetime import datetime

from app.core.db import get_db, KnowledgeGraphDocument
from app.models.notebook_models import (
    NotebookCreateRequest, NotebookUpdateRequest, NotebookResponse, NotebookDetailResponse,
    NotebookListResponse, NotebookDocumentAddRequest, NotebookDocumentBulkRequest,
    NotebookRAGRequest, NotebookRAGResponse, NotebookConversationRequest,
    NotebookDocumentBulkResponse, NotebookStatsResponse, NotebookOperationResponse,
    NotebookError, NotebookValidationError, NotebookDocumentResponse, 
    NotebookConversationResponse, DocumentDeleteRequest, DocumentDeleteResponse,
    DocumentUsageInfo, DocumentDeletionSummary, NotebookChatRequest, NotebookChatResponse
)
from pydantic import BaseModel
from app.services.notebook_service import NotebookService
from app.services.notebook_rag_service import NotebookRAGService
from app.services.document_admin_service import DocumentAdminService
from app.core.llm_settings_cache import get_llm_settings, get_main_llm_full_config
from app.core.notebook_llm_settings_cache import get_notebook_llm_full_config
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

# Initialize services
notebook_service = NotebookService()
notebook_rag_service = NotebookRAGService()

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
            if request.include_context:
                try:
                    rag_response = await notebook_rag_service.query_notebook(
                        db=db,
                        notebook_id=notebook_id,
                        query=request.message,
                        top_k=request.max_sources,
                        include_metadata=True
                    )
                    
                    yield json.dumps({
                        "status": "context_found",
                        "sources_count": len(rag_response.sources),
                        "collections_searched": rag_response.collections_searched,
                        "notebook_id": notebook_id,
                        "conversation_id": request.conversation_id
                    }) + "\n"
                    
                except Exception as e:
                    logger.warning(f"RAG query failed for notebook {notebook_id}: {str(e)}")
                    yield json.dumps({
                        "status": "context_warning",
                        "message": "Could not retrieve full context, proceeding with general response",
                        "notebook_id": notebook_id,
                        "conversation_id": request.conversation_id
                    }) + "\n"
            
            # Get notebook LLM configuration
            llm_config = get_notebook_llm_full_config()
            if not llm_config:
                yield json.dumps({
                    "error": "Notebook LLM configuration not available",
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
            
            # Initialize LLM with proper LLMConfig object
            llm_config_obj = LLMConfig(
                model_name=llm_config.get('model', 'qwen2.5:72b'),
                temperature=float(llm_config.get('temperature', 0.7)),
                top_p=float(llm_config.get('top_p', 0.9)),
                max_tokens=int(llm_config.get('max_tokens', 4096))
            )
            llm = OllamaLLM(llm_config_obj)
            
            # Build context-aware prompt
            system_prompt = "You are a helpful AI assistant for a notebook-based knowledge management system. "
            context_info = ""
            
            if rag_response and rag_response.sources:
                system_prompt += f"You have access to {len(rag_response.sources)} relevant document excerpts from this notebook. "
                system_prompt += "Use the provided context to give accurate, detailed answers. If the context doesn't contain enough information, say so clearly."
                
                context_info = "\n\n--- RELEVANT CONTEXT FROM NOTEBOOK ---\n"
                for i, source in enumerate(rag_response.sources[:request.max_sources], 1):
                    context_info += f"\n[Source {i} - {source.document_name or 'Unknown Document'}]:\n"
                    context_info += source.content[:1000] + ("..." if len(source.content) > 1000 else "")
                    context_info += "\n"
                context_info += "--- END CONTEXT ---\n\n"
            else:
                system_prompt += "No specific context was found in this notebook for this question. Provide a helpful general response and suggest ways the user might find relevant information."
            
            # Build the full prompt
            full_prompt = f"{system_prompt}\n\n{context_info}User Question: {request.message}\n\nResponse:"
            
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
                    for source in (rag_response.sources[:request.max_sources] if rag_response else [])
                ],
                "notebook_id": notebook_id,
                "conversation_id": request.conversation_id,
                "status": "complete"
            }
            
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

# Error handlers are handled at the app level, not router level
# The individual endpoints already have proper try/catch blocks with HTTPException