"""
Enhanced document upload endpoint supporting multiple file types
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
import os
import tempfile
import shutil
from datetime import datetime
import hashlib

# Document handlers
from app.document_handlers.base import ExtractedChunk
from app.document_handlers.excel_handler import ExcelHandler

# Existing imports for vector storage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from app.api.v1.endpoints.document import (
    ensure_milvus_collection, 
    HTTPEmbeddingFunction,
    hash_text,
    get_existing_hashes,
    get_existing_doc_ids
)
from app.core.embedding_settings_cache import get_embedding_settings
from app.core.vector_db_settings_cache import get_vector_db_settings
from pymilvus import Collection
from uuid import uuid4

router = APIRouter()

# Document handler registry
DOCUMENT_HANDLERS = {
    '.pdf': None,  # Will use existing PDF handler
    '.xlsx': ExcelHandler(),
    '.xls': ExcelHandler(),
    # Future handlers
    # '.docx': WordHandler(),
    # '.doc': WordHandler(),
    # '.pptx': PowerPointHandler(),
    # '.ppt': PowerPointHandler(),
}

# Allowed file extensions
ALLOWED_EXTENSIONS = {'.pdf', '.xlsx', '.xls'}  # Will expand as we add handlers

from fastapi.responses import StreamingResponse
import asyncio

@router.post("/upload-multi")
async def upload_document(
    file: UploadFile = File(...),
    preview: bool = Query(False, description="Preview extraction without storing"),
    sheets: Optional[str] = Query(None, description="Comma-separated sheet names for Excel"),
    exclude_sheets: Optional[str] = Query(None, description="Sheets to exclude"),
):
    """
    Upload and process various document types
    Supports: PDF, Excel (XLS/XLSX)
    """
    # Validate file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Save uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Save file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"📁 Processing {file.filename} ({file_ext})")
        
        # Route to appropriate handler
        if file_ext == '.pdf':
            # Use existing PDF processing
            from app.api.v1.endpoints.document import upload_pdf
            # Forward to existing PDF handler
            # Note: This is a simplified approach, in production you'd refactor
            return {"status": "PDF processing not refactored yet", "message": "Please use /upload_pdf endpoint"}
        
        # Get handler
        handler = DOCUMENT_HANDLERS.get(file_ext)
        if not handler:
            raise HTTPException(status_code=500, detail=f"No handler for {file_ext}")
        
        # Validate file
        is_valid, error_msg = handler.validate(temp_path)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Preview mode
        if preview:
            preview_result = handler.get_preview(temp_path)
            return {
                "status": "preview",
                "preview": preview_result.to_dict()
            }
        
        # Prepare extraction options
        extraction_options = {}
        if sheets:
            extraction_options['sheets'] = [s.strip() for s in sheets.split(',')]
        if exclude_sheets:
            extraction_options['exclude_sheets'] = [s.strip() for s in exclude_sheets.split(',')]
        
        # Extract content
        chunks = handler.extract(temp_path, extraction_options)
        
        if not chunks:
            return {
                "status": "error",
                "message": "No content extracted from file",
                "filename": file.filename
            }
        
        print(f"✅ Extracted {len(chunks)} chunks from {file.filename}")
        
        # Store in vector database
        result = await store_chunks_in_vectordb(chunks, file.filename, file_ext)
        
        return result
        
    except Exception as e:
        print(f"❌ Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Cleanup
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

async def store_chunks_in_vectordb(
    chunks: List[ExtractedChunk], 
    filename: str,
    file_type: str
) -> Dict[str, Any]:
    """Store extracted chunks in vector database"""
    
    # Get embedding configuration
    embedding_cfg = get_embedding_settings()
    embedding_model = embedding_cfg.get('embedding_model')
    embedding_endpoint = embedding_cfg.get('embedding_endpoint')
    
    if embedding_endpoint:
        embeddings = HTTPEmbeddingFunction(embedding_endpoint)
        print("✅ Using HTTP embedding endpoint")
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        print(f"✅ Using HuggingFace embeddings: {embedding_model}")
    
    # Get vector DB configuration
    vector_db_cfg = get_vector_db_settings()
    milvus_status = vector_db_cfg.get("milvus", {}).get("status", False)
    
    if not milvus_status:
        raise HTTPException(status_code=500, detail="Milvus vector database not configured")
    
    print("🔄 Using Milvus vector database")
    milvus_cfg = vector_db_cfg["milvus"]
    uri = milvus_cfg.get("MILVUS_URI")
    token = milvus_cfg.get("MILVUS_TOKEN")
    collection_name = milvus_cfg.get("MILVUS_DEFAULT_COLLECTION", "default_knowledge")
    vector_dim = int(milvus_cfg.get("dimension", 1536))
    
    # Ensure collection exists
    ensure_milvus_collection(collection_name, vector_dim=vector_dim, uri=uri, token=token)
    
    # Deduplication check
    collection_obj = Collection(collection_name)
    collection_obj.load()
    existing_hashes = get_existing_hashes(collection_obj)
    existing_doc_ids = get_existing_doc_ids(collection_obj)
    
    print(f"📊 Deduplication analysis:")
    print(f"   - Total chunks to process: {len(chunks)}")
    print(f"   - Existing hashes in DB: {len(existing_hashes)}")
    
    # Convert ExtractedChunk to format for vector DB
    unique_chunks = []
    duplicate_count = 0
    
    for chunk in chunks:
        # Generate hash
        chunk_hash = hash_text(chunk.content)
        chunk_id = chunk.metadata.get('chunk_id', str(uuid4()))
        
        # Check duplicates
        if chunk_hash in existing_hashes or chunk_id in existing_doc_ids:
            duplicate_count += 1
            continue
            
        # Add hash to metadata
        chunk.metadata['hash'] = chunk_hash
        chunk.metadata['doc_id'] = chunk_id
        unique_chunks.append(chunk)
    
    print(f"📊 After deduplication:")
    print(f"   - Unique chunks to insert: {len(unique_chunks)}")
    print(f"   - Duplicates filtered: {duplicate_count}")
    
    if not unique_chunks:
        return {
            "status": "skipped",
            "reason": "All chunks are duplicates",
            "total_chunks": len(chunks),
            "filename": filename
        }
    
    # Generate embeddings
    unique_ids = [str(uuid4()) for _ in unique_chunks]
    unique_texts = [chunk.content for chunk in unique_chunks]
    
    print(f"🔄 Generating embeddings for {len(unique_texts)} chunks...")
    try:
        embeddings_list = embeddings.embed_documents(unique_texts)
        print(f"✅ Generated {len(embeddings_list)} embeddings")
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {str(e)}")
    
    # Prepare data for insertion
    data = [
        unique_ids,
        embeddings_list,
        unique_texts,
        [chunk.metadata.get('source', filename) for chunk in unique_chunks],
        [chunk.metadata.get('page', 0) for chunk in unique_chunks],  # Use 0 for non-PDF
        [chunk.metadata.get('doc_type', file_type) for chunk in unique_chunks],
        [chunk.metadata.get('uploaded_at', datetime.now().isoformat()) for chunk in unique_chunks],
        [chunk.metadata.get('section', chunk.metadata.get('sheet_name', '')) for chunk in unique_chunks],
        [chunk.metadata.get('author', '') for chunk in unique_chunks],
        [chunk.metadata.get('hash', '') for chunk in unique_chunks],
        [chunk.metadata.get('doc_id', '') for chunk in unique_chunks],
    ]
    
    print(f"🔄 Inserting {len(unique_chunks)} chunks into Milvus collection '{collection_name}'...")
    try:
        insert_result = collection_obj.insert(data)
        collection_obj.flush()
        print(f"✅ Successfully inserted {len(unique_chunks)} chunks")
    except Exception as e:
        print(f"❌ Error inserting into Milvus: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to insert into Milvus: {str(e)}")
    
    # Prepare detailed response
    metadata_summary = {}
    if file_type in ['.xlsx', '.xls']:
        # Summarize by sheet
        sheets = {}
        for chunk in unique_chunks:
            sheet = chunk.metadata.get('sheet_name', 'Unknown')
            if sheet not in sheets:
                sheets[sheet] = 0
            sheets[sheet] += 1
        metadata_summary['sheets_processed'] = sheets
        metadata_summary['quality_scores'] = {
            'avg': sum(c.quality_score for c in unique_chunks) / len(unique_chunks),
            'min': min(c.quality_score for c in unique_chunks),
            'max': max(c.quality_score for c in unique_chunks)
        }
    
    return {
        "status": "success",
        "total_chunks": len(chunks),
        "unique_chunks": len(unique_chunks),
        "duplicates_filtered": duplicate_count,
        "filename": filename,
        "file_type": file_type,
        "collection": collection_name,
        "embedding_model": embedding_model or "HTTP endpoint",
        "metadata_summary": metadata_summary
    }

@router.get("/supported-types")
async def get_supported_types():
    """Get list of supported file types and their capabilities"""
    return {
        "supported_types": [
            {
                "extension": ".pdf",
                "description": "Portable Document Format",
                "features": ["Text extraction", "OCR support", "Page-based chunking"]
            },
            {
                "extension": ".xlsx",
                "description": "Excel Workbook (2007+)",
                "features": ["Multi-sheet support", "Table structure preservation", "Smart chunking"]
            },
            {
                "extension": ".xls", 
                "description": "Excel Workbook (97-2003)",
                "features": ["Multi-sheet support", "Table structure preservation", "Smart chunking"]
            }
        ],
        "coming_soon": [
            {"extension": ".docx", "description": "Word Document"},
            {"extension": ".doc", "description": "Word Document (97-2003)"},
            {"extension": ".pptx", "description": "PowerPoint Presentation"},
            {"extension": ".ppt", "description": "PowerPoint Presentation (97-2003)"}
        ]
    }