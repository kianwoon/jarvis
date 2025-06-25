"""
Enhanced document upload endpoint with progress tracking for multiple file types
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Query, Form
from fastapi.responses import StreamingResponse
from typing import Optional, List, Dict, Any
import os
import tempfile
import shutil
from datetime import datetime
import hashlib
import json
import asyncio

# Document handlers
from app.document_handlers.base import ExtractedChunk
from app.document_handlers.excel_handler import ExcelHandler
from app.document_handlers.word_handler import WordHandler
from app.document_handlers.powerpoint_handler import PowerPointHandler

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
from app.rag.bm25_processor import BM25Processor
from app.core.embedding_settings_cache import get_embedding_settings
from app.core.vector_db_settings_cache import get_vector_db_settings
from app.utils.metadata_extractor import MetadataExtractor
from pymilvus import Collection
from uuid import uuid4

router = APIRouter()

# Document handler registry
# Document handler registry - lazy initialization to avoid import errors
def get_document_handlers():
    """Get document handlers with lazy initialization"""
    handlers = {}
    
    # Initialize handlers that are available
    try:
        handlers['.xlsx'] = ExcelHandler()
        handlers['.xls'] = ExcelHandler()
    except ImportError:
        logger.warning("Excel handler not available")
    
    try:
        handlers['.docx'] = WordHandler()
    except ImportError:
        logger.warning("Word handler not available")
    
    try:
        from app.document_handlers.powerpoint_handler import PowerPointHandler
        handlers['.pptx'] = PowerPointHandler()
        handlers['.ppt'] = PowerPointHandler()
    except ImportError:
        logger.warning("PowerPoint handler not available")
    
    return handlers

# Cache the handlers after first initialization
_document_handlers = None

def get_handler_for_file(file_extension):
    """Get the appropriate handler for a file extension"""
    global _document_handlers
    if _document_handlers is None:
        _document_handlers = get_document_handlers()
    return _document_handlers.get(file_extension)

DOCUMENT_HANDLERS = None  # Will be lazy-loaded

# Allowed file extensions  
ALLOWED_EXTENSIONS = {'.pdf', '.xlsx', '.xls', '.docx', '.pptx', '.ppt'}  # .doc not fully supported yet

async def progress_generator(file_path: str, file_name: str, file_ext: str, options: Dict[str, Any], upload_params: Dict[str, Any] = None):
    """Generate progress updates for document processing"""
    total_steps = 6
    current_step = 0
    
    try:
        # Step 1: Validate file
        current_step += 1
        yield f"data: {json.dumps({'current_step': current_step, 'total_steps': total_steps, 'step_name': 'Validating file', 'progress_percent': int(current_step / total_steps * 100)})}\n\n"
        await asyncio.sleep(0.1)
        
        handler = get_handler_for_file(file_ext)
        if not handler:
            yield f"data: {json.dumps({'error': f'No handler for {file_ext}'})}\n\n"
            return
            
        is_valid, error_msg = handler.validate(file_path)
        if not is_valid:
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
            return
        
        # Step 2: Extract content
        current_step += 1
        yield f"data: {json.dumps({'current_step': current_step, 'total_steps': total_steps, 'step_name': 'Extracting content', 'progress_percent': int(current_step / total_steps * 100)})}\n\n"
        await asyncio.sleep(0.1)
        
        chunks = handler.extract(file_path, options)
        if not chunks:
            yield f"data: {json.dumps({'error': 'No content extracted from file'})}\n\n"
            return
        
        # Extract file metadata
        file_metadata = MetadataExtractor.extract_metadata(file_path, filename)
        
        # Add metadata to all chunks
        for chunk in chunks:
            if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata['creation_date'] = file_metadata['creation_date']
            chunk.metadata['last_modified_date'] = file_metadata['last_modified_date']
        
        # Step 3: Generate embeddings
        current_step += 1
        yield f"data: {json.dumps({'current_step': current_step, 'total_steps': total_steps, 'step_name': 'Generating embeddings', 'progress_percent': int(current_step / total_steps * 100), 'details': {'chunks_to_process': len(chunks)}})}\n\n"
        await asyncio.sleep(0.1)
        
        # Get embedding configuration
        embedding_cfg = get_embedding_settings()
        embedding_model = embedding_cfg.get('embedding_model')
        embedding_endpoint = embedding_cfg.get('embedding_endpoint')
        
        if embedding_endpoint:
            embeddings = HTTPEmbeddingFunction(embedding_endpoint)
        else:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Step 4: Check for duplicates
        current_step += 1
        yield f"data: {json.dumps({'current_step': current_step, 'total_steps': total_steps, 'step_name': 'Checking for duplicates', 'progress_percent': int(current_step / total_steps * 100)})}\n\n"
        await asyncio.sleep(0.1)
        
        # Get vector DB configuration
        vector_db_cfg = get_vector_db_settings()
        milvus_status = vector_db_cfg.get("milvus", {}).get("status", False)
        
        if not milvus_status:
            yield f"data: {json.dumps({'error': 'Milvus vector database not configured'})}\n\n"
            return
        
        milvus_cfg = vector_db_cfg["milvus"]
        uri = milvus_cfg.get("MILVUS_URI")
        token = milvus_cfg.get("MILVUS_TOKEN")
        # Use user-specified collection or fall back to default
        if upload_params and upload_params.get('target_collection'):
            collection_name = upload_params['target_collection']
        else:
            collection_name = milvus_cfg.get("MILVUS_DEFAULT_COLLECTION", "default_knowledge")
        vector_dim = int(milvus_cfg.get("dimension", 2560))
        
        # Ensure collection exists
        ensure_milvus_collection(collection_name, vector_dim=vector_dim, uri=uri, token=token)
        
        # Deduplication
        collection_obj = Collection(collection_name)
        collection_obj.load()
        existing_hashes = get_existing_hashes(collection_obj)
        existing_doc_ids = get_existing_doc_ids(collection_obj)
        
        # Initialize BM25 processor
        bm25_processor = BM25Processor()
        
        unique_chunks = []
        duplicate_count = 0
        
        for chunk in chunks:
            # Check if hash already exists in metadata (handlers might have set it)
            chunk_hash = chunk.metadata.get('hash', hash_text(chunk.content))
            # Use doc_id if set by handler, otherwise generate new one
            doc_id = chunk.metadata.get('doc_id', chunk.metadata.get('chunk_id', str(uuid4())))
            
            if chunk_hash in existing_hashes or doc_id in existing_doc_ids:
                duplicate_count += 1
                continue
                
            chunk.metadata['hash'] = chunk_hash
            chunk.metadata['doc_id'] = doc_id
            
            # Add BM25 preprocessing
            bm25_metadata = bm25_processor.prepare_document_for_bm25(chunk.content, chunk.metadata)
            chunk.metadata.update(bm25_metadata)
            
            unique_chunks.append(chunk)
        
        if not unique_chunks:
            yield f"data: {json.dumps({'current_step': total_steps, 'total_steps': total_steps, 'step_name': 'Complete', 'progress_percent': 100, 'details': {'status': 'skipped', 'reason': 'All chunks are duplicates', 'total_chunks': len(chunks), 'duplicates_filtered': duplicate_count}})}\n\n"
            return
        
        # Step 5: Generate embeddings
        current_step += 1
        yield f"data: {json.dumps({'current_step': current_step, 'total_steps': total_steps, 'step_name': 'Generating embeddings', 'progress_percent': int(current_step / total_steps * 100), 'details': {'embedding_progress': 0, 'total_embeddings': len(unique_chunks), 'unique_chunks': len(unique_chunks), 'duplicates': duplicate_count}})}\n\n"
        await asyncio.sleep(0.1)
        
        # Generate embeddings in batches for progress updates
        unique_ids = [str(uuid4()) for _ in unique_chunks]
        unique_texts = [chunk.content for chunk in unique_chunks]
        
        batch_size = 10
        all_embeddings = []
        
        for i in range(0, len(unique_chunks), batch_size):
            batch = unique_chunks[i:i + batch_size]
            batch_texts = [chunk.content for chunk in batch]
            
            batch_embeddings = embeddings.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
            # Update embedding progress
            yield f"data: {json.dumps({'current_step': current_step, 'total_steps': total_steps, 'step_name': 'Generating embeddings', 'progress_percent': int(current_step / total_steps * 100), 'details': {'embedding_progress': len(all_embeddings), 'total_embeddings': len(unique_chunks), 'unique_chunks': len(unique_chunks), 'duplicates': duplicate_count}})}\n\n"
            
            # Small delay to prevent overwhelming the client
            await asyncio.sleep(0.1)
        
        embeddings_list = all_embeddings
        
        # Step 5 final step: Insert into vector database
        current_step += 1
        yield f"data: {json.dumps({'current_step': current_step, 'total_steps': total_steps, 'step_name': 'Inserting into vector database', 'progress_percent': int(current_step / total_steps * 100), 'details': {'insertion': 'in_progress', 'unique_chunks': len(unique_chunks), 'duplicates': duplicate_count}})}\n\n"
        await asyncio.sleep(0.1)
        
        # Prepare data for insertion
        data = [
            unique_ids,
            embeddings_list,
            unique_texts,
            [chunk.metadata.get('source', file_name) for chunk in unique_chunks],
            [chunk.metadata.get('page', 0) for chunk in unique_chunks],
            [chunk.metadata.get('doc_type', file_ext[1:]) for chunk in unique_chunks],
            [chunk.metadata.get('uploaded_at', datetime.now().isoformat()) for chunk in unique_chunks],
            [chunk.metadata.get('section', chunk.metadata.get('sheet_name', '')) for chunk in unique_chunks],
            [chunk.metadata.get('author', '') for chunk in unique_chunks],
            [chunk.metadata.get('hash', '') for chunk in unique_chunks],
            [chunk.metadata.get('doc_id', '') for chunk in unique_chunks],
            # BM25 enhancement fields
            [chunk.metadata.get('bm25_tokens', '') for chunk in unique_chunks],
            [chunk.metadata.get('bm25_term_count', 0) for chunk in unique_chunks],
            [chunk.metadata.get('bm25_unique_terms', 0) for chunk in unique_chunks],
            [chunk.metadata.get('bm25_top_terms', '') for chunk in unique_chunks],
            # Date metadata fields
            [chunk.metadata.get('creation_date', '') for chunk in unique_chunks],
            [chunk.metadata.get('last_modified_date', '') for chunk in unique_chunks],
        ]
        
        insert_result = collection_obj.insert(data)
        collection_obj.flush()
        
        # Final success message
        metadata_summary = {}
        if file_ext in ['.xlsx', '.xls']:
            sheets = {}
            for chunk in unique_chunks:
                sheet = chunk.metadata.get('sheet_name', 'Unknown')
                if sheet not in sheets:
                    sheets[sheet] = 0
                sheets[sheet] += 1
            metadata_summary['sheets_processed'] = sheets
        elif file_ext in ['.pptx', '.ppt']:
            slides = {}
            for chunk in unique_chunks:
                slide = chunk.metadata.get('slide_number', 'Unknown')
                if slide not in slides:
                    slides[slide] = 0
                slides[slide] += 1
            metadata_summary['slides_processed'] = slides
        elif file_ext in ['.docx']:
            sections = {}
            for chunk in unique_chunks:
                section = chunk.metadata.get('section', 'Unknown')
                if section not in sections:
                    sections[section] = 0
                sections[section] += 1
            metadata_summary['sections_processed'] = sections
        
        # Quality scores for all file types
        if unique_chunks and hasattr(unique_chunks[0], 'quality_score'):
            metadata_summary['quality_scores'] = {
                'avg': sum(getattr(c, 'quality_score', 0.5) for c in unique_chunks) / len(unique_chunks),
                'min': min(getattr(c, 'quality_score', 0.5) for c in unique_chunks),
                'max': max(getattr(c, 'quality_score', 0.5) for c in unique_chunks)
            }
        
        yield f"data: {json.dumps({'current_step': total_steps, 'total_steps': total_steps, 'step_name': 'Complete', 'progress_percent': 100, 'details': {'status': 'success', 'total_chunks': len(chunks), 'unique_chunks_inserted': len(unique_chunks), 'duplicates_filtered': duplicate_count, 'metadata_summary': metadata_summary}})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@router.post("/upload-multi-progress")
async def upload_document_with_progress(
    file: UploadFile = File(...),
    sheets: Optional[str] = Query(None, description="Comma-separated sheet names for Excel"),
    exclude_sheets: Optional[str] = Query(None, description="Sheets to exclude"),
    collection_name: Optional[str] = Form(None),
    disable_auto_classification: Optional[str] = Form(None),
    force_collection: Optional[str] = Form(None),
    chunk_size: Optional[str] = Form(None),
    chunk_overlap: Optional[str] = Form(None),
    enable_bm25: Optional[str] = Form(None),
    bm25_weight: Optional[str] = Form(None)
):
    """
    Upload and process various document types with progress tracking
    Returns Server-Sent Events stream
    """
    # Validate file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # For PDF, redirect to existing endpoint
    if file_ext == '.pdf':
        raise HTTPException(
            status_code=400,
            detail="Please use /upload_pdf_progress endpoint for PDF files"
        )
    
    # Determine target collection
    target_collection = None
    if force_collection:
        target_collection = force_collection
    elif collection_name:
        target_collection = collection_name
    
    # Create upload parameters
    upload_params = {
        'target_collection': target_collection,
        'chunk_size': int(chunk_size) if chunk_size else None,
        'chunk_overlap': int(chunk_overlap) if chunk_overlap else None,
        'enable_bm25': enable_bm25 == 'true' if enable_bm25 else None,
        'bm25_weight': float(bm25_weight) if bm25_weight else None,
    }
    
    # Save uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Save file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Prepare extraction options
        extraction_options = {}
        if sheets:
            extraction_options['sheets'] = [s.strip() for s in sheets.split(',')]
        if exclude_sheets:
            extraction_options['exclude_sheets'] = [s.strip() for s in exclude_sheets.split(',')]
        
        # Create async generator for progress updates
        async def cleanup_and_stream():
            try:
                async for chunk in progress_generator(temp_path, file.filename.lower(), file_ext, extraction_options, upload_params):
                    yield chunk
            finally:
                # Cleanup
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
        
        return StreamingResponse(
            cleanup_and_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            }
        )
        
    except Exception as e:
        # Cleanup on error
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))