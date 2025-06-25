"""
Enhanced document upload with progress tracking via SSE
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from typing import Optional, List, Dict, Any
import json
import asyncio
from datetime import datetime
import hashlib
import os
import tempfile
from uuid import uuid4

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from pymilvus import Collection, utility

from app.core.embedding_settings_cache import get_embedding_settings
from app.core.vector_db_settings_cache import get_vector_db_settings
from app.api.v1.endpoints.document import HTTPEmbeddingFunction, ensure_milvus_collection
from utils.deduplication import hash_text, get_existing_hashes, get_existing_doc_ids
from app.rag.bm25_processor import BM25Processor
from app.utils.metadata_extractor import MetadataExtractor

router = APIRouter()

class UploadProgress:
    """Track upload progress"""
    def __init__(self, total_steps: int = 7):
        self.total_steps = total_steps
        self.current_step = 0
        self.current_step_name = ""
        self.details = {}
        self.error = None
        
    def update(self, step: int, step_name: str, details: Dict[str, Any] = None):
        self.current_step = step
        self.current_step_name = step_name
        if details:
            self.details.update(details)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress_percent": int((self.current_step / self.total_steps) * 100),
            "step_name": self.current_step_name,
            "details": self.details,
            "error": self.error
        }

async def progress_generator(file: UploadFile, progress: UploadProgress, upload_params: Dict[str, Any] = None):
    """Generate SSE events for upload progress"""
    
    try:
        # Step 1: Save file
        progress.update(1, "Saving uploaded file", {"filename": file.filename})
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        
        temp_path = None
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            temp_path = tmp_file.name
            content = await file.read()
            tmp_file.write(content)
            file_size_mb = len(content) / (1024 * 1024)
            progress.details["file_size_mb"] = round(file_size_mb, 2)
        
        # Step 2: Load PDF
        progress.update(2, "Loading PDF content", {"status": "processing"})
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        total_pages = len(docs)
        total_chars = sum(len(doc.page_content) for doc in docs)
        
        progress.details.update({
            "total_pages": total_pages,
            "total_characters": total_chars
        })
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        
        # Check if OCR is needed
        if all(not doc.page_content.strip() for doc in docs):
            progress.update(2, "Applying OCR (this may take longer)", {"ocr_required": True})
            yield f"data: {json.dumps(progress.to_dict())}\n\n"
            
            try:
                from langchain_community.document_loaders import UnstructuredPDFLoader
                loader = UnstructuredPDFLoader(temp_path, mode="elements", strategy="ocr_only")
                docs = loader.load()
                progress.details["ocr_applied"] = True
            except ImportError:
                raise HTTPException(status_code=400, detail="PDF appears to be empty and OCR is not available")
        
        # Step 3: Split into chunks
        progress.update(3, "Splitting into chunks", {"chunking": "in_progress"})
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
            length_function=len,
            is_separator_regex=False
        )
        
        chunks = splitter.split_documents(docs)
        progress.details["total_chunks"] = len(chunks)
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        
        # Step 4: Prepare embeddings
        progress.update(4, "Preparing embeddings", {"embedding_setup": "initializing"})
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        
        embedding_cfg = get_embedding_settings()
        embedding_endpoint = embedding_cfg.get('embedding_endpoint')
        
        if embedding_endpoint:
            embeddings = HTTPEmbeddingFunction(embedding_endpoint)
            progress.details["embedding_type"] = "HTTP endpoint"
        else:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name=embedding_cfg["embedding_model"])
            progress.details["embedding_type"] = f"HuggingFace: {embedding_cfg['embedding_model']}"
        
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        
        # Step 5: Check for duplicates
        progress.update(5, "Checking for duplicates", {"deduplication": "in_progress"})
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        
        # Compute file hash
        with open(temp_path, "rb") as f:
            file_content = f.read()
            file_id = hashlib.sha256(file_content).hexdigest()[:12]
        
        # Initialize BM25 processor
        bm25_processor = BM25Processor()
        
        # Extract file metadata
        file_metadata = MetadataExtractor.extract_metadata(temp_path, file.filename)
        
        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                chunk.metadata = {}
            
            original_page = chunk.metadata.get('page', 0)
            chunk.metadata.update({
                'source': file.filename.lower(),  # Normalize filename to lowercase
                'page': original_page,
                'doc_type': 'pdf',
                'uploaded_at': datetime.now().isoformat(),
                'section': f"chunk_{i}",
                'author': '',
                'chunk_index': i,
                'file_id': file_id,
                'hash': hash_text(chunk.page_content),
                'doc_id': f"{file_id}_p{original_page}_c{i}",
                'creation_date': file_metadata['creation_date'],
                'last_modified_date': file_metadata['last_modified_date']
            })
            
            # Add BM25 preprocessing
            bm25_metadata = bm25_processor.prepare_document_for_bm25(chunk.page_content, chunk.metadata)
            chunk.metadata.update(bm25_metadata)
        
        # Connect to vector DB and check duplicates
        vector_db_cfg = get_vector_db_settings()
        milvus_cfg = vector_db_cfg["milvus"]
        uri = milvus_cfg.get("MILVUS_URI")
        token = milvus_cfg.get("MILVUS_TOKEN")
        # Use user-specified collection or fall back to default
        if upload_params and upload_params.get('target_collection'):
            collection_name = upload_params['target_collection']
            progress.details["collection_method"] = "user_specified"
            progress.details["collection_source"] = "frontend_selection"
        else:
            collection_name = milvus_cfg.get("MILVUS_DEFAULT_COLLECTION", "default_knowledge")
            progress.details["collection_method"] = "default_fallback"
            progress.details["collection_source"] = "milvus_config"
        
        progress.details["target_collection"] = collection_name
        vector_dim = int(milvus_cfg.get("dimension", 2560))
        
        ensure_milvus_collection(collection_name, vector_dim=vector_dim, uri=uri, token=token)
        
        collection = Collection(collection_name)
        collection.load()
        existing_hashes = get_existing_hashes(collection)
        existing_doc_ids = get_existing_doc_ids(collection)
        
        # Filter duplicates
        unique_chunks = []
        duplicate_count = 0
        
        for chunk in chunks:
            chunk_hash = chunk.metadata.get('hash')
            doc_id = chunk.metadata.get('doc_id')
            
            if chunk_hash in existing_hashes or doc_id in existing_doc_ids:
                duplicate_count += 1
            else:
                unique_chunks.append(chunk)
        
        progress.details.update({
            "duplicates_found": duplicate_count,
            "unique_chunks": len(unique_chunks)
        })
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        
        if not unique_chunks:
            progress.update(7, "Complete - All chunks are duplicates", {
                "status": "skipped",
                "reason": "All chunks already exist in database"
            })
            yield f"data: {json.dumps(progress.to_dict())}\n\n"
            return
        
        # Step 6: Generate embeddings
        progress.update(6, "Generating embeddings", {
            "embedding_progress": 0,
            "total_embeddings": len(unique_chunks)
        })
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        
        # Process embeddings in batches for progress updates
        batch_size = 10
        all_embeddings = []
        
        for i in range(0, len(unique_chunks), batch_size):
            batch = unique_chunks[i:i + batch_size]
            batch_texts = [chunk.page_content for chunk in batch]
            
            batch_embeddings = embeddings.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
            progress.details["embedding_progress"] = len(all_embeddings)
            yield f"data: {json.dumps(progress.to_dict())}\n\n"
            
            # Small delay to prevent overwhelming the client
            await asyncio.sleep(0.1)
        
        # Step 7: Insert into Milvus
        progress.update(7, "Inserting into vector database", {"insertion": "in_progress"})
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        
        # Prepare data for insertion
        unique_ids = [str(uuid4()) for _ in unique_chunks]
        unique_texts = [chunk.page_content for chunk in unique_chunks]
        
        data = [
            unique_ids,
            all_embeddings,
            unique_texts,
            [chunk.metadata.get('source', '') for chunk in unique_chunks],
            [chunk.metadata.get('page', 0) for chunk in unique_chunks],
            [chunk.metadata.get('doc_type', 'pdf') for chunk in unique_chunks],
            [chunk.metadata.get('uploaded_at', '') for chunk in unique_chunks],
            [chunk.metadata.get('section', '') for chunk in unique_chunks],
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
        
        insert_result = collection.insert(data)
        collection.flush()
        
        # Final success
        progress.update(7, "Upload complete!", {
            "status": "success",
            "total_chunks": len(chunks),
            "unique_chunks_inserted": len(unique_chunks),
            "duplicates_filtered": duplicate_count,
            "collection": collection_name,
            "file_id": file_id
        })
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        
    except Exception as e:
        progress.error = str(e)
        progress.details["status"] = "error"
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        
    finally:
        # Cleanup
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

@router.post("/upload_pdf_progress")
async def upload_pdf_with_progress(
    file: UploadFile = File(...),
    collection_name: Optional[str] = Form(None),
    disable_auto_classification: Optional[str] = Form(None),
    force_collection: Optional[str] = Form(None),
    chunk_size: Optional[str] = Form(None),
    chunk_overlap: Optional[str] = Form(None),
    enable_bm25: Optional[str] = Form(None),
    bm25_weight: Optional[str] = Form(None)
):
    """Upload PDF with real-time progress updates via SSE"""
    
    print("=" * 60)
    print("🚨 BACKEND ENDPOINT HIT: upload_pdf_progress")
    print("=" * 60)
    
    # DEBUG: Log all received parameters
    print(f"BACKEND DEBUG: Received parameters:")
    print(f"  file.filename: {file.filename}")
    print(f"  collection_name: {collection_name}")
    print(f"  force_collection: {force_collection}")
    print(f"  disable_auto_classification: {disable_auto_classification}")
    print(f"  chunk_size: {chunk_size}")
    print(f"  chunk_overlap: {chunk_overlap}")
    print(f"  enable_bm25: {enable_bm25}")
    print(f"  bm25_weight: {bm25_weight}")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    progress = UploadProgress()
    
    # Determine target collection
    target_collection = None
    if force_collection:
        target_collection = force_collection
        print(f"BACKEND DEBUG: Using force_collection: {force_collection}")
    elif collection_name:
        target_collection = collection_name
        print(f"BACKEND DEBUG: Using collection_name: {collection_name}")
    else:
        print(f"BACKEND DEBUG: No collection specified, will use default")
    
    # Create upload parameters
    upload_params = {
        'target_collection': target_collection,
        'chunk_size': int(chunk_size) if chunk_size else None,
        'chunk_overlap': int(chunk_overlap) if chunk_overlap else None,
        'enable_bm25': enable_bm25 == 'true' if enable_bm25 else None,
        'bm25_weight': float(bm25_weight) if bm25_weight else None,
    }
    
    return StreamingResponse(
        progress_generator(file, progress, upload_params),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable proxy buffering
        }
    )