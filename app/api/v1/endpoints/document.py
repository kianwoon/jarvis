from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import requests
import json
import asyncio
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from uuid import uuid4
from datetime import datetime
import hashlib
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus, Qdrant
from app.core.embedding_settings_cache import get_embedding_settings
from app.core.vector_db_settings_cache import get_vector_db_settings
from app.core.http_embedding_function import HTTPEmbeddingFunction
from utils.deduplication import hash_text, get_existing_hashes, get_existing_doc_ids, filter_new_chunks
from app.rag.bm25_processor import BM25Processor
from app.core.document_classifier import get_document_classifier
from app.core.collection_registry_cache import get_collection_config
from app.utils.metadata_extractor import MetadataExtractor
from app.core.db import get_db, KnowledgeGraphDocument
from sqlalchemy.orm import Session

router = APIRouter()

class DocumentRequest(BaseModel):
    topic: str
    doc_id: str
    metadata: Optional[dict] = None

class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    file_type: Optional[str] = None
    file_size_bytes: Optional[int] = None
    processing_status: str
    milvus_collection: Optional[str] = None
    created_at: str

class DocumentsListResponse(BaseModel):
    documents: List[DocumentInfo]
    total_count: int

# HTTPEmbeddingFunction moved to app.core.http_embedding_function to avoid circular imports

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
            # BM25 enhancement fields
            FieldSchema(name="bm25_tokens", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="bm25_term_count", dtype=DataType.INT64),
            FieldSchema(name="bm25_unique_terms", dtype=DataType.INT64),
            FieldSchema(name="bm25_top_terms", dtype=DataType.VARCHAR, max_length=1000),
            # Date metadata fields
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

@router.post("/generate")
async def generate_document(
    request: DocumentRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate a document based on the given topic.
    This endpoint triggers a background task for document generation.
    """
    try:
        # TODO: Implement document generation logic
        # 1. Validate request
        # 2. Create background task
        # 3. Return task ID for tracking
        
        return {
            "message": "Document generation started",
            "doc_id": request.doc_id,
            "status": "processing"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start document generation: {str(e)}"
        )

@router.get("/", response_model=DocumentsListResponse)
async def list_documents(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    status: Optional[str] = Query(None, description="Filter by processing status"),
    file_type: Optional[str] = Query(None, description="Filter by file type"),
    db: Session = Depends(get_db)
):
    """
    List available documents that can be added to notebooks.
    
    Args:
        page: Page number (1-based)
        page_size: Number of items per page
        status: Filter by processing status (completed, processing, failed, etc.)
        file_type: Filter by file type (pdf, txt, etc.)
        db: Database session
        
    Returns:
        Paginated list of available documents
    """
    try:
        print(f"📋 Listing documents: page={page}, size={page_size}, status={status}, file_type={file_type}")
        
        # Build query - exclude memories, only show actual documents
        query = db.query(KnowledgeGraphDocument).filter(
            KnowledgeGraphDocument.content_type == 'document'
        )
        
        # Apply filters
        if status:
            query = query.filter(KnowledgeGraphDocument.processing_status == status)
        if file_type:
            query = query.filter(KnowledgeGraphDocument.file_type == file_type)
        
        # Get total count
        total_count = query.count()
        
        # Apply pagination
        documents = query.offset((page - 1) * page_size).limit(page_size).all()
        
        # Convert to response format
        document_list = []
        for doc in documents:
            document_list.append(DocumentInfo(
                document_id=doc.document_id,
                filename=doc.filename,
                file_type=doc.file_type,
                file_size_bytes=doc.file_size_bytes,
                processing_status=doc.processing_status,
                milvus_collection=doc.milvus_collection,
                created_at=doc.created_at.isoformat() if doc.created_at else ""
            ))
        
        print(f"✅ Found {len(document_list)} documents (total: {total_count})")
        
        return DocumentsListResponse(
            documents=document_list,
            total_count=total_count
        )
        
    except Exception as e:
        print(f"❌ Error listing documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list documents: {str(e)}"
        )

@router.post("/upload_pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    collection_name: Optional[str] = None,
    auto_classify: bool = True
):
    """Enhanced PDF upload with better chunking, metadata handling, and debugging"""
    
    # 1. Save file temporarily
    temp_path = f"/tmp/{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        print(f"📁 Saved {file.filename} to {temp_path}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to save file: {str(e)}")
    
    # 2. Load PDF with enhanced error handling
    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        print(f"✅ PDF loaded: {len(docs)} pages from {file.filename}")
        
        # Calculate total content for validation
        total_content_length = sum(len(doc.page_content) for doc in docs if doc.page_content)
        print(f"📊 Total content: {total_content_length} characters across {len(docs)} pages")
        
        # Check if all docs are empty, fallback to OCR
        if all(not doc.page_content.strip() for doc in docs):
            print("⚠️  All pages empty, trying OCR fallback...")
            try:
                from langchain_community.document_loaders import UnstructuredPDFLoader
                loader = UnstructuredPDFLoader(temp_path, mode="elements", strategy="ocr_only")
                docs = loader.load()
                print(f"✅ OCR loaded: {len(docs)} elements")
            except ImportError:
                print("❌ OCR fallback not available (UnstructuredPDFLoader not installed)")
                raise HTTPException(status_code=400, detail="PDF appears to be empty and OCR is not available")
            
    except Exception as e:
        print(f"❌ Error loading PDF: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to load PDF: {str(e)}")
    
    # 3. Enhanced text splitting for better context retention
    # Increased from 500 to 1500 chars for better context preservation
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,      # Larger chunks for better context
        chunk_overlap=200,    # Better overlap for continuity  
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        length_function=len,
        is_separator_regex=False
    )
    
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks (size: 1500, overlap: 200)")
    
    # Fallback if no chunks created
    if len(chunks) == 0:
        print("⚠️  No chunks from splitter, using whole pages as chunks")
        from langchain.schema import Document
        chunks = [Document(page_content=doc.page_content, metadata=doc.metadata) 
                 for doc in docs if doc.page_content.strip()]
        print(f"✅ Created {len(chunks)} page-based chunks")
    
    # 4. Get embedding configuration
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
    
    # 5. Enhanced metadata handling with proper page numbers and BM25 preprocessing
    # Compute file hash for doc_id uniqueness
    with open(temp_path, "rb") as f:
        file_content = f.read()
        file_id = hashlib.sha256(file_content).hexdigest()[:12]  # Shorter hash
    
    print(f"🔑 File ID: {file_id}")
    
    # Initialize BM25 processor
    bm25_processor = BM25Processor()
    
    # Extract file metadata
    file_metadata = MetadataExtractor.extract_metadata(temp_path, file.filename)
    
    # Enhanced metadata setting - preserve original page numbers from PyPDFLoader
    for i, chunk in enumerate(chunks):
        if not hasattr(chunk, 'metadata') or chunk.metadata is None:
            chunk.metadata = {}
        
        # IMPORTANT: Preserve original page number from PyPDFLoader - don't override with 0
        original_page = chunk.metadata.get('page', 0)
        
        chunk.metadata.update({
            'source': file.filename.lower(),  # Normalize filename to lowercase
            'page': original_page,  # Keep original page number
            'doc_type': 'pdf',
            'uploaded_at': datetime.now().isoformat(),
            'section': f"chunk_{i}",
            'author': '',
            'chunk_index': i,
            'file_id': file_id,
            'creation_date': file_metadata['creation_date'],
            'last_modified_date': file_metadata['last_modified_date']
        })
        
        # Add hash and robust doc_id
        chunk_hash = hash_text(chunk.page_content)
        chunk.metadata['hash'] = chunk_hash
        chunk.metadata['doc_id'] = f"{file_id}_p{original_page}_c{i}"
        
        # Add BM25 preprocessing
        bm25_metadata = bm25_processor.prepare_document_for_bm25(chunk.page_content, chunk.metadata)
        chunk.metadata.update(bm25_metadata)
    
    # 6. Determine target collection
    classified_type = None
    if collection_name:
        # Validate the provided collection exists
        collection_info = get_collection_config(collection_name)
        if not collection_info:
            raise HTTPException(status_code=400, detail=f"Collection '{collection_name}' not found")
        target_collection = collection_name
        print(f"📂 Using specified collection: {target_collection}")
    elif auto_classify:
        # Auto-classify document
        classifier = get_document_classifier()
        # Use first chunk for classification
        sample_content = chunks[0].page_content if chunks else ""
        doc_metadata = chunks[0].metadata if chunks else {}
        collection_type = classifier.classify_document(sample_content, doc_metadata)
        classified_type = collection_type  # Store for response
        target_collection = classifier.get_target_collection(collection_type)
        if not target_collection:
            target_collection = "default_knowledge"
        print(f"🤖 Auto-classified as '{collection_type}', using collection: {target_collection}")
        
        # Extract domain-specific metadata
        domain_metadata = classifier.extract_domain_metadata(sample_content, collection_type)
        for chunk in chunks:
            chunk.metadata.update(domain_metadata)
    else:
        # Use default collection
        target_collection = "default_knowledge"
        print(f"📂 Using default collection: {target_collection}")
    
    # Add collection info to chunk metadata
    for chunk in chunks:
        chunk.metadata['collection_name'] = target_collection
    
    # 7. Vector DB storage with enhanced deduplication logic
    from app.utils.vector_db_migration import migrate_vector_db_settings
    vector_db_cfg = migrate_vector_db_settings(get_vector_db_settings())
    
    # Find active Milvus database
    milvus_db = None
    for db in vector_db_cfg.get("databases", []):
        if db.get("id") == "milvus" and db.get("enabled"):
            milvus_db = db
            break
    
    if milvus_db:
        print("🔄 Using Milvus vector database")
        milvus_cfg = milvus_db.get("config", {})
        uri = milvus_cfg.get("MILVUS_URI")
        token = milvus_cfg.get("MILVUS_TOKEN")
        collection = target_collection  # Use the determined target collection
        vector_dim = int(milvus_cfg.get("dimension", 2560))
        
        # Ensure collection exists
        ensure_milvus_collection(collection, vector_dim=vector_dim, uri=uri, token=token)
        
        # Enhanced deduplication check
        collection_obj = Collection(collection)
        collection_obj.load()
        existing_hashes = get_existing_hashes(collection_obj)
        existing_doc_ids = get_existing_doc_ids(collection_obj)
        
        print(f"📊 Deduplication analysis:")
        print(f"   - Total chunks to process: {len(chunks)}")
        print(f"   - Existing hashes in DB: {len(existing_hashes)}")
        print(f"   - Existing doc_ids in DB: {len(existing_doc_ids)}")
        
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
        
        print(f"📊 After deduplication:")
        print(f"   - Unique chunks to insert: {len(unique_chunks)}")
        print(f"   - Duplicates filtered: {duplicate_count}")
        
        if not unique_chunks:
            # Cleanup temp file
            try:
                os.remove(temp_path)
            except:
                pass
            
            return {
                "status": "skipped", 
                "reason": "All chunks are duplicates", 
                "total_chunks": len(chunks),
                "filename": file.filename
            }
        
        # Generate embeddings and insert into Milvus
        unique_ids = [str(uuid4()) for _ in unique_chunks]
        unique_texts = [chunk.page_content for chunk in unique_chunks]
        
        print(f"🔄 Generating embeddings for {len(unique_texts)} chunks...")
        try:
            embeddings_list = embeddings.embed_documents(unique_texts)
            print(f"✅ Generated {len(embeddings_list)} embeddings")
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            # Cleanup temp file
            try:
                os.remove(temp_path)
            except:
                pass
            raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {str(e)}")
        
        # Prepare data for insertion
        data = [
            unique_ids,
            embeddings_list,
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
        
        print(f"🔄 Inserting {len(unique_chunks)} chunks into Milvus collection '{collection}'...")
        try:
            insert_result = collection_obj.insert(data)
            collection_obj.flush()  # Ensure data is written
            print(f"✅ Successfully inserted {len(unique_chunks)} chunks")
            
            # Update collection statistics
            try:
                from app.core.collection_statistics import update_collection_statistics
                stats_result = update_collection_statistics(
                    collection_name=collection,
                    chunks_added=len(unique_chunks),
                    uri=uri,
                    token=token
                )
                print(f"📊 Updated collection statistics: {stats_result}")
            except Exception as stats_error:
                print(f"⚠️  Failed to update collection statistics: {stats_error}")
                
        except Exception as e:
            print(f"❌ Error inserting into Milvus: {e}")
            # Cleanup temp file
            try:
                os.remove(temp_path)
            except:
                pass
            raise HTTPException(status_code=500, detail=f"Failed to insert into Milvus: {str(e)}")
        
        # Cleanup temporary file
        try:
            os.remove(temp_path)
            print(f"🧹 Cleaned up temporary file: {temp_path}")
        except Exception as e:
            print(f"⚠️  Could not remove temp file: {e}")
        
        response = {
            "status": "success", 
            "total_chunks": len(chunks),
            "unique_chunks": len(unique_chunks),
            "duplicates_filtered": duplicate_count,
            "filename": file.filename,
            "file_id": file_id,
            "collection": collection,
            "pages_processed": len(docs),
            "content_length": total_content_length,
            "chunk_size": 1500,
            "embedding_model": embedding_model or "HTTP endpoint"
        }
        
        if classified_type:
            response["classified_type"] = classified_type
            response["auto_classified"] = True
            
        return response
    
    elif qdrant_status:
        print("🔄 Using Qdrant vector database")
        qdrant_cfg = vector_db_cfg["qdrant"]
        host = qdrant_cfg.get("QDRANT_HOST", "localhost")
        port = qdrant_cfg.get("QDRANT_PORT", 6333)
        collection = target_collection  # Use the determined target collection
        
        vector_store = Qdrant(
            embedding_function=embeddings,
            collection_name=collection,
            url=f"http://{host}:{port}"
        )
        
        print(f"🔄 Adding {len(chunks)} chunks to Qdrant...")
        vector_store.add_documents(chunks)
        
        # Cleanup temporary file
        try:
            os.remove(temp_path)
        except Exception as e:
            print(f"⚠️  Could not remove temp file: {e}")
        
        response = {
            "status": "success", 
            "chunks": len(chunks),
            "filename": file.filename,
            "collection": collection,
            "pages_processed": len(docs),
            "vector_db": "qdrant"
        }
        
        if classified_type:
            response["classified_type"] = classified_type
            response["auto_classified"] = True
            
        return response
    
    else:
        # Cleanup temporary file even on error
        try:
            os.remove(temp_path)
        except:
            pass
        raise HTTPException(status_code=400, detail="No active vector DB configured")

@router.get("/debug_search/{query}")
async def debug_search(query: str, limit: int = 10, min_score: float = 0.5, min_length: int = 50):
    """Enhanced debug endpoint with quality filtering"""
    
    # Get configuration
    embedding_cfg = get_embedding_settings()
    vector_db_cfg = get_vector_db_settings()
    
    # Set up embeddings
    embedding_endpoint = embedding_cfg.get("embedding_endpoint")
    if embedding_endpoint:
        embeddings = HTTPEmbeddingFunction(embedding_endpoint)
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name=embedding_cfg["embedding_model"])
    
    # Connect to vector store
    from app.utils.vector_db_migration import migrate_vector_db_settings
    vector_db_cfg = migrate_vector_db_settings(vector_db_cfg)
    
    # Find active Milvus database
    milvus_db = None
    for db in vector_db_cfg.get("databases", []):
        if db.get("id") == "milvus" and db.get("enabled"):
            milvus_db = db
            break
    
    if not milvus_db:
        raise HTTPException(status_code=500, detail="No active Milvus configuration found")
        
    milvus_cfg = milvus_db.get("config", {})
    collection = milvus_cfg.get("MILVUS_DEFAULT_COLLECTION", "default_knowledge")
    uri = milvus_cfg.get("MILVUS_URI")
    token = milvus_cfg.get("MILVUS_TOKEN")
    
    milvus_store = Milvus(
        embedding_function=embeddings,
        collection_name=collection,
        connection_args={"uri": uri, "token": token},
        text_field="content"
    )
    
    # Search with more results to filter from
    try:
        # Normalize query for consistent vector search
        normalized_query = query.lower().strip()
        if hasattr(milvus_store, 'similarity_search_with_score'):
            docs = milvus_store.similarity_search_with_score(normalized_query, k=limit * 3)  # Get 3x more to filter
        else:
            docs_without_score = milvus_store.similarity_search(normalized_query, k=limit * 3)
            docs = [(doc, 1.0) for doc in docs_without_score]
        
        results = []
        query_lower = query.lower()
        
        for doc, score in docs:
            content = doc.page_content.strip()
            content_lower = content.lower()
            
            # Filter 1: Skip very short fragments
            if len(content) < min_length:
                continue
                
            # Filter 2: Skip low-confidence results
            # For COSINE distance: lower score = better match
            # Convert distance to similarity for filtering
            similarity = 1 - (score / 2)  # Convert [0,2] range to [1,0] similarity
            if similarity < min_score:
                continue
            
            # Boost 1: Exact keyword matches
            boost_factor = 1.0
            if query_lower in content_lower:
                boost_factor = 1.3  # 30% boost for exact matches
            
            # Boost 2: Multi-word query handling
            query_words = query_lower.split()
            if len(query_words) > 1:
                word_matches = sum(1 for word in query_words if word in content_lower)
                if word_matches >= len(query_words) * 0.5:  # At least 50% of words match
                    boost_factor = max(boost_factor, 1.2)
            
            # Apply boost to similarity, not distance
            adjusted_similarity = similarity * boost_factor
            
            results.append({
                "score": float(adjusted_similarity),
                "original_score": float(score),
                "similarity": float(similarity),
                "boost_applied": boost_factor > 1.0,
                "content_preview": content[:300] + "..." if len(content) > 300 else content,
                "content_length": len(content),
                "metadata": doc.metadata,
                "exact_keyword_match": query_lower in content_lower,
                "partial_keyword_matches": sum(1 for word in query_lower.split() if word in content_lower)
            })
            
            # Stop when we have enough good results
            if len(results) >= limit:
                break
        
        # Sort by adjusted score
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:limit]
        
        return {
            "query": query,
            "results_count": len(results),
            "collection": collection,
            "filters_applied": {
                "min_score": min_score,
                "min_length": min_length,
                "keyword_boosting": True
            },
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


