from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
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
from utils.deduplication import hash_text, get_existing_hashes, get_existing_doc_ids, filter_new_chunks
from app.rag.bm25_processor import BM25Processor
from app.core.document_classifier import get_document_classifier
from app.core.collection_registry_cache import get_collection_config

router = APIRouter()

class DocumentRequest(BaseModel):
    topic: str
    doc_id: str
    metadata: Optional[dict] = None

# Custom embedding function for HTTP endpoint
class HTTPEmbeddingFunction:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            # Use lowercase for consistent embeddings
            normalized_text = text.lower().strip()
            payload = {"texts": [normalized_text]}
            resp = requests.post("http://qwen-embedder:8050/embed", json=payload)
            if resp.status_code == 422:
                print("Response content:", resp.content)
            resp.raise_for_status()
            embeddings.append(resp.json()["embeddings"][0])
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        # Also normalize queries to lowercase
        return self.embed_documents([text])[0]

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
    vector_db_cfg = get_vector_db_settings()
    milvus_status = vector_db_cfg.get("milvus", {}).get("status", False)
    qdrant_status = vector_db_cfg.get("qdrant", {}).get("status", False)
    
    if milvus_status:
        print("🔄 Using Milvus vector database")
        milvus_cfg = vector_db_cfg["milvus"]
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
        ]
        
        print(f"🔄 Inserting {len(unique_chunks)} chunks into Milvus collection '{collection}'...")
        try:
            insert_result = collection_obj.insert(data)
            collection_obj.flush()  # Ensure data is written
            print(f"✅ Successfully inserted {len(unique_chunks)} chunks")
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
    milvus_cfg = vector_db_cfg["milvus"]
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

class UploadProgress:
    """Track upload progress"""
    def __init__(self, total_steps: int = 8):
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

async def progress_generator(file: UploadFile, progress: UploadProgress, collection_name: Optional[str] = None, auto_classify: bool = True):
    """Generate SSE events for upload progress"""
    import tempfile
    
    temp_path = None
    classified_type = None
    try:
        # Step 1: Save file
        progress.update(1, "Saving uploaded file", {"filename": file.filename})
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        
        # Read file content (should already be available from wrapper)
        file_content = await file.read()
        
        if not file_content:
            raise ValueError("File is empty")
            
        file_size_mb = len(file_content) / (1024 * 1024)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            temp_path = tmp_file.name
            tmp_file.write(file_content)
            
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
        
        # Step 5: Determine target collection
        progress.update(5, "Determining collection", {"collection_selection": "in_progress"})
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        
        if collection_name:
            # Validate the provided collection exists
            collection_info = get_collection_config(collection_name)
            if not collection_info:
                raise HTTPException(status_code=400, detail=f"Collection '{collection_name}' not found")
            target_collection = collection_name
            progress.details["collection_method"] = "specified"
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
            progress.details["collection_method"] = "auto_classified"
            progress.details["classified_type"] = collection_type
            
            # Extract domain-specific metadata
            domain_metadata = classifier.extract_domain_metadata(sample_content, collection_type)
            for chunk in chunks:
                chunk.metadata.update(domain_metadata)
        else:
            # Use default collection
            target_collection = "default_knowledge"
            progress.details["collection_method"] = "default"
        
        progress.details["target_collection"] = target_collection
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        
        # Step 6: Check for duplicates
        progress.update(6, "Checking for duplicates", {"deduplication": "in_progress"})
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        
        # Compute file hash (using the already-read content)
        file_id = hashlib.sha256(file_content).hexdigest()[:12]
        
        # Initialize BM25 processor
        bm25_processor = BM25Processor()
        
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
                'collection_name': target_collection
            })
            
            # Add BM25 preprocessing
            bm25_metadata = bm25_processor.prepare_document_for_bm25(chunk.page_content, chunk.metadata)
            chunk.metadata.update(bm25_metadata)
        
        # Connect to vector DB and check duplicates
        vector_db_cfg = get_vector_db_settings()
        milvus_cfg = vector_db_cfg["milvus"]
        uri = milvus_cfg.get("MILVUS_URI")
        token = milvus_cfg.get("MILVUS_TOKEN")
        collection_name = target_collection  # Use the determined target collection
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
            progress.update(8, "Complete - All chunks are duplicates", {
                "status": "skipped",
                "reason": "All chunks already exist in database"
            })
            yield f"data: {json.dumps(progress.to_dict())}\n\n"
            return
        
        # Step 7: Generate embeddings
        progress.update(7, "Generating embeddings", {
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
        
        # Step 8: Insert into Milvus
        progress.update(8, "Inserting into vector database", {"insertion": "in_progress"})
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
        ]
        
        insert_result = collection.insert(data)
        collection.flush()
        
        # Final success
        progress.update(8, "Upload complete!", {
            "status": "success",
            "total_chunks": len(chunks),
            "unique_chunks_inserted": len(unique_chunks),
            "duplicates_filtered": duplicate_count,
            "collection": collection_name,
            "file_id": file_id,
            "classified_type": classified_type,
            "auto_classified": bool(classified_type)
        })
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        progress.error = error_msg
        progress.details["status"] = "error"
        progress.details["error_traceback"] = traceback.format_exc()
        print(f"[ERROR] Upload failed: {error_msg}")
        print(traceback.format_exc())
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        
    finally:
        # Cleanup
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"[INFO] Cleaned up temp file: {temp_path}")
            except Exception as e:
                print(f"[WARNING] Failed to remove temp file {temp_path}: {str(e)}")

@router.post("/upload_pdf_progress")
async def upload_pdf_with_progress(
    file: UploadFile = File(...),
    collection_name: Optional[str] = None,
    auto_classify: bool = True
):
    """Upload PDF with real-time progress updates via SSE"""
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
        
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    print(f"[INFO] Starting upload for file: {file.filename}")
    
    # Read the file content here before passing to generator
    try:
        file_content = await file.read()
        if not file_content:
            # Try reading using the file object directly
            file.file.seek(0)
            file_content = file.file.read()
            
        if not file_content:
            raise HTTPException(status_code=400, detail="File is empty")
            
    except Exception as e:
        print(f"[ERROR] Failed to read uploaded file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
    
    # Create a simple wrapper that provides the content
    class FileContentWrapper:
        def __init__(self, filename, content):
            self.filename = filename
            self._content = content
            
        async def read(self):
            return self._content
            
        async def seek(self, pos):
            pass
    
    wrapped_file = FileContentWrapper(file.filename, file_content)
    progress = UploadProgress()
    
    return StreamingResponse(
        progress_generator(wrapped_file, progress, collection_name, auto_classify),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable proxy buffering
        }
    )