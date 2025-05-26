from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List
import requests
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

router = APIRouter()

class DocumentRequest(BaseModel):
    topic: str
    doc_id: str
    metadata: Optional[dict] = None

# Custom embedding function for HTTP endpoint
class HTTPEndeddingFunction:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            payload = {"texts": [text]}
            resp = requests.post("http://qwen-embedder:8050/embed", json=payload)
            if resp.status_code == 422:
                print("Response content:", resp.content)
            resp.raise_for_status()
            embeddings.append(resp.json()["embeddings"][0])
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
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
async def upload_pdf(file: UploadFile = File(...)):
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
        embeddings = HTTPEndeddingFunction(embedding_endpoint)
        print("✅ Using HTTP embedding endpoint")
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        print(f"✅ Using HuggingFace embeddings: {embedding_model}")
    
    # 5. Enhanced metadata handling with proper page numbers
    # Compute file hash for doc_id uniqueness
    with open(temp_path, "rb") as f:
        file_content = f.read()
        file_id = hashlib.sha256(file_content).hexdigest()[:12]  # Shorter hash
    
    print(f"🔑 File ID: {file_id}")
    
    # Enhanced metadata setting - preserve original page numbers from PyPDFLoader
    for i, chunk in enumerate(chunks):
        if not hasattr(chunk, 'metadata') or chunk.metadata is None:
            chunk.metadata = {}
        
        # IMPORTANT: Preserve original page number from PyPDFLoader - don't override with 0
        original_page = chunk.metadata.get('page', 0)
        
        chunk.metadata.update({
            'source': file.filename,
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
    
    # 6. Vector DB storage with enhanced deduplication logic
    vector_db_cfg = get_vector_db_settings()
    milvus_status = vector_db_cfg.get("milvus", {}).get("status", False)
    qdrant_status = vector_db_cfg.get("qdrant", {}).get("status", False)
    
    if milvus_status:
        print("🔄 Using Milvus vector database")
        milvus_cfg = vector_db_cfg["milvus"]
        uri = milvus_cfg.get("MILVUS_URI")
        token = milvus_cfg.get("MILVUS_TOKEN")
        collection = milvus_cfg.get("MILVUS_DEFAULT_COLLECTION", "default_knowledge")
        vector_dim = int(milvus_cfg.get("dimension", 1536))
        
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
        
        return {
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
    
    elif qdrant_status:
        print("🔄 Using Qdrant vector database")
        qdrant_cfg = vector_db_cfg["qdrant"]
        host = qdrant_cfg.get("QDRANT_HOST", "localhost")
        port = qdrant_cfg.get("QDRANT_PORT", 6333)
        collection = qdrant_cfg.get("QDRANT_COLLECTION", "default_knowledge")
        
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
        
        return {
            "status": "success", 
            "chunks": len(chunks),
            "filename": file.filename,
            "pages_processed": len(docs),
            "vector_db": "qdrant"
        }
    
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
        embeddings = HTTPEndeddingFunction(embedding_endpoint)
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
        if hasattr(milvus_store, 'similarity_search_with_score'):
            docs = milvus_store.similarity_search_with_score(query, k=limit * 3)  # Get 3x more to filter
        else:
            docs_without_score = milvus_store.similarity_search(query, k=limit * 3)
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