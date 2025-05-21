from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List
import requests
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from uuid import uuid4
from datetime import datetime
import hashlib

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
            print("Request payload:", payload)
            resp = requests.post("http://qwen-embedder:8050/embed", json=payload)
            if resp.status_code == 422:
                print("Response content:", resp.content)
            resp.raise_for_status()
            embeddings.append(resp.json()["embeddings"][0])
        return embeddings
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

def ensure_milvus_collection(collection_name: str, vector_dim: int, uri: str, token: str):
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
    # 1. Save file temporarily
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    # 2. Load PDF
    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        print(f"Number of docs loaded from PDF (PyPDFLoader): {len(docs)}")
        # Check if all docs are empty, fallback to UnstructuredPDFLoader with OCR
        if all(not doc.page_content.strip() for doc in docs):
            print("All docs are empty. Falling back to UnstructuredPDFLoader with OCR.")
            from langchain_community.document_loaders import UnstructuredPDFLoader
            loader = UnstructuredPDFLoader(temp_path, mode="elements", strategy="ocr_only")
            docs = loader.load()
            print(f"Number of docs loaded from PDF (UnstructuredPDFLoader OCR): {len(docs)}")
    except Exception as e:
        print(f"Error loading PDF: {e}")
        raise
    # 3. Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"Number of chunks created: {len(chunks)}")
    # Fallback: treat short docs as single chunks if chunking produced zero chunks
    if len(chunks) == 0:
        print("No chunks created by splitter. Falling back to treating each doc as a single chunk.")
        from langchain.schema import Document
        chunks = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in docs if doc.page_content.strip()]
        print(f"Number of fallback single-chunks created: {len(chunks)}")
    # 4. Get embedding config
    embedding_cfg = get_embedding_settings()
    embedding_model = embedding_cfg.get('embedding_model')
    embedding_endpoint = embedding_cfg.get('embedding_endpoint')
    # Use HTTP endpoint if set, otherwise fallback to HuggingFaceEmbeddings
    if embedding_endpoint:
        embeddings = HTTPEndeddingFunction(embedding_endpoint)
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    # 5. Get vector DB config
    vector_db_cfg = get_vector_db_settings()
    milvus_status = vector_db_cfg.get("milvus", {}).get("status", False)
    qdrant_status = vector_db_cfg.get("qdrant", {}).get("status", False)
    # 6. Store in vector DB
    if milvus_status:
        milvus_cfg = vector_db_cfg["milvus"]
        uri = milvus_cfg.get("MILVUS_URI")
        token = milvus_cfg.get("MILVUS_TOKEN")
        collection = milvus_cfg.get("MILVUS_DEFAULT_COLLECTION", "default_knowledge")
        vector_dim = int(milvus_cfg.get("dimension", 1536))
        # Ensure collection exists with correct schema (configurable dim)
        ensure_milvus_collection(collection, vector_dim=vector_dim, uri=uri, token=token)
        from pymilvus import Collection
        # Compute file hash for doc_id uniqueness
        with open(temp_path, "rb") as f:
            file_id = hashlib.sha256(f.read()).hexdigest()
        ids = [str(uuid4()) for _ in chunks]
        texts = [chunk.page_content for chunk in chunks]
        # Set hash and doc_id for each chunk, always update chunk.metadata in-place
        for i, chunk in enumerate(chunks):
            if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata.setdefault('source', file.filename)
            chunk.metadata.setdefault('page', 0)
            chunk.metadata.setdefault('doc_type', 'pdf')
            chunk.metadata.setdefault('uploaded_at', datetime.now().isoformat())
            chunk.metadata.setdefault('section', '')
            chunk.metadata.setdefault('author', '')
            # Add hash and robust doc_id
            chunk_hash = hash_text(chunk.page_content)
            chunk.metadata['hash'] = chunk_hash
            chunk.metadata['doc_id'] = f"{file_id}_page_{chunk.metadata['page']}"
            # Comprehensive debug log for each chunk
            print(f"Chunk {i} content: {repr(chunk.page_content)[:200]}")
            print(f"Chunk {i} hash: {chunk.metadata['hash']}")
            print(f"Chunk {i} doc_id: {chunk.metadata['doc_id']}")
            print(f"Chunk {i} metadata: {chunk.metadata}")
        # Deduplication: filter out chunks with existing hash or doc_id
        collection_obj = Collection(collection)
        collection_obj.load()
        existing_hashes = get_existing_hashes(collection_obj)
        existing_doc_ids = get_existing_doc_ids(collection_obj)
        print(f"Total existing hashes: {len(existing_hashes)}")
        print(f"Total existing doc_ids: {len(existing_doc_ids)}")
        print(f"Existing hashes in collection: {list(existing_hashes)[:5]}... (total {len(existing_hashes)})")
        print(f"Existing doc_ids in collection: {list(existing_doc_ids)[:5]}... (total {len(existing_doc_ids)})")
        unique_chunks = filter_new_chunks(chunks, existing_hashes, existing_doc_ids)
        print(f"Unique chunks to insert: {len(unique_chunks)}")
        if not unique_chunks:
            return {"status": "skipped", "reason": "All chunks are duplicates."}
        # Prepare data for insertion
        unique_ids = [str(uuid4()) for _ in unique_chunks]
        unique_texts = [chunk.page_content for chunk in unique_chunks]
        embeddings_list = embeddings.embed_documents(unique_texts)
        id_list = []
        vector_list = []
        content_list = []
        source_list = []
        page_list = []
        doc_type_list = []
        uploaded_at_list = []
        section_list = []
        author_list = []
        hash_list = []
        doc_id_list = []
        for i, chunk in enumerate(unique_chunks):
            meta = chunk.metadata if hasattr(chunk, 'metadata') else {}
            id_list.append(unique_ids[i])
            vector_list.append(embeddings_list[i])
            content_list.append(chunk.page_content)
            source_list.append(meta.get('source', ''))
            page_list.append(meta.get('page', 0))
            doc_type_list.append(meta.get('doc_type', 'pdf'))
            uploaded_at_list.append(meta.get('uploaded_at', ''))
            section_list.append(meta.get('section', ''))
            author_list.append(meta.get('author', ''))
            hash_list.append(meta.get('hash', ''))
            doc_id_list.append(meta.get('doc_id', ''))
        data = [
            id_list,
            vector_list,
            content_list,
            source_list,
            page_list,
            doc_type_list,
            uploaded_at_list,
            section_list,
            author_list,
            hash_list,
            doc_id_list,
        ]
        collection_obj.insert(data)
    elif qdrant_status:
        qdrant_cfg = vector_db_cfg["qdrant"]
        host = qdrant_cfg.get("QDRANT_HOST", "localhost")
        port = qdrant_cfg.get("QDRANT_PORT", 6333)
        collection = qdrant_cfg.get("QDRANT_COLLECTION", "default_knowledge")
        vector_store = Qdrant(
            embedding_function=embeddings,
            collection_name=collection,
            url=f"http://{host}:{port}"
        )
        vector_store.add_documents(chunks)
    else:
        raise HTTPException(status_code=400, detail="No active vector DB.")
    return {"status": "success", "chunks": len(chunks)} 