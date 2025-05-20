from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List
import requests
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from uuid import uuid4
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus, Qdrant
from app.core.embedding_settings_cache import get_embedding_settings
from app.core.vector_db_settings_cache import get_vector_db_settings

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
            payload = {"text": text}
            print("Request payload:", payload)
            resp = requests.post("http://qwen-embedder:8050/embed", json=payload)
            if resp.status_code == 422:
                print("Response content:", resp.content)
            resp.raise_for_status()
            embeddings.append(resp.json()["embedding"])
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
        ]
        schema = CollectionSchema(fields, description="Knowledge base with metadata")
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
    loader = PyPDFLoader(temp_path)
    docs = loader.load()
    # 3. Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
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
        ids = [str(uuid4()) for _ in chunks]
        texts = [chunk.page_content for chunk in chunks]
        embeddings_list = embeddings.embed_documents(texts)
        id_list = []
        vector_list = []
        content_list = []
        source_list = []
        page_list = []
        doc_type_list = []
        uploaded_at_list = []
        section_list = []
        author_list = []
        for i, chunk in enumerate(chunks):
            meta = chunk.metadata if hasattr(chunk, 'metadata') else {}
            meta.setdefault('source', file.filename)
            meta.setdefault('page', 0)
            meta.setdefault('doc_type', 'pdf')
            meta.setdefault('uploaded_at', datetime.now().isoformat())
            meta.setdefault('section', '')
            meta.setdefault('author', '')
            id_list.append(ids[i])
            vector_list.append(embeddings_list[i])
            content_list.append(chunk.page_content)
            source_list.append(meta.get('source', ''))
            page_list.append(meta.get('page', 0))
            doc_type_list.append(meta.get('doc_type', 'pdf'))
            uploaded_at_list.append(meta.get('uploaded_at', ''))
            section_list.append(meta.get('section', ''))
            author_list.append(meta.get('author', ''))
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
        ]
        collection_obj = Collection(collection)
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