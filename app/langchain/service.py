import requests
import re
import httpx
from app.core.llm_settings_cache import get_llm_settings
from app.core.embedding_settings_cache import get_embedding_settings
from app.core.vector_db_settings_cache import get_vector_db_settings
from app.api.v1.endpoints.document import HTTPEndeddingFunction
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus

# Import LangChain classes (replace with actual imports for your environment)
# from langchain_community.llms import Qwen2
# from langchain_community.vectorstores import Qdrant

# Placeholder for actual LangChain imports
Qwen2 = object  # Replace with: from langchain_community.llms import Qwen2
Qdrant = object  # Replace with: from langchain_community.vectorstores import Qdrant


def rag_answer(question: str) -> dict:
    """
    Retrieve relevant context from the vector DB and generate an answer using the LLM.
    Returns a dict with 'answer' and 'context'.
    """
    # 1. Load configs
    llm_cfg = get_llm_settings()
    embedding_cfg = get_embedding_settings()
    vector_db_cfg = get_vector_db_settings()

    # 2. Instantiate Embeddings
    embedding_model = embedding_cfg.get("embedding_model")
    embedding_endpoint = embedding_cfg.get("embedding_endpoint")
    if embedding_endpoint:
        embeddings = HTTPEndeddingFunction(embedding_endpoint)
        embed_query = embeddings.embed_query
    else:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        embed_query = embeddings.embed_query

    # 3. Instantiate Milvus Vector Store
    milvus_cfg = vector_db_cfg.get("milvus", {})
    collection = milvus_cfg.get("MILVUS_DEFAULT_COLLECTION", "default_knowledge")
    uri = milvus_cfg.get("MILVUS_URI")
    token = milvus_cfg.get("MILVUS_TOKEN")
    vector_dim = int(milvus_cfg.get("dimension", 1536))
    milvus_store = Milvus(
        embedding_function=embeddings,
        collection_name=collection,
        connection_args={"uri": uri, "token": token},
        text_field="content"
    )

    # 4. Retrieve relevant documents
    docs = milvus_store.similarity_search(question, k=4)
    context = '\n'.join([doc.page_content for doc in docs])

    # 5. Construct prompt for LLM
    prompt = f"""Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"""

    # 6. Call internal LLM API using streaming
    llm_api_url = "http://localhost:8000/api/v1/generate_stream"
    payload = {
        "prompt": prompt,
        "temperature": llm_cfg.get("temperature", 0.7),
        "top_p": llm_cfg.get("top_p", 1.0),
        "max_tokens": llm_cfg.get("max_tokens", 2048)
    }
    text = ""
    with httpx.Client(timeout=None) as client:
        with client.stream("POST", llm_api_url, json=payload) as response:
            for line in response.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                if line.startswith("data: "):
                    token = line.replace("data: ", "")
                    text += token
    reasoning = re.findall(r"<think>(.*?)</think>", text, re.DOTALL)
    answer = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return {"answer": answer, "context": context, "reasoning": reasoning[0] if reasoning else None, "raw": text} 