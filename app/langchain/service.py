import requests
import re
import httpx
import json
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


def build_prompt(prompt: str, thinking: bool = False, is_internal: bool = False) -> str:
    """Helper to prepend /no_think to prompt if needed."""
    if is_internal or not thinking:
        return f"/no_think\n{prompt}"
    return prompt

def classify_rag_need(question: str, llm_cfg) -> str:
    """Use LLM to classify if a query needs RAG."""
    router_prompt = (
        "You are an expert AI assistant. Only route to RAG if the question is about our company, internal data, business processes, proprietary knowledge, or client-specific information. For general knowledge, technology, or public information questions, use the LLM only.\n"
        "Label as:\n"
        "- 'RAG' if the question is about our business, internal data, company policies, client SOWs, or proprietary info.\n"
        "- 'NO_RAG' if the question is about general knowledge, technology, public facts, or anything not specific to our company.\n"
        "\nExamples:\n"
        "Q: What is the capital of France?\nA: NO_RAG\n"
        "Q: Summarize the latest company quarterly report.\nA: RAG\n"
        "Q: Tell me a joke.\nA: NO_RAG\n"
        "Q: What are the main points from the document I uploaded?\nA: RAG\n"
        "Q: Who won the FIFA World Cup in 2018?\nA: NO_RAG\n"
        "Q: What are the recent updates in EU data privacy laws?\nA: NO_RAG\n"
        "Q: What is our client OCBC's SOW for 2024?\nA: RAG\n"
        "Q: Compare OceanBase and MySQL.\nA: NO_RAG\n"
        "Q: What is our internal process for onboarding new employees?\nA: RAG\n"
        "\nNow, decide for this query:\n"
        f"Query: \"{question}\"\nAnswer with only: 'RAG' or 'NO_RAG'"
    )
    prompt = build_prompt(router_prompt, is_internal=True)
    llm_api_url = "http://localhost:8000/api/v1/generate_stream"
    payload = {
        "prompt": prompt,
        "temperature": llm_cfg.get("temperature", 0.0),
        "top_p": 1.0,
        "max_tokens": 10
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
    label = text.strip().upper()
    return label if label in ("RAG", "NO_RAG") else "RAG"

def rag_answer(question: str, thinking: bool = False, stream: bool = False):
    """
    Retrieve relevant context from the vector DB and generate an answer using the LLM.
    If stream=True, yield tokens as they arrive from the LLM (for StreamingResponse).
    If stream=False, return a dict with 'answer', 'context', and 'source'.
    """
    # 1. Load configs
    llm_cfg = get_llm_settings()
    embedding_cfg = get_embedding_settings()
    vector_db_cfg = get_vector_db_settings()

    # 2. Use LLM to classify if RAG is needed
    route = classify_rag_need(question, llm_cfg)
    print(f"[RAG ROUTER] Route decision: {route}")
    context = ""
    reasoning = None
    answer = ""
    raw = ""
    source = "LLM"

    SIMILARITY_THRESHOLD = 0.7  # Only use context above this threshold

    if route == "RAG":
        # --- RAG pipeline ---
        embedding_endpoint = embedding_cfg.get("embedding_endpoint")
        if embedding_endpoint:
            embeddings = HTTPEndeddingFunction(embedding_endpoint)
        else:
            embeddings = HuggingFaceEmbeddings(model_name=embedding_cfg["embedding_model"])
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
        # Retrieve docs with similarity scores if possible
        docs = milvus_store.similarity_search_with_score(question, k=4) if hasattr(milvus_store, 'similarity_search_with_score') else [(doc, 1.0) for doc in milvus_store.similarity_search(question, k=4)]
        filtered_docs = [doc for doc, score in docs if score >= SIMILARITY_THRESHOLD]
        context = "\n\n".join([doc.page_content for doc in filtered_docs])
        if context.strip():
            user_prompt = (
                "The following context is from our internal corporate knowledge base (business, data, internal docs). If the context is not relevant to the question, answer based on your own knowledge.\n\n"
                f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
            )
            prompt = build_prompt(user_prompt, thinking=thinking)
            source = "RAG"
        else:
            # Fallback to LLM if no context found
            prompt = build_prompt(question, thinking=thinking)
            source = "LLM"
    else:
        # --- Direct LLM pipeline ---
        prompt = build_prompt(question, thinking=thinking)
        source = "LLM"
    print(f"[RAG ROUTER] Prompt sent to LLM:\n{prompt}")

    # 3. Call internal LLM API using streaming
    llm_api_url = "http://localhost:8000/api/v1/generate_stream"
    payload = {
        "prompt": prompt,
        "temperature": llm_cfg.get("temperature", 0.7),
        "top_p": llm_cfg.get("top_p", 1.0),
        "max_tokens": llm_cfg.get("max_tokens", 2048)
    }
    if stream:
        def token_stream():
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
                            yield json.dumps({"token": token}) + "\n"
            # After streaming all tokens, yield the final metadata
            reasoning = re.findall(r"<think>(.*?)</think>", text, re.DOTALL)
            answer = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            yield json.dumps({
                "answer": answer,
                "source": source,
                "context": context,
                "reasoning": reasoning[0] if reasoning else None
            }) + "\n"
        return token_stream()
    else:
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
        print(f"[RAG ROUTER] Final answer: {answer}")
        return {"answer": answer, "context": context, "reasoning": reasoning[0] if reasoning else None, "raw": text, "route": route, "source": source} 