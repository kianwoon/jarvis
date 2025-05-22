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
    if thinking:
        return (
            "Please show your reasoning step by step before giving the final answer.\n"
            + prompt
        )
    return prompt

def classify_rag_need(question: str, llm_cfg) -> str:
    print(f"[DEBUG] classify_rag_need: question = {question}")
    router_prompt = (
        "NO_THINK\nuser asking this question: '" + question + "'\n"
        "do we need to check internal company info repository?"
        "Just say 'YES' or 'NO'."
    )
    print(f"[DEBUG] classify_rag_need: router_prompt = {router_prompt}")
    prompt = build_prompt(router_prompt, is_internal=True)
    print(f"[DEBUG] classify_rag_need: prompt after build_prompt = {prompt}")
    llm_api_url = "http://localhost:8000/api/v1/generate_stream"
    mode = llm_cfg["non_thinking_mode"]
    payload = {
        "prompt": prompt,
        "temperature": mode.get("temperature", 0.99),
        "top_p": mode.get("top_p", 0.2),
        "max_tokens": 100
    }
    print(f"[DEBUG] classify_rag_need: payload = {payload}")
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
    print(f"[DEBUG] classify_rag_need: LLM raw output = {text}")
    # Detect <think>...</think>NO or <think>...</think>YES
    match = re.match(r"<think>.*?</think>\s*(NO|YES)$", text.strip(), re.DOTALL | re.IGNORECASE)
    if match:
        answer = match.group(1).upper()
        if answer == "NO":
            label = "NO RAG"
        elif answer == "YES":
            label = "RAG"
        else:
            label = "NO RAG"
    else:
        label = "NO RAG"
    print(f"[DEBUG] classify_rag_need: extracted label = {label}")
    return label

def rag_answer(question: str, thinking: bool = False, stream: bool = False):
    print(f"[DEBUG] rag_answer: incoming question = {question}")
    llm_cfg = get_llm_settings()
    required_fields = ["model", "thinking_mode", "non_thinking_mode", "max_tokens"]
    missing = [f for f in required_fields if f not in llm_cfg or llm_cfg[f] is None]
    if missing:
        raise RuntimeError(f"Missing required LLM config fields: {', '.join(missing)}")
    embedding_cfg = get_embedding_settings()
    vector_db_cfg = get_vector_db_settings()

    route = classify_rag_need(question, llm_cfg)
    print(f"[DEBUG] rag_answer: route = {route}")
    context = ""
    reasoning = None
    answer = ""
    raw = ""
    source = "LLM"

    SIMILARITY_THRESHOLD = 0.7

    if route == "RAG":
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
        docs = milvus_store.similarity_search_with_score(question, k=4) if hasattr(milvus_store, 'similarity_search_with_score') else [(doc, 1.0) for doc in milvus_store.similarity_search(question, k=4)]
        print(f"[DEBUG] rag_answer: docs = {docs}")
        filtered_docs = [doc for doc, score in docs if score >= SIMILARITY_THRESHOLD]
        print(f"[DEBUG] rag_answer: filtered_docs = {filtered_docs}")
        context = "\n\n".join([doc.page_content for doc in filtered_docs])
        print(f"[DEBUG] rag_answer: context = {context}")
        if context.strip():
            user_prompt = (
                "The following context is from our internal corporate knowledge base (business, data, internal docs). If the context is not relevant to the question, answer based on your own knowledge.\n\n"
                f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
            )
            prompt = build_prompt(user_prompt, thinking=thinking)
            source = "RAG"
        else:
            prompt = build_prompt(question, thinking=thinking)
            source = "LLM"
    else:
        prompt = build_prompt(question, thinking=thinking)
        source = "LLM"
    print(f"[DEBUG] rag_answer: final prompt = {prompt}")

    # 3. Call internal LLM API using streaming
    llm_api_url = "http://localhost:8000/api/v1/generate_stream"
    mode = llm_cfg["thinking_mode"] if thinking else llm_cfg["non_thinking_mode"]
    payload = {
        "prompt": prompt,
        "temperature": mode.get("temperature", 0.7),
        "top_p": mode.get("top_p", 1.0),
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