"""
LangGraph-based RAG implementation with context management and Redis persistence
"""
import json
import re
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime
from langgraph.graph import Graph, StateGraph, END
from langgraph.checkpoint.redis import RedisSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.core.llm_settings_cache import get_llm_settings
from app.core.embedding_settings_cache import get_embedding_settings
from app.core.vector_db_settings_cache import get_vector_db_settings
from app.core.mcp_tools_cache import get_enabled_mcp_tools
from app.api.v1.endpoints.document import HTTPEmbeddingFunction
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# Redis configuration for conversation persistence
from app.core.redis_client import get_redis_client_for_langgraph, get_redis_client
CONVERSATION_TTL = 3600 * 24  # 24 hours

# Context window limits
MAX_CONTEXT_CHARS = 32000  # Conservative limit for most LLMs
MAX_DOCS_PER_RETRIEVAL = 10
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

class RAGState(TypedDict):
    """State for the RAG workflow"""
    query: str
    conversation_id: str
    thinking: bool
    messages: List[BaseMessage]
    documents: List[Dict[str, Any]]
    compressed_context: str
    tool_calls: List[Dict[str, Any]]
    classification: str
    final_answer: str
    error: Optional[str]
    metadata: Dict[str, Any]

class ContextCompressor:
    """Handles context compression and summarization"""
    
    def __init__(self, llm_cfg: Dict[str, Any]):
        self.llm_cfg = llm_cfg
        self.llm_api_url = "http://localhost:8000/api/v1/generate_stream"
    
    def compress_documents(self, documents: List[Dict[str, Any]], query: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
        """Compress documents to fit within context window"""
        if not documents:
            return ""
        
        # First, try to fit all documents
        full_context = "\n\n".join([doc.get("content", "") for doc in documents])
        if len(full_context) <= max_chars:
            return full_context
        
        # If too large, use progressive summarization
        return self._progressive_summarization(documents, query, max_chars)
    
    def _progressive_summarization(self, documents: List[Dict[str, Any]], query: str, max_chars: int) -> str:
        """Progressively summarize documents to fit context window"""
        # Group documents into chunks that can be summarized together
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_limit = max_chars // 3  # Process in thirds
        
        for doc in documents:
            doc_content = doc.get("content", "")
            doc_size = len(doc_content)
            
            if current_size + doc_size > chunk_limit and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [doc]
                current_size = doc_size
            else:
                current_chunk.append(doc)
                current_size += doc_size
        
        if current_chunk:
            chunks.append(current_chunk)
        
        # Summarize each chunk in parallel
        summaries = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_chunk = {
                executor.submit(self._summarize_chunk, chunk, query): i
                for i, chunk in enumerate(chunks)
            }
            
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    summary = future.result()
                    summaries.append((chunk_idx, summary))
                except Exception as e:
                    print(f"[ERROR] Failed to summarize chunk {chunk_idx}: {str(e)}")
        
        # Sort summaries by original order
        summaries.sort(key=lambda x: x[0])
        compressed = "\n\n".join([s[1] for _, s in summaries])
        
        # If still too large, do a final compression
        if len(compressed) > max_chars:
            compressed = self._final_compression(compressed, query, max_chars)
        
        return compressed
    
    def _summarize_chunk(self, documents: List[Dict[str, Any]], query: str) -> str:
        """Summarize a chunk of documents"""
        combined_text = "\n\n".join([doc.get("content", "") for doc in documents])
        
        prompt = f"""Summarize the following documents while preserving information relevant to this query: "{query}"

Documents:
{combined_text}

Provide a concise summary that retains all information relevant to answering the query. Focus on facts, data, and specific details."""

        mode = self.llm_cfg["non_thinking_mode"]
        payload = {
            "prompt": prompt,
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 1000
        }
        
        text = ""
        with httpx.Client(timeout=60) as client:
            with client.stream("POST", self.llm_api_url, json=payload) as response:
                for line in response.iter_lines():
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        line = line.decode("utf-8")
                    if line.startswith("data: "):
                        token = line.replace("data: ", "")
                        text += token
        
        return text.strip()
    
    def _final_compression(self, text: str, query: str, max_chars: int) -> str:
        """Final compression if needed"""
        prompt = f"""The following text needs to be compressed to under {max_chars} characters while preserving all information relevant to this query: "{query}"

Text:
{text}

Provide a compressed version that retains all critical information for answering the query."""

        mode = self.llm_cfg["non_thinking_mode"]
        payload = {
            "prompt": prompt,
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": max_chars // 4  # Approximate token count
        }
        
        compressed = ""
        with httpx.Client(timeout=60) as client:
            with client.stream("POST", self.llm_api_url, json=payload) as response:
                for line in response.iter_lines():
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        line = line.decode("utf-8")
                    if line.startswith("data: "):
                        token = line.replace("data: ", "")
                        compressed += token
        
        return compressed.strip()

class ConversationMemory:
    """Manages conversation history with Redis"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.max_messages = 20  # Keep last 20 messages
    
    def get_conversation(self, conversation_id: str) -> List[BaseMessage]:
        """Retrieve conversation history"""
        key = f"conversation:{conversation_id}"
        messages_json = self.redis_client.lrange(key, 0, -1)
        
        messages = []
        for msg_json in messages_json:
            msg_data = json.loads(msg_json)
            if msg_data["type"] == "human":
                messages.append(HumanMessage(content=msg_data["content"]))
            elif msg_data["type"] == "ai":
                messages.append(AIMessage(content=msg_data["content"]))
            elif msg_data["type"] == "system":
                messages.append(SystemMessage(content=msg_data["content"]))
        
        return messages
    
    def add_message(self, conversation_id: str, message: BaseMessage):
        """Add message to conversation history"""
        key = f"conversation:{conversation_id}"
        
        msg_data = {
            "type": message.__class__.__name__.lower().replace("message", ""),
            "content": message.content,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.redis_client.rpush(key, json.dumps(msg_data))
        
        # Trim to max messages
        self.redis_client.ltrim(key, -self.max_messages, -1)
        
        # Set TTL
        self.redis_client.expire(key, CONVERSATION_TTL)
    
    def get_summary(self, conversation_id: str) -> Optional[str]:
        """Get conversation summary if exists"""
        key = f"conversation_summary:{conversation_id}"
        return self.redis_client.get(key)
    
    def set_summary(self, conversation_id: str, summary: str):
        """Store conversation summary"""
        key = f"conversation_summary:{conversation_id}"
        self.redis_client.setex(key, CONVERSATION_TTL, summary)

def create_rag_graph() -> StateGraph:
    """Create the LangGraph RAG workflow"""
    
    # Initialize Redis for checkpointing using pooled connection
    redis_for_checkpointer = get_redis_client_for_langgraph()
    redis_client = get_redis_client(decode_responses=True)
    
    # Create RedisSaver with pooled connection
    if redis_for_checkpointer:
        try:
            checkpointer = RedisSaver(redis_for_checkpointer)
            print("[INFO] RedisSaver initialized with pooled Redis connection")
        except Exception as e:
            print(f"[WARNING] RedisSaver initialization failed: {e}, disabling checkpointing")
            checkpointer = None
    else:
        print("[WARNING] Redis connection pool not available, disabling checkpointing")
        checkpointer = None
    
    # Initialize conversation memory
    memory = ConversationMemory(redis_client)
    
    # Create workflow
    workflow = StateGraph(RAGState)
    
    # Define nodes
    workflow.add_node("classify_query", classify_query_node)
    workflow.add_node("retrieve_documents", retrieve_documents_node)
    workflow.add_node("compress_context", compress_context_node)
    workflow.add_node("execute_tools", execute_tools_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.add_node("update_memory", update_memory_node)
    
    # Define edges
    workflow.set_entry_point("classify_query")
    
    # Conditional routing based on classification
    workflow.add_conditional_edges(
        "classify_query",
        route_by_classification,
        {
            "RAG": "retrieve_documents",
            "TOOLS": "execute_tools",
            "LLM": "generate_response",
            "RAG+TOOLS": "retrieve_documents"
        }
    )
    
    # RAG path
    workflow.add_edge("retrieve_documents", "compress_context")
    workflow.add_edge("compress_context", "generate_response")
    
    # Tools path
    workflow.add_conditional_edges(
        "execute_tools",
        check_needs_rag_after_tools,
        {
            "generate": "generate_response",
            "retrieve": "retrieve_documents"
        }
    )
    
    # Final steps
    workflow.add_edge("generate_response", "update_memory")
    workflow.add_edge("update_memory", END)
    
    # Compile with checkpointer
    return workflow.compile(checkpointer=checkpointer)

def classify_query_node(state: RAGState) -> RAGState:
    """Classify the query type"""
    from app.langchain.service import classify_query_type
    
    llm_cfg = get_llm_settings()
    classification = classify_query_type(state["query"], llm_cfg)
    
    # Check if both RAG and TOOLS might be needed
    query_lower = state["query"].lower()
    available_tools = get_enabled_mcp_tools()
    
    needs_tools = False
    if available_tools:
        tool_keywords = ["time", "date", "email", "send", "calculate", "weather"]
        needs_tools = any(kw in query_lower for kw in tool_keywords)
    
    needs_rag = any(kw in query_lower for kw in ["our", "company", "document", "policy"]) or \
                classification == "RAG"
    
    if needs_tools and needs_rag:
        state["classification"] = "RAG+TOOLS"
    else:
        state["classification"] = classification
    
    state["metadata"]["classification_time"] = datetime.utcnow().isoformat()
    
    return state

def retrieve_documents_node(state: RAGState) -> RAGState:
    """Retrieve relevant documents from vector store"""
    embedding_cfg = get_embedding_settings()
    vector_db_cfg = get_vector_db_settings()
    
    # Set up embeddings
    embedding_endpoint = embedding_cfg.get("embedding_endpoint")
    if embedding_endpoint:
        embeddings = HTTPEmbeddingFunction(embedding_endpoint)
    else:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_cfg["embedding_model"])
    
    # Connect to Milvus
    milvus_cfg = vector_db_cfg["milvus"]
    milvus_store = Milvus(
        embedding_function=embeddings,
        collection_name=milvus_cfg.get("MILVUS_DEFAULT_COLLECTION", "default_knowledge"),
        connection_args={
            "uri": milvus_cfg.get("MILVUS_URI"),
            "token": milvus_cfg.get("MILVUS_TOKEN")
        },
        text_field="content"
    )
    
    try:
        # Retrieve documents
        docs_with_scores = milvus_store.similarity_search_with_score(
            state["query"], 
            k=MAX_DOCS_PER_RETRIEVAL
        )
        
        # Convert to our format
        documents = []
        for doc, score in docs_with_scores:
            documents.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
                "id": hashlib.md5(doc.page_content.encode()).hexdigest()[:8]
            })
        
        state["documents"] = documents
        state["metadata"]["retrieval_count"] = len(documents)
        
    except Exception as e:
        state["error"] = f"Document retrieval failed: {str(e)}"
        state["documents"] = []
    
    return state

def compress_context_node(state: RAGState) -> RAGState:
    """Compress retrieved documents to fit context window"""
    if not state.get("documents"):
        state["compressed_context"] = ""
        return state
    
    llm_cfg = get_llm_settings()
    compressor = ContextCompressor(llm_cfg)
    
    # Calculate available space considering conversation history
    messages_chars = sum(len(msg.content) for msg in state.get("messages", []))
    available_chars = MAX_CONTEXT_CHARS - messages_chars - 2000  # Reserve space for response
    
    compressed = compressor.compress_documents(
        state["documents"], 
        state["query"],
        max_chars=available_chars
    )
    
    state["compressed_context"] = compressed
    state["metadata"]["compression_ratio"] = len(compressed) / sum(
        len(doc["content"]) for doc in state["documents"]
    ) if state["documents"] else 1.0
    
    return state

def execute_tools_node(state: RAGState) -> RAGState:
    """Execute relevant tools"""
    from app.langchain.service import execute_tools_first
    
    tool_results, updated_question, tool_context = execute_tools_first(
        state["query"], 
        state.get("thinking", False)
    )
    
    state["tool_calls"] = tool_results
    state["metadata"]["tool_context"] = tool_context
    
    return state

def generate_response_node(state: RAGState) -> RAGState:
    """Generate final response using all available context"""
    llm_cfg = get_llm_settings()
    
    # Build comprehensive prompt
    prompt_parts = []
    
    # Add conversation history if exists
    if state.get("messages"):
        prompt_parts.append("Previous conversation:")
        for msg in state["messages"][-6:]:  # Last 6 messages
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            prompt_parts.append(f"{role}: {msg.content}")
        prompt_parts.append("")
    
    # Add compressed context
    if state.get("compressed_context"):
        prompt_parts.append("Relevant information from knowledge base:")
        prompt_parts.append(state["compressed_context"])
        prompt_parts.append("")
    
    # Add tool results
    if state.get("tool_calls"):
        prompt_parts.append("Tool execution results:")
        prompt_parts.append(state["metadata"].get("tool_context", ""))
        prompt_parts.append("")
    
    # Add the question
    prompt_parts.append(f"Current question: {state['query']}")
    prompt_parts.append("")
    prompt_parts.append("Please provide a comprehensive answer based on all the information above.")
    
    full_prompt = "\n".join(prompt_parts)
    
    # Generate response
    llm_api_url = "http://localhost:8000/api/v1/generate_stream"
    mode = llm_cfg["thinking_mode"] if state.get("thinking") else llm_cfg["non_thinking_mode"]
    
    payload = {
        "prompt": full_prompt,
        "temperature": mode.get("temperature", 0.7),
        "top_p": mode.get("top_p", 1.0),
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
    
    # Clean response
    answer = text.strip()
    state["final_answer"] = answer
    
    return state

def update_memory_node(state: RAGState) -> RAGState:
    """Update conversation memory"""
    redis_client = get_redis_client(decode_responses=True)
    if redis_client:
        memory = ConversationMemory(redis_client)
    else:
        return state  # Skip memory update if Redis not available
    
    # Add messages to history
    memory.add_message(state["conversation_id"], HumanMessage(content=state["query"]))
    memory.add_message(state["conversation_id"], AIMessage(content=state["final_answer"]))
    
    # Update summary if conversation is getting long
    messages = memory.get_conversation(state["conversation_id"])
    if len(messages) > 10:
        # Generate summary of older messages
        older_messages = messages[:-10]
        summary_prompt = "Summarize the key points from this conversation:\n"
        for msg in older_messages:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            summary_prompt += f"{role}: {msg.content}\n"
        
        # TODO: Generate summary using LLM
        # For now, just store a placeholder
        memory.set_summary(state["conversation_id"], "Conversation summary pending...")
    
    return state

def route_by_classification(state: RAGState) -> str:
    """Route based on query classification"""
    return state["classification"]

def check_needs_rag_after_tools(state: RAGState) -> str:
    """Check if we need RAG after tool execution"""
    # If original classification was RAG+TOOLS, go to retrieve
    if state["classification"] == "RAG+TOOLS":
        return "retrieve"
    return "generate"

# Export main function
def enhanced_rag_answer(
    question: str, 
    conversation_id: Optional[str] = None,
    thinking: bool = False,
    stream: bool = False
) -> Dict[str, Any]:
    """
    Enhanced RAG answer with LangGraph workflow
    
    Args:
        question: User query
        conversation_id: Optional conversation ID for history
        thinking: Whether to use thinking mode
        stream: Whether to stream response
        
    Returns:
        Dict with answer and metadata
    """
    import uuid
    
    # Generate conversation ID if not provided
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    # Initialize state
    initial_state = RAGState(
        query=question,
        conversation_id=conversation_id,
        thinking=thinking,
        messages=[],
        documents=[],
        compressed_context="",
        tool_calls=[],
        classification="",
        final_answer="",
        error=None,
        metadata={}
    )
    
    # Get conversation history
    redis_client = get_redis_client(decode_responses=True)
    if redis_client:
        memory = ConversationMemory(redis_client)
        initial_state["messages"] = memory.get_conversation(conversation_id)
    else:
        initial_state["messages"] = []  # Empty conversation history if Redis not available
    
    # Create and run workflow
    app = create_rag_graph()
    
    if stream:
        # For streaming, we need to handle differently
        # This is a simplified version - you might want to enhance this
        config = {"configurable": {"thread_id": conversation_id}}
        
        for output in app.stream(initial_state, config):
            if "generate_response" in output and "final_answer" in output["generate_response"]:
                yield json.dumps({
                    "token": output["generate_response"]["final_answer"]
                }) + "\n"
        
        # Final response
        final_state = output.get("update_memory", output.get("generate_response", {}))
        yield json.dumps({
            "answer": final_state.get("final_answer", ""),
            "source": final_state.get("classification", ""),
            "context": final_state.get("compressed_context", ""),
            "tool_calls": final_state.get("tool_calls", []),
            "conversation_id": conversation_id,
            "metadata": final_state.get("metadata", {})
        }) + "\n"
    else:
        # Run synchronously
        config = {"configurable": {"thread_id": conversation_id}}
        final_state = app.invoke(initial_state, config)
        
        return {
            "answer": final_state.get("final_answer", ""),
            "source": final_state.get("classification", ""),
            "context": final_state.get("compressed_context", ""),
            "tool_calls": final_state.get("tool_calls", []),
            "conversation_id": conversation_id,
            "metadata": final_state.get("metadata", {}),
            "error": final_state.get("error")
        }