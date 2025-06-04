from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from app.langchain.service import rag_answer, get_llm_response_direct
from app.langchain.multi_agent_system_simple import MultiAgentSystem
from app.langchain.enhanced_query_classifier import EnhancedQueryClassifier, QueryType
from fastapi.responses import StreamingResponse
import json
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize enhanced query classifier
query_classifier = EnhancedQueryClassifier()

# Thread pool for parallel execution
executor = ThreadPoolExecutor(max_workers=4)

class RAGRequest(BaseModel):
    question: str
    thinking: bool = False
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None  # For backward compatibility
    use_langgraph: bool = True
    collections: Optional[List[str]] = None
    collection_strategy: str = "auto"
    skip_classification: bool = False
    force_hybrid: bool = False  # Force hybrid execution even for single-type queries

class RAGResponse(BaseModel):
    answer: str
    source: str
    conversation_id: Optional[str] = None

class HybridExecutor:
    """Handles execution of hybrid queries"""
    
    def __init__(self):
        self.results = {}
        self.lock = threading.Lock()
        
    async def execute_hybrid(self, 
                           request: RAGRequest, 
                           routing: Dict[str, Any],
                           conversation_id: str):
        """Execute hybrid query with multiple handlers"""
        handlers = routing["routing"]["handlers"]
        execution_mode = routing["routing"]["execution_mode"]
        
        if execution_mode == "parallel":
            # Execute handlers in parallel
            tasks = []
            for handler_config in handlers:
                handler_type = handler_config["type"]
                handler_name = handler_config["handler"]
                weight = handler_config["weight"]
                
                task = self._execute_handler(
                    handler_type, 
                    handler_name, 
                    request, 
                    routing,
                    conversation_id,
                    weight
                )
                tasks.append(task)
            
            # Wait for all handlers to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            combined_result = await self._combine_results(results, handlers)
            return combined_result
        else:
            # Sequential execution
            final_result = None
            for handler_config in handlers:
                handler_type = handler_config["type"]
                handler_name = handler_config["handler"]
                
                result = await self._execute_handler(
                    handler_type,
                    handler_name,
                    request,
                    routing,
                    conversation_id,
                    1.0
                )
                
                if final_result is None:
                    final_result = result
                else:
                    # Merge with previous result
                    final_result = await self._merge_sequential_results(final_result, result)
            
            return final_result
    
    async def _execute_handler(self, 
                             handler_type: str,
                             handler_name: str,
                             request: RAGRequest,
                             routing: Dict,
                             conversation_id: str,
                             weight: float) -> Dict:
        """Execute a specific handler"""
        try:
            if handler_name == "rag":
                return await self._execute_rag(request, conversation_id, weight)
            elif handler_name == "tool_handler":
                return await self._execute_tools(request, routing, conversation_id, weight)
            elif handler_name == "llm":
                return await self._execute_llm(request, conversation_id, weight)
            elif handler_name == "multi_agent":
                return await self._execute_multi_agent(request, routing, conversation_id, weight)
            else:
                logger.warning(f"Unknown handler: {handler_name}")
                return {"error": f"Unknown handler: {handler_name}", "weight": weight}
        except Exception as e:
            logger.error(f"Handler {handler_name} failed: {e}")
            return {"error": str(e), "handler": handler_name, "weight": weight}
    
    async def _execute_rag(self, request: RAGRequest, conversation_id: str, weight: float) -> Dict:
        """Execute RAG search"""
        chunks = []
        
        def collect_chunks():
            for chunk in rag_answer(
                request.question,
                thinking=request.thinking,
                stream=True,
                conversation_id=conversation_id,
                use_langgraph=request.use_langgraph,
                collections=request.collections,
                collection_strategy=request.collection_strategy
            ):
                chunks.append(chunk)
        
        # Run in thread
        await asyncio.get_event_loop().run_in_executor(executor, collect_chunks)
        
        return {
            "type": "rag",
            "chunks": chunks,
            "weight": weight
        }
    
    async def _execute_tools(self, request: RAGRequest, routing: Dict, conversation_id: str, weight: float) -> Dict:
        """Execute tool-based query"""
        suggested_tools = routing.get("routing", {}).get("suggested_tools", [])
        
        # For now, use multi-agent system with tool executors
        system = MultiAgentSystem(conversation_id=conversation_id)
        chunks = []
        
        async for event in system.stream_events(
            request.question,
            selected_agents=["tool_executor"],
            max_iterations=5
        ):
            chunks.append(json.dumps(event, ensure_ascii=False) + "\n")
        
        return {
            "type": "tool",
            "chunks": chunks,
            "suggested_tools": suggested_tools,
            "weight": weight
        }
    
    async def _execute_llm(self, request: RAGRequest, conversation_id: str, weight: float) -> Dict:
        """Execute direct LLM query"""
        chunks = []
        
        def collect_chunks():
            for chunk in get_llm_response_direct(
                request.question,
                thinking=request.thinking,
                stream=True,
                conversation_id=conversation_id
            ):
                chunks.append(chunk)
        
        # Run in thread
        await asyncio.get_event_loop().run_in_executor(executor, collect_chunks)
        
        return {
            "type": "llm",
            "chunks": chunks,
            "weight": weight
        }
    
    async def _execute_multi_agent(self, request: RAGRequest, routing: Dict, conversation_id: str, weight: float) -> Dict:
        """Execute multi-agent query"""
        suggested_agents = routing.get("routing", {}).get("suggested_agents", [])
        
        system = MultiAgentSystem(conversation_id=conversation_id)
        chunks = []
        
        async for event in system.stream_events(
            request.question,
            selected_agents=suggested_agents if suggested_agents else None,
            max_iterations=10
        ):
            chunks.append(json.dumps(event, ensure_ascii=False) + "\n")
        
        return {
            "type": "multi_agent",
            "chunks": chunks,
            "suggested_agents": suggested_agents,
            "weight": weight
        }
    
    async def _combine_results(self, results: List[Dict], handlers: List[Dict]) -> Dict:
        """Combine results from multiple handlers based on weights"""
        combined = {
            "type": "hybrid",
            "components": [],
            "merged_chunks": []
        }
        
        # Collect valid results
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, dict) and "error" not in result:
                result["handler"] = handlers[i]["handler"]
                valid_results.append(result)
                combined["components"].append({
                    "type": result["type"],
                    "weight": result["weight"],
                    "chunk_count": len(result.get("chunks", []))
                })
        
        if not valid_results:
            return {"error": "All handlers failed", "results": results}
        
        # Merge chunks based on weights
        # For now, we'll interleave chunks proportionally based on weights
        total_weight = sum(r["weight"] for r in valid_results)
        
        # Normalize weights
        for result in valid_results:
            result["normalized_weight"] = result["weight"] / total_weight
        
        # Create merged stream
        combined["merged_chunks"] = self._interleave_chunks(valid_results)
        
        return combined
    
    def _interleave_chunks(self, results: List[Dict]) -> List[str]:
        """Interleave chunks from multiple results based on weights"""
        merged = []
        
        # Add header indicating hybrid response
        merged.append(json.dumps({
            "type": "hybrid_response",
            "components": [r["type"] for r in results],
            "weights": {r["type"]: r["normalized_weight"] for r in results}
        }) + "\n")
        
        # Get all chunks
        chunk_iterators = []
        for result in results:
            chunks = result.get("chunks", [])
            if chunks:
                chunk_iterators.append({
                    "type": result["type"],
                    "chunks": chunks,
                    "index": 0,
                    "weight": result["normalized_weight"]
                })
        
        # Interleave based on weights
        while chunk_iterators:
            # Select next source based on weights
            total = sum(it["weight"] for it in chunk_iterators)
            if total == 0:
                break
                
            # Simple round-robin for now, can be improved with weighted selection
            for iterator in chunk_iterators[:]:
                if iterator["index"] < len(iterator["chunks"]):
                    chunk = iterator["chunks"][iterator["index"]]
                    
                    # Wrap chunk with source info
                    if chunk.strip():
                        try:
                            chunk_data = json.loads(chunk.strip())
                            chunk_data["source"] = iterator["type"]
                            merged.append(json.dumps(chunk_data) + "\n")
                        except:
                            # If not JSON, wrap it
                            merged.append(json.dumps({
                                "type": "content",
                                "source": iterator["type"],
                                "content": chunk
                            }) + "\n")
                    
                    iterator["index"] += 1
                else:
                    chunk_iterators.remove(iterator)
        
        return merged
    
    async def _merge_sequential_results(self, result1: Dict, result2: Dict) -> Dict:
        """Merge two sequential results"""
        merged = {
            "type": "sequential",
            "components": [result1.get("type", "unknown"), result2.get("type", "unknown")],
            "chunks": result1.get("chunks", []) + result2.get("chunks", [])
        }
        return merged

# Global hybrid executor instance
hybrid_executor = HybridExecutor()

@router.post("/rag")
async def enhanced_rag_endpoint(request: RAGRequest):
    """Enhanced RAG endpoint with hybrid query support"""
    # Use session_id as conversation_id if not provided
    conversation_id = request.conversation_id or request.session_id
    
    # Get routing recommendation
    routing = None
    if not request.skip_classification:
        routing = query_classifier.get_routing_recommendation(request.question)
        logger.info(f"Query routing: {routing}")
        
    async def stream():
        try:
            # Send classification info first
            if routing:
                classification_chunk = json.dumps({
                    "type": "classification",
                    "routing": routing
                }) + "\n"
                yield classification_chunk
            
            # Check if hybrid execution is needed
            if routing and (routing["is_hybrid"] or request.force_hybrid):
                # Execute hybrid query
                logger.info("Executing hybrid query")
                result = await hybrid_executor.execute_hybrid(request, routing, conversation_id)
                
                if "error" in result:
                    yield json.dumps({"error": result["error"]}) + "\n"
                else:
                    # Stream merged chunks
                    for chunk in result.get("merged_chunks", []):
                        yield chunk
            else:
                # Single handler execution
                primary_type = routing["primary_type"] if routing else "rag"
                
                if primary_type == "tool":
                    # Execute tool query
                    result = await hybrid_executor._execute_tools(request, routing, conversation_id, 1.0)
                    for chunk in result.get("chunks", []):
                        yield chunk
                elif primary_type == "llm" and not routing.get("routing", {}).get("use_rag", True):
                    # Direct LLM
                    result = await hybrid_executor._execute_llm(request, conversation_id, 1.0)
                    for chunk in result.get("chunks", []):
                        yield chunk
                elif primary_type == "multi_agent":
                    # Multi-agent
                    result = await hybrid_executor._execute_multi_agent(request, routing, conversation_id, 1.0)
                    for chunk in result.get("chunks", []):
                        yield chunk
                else:
                    # Default to RAG
                    result = await hybrid_executor._execute_rag(request, conversation_id, 1.0)
                    for chunk in result.get("chunks", []):
                        yield chunk
                        
        except Exception as e:
            logger.error(f"Enhanced RAG endpoint error: {e}")
            import traceback
            traceback.print_exc()
            yield json.dumps({"error": f"Processing failed: {str(e)}"}) + "\n"
    
    return StreamingResponse(stream(), media_type="application/json")

@router.post("/reload-patterns")
async def reload_patterns():
    """Reload query patterns configuration"""
    try:
        query_classifier.reload_config()
        return {"status": "success", "message": "Query patterns reloaded successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/query-patterns")
async def get_query_patterns():
    """Get current query patterns configuration"""
    return {
        "config_path": query_classifier.config_path,
        "patterns": query_classifier.config,
        "compiled_count": {k: len(v) for k, v in query_classifier.compiled_patterns.items()}
    }

@router.post("/test-classification")
async def test_classification(query: str):
    """Test query classification without executing"""
    routing = query_classifier.get_routing_recommendation(query)
    return routing

# Copy existing endpoints from original langchain.py
@router.post("/multi-agent")
async def multi_agent_endpoint(request: MultiAgentRequest):
    """Process a question using the multi-agent system"""
    system = MultiAgentSystem(conversation_id=request.conversation_id)
    
    async def stream_events():
        async for event in system.stream_events(
            request.question,
            selected_agents=request.selected_agents,
            max_iterations=request.max_iterations,
            conversation_history=request.conversation_history
        ):
            if not isinstance(event, dict):
                continue
            yield json.dumps(event, ensure_ascii=False) + "\n"
    
    return StreamingResponse(
        stream_events(),
        media_type="application/json"
    )

class MultiAgentRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    selected_agents: Optional[List[str]] = None
    max_iterations: int = 10
    conversation_history: Optional[List[Dict]] = None

class LargeGenerationRequest(BaseModel):
    task_description: str
    target_count: int = 100
    chunk_size: Optional[int] = None
    conversation_id: Optional[str] = None
    use_redis: bool = True
    conversation_history: Optional[List[Dict]] = None

@router.post("/large-generation")
async def large_generation_endpoint(request: LargeGenerationRequest):
    """Handle large generation tasks"""
    system = MultiAgentSystem(conversation_id=request.conversation_id)
    
    async def stream_events():
        try:
            async for event in system.stream_large_generation_events(
                query=request.task_description,
                target_count=request.target_count,
                chunk_size=request.chunk_size,
                conversation_history=request.conversation_history
            ):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as e:
            error_event = {
                "type": "large_generation_error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            yield json.dumps(error_event, ensure_ascii=False) + "\n"
    
    return StreamingResponse(
        stream_events(),
        media_type="application/json"
    )