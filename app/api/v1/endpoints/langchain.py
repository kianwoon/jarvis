from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from app.langchain.multi_agent_system_simple import MultiAgentSystem
from app.langchain.enhanced_query_classifier import EnhancedQueryClassifier, QueryType
from app.core.langfuse_integration import get_tracer
from app.core.query_classifier_settings_cache import get_query_classifier_settings
from app.core.temp_document_manager import TempDocumentManager
from app.langchain.hybrid_rag_orchestrator import HybridRAGOrchestrator
from app.core.simple_conversation_manager import conversation_manager
from fastapi.responses import StreamingResponse
import json as json_module  # Import with alias to avoid scope issues
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize enhanced query classifier
query_classifier = EnhancedQueryClassifier()

# Initialize temp document manager
temp_doc_manager = TempDocumentManager()

# Initialize hybrid RAG orchestrator
hybrid_rag_orchestrator = HybridRAGOrchestrator()

class RAGRequest(BaseModel):
    question: str
    thinking: bool = False
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None  # Add this for backward compatibility
    use_langgraph: bool = True
    collections: Optional[List[str]] = None  # Specific collections to search
    collection_strategy: str = "auto"  # "auto", "specific", or "all"
    skip_classification: bool = False  # Allow bypassing classification if needed
    include_temp_docs: Optional[bool] = None  # Include temporary documents in search
    active_temp_doc_ids: Optional[List[str]] = None  # Specific temp doc IDs to include
    # Hybrid RAG mode parameters
    use_hybrid_rag: bool = False  # Enable hybrid RAG mode
    hybrid_strategy: str = "temp_priority"  # "temp_priority", "parallel_fusion", "persistent_only"
    fallback_to_persistent: bool = True  # Fallback to persistent RAG if no temp results
    temp_results_weight: float = 0.7  # Weight for temp results in fusion (0.0-1.0)

class RAGResponse(BaseModel):
    answer: str
    source: str
    conversation_id: Optional[str] = None

@router.post("/rag")
async def rag_endpoint(request: RAGRequest):
    # Initialize execution tracking to prevent duplicate MCP calls
    request._rag_knowledge_search_executed = False
    request._rag_documents = None
    
    # Initialize Langfuse tracing
    tracer = get_tracer()
    trace = None
    generation = None
    
    # Get model name from LLM settings for proper tracing
    from app.core.llm_settings_cache import get_llm_settings, get_main_llm_full_config
    llm_settings = get_llm_settings()
    main_llm_config = get_main_llm_full_config(llm_settings)
    model_name = main_llm_config.get("model", "unknown")
    
    if not tracer._initialized:
        tracer.initialize()
    
    # Initialize rag_span outside conditional to ensure it's always defined
    rag_span = None
    
    if tracer.is_enabled():
        trace = tracer.create_trace(
            name="rag-workflow",
            input=request.question,
            metadata={
                "endpoint": "/api/v1/langchain/rag",
                "conversation_id": request.conversation_id,
                "thinking": request.thinking,
                "collections": request.collections,
                "collection_strategy": request.collection_strategy,
                "use_langgraph": request.use_langgraph,
                "model": model_name
            }
        )
        
        # Create RAG execution span for proper hierarchy
        if trace:
            rag_span = tracer.create_span(
                trace,
                name="rag-execution",
                metadata={
                    "operation": "rag_search_and_generation",
                    "thinking": request.thinking,
                    "collections": request.collections,
                    "collection_strategy": request.collection_strategy,
                    "use_langgraph": request.use_langgraph,
                    "model": model_name
                }
            )
        
        # Create generation within the RAG span for detailed observability
        if rag_span:
            generation = tracer.create_generation_with_usage(
                trace=trace,
                name="rag-generation",
                model=model_name,
                input_text=request.question,
                parent_span=rag_span,
                metadata={
                    "thinking": request.thinking,
                    "collections": request.collections,
                    "collection_strategy": request.collection_strategy,
                    "use_langgraph": request.use_langgraph,
                    "model": model_name,
                    "endpoint": "rag"
                }
            )
    
    # Use session_id as conversation_id if conversation_id is not provided
    conversation_id = request.conversation_id or request.session_id
    
    # Unified function to prevent duplicate rag_knowledge_search executions
    async def execute_rag_search_once(enhanced_question, trace_span=None):
        """Execute rag_knowledge_search only once per request to prevent duplicates"""
        if request._rag_knowledge_search_executed:
            logger.info(f"[EXECUTION TRACKER] SKIPPING duplicate rag_knowledge_search call - already executed")
            print(f"[EXECUTION TRACKER] SKIPPING duplicate rag_knowledge_search call - already executed")
            return request._rag_documents
        
        # Mark as executed to prevent future calls
        request._rag_knowledge_search_executed = True
        logger.info(f"[EXECUTION TRACKER] EXECUTION #1: Direct MCP call to rag_knowledge_search (unified)")
        print(f"[EXECUTION TRACKER] EXECUTION #1: Direct MCP call to rag_knowledge_search (unified)")
        
        # Prepare parameters for rag_knowledge_search tool
        from app.core.rag_settings_cache import get_document_retrieval_settings
        doc_settings = get_document_retrieval_settings()
        max_docs = doc_settings.get('max_documents_mcp', 8)
        
        tool_parameters = {
            "query": request.question,  # Use original question for document search, not conversation history
            "include_content": True,
            "max_documents": max_docs
        }
        
        # Add collections if specified
        if request.collections:
            tool_parameters["collections"] = request.collections
        
        print(f"[DEBUG] Unified RAG search - tool parameters: {tool_parameters}")
        
        # Execute rag_knowledge_search tool
        from app.langchain.service import call_mcp_tool
        rag_tool_result = call_mcp_tool(
            tool_name="rag_knowledge_search",
            parameters=tool_parameters,
            trace=trace_span or trace
        )
        
        # Extract documents from tool result
        extracted_documents = []
        if isinstance(rag_tool_result, dict):
            result_data = rag_tool_result
            if 'result' in result_data and isinstance(result_data['result'], dict):
                rag_result = result_data['result']
                if 'documents' in rag_result:
                    extracted_documents.extend(rag_result['documents'])
        
        # Cache the result for future use in this request
        request._rag_documents = extracted_documents
        logger.info(f"[EXECUTION TRACKER] Unified search found {len(extracted_documents)} documents")
        
        return extracted_documents
    
    # Check if we should bypass classification for hybrid RAG mode
    if request.use_hybrid_rag and conversation_id:
        logger.info(f"[HYBRID RAG] Bypassing classification for hybrid RAG mode")
        # Skip classification and go directly to RAG handling
        request.skip_classification = True
    
    # Classify the query unless explicitly skipped
    routing = None
    if not request.skip_classification:
        # Get classification thresholds from settings
        classifier_settings = get_query_classifier_settings()
        tool_threshold = classifier_settings.get('direct_execution_threshold', 0.55)
        llm_threshold = classifier_settings.get('llm_direct_threshold', 0.8)
        multi_agent_threshold = classifier_settings.get('multi_agent_threshold', 0.6)
        
        routing = await query_classifier.get_routing_recommendation(request.question, trace=trace)
        logger.info(f"Query routing: {routing['primary_type']} (confidence: {routing['confidence']:.2f})")
        
        # Handle based on classification (using enhanced classifier types)
        primary_type = routing['primary_type']
        is_hybrid = routing.get('is_hybrid', False)
        confidence = routing['confidence']
        
        logger.info(f"[ROUTING DEBUG] primary_type='{primary_type}', QueryType.TOOL.value='{QueryType.TOOL.value}', confidence={confidence}, tool_threshold={tool_threshold}, is_hybrid={is_hybrid}")
        
        # Check for hybrid queries that need multiple handlers
        if is_hybrid:
            logger.info(f"[ROUTING DEBUG] Taking hybrid query path")
            # Handle hybrid queries (TOOL_RAG, TOOL_LLM, RAG_LLM, TOOL_RAG_LLM)
            return handle_hybrid_query(request, routing)
        elif primary_type == QueryType.TOOL.value and confidence >= tool_threshold:
            # CRITICAL FIX: Check if the "tool" is actually rag_knowledge_search
            # If so, route to RAG synthesis instead of tool calling to prevent 
            # showing <tool>rag_knowledge_search(...) instead of synthesized content
            suggested_tools = routing.get('suggested_tools', [])
            if not suggested_tools and 'routing' in routing:
                suggested_tools = routing['routing'].get('suggested_tools', [])
            
            is_rag_query = any(
                tool in ['rag_knowledge_search', 'knowledge_search', 'document_search'] 
                for tool in suggested_tools
            )
            
            if is_rag_query:
                logger.info(f"[ROUTING DEBUG] RAG query detected in TOOL classification - routing to RAG synthesis instead")
                # Route to RAG synthesis path instead of tool calling
                # This prevents showing <tool>rag_knowledge_search(...) and ensures proper synthesis
                # Fall through to RAG path below
            else:
                logger.info(f"[ROUTING DEBUG] Taking direct tool execution path")
                # Direct tool execution for confident classifications
                # Use the suggested tool from classification instead of planning
                return handle_direct_tool_query(request, routing, trace=trace)
        elif primary_type == QueryType.LLM.value and confidence > llm_threshold:
            # Route to direct LLM only if high confidence
            return handle_direct_llm_query(request, routing)
        elif primary_type == QueryType.MULTI_AGENT.value and confidence > multi_agent_threshold:
            # Route to multi-agent system
            return handle_multi_agent_query(request, routing)
        else:
            logger.info(f"[ROUTING DEBUG] Taking fallback RAG path")
        # Fall through to RAG for RAG_SEARCH or low confidence cases
    
    async def stream():
        try:
            chunk_count = 0
            
            # Store user message in conversation history
            if conversation_id:
                await conversation_manager.add_message(
                    conversation_id=conversation_id,
                    role="user",
                    content=request.question
                )
            
            # Retrieve conversation history and enhance question with context
            enhanced_question = request.question
            if conversation_id and not request.skip_classification:
                # Get last 3 exchanges (6 messages: 3 user + 3 assistant)
                history = await conversation_manager.get_conversation_history(conversation_id, limit=6)
                if history:
                    # Format history for LLM context
                    enhanced_question = conversation_manager.format_history_for_prompt(history, request.question)
                    logger.info(f"[CONVERSATION] Retrieved {len(history)} messages for conversation {conversation_id}")
            
            # Add classification metadata to first chunk if available
            if not request.skip_classification and routing:
                classification_chunk = json_module.dumps({
                    "type": "classification",
                    "routing": routing,
                    "handler": "rag"
                }) + "\n"
                yield classification_chunk
                
            print(f"[DEBUG] API endpoint - About to call rag_answer with stream=True")
            
            # Map enhanced classifier types to simple optimization types
            simple_query_type = None
            
            # Force RAG type when using hybrid RAG mode
            if request.use_hybrid_rag:
                simple_query_type = "RAG"
                print(f"[DEBUG] API endpoint - Forcing query_type='RAG' for hybrid RAG mode")
            elif routing:
                enhanced_type = routing['primary_type']
                
                # CRITICAL FIX: Check if this was a TOOL query that we detected as RAG
                # If so, override the mapping to ensure proper RAG synthesis
                suggested_tools = routing.get('suggested_tools', [])
                if not suggested_tools and 'routing' in routing:
                    suggested_tools = routing['routing'].get('suggested_tools', [])
                
                is_rag_query = any(
                    tool in ['rag_knowledge_search', 'knowledge_search', 'document_search'] 
                    for tool in suggested_tools
                )
                
                type_mapping = {
                    "rag": "RAG",
                    "tool": "RAG" if is_rag_query else "TOOLS",  # Map tool->RAG for knowledge queries
                    "llm": "LLM",
                    "code": "LLM",
                    "multi_agent": "RAG",
                    "tool_rag": "TOOLS",
                    "tool_llm": "TOOLS", 
                    "rag_llm": "RAG",
                    "tool_rag_llm": "TOOLS"
                }
                simple_query_type = type_mapping.get(enhanced_type, "LLM")
                print(f"[DEBUG] API endpoint - Mapped '{enhanced_type}' to '{simple_query_type}'")
                print(f"[DEBUG] API endpoint - About to pass query_type='{simple_query_type}' to rag_answer")
            
            # Handle hybrid RAG mode or traditional temp document integration
            # enhanced_question already set above with conversation context
            hybrid_context = None
            
            if conversation_id and request.use_hybrid_rag:
                try:
                    logger.info(f"[HYBRID RAG] Using hybrid mode with strategy: {request.hybrid_strategy}")
                    
                    # Use hybrid orchestrator for intelligent result fusion
                    hybrid_result = await hybrid_rag_orchestrator.query(
                        query=request.question,
                        conversation_id=conversation_id,
                        strategy=request.hybrid_strategy,
                        fallback_to_persistent=request.fallback_to_persistent,
                        temp_results_weight=request.temp_results_weight,
                        top_k=5
                    )
                    
                    if hybrid_result and hybrid_result.final_sources:
                        # Build enhanced context from hybrid results
                        context_parts = []
                        for source in hybrid_result.final_sources[:5]:  # Top 5 results
                            content = source.get('content', '')
                            filename = source.get('filename', 'Unknown document')
                            context_parts.append(f"From {filename}:\n{content}")
                        
                        temp_doc_context = "\n\n=== TEMPORARY DOCUMENTS ===\n" + "\n\n".join(context_parts) + "\n=== END TEMPORARY DOCUMENTS ===\n\n"
                        
                        # Enhance the question with hybrid context
                        enhanced_question = f"""Context from uploaded documents:
{temp_doc_context}

User question: {request.question}

Please answer the user's question using the information from the uploaded documents above, along with any other relevant knowledge."""
                        
                        hybrid_context = {
                            'sources': hybrid_result.final_sources,
                            'strategy_used': hybrid_result.strategy_used,
                            'sources_queried': hybrid_result.sources_queried,
                            'fusion_metadata': hybrid_result.fusion_metadata
                        }
                        logger.info(f"[HYBRID RAG] Enhanced question with {len(hybrid_result.final_sources)} hybrid sources")
                    else:
                        logger.info(f"[HYBRID RAG] No hybrid results found")
                        
                except Exception as e:
                    logger.warning(f"Hybrid RAG failed, falling back to traditional mode: {e}")
                    import traceback
                    logger.warning(f"Full traceback: {traceback.format_exc()}")
            
            elif conversation_id and (request.include_temp_docs is not False):
                # Traditional temp document integration (backward compatibility)
                try:
                    logger.info(f"[TEMP DOC DEBUG] Searching for temp docs with conversation_id: {conversation_id}")
                    
                    # First check if any temp docs exist for this conversation
                    all_docs = await temp_doc_manager.get_conversation_documents(conversation_id)
                    logger.info(f"[TEMP DOC DEBUG] Found {len(all_docs)} total temp documents for conversation")
                    
                    for doc in all_docs:
                        logger.info(f"[TEMP DOC DEBUG] Doc: {doc.get('filename', 'Unknown')} - included: {doc.get('is_included', False)}")
                    
                    # Query temp documents if they exist
                    temp_results = await temp_doc_manager.query_conversation_documents(
                        conversation_id=conversation_id,
                        query=request.question,
                        include_all=False,  # Only active documents
                        top_k=5
                    )
                    
                    logger.info(f"[TEMP DOC DEBUG] Query returned {len(temp_results)} results")
                    
                    if temp_results:
                        logger.info(f"Found {len(temp_results)} temporary document results")
                        temp_context_parts = []
                        for result in temp_results:
                            filename = result.get('filename', 'Unknown document')
                            content = result.get('content', '')
                            logger.info(f"[TEMP DOC DEBUG] Result from {filename}: {content[:100]}...")
                            temp_context_parts.append(f"From {filename}:\n{content}")
                        
                        temp_doc_context = "\n\n=== TEMPORARY DOCUMENTS ===\n" + "\n\n".join(temp_context_parts) + "\n=== END TEMPORARY DOCUMENTS ===\n\n"
                        
                        # Enhance the question with temp document context
                        enhanced_question = f"""Context from uploaded documents:
{temp_doc_context}

User question: {request.question}

Please answer the user's question using the information from the uploaded documents above, along with any other relevant knowledge."""
                        
                        logger.info(f"[TEMP DOC DEBUG] Enhanced question length: {len(enhanced_question)}")
                    else:
                        logger.info(f"[TEMP DOC DEBUG] No temp document results found for query")
                        
                except Exception as e:
                    logger.warning(f"Failed to get temporary document context: {e}")
                    import traceback
                    logger.warning(f"Full traceback: {traceback.format_exc()}")
            
            # Use unified RAG search to prevent duplicate executions
            try:
                yield json_module.dumps({
                    "type": "status", 
                    "message": "Searching knowledge base..."
                }) + "\n"
                
                # Execute rag_knowledge_search once using unified function
                extracted_documents = await execute_rag_search_once(enhanced_question, rag_span or trace)
                
                print(f"[DEBUG] API endpoint - Extracted {len(extracted_documents)} documents from unified RAG search")
                
                if extracted_documents:
                    yield json_module.dumps({
                        "type": "status", 
                        "message": f"Found {len(extracted_documents)} relevant documents, generating response..."
                    }) + "\n"
                    
                    # Build synthesis prompt (similar to tool path approach)
                    # Extract document content for context
                    document_contexts = []
                    for doc in extracted_documents:
                        content = doc.get('content', '')
                        # Try multiple ways to get file information from MCP response
                        file_info = (
                            doc.get('file') or 
                            doc.get('filename') or 
                            doc.get('source') or 
                            doc.get('metadata', {}).get('source') or
                            doc.get('metadata', {}).get('file') or
                            'Unknown source'
                        )
                        print(f"[DEBUG] Document metadata: {doc.keys()}")
                        print(f"[DEBUG] Extracted file_info: {file_info}")
                        document_contexts.append(f"**📄 {file_info}:**\n{content}")
                    
                    documents_text = "\n\n".join(document_contexts)
                    
                    # Include conversation history in synthesis prompt
                    synthesis_prompt = f"""Based on the search results below, provide a comprehensive answer to the user's question.

{enhanced_question}

📚 Internal Knowledge Base Results:
{documents_text}

Please provide a detailed, helpful response based on the information found in the knowledge base."""

                    # Generate response using LLM synthesis (skip RAG search)
                    from app.langchain.service import rag_answer
                    
                    # Use rag_answer in direct synthesis mode (skip tool planning)
                    # Pass the original enhanced_question (which already has conversation history)
                    # and include the documents in hybrid_context
                    synthesis_context = {
                        "sources": extracted_documents,
                        "strategy": "unified_search",
                        "pre_formatted_context": documents_text
                    }
                    
                    rag_stream = await rag_answer(
                        enhanced_question,  # Use enhanced_question to maintain conversation context
                        thinking=request.thinking, 
                        stream=True,
                        conversation_id=conversation_id,
                        use_langgraph=False,  # Disable langgraph to prevent tool planning
                        collections=None,  # Don't search again, use provided context
                        collection_strategy="all",  # Use available collections but we won't search
                        query_type="SYNTHESIS",  # Custom type to skip tool planning
                        trace=rag_span or trace,
                        hybrid_context=synthesis_context  # Pass document context via hybrid_context
                    )
                    
                    print(f"[DEBUG] API endpoint - Got synthesis stream: {type(rag_stream)}")
                    
                    # Track complete response for final message
                    complete_response = ""
                    
                    try:
                        # Stream the synthesis results
                        async for chunk in rag_stream:
                            try:
                                if chunk:  # Only yield non-empty chunks
                                    chunk_count += 1
                                    yield chunk
                                    
                                    # Parse chunk to accumulate complete response
                                    try:
                                        chunk_data = json_module.loads(chunk.strip())
                                        if "token" in chunk_data:
                                            complete_response += chunk_data["token"]
                                        elif "answer" in chunk_data:
                                            # Some responses send complete answer
                                            complete_response = chunk_data["answer"]
                                    except:
                                        pass  # Not a JSON chunk
                                    
                                    # Add small delay to ensure streaming
                                    import asyncio
                                    await asyncio.sleep(0)
                            except GeneratorExit:
                                # Client disconnected, log and stop gracefully
                                print(f"[INFO] Client disconnected after {chunk_count} chunks")
                                break
                            except Exception as chunk_error:
                                print(f"[ERROR] Error yielding chunk {chunk_count}: {chunk_error}")
                                # Try to yield error message, but don't fail the whole stream
                                try:
                                    yield json_module.dumps({"error": f"Chunk error: {str(chunk_error)}"}) + "\n"
                                except:
                                    break  # If we can't even send error, client is likely gone
                    finally:
                        # ALWAYS send completion event with documents for UI accordion, even if streaming was interrupted
                        if extracted_documents:
                            # Format documents for frontend
                            formatted_docs = []
                            for doc in extracted_documents:
                                formatted_doc = {
                                    "content": doc.get('content', ''),
                                    "source": (
                                        doc.get('file') or 
                                        doc.get('filename') or 
                                        doc.get('source') or 
                                        doc.get('metadata', {}).get('source') or
                                        doc.get('metadata', {}).get('file') or
                                        'Unknown source'
                                    ),
                                    "relevance_score": doc.get('score', 0.0)
                                }
                                formatted_docs.append(formatted_doc)
                            
                            # Send completion event matching frontend expectations
                            yield json_module.dumps({
                                "answer": complete_response,
                                "source": "RAG",
                                "conversation_id": conversation_id,
                                "documents": formatted_docs
                            }) + "\n"
                else:
                    # No documents found, provide a fallback response
                    yield json_module.dumps({
                        "type": "status", 
                        "message": "No relevant documents found, providing general response..."
                    }) + "\n"
                    
                    # Fallback to direct LLM response
                    fallback_prompt = f"I couldn't find specific information in the knowledge base for: {request.question}\n\nPlease provide a helpful response based on general knowledge."
                    
                    from app.langchain.service import rag_answer
                    rag_stream = await rag_answer(
                        fallback_prompt,
                        thinking=request.thinking, 
                        stream=True,
                        conversation_id=conversation_id,
                        use_langgraph=request.use_langgraph,
                        collections=None,
                        collection_strategy="skip",
                        query_type="LLM",
                        trace=rag_span or trace,
                        hybrid_context=hybrid_context
                    )
                    
                    async for chunk in rag_stream:
                        try:
                            if chunk:
                                chunk_count += 1
                                yield chunk
                                import asyncio
                                await asyncio.sleep(0)
                        except GeneratorExit:
                            print(f"[INFO] Client disconnected after {chunk_count} chunks")
                            break
                        except Exception as chunk_error:
                            print(f"[ERROR] Error yielding chunk {chunk_count}: {chunk_error}")
                            try:
                                yield json_module.dumps({"error": f"Chunk error: {str(chunk_error)}"}) + "\n"
                            except:
                                break
                                
            except Exception as tool_error:
                print(f"[ERROR] RAG tool execution failed: {tool_error}")
                # Fallback to original rag_answer approach if tool fails
                yield json_module.dumps({
                    "type": "status", 
                    "message": "Tool execution failed, falling back to direct search..."
                }) + "\n"
                
                from app.langchain.service import rag_answer
                rag_stream = await rag_answer(
                    enhanced_question, 
                    thinking=request.thinking, 
                    stream=True,
                    conversation_id=conversation_id,
                    use_langgraph=request.use_langgraph,
                    collections=request.collections,
                    collection_strategy=request.collection_strategy,
                    query_type=simple_query_type,
                    trace=rag_span or trace,
                    hybrid_context=hybrid_context
                )
                
                async for chunk in rag_stream:
                    try:
                        if chunk:
                            chunk_count += 1
                            yield chunk
                            import asyncio
                            await asyncio.sleep(0)
                    except GeneratorExit:
                        print(f"[INFO] Client disconnected after {chunk_count} chunks")
                        break
                    except Exception as chunk_error:
                        print(f"[ERROR] Error yielding chunk {chunk_count}: {chunk_error}")
                        try:
                            yield json_module.dumps({"error": f"Chunk error: {str(chunk_error)}"}) + "\n"
                        except:
                            break
        except (ConnectionError, BrokenPipeError) as conn_error:
            print(f"[INFO] Connection lost during RAG processing: {conn_error}")
            # Client disconnected, don't try to send anything
        except Exception as e:
            print(f"[ERROR] RAG endpoint error: {e}")
            import traceback
            traceback.print_exc()
            try:
                yield json_module.dumps({"error": f"RAG processing failed: {str(e)}"}) + "\n"
            except:
                # If we can't send the error, client is likely disconnected
                print(f"[ERROR] Could not send error response, client likely disconnected")
    
    print(f"[DEBUG] API endpoint - Returning StreamingResponse")
    
    # Create a wrapper to capture the output for Langfuse tracing
    async def stream_with_tracing():
        logger.info(f"[TRACE DEBUG] Stream wrapper started for tracing")
        collected_output = ""
        final_answer = ""
        source_info = ""
        
        try:
            async for chunk in stream():
                # Collect the streamed data
                if chunk and chunk.strip():
                    collected_output += chunk
                    # logger.info(f"[TRACE DEBUG] Chunk received: {chunk[:200]}...")  # Commented out to reduce log noise
                    
                    # Try to extract the final answer from the response chunks
                    if '"answer"' in chunk:
                        try:
                            chunk_data = json_module.loads(chunk.strip())
                            if "answer" in chunk_data:
                                final_answer = chunk_data["answer"]
                                # logger.info(f"[TRACE DEBUG] Extracted final answer: {final_answer[:100]}...")  # Commented out
                                if final_answer:
                                    pass  # logger.info(f"[TRACE DEBUG] Final answer: {final_answer[:100]}...")  # Commented out
                            if "source" in chunk_data:
                                source_info = chunk_data["source"]
                                logger.info(f"[TRACE DEBUG] Extracted source info: {source_info}")
                        except Exception as e:
                            logger.warning(f"[TRACE DEBUG] Failed to parse chunk: {e}")
                            pass  # Continue streaming even if parsing fails
                
                yield chunk
            
            # Save assistant response to conversation history
            if conversation_id and final_answer:
                await conversation_manager.add_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=final_answer,
                    metadata={"source": source_info or "rag-chat"}
                )
                logger.info(f"[CONVERSATION] Saved assistant response to conversation {conversation_id}")
            
            # Update Langfuse generation and trace with the final output
            logger.info(f"[TRACE DEBUG] Stream completed, updating trace. Final answer length: {len(final_answer) if final_answer else 0}")
            if tracer.is_enabled():
                try:
                    # Ensure we have meaningful output for the generation
                    generation_output = final_answer if final_answer else "Response generated successfully"
                    logger.info(f"[TRACE DEBUG] Final generation output for tracing: {len(generation_output)} chars, has_final_answer: {bool(final_answer)}")
                    
                    # Estimate token usage for cost tracking
                    usage = tracer.estimate_token_usage(request.question, generation_output)
                    
                    # End the generation with results including usage
                    if generation:
                        generation.end(
                            output=generation_output,
                            usage=usage,
                            metadata={
                                "response_length": len(final_answer) if final_answer else len(collected_output),
                                "source": source_info or "rag-chat",
                                "streaming": True,
                                "has_answer": bool(final_answer),
                                "conversation_id": conversation_id,
                                "output_captured": bool(final_answer),
                                "model": model_name,
                                "input_length": len(request.question),
                                "output_length": len(generation_output),
                                "estimated_tokens": usage
                            }
                        )
                    
                    # Update trace with final result
                    if trace:
                        trace.update(
                            output=final_answer if final_answer else collected_output[:500],
                            metadata={
                                "success": True,
                                "source": source_info,
                                "response_length": len(final_answer) if final_answer else len(collected_output),
                                "streaming": True,
                                "conversation_id": conversation_id
                            }
                        )
                    
                    tracer.flush()
                except Exception as e:
                    print(f"[WARNING] Failed to update Langfuse trace/generation: {e}")
            
            # Note: workflow_span is only for multi-agent mode, not standard RAG
                    
        except Exception as e:
            # Update generation and trace with error
            if tracer.is_enabled():
                try:
                    # Estimate usage even for errors
                    error_output = f"Error: {str(e)}"
                    usage = tracer.estimate_token_usage(request.question, error_output)
                    
                    # End generation with error
                    if generation:
                        generation.end(
                            output=error_output,
                            usage=usage,
                            metadata={
                                "success": False,
                                "error": str(e),
                                "conversation_id": conversation_id,
                                "model": model_name,
                                "estimated_tokens": usage
                            }
                        )
                    
                    # Update trace with error
                    if trace:
                        trace.update(
                            output=f"Error: {str(e)}",
                            metadata={
                                "success": False,
                                "error": str(e),
                                "conversation_id": conversation_id
                            }
                        )
                    
                    tracer.flush()
                except:
                    pass  # Don't fail the request if tracing fails
            raise
    
    return StreamingResponse(stream_with_tracing(), media_type="application/json")

class MultiAgentRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    selected_agents: Optional[List[str]] = None  # Specific agents to use
    max_iterations: int = 10
    conversation_history: Optional[List[Dict]] = None  # Previous messages for context

class MultiAgentResponse(BaseModel):
    final_answer: str
    agent_responses: List[Dict[str, str]]
    conversation_id: str

class LargeGenerationRequest(BaseModel):
    task_description: str
    target_count: int = 100
    chunk_size: Optional[int] = None  # Auto-calculated if not provided
    conversation_id: Optional[str] = None
    use_redis: bool = True  # Use Redis for state management
    conversation_history: Optional[List[Dict]] = None

class GenerationProgressRequest(BaseModel):
    session_id: str

@router.post("/multi-agent")
async def multi_agent_endpoint(request: MultiAgentRequest):
    """Process a question using the LangGraph multi-agent system"""
    print(f"[DEBUG] 🚀 MULTI-AGENT ENDPOINT CALLED with question: {request.question[:50]}...")
    
    # Initialize Langfuse tracing
    tracer = get_tracer()
    trace = None
    generation = None
    
    # Get model name from LLM settings for proper tracing
    from app.core.llm_settings_cache import get_llm_settings, get_main_llm_full_config
    llm_settings = get_llm_settings()
    main_llm_config = get_main_llm_full_config(llm_settings)
    model_name = main_llm_config.get("model", "unknown")
    
    if not tracer._initialized:
        tracer.initialize()
    
    if tracer.is_enabled():
        trace = tracer.create_trace(
            name="langgraph-multi-agent-workflow",
            input=request.question,
            metadata={
                "endpoint": "/api/v1/langchain/multi-agent",
                "conversation_id": request.conversation_id,
                "selected_agents": request.selected_agents,
                "max_iterations": request.max_iterations,
                "model": model_name,
                "system_type": "langgraph"
            }
        )
        
        # Create generation for LangGraph multi-agent workflow
        if trace:
            generation = tracer.create_generation_with_usage(
                trace=trace,
                name="langgraph-multi-agent-generation",
                model=model_name,
                input_text=request.question,
                metadata={
                    "selected_agents": request.selected_agents,
                    "max_iterations": request.max_iterations,
                    "system_type": "langgraph",
                    "model": model_name,
                    "endpoint": "multi-agent"
                }
            )
    
    # Create multi-agent workflow span for proper hierarchy
    workflow_span = None
    if trace:
        try:
            if tracer.is_enabled():
                workflow_span = tracer.create_multi_agent_workflow_span(
                    trace, "langgraph-multi-agent", request.selected_agents or []
                )
        except Exception as e:
            print(f"[DEBUG] Failed to create multi-agent workflow span: {e}")
    
    # Use LangGraph multi-agent system with streaming callback
    from app.langchain.langgraph_multi_agent_system import LangGraphMultiAgentSystem
    
    async def stream_events_with_tracing():
        print(f"[DEBUG] ✅ STREAM FUNCTION CALLED FOR MULTI-AGENT")
        collected_output = ""
        final_response = ""
        agent_outputs = []  # Initialize to track agent outputs
        
        try:
            # Use streaming multi-agent function like working RAG pattern
            from app.langchain.langgraph_multi_agent_system import langgraph_multi_agent_answer
            
            # Get streaming generator (same pattern as rag_answer)
            multi_agent_stream = await langgraph_multi_agent_answer(
                question=request.question,
                conversation_id=request.conversation_id,
                stream=True,
                trace=workflow_span or trace
            )
            print(f"[DEBUG] API endpoint - Got multi_agent_stream: {type(multi_agent_stream)}")
            
            # Stream chunks exactly like RAG endpoint does
            chunk_count = 0
            async for chunk in multi_agent_stream:
                try:
                    if chunk:  # Only yield non-empty chunks
                        chunk_count += 1
                        yield chunk
                        # Collect agent outputs from chunks
                        try:
                            if isinstance(chunk, str):
                                chunk_data = json_module.loads(chunk.strip())
                                if chunk_data.get("type") == "agent_response":
                                    agent_outputs.append({
                                        "agent": chunk_data.get("agent", "unknown"),
                                        "content": chunk_data.get("response", "")
                                    })
                        except:
                            pass  # Ignore parsing errors
                        # Add small delay to ensure streaming
                        import asyncio
                        await asyncio.sleep(0.01)
                except GeneratorExit:
                    # Client disconnected, log and stop gracefully
                    print(f"[INFO] Client disconnected after {chunk_count} chunks")
                    break
                except Exception as chunk_error:
                    print(f"[ERROR] Error yielding chunk {chunk_count}: {chunk_error}")
                    # Try to yield error message, but don't fail the whole stream
                    try:
                        yield json_module.dumps({"error": f"Chunk error: {str(chunk_error)}"}) + "\n"
                    except:
                        break  # If we can't even send error, client is likely gone
                        
            print(f"[DEBUG] Multi-agent streaming completed with {chunk_count} chunks")
            collected_output = f"Multi-agent streaming completed with {chunk_count} chunks"
            
            # Update Langfuse generation and trace with the final output
            if tracer.is_enabled():
                try:
                    generation_output = final_response if final_response else "Multi-agent processing completed"
                    
                    # Estimate token usage for cost tracking
                    usage = tracer.estimate_token_usage(request.question, generation_output)
                    
                    # End the generation with results including usage
                    if generation:
                        generation.end(
                            output=generation_output,
                            usage=usage,
                            metadata={
                                "response_length": len(final_response) if final_response else len(collected_output),
                                "source": "multi-agent",
                                "streaming": True,
                                "has_final_response": bool(final_response),
                                "conversation_id": request.conversation_id,
                                "agent_count": len(agent_outputs),
                                "model": model_name,
                                "input_length": len(request.question),
                                "output_length": len(generation_output),
                                "estimated_tokens": usage
                            }
                        )
                    
                    # Update trace with final result
                    if trace:
                        trace.update(
                            output=generation_output,
                            metadata={
                                "success": True,
                                "source": "multi-agent",
                                "agent_outputs": agent_outputs,
                                "response_length": len(final_response) if final_response else len(collected_output),
                                "streaming": True,
                                "conversation_id": request.conversation_id
                            }
                        )
                    
                    tracer.flush()
                except Exception as e:
                    print(f"[WARNING] Failed to update Langfuse trace/generation: {e}")
            
            # End multi-agent workflow span with success
            if workflow_span:
                try:
                    tracer.end_span_with_result(
                        workflow_span,
                        {
                            "agent_count": len(agent_outputs),
                            "successful_agents": len([a for a in agent_outputs if a.get("content")]),
                            "final_response_length": len(final_response) if final_response else 0,
                            "total_output_length": len(collected_output)
                        },
                        success=bool(final_response)
                    )
                except Exception as e:
                    print(f"[WARNING] Failed to end multi-agent workflow span: {e}")
                    
        except Exception as e:
            # Update generation and trace with error
            if tracer.is_enabled():
                try:
                    # Estimate usage even for errors
                    error_output = f"Error: {str(e)}"
                    usage = tracer.estimate_token_usage(request.question, error_output)
                    
                    # Individual agent generations handle their own errors
                    # No top-level generation to end in multi-agent mode
                    
                    # Update trace with error
                    if trace:
                        trace.update(
                            output=f"Error: {str(e)}",
                            metadata={
                                "success": False,
                                "error": str(e),
                                "conversation_id": request.conversation_id
                            }
                        )
                    
                    tracer.flush()
                except:
                    pass  # Don't fail the request if tracing fails
            
            # End multi-agent workflow span with error
            if workflow_span:
                try:
                    tracer.end_span_with_result(workflow_span, None, False, str(e))
                except Exception as span_error:
                    print(f"[WARNING] Failed to end multi-agent workflow span with error: {span_error}")
            
            raise
    
    print(f"[DEBUG] 🎯 ABOUT TO RETURN STREAMING RESPONSE FOR MULTI-AGENT")
    return StreamingResponse(
        stream_events_with_tracing(),
        media_type="application/json"
    )

@router.post("/large-generation")
async def large_generation_endpoint(request: LargeGenerationRequest):
    """
    Handle large generation tasks that transcend context limits
    using intelligent chunking and Redis-based state management
    """
    # Initialize Langfuse tracing
    tracer = get_tracer()
    trace = None
    generation = None
    
    # Get model name from LLM settings for proper tracing
    from app.core.llm_settings_cache import get_llm_settings, get_main_llm_full_config
    llm_settings = get_llm_settings()
    main_llm_config = get_main_llm_full_config(llm_settings)
    model_name = main_llm_config.get("model", "unknown")
    
    if not tracer._initialized:
        tracer.initialize()
    
    if tracer.is_enabled():
        trace = tracer.create_trace(
            name="large-generation-workflow",
            input=request.task_description,
            metadata={
                "endpoint": "/api/v1/langchain/large-generation",
                "conversation_id": request.conversation_id,
                "target_count": request.target_count,
                "chunk_size": request.chunk_size,
                "use_redis": request.use_redis,
                "model": model_name
            }
        )
        
        # Create generation within the trace for detailed observability
        if trace:
            generation = tracer.create_generation_with_usage(
                trace=trace,
                name="large-generation-generation",
                model=model_name,
                input_text=request.task_description,
                metadata={
                    "target_count": request.target_count,
                    "chunk_size": request.chunk_size,
                    "use_redis": request.use_redis,
                    "model": model_name,
                    "endpoint": "large-generation"
                }
            )
    
    system = MultiAgentSystem(conversation_id=request.conversation_id, trace=trace)
    
    async def stream_events_with_tracing():
        collected_output = ""
        final_result = ""
        chunks_processed = 0
        
        try:
            # Choose Redis or in-memory continuity manager
            if request.use_redis:
                # Use Redis-based approach for production
                async for event in system.stream_large_generation_events(
                    query=request.task_description,
                    target_count=request.target_count,
                    chunk_size=request.chunk_size,
                    conversation_history=request.conversation_history
                ):
                    # Collect event data for Langfuse
                    collected_output += json_module.dumps(event) + "\n"
                    
                    # Track chunk processing
                    if event.get('type') == 'chunk_complete':
                        chunks_processed += 1
                    elif event.get('type') == 'generation_complete':
                        final_result = event.get('final_result', '')
                    
                    yield json_module.dumps(event, ensure_ascii=False) + "\n"
            else:
                # Use in-memory approach for testing
                async for event in system.stream_large_generation_events(
                    query=request.task_description,
                    target_count=request.target_count,
                    chunk_size=request.chunk_size,
                    conversation_history=request.conversation_history
                ):
                    # Collect event data for Langfuse
                    collected_output += json_module.dumps(event) + "\n"
                    
                    # Track chunk processing
                    if event.get('type') == 'chunk_complete':
                        chunks_processed += 1
                    elif event.get('type') == 'generation_complete':
                        final_result = event.get('final_result', '')
                    
                    yield json_module.dumps(event, ensure_ascii=False) + "\n"
            
            # Update Langfuse generation and trace with the final output
            if tracer.is_enabled():
                try:
                    generation_output = final_result if final_result else f"Large generation completed - {chunks_processed} chunks processed"
                    
                    # Estimate token usage for cost tracking (large generation can be expensive)
                    usage = tracer.estimate_token_usage(request.task_description, generation_output)
                    
                    # End the generation with results including usage
                    if generation:
                        generation.end(
                            output=generation_output,
                            usage=usage,
                            metadata={
                                "response_length": len(final_result) if final_result else len(collected_output),
                                "source": "large-generation",
                                "streaming": True,
                                "chunks_processed": chunks_processed,
                                "target_count": request.target_count,
                                "conversation_id": request.conversation_id,
                                "use_redis": request.use_redis,
                                "model": model_name,
                                "input_length": len(request.task_description),
                                "output_length": len(generation_output),
                                "estimated_tokens": usage
                            }
                        )
                    
                    # Update trace with final result
                    if trace:
                        trace.update(
                            output=generation_output,
                            metadata={
                                "success": True,
                                "source": "large-generation",
                                "chunks_processed": chunks_processed,
                                "response_length": len(final_result) if final_result else len(collected_output),
                                "streaming": True,
                                "conversation_id": request.conversation_id
                            }
                        )
                    
                    tracer.flush()
                except Exception as e:
                    print(f"[WARNING] Failed to update Langfuse trace/generation: {e}")
            
            # Large generation doesn't use workflow spans - removed incorrect code
                    
        except Exception as e:
            # Update generation and trace with error
            if tracer.is_enabled():
                try:
                    # Estimate usage even for errors
                    error_output = f"Error: {str(e)}"
                    usage = tracer.estimate_token_usage(request.task_description, error_output)
                    
                    # End generation with error
                    if generation:
                        generation.end(
                            output=error_output,
                            usage=usage,
                            metadata={
                                "success": False,
                                "error": str(e),
                                "conversation_id": request.conversation_id,
                                "chunks_processed": chunks_processed,
                                "model": model_name,
                                "estimated_tokens": usage
                            }
                        )
                    
                    # Update trace with error
                    if trace:
                        trace.update(
                            output=f"Error: {str(e)}",
                            metadata={
                                "success": False,
                                "error": str(e),
                                "conversation_id": request.conversation_id,
                                "chunks_processed": chunks_processed
                            }
                        )
                    
                    tracer.flush()
                except:
                    pass  # Don't fail the request if tracing fails
            
            error_event = {
                "type": "large_generation_error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            yield json_module.dumps(error_event, ensure_ascii=False) + "\n"
    
    return StreamingResponse(
        stream_events_with_tracing(),
        media_type="application/json"
    )

@router.get("/large-generation/progress/{session_id}")
async def get_generation_progress(session_id: str):
    """Get progress for an ongoing large generation task"""
    try:
        # Try Redis-based progress first
        from app.agents.redis_continuation_manager import RedisContinuityManager
        
        redis_manager = RedisContinuityManager(session_id=session_id)
        progress = await redis_manager.get_progress_status()
        
        return {
            "session_id": session_id,
            "progress": progress,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "session_id": session_id,
            "error": str(e),
            "available": False,
            "timestamp": datetime.now().isoformat()
        }

@router.post("/large-generation/resume")
async def resume_generation(request: dict):
    """Resume a large generation task from a specific point"""
    session_id = request.get("session_id")
    from_chunk = request.get("from_chunk", 0)
    
    if not session_id:
        return {"error": "session_id is required"}
    
    try:
        system = MultiAgentSystem()
        
        async def stream_resume_events():
            async for event in system.resume_chunked_generation(
                session_id=session_id,
                from_chunk=from_chunk
            ):
                yield json_module.dumps(event, ensure_ascii=False) + "\n"
        
        return StreamingResponse(
            stream_resume_events(),
            media_type="application/json"
        )
        
    except Exception as e:
        error_event = {
            "type": "resume_error",
            "error": str(e),
            "session_id": session_id
        }
        return StreamingResponse(
            iter([json_module.dumps(error_event, ensure_ascii=False) + "\n"]),
            media_type="application/json"
        )

@router.delete("/large-generation/cleanup/{session_id}")
async def cleanup_generation_session(session_id: str):
    """Clean up Redis keys and resources for a completed generation session"""
    try:
        from app.agents.redis_continuation_manager import RedisContinuityManager
        
        redis_manager = RedisContinuityManager(session_id=session_id)
        await redis_manager.cleanup_session()
        
        return {
            "session_id": session_id,
            "status": "cleaned_up",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "session_id": session_id,
            "error": str(e),
            "status": "cleanup_failed",
            "timestamp": datetime.now().isoformat()
        }

def handle_tool_query(request: RAGRequest, routing: Dict):
    """Handle queries that require tool usage"""
    conversation_id = request.conversation_id or request.session_id
    
    def stream():
        try:
            # Send classification info
            yield json_module.dumps({
                "type": "classification",
                "routing": routing,
                "handler": "tool_handler",
                "message": "This query requires external tools."
            }) + "\n"
            
            # Tool execution will be handled by rag_answer internally
            
            yield json_module.dumps({
                "type": "status",
                "message": f"🔧 Tool execution required. Using dynamic tool selection..."
            }) + "\n"
            
            # Let rag_answer handle tool execution internally to avoid duplication
            # This prevents the enhanced question issue that triggers large generation detection
            
            try:
                # Use original question and let rag_answer handle tool execution
                # The service module already has logic to execute tools when query_type = "TOOLS"
                from app.langchain.service import rag_answer
                for chunk in rag_answer(
                    request.question,
                    thinking=request.thinking,
                    stream=True,
                    conversation_id=conversation_id,
                    use_langgraph=False,
                    collections=request.collections,
                    collection_strategy=request.collection_strategy
                ):
                    yield chunk
                    
            except Exception as e:
                # Fallback to regular RAG if tool execution fails
                yield json_module.dumps({
                    "type": "tool_error",
                    "message": f"Tool execution failed: {str(e)}. Falling back to regular response."
                }) + "\n"
                
                from app.langchain.service import rag_answer
                
                for chunk in rag_answer(
                    request.question,
                    thinking=request.thinking,
                    stream=True,
                    conversation_id=conversation_id,
                    use_langgraph=False,
                    collections=request.collections,
                    collection_strategy=request.collection_strategy
                ):
                    yield chunk
                
        except Exception as e:
            logger.error(f"Tool handler error: {e}")
            yield json_module.dumps({"error": f"Tool handling failed: {str(e)}"}) + "\n"
            
    # Create a wrapper to capture the output for Langfuse tracing in direct tool execution
    async def stream_with_tracing():
        logger.info(f"[TRACE DEBUG] Direct tool stream wrapper started")
        collected_output = ""
        final_answer = ""
        source_info = ""
        
        try:
            async for chunk in stream():
                # Collect the streamed data
                if chunk and chunk.strip():
                    collected_output += chunk
                    # logger.info(f"[TRACE DEBUG] Direct tool chunk received: {chunk[:200]}...")  # Commented out to reduce log noise
                    
                    # Try to extract the final answer from the response chunks
                    if '"answer"' in chunk:
                        try:
                            chunk_data = json_module.loads(chunk.strip())
                            if "answer" in chunk_data:
                                final_answer = chunk_data["answer"]
                                # logger.info(f"[TRACE DEBUG] Direct tool extracted final answer: {final_answer[:100]}...")  # Commented out
                                if final_answer:
                                    pass  # logger.info(f"[TRACE DEBUG] Direct tool final answer: {final_answer[:100]}...")  # Commented out
                            if "source" in chunk_data:
                                source_info = chunk_data["source"]
                                logger.info(f"[TRACE DEBUG] Direct tool extracted source info: {source_info}")
                        except Exception as e:
                            logger.warning(f"[TRACE DEBUG] Direct tool failed to parse chunk: {e}")
                            pass  # Continue streaming even if parsing fails
                
                yield chunk
            
            logger.info(f"[TRACE DEBUG] Direct tool stream completed, final answer length: {len(final_answer) if final_answer else 0}")
            
        except Exception as e:
            logger.error(f"Direct tool streaming error: {e}")
            yield json_module.dumps({"error": f"Direct tool streaming failed: {str(e)}"}) + "\n"
    
    return StreamingResponse(stream_with_tracing(), media_type="application/json")

def handle_direct_tool_query(request: RAGRequest, routing: Dict, trace=None):
    """Handle confident tool classifications with direct execution"""
    # logger.info(f"[DIRECT HANDLER DEBUG] Function called! routing: {routing}")  # Commented out to avoid logging entire routing object
    logger.info(f"[DIRECT HANDLER] handle_direct_tool_query called for tool query")
    conversation_id = request.conversation_id or request.session_id
    
    async def stream():
        logger.info(f"[DIRECT HANDLER] stream() function started")
        # Initialize enhanced_question early to avoid UnboundLocalError
        enhanced_question = request.question
        
        try:
            # Store user message in conversation history
            if conversation_id:
                await conversation_manager.add_message(
                    conversation_id=conversation_id,
                    role="user",
                    content=request.question
                )
            
            # Initialize enhanced_question 
            enhanced_question = request.question
            logger.info(f"[DIRECT HANDLER] About to send classification info")
            # Send classification info
            yield json_module.dumps({
                "type": "classification",
                "routing": routing,
                "handler": "direct_tool_execution",
                "message": f"Directly executing tool with {routing['confidence']:.2f} confidence"
            }) + "\n"
            
            # Get the suggested tool from classification
            # logger.info(f"[DIRECT HANDLER] Full routing structure: {routing}")
            suggested_tools = routing.get('suggested_tools', [])
            # Also check nested structure
            if not suggested_tools and 'routing' in routing:
                suggested_tools = routing['routing'].get('suggested_tools', [])
            logger.info(f"[DIRECT HANDLER] Suggested tools: {suggested_tools}")
            
            # Skip conversation history for real-time/current data queries to avoid contamination
            tools_for_current_data = ['get_datetime', 'get_current_time', 'current_time', 'datetime', 'now']  
            is_current_data_query = any(tool in suggested_tools for tool in tools_for_current_data)
            
            logger.info(f"[DIRECT HANDLER] Tools for current data: {tools_for_current_data}")
            logger.info(f"[DIRECT HANDLER] Is current data query: {is_current_data_query}")
            
            # Retrieve conversation history and enhance question with context
            if conversation_id and not is_current_data_query:
                # Get last 3 exchanges (6 messages: 3 user + 3 assistant)
                history = await conversation_manager.get_conversation_history(conversation_id, limit=6)
                if history:
                    # Format history for LLM context
                    enhanced_question = conversation_manager.format_history_for_prompt(history, request.question)
                    logger.info(f"[DIRECT HANDLER] Retrieved {len(history)} messages for conversation {conversation_id}")
                    logger.info(f"[DIRECT HANDLER] Enhanced question with conversation context")
            elif is_current_data_query:
                logger.info(f"[DIRECT HANDLER] Skipping conversation history for current data query to avoid contamination")
                # enhanced_question already set to request.question above
            else:
                logger.info(f"[DIRECT HANDLER] No conversation ID, using original question")
                # enhanced_question already set to request.question above
            if not suggested_tools:
                # Fallback to rag_answer if no tools suggested  
                logger.info(f"[DIRECT HANDLER] No suggested tools found, falling back to regular rag_answer flow")
                yield json_module.dumps({
                    "type": "status", 
                    "message": "No specific tool suggested, falling back to planning..."
                }) + "\n"
                
                from app.langchain.service import rag_answer
                rag_stream = await rag_answer(
                    enhanced_question,  # Use enhanced_question which includes conversation history
                    thinking=request.thinking,
                    stream=True,
                    conversation_id=conversation_id,
                    use_langgraph=request.use_langgraph,
                    collections=request.collections,
                    collection_strategy=request.collection_strategy,
                    query_type="TOOLS",
                    trace=trace
                )
                async for chunk in rag_stream:
                    yield chunk
                return
            
            # Execute tools directly without redundant classification
            logger.info(f"[DIRECT HANDLER] Executing tools directly: {suggested_tools}")
            
            yield json_module.dumps({
                "type": "status", 
                "message": f"Executing {suggested_tools[0]} directly..."
            }) + "\n"
            
            # Import tool execution function
            from app.langchain.service import call_mcp_tool
            
            tool_results = []
            for tool_name in suggested_tools[:1]:  # Execute primary tool only
                try:
                    logger.info(f"[DIRECT HANDLER] Executing tool: {tool_name}")
                    
                    # Get tool schema to determine if parameters are needed
                    from app.core.mcp_tools_cache import get_enabled_mcp_tools
                    available_tools = get_enabled_mcp_tools()
                    tool_info = available_tools.get(tool_name, {})
                    tool_params_schema = tool_info.get('parameters', {})
                    
                    # Check if tool requires parameters based on schema
                    required_params = []
                    if isinstance(tool_params_schema, dict):
                        required_params = tool_params_schema.get('required', [])
                    
                    parameters = {}
                    if not required_params:
                        # Tool needs no parameters - execute immediately
                        logger.info(f"[DIRECT HANDLER] Tool {tool_name} requires no parameters")
                    else:
                        # Tool needs parameters - determine them intelligently
                        yield json_module.dumps({
                            "type": "status", 
                            "message": f"Planning parameters for {tool_name}..."
                        }) + "\n"
                        
                        # Use simple parameter mapping for common patterns
                        if "query" in required_params:
                            parameters["query"] = request.question
                        
                        # For complex tools with multiple required params, use intelligent planning
                        if len(required_params) > 1 or ("query" not in required_params and required_params):
                            from app.langchain.intelligent_tool_planner import get_tool_planner
                            planner = get_tool_planner()
                            
                            execution_plan = await planner.plan_tool_execution(
                                task=request.question,
                                context={"user_query": request.question},
                                mode="standard"
                            )
                            
                            # Find the matching tool in the plan
                            tool_plan = None
                            for planned_tool in execution_plan.tools:
                                if planned_tool.tool_name == tool_name:
                                    tool_plan = planned_tool
                                    break
                            
                            if tool_plan:
                                parameters = tool_plan.parameters
                    
                    # Send progress update before execution
                    yield json_module.dumps({
                        "type": "status", 
                        "message": f"Executing {tool_name}..."
                    }) + "\n"
                    
                    logger.info(f"[DIRECT HANDLER] Using planned parameters for {tool_name}: {parameters}")
                    
                    result = call_mcp_tool(
                        tool_name=tool_name,
                        parameters=parameters,
                        trace=trace
                    )
                    
                    tool_results.append({
                        "tool": tool_name,
                        "success": True,
                        "result": result
                    })
                    
                    yield json_module.dumps({
                        "type": "tool_execution",
                        "tool": tool_name,
                        "success": True,
                        "result": result
                    }) + "\n"
                    
                except Exception as e:
                    logger.error(f"[DIRECT HANDLER] Tool {tool_name} failed: {e}")
                    tool_results.append({
                        "tool": tool_name,
                        "success": False,
                        "error": str(e)
                    })
            
            # Generate final response with tool results
            logger.info(f"[DIRECT HANDLER] Processing {len(tool_results)} tool results")
            logger.info(f"[DEBUG TOOL_RESULTS] Tool results structure: {tool_results}")
            logger.info(f"[DEBUG TOOL_RESULTS] Success check: tool_results={bool(tool_results)}, any_success={any(r.get('success') for r in tool_results) if tool_results else False}")
            if tool_results and any(r.get('success') for r in tool_results):
                try:
                    # Send progress update before synthesis
                    yield json_module.dumps({
                        "type": "status", 
                        "message": "Generating response from tool results..."
                    }) + "\n"
                    
                    # Build context from tool results and extract documents
                    tool_context = ""
                    extracted_documents = []
                    
                    for tr in tool_results:
                        if tr.get('success'):
                            # Extract readable text from tool result
                            tool_name = tr['tool']
                            raw_result = tr['result']
                            
                            # Format tool result in a readable way
                            if isinstance(raw_result, dict) and 'content' in raw_result:
                                # Extract text from content array
                                content_parts = []
                                for item in raw_result['content']:
                                    if isinstance(item, dict) and item.get('type') == 'text':
                                        content_parts.append(item.get('text', ''))
                                readable_result = ' '.join(content_parts) if content_parts else str(raw_result)
                            else:
                                readable_result = str(raw_result)
                            
                            tool_context += f"\n{tool_name}: {readable_result}\n"
                            logger.info(f"[DIRECT HANDLER] Formatted {tool_name} result: {readable_result}")
                            
                            # Extract documents from RAG search results
                            # Import is_rag_tool function for flexible tool name matching
                            from app.langchain.service import is_rag_tool
                            if is_rag_tool(tr['tool']) and isinstance(tr.get('result'), dict):
                                # Parse JSON-RPC response
                                result_data = tr['result']
                                if 'result' in result_data and isinstance(result_data['result'], dict):
                                    rag_result = result_data['result']
                                    if 'documents' in rag_result:
                                        extracted_documents.extend(rag_result['documents'])
                    
                    logger.info(f"[DIRECT HANDLER] Tool context built: {tool_context[:200]}...")
                    logger.info(f"[DIRECT HANDLER] Extracted {len(extracted_documents)} documents from tool results")
                    
                    # Use unified synthesis with proper message format
                    from app.langchain.service import build_messages_for_synthesis
                    
                    logger.info(f"[DEBUG CALL] About to call build_messages_for_synthesis with question='{enhanced_question[:100]}...', tool_context='{tool_context[:100]}...'")
                    
                    try:
                        messages, source_label, full_context, system_prompt = build_messages_for_synthesis(
                            question=enhanced_question,
                            query_type="TOOLS",
                            tool_context=tool_context,
                            thinking=request.thinking if hasattr(request, 'thinking') else False
                        )
                        logger.info(f"[DEBUG CALL] build_messages_for_synthesis returned successfully, messages count: {len(messages)}")
                    except Exception as e:
                        logger.error(f"[DEBUG CALL] build_messages_for_synthesis failed with error: {e}")
                        raise
                    
                    # For backward compatibility, create a single prompt
                    synthesis_prompt = "\n\n".join([msg["content"] for msg in messages])
                    
                    logger.info(f"[DIRECT HANDLER] About to stream synthesis response")
                    
                    # Get LLM settings first
                    from app.llm.ollama import OllamaLLM
                    from app.llm.base import LLMConfig
                    from app.core.llm_settings_cache import get_llm_settings, get_main_llm_full_config
                    
                    llm_settings = get_llm_settings()
                    
                    # Apply dynamic detection and model fallback for main LLM configuration
                    main_llm_base = llm_settings.get('main_llm', {})
                    model_name = main_llm_base.get('model', '')
                    
                    # Disable automatic fallback to investigate the real issue
                    model_fallback_applied = False
                    original_model = model_name
                    
                    # DISABLED FALLBACK - Let's find the real issue
                    # if 'qwen3:30b-a3b-instruct-2507-q4_K_M' in model_name:
                    #     fallback_model = 'qwen3:30b-a3b-q4_K_M'
                    #     logger.warning(f"[MAIN LLM FALLBACK] Detected problematic model {model_name}, switching to {fallback_model}")
                    #     # ... fallback logic disabled
                    
                    if True:  # Always use original model now
                        # Try to get cached behavior or use model name heuristics for other models
                        detected_mode = None
                        try:
                            from app.llm.response_analyzer import response_analyzer
                            cached_profile = response_analyzer._get_cached_behavior(model_name)
                            if cached_profile and cached_profile.confidence > 0.7:
                                detected_mode = 'thinking' if cached_profile.is_thinking_model else 'non-thinking'
                                logger.info(f"[MAIN LLM DETECTION] Using cached behavior for {model_name}: {detected_mode} (confidence: {cached_profile.confidence:.2f})")
                            else:
                                # Use model name heuristics for instruct models
                                if 'instruct' in model_name.lower() and '2507' in model_name:
                                    detected_mode = 'non-thinking'
                                    logger.info(f"[MAIN LLM DETECTION] Using heuristic detection for {model_name}: {detected_mode}")
                        except Exception as e:
                            logger.warning(f"[MAIN LLM DETECTION] Failed to apply dynamic detection: {e}")
                        
                        main_llm_config = get_main_llm_full_config(llm_settings, override_mode=detected_mode)
                    
                    # Log effective configuration
                    logger.info(f"[MAIN LLM CONFIG] Original model: {original_model}")
                    logger.info(f"[MAIN LLM CONFIG] Final model: {main_llm_config.get('model')}")
                    logger.info(f"[MAIN LLM CONFIG] Model fallback applied: {model_fallback_applied}")
                    logger.info(f"[MAIN LLM CONFIG] Effective mode: {main_llm_config.get('effective_mode', 'unknown')}")
                    logger.info(f"[MAIN LLM CONFIG] Mode overridden: {main_llm_config.get('mode_overridden', False)}")
                    logger.info(f"[MAIN LLM CONFIG] Temperature: {main_llm_config.get('temperature', 'not_set')}")
                    logger.info(f"[MAIN LLM CONFIG] Top-p: {main_llm_config.get('top_p', 'not_set')}")
                    logger.info(f"[MAIN LLM CONFIG] Max tokens: {main_llm_config.get('max_tokens', 'not_set')}")
                    
                    # Create synthesis generation span for Langfuse tracing
                    synthesis_generation_span = None
                    if trace:
                        try:
                            from app.core.langfuse_integration import get_tracer
                            tracer = get_tracer()
                            if tracer.is_enabled():
                                synthesis_generation_span = tracer.create_llm_generation_span(
                                    trace,
                                    model=main_llm_config.get('model'),
                                    prompt=messages,  # Pass messages instead of single prompt
                                    operation="direct_tool_synthesis"
                                )
                                logger.info(f"[DIRECT HANDLER] Created synthesis generation span")
                        except Exception as e:
                            logger.warning(f"Failed to create synthesis generation span: {e}")
                    
                    # Configure LLM for streaming using refactored config
                    max_tokens_from_config = main_llm_config.get('max_tokens', 4000)
                    print(f"[DEBUG] Direct handler max_tokens from config: {max_tokens_from_config}")
                    llm_config = LLMConfig(
                        model_name=main_llm_config.get('model'),
                        temperature=float(main_llm_config.get('temperature', 0.8)),
                        max_tokens=int(max_tokens_from_config),
                        top_p=float(main_llm_config.get('top_p', 0.9))
                    )
                    
                    # Use same model server detection as service
                    import os
                    model_server = os.environ.get("OLLAMA_BASE_URL")
                    if not model_server:
                        model_server = main_llm_config.get('model_server', '').strip()
                        if not model_server:
                            model_server = "http://ollama:11434"
                    
                    llm = OllamaLLM(llm_config, base_url=model_server)
                    
                    # Stream synthesis response token by token using chat with messages
                    final_response = ""
                    # Log system prompt being used
                    logger.info(f"[DIRECT HANDLER] Using system prompt: {system_prompt[:100]}...")
                    logger.info(f"[DIRECT HANDLER] Full system prompt length: {len(system_prompt)}")
                    logger.info(f"[DIRECT HANDLER] Messages array length: {len(messages)}")
                    for i, msg in enumerate(messages):
                        logger.info(f"[DIRECT HANDLER] Message {i} - Role: {msg.get('role')}, Content length: {len(msg.get('content', ''))}")
                        if msg.get('role') == 'system':
                            logger.info(f"[DIRECT HANDLER] System message preview: {msg.get('content', '')[:200]}...")
                    
                    # Use chat_stream with messages instead of generate_stream
                    # Track response for dynamic behavior detection
                    response_tokens = []
                    first_analysis_done = False
                    chunk_count = 0
                    
                    logger.info(f"[SYNTHESIS DEBUG] Starting LLM streaming with model: {main_llm_config.get('model')}")
                    logger.info(f"[SYNTHESIS DEBUG] LLM config - temp: {main_llm_config.get('temperature')}, top_p: {main_llm_config.get('top_p')}, max_tokens: {main_llm_config.get('max_tokens')}")
                    
                    async for response_chunk in llm.chat_stream(messages):
                        chunk_count += 1
                        chunk_text = response_chunk.text
                        final_response += chunk_text
                        response_tokens.append(chunk_text)
                        
                        # Log first few chunks for debugging
                        if chunk_count <= 5:
                            logger.info(f"[SYNTHESIS DEBUG] Chunk {chunk_count}: '{chunk_text}' (length: {len(chunk_text)})")
                        
                        # Log every 100 chunks to track progress
                        if chunk_count % 100 == 0:
                            logger.info(f"[SYNTHESIS DEBUG] Processed {chunk_count} chunks, total length: {len(final_response)}")
                        
                        # Early termination detection
                        if chunk_count > 10 and len(final_response) < 200:
                            logger.warning(f"[SYNTHESIS DEBUG] Potential early termination detected at chunk {chunk_count} with only {len(final_response)} characters")
                        
                        # Analyze first 10 tokens for thinking behavior detection
                        if not first_analysis_done and len(response_tokens) >= 10:
                            first_analysis_done = True
                            try:
                                from app.llm.response_analyzer import detect_model_thinking_behavior
                                sample_text = ''.join(response_tokens[:20])  # First 20 tokens
                                model_name = main_llm_config.get('model', 'unknown')
                                is_thinking, confidence = detect_model_thinking_behavior(sample_text, model_name)
                                
                                logger.info(f"[DYNAMIC DETECTION] Model: {model_name}")
                                logger.info(f"[DYNAMIC DETECTION] Detected thinking behavior: {is_thinking} (confidence: {confidence:.2f})")
                                logger.info(f"[DYNAMIC DETECTION] Sample analyzed: {sample_text[:100]}...")
                                
                                # Log if detection differs from configured mode
                                configured_mode = main_llm_config.get('mode', 'thinking')
                                expected_thinking = configured_mode == 'thinking'
                                if is_thinking != expected_thinking and confidence > 0.7:
                                    logger.warning(f"[DYNAMIC DETECTION] Mode mismatch! Configured: {configured_mode}, Detected: {'thinking' if is_thinking else 'non-thinking'}")
                                    
                            except Exception as e:
                                logger.warning(f"[DYNAMIC DETECTION] Analysis failed: {e}")
                        
                        if response_chunk.text.strip():
                            yield json_module.dumps({
                                "token": response_chunk.text
                            }) + "\n"
                    
                    logger.info(f"[DIRECT HANDLER] Synthesis completed, response length: {len(final_response)}")
                    logger.info(f"[SYNTHESIS DEBUG] Final stats - Total chunks: {chunk_count}, Final response length: {len(final_response)}")
                    logger.info(f"[SYNTHESIS DEBUG] Response preview: {final_response[:300]}...")
                    logger.info(f"[SYNTHESIS DEBUG] Response ends with: ...{final_response[-100:] if len(final_response) > 100 else final_response}")
                    
                    # End synthesis generation span with output
                    if synthesis_generation_span:
                        try:
                            tracer = get_tracer()
                            usage = tracer.estimate_token_usage(synthesis_prompt, final_response)
                            synthesis_generation_span.end(
                                output=final_response,
                                usage=usage,
                                metadata={
                                    "response_length": len(final_response),
                                    "operation": "direct_tool_synthesis",
                                    "tool_results_count": len(tool_results),
                                    "optimization": "direct_execution"
                                }
                            )
                            logger.info(f"[DIRECT HANDLER] Synthesis generation span ended with output")
                        except Exception as e:
                            logger.warning(f"Failed to end synthesis generation span: {e}")
                    
                    # Process final response based on detected model behavior
                    processed_response = final_response
                    try:
                        from app.llm.response_analyzer import detect_model_thinking_behavior
                        model_name = main_llm_config.get('model', 'unknown')
                        is_thinking, confidence = detect_model_thinking_behavior(final_response, model_name)
                        
                        if is_thinking and confidence > 0.8:
                            # Remove thinking tags from response
                            import re
                            processed_response = re.sub(r'<think>.*?</think>', '', final_response, flags=re.DOTALL | re.IGNORECASE)
                            processed_response = processed_response.strip()
                            
                            logger.info(f"[RESPONSE PROCESSING] Removed thinking tags from response")
                            logger.info(f"[RESPONSE PROCESSING] Original length: {len(final_response)}, Processed length: {len(processed_response)}")
                        
                    except Exception as e:
                        logger.warning(f"[RESPONSE PROCESSING] Failed to process thinking tags: {e}")
                        processed_response = final_response
                    
                    # Send final answer with documents
                    response_data = {
                        "answer": processed_response,
                        "source": "DIRECT_TOOL_EXECUTION", 
                        "context": tool_context,
                        "query_type": "TOOLS",
                        "metadata": {
                            "tools_executed": [tr['tool'] for tr in tool_results if tr.get('success')],
                            "direct_execution": True,
                            "optimization": "eliminated_redundant_classification"
                        }
                    }
                    
                    # Add documents if any were extracted
                    if extracted_documents:
                        response_data["documents"] = extracted_documents
                        response_data["metadata"]["documents_found"] = len(extracted_documents)
                    
                    yield json_module.dumps(response_data) + "\n"
                    
                    logger.info(f"[DIRECT HANDLER] Final answer sent to frontend")
                    
                    # Save assistant response to conversation history (use processed response)
                    if conversation_id and processed_response:
                        await conversation_manager.add_message(
                            conversation_id=conversation_id,
                            role="assistant",
                            content=processed_response,
                            metadata={"source": "DIRECT_TOOL_EXECUTION", "tools_executed": [tr['tool'] for tr in tool_results if tr.get('success')]}
                        )
                    
                except Exception as synthesis_error:
                    logger.error(f"[DIRECT HANDLER] Synthesis failed: {synthesis_error}")
                    yield json_module.dumps({
                        "type": "error",
                        "message": f"Response synthesis failed: {str(synthesis_error)}"
                    }) + "\n"
            else:
                # Fallback if tools failed
                logger.warning(f"[DIRECT HANDLER] No successful tool results")
                yield json_module.dumps({
                    "type": "error",
                    "message": "Tool execution failed, please try again"
                }) + "\n"
                    
        except Exception as e:
            yield json_module.dumps({
                "type": "handler_error",
                "error": str(e),
                "message": "Direct tool execution failed, falling back to planning..."
            }) + "\n"
            
            # Fallback to regular rag_answer
            from app.langchain.service import rag_answer
            rag_stream = await rag_answer(
                enhanced_question,
                thinking=request.thinking,
                stream=True,
                conversation_id=conversation_id,
                use_langgraph=request.use_langgraph,
                collections=request.collections,
                collection_strategy=request.collection_strategy,
                query_type="TOOLS",
                trace=trace
            )
            async for chunk in rag_stream:
                yield chunk
                    
        except Exception as e:
            yield json_module.dumps({
                "type": "handler_error",
                "error": str(e),
                "message": "Direct tool handler failed completely"
            }) + "\n"
    
    return StreamingResponse(stream(), media_type="application/json")

def handle_direct_llm_query(request: RAGRequest, routing: Dict):
    """Handle queries that can be answered directly by LLM without RAG"""
    conversation_id = request.conversation_id or request.session_id
    
    def stream():
        try:
            # Send classification info
            yield json_module.dumps({
                "type": "classification",
                "routing": routing,
                "handler": "direct_llm",
                "message": "This is a general query. Using direct LLM response."
            }) + "\n"
            
            # Use the service module but skip RAG retrieval
            from app.langchain.service import get_llm_response_direct
            
            # Stream direct LLM response
            chunk_count = 0
            for chunk in get_llm_response_direct(
                request.question,
                thinking=request.thinking,
                stream=True,
                conversation_id=conversation_id
            ):
                if chunk:
                    chunk_count += 1
                    yield chunk
                    
        except Exception as e:
            logger.error(f"Direct LLM handler error: {e}")
            yield json_module.dumps({"error": f"Direct LLM handling failed: {str(e)}"}) + "\n"
            
    return StreamingResponse(stream(), media_type="application/json")

def handle_multi_agent_query(request: RAGRequest, routing: Dict):
    """Handle complex queries that need multiple agents"""
    conversation_id = request.conversation_id or request.session_id
    
    async def stream():
        try:
            # Send classification info
            yield json_module.dumps({
                "type": "classification",
                "routing": routing,
                "handler": "langgraph_multi_agent",
                "message": "This is a complex query. Using LangGraph multi-agent system."
            }) + "\n"
            
            # Use LangGraph multi-agent system
            from app.langchain.langgraph_multi_agent_system import langgraph_multi_agent_answer
            
            yield json_module.dumps({
                "type": "status",
                "message": "🤖 Initializing LangGraph multi-agent workflow..."
            }) + "\n"
            
            # Process with LangGraph system
            result = await langgraph_multi_agent_answer(
                question=request.question,
                conversation_id=conversation_id
            )
            
            # Stream the results
            yield json_module.dumps({
                "type": "agent_selection",
                "agents_selected": result.get("agents_used", []),
                "execution_pattern": result.get("execution_pattern", ""),
                "selection_reasoning": result.get("agent_selection_reasoning", ""),
                "confidence_score": result.get("confidence_score", 0.0)
            }) + "\n"
            
            yield json_module.dumps({
                "type": "status",
                "message": f"🔄 Executing {result.get('execution_pattern', 'sequential')} collaboration pattern..."
            }) + "\n"
            
            # Stream final response
            final_answer = result.get("answer", "")
            
            # Stream token by token for consistency with other handlers
            import asyncio
            for i in range(0, len(final_answer), 10):  # Stream in chunks of 10 characters
                chunk = final_answer[i:i+10]
                yield json_module.dumps({
                    "token": chunk
                }) + "\n"
                await asyncio.sleep(0.01)  # Small delay for streaming effect
            
            # Send final answer
            yield json_module.dumps({
                "answer": final_answer,
                "source": "LANGGRAPH_MULTI_AGENT",
                "agents_used": result.get("agents_used", []),
                "execution_pattern": result.get("execution_pattern", ""),
                "confidence_score": result.get("confidence_score", 0.0),
                "conversation_id": conversation_id,
                "metadata": result.get("metadata", {})
            }) + "\n"
                
        except Exception as e:
            logger.error(f"LangGraph multi-agent handler error: {e}")
            yield json_module.dumps({"error": f"LangGraph multi-agent handling failed: {str(e)}"}) + "\n"
            
    return StreamingResponse(stream(), media_type="application/json")

def handle_hybrid_query(request: RAGRequest, routing: Dict):
    """Handle hybrid queries that require multiple components (Tools + RAG + LLM)"""
    conversation_id = request.conversation_id or request.session_id
    
    async def stream():
        try:
            # Store user message in conversation history
            if conversation_id:
                await conversation_manager.add_message(
                    conversation_id=conversation_id,
                    role="user",
                    content=request.question
                )
            
            # Retrieve conversation history and enhance question with context
            enhanced_question = request.question
            if conversation_id:
                # Get last 3 exchanges (6 messages: 3 user + 3 assistant)
                history = await conversation_manager.get_conversation_history(conversation_id, limit=6)
                if history:
                    # Format history for LLM context
                    enhanced_question = conversation_manager.format_history_for_prompt(history, request.question)
                    logger.info(f"[HYBRID HANDLER] Enhanced question with conversation context")
            
            # Send classification info
            yield json_module.dumps({
                "type": "classification",
                "routing": routing,
                "handler": "hybrid",
                "message": f"This is a hybrid query requiring: {routing['primary_type']}"
            }) + "\n"
            
            # Extract component types
            primary_type = routing['primary_type']
            components = []
            
            # Determine which components are needed
            if primary_type == QueryType.TOOL_RAG.value:
                components = ["tool", "rag"]
            elif primary_type == QueryType.TOOL_LLM.value:
                components = ["tool", "llm"]
            elif primary_type == QueryType.RAG_LLM.value:
                components = ["rag", "llm"]
            elif primary_type == QueryType.TOOL_RAG_LLM.value:
                components = ["tool", "rag", "llm"]
            
            # Collect results from each component
            results = {}
            
            # 1. Tool component - Get real-time data
            if "tool" in components:
                yield json_module.dumps({
                    "type": "status",
                    "message": "🔧 Searching for real-time information..."
                }) + "\n"
                
                # Get MCP tools
                from app.core.mcp_tools_cache import get_enabled_mcp_tools
                available_tools = get_enabled_mcp_tools()
                
                # TODO: Execute appropriate tools based on query
                # For now, we'll indicate tool execution
                tool_results = {
                    "status": "Tool search would be executed here",
                    "suggested_tools": routing.get("routing", {}).get("suggested_tools", [])
                }
                results["tool"] = tool_results
                
                yield json_module.dumps({
                    "type": "tool_result",
                    "data": tool_results
                }) + "\n"
            
            # 2. RAG component - Search knowledge base
            if "rag" in components:
                yield json_module.dumps({
                    "type": "status",
                    "message": "📚 Searching knowledge base..."
                }) + "\n"
                
                # Get RAG results using the service
                from app.langchain.service import handle_rag_query
                
                try:
                    # Get RAG results
                    context, _ = handle_rag_query(
                        question=request.question,
                        thinking=request.thinking,
                        collections=request.collections or ["documents"],
                        collection_strategy=request.collection_strategy or "auto"
                    )
                    
                    results = {
                        "documents": context if isinstance(context, list) else [],
                        "context": context if isinstance(context, str) else ""
                    }
                    
                    # Format context
                    rag_context = "\n\n".join([
                        f"Source {i+1}:\n{result['text'][:500]}..."
                        for i, result in enumerate(results[:3])
                    ]) if results else None
                    
                except Exception as e:
                    logger.warning(f"RAG retrieval error: {e}")
                    rag_context = None
                
                results["rag"] = rag_context
                
                if rag_context:
                    yield json_module.dumps({
                        "type": "rag_result",
                        "data": {
                            "context_found": True,
                            "num_sources": len(rag_context.split('\n\n')) if isinstance(rag_context, str) else 0
                        }
                    }) + "\n"
            
            # 3. Synthesize with LLM
            yield json_module.dumps({
                "type": "status",
                "message": "🤖 Synthesizing comprehensive answer..."
            }) + "\n"
            
            # Build enhanced prompt with all component results
            enhanced_prompt = build_hybrid_prompt(enhanced_question, results, components)
            
            # Stream the final synthesized answer using rag_answer
            from app.langchain.service import rag_answer
            
            # Create hybrid context with the results
            hybrid_synthesis_context = {
                "sources": [],
                "strategy": "hybrid_synthesis",
                "pre_formatted_context": enhanced_prompt,
                "component_results": results
            }
            
            # Use rag_answer with the enhanced question and pass context via hybrid_context
            final_answer = ""
            async for chunk in rag_answer(
                enhanced_question,  # Pass enhanced_question to maintain conversation context
                thinking=request.thinking,
                stream=True,
                conversation_id=conversation_id,
                use_langgraph=False,  # Direct LLM for synthesis
                collections=request.collections,
                collection_strategy="none",  # We already have context
                query_type="SYNTHESIS",  # Use SYNTHESIS mode to skip additional searches
                hybrid_context=hybrid_synthesis_context
            ):
                # Extract token from chunk for accumulation
                if isinstance(chunk, str):
                    try:
                        data = json_module.loads(chunk.strip())
                        if data.get("token"):
                            final_answer += data["token"]
                        elif data.get("answer"):
                            final_answer = data["answer"]
                    except:
                        pass
                yield chunk
            
            # Save assistant response to conversation history
            if conversation_id and final_answer:
                await conversation_manager.add_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=final_answer,
                    metadata={"source": "hybrid_handler", "components": components}
                )
                
        except Exception as e:
            logger.error(f"Hybrid handler error: {e}")
            yield json_module.dumps({"error": f"Hybrid handling failed: {str(e)}"}) + "\n"
    
    return StreamingResponse(stream(), media_type="application/json")

def build_hybrid_prompt(question: str, results: Dict, components: List[str]) -> str:
    """Build an enhanced prompt that combines results from all components"""
    prompt_parts = [f"User Question: {question}\n"]
    
    if "tool" in components and "tool" in results:
        tool_data = results["tool"]
        prompt_parts.append("\n## Real-time Information:")
        if tool_data.get("suggested_tools"):
            prompt_parts.append(f"Suggested tools: {', '.join(tool_data['suggested_tools'])}")
        prompt_parts.append("(Tool execution results would appear here)")
    
    if "rag" in components and "rag" in results:
        rag_data = results["rag"]
        if rag_data:
            prompt_parts.append("\n## Knowledge Base Context:")
            prompt_parts.append(str(rag_data))
    
    prompt_parts.append("\n## Task:")
    prompt_parts.append("Based on the above information, provide a comprehensive answer that:")
    prompt_parts.append("1. Incorporates relevant real-time data (if available)")
    prompt_parts.append("2. Uses knowledge from the knowledge base (if available)")
    prompt_parts.append("3. Synthesizes all information into a coherent response")
    prompt_parts.append("4. Clearly indicates which information comes from which source")
    
    return "\n".join(prompt_parts)

# ===== TEMPORARY DOCUMENT ENDPOINTS =====

@router.post("/upload-document")
async def upload_temp_document(
    file: UploadFile = File(...),
    conversation_id: str = Form(...),
    ttl_hours: int = Form(2),
    auto_include: bool = Form(True)
):
    """Upload a temporary document for conversation-scoped indexing."""
    try:
        # Read file content
        file_content = await file.read()
        
        # Process document
        result = await temp_doc_manager.upload_and_process_document(
            file_content=file_content,
            filename=file.filename,
            conversation_id=conversation_id,
            ttl_hours=ttl_hours,
            auto_include=auto_include
        )
        
        if result['success']:
            return {
                "success": True,
                "temp_doc_id": result['temp_doc_id'],
                "message": f"Document '{file.filename}' uploaded and indexed successfully",
                "metadata": result['metadata']
            }
        else:
            raise HTTPException(status_code=400, detail=result['error'])
            
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}")

@router.get("/temp-documents/{conversation_id}")
async def get_temp_documents(conversation_id: str):
    """Get all temporary documents for a conversation."""
    try:
        documents = await temp_doc_manager.get_conversation_documents(conversation_id)
        return {
            "conversation_id": conversation_id,
            "documents": documents,
            "total_count": len(documents),
            "active_count": len([doc for doc in documents if doc.get('is_included', False)])
        }
    except Exception as e:
        logger.error(f"Failed to get temp documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/temp-documents/{temp_doc_id}")
async def delete_temp_document(temp_doc_id: str):
    """Delete a temporary document."""
    try:
        success = await temp_doc_manager.delete_document(temp_doc_id)
        return {"success": success}
    except Exception as e:
        logger.error(f"Failed to delete temp document: {str(e)}")
        return {"success": True}  # Idempotent - return success even if not found

@router.put("/temp-documents/{temp_doc_id}/preferences")
async def update_temp_document_preferences(
    temp_doc_id: str,
    preferences: Dict[str, Any]
):
    """Update preferences for a temporary document."""
    try:
        success = await temp_doc_manager.update_document_preferences(temp_doc_id, preferences)
        if success:
            return {"success": True, "message": "Preferences updated"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(f"Failed to update preferences: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/conversation/{conversation_id}")
async def clear_conversation_history(conversation_id: str):
    """Clear conversation history from Redis for a specific conversation ID."""
    try:
        await conversation_manager.clear_conversation(conversation_id)
        return {"success": True, "message": f"Conversation {conversation_id} cleared successfully"}
    except Exception as e:
        logger.error(f"Failed to clear conversation {conversation_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear conversation: {str(e)}") 