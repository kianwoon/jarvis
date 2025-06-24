"""
Intelligent Chat Endpoint - Rearchitected Standard Chat
Uses LLM-driven decision making with structured function calling
No hardcoded JSON parsing - lets LLM decide what tools/RAG to use
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import logging
import asyncio
from datetime import datetime

from app.core.llm_settings_cache import get_llm_settings
from app.core.mcp_tools_cache import get_enabled_mcp_tools
from app.core.collection_registry_cache import get_all_collections
from app.core.query_classifier_settings_cache import get_query_classifier_settings
from app.llm.ollama import OllamaLLM
from app.llm.base import LLMConfig
from app.langchain.enhanced_query_classifier import EnhancedQueryClassifier, QueryType

logger = logging.getLogger(__name__)
router = APIRouter()

class IntelligentChatRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    thinking: bool = False

class IntelligentChatResponse(BaseModel):
    answer: str
    tool_calls: List[Dict[str, Any]] = []
    rag_searches: List[Dict[str, Any]] = []
    conversation_id: Optional[str] = None
    confidence: float = 0.0
    classification: str = ""

async def execute_mcp_tool(tool_name: str, parameters: Dict[str, Any], trace=None, parent_span=None) -> Dict[str, Any]:
    """Phase 2: Execute MCP tool with just-in-time schema loading and validation"""
    # Import here to avoid circular imports
    from app.langchain.service import call_mcp_tool
    from app.core.langfuse_integration import get_tracer
    
    tracer = get_tracer()
    tool_span = None
    
    # Create tool execution span if tracing is enabled and trace is provided
    if trace and tracer.is_enabled():
        print(f"[DEBUG] Standard chat creating tool span for {tool_name}")
        # Use parent_span if provided, otherwise use trace
        span_parent = parent_span if parent_span else trace
        tool_span = tracer.create_tool_span(span_parent, tool_name, parameters)
        print(f"[DEBUG] Standard chat tool span created: {tool_span is not None}")
    else:
        print(f"[DEBUG] Standard chat no tool span created - trace: {trace is not None}, tracer enabled: {tracer.is_enabled()}")
    
    try:
        logger.info(f"Phase 2 - Executing MCP tool: {tool_name} with parameters: {parameters}")
        
        # Just-in-time: Load full tool schema only when executing
        tools = get_enabled_mcp_tools()
        if tool_name not in tools:
            raise ValueError(f"Tool {tool_name} not available or not enabled")
        
        tool_info = tools[tool_name]
        
        # Get full parameter schema for execution-time validation
        tool_schema = tool_info.get('parameters', {})
        
        # Enhanced parameter validation with full schema context
        validated_params = validate_nested_parameters_with_schema(parameters, tool_schema, tool_name)
        
        logger.info(f"Phase 2 - Validated parameters for {tool_name}: {validated_params}")
        
        # Call the tool using existing infrastructure
        result = call_mcp_tool(tool_name, validated_params)
        
        # Check if result indicates an error
        success = True
        if isinstance(result, dict):
            if "error" in result:
                success = False
            elif "message" in result and "error" in str(result.get("message", "")).lower():
                success = False
        
        # End tool span with result
        if tool_span:
            print(f"[DEBUG] Standard chat ending tool span for {tool_name} with success: {success}")
            tracer.end_span_with_result(tool_span, result, success)
        
        # Return in same format as existing system
        return {
            "tool": tool_name,
            "parameters": validated_params,
            "result": result,
            "success": success
        }
    except Exception as e:
        logger.error(f"Phase 2 - MCP tool execution failed for {tool_name}: {e}")
        
        # End tool span with error
        if tool_span:
            print(f"[DEBUG] Standard chat ending tool span for {tool_name} with error: {str(e)}")
            tracer.end_span_with_result(tool_span, None, False, str(e))
        
        return {
            "tool": tool_name,
            "parameters": parameters,
            "error": str(e),
            "success": False
        }

def validate_nested_parameters_with_schema(parameters: Dict[str, Any], schema: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
    """Phase 2: Enhanced parameter validation with full schema context and logging"""
    if not schema or not isinstance(schema, dict):
        logger.warning(f"No schema available for tool {tool_name}, using parameters as-is")
        return parameters
    
    try:
        properties = schema.get('properties', {})
        required = schema.get('required', [])
        validated = {}
        
        logger.info(f"Validating parameters for {tool_name} - Required: {required}, Available properties: {list(properties.keys())}")
        
        # Validate each parameter
        for param_name, param_value in parameters.items():
            if param_name in properties:
                param_schema = properties[param_name]
                param_type = param_schema.get('type', 'string')
                
                # Handle nested objects
                if param_type == 'object' and isinstance(param_value, dict):
                    validated[param_name] = validate_nested_parameters_with_schema(param_value, param_schema, f"{tool_name}.{param_name}")
                # Handle arrays
                elif param_type == 'array' and isinstance(param_value, list):
                    validated[param_name] = param_value  # Basic array validation
                else:
                    validated[param_name] = param_value
            else:
                # Allow extra parameters for flexibility
                validated[param_name] = param_value
                logger.info(f"Parameter '{param_name}' not in schema for {tool_name}, allowing as extra parameter")
        
        # Check required parameters
        missing_required = []
        for req_param in required:
            if req_param not in validated:
                missing_required.append(req_param)
        
        if missing_required:
            logger.warning(f"Tool {tool_name} missing required parameters: {missing_required}")
        
        return validated
        
    except Exception as e:
        logger.warning(f"Parameter validation failed for {tool_name}: {e}, using original parameters")
        return parameters

def validate_nested_parameters(parameters: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """Validate parameters against MCP tool schema with nested object support"""
    if not schema or not isinstance(schema, dict):
        return parameters
    
    try:
        properties = schema.get('properties', {})
        required = schema.get('required', [])
        validated = {}
        
        # Validate each parameter
        for param_name, param_value in parameters.items():
            if param_name in properties:
                param_schema = properties[param_name]
                param_type = param_schema.get('type', 'string')
                
                # Handle nested objects
                if param_type == 'object' and isinstance(param_value, dict):
                    validated[param_name] = validate_nested_parameters(param_value, param_schema)
                # Handle arrays
                elif param_type == 'array' and isinstance(param_value, list):
                    validated[param_name] = param_value  # Basic array validation
                else:
                    validated[param_name] = param_value
            else:
                # Allow extra parameters for flexibility
                validated[param_name] = param_value
        
        # Check required parameters
        for req_param in required:
            if req_param not in validated:
                logger.warning(f"Required parameter '{req_param}' missing")
        
        return validated
        
    except Exception as e:
        logger.warning(f"Parameter validation failed: {e}, using original parameters")
        return parameters

async def execute_rag_search(query: str, collections: List[str] = None, trace=None, parent_span=None) -> Dict[str, Any]:
    """Phase 2: Execute RAG search with just-in-time collection validation"""
    # Import here to avoid circular imports
    from app.langchain.service import handle_rag_query
    from app.core.langfuse_integration import get_tracer
    
    tracer = get_tracer()
    rag_span = None
    
    # Create RAG search span if tracing is enabled and trace is provided
    if trace and tracer.is_enabled():
        # Use parent_span if provided, otherwise use trace
        span_parent = parent_span if parent_span else trace
        rag_span = tracer.create_rag_span(span_parent, query, collections)
    
    try:
        logger.info(f"Phase 2 - Performing RAG search for: {query[:100]}...")
        
        # Just-in-time: Validate collections against available collections
        if collections:
            available_collections = get_all_collections()
            available_names = [c.get('collection_name', '') for c in available_collections] if available_collections else []
            
            # Validate requested collections exist
            validated_collections = []
            for requested_collection in collections:
                if requested_collection in available_names:
                    validated_collections.append(requested_collection)
                else:
                    logger.warning(f"Requested collection '{requested_collection}' not found. Available: {available_names}")
            
            if not validated_collections:
                logger.warning(f"No valid collections found from request: {collections}, using auto strategy")
                collections = None
            else:
                collections = validated_collections
                logger.info(f"Phase 2 - Validated collections: {collections}")
        
        # Use existing RAG query handler to get context and sources
        context, sources = handle_rag_query(
            question=query,
            thinking=False,
            collections=collections,
            collection_strategy="specific" if collections else "auto"
        )
        
        # Parse sources to extract document information
        documents = []
        if sources:
            for source in sources[:5]:  # Limit to top 5 sources
                doc_info = {
                    "content": source.get("content", "")[:500] + "..." if len(source.get("content", "")) > 500 else source.get("content", ""),
                    "source": source.get("source", "Unknown"),
                    "relevance_score": source.get("relevance_score", 0.0),
                    "metadata": {
                        "page": source.get("page"),
                        "doc_id": source.get("doc_id"),
                        "collection": source.get("collection_name")
                    }
                }
                documents.append(doc_info)
        
        result = {
            "query": query,
            "collections": collections or "auto",
            "context": context[:1000] + "..." if len(context) > 1000 else context,  # Truncate for response
            "documents": documents,
            "success": True,
            "document_count": len(documents)
        }
        
        # End RAG span with result
        if rag_span:
            tracer.end_span_with_result(rag_span, result, True)
        
        return result
    except Exception as e:
        logger.error(f"Phase 2 - RAG search failed for query '{query}': {e}")
        
        result = {
            "query": query,
            "collections": collections,
            "error": str(e),
            "success": False
        }
        
        # End RAG span with error
        if rag_span:
            tracer.end_span_with_result(rag_span, None, False, str(e))
        
        return result

def build_lightweight_decision_context() -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Build lightweight context for decision-making phase - only names and descriptions"""
    
    # Phase 1: Lightweight MCP tools info for decision making
    tools_summary = []
    try:
        mcp_tools = get_enabled_mcp_tools()
        if mcp_tools:
            for tool_name, tool_info in mcp_tools.items():
                # Get basic description only
                description = "Available MCP tool"
                if isinstance(tool_info, dict):
                    description = tool_info.get('description', description)
                    # Try to get description from manifest
                    manifest = tool_info.get('manifest', {})
                    if manifest and 'tools' in manifest:
                        for tool_def in manifest['tools']:
                            if tool_def.get('name') == tool_name:
                                description = tool_def.get('description', description)
                                break
                
                tools_summary.append({
                    "name": tool_name,
                    "description": description
                })
    except Exception as e:
        logger.warning(f"Failed to load MCP tools summary: {e}")
    
    # Phase 1: Lightweight RAG collections info for decision making  
    collections_summary = []
    try:
        collections = get_all_collections()
        if collections:
            for collection in collections:
                name = collection.get('collection_name', '')
                description = collection.get('description', 'No description')
                
                collections_summary.append({
                    "name": name,
                    "description": description
                })
    except Exception as e:
        logger.warning(f"Failed to load collections summary: {e}")
    
    return tools_summary, collections_summary

def build_lightweight_decision_prompt() -> str:
    """Build lightweight prompt for decision-making phase - clean and focused"""
    
    # Get base system prompt from settings
    llm_settings = get_llm_settings()
    base_prompt = llm_settings.get('system_prompt', 'You are Jarvis, an AI assistant.')
    
    # Get lightweight context for decision making
    tools_summary, collections_summary = build_lightweight_decision_context()
    
    # Build simple tool list
    tools_list = []
    for tool in tools_summary:
        tools_list.append(f"• {tool['name']}: {tool['description']}")
    
    # Build simple collections list
    collections_list = []
    for collection in collections_summary:
        collections_list.append(f"• {collection['name']}: {collection['description']}")
    
    # Build clean, focused instructions
    instructions = f"""

**AVAILABLE RESOURCES FOR DECISION MAKING**

**Available Tools:**
{chr(10).join(tools_list) if tools_list else "No tools available"}

**Available Knowledge Collections:**
{chr(10).join(collections_list) if collections_list else "No collections available"}

**DECISION FRAMEWORK**

**Use Tools when:**
- Need real-time/current information (news, weather, time)
- Need to perform actions (send email, create tasks)
- Need external data or services

**Use Knowledge Search when:**
- Need company-specific information (policies, procedures)
- Need detailed documentation or established knowledge
- Query involves internal/historical data

**Function Call Format:**
- For tools: call_tool_TOOLNAME(param1="value1", param2="value2")
- For knowledge: search_knowledge(query="search terms", collections=["collection_name"])

**Instructions:**
1. Analyze the user's question
2. Decide if you need tools, knowledge search, both, or neither
3. Make function calls with appropriate parameters
4. Provide a comprehensive response based on results

Keep your reasoning clear and make intelligent decisions about what resources to use.
"""
    
    return base_prompt + instructions

async def intelligent_routing(question: str) -> Dict[str, Any]:
    """Use existing enhanced classifier with confidence fallback to online search"""
    try:
        # Use existing enhanced classifier
        classifier = EnhancedQueryClassifier()
        results = await classifier.classify(question)
        
        # Get the primary classification result
        result = results[0] if results else None
        
        # Get confidence threshold from Redis cache
        llm_settings = get_llm_settings()
        min_confidence = float(llm_settings.get("query_classifier", {}).get("min_confidence_threshold", 0.1))
        
        # Handle case where no classification was returned
        if not result:
            logger.info("No classification returned, falling back to online search")
            return await execute_online_search_fallback(question)
        
        logger.info(f"Query classification: {result.query_type.value}, confidence: {result.confidence}")
        
        # If confidence below threshold, fallback to online search
        if result.confidence < min_confidence:
            logger.info(f"Low confidence {result.confidence} < {min_confidence}, falling back to online search")
            return await execute_online_search_fallback(question)
        
        # Let LLM decide with structured functions - no hardcoded routing
        return {
            "route": "llm_structured_functions", 
            "classification": result.query_type.value,
            "confidence": result.confidence,
            "suggested_tools": result.suggested_tools
        }
            
    except Exception as e:
        logger.error(f"Intelligent routing failed: {e}")
        # Fallback to online search on any error
        return await execute_online_search_fallback(question)

async def execute_online_search_fallback(question: str) -> Dict[str, Any]:
    """Execute online search when confidence is low or routing fails"""
    try:
        # Check available search tools from cache
        tools = get_enabled_mcp_tools()
        search_tools = [name for name in tools.keys() if 'search' in name.lower()]
        
        if 'google_search' in search_tools:
            result = await execute_mcp_tool('google_search', {'query': question})
            return {"route": "online_search", "tool": "google_search", "result": result}
        elif 'web_search' in search_tools:
            result = await execute_mcp_tool('web_search', {'query': question})
            return {"route": "online_search", "tool": "web_search", "result": result}
        elif 'tavily' in search_tools:
            result = await execute_mcp_tool('tavily', {'query': question})
            return {"route": "online_search", "tool": "tavily", "result": result}
        else:
            # No search tools available, return indication for basic LLM response
            return {"route": "basic_llm", "message": "No search tools available"}
            
    except Exception as e:
        logger.error(f"Online search fallback failed: {e}")
        return {"route": "basic_llm", "error": str(e)}

async def parse_function_calls(response_text: str) -> tuple[str, List[Dict], List[Dict]]:
    """Parse LLM response to extract function calls using structured function calling"""
    tool_calls = []
    rag_searches = []
    
    import re
    
    # Pattern to match function calls: function_name(param1="value1", param2="value2") or function_name()
    function_pattern = r'(call_tool_\w+|search_knowledge)\s*\(([^)]*)\)'
    
    matches = re.finditer(function_pattern, response_text, re.MULTILINE | re.DOTALL)
    
    for match in matches:
        function_name = match.group(1)
        params_str = match.group(2)
        
        try:
            # Parse parameters from function call (handle empty parameters)
            parameters = parse_function_parameters(params_str) if params_str.strip() else {}
            
            if function_name.startswith('call_tool_'):
                # Extract tool name
                tool_name = function_name.replace('call_tool_', '')
                tool_calls.append({
                    "tool": tool_name,
                    "parameters": parameters
                })
            elif function_name == 'search_knowledge':
                rag_searches.append({
                    "query": parameters.get("query", ""),
                    "collections": parameters.get("collections")
                })
        except Exception as e:
            logger.warning(f"Failed to parse function call {function_name}: {e}")
            continue
    
    # Clean response text by removing function calls
    clean_response = response_text
    for match in re.finditer(function_pattern, response_text, re.MULTILINE | re.DOTALL):
        clean_response = clean_response.replace(match.group(0), "")
    
    return clean_response.strip(), tool_calls, rag_searches

def parse_function_parameters(params_str: str) -> Dict[str, Any]:
    """Parse function parameters from string format"""
    parameters = {}
    
    # Simple parameter parsing for key="value" format
    import re
    
    # Pattern to match parameter assignments
    param_pattern = r'(\w+)\s*=\s*(["\'"]?)([^,"\']*)\2'
    
    matches = re.finditer(param_pattern, params_str)
    
    for match in matches:
        param_name = match.group(1)
        param_value = match.group(3)
        
        # Try to parse as JSON for complex types
        if param_value.startswith('[') or param_value.startswith('{'):
            try:
                parameters[param_name] = json.loads(param_value)
            except json.JSONDecodeError:
                parameters[param_name] = param_value
        # Handle boolean values
        elif param_value.lower() in ['true', 'false']:
            parameters[param_name] = param_value.lower() == 'true'
        # Handle numeric values
        elif param_value.isdigit():
            parameters[param_name] = int(param_value)
        else:
            parameters[param_name] = param_value
    
    return parameters

@router.post("/intelligent-chat")
async def intelligent_chat_endpoint(request: IntelligentChatRequest):
    """Intelligent chat endpoint with LLM-driven decision making and structured function calling"""
    
    # Initialize Langfuse tracing
    from app.core.langfuse_integration import get_tracer
    tracer = get_tracer()
    trace = None
    generation = None
    
    # Get model name from LLM settings for proper tracing
    llm_settings = get_llm_settings()
    model_name = llm_settings.get("model", "unknown")
    
    if not tracer._initialized:
        tracer.initialize()
    
    print(f"[DEBUG] Standard chat tracer enabled: {tracer.is_enabled()}")
    if tracer.is_enabled():
        trace = tracer.create_trace(
            name="chat-workflow",
            input=request.question,
            metadata={
                "endpoint": "/api/v1/intelligent-chat",
                "conversation_id": request.conversation_id,
                "thinking": request.thinking,
                "model": model_name
            }
        )
        print(f"[DEBUG] Standard chat trace created: {trace is not None}")
        
        # Create chat execution span for proper hierarchy
        chat_span = None
        if trace:
            chat_span = tracer.create_span(
                trace,
                name="chat-execution",
                metadata={
                    "operation": "intelligent_chat",
                    "thinking": request.thinking,
                    "model": model_name
                }
            )
            print(f"[DEBUG] Standard chat span created: {chat_span is not None}")
        
        # Create generation within the chat span for detailed observability
        if chat_span:
            generation = tracer.create_generation_with_usage(
                trace=trace,
                name="chat-generation",
                model=model_name,
                input_text=request.question,
                parent_span=chat_span,
                metadata={
                    "thinking": request.thinking,
                    "model": model_name,
                    "endpoint": "intelligent-chat"
                }
            )
    
    async def stream():
        collected_output = ""
        final_answer = ""
        tool_calls_made = []
        rag_searches_made = []
        
        try:
            # Step 1: Use intelligent routing to check confidence
            routing_result = await intelligent_routing(request.question)
            
            # If low confidence, use direct online search result
            if routing_result.get("route") == "online_search":
                yield json.dumps({
                    "type": "fallback_search",
                    "tool": routing_result["tool"],
                    "confidence": 0.0
                }) + "\n"
                
                # Return the search result directly
                search_result = routing_result["result"]
                if search_result.get("success"):
                    result_text = json.dumps(search_result["result"], indent=2)
                else:
                    result_text = f"Search failed: {search_result.get('error', 'Unknown error')}"
                
                yield json.dumps({
                    "answer": result_text,
                    "source": "online_search_fallback",
                    "conversation_id": request.conversation_id
                }) + "\n"
                return
            
            # Step 1.5: Check for high-confidence direct execution bypass
            confidence = routing_result.get("confidence", 0.0)
            classification = routing_result.get("classification", "")
            
            # Get direct execution threshold from settings
            classifier_settings = get_query_classifier_settings()
            direct_execution_threshold = classifier_settings.get("direct_execution_threshold", 0.9)
            
            # If high confidence and tool classification, bypass intelligent planning
            if (confidence >= direct_execution_threshold and 
                classification.upper() in ["TOOL", "TOOLS"] and 
                routing_result.get("suggested_tools")):
                
                logger.info(f"Direct execution bypass: confidence={confidence:.3f} >= {direct_execution_threshold}, classification={classification}")
                
                # Send initial event indicating direct execution
                yield json.dumps({
                    "type": "chat_start",
                    "conversation_id": request.conversation_id,
                    "model": "direct-execution",
                    "thinking": request.thinking,
                    "confidence": confidence,
                    "classification": classification,
                    "bypass_planning": True
                }) + "\n"
                
                # Execute the highest-confidence tool directly
                suggested_tools = routing_result.get("suggested_tools", [])
                tool_name = suggested_tools[0]  # Use the first/best suggestion
                
                # Get basic parameters for the tool (empty for simple tools like datetime)
                tool_parameters = {}
                
                # Create tool call structure
                direct_tool_calls = [{
                    "tool": tool_name,
                    "parameters": tool_parameters
                }]
                
                logger.info(f"Direct execution: {tool_name} with parameters: {tool_parameters}")
                
                # Skip to tool execution (Step 5)
                yield json.dumps({
                    "type": "tools_start",
                    "tool_count": 1
                }) + "\n"
                
                # Execute the tool directly
                for tool_call in direct_tool_calls:
                    tool_result = await execute_mcp_tool(
                        tool_call["tool"],
                        tool_call["parameters"],
                        trace,
                        chat_span
                    )
                    
                    yield json.dumps({
                        "type": "tool_result",
                        "tool_result": tool_result
                    }) + "\n"
                    
                    # Generate final response from tool result
                    if tool_result.get("success") and "result" in tool_result:
                        # For datetime and simple tools, format the result nicely
                        if tool_name == "get_datetime" and isinstance(tool_result["result"], str):
                            try:
                                from datetime import datetime
                                dt = datetime.fromisoformat(tool_result["result"].replace('Z', '+00:00'))
                                formatted_date = dt.strftime("%A, %B %d, %Y")
                                final_answer = f"Today's date is {formatted_date}."
                            except:
                                final_answer = f"Today's date is {tool_result['result']}."
                        else:
                            # Generic tool result formatting
                            result_data = tool_result["result"]
                            if isinstance(result_data, dict):
                                final_answer = json.dumps(result_data, indent=2)
                            else:
                                final_answer = str(result_data)
                        
                        yield json.dumps({
                            "answer": final_answer,
                            "source": "direct_tool_execution",
                            "conversation_id": request.conversation_id,
                            "tool_used": tool_name
                        }) + "\n"
                        return
                    else:
                        # Tool execution failed, continue with normal flow
                        logger.warning(f"Direct execution failed for {tool_name}: {tool_result.get('error', 'Unknown error')}")
                        break
            
            # Step 2: Build lightweight decision prompt - Phase 1 approach
            system_prompt = build_lightweight_decision_prompt()
            
            # Prepare the conversation
            full_prompt = f"{system_prompt}\n\nUser: {request.question}\n\nAssistant:"
            
            # Get LLM settings and create LLM instance
            llm_settings = get_llm_settings()
            
            # Choose mode based on thinking parameter
            mode_config = llm_settings["thinking_mode"] if request.thinking else llm_settings["non_thinking_mode"]
            
            llm_config = LLMConfig(
                model_name=llm_settings["model"],
                temperature=float(mode_config["temperature"]),
                top_p=float(mode_config["top_p"]),
                max_tokens=int(llm_settings["max_tokens"])
            )
            
            # Get model server URL
            import os
            model_server = os.environ.get("OLLAMA_BASE_URL")
            if not model_server:
                model_server = llm_settings.get('model_server', '').strip()
                if not model_server:
                    model_server = "http://ollama:11434"
            
            llm = OllamaLLM(llm_config, base_url=model_server)
            
            # Send initial event
            yield json.dumps({
                "type": "chat_start",
                "conversation_id": request.conversation_id,
                "model": llm_settings["model"],
                "thinking": request.thinking,
                "confidence": routing_result.get("confidence", 0.0),
                "classification": routing_result.get("classification", "unknown")
            }) + "\n"
            
            # Step 3: Stream initial LLM response (exact pattern from service.py)
            initial_response = ""
            
            async for response_chunk in llm.generate_stream(full_prompt):
                initial_response += response_chunk.text
                
                # Stream tokens in real-time (exact format from service.py)
                if response_chunk.text.strip():
                    yield json.dumps({
                        "token": response_chunk.text
                    }) + "\n"
            
            # Step 4: Parse for function calls after streaming completes
            clean_response, tool_calls, rag_searches = await parse_function_calls(initial_response)
            
            # Log parsing results
            logger.info(f"Parsed - Tool calls: {tool_calls}")
            logger.info(f"Parsed - RAG searches: {rag_searches}")
            
            # Step 5: Execute tool calls if requested
            tool_results = []
            if tool_calls:
                yield json.dumps({
                    "type": "tools_start",
                    "tool_count": len(tool_calls)
                }) + "\n"
                
                for tool_call in tool_calls:
                    print(f"[DEBUG] Standard chat executing tool {tool_call['tool']} with trace: {trace is not None}")
                    tool_result = await execute_mcp_tool(
                        tool_call["tool"],
                        tool_call["parameters"],
                        trace,  # Pass trace for span creation
                        chat_span  # Pass chat_span as parent for proper hierarchy
                    )
                    tool_results.append(tool_result)
                    
                    yield json.dumps({
                        "type": "tool_result",
                        "tool_result": tool_result
                    }) + "\n"
            
            # Step 6: Execute RAG searches if requested
            rag_results = []
            if rag_searches:
                yield json.dumps({
                    "type": "rag_start",
                    "search_count": len(rag_searches)
                }) + "\n"
                
                for rag_search_request in rag_searches:
                    rag_result = await execute_rag_search(
                        rag_search_request["query"],
                        rag_search_request["collections"],
                        trace,  # Pass trace for span creation
                        chat_span  # Pass chat_span as parent for proper hierarchy
                    )
                    rag_results.append(rag_result)
                    
                    yield json.dumps({
                        "type": "rag_result",
                        "rag_result": rag_result
                    }) + "\n"
            
            # Step 7: Synthesize final response if we have tool or RAG results
            if tool_results or rag_results:
                # Build context from results
                tool_context = ""
                if tool_results:
                    tool_context = "\n\nTool Results:\n"
                    for result in tool_results:
                        if result.get("success") and "result" in result:
                            tool_context += f"\n{result['tool']}: {json.dumps(result['result'], indent=2)}\n"
                        elif "error" in result:
                            tool_context += f"\n{result['tool']}: Error - {result['error']}\n"
                
                rag_context = ""
                if rag_results:
                    for result in rag_results:
                        if result.get("success") and result.get("context"):
                            rag_context += result["context"]
                
                # Use unified synthesis approach from existing system
                from app.langchain.service import unified_llm_synthesis
                
                synthesis_prompt, _, _ = unified_llm_synthesis(
                    question=request.question,
                    query_type="TOOLS" if tool_results else "RAG",
                    rag_context=rag_context,
                    tool_context=tool_context,
                    conversation_history="",
                    thinking=request.thinking
                )
                
                # Stream synthesis
                yield json.dumps({
                    "type": "synthesis_start"
                }) + "\n"
                
                # Stream final response tokens (exact pattern from service.py)
                final_response = ""
                async for response_chunk in llm.generate_stream(synthesis_prompt):
                    final_response += response_chunk.text
                    if response_chunk.text.strip():
                        yield json.dumps({
                            "token": response_chunk.text
                        }) + "\n"
            else:
                # Use initial response as final response
                final_response = clean_response
            
            # Step 8: Send completion event (EXACT pattern from service.py)
            source = "intelligent_chat"
            if tool_results:
                source += "+TOOLS"
                tool_calls_made = tool_results
            if rag_results:
                source += "+RAG"
                rag_searches_made = rag_results
            
            # Extract documents from RAG results for transparency
            all_documents = []
            if rag_results:
                for rag_result in rag_results:
                    if rag_result.get("success") and rag_result.get("documents"):
                        all_documents.extend(rag_result["documents"])
            
            final_answer = final_response
            collected_output += json.dumps({
                "answer": final_response,
                "source": source,
                "conversation_id": request.conversation_id,
                "documents": all_documents
            })
            
            yield json.dumps({
                "answer": final_response,
                "source": source,
                "conversation_id": request.conversation_id,
                "documents": all_documents
            }) + "\n"
            
            # Update Langfuse generation and trace with the final output
            if tracer.is_enabled():
                try:
                    generation_output = final_answer if final_answer else "Intelligent chat completed"
                    
                    # Estimate token usage for cost tracking
                    usage = tracer.estimate_token_usage(request.question, generation_output)
                    
                    # End the generation with results including usage
                    if generation:
                        generation.end(
                            output=generation_output,
                            usage_details=usage,
                            metadata={
                                "response_length": len(final_answer) if final_answer else len(collected_output),
                                "source": source,
                                "streaming": True,
                                "has_final_answer": bool(final_answer),
                                "conversation_id": request.conversation_id,
                                "tool_calls_count": len(tool_calls_made),
                                "rag_searches_count": len(rag_searches_made),
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
                                "source": source,
                                "tool_calls": tool_calls_made[:3] if tool_calls_made else [],  # Limit for readability
                                "rag_searches": rag_searches_made[:3] if rag_searches_made else [],  # Limit for readability
                                "response_length": len(final_answer) if final_answer else len(collected_output),
                                "streaming": True,
                                "conversation_id": request.conversation_id
                            }
                        )
                    
                    tracer.flush()
                except Exception as e:
                    print(f"[WARNING] Failed to update Langfuse trace/generation: {e}")
            
        except Exception as e:
            logger.error(f"Intelligent chat error: {e}")
            
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
                            usage_details=usage,
                            metadata={
                                "success": False,
                                "error": str(e),
                                "conversation_id": request.conversation_id,
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
                                "conversation_id": request.conversation_id
                            }
                        )
                    
                    tracer.flush()
                except:
                    pass  # Don't fail the request if tracing fails
            
            # Always send final answer even on errors (prevents content disappearing)
            yield json.dumps({
                "answer": f"Error: {str(e)}",
                "source": "ERROR",
                "conversation_id": request.conversation_id
            }) + "\n"
    
    return StreamingResponse(stream(), media_type="application/json")

@router.get("/intelligent-chat/status")
async def get_intelligent_chat_status():
    """Get current intelligent chat system status"""
    try:
        # Check available resources
        mcp_tools = get_enabled_mcp_tools()
        collections = get_all_collections()
        llm_settings = get_llm_settings()
        
        # Get classifier settings
        classifier_settings = llm_settings.get("query_classifier", {})
        
        return {
            "status": "ready",
            "available_tools": len(mcp_tools),
            "available_collections": len(collections),
            "current_model": llm_settings.get("model", "unknown"),
            "confidence_threshold": classifier_settings.get("min_confidence_threshold", 0.1),
            "tools": list(mcp_tools.keys()) if mcp_tools else [],
            "collections": [c.get("collection_name") for c in collections] if collections else [],
            "features": [
                "llm_driven_decision_making",
                "structured_function_calling", 
                "confidence_based_fallback",
                "online_search_fallback",
                "nested_parameter_support"
            ]
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }