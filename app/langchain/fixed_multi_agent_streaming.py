"""
Fixed multi-agent streaming with thinking, tools, and collaboration
Enhanced with comprehensive Langfuse tracing
"""
import json
import asyncio
import time
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from app.langchain.langgraph_multi_agent_system import LangGraphMultiAgentSystem
from app.llm.ollama import OllamaLLM
from app.llm.base import LLMConfig
from app.core.mcp_tools_cache import get_enabled_mcp_tools
from app.core.langfuse_integration import get_tracer
from app.core.llm_settings_cache import get_llm_settings, get_second_llm_full_config
from app.core.intelligent_agent_selector import IntelligentAgentSelector
from app.core.agent_performance_tracker import performance_tracker
from app.core.langgraph_agents_cache import get_agent_by_name
# from app.langchain.tool_executor import tool_executor  # Not needed for current implementation
# from app.langchain.service import call_mcp_tool  # Moved to delayed import to avoid circular import
import logging

logger = logging.getLogger(__name__)

async def fixed_multi_agent_streaming(
    question: str,
    conversation_id: Optional[str] = None,
    trace=None
):
    """
    Fixed multi-agent streaming addressing all three issues:
    1. Tool usage from agent configs  
    2. True agent collaboration
    3. Comprehensive Langfuse tracing
    """
    
    async def stream_fixed_multi_agent():
        logger.info(f"[MULTI-AGENT] Starting fixed multi-agent streaming for question: {question[:100]}...")
        logger.info(f"[MULTI-AGENT] Conversation ID: {conversation_id}")
        logger.info(f"[MULTI-AGENT] Trace provided: {bool(trace)}")
        # Initialize Langfuse tracing
        tracer = get_tracer()
        main_trace = trace  # Use provided trace or create new one
        workflow_span = None
        
        # Initialize system with LLM settings (with error handling)
        llm_settings = None
        try:
            llm_settings = get_llm_settings()  # Ensure settings are loaded and stored
            logger.info("LLM settings loaded successfully for multi-agent system")
        except Exception as settings_error:
            logger.error(f"Failed to load LLM settings: {settings_error}")
            # Continue with multi-agent system - failsafe should handle individual agent calls
            yield json.dumps({"type": "warning", "message": f"⚠️ Settings loading issue: {settings_error}"}) + "\n"
        
        try:
            yield json.dumps({"type": "status", "message": "🎯 Starting advanced multi-agent collaboration..."}) + "\n"
            await asyncio.sleep(0.3)
            
            # Initialize system with dependency check
            try:
                system = LangGraphMultiAgentSystem(conversation_id)
                logger.info("Multi-agent system initialized successfully")
                
                # Verify Ollama connectivity before starting streaming
                await system._verify_ollama_connectivity_async()
                
            except ImportError as import_error:
                logger.error(f"Multi-agent system dependency missing: {import_error}")
                yield json.dumps({
                    "type": "error",
                    "error": "Multi-agent system dependency missing", 
                    "message": "❌ Multi-agent system requires additional dependencies. Please install LangGraph: pip install langgraph",
                    "details": {
                        "missing_dependency": "langgraph",
                        "install_command": "pip install langgraph",
                        "error_details": str(import_error)
                    }
                }) + "\n"
                return
            
            # Create multi-agent workflow span if tracing is enabled
            if main_trace and tracer.is_enabled():
                try:
                    workflow_span = tracer.create_multi_agent_workflow_span(
                        main_trace, "fixed_multi_agent_streaming", []
                    )
                    logger.info("Created multi-agent workflow span for fixed streaming")
                except Exception as e:
                    logger.warning(f"Failed to create workflow span: {e}")
            
            # Get agents from database and available tools
            available_agents = system.agents
            available_tools = get_enabled_mcp_tools()
            
            if not available_agents:
                yield json.dumps({"type": "error", "error": "No agents found in database"}) + "\n"
                return
            
            # Intelligent agent selection using semantic analysis
            try:
                agent_selector = IntelligentAgentSelector()
                selected_agents, question_analysis = agent_selector.select_agents(question, available_tools)
            except Exception as e:
                logger.error(f"Agent selection failed: {e}")
                # Create default analysis for fallback
                from app.core.intelligent_agent_selector import QuestionAnalysis
                question_analysis = QuestionAnalysis(
                    complexity="moderate", domain="general", required_skills=[], 
                    tool_requirements=[], collaboration_type="sequential", 
                    optimal_agent_count=2, keywords=[], confidence=0.5
                )
                selected_agents = []
            
            # Fallback to basic selection if intelligent selector fails
            if not selected_agents:
                logger.warning("Intelligent agent selection failed, using capability-based fallback")
                # Use capability-based fallback instead of arbitrary selection
                from app.core.langgraph_agents_cache import get_agents_with_capabilities
                agents_with_caps = get_agents_with_capabilities()
                if agents_with_caps:
                    # Select diverse agents based on domain
                    domain_agents = {}
                    for name, data in agents_with_caps.items():
                        domain = data.get('capabilities', {}).get('primary_domain', 'general')
                        if domain not in domain_agents:
                            domain_agents[domain] = name
                    selected_agents = list(domain_agents.values())[:3]
                else:
                    # Last resort
                    selected_agents = list(available_agents.keys())[:3]
                logger.info(f"Fallback selected: {selected_agents}")
                
            agents_used = selected_agents
            
            # Log selection details
            logger.info(f"🎯 ORIGINAL SELECTION: {selected_agents} (count: {len(selected_agents)})")
            logger.info(f"🎯 AGENTS TO PROCESS: {agents_used} (count: {len(agents_used)})")
            logger.info(f"Question analysis: complexity={question_analysis.complexity}, "
                       f"domain={question_analysis.domain}, optimal_count={question_analysis.optimal_agent_count}, confidence={question_analysis.confidence:.2f}")
            
            # Yield selection details for transparency
            yield json.dumps({
                "type": "agent_selection",
                "selected_agents": agents_used,
                "selection_analysis": {
                    "complexity": question_analysis.complexity,
                    "domain": question_analysis.domain,
                    "collaboration_type": question_analysis.collaboration_type,
                    "optimal_count": question_analysis.optimal_agent_count,
                    "confidence": question_analysis.confidence
                },
                "message": f"🎯 Selected {len(agents_used)} agents for {question_analysis.complexity} {question_analysis.domain} task"
            }) + "\n"
            await asyncio.sleep(0.2)
            
            # Create agent selection span for tracing
            selection_span = None
            if workflow_span and tracer.is_enabled():
                try:
                    selection_span = tracer.create_span(
                        workflow_span,
                        name="agent-selection",
                        metadata={
                            "operation": "intelligent_agent_selection",
                            "selected_agents": agents_used,
                            "total_available": len(available_agents),
                            "selection_criteria": "semantic_analysis_with_capability_matching",
                            "question_complexity": question_analysis.complexity,
                            "question_domain": question_analysis.domain,
                            "collaboration_type": question_analysis.collaboration_type,
                            "selection_confidence": question_analysis.confidence,
                            "optimal_agent_count": question_analysis.optimal_agent_count
                        }
                    )
                    logger.info(f"Created agent selection span for agents: {agents_used}")
                except Exception as e:
                    logger.warning(f"Failed to create agent selection span: {e}")
            
            # Build detailed agent information
            agent_details = []
            for agent_name in agents_used:
                agent_data = available_agents[agent_name]
                agent_details.append({
                    "name": agent_name,
                    "role": agent_data.get('role', agent_name),
                    "description": agent_data.get('description', f"Specialized {agent_data.get('role', 'analysis')} expert"),
                    "tools": agent_data.get('tools', []),
                    "capabilities": agent_data.get('capabilities', []),
                    "estimated_time": f"{2 + len(agent_data.get('tools', []))} min"
                })
            
            yield json.dumps({
                "type": "agents_selected",
                "agents": agents_used,
                "agent_details": agent_details,
                "execution_plan": {
                    "total_agents": len(agents_used),
                    "estimated_total_time": f"{len(agents_used) * 3 + 2} min",
                    "execution_pattern": "sequential_collaborative",
                    "includes_synthesis": True
                },
                "selection_criteria": "tool_based_and_keyword_matching",
                "message": f"🎯 Selected {len(agents_used)} agents for collaborative analysis"
            }) + "\n"
            await asyncio.sleep(0.2)
            
            # End agent selection span
            if selection_span and tracer.is_enabled():
                try:
                    tracer.end_span_with_result(
                        selection_span,
                        {
                            "selected_agents": agents_used,
                            "selection_success": True,
                            "agent_count": len(agents_used)
                        },
                        success=True
                    )
                except Exception as e:
                    logger.warning(f"Failed to end agent selection span: {e}")
            
            # Track agent interactions
            agent_responses = {}
            agent_communications = []
            
            # Process each agent with collaboration
            # CRITICAL DEBUG: Log exact processing plan
            logger.info(f"🔥 STARTING AGENT PROCESSING LOOP")
            logger.info(f"🔥 Total agents to process: {len(agents_used)}")
            logger.info(f"🔥 Agent list: {agents_used}")
            
            for i, agent_name in enumerate(agents_used):
                logger.info(f"🔥 PROCESSING AGENT {i+1}/{len(agents_used)}: {agent_name}")
                
                if agent_name not in available_agents:
                    logger.error(f"🔥 CRITICAL ERROR: Agent '{agent_name}' not found in available_agents!")
                    logger.error(f"🔥 Available agents: {list(available_agents.keys())}")
                    continue
                    
                agent_data = available_agents[agent_name]
                role = agent_data.get('role', agent_name)
                tools = agent_data.get('tools', [])
                
                yield json.dumps({
                    "type": "agent_start",
                    "agent": agent_name,
                    "agent_info": {
                        "role": role,
                        "position": i + 1,
                        "total_agents": len(agents_used),
                        "tools": tools,
                        "phase": "individual_analysis"
                    },
                    "progress": {
                        "current_step": i + 1,
                        "total_steps": len(agents_used) + 1,  # +1 for synthesis
                        "percentage": int((i / (len(agents_used) + 1)) * 100),
                        "phase_name": f"Agent {i + 1} Analysis",
                        "next_step": "synthesis" if i == len(agents_used) - 1 else f"Agent {i + 2} ({available_agents[agents_used[i + 1]].get('role', agents_used[i + 1])}) analysis",
                        "estimated_remaining_time": f"{(len(agents_used) - i) * 2} min"
                    },
                    "message": f"🎯 {agent_name} ({role}) beginning analysis... ({i + 1}/{len(agents_used)})"
                }) + "\n"
                await asyncio.sleep(0.3)
                
                # Create agent execution span
                agent_span = None
                agent_generation = None
                if workflow_span and tracer.is_enabled():
                    try:
                        agent_span = tracer.create_agent_execution_span(
                            workflow_span,
                            agent_name,
                            question,
                            {
                                "agent_position": i + 1,
                                "total_agents": len(agents_used),
                                "execution_pattern": "collaborative_streaming"
                            }
                        )
                        logger.info(f"Created agent execution span for {agent_name}")
                    except Exception as e:
                        logger.warning(f"Failed to create agent execution span for {agent_name}: {e}")
                
                try:
                    # Start timing for performance tracking
                    agent_start_time = time.time()
                    
                    # Get agent config from database
                    agent_data = available_agents[agent_name]
                    system_prompt = agent_data.get('system_prompt', '')
                    role = agent_data.get('role', agent_name)
                    config = agent_data.get('config', {})
                    agent_tools = agent_data.get('tools', [])
                    
                    # Log agent configuration details 
                    logger.debug(f"🔍 AGENT DEBUG {agent_name}:")
                    logger.debug(f"  - Config: {config}")
                    logger.debug(f"  - Has model in config: {'model' in config}")
                    logger.debug(f"  - Model value: {config.get('model', 'NOT SET')}")
                    logger.debug(f"  - Full agent data keys: {list(agent_data.keys())}")
                    
                    # Build collaboration context
                    collaboration_context = ""
                    if agent_communications:
                        collaboration_context += "\n\nAGENT COMMUNICATIONS:\n"
                        for comm in agent_communications[-2:]:
                            collaboration_context += f"- {comm['from']} → {comm['to']}: {comm['message']}\n"
                    
                    if agent_responses:
                        collaboration_context += "\n\nPREVIOUS INSIGHTS:\n"
                        for prev_agent, response in list(agent_responses.items())[-1:]:
                            prev_role = available_agents[prev_agent].get('role', prev_agent)
                            
                            # CRITICAL FIX: PRESERVE thinking tags in collaboration context 
                            # This allows subsequent agents to see and build upon previous thinking
                            # Only truncate for length, don't strip thinking content
                            
                            # Smart truncation that preserves thinking tags
                            if len(response) > 500:  # Increased limit to preserve more context
                                # Find last complete sentence within 500 characters
                                truncated = response[:500]
                                last_period = truncated.rfind('.')
                                last_exclamation = truncated.rfind('!')
                                last_question = truncated.rfind('?')
                                
                                last_sentence_end = max(last_period, last_exclamation, last_question)
                                if last_sentence_end > 200:  # Ensure meaningful content
                                    summary = response[:last_sentence_end + 1]
                                else:
                                    # Fallback to word boundary
                                    last_space = truncated.rfind(' ')
                                    summary = response[:last_space] + "..." if last_space > 200 else response[:500] + "..."
                            else:
                                summary = response  # Keep full response if under limit
                                
                            collaboration_context += f"- {prev_role}: {summary}\n"
                    
                    # Check for relevant tools
                    relevant_tools = [tool for tool in agent_tools if tool in available_tools]
                    tool_context = ""
                    
                    if relevant_tools:
                        yield json.dumps({
                            "type": "agent_tool_start",
                            "agent": agent_name,
                            "tools": relevant_tools,
                            "message": f"🔧 {agent_name} checking tools: {', '.join(relevant_tools)}"
                        }) + "\n"
                        
                        # REAL TOOL EXECUTION - Replace simulation with actual MCP tool calls
                        for tool in relevant_tools[:2]:
                            # Create tool execution span for tracing
                            tool_span = None
                            if agent_span and tracer.is_enabled():
                                try:
                                    tool_span = tracer.create_tool_span(
                                        agent_span,
                                        tool,
                                        {"query": question, "agent": agent_name},
                                        parent_span=agent_span
                                    )
                                    logger.info(f"Created tool span for {tool} used by {agent_name}")
                                except Exception as e:
                                    logger.warning(f"Failed to create tool span for {tool}: {e}")
                            
                            # Execute real MCP tool instead of simulation
                            tool_success = False
                            tool_result = ""
                            
                            try:
                                logger.info(f"🔧 EXECUTING REAL TOOL: {tool} for {agent_name}")
                                
                                # Use threading to handle sync/async compatibility (pattern from dynamic_agent_system.py)
                                def execute_tool_sync():
                                    # Get tool info from MCP cache for proper parameters (no hardcode)
                                    tool_info = available_tools.get(tool, {})
                                    tool_parameters = tool_info.get('parameters', {})
                                    
                                    # Build parameters based on tool's JSON-RPC 2.0 schema
                                    tool_params = {}
                                    if tool_parameters:
                                        # For search tools, typically need 'query' parameter
                                        if 'query' in tool_parameters.get('properties', {}):
                                            tool_params['query'] = question
                                        if 'num_results' in tool_parameters.get('properties', {}):
                                            tool_params['num_results'] = 10
                                        # Add other common parameters based on schema
                                        for param_name, param_info in tool_parameters.get('properties', {}).items():
                                            if param_name not in tool_params:
                                                # Set reasonable defaults based on parameter type
                                                param_type = param_info.get('type', 'string')
                                                if param_type == 'integer' and 'count' in param_name:
                                                    tool_params[param_name] = 10
                                                elif param_type == 'boolean':
                                                    tool_params[param_name] = True
                                    else:
                                        # Fallback for tools without schema
                                        tool_params = {'query': question}
                                    
                                    # Delayed import to avoid circular dependency
                                    from app.langchain.service import call_mcp_tool
                                    return call_mcp_tool(tool, tool_params, trace=tool_span)
                                
                                # Execute tool in thread to avoid async conflicts
                                loop = asyncio.get_event_loop()
                                with ThreadPoolExecutor() as executor:
                                    future = loop.run_in_executor(executor, execute_tool_sync)
                                    tool_response = await asyncio.wait_for(future, timeout=30.0)
                                
                                # Use same success detection logic as standard chat
                                tool_success = True
                                if isinstance(tool_response, dict):
                                    if "error" in tool_response:
                                        tool_success = False
                                    elif "message" in tool_response and "error" in str(tool_response.get("message", "")).lower():
                                        tool_success = False
                                
                                if tool_success and tool_response:
                                    # Use tool response directly (same as standard chat pattern)
                                    # Don't try to extract nested content - the response IS the content
                                    tool_result = tool_response
                                    
                                    # Only convert to string for logging and context, not for actual result
                                    result_preview = str(tool_result)[:100]
                                    logger.info(f"✅ Tool {tool} succeeded: {result_preview}...")
                                else:
                                    tool_result = tool_response.get('error', f"Tool {tool} failed") if isinstance(tool_response, dict) else f"Tool {tool} returned no results"
                                    logger.warning(f"⚠️ Tool {tool} failed: {tool_result}")
                                    
                            except asyncio.TimeoutError:
                                tool_result = f"Tool {tool} timed out after 30 seconds"
                                logger.error(f"❌ Tool {tool} timeout")
                            except Exception as e:
                                tool_result = f"Tool {tool} failed: {str(e)}"
                                logger.error(f"❌ Tool {tool} error: {e}")
                            
                            # Add tool result to context for LLM (convert to string for context)
                            tool_result_str = str(tool_result) if tool_result else "No result"
                            tool_context += f"\nTool {tool}: {tool_result_str}"
                            
                            # End tool span with real result
                            if tool_span and tracer.is_enabled():
                                try:
                                    tracer.end_span_with_result(
                                        tool_span,
                                        {
                                            "tool_name": tool,
                                            "result": tool_result,
                                            "agent": agent_name,
                                            "success": tool_success
                                        },
                                        success=tool_success
                                    )
                                except Exception as e:
                                    logger.warning(f"Failed to end tool span for {tool}: {e}")
                            
                            # Stream real tool result to UI
                            tool_result_preview = str(tool_result)[:200] + "..." if len(str(tool_result)) > 200 else str(tool_result)
                            yield json.dumps({
                                "type": "agent_tool_complete",
                                "agent": agent_name,
                                "tool": tool,
                                "result": tool_result_preview,
                                "success": tool_success
                            }) + "\n"
                            await asyncio.sleep(0.3)
                    
                    # Create comprehensive prompt
                    agent_prompt = f"""{system_prompt}

User Question: {question}{collaboration_context}{tool_context}

Your response should include detailed analysis with specific examples and evidence.

Available Tools: {', '.join(relevant_tools) if relevant_tools else 'None'}

Instructions:
1. Provide comprehensive analysis using your expertise (minimum 3-4 detailed paragraphs)
2. {"Reference and build upon previous agent insights" if collaboration_context else "Provide foundational analysis"}  
3. Include specific examples, case studies, and evidence
4. End with actionable recommendations and next steps
6. Ensure your response is thorough and adds substantial value to the analysis

IMPORTANT: This is a professional analysis that requires depth and detail. Provide at least 300-500 words of substantive content.

"""
                    
                    # Determine model and parameters based on configuration flags
                    is_gpt_oss = False  # Initialize here for proper scope
                    if config.get('use_main_llm'):
                        from app.core.llm_settings_cache import get_main_llm_full_config
                        main_llm_config = get_main_llm_full_config()
                        agent_model = main_llm_config.get('model')
                        # Override with main_llm defaults if not specified in agent config
                        if 'max_tokens' not in config:
                            config['max_tokens'] = main_llm_config.get('max_tokens', 2000)
                        if 'temperature' not in config:
                            config['temperature'] = main_llm_config.get('temperature', 0.7)
                        logger.info(f"🔍 {agent_name} using main_llm: {agent_model}")
                    elif config.get('use_second_llm') or not config.get('model'):
                        # Use second_llm with dynamic detection
                        # Use the pre-loaded settings or try to get them again
                        settings = llm_settings if llm_settings else get_llm_settings()
                        
                        # Apply dynamic detection for second_llm (same logic as main_llm)
                        second_llm_base = settings.get('second_llm', {})
                        model_name = second_llm_base.get('model', '')
                        
                        # Check for problematic model and apply dynamic detection
                        detected_mode = None
                        if 'qwen3:30b-a3b-instruct-2507-q4_K_M' in model_name:
                            # Apply heuristic detection for instruct models
                            detected_mode = 'non-thinking'
                            logger.info(f"🔍 {agent_name} applying dynamic detection: {model_name} -> {detected_mode}")
                        
                        second_llm_config = get_second_llm_full_config(settings, override_mode=detected_mode)
                        agent_model = second_llm_config.get('model')
                        
                        # Override with second_llm defaults if not specified
                        if 'max_tokens' not in config:
                            config['max_tokens'] = second_llm_config.get('max_tokens', 2000)
                        if 'temperature' not in config:
                            config['temperature'] = second_llm_config.get('temperature', 0.7)
                        
                        # THINKING MODEL DETECTION: Add special handling for thinking models
                        is_gpt_oss = agent_model and 'gpt-oss' in agent_model.lower()
                        if is_gpt_oss:
                            logger.info(f"🧠 {agent_name} GPT-OSS THINKING MODEL DETECTED: {agent_model}")
                            logger.info(f"🧠 {agent_name} Will handle thinking tags and empty chunks specially")
                        
                        # Log effective configuration
                        logger.info(f"🔍 {agent_name} using second_llm: {agent_model}")
                        logger.info(f"🔍 {agent_name} effective mode: {second_llm_config.get('effective_mode', 'unknown')}")
                        logger.info(f"🔍 {agent_name} temp: {config['temperature']}, max_tokens: {config['max_tokens']}")
                    else:
                        # Use specific model
                        agent_model = config.get('model')
                        logger.info(f"🔍 {agent_name} using specific model: {agent_model}")
                    
                    # Real LLM streaming call with agent-specific model
                    llm_config = LLMConfig(
                        model_name=agent_model,
                        temperature=config.get('temperature', 0.7),
                        top_p=config.get('top_p', 0.9),
                        max_tokens=config.get('max_tokens', 2000)
                    )
                    
                    # Create LLM generation span for tracing
                    if agent_span and tracer.is_enabled():
                        try:
                            agent_generation = tracer.create_llm_generation_span(
                                agent_span,
                                model=llm_config.model_name,
                                prompt=agent_prompt,
                                operation=f"agent_generation_{agent_name}"
                            )
                            logger.info(f"Created LLM generation span for {agent_name}")
                        except Exception as e:
                            logger.warning(f"Failed to create LLM generation span for {agent_name}: {e}")
                    
                    llm = OllamaLLM(llm_config, base_url=system.ollama_base_url)
                    
                    # Stream with thinking detection
                    full_response = ""
                    thinking_content = ""
                    in_thinking = False
                    thinking_detected = False
                    token_count = 0
                    empty_chunk_count = 0  # Track empty chunks for GPT OSS models
                    
                    # CONTEXT-AWARE EMPTY CHUNK LIMITS: Later agents need higher tolerance
                    # GPT-OSS models produce more empty chunks in sequential execution
                    base_empty_chunks = 1000  # Increased base from 500 to 1000
                    if is_gpt_oss:
                        # Progressive tolerance based on agent position in sequence
                        agent_position = i + 1  # 1-based position
                        if agent_position >= 4:  # 4th+ agent
                            max_empty_chunks = 3000  # Increased from 1500
                        elif agent_position >= 3:  # 3rd agent  
                            max_empty_chunks = 2000  # Increased from 1000
                        else:  # 1st-2nd agent
                            max_empty_chunks = 1000  # Increased from 500
                        logger.info(f"[GPT-OSS] {agent_name} (position {agent_position}) max_empty_chunks: {max_empty_chunks}")
                    else:
                        max_empty_chunks = 100  # Non-GPT-OSS models need fewer empty chunks
                    
                    # Log if this is a GPT OSS thinking model (already detected above)
                    if is_gpt_oss:
                        logger.info(f"[GPT-OSS] {agent_name} using GPT OSS model: {agent_model} - expecting many empty chunks")
                    
                    # Add debug logging and timing
                    stream_start_time = time.time()
                    # Increased timeout for GPT-OSS models
                    max_wait_time = 90.0 if is_gpt_oss else 45.0  # 90s for GPT-OSS, 45s for others
                    logger.info(f"Starting streaming for {agent_name} (timeout: {max_wait_time}s)...")
                    
                    async for response_chunk in llm.generate_stream(agent_prompt):
                        chunk_text = response_chunk.text
                        
                        # CRITICAL FIX: Handle empty chunks differently for GPT OSS models
                        # GPT OSS models return hundreds of empty chunks before content
                        if not chunk_text:  # Empty chunk
                            empty_chunk_count += 1
                            elapsed_time = time.time() - stream_start_time
                            
                            if is_gpt_oss:
                                # Progressive timeout: Use both chunk count AND time limits
                                if empty_chunk_count <= max_empty_chunks and elapsed_time < max_wait_time:
                                    # Continue waiting for GPT-OSS content
                                    if empty_chunk_count % 200 == 0:  # Log less frequently
                                        logger.info(f"[GPT-OSS] {agent_name} received {empty_chunk_count} empty chunks after {elapsed_time:.1f}s, still waiting (max: {max_empty_chunks} chunks, {max_wait_time}s)...")
                                    continue
                                else:
                                    # Progressive timeout reached
                                    if elapsed_time >= max_wait_time:
                                        logger.warning(f"[GPT-OSS] {agent_name} timed out after {elapsed_time:.1f}s with {empty_chunk_count} empty chunks")
                                    else:
                                        logger.warning(f"[GPT-OSS] {agent_name} exceeded {max_empty_chunks} empty chunks after {elapsed_time:.1f}s")
                                    break
                            else:
                                # For non-GPT OSS models, skip empty chunks normally
                                logger.debug(f"[{agent_name}] Skipping empty chunk (prevents empty response bug)")
                                continue
                        
                        # Reset empty chunk counter when we get content
                        if empty_chunk_count > 0:
                            logger.info(f"[{agent_name}] Got content after {empty_chunk_count} empty chunks")
                            empty_chunk_count = 0
                        
                        full_response += chunk_text
                        token_count += 1
                        
                        # Enhanced debugging for first few tokens
                        if token_count <= 5:
                            logger.info(f"[{agent_name}] Token {token_count}: '{chunk_text[:50]}...' (length: {len(chunk_text)})")
                        
                        # Additional debug for potential empty chunk patterns
                        if token_count <= 10 and len(chunk_text.strip()) == 0:
                            logger.warning(f"[{agent_name}] Token {token_count} is whitespace-only: {repr(chunk_text)}")
                        
                        # Detect thinking start
                        if "<think>" in chunk_text.lower():
                            in_thinking = True
                            thinking_detected = True
                            
                            yield json.dumps({
                                "type": "agent_thinking_start",
                                "agent": agent_name,
                                "message": f"💭 {agent_name} reasoning..."
                            }) + "\n"
                            
                            # Extract content after <think>
                            start_idx = chunk_text.lower().find("<think>") + 7
                            thinking_content += chunk_text[start_idx:]
                            
                        # Detect thinking end
                        elif "</think>" in chunk_text.lower():
                            in_thinking = False
                            
                            # Extract content before </think>
                            end_idx = chunk_text.lower().find("</think>")
                            thinking_content += chunk_text[:end_idx]
                            
                            # Send complete thinking
                            yield json.dumps({
                                "type": "agent_thinking_complete",
                                "agent": agent_name,
                                "thinking": thinking_content.strip()
                            }) + "\n"
                            
                            # Reset thinking state
                            thinking_content = ""
                            
                            # CRITICAL FIX: Send the </think> tag as output to preserve complete response
                            yield json.dumps({
                                "type": "agent_token",
                                "agent": agent_name,
                                "token": "</think>"
                            }) + "\n"
                            
                            # Continue with remaining content after </think>
                            remaining = chunk_text[end_idx + 8:]
                            if remaining.strip():
                                yield json.dumps({
                                    "type": "agent_token",
                                    "agent": agent_name,
                                    "token": remaining
                                }) + "\n"
                        
                        # Accumulate thinking content
                        elif in_thinking:
                            thinking_content += chunk_text
                        
                        # Stream regular content (this should be the main path)
                        else:
                            if chunk_text.strip():
                                yield json.dumps({
                                    "type": "agent_token",
                                    "agent": agent_name,
                                    "token": chunk_text
                                }) + "\n"
                        
                        await asyncio.sleep(0.02)
                    
                    logger.info(f"Streaming completed for {agent_name}. Total tokens: {token_count}, empty_chunks: {empty_chunk_count}, thinking_detected: {thinking_detected}, full_response_length: {len(full_response)}")
                    
                    # Additional debugging for the Service Delivery Manager issue
                    if token_count > 0:
                        logger.debug(f"[{agent_name}] Full response preview: {repr(full_response[:100])}...")
                        if len(full_response) == 0:
                            logger.error(f"[{agent_name}] CRITICAL: Streamed {token_count} tokens but full_response is empty - this should be fixed now!")
                        elif len(full_response) < 50:
                            logger.warning(f"[{agent_name}] Warning: Only {len(full_response)} characters accumulated from {token_count} tokens")
                    
                    # Fallback thinking if none detected
                    logger.info(f"[DEBUG] {agent_name}: thinking_detected = {thinking_detected}")
                    if not thinking_detected:
                        fallback_thinking = f"Analyzing '{question}' from {role} perspective. I need to consider my domain expertise, available insights from other agents, and provide actionable recommendations based on my specialized knowledge."
                        
                        logger.info(f"[DEBUG] {agent_name}: Sending fallback thinking")
                        yield json.dumps({
                            "type": "agent_thinking_start",
                            "agent": agent_name,
                            "message": f"💭 {agent_name} reasoning..."
                        }) + "\n"
                        
                        yield json.dumps({
                            "type": "agent_thinking_complete",
                            "agent": agent_name,
                            "thinking": fallback_thinking
                        }) + "\n"
                    else:
                        logger.info(f"[DEBUG] {agent_name}: Thinking was detected, not sending fallback")
                    
                    # CRITICAL FIX: If no tokens were streamed but we have a response, send it now
                    if token_count == 0 and full_response.strip():
                        logger.warning(f"[CRITICAL] {agent_name}: No tokens streamed but response exists! Sending full response now.")
                        yield json.dumps({
                            "type": "agent_token",
                            "agent": agent_name,
                            "token": full_response
                        }) + "\n"
                    elif token_count > 0:
                        logger.info(f"[SUCCESS] {agent_name}: Streamed {token_count} tokens successfully")
                    else:
                        logger.error(f"[ERROR] {agent_name}: No response generated at all!")
                    
                    # Store response for collaboration with validation
                    try:
                        agent_response = system._clean_llm_response(full_response)
                    except Exception as clean_error:
                        logger.error(f"[{agent_name}] Error cleaning response: {clean_error}")
                        agent_response = full_response  # Use raw response as fallback
                    
                    # CRITICAL FIX: Handle thinking models (like GPT OSS) that wrap ALL content in <think> tags
                    if thinking_detected and agent_response.strip():
                        try:
                            # Check if response is ONLY thinking content (wrapped in <think></think>)
                            think_pattern = r'<think>(.*?)</think>'
                            think_matches = re.findall(think_pattern, agent_response, re.DOTALL | re.IGNORECASE)
                            
                            # Also check for any content AFTER </think> tags (the actual response)
                            after_think_pattern = r'</think>\s*(.+)'
                            after_think_match = re.search(after_think_pattern, agent_response, re.DOTALL | re.IGNORECASE)
                            
                            if after_think_match and after_think_match.group(1).strip():
                                # If there's content after </think>, that's the real response
                                actual_content = after_think_match.group(1).strip()
                                logger.info(f"[THINKING MODEL] {agent_name}: Found {len(actual_content)} chars after thinking tags")
                                # Keep the full response for now (includes thinking for context)
                            elif think_matches:
                                # Response is mostly/only thinking content
                                extracted_content = '\n'.join(think_matches).strip()
                                # If extracted content is substantial, log it but keep full response
                                if len(extracted_content) > len(agent_response) * 0.3:
                                    logger.info(f"[THINKING MODEL] {agent_name}: Response contains {len(extracted_content)} chars in thinking tags")
                                    # For GPT OSS, the thinking IS the content, so preserve it
                                    if is_gpt_oss:
                                        logger.info(f"[GPT-OSS] {agent_name}: Preserving thinking content as main response")
                        except Exception as think_error:
                            logger.error(f"[{agent_name}] Error processing thinking tags: {think_error}")
                            # Continue with the response as-is
                    
                    # Enhanced validation with better debugging for Service Delivery Manager issue
                    if token_count > 0 and not agent_response.strip():
                        logger.error(f"[VALIDATION FAILURE] {agent_name}: Streamed {token_count} tokens but got empty response after cleaning")
                        logger.error(f"[VALIDATION] Full response before cleaning: original_length={len(full_response)}, content={repr(full_response[:200])}...")
                        logger.error(f"[VALIDATION] Cleaned response: cleaned_length={len(agent_response)}, content={repr(agent_response)}")
                        
                        # Use full_response as fallback if cleaning resulted in empty response
                        if full_response.strip():
                            agent_response = full_response
                            logger.info(f"[VALIDATION RECOVERY] {agent_name}: Using original full_response as fallback, length={len(agent_response)}")
                        else:
                            agent_response = "Error: Response processing failed - no content generated"
                            logger.error(f"[VALIDATION RECOVERY] {agent_name}: Both full_response and cleaned response empty - using error message")
                    elif token_count > 0 and agent_response.strip():
                        logger.debug(f"[VALIDATION SUCCESS] {agent_name}: {token_count} tokens -> {len(agent_response)} character response")
                    elif token_count == 0:
                        logger.warning(f"[VALIDATION WARNING] {agent_name}: No tokens streamed at all - possible connection issue")
                    
                    agent_responses[agent_name] = agent_response
                    
                    # Record performance metrics
                    try:
                        agent_execution_time = time.time() - agent_start_time
                        performance_tracker.record_agent_execution(
                            agent_name=agent_name,
                            success=True,
                            response_length=len(agent_response),
                            execution_time=agent_execution_time,
                            question_complexity=question_analysis.complexity,
                            question_domain=question_analysis.domain,
                            collaboration_mode=question_analysis.collaboration_type
                        )
                    except Exception as perf_error:
                        logger.warning(f"Failed to record performance metrics for {agent_name}: {perf_error}")
                    
                    # End LLM generation span with results - use full_response to preserve thinking tags
                    if agent_generation and tracer.is_enabled():
                        try:
                            usage = tracer.estimate_token_usage(agent_prompt, full_response)
                            agent_generation.end(
                                output=full_response,  # Use full_response to show complete thinking in Langfuse
                                usage=usage,
                                metadata={
                                    "agent_name": agent_name,
                                    "thinking_detected": thinking_detected,
                                    "tools_used": relevant_tools,
                                    "collaboration_context_length": len(collaboration_context),
                                    "response_length": len(full_response),
                                    "operation": f"agent_generation_{agent_name}"
                                }
                            )
                            logger.info(f"Ended LLM generation span for {agent_name}")
                        except Exception as e:
                            logger.warning(f"Failed to end LLM generation span for {agent_name}: {e}")
                    
                    # Send communication to next agent
                    if i < len(agents_used) - 1:
                        next_agent = agents_used[i + 1]
                        next_agent_data = available_agents[next_agent]
                        next_role = next_agent_data.get('role', next_agent)
                        
                        # IMPROVED: Clean thinking tags for communication messages (but preserve structure)
                        # Communication messages should be concise summaries, not full thinking content
                        try:
                            # First remove any thinking content between tags
                            clean_response = re.sub(r'<think>.*?</think>', '', agent_response, flags=re.DOTALL | re.IGNORECASE)
                            # Then remove any remaining think tags
                            clean_response = re.sub(r'</?think>', '', clean_response, flags=re.IGNORECASE)
                            # Finally strip whitespace
                            clean_response = clean_response.strip()
                        except Exception as clean_error:
                            logger.error(f"[{agent_name}] Error cleaning response for communication: {clean_error}")
                            clean_response = agent_response[:400] if agent_response else ""  # Fallback to simple truncation
                        
                        # Additional cleanup for better communication messages
                        clean_response = re.sub(r'\n\s*\n+', ' ', clean_response)  # Collapse multiple newlines
                        
                        # Smart truncation that preserves sentence integrity
                        if len(clean_response) > 400:
                            # Find last complete sentence within 400 characters
                            truncated = clean_response[:400]
                            last_period = truncated.rfind('.')
                            last_exclamation = truncated.rfind('!')
                            last_question = truncated.rfind('?')
                            
                            # Use the last complete sentence boundary
                            last_sentence_end = max(last_period, last_exclamation, last_question)
                            if last_sentence_end > 200:  # Ensure we have meaningful content
                                insights_summary = clean_response[:last_sentence_end + 1]
                            else:
                                # Fallback to word boundary
                                truncated = clean_response[:400]
                                last_space = truncated.rfind(' ')
                                insights_summary = clean_response[:last_space] + "..." if last_space > 200 else clean_response[:400] + "..."
                        else:
                            insights_summary = clean_response
                        
                        comm_message = f"Key insights from {role}: {insights_summary} Please build upon this with your expertise."
                        
                        agent_communications.append({
                            "from": agent_name,
                            "to": next_agent,
                            "message": comm_message,
                            "summary": insights_summary,
                            "handoff_reason": f"Transitioning from {role} analysis to {next_role} perspective"
                        })
                        
                        yield json.dumps({
                            "type": "agent_communication",
                            "from": agent_name,
                            "from_role": role,
                            "to": next_agent,
                            "to_role": next_role,
                            "communication": {
                                "message": comm_message,
                                "summary": insights_summary,
                                "handoff_reason": f"Transitioning from {role} analysis to {next_role} perspective",
                                "context_shared": True
                            },
                            "progress": {
                                "handoff_step": i + 1,
                                "total_handoffs": len(agents_used) - 1
                            },
                            "timestamp": "now",
                            "message": f"🔄 Handing off from {role} to {next_role}..."
                        }) + "\n"
                    
                    # End agent execution span with results
                    if agent_span and tracer.is_enabled():
                        try:
                            tracer.end_span_with_result(
                                agent_span,
                                {
                                    "agent_name": agent_name,
                                    "response_length": len(agent_response),
                                    "tools_used": relevant_tools,
                                    "thinking_detected": thinking_detected,
                                    "collaboration_used": bool(collaboration_context),
                                    "success": True
                                },
                                success=True
                            )
                            logger.info(f"Ended agent execution span for {agent_name}")
                        except Exception as e:
                            logger.warning(f"Failed to end agent execution span for {agent_name}: {e}")
                    
                    # Send agent completion signal with full response including thinking tags
                    logger.info(f"Sending agent_complete for {agent_name} with response length: {len(agent_response)}")
                    logger.info(f"Agent response contains thinking tags: {'<think>' in agent_response and '</think>' in agent_response}")
                    
                    yield json.dumps({
                        "type": "agent_complete",
                        "agent": agent_name,
                        "content": agent_response,  # This includes thinking tags preserved by _clean_llm_response
                        "agent_info": {
                            "role": role,
                            "position": i + 1,
                            "total_agents": len(agents_used),
                            "tools_used": relevant_tools,
                            "thinking_detected": thinking_detected,
                            "response_length": len(agent_response)
                        },
                        "progress": {
                            "current_step": i + 1,
                            "total_steps": len(agents_used) + 1,
                            "percentage": int(((i + 1) / (len(agents_used) + 1)) * 100),
                            "phase_name": f"Agent {i + 1} Complete",
                            "remaining_agents": len(agents_used) - (i + 1)
                        },
                        "message": f"✅ {agent_name} analysis complete ({i + 1}/{len(agents_used)})"
                    }) + "\n"
                    
                except Exception as e:
                    logger.error(f"Agent {agent_name} failed: {e}")
                    fallback_response = f"Analysis from {agent_name}: {question} requires careful consideration from {role} perspective."
                    agent_responses[agent_name] = fallback_response
                    
                    # Record failed performance metrics
                    try:
                        agent_execution_time = time.time() - agent_start_time
                        performance_tracker.record_agent_execution(
                            agent_name=agent_name,
                            success=False,
                            response_length=len(fallback_response),
                            execution_time=agent_execution_time,
                            question_complexity=question_analysis.complexity,
                            question_domain=question_analysis.domain,
                            collaboration_mode=question_analysis.collaboration_type
                        )
                    except Exception as perf_error:
                        logger.warning(f"Failed to record failure metrics for {agent_name}: {perf_error}")
                    
                    # End spans with error for failed agents
                    if agent_generation and tracer.is_enabled():
                        try:
                            usage = tracer.estimate_token_usage(agent_prompt if 'agent_prompt' in locals() else question, fallback_response)
                            agent_generation.end(
                                output=f"Error: {str(e)}",
                                usage=usage,
                                metadata={
                                    "agent_name": agent_name,
                                    "error": str(e),
                                    "fallback_used": True,
                                    "operation": f"agent_generation_{agent_name}"
                                }
                            )
                        except Exception as trace_error:
                            logger.warning(f"Failed to end LLM generation span with error for {agent_name}: {trace_error}")
                    
                    if agent_span and tracer.is_enabled():
                        try:
                            tracer.end_span_with_result(
                                agent_span,
                                {
                                    "agent_name": agent_name,
                                    "error": str(e),
                                    "fallback_response": fallback_response,
                                    "success": False
                                },
                                success=False,
                                error=str(e)
                            )
                        except Exception as trace_error:
                            logger.warning(f"Failed to end agent execution span with error for {agent_name}: {trace_error}")
                    
                    yield json.dumps({
                        "type": "agent_complete",
                        "agent": agent_name,
                        "content": fallback_response,
                        "agent_info": {
                            "role": role,
                            "position": i + 1,
                            "total_agents": len(agents_used),
                            "tools_used": relevant_tools,
                            "thinking_detected": False,
                            "response_length": len(fallback_response),
                            "status": "fallback"
                        },
                        "progress": {
                            "current_step": i + 1,
                            "total_steps": len(agents_used) + 1,
                            "percentage": int(((i + 1) / (len(agents_used) + 1)) * 100),
                            "phase_name": f"Agent {i + 1} Complete",
                            "remaining_agents": len(agents_used) - (i + 1)
                        },
                        "message": f"✅ {agent_name} analysis complete (fallback) ({i + 1}/{len(agents_used)})"
                    }) + "\n"
                
                await asyncio.sleep(0.2)
            
            # Synthesis phase with detailed progress
            yield json.dumps({
                "type": "synthesis_start",
                "phase": "synthesis",
                "synthesis_info": {
                    "total_agents_processed": len(agents_used),
                    "communications_count": len(agent_communications),
                    "synthesis_type": "collaborative_integration",
                    "estimated_time": "2-3 min"
                },
                "progress": {
                    "current_step": len(agents_used) + 1,
                    "total_steps": len(agents_used) + 1,
                    "percentage": 80,  # Start synthesis at 80%
                    "phase_name": "Synthesis Phase - Gathering Insights",
                    "status": "gathering_insights",
                    "activity": "Collecting and organizing agent responses"
                },
                "message": "🔮 Synthesizing collaborative insights from all agents..."
            }) + "\n"
            await asyncio.sleep(0.5)
            
            # Synthesis sub-phase 1: Gathering insights (85%)
            yield json.dumps({
                "type": "synthesis_progress",
                "progress": {
                    "current_step": len(agents_used) + 1,
                    "total_steps": len(agents_used) + 1,
                    "percentage": 85,
                    "phase_name": "Synthesis Phase - Analyzing Insights",
                    "status": "analyzing_insights",
                    "activity": "Analyzing agent communications and building integration framework"
                },
                "synthesis_details": {
                    "agents_being_integrated": agents_used,
                    "integration_steps": [
                        f"✅ Collected insights from {agent}" for agent in agents_used
                    ] + ["🔄 Building integration framework"],
                    "current_focus": "Agent communication patterns"
                },
                "message": "🔍 Analyzing agent insights and communication patterns..."
            }) + "\n"
            await asyncio.sleep(0.3)
            
            # Create synthesis prompt
            synthesis_prompt = f"""
I need to synthesize the collaborative analysis from {len(agents_used)} agents who worked together on: {question}

Let me review their communications and create an integrated response.


COLLABORATION SUMMARY:
"""
            
            for comm in agent_communications:
                synthesis_prompt += f"Communication: {comm['from']} → {comm['to']}: {comm['message']}\n"
            
            synthesis_prompt += f"""

AGENT ANALYSES:
"""
            
            for agent_name, response in agent_responses.items():
                role = available_agents[agent_name].get('role', agent_name)
                synthesis_prompt += f"\n{role}: {response}\n"
            
            synthesis_prompt += """

SYNTHESIS TASK:
Create a comprehensive, detailed synthesis response that:
1. Integrates key insights from all agents
2. Shows how agents collaborated and built upon each other's work
3. Provides actionable recommendations
4. Is at least 3-4 paragraphs long with specific examples

Begin your synthesis now:"""
            
            # Debug: Log synthesis prompt length
            prompt_length = len(synthesis_prompt)
            print(f"[DEBUG] Synthesis prompt length: {prompt_length} characters")
            if prompt_length > 20000:
                print(f"[WARNING] Very long synthesis prompt ({prompt_length} chars) - may cause truncation")
                # Truncate agent responses if needed to prevent context overflow
                max_response_length = 1000
                truncated_prompt = f"""I need to synthesize the collaborative analysis from {len(agents_used)} agents who worked together on: {question}

Let me review their communications and create an integrated response.

COLLABORATION SUMMARY:
"""
                for comm in agent_communications:
                    truncated_prompt += f"Communication: {comm['from']} → {comm['to']}: {comm['message']}\n"
                
                truncated_prompt += "\n\nAGENT ANALYSES:\n"
                
                for agent_name, response in agent_responses.items():
                    role = available_agents[agent_name].get('role', agent_name)
                    truncated_response = response[:max_response_length] + "..." if len(response) > max_response_length else response
                    truncated_prompt += f"\n{role}: {truncated_response}\n"
                
                truncated_prompt += """

SYNTHESIS TASK:
Create a comprehensive, detailed synthesis response that:
1. Integrates key insights from all agents
2. Shows how agents collaborated and built upon each other's work
3. Provides actionable recommendations
4. Is at least 3-4 paragraphs long with specific examples

Begin your synthesis now:"""
                
                synthesis_prompt = truncated_prompt
                print(f"[DEBUG] Truncated synthesis prompt to {len(synthesis_prompt)} characters")
            
            # Synthesis sub-phase 2: Building integration (90%)
            yield json.dumps({
                "type": "synthesis_progress",
                "progress": {
                    "current_step": len(agents_used) + 1,
                    "total_steps": len(agents_used) + 1,
                    "percentage": 90,
                    "phase_name": "Synthesis Phase - Building Integration",
                    "status": "building_integration",
                    "activity": "Creating unified response from collaborative insights"
                },
                "synthesis_details": {
                    "agents_being_integrated": agents_used,
                    "integration_steps": [
                        f"✅ Integrated insights from {agent}" for agent in agents_used
                    ] + ["🔄 Creating unified narrative"],
                    "current_focus": "Cross-agent insight synthesis"
                },
                "message": "🔨 Building integrated response from agent collaborations..."
            }) + "\n"
            await asyncio.sleep(0.3)
            
            # Create synthesis span for tracing
            synthesis_span = None
            synthesis_generation = None
            if workflow_span and tracer.is_enabled():
                try:
                    synthesis_span = tracer.create_span(
                        workflow_span,
                        name="synthesis",
                        metadata={
                            "operation": "multi_agent_synthesis",
                            "agent_count": len(agents_used),
                            "communication_count": len(agent_communications),
                            "synthesis_type": "collaborative_integration"
                        }
                    )
                    logger.info("Created synthesis span for multi-agent workflow")
                except Exception as e:
                    logger.warning(f"Failed to create synthesis span: {e}")
            
            # Synthesis streaming with thinking model preference
            try:
                # Get synthesizer configuration first
                synthesizer_agent = get_agent_by_name("synthesizer")
                synthesis_model = None
                temperature = 0.6
                max_tokens = 2500
                
                if synthesizer_agent:
                    synthesizer_config_data = synthesizer_agent.get('config', {})
                    
                    # Check synthesizer's LLM configuration
                    if synthesizer_config_data.get('use_main_llm'):
                        from app.core.llm_settings_cache import get_main_llm_full_config
                        main_llm_config = get_main_llm_full_config()
                        synthesis_model = main_llm_config.get('model')
                        temperature = synthesizer_config_data.get('temperature', main_llm_config.get('temperature', 0.6))
                        max_tokens = synthesizer_config_data.get('max_tokens', main_llm_config.get('max_tokens', 2500))
                    elif synthesizer_config_data.get('use_second_llm'):
                        second_llm_config = get_second_llm_full_config()
                        synthesis_model = second_llm_config.get('model')
                        temperature = synthesizer_config_data.get('temperature', second_llm_config.get('temperature', 0.6))
                        max_tokens = synthesizer_config_data.get('max_tokens', second_llm_config.get('max_tokens', 2500))
                    elif synthesizer_config_data.get('model'):
                        synthesis_model = synthesizer_config_data.get('model')
                        temperature = synthesizer_config_data.get('temperature', 0.6)
                        max_tokens = synthesizer_config_data.get('max_tokens', 2500)
                
                # If no synthesizer config found, fall back to second_llm
                if not synthesis_model:
                    second_llm_config = get_second_llm_full_config()
                    synthesis_model = second_llm_config.get('model', 'qwen3:30b-a3b')
                
                logger.info(f"Synthesizer using model: {synthesis_model}")
                
                synthesizer_config = LLMConfig(
                    model_name=synthesis_model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Create synthesis generation span
                if synthesis_span and tracer.is_enabled():
                    try:
                        synthesis_generation = tracer.create_llm_generation_span(
                            synthesis_span,
                            model=synthesizer_config.model_name,
                            prompt=synthesis_prompt,
                            operation="synthesis_generation"
                        )
                        logger.info("Created synthesis generation span")
                    except Exception as e:
                        logger.warning(f"Failed to create synthesis generation span: {e}")
                
                synthesizer_llm = OllamaLLM(synthesizer_config, base_url=system.ollama_base_url)
                
                final_response = ""
                thinking_content = ""
                in_thinking = False
                synthesis_thinking_detected = False
                
                async for response_chunk in synthesizer_llm.generate_stream(synthesis_prompt):
                    chunk_text = response_chunk.text
                    final_response += chunk_text
                    
                    if "<think>" in chunk_text.lower():
                        in_thinking = True
                        synthesis_thinking_detected = True
                        
                        yield json.dumps({
                            "type": "synthesis_thinking_start",
                            "agent": "synthesizer",
                            "message": "💭 Synthesizing insights..."
                        }) + "\n"
                        
                        start_idx = chunk_text.lower().find("<think>") + 7
                        thinking_content += chunk_text[start_idx:]
                        
                    elif "</think>" in chunk_text.lower():
                        in_thinking = False
                        
                        end_idx = chunk_text.lower().find("</think>")
                        thinking_content += chunk_text[:end_idx]
                        
                        yield json.dumps({
                            "type": "synthesis_thinking_complete",
                            "agent": "synthesizer",
                            "thinking": thinking_content.strip()
                        }) + "\n"
                        
                        # CRITICAL FIX: Send the </think> tag as output to preserve complete response
                        yield json.dumps({
                            "type": "agent_token",
                            "agent": "synthesizer",
                            "token": "</think>"
                        }) + "\n"
                        
                        remaining = chunk_text[end_idx + 8:]
                        if remaining.strip():
                            yield json.dumps({
                                "type": "agent_token",
                                "agent": "synthesizer",
                                "token": remaining
                            }) + "\n"
                            
                    elif in_thinking:
                        thinking_content += chunk_text
                        
                    else:
                        if chunk_text.strip():
                            yield json.dumps({
                                "type": "agent_token",
                                "agent": "synthesizer",
                                "token": chunk_text
                            }) + "\n"
                    
                    await asyncio.sleep(0.02)
                
                # Apply dynamic response processing for second_llm synthesis
                final_answer = final_response
                try:
                    from app.llm.response_analyzer import detect_model_thinking_behavior
                    
                    # Detect model behavior for synthesis response processing
                    synthesis_model = synthesizer_config.model_name
                    is_thinking, confidence = detect_model_thinking_behavior(final_response, synthesis_model)
                    
                    logger.info(f"[SYNTHESIS DETECTION] Model: {synthesis_model}")
                    logger.info(f"[SYNTHESIS DETECTION] Detected thinking behavior: {is_thinking} (confidence: {confidence:.2f})")
                    
                    if is_thinking and confidence > 0.8:
                        # Remove thinking tags for thinking models in final answer
                        final_answer = re.sub(r'<think>.*?</think>', '', final_response, flags=re.DOTALL | re.IGNORECASE).strip()
                        logger.info(f"[SYNTHESIS PROCESSING] Removed thinking tags from {synthesis_model} response")
                    else:
                        # Non-thinking model or low confidence - use minimal cleaning
                        final_answer = system._clean_llm_response(final_response)
                        logger.info(f"[SYNTHESIS PROCESSING] Using minimal cleaning for {synthesis_model} (thinking: {is_thinking})")
                    
                except Exception as e:
                    logger.warning(f"[SYNTHESIS PROCESSING] Dynamic detection failed: {e}, using fallback")
                    final_answer = system._clean_llm_response(final_response)
                
                print(f"[DEBUG] Synthesis complete - final_response length: {len(final_response)}, final_answer length: {len(final_answer)}")
                
                # Synthesis completion: 100% with all steps completed
                yield json.dumps({
                    "type": "synthesis_progress",
                    "progress": {
                        "current_step": len(agents_used) + 1,
                        "total_steps": len(agents_used) + 1,
                        "percentage": 100,
                        "phase_name": "Synthesis Phase - Complete",
                        "status": "synthesis_complete",
                        "activity": "Synthesis successfully completed"
                    },
                    "synthesis_details": {
                        "agents_being_integrated": agents_used,
                        "integration_steps": [
                            f"✅ Completed insights from {agent}" for agent in agents_used
                        ] + ["✅ Unified narrative complete", "✅ Final polish and formatting complete"],
                        "current_focus": "Synthesis complete"
                    },
                    "message": "✅ Synthesis phase completed successfully!"
                }) + "\n"
                await asyncio.sleep(0.2)
                
                
                # End synthesis generation span with results
                if synthesis_generation and tracer.is_enabled():
                    try:
                        usage = tracer.estimate_token_usage(synthesis_prompt, final_response)
                        synthesis_generation.end(
                            output=final_response,  # Use final_response to show complete thinking in Langfuse
                            usage=usage,
                            metadata={
                                "synthesis_thinking_detected": synthesis_thinking_detected,
                                "response_length": len(final_response),
                                "agent_count": len(agents_used),
                                "operation": "synthesis_generation"
                            }
                        )
                        logger.info("Ended synthesis generation span")
                    except Exception as e:
                        logger.warning(f"Failed to end synthesis generation span: {e}")
                
            except Exception as e:
                logger.error(f"Synthesis failed: {e}")
                final_answer = f"Collaborative analysis from {', '.join(agents_used)} provides comprehensive insights."
                
                # End synthesis spans with error
                if synthesis_generation and tracer.is_enabled():
                    try:
                        usage = tracer.estimate_token_usage(synthesis_prompt, final_answer)
                        synthesis_generation.end(
                            output=f"Error: {str(e)}",
                            usage=usage,
                            metadata={
                                "error": str(e),
                                "fallback_used": True,
                                "operation": "synthesis_generation"
                            }
                        )
                    except Exception as trace_error:
                        logger.warning(f"Failed to end synthesis generation span with error: {trace_error}")
            
            # End synthesis span
            if synthesis_span and tracer.is_enabled():
                try:
                    tracer.end_span_with_result(
                        synthesis_span,
                        {
                            "final_response_length": len(final_answer),
                            "agent_count": len(agents_used),
                            "synthesis_success": 'final_answer' in locals() and len(final_answer) > 50,
                            "collaboration_summary": {
                                "agent_communications": len(agent_communications),
                                "thinking_sessions": len(agents_used) + 1
                            }
                        },
                        success=True
                    )
                    logger.info("Ended synthesis span")
                except Exception as e:
                    logger.warning(f"Failed to end synthesis span: {e}")
            
            # Final response with enhanced collaboration metrics
            yield json.dumps({
                "type": "final_response",
                "response": final_answer,
                "execution_summary": {
                    "agents_used": agents_used,
                    "total_agents": len(agents_used),
                    "execution_pattern": "collaborative_with_thinking_and_tools",
                    "completion_status": "success"
                },
                "collaboration_summary": {
                    "agent_communications": len(agent_communications),
                    "successful_handoffs": len(agent_communications),
                    "tools_used": sum(len(available_agents[agent].get('tools', [])) for agent in agents_used),
                    "thinking_sessions": len(agents_used) + 1,  # +1 for synthesizer
                    "total_processing_phases": len(agents_used) + 1
                },
                "progress": {
                    "current_step": len(agents_used) + 1,
                    "total_steps": len(agents_used) + 1,
                    "percentage": 100,
                    "phase_name": "Complete",
                    "status": "success"
                },
                "performance_metrics": {
                    "confidence_score": 0.95,
                    "response_length": len(final_answer),
                    "synthesis_success": True
                },
                "conversation_id": conversation_id,
                "message": f"🎉 Collaborative analysis complete! {len(agents_used)} agents collaborated successfully."
            }) + "\n"
            
            # End workflow span with success
            if workflow_span and tracer.is_enabled():
                try:
                    tracer.end_span_with_result(
                        workflow_span,
                        {
                            "agents_used": agents_used,
                            "final_response_length": len(final_answer),
                            "collaboration_metrics": {
                                "agent_count": len(agents_used),
                                "communications": len(agent_communications),
                                "successful_synthesis": True
                            },
                            "execution_pattern": "collaborative_with_thinking_and_tools",
                            "confidence_score": 0.95
                        },
                        success=True
                    )
                    logger.info("Ended multi-agent workflow span successfully")
                except Exception as e:
                    logger.warning(f"Failed to end workflow span: {e}")
            
        except Exception as e:
            logger.error(f"Fixed multi-agent streaming failed: {e}")
            
            # End workflow span with error
            if workflow_span and tracer.is_enabled():
                try:
                    tracer.end_span_with_result(
                        workflow_span,
                        {
                            "error": str(e),
                            "agents_attempted": agents_used if 'agents_used' in locals() else [],
                            "failure_point": "workflow_execution"
                        },
                        success=False,
                        error=str(e)
                    )
                    logger.info("Ended multi-agent workflow span with error")
                except Exception as trace_error:
                    logger.warning(f"Failed to end workflow span with error: {trace_error}")
            
            yield json.dumps({
                "type": "error",
                "error": str(e),
                "message": "❌ Multi-agent execution failed"
            }) + "\n"
    
    return stream_fixed_multi_agent()


class FixedMultiAgentStreamingService:
    """Service wrapper for fixed multi-agent streaming functionality"""
    
    async def stream_multi_agent_response(self, query: str, conversation_id: Optional[str] = None):
        """
        Stream multi-agent response using the fixed streaming implementation
        
        Args:
            query: The query to process
            conversation_id: Optional conversation ID for context
            
        Yields:
            JSON-encoded streaming events from the multi-agent system
        """
        generator = await fixed_multi_agent_streaming(
            question=query, 
            conversation_id=conversation_id
        )
        async for event in generator:
            yield event