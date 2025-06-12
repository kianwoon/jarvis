"""
Simple tool executor that doesn't rely on LLM formatting
"""
import json
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

def identify_and_execute_tools(question: str) -> Tuple[List[Dict], str]:
    """
    Identify which tools to use based on the question and execute them directly
    
    Returns:
        (tool_results, tool_context)
    """
    from app.core.mcp_tools_cache import get_enabled_mcp_tools
    
    question_lower = question.lower()
    tool_results = []
    
    # Get available tools
    enabled_tools = get_enabled_mcp_tools()
    
    # Simple pattern matching for tool selection
    if any(word in question_lower for word in ['search', 'internet', 'web', 'google', 'find', 'latest', 'current', 'news']):
        # Execute google_search if available
        if 'google_search' in enabled_tools:
            logger.info("[TOOL] Executing google_search based on query keywords")
            
            # Extract search query by removing instruction words
            search_query = question
            for word in ['search', 'internet', 'web', 'google', 'find', 'about', 'for']:
                search_query = search_query.replace(word, '').strip()
            
            # Execute the tool
            try:
                from app.langchain.service import call_mcp_tool
                
                result = call_mcp_tool('google_search', {
                    'query': search_query,
                    'num_results': 10
                })
                
                tool_results.append({
                    'tool': 'google_search',
                    'success': True,
                    'result': result
                })
                
                logger.info(f"[TOOL] google_search executed successfully, got {len(str(result))} chars")
            except Exception as e:
                logger.error(f"[TOOL] google_search failed: {e}")
                tool_results.append({
                    'tool': 'google_search',
                    'success': False,
                    'error': str(e)
                })
    
    elif any(word in question_lower for word in ['time', 'date', 'today', 'now', 'current time', 'current date']):
        # Execute get_datetime if available
        if 'get_datetime' in enabled_tools:
            logger.info("[TOOL] Executing get_datetime based on query keywords")
            
            try:
                from app.langchain.service import call_mcp_tool
                
                result = call_mcp_tool('get_datetime', {})
                
                tool_results.append({
                    'tool': 'get_datetime',
                    'success': True,
                    'result': result
                })
                
                logger.info(f"[TOOL] get_datetime executed successfully")
            except Exception as e:
                logger.error(f"[TOOL] get_datetime failed: {e}")
                tool_results.append({
                    'tool': 'get_datetime',
                    'success': False,
                    'error': str(e)
                })
    
    # Build tool context
    tool_context = ""
    if tool_results:
        tool_context = "Tool Execution Results:\n\n"
        for result in tool_results:
            if result['success']:
                tool_context += f"🔧 {result['tool']}:\n"
                tool_context += f"{json.dumps(result['result'], indent=2)}\n\n"
            else:
                tool_context += f"❌ {result['tool']} failed: {result.get('error', 'Unknown error')}\n\n"
    
    return tool_results, tool_context