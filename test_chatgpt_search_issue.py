#!/usr/bin/env python3
"""
Test script to debug ChatGPT Pro search result inaccuracy issue
"""

import asyncio
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_google_search():
    """Test Google search directly to see what results we get"""
    from app.core.unified_mcp_service import UnifiedMCPService
    
    service = UnifiedMCPService()
    query = "ChatGPT Pro subscription usage limits 2024 2025"
    
    logger.info(f"Testing Google search with query: {query}")
    
    try:
        result = await service._direct_google_search({
            "query": query,
            "num_results": 5
        })
        
        logger.info("Search result structure:")
        logger.info(json.dumps(result, indent=2))
        
        # Extract and log the actual content
        if 'content' in result and len(result['content']) > 0:
            text_content = result['content'][0].get('text', '')
            logger.info(f"Search result text content:\n{text_content}")
            
            # Check for suspicious terms
            if 'o3' in text_content.lower() or 'o4-mini' in text_content.lower():
                logger.warning("⚠️ Found suspicious model names (o3/o4-mini) in search results!")
            else:
                logger.info("✅ No suspicious model names found in search results")
                
    except Exception as e:
        logger.error(f"Search failed: {e}")
    finally:
        await service.close()

async def test_temporal_context():
    """Test temporal context manager to see what date it returns"""
    from app.core.temporal_context_manager import get_temporal_context_manager
    
    manager = get_temporal_context_manager()
    context = manager.get_current_time_context()
    
    logger.info("Temporal context:")
    logger.info(json.dumps(context, indent=2, default=str))
    
    # Check if date is in future
    if '2025' in str(context.get('current_datetime', '')):
        logger.warning("⚠️ System is reporting a future date (2025)!")
    else:
        logger.info("✅ System date appears correct")

async def test_synthesis_prompt():
    """Test how synthesis prompt is built"""
    from app.langchain.service import build_messages_for_synthesis
    
    # Simulate tool context with search results
    tool_context = """google_search: Found 5 search results for 'ChatGPT Pro subscription usage limits':

**ChatGPT Plus Usage Limits and Features**
ChatGPT Plus subscribers get priority access and up to 80 messages every 3 hours with GPT-4
https://openai.com/chatgpt/pricing

**OpenAI Usage Limits Guide**
GPT-4 has different rate limits: 40 messages/3 hours for standard, 80 messages/3 hours for Plus
https://platform.openai.com/docs/guides/rate-limits"""

    question = "what are the usage limit of chatgpt PRO subscription?"
    
    messages, source_label, full_context, system_prompt = build_messages_for_synthesis(
        question=question,
        query_type="TOOLS",
        tool_context=tool_context,
        thinking=False
    )
    
    logger.info("System prompt preview:")
    logger.info(system_prompt[:500])
    
    logger.info("\nMessages structure:")
    for i, msg in enumerate(messages):
        logger.info(f"Message {i} - Role: {msg['role']}")
        if msg['role'] == 'system':
            logger.info(f"System content preview: {msg['content'][:300]}...")
        elif msg['role'] == 'user':
            logger.info(f"User content preview: {msg['content'][:300]}...")

async def test_system_date():
    """Test system date using datetime fallback"""
    from app.core.datetime_fallback import get_current_datetime
    
    result = get_current_datetime()
    logger.info("Datetime fallback result:")
    logger.info(json.dumps(result, indent=2))
    
    # Also check Python's datetime
    from datetime import datetime
    import platform
    
    logger.info(f"Python datetime.now(): {datetime.now()}")
    logger.info(f"System platform: {platform.system()}")
    logger.info(f"System version: {platform.version()}")

async def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("Starting ChatGPT search issue diagnosis")
    logger.info("=" * 60)
    
    # Test 1: Check system date
    logger.info("\n📅 TEST 1: System Date Check")
    await test_system_date()
    
    # Test 2: Check temporal context
    logger.info("\n⏰ TEST 2: Temporal Context Check")
    await test_temporal_context()
    
    # Test 3: Direct Google search
    logger.info("\n🔍 TEST 3: Direct Google Search")
    await test_google_search()
    
    # Test 4: Synthesis prompt building
    logger.info("\n💬 TEST 4: Synthesis Prompt Building")
    await test_synthesis_prompt()
    
    logger.info("\n" + "=" * 60)
    logger.info("Diagnosis complete!")
    logger.info("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())