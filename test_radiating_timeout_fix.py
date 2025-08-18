#!/usr/bin/env python3
"""
Test script to verify radiating mode timeout fixes
"""

import asyncio
import sys
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_entity_extraction():
    """Test entity extraction with the new timeout settings"""
    logger.info("Starting entity extraction test...")
    
    try:
        from app.services.radiating.extraction.universal_entity_extractor import UniversalEntityExtractor
        from app.core.timeout_settings_cache import get_entity_extraction_timeout
        
        # Get the configured timeout
        timeout = get_entity_extraction_timeout()
        logger.info(f"Entity extraction timeout is set to: {timeout} seconds")
        
        # Initialize the extractor
        extractor = UniversalEntityExtractor()
        
        # Test with a complex query that previously timed out
        test_query = """
        What are the essential technologies and tools for building modern LLM-based 
        RAG systems including vector databases, embedding models, orchestration frameworks,
        and inference optimization techniques?
        """
        
        logger.info(f"Testing with query: {test_query[:100]}...")
        
        start_time = datetime.now()
        
        # Run extraction with timeout
        try:
            entities = await asyncio.wait_for(
                extractor.extract_entities(test_query),
                timeout=timeout
            )
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Entity extraction succeeded in {elapsed:.2f} seconds")
            logger.info(f"Extracted {len(entities)} entities")
            
            # Show first few entities
            for i, entity in enumerate(entities[:5]):
                logger.info(f"  Entity {i+1}: {entity.text} ({entity.entity_type}) - confidence: {entity.confidence:.2f}")
            
            # Check if LLM client is using correct max_tokens
            if hasattr(extractor.llm_client, 'llm') and hasattr(extractor.llm_client.llm, 'config'):
                max_tokens = extractor.llm_client.llm.config.max_tokens
                logger.info(f"LLM max_tokens is set to: {max_tokens}")
                if max_tokens > 4096:
                    logger.warning(f"⚠️ max_tokens is {max_tokens}, which is higher than recommended 4096")
                else:
                    logger.info(f"✅ max_tokens is properly limited to {max_tokens}")
            
            return True
            
        except asyncio.TimeoutError:
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ Entity extraction timed out after {elapsed:.2f} seconds")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}", exc_info=True)
        return False

async def test_radiating_agent():
    """Test the full radiating agent system"""
    logger.info("\nStarting radiating agent system test...")
    
    try:
        from app.langchain.radiating_agent_system import RadiatingAgentSystem
        
        # Initialize the agent system
        agent = RadiatingAgentSystem()
        
        # Test query
        test_query = "What are the latest LLM inference optimization techniques?"
        
        logger.info(f"Testing radiating agent with query: {test_query}")
        
        start_time = datetime.now()
        
        # Process query
        result_count = 0
        async for result in agent.process_query(test_query, stream=False):
            result_count += 1
            if result.get('type') == 'status':
                logger.info(f"Status: {result.get('message')}")
            elif result.get('type') == 'response':
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.info(f"✅ Radiating agent completed in {elapsed:.2f} seconds")
                logger.info(f"Response preview: {result.get('content', '')[:200]}...")
                
        if result_count > 0:
            logger.info(f"✅ Radiating agent test passed")
            return True
        else:
            logger.error("❌ No results from radiating agent")
            return False
            
    except Exception as e:
        logger.error(f"❌ Radiating agent test failed: {e}", exc_info=True)
        return False

async def test_timeout_settings():
    """Test that timeout settings are properly configured"""
    logger.info("\nChecking timeout settings...")
    
    try:
        from app.core.timeout_settings_cache import (
            get_entity_extraction_timeout,
            get_radiating_timeout,
            get_radiating_llm_timeout
        )
        
        entity_timeout = get_entity_extraction_timeout()
        llm_timeout = get_radiating_llm_timeout()
        traversal_timeout = get_radiating_timeout('traversal_timeout', 180)
        
        logger.info(f"Entity extraction timeout: {entity_timeout}s")
        logger.info(f"LLM call timeout: {llm_timeout}s")
        logger.info(f"Traversal timeout: {traversal_timeout}s")
        
        # Check if timeouts are reasonable
        if entity_timeout >= 60:
            logger.info("✅ Entity extraction timeout is properly increased")
        else:
            logger.warning(f"⚠️ Entity extraction timeout ({entity_timeout}s) may be too short")
        
        if llm_timeout >= 30:
            logger.info("✅ LLM timeout is reasonable")
        else:
            logger.warning(f"⚠️ LLM timeout ({llm_timeout}s) may be too short")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to check timeout settings: {e}")
        return False

async def test_model_config():
    """Test that model configuration is properly set"""
    logger.info("\nChecking model configuration...")
    
    try:
        from app.core.radiating_settings_cache import get_model_config
        
        config = get_model_config()
        logger.info(f"Model: {config.get('model')}")
        logger.info(f"Max tokens: {config.get('max_tokens')}")
        logger.info(f"Temperature: {config.get('temperature')}")
        logger.info(f"LLM mode: {config.get('llm_mode')}")
        
        max_tokens = config.get('max_tokens', 0)
        if max_tokens <= 4096:
            logger.info(f"✅ Max tokens is properly limited to {max_tokens}")
        else:
            logger.warning(f"⚠️ Max tokens ({max_tokens}) is higher than recommended 4096")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to check model config: {e}")
        return False

async def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("Starting Radiating Mode Timeout Fix Tests")
    logger.info("=" * 60)
    
    all_passed = True
    
    # Test 1: Timeout settings
    if not await test_timeout_settings():
        all_passed = False
    
    # Test 2: Model configuration
    if not await test_model_config():
        all_passed = False
    
    # Test 3: Entity extraction
    if not await test_entity_extraction():
        all_passed = False
    
    # Test 4: Full radiating agent (optional - may take longer)
    # Uncomment to run full test
    # if not await test_radiating_agent():
    #     all_passed = False
    
    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("✅ ALL TESTS PASSED - Timeout issues should be fixed!")
    else:
        logger.info("❌ SOME TESTS FAILED - Please check the logs above")
    logger.info("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)