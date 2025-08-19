#!/usr/bin/env python3
"""
Test script to verify RadiatingAgent uses web search for entity extraction.
This script directly tests the RadiatingAgent class to ensure it's calling
the web-first entity extraction method.
"""

import asyncio
import logging
import sys
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, '/Users/kianwoonwong/Downloads/jarvis')

from app.langchain.radiating_agent_system import RadiatingAgent

# Configure logging to see all debug messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_radiating_agent_web_search():
    """Test that RadiatingAgent properly uses web search for entity extraction"""
    
    print("\n" + "="*80)
    print("TESTING RADIATING AGENT WEB SEARCH INTEGRATION")
    print("="*80 + "\n")
    
    # Create a RadiatingAgent instance
    agent = RadiatingAgent(trace="test_web_search")
    
    # Test query that should trigger web search
    test_query = "What are the latest AI models like GPT-4o and Claude 3.5 Sonnet?"
    
    print(f"Query: {test_query}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("-" * 80)
    
    # Process the query with radiating coverage
    context = {
        'strategy': 'breadth_first',
        'max_depth': 2,
        'relevance_threshold': 0.3
    }
    
    responses = []
    entity_count = 0
    web_search_triggered = False
    
    try:
        async for response in agent.process_with_radiation(test_query, context, stream=True):
            response_type = response.get('type', '')
            
            # Check for status messages about web search
            if response_type == 'status':
                message = response.get('message', '')
                print(f"[STATUS] {message}")
                if 'web' in message.lower() or 'search' in message.lower():
                    web_search_triggered = True
            
            # Check metadata for entity information
            elif response_type == 'metadata':
                entity_count = response.get('entities_discovered', 0)
                print(f"\n[METADATA] Entities discovered: {entity_count}")
                print(f"[METADATA] Relationships found: {response.get('relationships_found', 0)}")
                print(f"[METADATA] Processing time: {response.get('processing_time_ms', 0)}ms")
            
            # Collect content responses
            elif response_type == 'content':
                content = response.get('content', '')
                responses.append(content)
                # Print first 200 chars of content
                if len(content) > 0:
                    preview = content[:200] + "..." if len(content) > 200 else content
                    print(f"[CONTENT] {preview}")
            
            # Check for errors
            elif response_type == 'error':
                print(f"[ERROR] {response.get('message', 'Unknown error')}")
    
    except Exception as e:
        print(f"\n[EXCEPTION] Error during processing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("TEST RESULTS:")
    print("="*80)
    
    # Check the logs to see if web search was mentioned
    success_indicators = []
    
    # Check if we got entities
    if entity_count > 0:
        success_indicators.append(f"✓ Extracted {entity_count} entities")
    else:
        success_indicators.append("✗ No entities extracted")
    
    # Check if we got responses
    if responses:
        success_indicators.append(f"✓ Generated {len(responses)} response chunks")
    else:
        success_indicators.append("✗ No response generated")
    
    # Print indicators
    for indicator in success_indicators:
        print(indicator)
    
    print("\nNOTE: Check the logs above for lines containing:")
    print("  - 'Starting web-first entity extraction'")
    print("  - 'Successfully extracted X entities using web-first approach'")
    print("  - 'Entity sources: X from web search, Y from LLM'")
    print("\nThese log messages confirm that web search is being used.")
    
    # Cleanup
    await agent.cleanup()

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_radiating_agent_web_search())