#!/usr/bin/env python3
"""
Test script to verify radiating entity extraction is working correctly
"""

import asyncio
import json
import logging
from app.services.radiating.extraction.universal_entity_extractor import UniversalEntityExtractor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_entity_extraction():
    """Test entity extraction with various queries"""
    
    # Initialize the extractor
    extractor = UniversalEntityExtractor()
    
    # Test queries
    test_queries = [
        "What are the essential technologies for building RAG systems with LLMs?",
        "Tell me about GPT-4, Claude, and Gemini models",
        "Python frameworks like FastAPI and Django for web development"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        # Extract entities
        entities = await extractor.extract_entities(query)
        
        print(f"\nExtracted {len(entities)} entities:")
        for i, entity in enumerate(entities, 1):
            print(f"{i}. {entity.text} ({entity.entity_type}) - confidence: {entity.confidence:.2f}")
            if hasattr(entity, 'metadata') and 'original_fields' in entity.metadata:
                print(f"   Original fields from LLM: {entity.metadata['original_fields']}")
        
        # Also test comprehensive extraction for the first query
        if "essential technologies" in query.lower():
            print(f"\n{'='*60}")
            print("Testing with comprehensive extraction (web search):")
            print('='*60)
            
            web_entities = await extractor.extract_entities_with_web_search(
                query,
                force_web_search=True
            )
            
            print(f"\nExtracted {len(web_entities)} entities with web search:")
            for i, entity in enumerate(web_entities[:10], 1):  # Show first 10
                print(f"{i}. {entity.text} ({entity.entity_type}) - confidence: {entity.confidence:.2f}")

if __name__ == "__main__":
    print("Testing Radiating Entity Extraction...")
    print("This will test both regular and comprehensive extraction modes.\n")
    
    try:
        asyncio.run(test_entity_extraction())
        print("\n✅ Test completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()