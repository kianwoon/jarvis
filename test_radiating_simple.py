#!/usr/bin/env python3
"""
Simple test to verify entity extraction field mapping is working
"""

import asyncio
import json
import logging
from app.services.radiating.extraction.universal_entity_extractor import UniversalEntityExtractor

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_extraction_only():
    """Test just the extraction part without type discovery"""
    
    # Initialize the extractor
    extractor = UniversalEntityExtractor()
    
    # Disable type discovery to test extraction only
    extractor.config['enable_universal_discovery'] = False
    
    query = "What are the essential technologies for building RAG systems with LLMs?"
    
    print(f"Query: {query}\n")
    
    # Test extraction
    entities = await extractor.extract_entities(query)
    
    print(f"Extracted {len(entities)} entities:")
    for i, entity in enumerate(entities, 1):
        print(f"{i}. {entity.text} ({entity.entity_type}) - confidence: {entity.confidence:.2f}")
        if hasattr(entity, 'metadata') and 'original_fields' in entity.metadata:
            print(f"   Original fields: {entity.metadata['original_fields']}")

if __name__ == "__main__":
    print("Testing Entity Extraction Field Mapping...\n")
    
    try:
        asyncio.run(test_extraction_only())
        print("\n✅ Test completed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
