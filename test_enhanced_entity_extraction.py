#!/usr/bin/env python3
"""
Test script for enhanced UniversalEntityExtractor
Verifies extraction of 30+ entities for technology queries
"""

import asyncio
import json
from app.services.radiating.extraction.universal_entity_extractor import UniversalEntityExtractor

async def test_comprehensive_extraction():
    """Test entity extraction for technology queries"""
    
    # Initialize the extractor
    extractor = UniversalEntityExtractor()
    
    # Test queries
    test_queries = [
        "what are the essential technologies of AI implementation, favor open source",
        "list all machine learning frameworks and tools",
        "what technologies are used for building LLM applications",
        "essential tools for data science and analytics"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Testing query: {query}")
        print(f"{'='*80}")
        
        # Check if comprehensive detection works
        is_comprehensive = extractor._is_comprehensive_technology_query(query)
        print(f"Detected as comprehensive query: {is_comprehensive}")
        
        # Extract entities
        entities = await extractor.extract_entities(query)
        
        print(f"\nExtracted {len(entities)} entities:")
        
        # Group entities by type
        entities_by_type = {}
        for entity in entities:
            entity_type = entity.entity_type
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)
        
        # Display entities by type
        for entity_type, type_entities in entities_by_type.items():
            print(f"\n{entity_type} ({len(type_entities)} entities):")
            for entity in type_entities[:10]:  # Show first 10 of each type
                print(f"  - {entity.text} (confidence: {entity.confidence:.2f})")
                if entity.metadata.get('reason'):
                    print(f"    Reason: {entity.metadata['reason']}")
            if len(type_entities) > 10:
                print(f"  ... and {len(type_entities) - 10} more")
        
        # Summary statistics
        print(f"\n{'-'*40}")
        print(f"Summary:")
        print(f"  Total entities extracted: {len(entities)}")
        print(f"  Unique entity types: {len(entities_by_type)}")
        print(f"  Average confidence: {sum(e.confidence for e in entities) / len(entities) if entities else 0:.2f}")
        print(f"  Extraction method: {entities[0].metadata.get('extraction_method', 'unknown') if entities else 'N/A'}")
        
        # Verify we get 30+ entities for AI/ML queries
        if 'ai' in query.lower() or 'machine learning' in query.lower():
            if len(entities) >= 30:
                print(f"  ✅ SUCCESS: Extracted {len(entities)} entities (>= 30 required)")
            else:
                print(f"  ❌ FAILURE: Only extracted {len(entities)} entities (expected >= 30)")

async def test_regular_vs_comprehensive():
    """Compare regular extraction vs comprehensive extraction"""
    
    print(f"\n{'='*80}")
    print("Comparing Regular vs Comprehensive Extraction")
    print(f"{'='*80}")
    
    extractor = UniversalEntityExtractor()
    
    # Test with a simple non-technology query
    simple_query = "The meeting is scheduled for tomorrow at 3pm"
    print(f"\nSimple query: {simple_query}")
    
    is_comprehensive = extractor._is_comprehensive_technology_query(simple_query)
    print(f"Detected as comprehensive: {is_comprehensive}")
    
    entities = await extractor.extract_entities(simple_query)
    print(f"Entities extracted: {len(entities)}")
    for entity in entities:
        print(f"  - {entity.text} ({entity.entity_type})")

if __name__ == "__main__":
    print("Testing Enhanced Universal Entity Extractor")
    print("="*80)
    
    # Run the async tests
    asyncio.run(test_comprehensive_extraction())
    asyncio.run(test_regular_vs_comprehensive())
    
    print("\n" + "="*80)
    print("Test complete!")