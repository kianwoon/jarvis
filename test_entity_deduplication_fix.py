#!/usr/bin/env python3
"""
Test script to validate the entity deduplication fix in multi-chunk processing
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.multi_chunk_processor import get_multi_chunk_processor
from app.services.knowledge_graph_types import ExtractedEntity, ExtractedRelationship
from app.document_handlers.base import ExtractedChunk

async def test_entity_deduplication_fix():
    """Test that the entity deduplication fix preserves entities from different chunks"""
    print("🧪 Testing entity deduplication fix...")
    
    # Create test entities from different chunks
    entities_chunk1 = [
        ExtractedEntity(
            text="DBS Bank",
            canonical_form="DBS Bank",
            label="ORGANIZATION",
            confidence=0.9,
            properties={"chunk_id": "chunk_1"}
        ),
        ExtractedEntity(
            text="Oracle Database",
            canonical_form="Oracle Database", 
            label="TECHNOLOGY",
            confidence=0.8,
            properties={"chunk_id": "chunk_1"}
        ),
        ExtractedEntity(
            text="Cloud Migration",
            canonical_form="Cloud Migration",
            label="CONCEPT", 
            confidence=0.7,
            properties={"chunk_id": "chunk_1"}
        )
    ]
    
    entities_chunk2 = [
        ExtractedEntity(
            text="DBS Bank",  # Same name, should be deduplicated carefully
            canonical_form="DBS Bank",
            label="ORGANIZATION",
            confidence=0.85,  # Slightly lower confidence
            properties={"chunk_id": "chunk_2"}
        ),
        ExtractedEntity(
            text="MariaDB",
            canonical_form="MariaDB",
            label="TECHNOLOGY",
            confidence=0.9,
            properties={"chunk_id": "chunk_2"}
        ),
        ExtractedEntity(
            text="Digital Transformation",
            canonical_form="Digital Transformation",
            label="CONCEPT",
            confidence=0.8,
            properties={"chunk_id": "chunk_2"}
        )
    ]
    
    # Cross-chunk entities (new entities discovered in multi-chunk analysis)
    cross_chunk_entities = [
        ExtractedEntity(
            text="Technology Stack Modernization",
            canonical_form="Technology Stack Modernization", 
            label="CONCEPT",
            confidence=0.6,
            properties={"cross_chunk": True, "source_window": "1_2"}
        )
    ]
    
    # Combine all entities
    all_entities = entities_chunk1 + entities_chunk2 + cross_chunk_entities
    print(f"📊 Input entities: {len(all_entities)}")
    for i, entity in enumerate(all_entities, 1):
        chunk_info = entity.properties.get('chunk_id', 'cross-chunk') if entity.properties else 'unknown'
        print(f"   {i}. {entity.canonical_form} ({entity.label}) - {chunk_info}")
    
    # Test the enhanced deduplication
    processor = get_multi_chunk_processor()
    high_freq_entities = {
        "dbs bank": {"count": 2}  # Appears in both chunks
    }
    
    # Test the fixed deduplication
    deduplicated = processor._deduplicate_entities_enhanced(all_entities, high_freq_entities)
    
    print(f"\n🔧 After enhanced deduplication: {len(deduplicated)}")
    for i, entity in enumerate(deduplicated, 1):
        chunk_info = entity.properties.get('chunk_id', 'cross-chunk') if entity.properties else 'unknown'
        print(f"   {i}. {entity.canonical_form} ({entity.label}) - {chunk_info} - Conf: {entity.confidence:.2f}")
    
    # Analyze results
    chunk1_entities = [e for e in deduplicated if e.properties and e.properties.get('chunk_id') == 'chunk_1']
    chunk2_entities = [e for e in deduplicated if e.properties and e.properties.get('chunk_id') == 'chunk_2']
    cross_chunk_entities_result = [e for e in deduplicated if e.properties and e.properties.get('cross_chunk')]
    
    print(f"\n📈 Results Analysis:")
    print(f"   Chunk 1 entities: {len(chunk1_entities)}")
    print(f"   Chunk 2 entities: {len(chunk2_entities)}")
    print(f"   Cross-chunk entities: {len(cross_chunk_entities_result)}")
    print(f"   Total preserved: {len(deduplicated)}")
    
    # Verify the fix works
    expected_min_entities = 6  # Should preserve most entities from different chunks
    if len(deduplicated) >= expected_min_entities:
        print(f"✅ DEDUPLICATION FIX SUCCESSFUL: Preserved {len(deduplicated)} entities (≥{expected_min_entities} expected)")
        return True
    else:
        print(f"❌ DEDUPLICATION FIX FAILED: Only preserved {len(deduplicated)} entities (<{expected_min_entities} expected)")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_entity_deduplication_fix())
    sys.exit(0 if success else 1)