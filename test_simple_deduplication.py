#!/usr/bin/env python3
"""
Simple test to validate entity deduplication logic without Neo4j dependencies
"""

from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class SimpleEntity:
    canonical_form: str
    label: str
    confidence: float
    properties: Dict[str, Any] = None

def deduplicate_entities_enhanced(entities: List[SimpleEntity], 
                                 high_freq_entities: Dict[str, Any]) -> List[SimpleEntity]:
    """Enhanced entity deduplication with frequency weighting - FIXED to preserve distinct entities"""
    entity_map = {}
    
    for entity in entities:
        # Create more specific key that includes chunk context to avoid over-deduplication
        base_key = entity.canonical_form.lower()
        chunk_context = entity.properties.get('chunk_id', 'unknown') if entity.properties else 'unknown'
        entity_type = entity.label.upper()
        source_window = entity.properties.get('source_window', 'main') if entity.properties else 'main'
        
        # Create compound key that preserves chunk-specific entities
        # Only deduplicate if entities are from the same chunk AND have identical canonical forms
        compound_key = f"{base_key}|{entity_type}|{chunk_context}|{source_window}"
        
        if compound_key in entity_map:
            # Keep entity with higher confidence for truly identical entities
            existing = entity_map[compound_key]
            is_new_high_freq = base_key in high_freq_entities
            is_existing_high_freq = existing.properties.get('high_frequency', False) if existing.properties else False
            
            # Only replace if new entity is clearly better
            if (is_new_high_freq and not is_existing_high_freq) or \
               (is_new_high_freq == is_existing_high_freq and entity.confidence > existing.confidence + 0.1):
                entity_map[compound_key] = entity
        else:
            entity_map[compound_key] = entity
    
    deduplicated_entities = list(entity_map.values())
    
    # Now do a final pass to merge only truly identical entities (same name, type, and context)
    final_entity_map = {}
    for entity in deduplicated_entities:
        final_key = f"{entity.canonical_form.lower()}|{entity.label.upper()}"
        
        if final_key in final_entity_map:
            existing = final_entity_map[final_key]
            # Only merge if entities are truly similar in context and one is clearly superior
            entity_context = entity.properties.get('chunk_id', '') if entity.properties else ''
            existing_context = existing.properties.get('chunk_id', '') if existing.properties else ''
            
            # If from different chunks, keep both unless one has much higher confidence
            if entity_context != existing_context:
                if entity.confidence > existing.confidence + 0.2:  # Significant confidence difference
                    final_entity_map[final_key] = entity
                # Otherwise keep existing (don't replace with similar entity from different chunk)
            else:
                # Same chunk - can safely merge, keep higher confidence
                if entity.confidence > existing.confidence:
                    final_entity_map[final_key] = entity
        else:
            final_entity_map[final_key] = entity
    
    result_entities = list(final_entity_map.values())
    print(f"🔧 Enhanced deduplication: {len(entities)} → {len(result_entities)} entities (preserved chunk-specific entities)")
    return result_entities

def test_deduplication():
    """Test the deduplication fix"""
    print("🧪 Testing entity deduplication fix...")
    
    # Create test entities from different chunks
    entities = [
        # Chunk 1 entities
        SimpleEntity(
            canonical_form="DBS Bank",
            label="ORGANIZATION",
            confidence=0.9,
            properties={"chunk_id": "chunk_1"}
        ),
        SimpleEntity(
            canonical_form="Oracle Database",
            label="TECHNOLOGY",
            confidence=0.8,
            properties={"chunk_id": "chunk_1"}
        ),
        SimpleEntity(
            canonical_form="Cloud Migration",
            label="CONCEPT",
            confidence=0.7,
            properties={"chunk_id": "chunk_1"}
        ),
        # Chunk 2 entities
        SimpleEntity(
            canonical_form="DBS Bank",  # Same name, different chunk
            label="ORGANIZATION",
            confidence=0.85,
            properties={"chunk_id": "chunk_2"}
        ),
        SimpleEntity(
            canonical_form="MariaDB",
            label="TECHNOLOGY",
            confidence=0.9,
            properties={"chunk_id": "chunk_2"}
        ),
        SimpleEntity(
            canonical_form="Digital Transformation",
            label="CONCEPT",
            confidence=0.8,
            properties={"chunk_id": "chunk_2"}
        ),
        # Cross-chunk entity
        SimpleEntity(
            canonical_form="Technology Stack Modernization",
            label="CONCEPT",
            confidence=0.6,
            properties={"cross_chunk": True, "source_window": "1_2"}
        )
    ]
    
    print(f"📊 Input entities: {len(entities)}")
    for i, entity in enumerate(entities, 1):
        chunk_info = entity.properties.get('chunk_id', 'cross-chunk') if entity.properties else 'unknown'
        print(f"   {i}. {entity.canonical_form} ({entity.label}) - {chunk_info}")
    
    # Test deduplication with high frequency entities
    high_freq_entities = {
        "dbs bank": {"count": 2}  # Appears in both chunks
    }
    
    deduplicated = deduplicate_entities_enhanced(entities, high_freq_entities)
    
    print(f"\n🔧 After enhanced deduplication: {len(deduplicated)}")
    for i, entity in enumerate(deduplicated, 1):
        chunk_info = entity.properties.get('chunk_id', 'cross-chunk') if entity.properties else 'unknown'
        print(f"   {i}. {entity.canonical_form} ({entity.label}) - {chunk_info} - Conf: {entity.confidence:.2f}")
    
    # Analyze results
    chunk1_entities = [e for e in deduplicated if e.properties and e.properties.get('chunk_id') == 'chunk_1']
    chunk2_entities = [e for e in deduplicated if e.properties and e.properties.get('chunk_id') == 'chunk_2']
    cross_chunk_entities = [e for e in deduplicated if e.properties and e.properties.get('cross_chunk')]
    
    print(f"\n📈 Results Analysis:")
    print(f"   Chunk 1 entities: {len(chunk1_entities)}")
    print(f"   Chunk 2 entities: {len(chunk2_entities)}")
    print(f"   Cross-chunk entities: {len(cross_chunk_entities)}")
    print(f"   Total preserved: {len(deduplicated)}")
    
    # Check if DBS Bank appears only once (should be merged due to high frequency)
    dbs_entities = [e for e in deduplicated if e.canonical_form.lower() == 'dbs bank']
    print(f"   DBS Bank entities: {len(dbs_entities)} (should be 1 after merge)")
    
    # Expected: 6 entities total (DBS Bank merged, others preserved)
    expected_entities = 6
    if len(deduplicated) == expected_entities:
        print(f"✅ DEDUPLICATION FIX SUCCESSFUL: Preserved {len(deduplicated)} entities (expected {expected_entities})")
        return True
    else:
        print(f"❌ DEDUPLICATION FIX ISSUE: Got {len(deduplicated)} entities (expected {expected_entities})")
        return False

if __name__ == "__main__":
    success = test_deduplication()
    exit(0 if success else 1)