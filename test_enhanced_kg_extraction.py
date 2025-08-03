#!/usr/bin/env python3
"""
Test Enhanced Knowledge Graph Extraction System

Tests the new ultra-aggressive business entity extraction against the DBS Technology Strategy document.
Target: 4x improvement (60-70+ entities from 32k characters).
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from app.services.llm_knowledge_extractor import get_llm_knowledge_extractor
from app.document_handlers.base import ExtractedChunk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_enhanced_extraction():
    """Test the enhanced extraction system with the DBS document"""
    
    # Load the DBS document
    dbs_file_path = Path("test_dbs_sample.txt")
    
    if not dbs_file_path.exists():
        # Try alternative names
        alternatives = ["DBS Technology Strategy (Confidential).pdf", "test_dbs_document.txt"]
        for alt in alternatives:
            alt_path = Path(alt)
            if alt_path.exists():
                dbs_file_path = alt_path
                break
        else:
            logger.error("❌ DBS document not found. Please ensure test file exists.")
            return
    
    # Read the document content
    try:
        if dbs_file_path.suffix == '.txt':
            content = dbs_file_path.read_text(encoding='utf-8')
        else:
            logger.error("❌ PDF reading not implemented in this test. Please convert to .txt first.")
            return
    except Exception as e:
        logger.error(f"❌ Failed to read document: {e}")
        return
    
    logger.info(f"📄 Loaded DBS document: {len(content):,} characters")
    
    # Create test chunks (simulate chunking)
    chunk_size = 2000  # 2K chars per chunk to preserve granularity
    chunks = []
    
    for i in range(0, len(content), chunk_size):
        chunk_content = content[i:i + chunk_size]
        if len(chunk_content.strip()) > 100:  # Skip tiny chunks
            chunk = ExtractedChunk(
                content=chunk_content,
                metadata={
                    'chunk_id': f'dbs_chunk_{i//chunk_size + 1}',
                    'start_char': i,
                    'end_char': i + len(chunk_content),
                    'chunk_index': i // chunk_size
                },
                quality_score=0.8
            )
            chunks.append(chunk)
    
    logger.info(f"🔪 Created {len(chunks)} chunks for processing")
    logger.info(f"   Average chunk size: {sum(len(c.content) for c in chunks) // len(chunks):,} characters")
    
    # Get the enhanced LLM extractor
    extractor = get_llm_knowledge_extractor()
    
    # Test individual chunk processing
    logger.info("\n🚀 TESTING ULTRA-AGGRESSIVE EXTRACTION")
    logger.info("=" * 80)
    
    start_time = time.time()
    all_entities = []
    all_relationships = []
    
    for i, chunk in enumerate(chunks[:10], 1):  # Test first 10 chunks
        logger.info(f"\n📊 Processing chunk {i}/{min(10, len(chunks))} ({len(chunk.content):,} chars)")
        
        try:
            # Use enhanced business extraction
            domain_hints = ['business', 'strategy', 'technology', 'banking', 'financial']
            context = {
                'document_type': 'business_strategy_confidential',
                'chunk_index': i,
                'total_chunks': len(chunks),
                'source': 'DBS Technology Strategy'
            }
            
            result = await extractor.extract_with_llm(chunk.content, context, domain_hints)
            
            logger.info(f"   ✅ Chunk {i}: {len(result.entities)} entities, {len(result.relationships)} relationships")
            logger.info(f"   📈 Entity density: {len(result.entities) / (len(chunk.content) / 1000):.1f} entities per 1K chars")
            
            # Show sample entities
            if result.entities:
                sample_entities = result.entities[:5]
                entity_names = [e.canonical_form for e in sample_entities]
                logger.info(f"   🏷️  Sample entities: {', '.join(entity_names)}")
            
            all_entities.extend(result.entities)
            all_relationships.extend(result.relationships)
            
        except Exception as e:
            logger.error(f"   ❌ Chunk {i} failed: {e}")
            continue
    
    processing_time = time.time() - start_time
    
    # Calculate results
    total_entities = len(all_entities)
    total_relationships = len(all_relationships)
    total_chars_processed = sum(len(c.content) for c in chunks[:10])
    entity_density = total_entities / (total_chars_processed / 1000)
    
    logger.info("\n" + "=" * 80)
    logger.info("🎯 ENHANCED EXTRACTION RESULTS")
    logger.info("=" * 80)
    logger.info(f"📊 Total entities extracted: {total_entities}")
    logger.info(f"🔗 Total relationships extracted: {total_relationships}")
    logger.info(f"📄 Characters processed: {total_chars_processed:,}")
    logger.info(f"📈 Entity density: {entity_density:.1f} entities per 1K chars")
    logger.info(f"⏱️  Processing time: {processing_time:.1f} seconds")
    
    # Compare with target
    target_density = 2.0  # Target: 2 entities per 1K chars
    improvement_factor = entity_density / 0.53  # Original was 0.53 entities per 1K
    
    logger.info(f"\n🚀 PERFORMANCE ANALYSIS:")
    logger.info(f"   Target density: {target_density:.1f} entities per 1K chars")
    logger.info(f"   Achieved density: {entity_density:.1f} entities per 1K chars")
    logger.info(f"   Improvement factor: {improvement_factor:.1f}x (target: 4x)")
    
    if improvement_factor >= 4.0:
        logger.info("   ✅ SUCCESS: Achieved 4x improvement target!")
    elif improvement_factor >= 3.0:
        logger.info("   🟡 GOOD: Close to 4x improvement target")
    else:
        logger.info("   🔴 NEEDS IMPROVEMENT: Below target - consider further tuning")
    
    # Show entity type distribution
    entity_types = {}
    for entity in all_entities:
        entity_types[entity.label] = entity_types.get(entity.label, 0) + 1
    
    logger.info(f"\n📊 ENTITY TYPE DISTRIBUTION:")
    for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True)[:15]:
        logger.info(f"   {entity_type}: {count}")
    
    # Show sample high-confidence entities
    high_conf_entities = [e for e in all_entities if e.confidence >= 0.7]
    logger.info(f"\n🎯 HIGH-CONFIDENCE ENTITIES ({len(high_conf_entities)}):")
    for entity in high_conf_entities[:20]:
        logger.info(f"   {entity.canonical_form} ({entity.label}) - {entity.confidence:.2f}")
    
    # Estimate full document projection
    full_doc_entity_projection = int(entity_density * (len(content) / 1000))
    logger.info(f"\n🔮 FULL DOCUMENT PROJECTION:")
    logger.info(f"   Estimated total entities: {full_doc_entity_projection}")
    logger.info(f"   Target was: 60-70 entities")
    
    if full_doc_entity_projection >= 60:
        logger.info("   ✅ SUCCESS: Projected to meet 60-70 entity target!")
    else:
        logger.info("   🔴 Below target projection")
    
    return {
        'total_entities': total_entities,
        'total_relationships': total_relationships,
        'entity_density': entity_density,
        'improvement_factor': improvement_factor,
        'processing_time': processing_time,
        'full_doc_projection': full_doc_entity_projection
    }

if __name__ == "__main__":
    try:
        result = asyncio.run(test_enhanced_extraction())
        print(f"\n🎯 Test completed successfully!")
    except KeyboardInterrupt:
        print(f"\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()