#!/usr/bin/env python3
"""
Complete Knowledge Graph Pipeline Validation Test

This script comprehensively tests the entire knowledge graph ingestion pipeline
to verify all fixes are working correctly:

✅ Fixed LLM connectivity (host.docker.internal:11434)
✅ Fixed Neo4j connectivity (neo4j:7687) 
✅ Fixed chunking strategy (5-8 balanced chunks instead of mega chunks)

Test Coverage:
1. LLM Service Connectivity Test
2. Neo4j Service Connectivity Test
3. Chunking Strategy Validation
4. Entity Extraction Validation
5. Relationship Extraction Validation
6. Relationship Ratio Enforcement (≤4 per entity)
7. Neo4j Storage Validation
8. End-to-End Pipeline Integration Test
"""

import asyncio
import logging
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class KnowledgeGraphPipelineValidator:
    """Comprehensive validator for the knowledge graph ingestion pipeline"""
    
    def __init__(self):
        self.test_results = {
            'llm_connectivity': None,
            'neo4j_connectivity': None,
            'chunking_strategy': None,
            'entity_extraction': None,
            'relationship_extraction': None,
            'relationship_ratio': None,
            'neo4j_storage': None,
            'end_to_end_pipeline': None
        }
        self.test_document_path = project_root / "test_kg_pipeline_validation.txt"
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive pipeline validation tests"""
        logger.info("🚀 Starting Complete Knowledge Graph Pipeline Validation")
        logger.info("=" * 70)
        
        try:
            # Test 1: LLM Service Connectivity
            logger.info("\n1️⃣  Testing LLM Service Connectivity...")
            self.test_results['llm_connectivity'] = await self.test_llm_connectivity()
            
            # Test 2: Neo4j Service Connectivity  
            logger.info("\n2️⃣  Testing Neo4j Service Connectivity...")
            self.test_results['neo4j_connectivity'] = await self.test_neo4j_connectivity()
            
            # Test 3: Chunking Strategy Validation
            logger.info("\n3️⃣  Testing Chunking Strategy...")
            self.test_results['chunking_strategy'] = await self.test_chunking_strategy()
            
            # Test 4: Entity Extraction Validation
            logger.info("\n4️⃣  Testing Entity Extraction...")
            self.test_results['entity_extraction'] = await self.test_entity_extraction()
            
            # Test 5: Relationship Extraction Validation
            logger.info("\n5️⃣  Testing Relationship Extraction...")
            self.test_results['relationship_extraction'] = await self.test_relationship_extraction()
            
            # Test 6: Relationship Ratio Enforcement
            logger.info("\n6️⃣  Testing Relationship Ratio Enforcement...")
            self.test_results['relationship_ratio'] = await self.test_relationship_ratio()
            
            # Test 7: Neo4j Storage Validation
            logger.info("\n7️⃣  Testing Neo4j Storage...")
            self.test_results['neo4j_storage'] = await self.test_neo4j_storage()
            
            # Test 8: End-to-End Pipeline Integration
            logger.info("\n8️⃣  Testing End-to-End Pipeline Integration...")
            self.test_results['end_to_end_pipeline'] = await self.test_end_to_end_pipeline()
            
            # Generate final report
            self.generate_final_report()
            
            return self.test_results
            
        except Exception as e:
            logger.error(f"❌ Pipeline validation failed: {e}")
            logger.exception("Full exception details:")
            return self.test_results
    
    async def test_llm_connectivity(self) -> Dict[str, Any]:
        """Test LLM service connectivity with fixed host configuration"""
        try:
            from app.core.llm_settings_cache import get_llm_settings
            from app.llm.inference import LLMInferenceService
            
            # Get LLM settings to verify host configuration
            llm_settings = get_llm_settings()
            llm_host = llm_settings.get('host', 'localhost')
            llm_port = llm_settings.get('port', 11434)
            
            logger.info(f"🔍 Testing LLM connectivity at {llm_host}:{llm_port}")
            
            # Expected: host.docker.internal:11434 (fixed configuration)
            expected_host = "host.docker.internal"
            if llm_host != expected_host:
                return {
                    'success': False,
                    'error': f'LLM host not properly configured: {llm_host} (expected: {expected_host})',
                    'details': {'current_host': llm_host, 'expected_host': expected_host, 'port': llm_port}
                }
            
            # Test actual connectivity
            llm_service = LLMInferenceService()
            
            # Try a simple test query
            test_prompt = "Test connectivity. Respond with: CONNECTED"
            start_time = time.time()
            
            response = await llm_service.generate_response(
                prompt=test_prompt,
                temperature=0.1,
                max_tokens=10,
                timeout=30
            )
            
            response_time = time.time() - start_time
            
            if response and 'CONNECTED' in response.upper():
                return {
                    'success': True,
                    'host': llm_host,
                    'port': llm_port,
                    'response_time_ms': response_time * 1000,
                    'response_preview': response[:50] + "..." if len(response) > 50 else response
                }
            else:
                return {
                    'success': False,
                    'error': 'LLM service not responding correctly',
                    'details': {'host': llm_host, 'port': llm_port, 'response': response}
                }
                
        except Exception as e:
            logger.error(f"❌ LLM connectivity test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'details': {'exception_type': type(e).__name__}
            }
    
    async def test_neo4j_connectivity(self) -> Dict[str, Any]:
        """Test Neo4j service connectivity with fixed host configuration"""
        try:
            from app.services.neo4j_service import get_neo4j_service
            
            neo4j_service = get_neo4j_service()
            
            if not neo4j_service.is_enabled():
                return {
                    'success': False,
                    'error': 'Neo4j service not enabled',
                    'details': {'enabled': False}
                }
            
            # Test basic connectivity
            logger.info("🔍 Testing Neo4j basic connectivity...")
            
            # Get database info
            db_info = neo4j_service.get_database_info()
            if not db_info:
                return {
                    'success': False,
                    'error': 'Cannot retrieve Neo4j database info',
                    'details': {'db_info': None}
                }
            
            # Test simple query
            test_query = "RETURN 'CONNECTED' as status, datetime() as timestamp"
            result = neo4j_service.execute_cypher(test_query)
            
            if result and len(result) > 0 and result[0].get('status') == 'CONNECTED':
                # Get current entity and relationship counts
                entity_count = neo4j_service.get_total_entity_count()
                relationship_count = neo4j_service.get_total_relationship_count()
                current_ratio = relationship_count / max(entity_count, 1)
                
                return {
                    'success': True,
                    'database_info': db_info,
                    'connectivity_test': result[0],
                    'current_stats': {
                        'entities': entity_count,
                        'relationships': relationship_count,
                        'ratio': round(current_ratio, 2)
                    }
                }
            else:
                return {
                    'success': False,
                    'error': 'Neo4j connectivity test query failed',
                    'details': {'query_result': result}
                }
                
        except Exception as e:
            logger.error(f"❌ Neo4j connectivity test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'details': {'exception_type': type(e).__name__}
            }
    
    async def test_chunking_strategy(self) -> Dict[str, Any]:
        """Test that chunking produces balanced chunks (5-8 chunks, not mega chunks)"""
        try:
            from app.services.dynamic_chunk_sizing import DynamicChunkSizeCalculator
            
            logger.info("🔍 Testing chunking strategy with test document...")
            
            # Read test document
            if not self.test_document_path.exists():
                return {
                    'success': False,
                    'error': f'Test document not found: {self.test_document_path}',
                    'details': {'file_exists': False}
                }
            
            with open(self.test_document_path, 'r', encoding='utf-8') as f:
                test_content = f.read()
            
            logger.info(f"📄 Test document size: {len(test_content)} characters")
            
            # Test dynamic chunking
            chunk_calculator = DynamicChunkSizeCalculator()
            
            # Calculate chunk parameters
            chunk_params = chunk_calculator.calculate_optimal_chunk_size(
                content_length=len(test_content),
                target_chunks=7,  # Target 7 chunks (within 5-8 range)
                min_chunk_size=800,
                max_chunk_size=2000
            )
            
            logger.info(f"🔧 Calculated chunk parameters: {chunk_params}")
            
            # Simulate chunking (basic implementation)
            chunk_size = chunk_params['chunk_size']
            overlap = chunk_params['overlap']
            
            chunks = []
            start = 0
            chunk_id = 1
            
            while start < len(test_content):
                end = min(start + chunk_size, len(test_content))
                chunk_text = test_content[start:end]
                
                if chunk_text.strip():  # Only add non-empty chunks
                    chunks.append({
                        'id': f'chunk_{chunk_id}',
                        'text': chunk_text,
                        'size': len(chunk_text),
                        'start_pos': start,
                        'end_pos': end
                    })
                    chunk_id += 1
                
                # Move start position with overlap
                start = end - overlap
                if start >= len(test_content):
                    break
            
            chunk_count = len(chunks)
            avg_chunk_size = sum(c['size'] for c in chunks) / max(chunk_count, 1)
            min_chunk_size = min(c['size'] for c in chunks) if chunks else 0
            max_chunk_size = max(c['size'] for c in chunks) if chunks else 0
            
            # Validate chunking results
            success = 5 <= chunk_count <= 8  # Must be within 5-8 chunks
            mega_chunk_detected = max_chunk_size > 5000  # Mega chunk = >5KB
            
            logger.info(f"📊 Chunking results: {chunk_count} chunks, avg size: {avg_chunk_size:.0f}")
            
            if success and not mega_chunk_detected:
                logger.info("✅ Chunking strategy validation PASSED")
            else:
                logger.warning("⚠️ Chunking strategy validation FAILED")
            
            return {
                'success': success and not mega_chunk_detected,
                'chunk_count': chunk_count,
                'target_range': '5-8 chunks',
                'within_target': success,
                'mega_chunk_detected': mega_chunk_detected,
                'chunk_stats': {
                    'count': chunk_count,
                    'avg_size': round(avg_chunk_size),
                    'min_size': min_chunk_size,
                    'max_size': max_chunk_size,
                    'overlap': overlap
                },
                'chunk_previews': [
                    {
                        'id': c['id'],
                        'size': c['size'],
                        'preview': c['text'][:100] + "..." if len(c['text']) > 100 else c['text']
                    }
                    for c in chunks[:3]  # Show first 3 chunks
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Chunking strategy test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'details': {'exception_type': type(e).__name__}
            }
    
    async def test_entity_extraction(self) -> Dict[str, Any]:
        """Test entity extraction from test document chunk"""
        try:
            from app.services.llm_knowledge_extractor import LLMKnowledgeExtractor
            
            logger.info("🔍 Testing entity extraction...")
            
            # Use a sample chunk from test document
            sample_text = """
            DBS Bank is a leading financial services group in Southeast Asia with strong positions in consumer banking, 
            treasury and markets, asset management, and insurance. The bank leverages advanced technologies including 
            artificial intelligence, machine learning, and blockchain to enhance customer experience. The Chief Technology 
            Officer leads digital transformation initiatives across Singapore, Hong Kong, and other Asian markets.
            """
            
            llm_extractor = LLMKnowledgeExtractor()
            
            start_time = time.time()
            extraction_result = await llm_extractor.extract_knowledge(
                text=sample_text,
                context={
                    'document_id': 'test_doc',
                    'chunk_id': 'test_chunk_1',
                    'extraction_mode': 'comprehensive'
                }
            )
            extraction_time = time.time() - start_time
            
            entities = extraction_result.entities
            entity_count = len(entities)
            
            # Analyze extracted entities
            entity_types = {}
            entity_names = []
            for entity in entities:
                entity_type = entity.label
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
                entity_names.append(f"{entity.text} ({entity.label})")
            
            # Validation criteria
            min_expected_entities = 5  # Should extract at least 5 entities from this rich text
            max_expected_entities = 15  # But not too many (avoid over-extraction)
            
            success = min_expected_entities <= entity_count <= max_expected_entities
            
            logger.info(f"📊 Extracted {entity_count} entities in {extraction_time:.2f}s")
            
            if success:
                logger.info("✅ Entity extraction validation PASSED")
            else:
                logger.warning("⚠️ Entity extraction validation FAILED")
                
            return {
                'success': success,
                'entity_count': entity_count,
                'expected_range': f'{min_expected_entities}-{max_expected_entities}',
                'within_range': success,
                'extraction_time_ms': extraction_time * 1000,
                'entity_types_distribution': entity_types,
                'sample_entities': entity_names[:10],  # Show first 10
                'extraction_metadata': {
                    'confidence_score': extraction_result.confidence_score,
                    'model_used': extraction_result.llm_model_used
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Entity extraction test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'details': {'exception_type': type(e).__name__}
            }
    
    async def test_relationship_extraction(self) -> Dict[str, Any]:
        """Test relationship extraction from test document chunk"""
        try:
            from app.services.llm_knowledge_extractor import LLMKnowledgeExtractor
            
            logger.info("🔍 Testing relationship extraction...")
            
            # Use a relationship-rich sample
            sample_text = """
            DBS Bank operates in Singapore and Hong Kong, providing digital banking services to retail customers. 
            The Chief Technology Officer oversees artificial intelligence projects that enhance mobile banking platforms. 
            These AI systems process customer data to deliver personalized financial products through the mobile app.
            """
            
            llm_extractor = LLMKnowledgeExtractor()
            
            start_time = time.time()
            extraction_result = await llm_extractor.extract_knowledge(
                text=sample_text,
                context={
                    'document_id': 'test_doc',
                    'chunk_id': 'test_chunk_2', 
                    'extraction_mode': 'comprehensive'
                }
            )
            extraction_time = time.time() - start_time
            
            relationships = extraction_result.relationships
            relationship_count = len(relationships)
            
            # Analyze extracted relationships
            relationship_types = {}
            relationship_details = []
            for rel in relationships:
                rel_type = rel.relationship_type
                relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
                relationship_details.append({
                    'source': rel.source_entity,
                    'type': rel.relationship_type,
                    'target': rel.target_entity,
                    'confidence': rel.confidence
                })
            
            # Validation criteria
            min_expected_relationships = 3  # Should extract at least 3 relationships
            max_expected_relationships = 10  # But not excessive
            
            success = min_expected_relationships <= relationship_count <= max_expected_relationships
            
            logger.info(f"📊 Extracted {relationship_count} relationships in {extraction_time:.2f}s")
            
            if success:
                logger.info("✅ Relationship extraction validation PASSED")
            else:
                logger.warning("⚠️ Relationship extraction validation FAILED")
                
            return {
                'success': success,
                'relationship_count': relationship_count,
                'expected_range': f'{min_expected_relationships}-{max_expected_relationships}',
                'within_range': success,
                'extraction_time_ms': extraction_time * 1000,
                'relationship_types_distribution': relationship_types,
                'sample_relationships': relationship_details[:8],  # Show first 8
                'avg_confidence': sum(r.confidence for r in relationships) / max(relationship_count, 1)
            }
            
        except Exception as e:
            logger.error(f"❌ Relationship extraction test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'details': {'exception_type': type(e).__name__}
            }
    
    async def test_relationship_ratio(self) -> Dict[str, Any]:
        """Test that relationship ratio stays ≤4 per entity"""
        try:
            from app.services.neo4j_service import get_neo4j_service
            from app.services.knowledge_graph_service import GlobalRelationshipBudget
            
            logger.info("🔍 Testing relationship ratio enforcement...")
            
            neo4j_service = get_neo4j_service()
            if not neo4j_service.is_enabled():
                return {
                    'success': False,
                    'error': 'Neo4j not enabled - cannot test ratio',
                    'details': {'neo4j_enabled': False}
                }
            
            # Get current database statistics
            total_entities = neo4j_service.get_total_entity_count()
            total_relationships = neo4j_service.get_total_relationship_count()
            
            if total_entities == 0:
                logger.info("📊 No entities in database - ratio test not applicable")
                return {
                    'success': True,
                    'entities': 0,
                    'relationships': 0,
                    'ratio': 0.0,
                    'within_limit': True,
                    'note': 'No entities in database - test not applicable'
                }
            
            current_ratio = total_relationships / total_entities
            
            # Test relationship budget system
            global_budget = GlobalRelationshipBudget()
            max_allowed_ratio = global_budget.max_ratio  # Should be 4.0
            
            ratio_within_limit = current_ratio <= max_allowed_ratio
            budget_check_passed = not global_budget.check_ratio_limit(neo4j_service)
            
            logger.info(f"📊 Current ratio: {current_ratio:.2f} (limit: {max_allowed_ratio})")
            
            success = ratio_within_limit and budget_check_passed
            
            if success:
                logger.info("✅ Relationship ratio enforcement PASSED")
            else:
                logger.warning("⚠️ Relationship ratio enforcement FAILED")
            
            return {
                'success': success,
                'entities': total_entities,
                'relationships': total_relationships,
                'current_ratio': round(current_ratio, 2),
                'max_allowed_ratio': max_allowed_ratio,
                'within_limit': ratio_within_limit,
                'budget_system_active': True,
                'budget_check_passed': budget_check_passed,
                'ratio_status': 'PASS' if ratio_within_limit else 'FAIL'
            }
            
        except Exception as e:
            logger.error(f"❌ Relationship ratio test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'details': {'exception_type': type(e).__name__}
            }
    
    async def test_neo4j_storage(self) -> Dict[str, Any]:
        """Test Neo4j storage functionality"""
        try:
            from app.services.neo4j_service import get_neo4j_service
            from app.services.knowledge_graph_types import ExtractedEntity, ExtractedRelationship
            from app.document_handlers.base import ExtractedChunk
            from app.services.knowledge_graph_service import get_knowledge_graph_service
            
            logger.info("🔍 Testing Neo4j storage functionality...")
            
            neo4j_service = get_neo4j_service()
            if not neo4j_service.is_enabled():
                return {
                    'success': False,
                    'error': 'Neo4j not enabled - cannot test storage',
                    'details': {'neo4j_enabled': False}
                }
            
            # Record initial state
            initial_entities = neo4j_service.get_total_entity_count()
            initial_relationships = neo4j_service.get_total_relationship_count()
            
            # Create test entities
            test_entities = [
                ExtractedEntity(
                    text="Test Bank Pipeline",
                    label="ORGANIZATION", 
                    canonical_form="Test Bank Pipeline",
                    confidence=0.9,
                    properties={}
                ),
                ExtractedEntity(
                    text="AI Technology",
                    label="TECHNOLOGY",
                    canonical_form="AI Technology", 
                    confidence=0.8,
                    properties={}
                )
            ]
            
            # Create test relationship
            test_relationships = [
                ExtractedRelationship(
                    source_entity="Test Bank Pipeline",
                    target_entity="AI Technology",
                    relationship_type="USES",
                    confidence=0.85,
                    context="Test bank uses AI technology",
                    properties={}
                )
            ]
            
            # Create mock extraction result
            from app.services.knowledge_graph_types import GraphExtractionResult
            test_result = GraphExtractionResult(
                chunk_id="test_storage_chunk",
                entities=test_entities,
                relationships=test_relationships,
                processing_time_ms=100.0,
                source_metadata={},
                warnings=[]
            )
            
            # Test storage
            kg_service = get_knowledge_graph_service()
            storage_result = await kg_service.store_in_neo4j(test_result, document_id="test_storage_doc")
            
            # Verify storage success
            storage_success = storage_result.get('success', False)
            entities_stored = storage_result.get('entities_stored', 0)
            relationships_stored = storage_result.get('relationships_stored', 0)
            
            # Get final counts
            final_entities = neo4j_service.get_total_entity_count()
            final_relationships = neo4j_service.get_total_relationship_count()
            
            # Calculate changes
            entities_added = final_entities - initial_entities
            relationships_added = final_relationships - initial_relationships
            
            logger.info(f"📊 Storage test: +{entities_added} entities, +{relationships_added} relationships")
            
            success = storage_success and entities_added >= 0 and relationships_added >= 0
            
            if success:
                logger.info("✅ Neo4j storage validation PASSED")
            else:
                logger.warning("⚠️ Neo4j storage validation FAILED")
            
            return {
                'success': success,
                'storage_result': storage_result,
                'database_changes': {
                    'entities_before': initial_entities,
                    'entities_after': final_entities,
                    'entities_added': entities_added,
                    'relationships_before': initial_relationships,
                    'relationships_after': final_relationships,
                    'relationships_added': relationships_added
                },
                'test_data': {
                    'test_entities': len(test_entities),
                    'test_relationships': len(test_relationships),
                    'entities_stored': entities_stored,
                    'relationships_stored': relationships_stored
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Neo4j storage test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'details': {'exception_type': type(e).__name__}
            }
    
    async def test_end_to_end_pipeline(self) -> Dict[str, Any]:
        """Test complete end-to-end pipeline integration"""
        try:
            logger.info("🔍 Testing end-to-end pipeline integration...")
            
            # This test brings together all components
            
            # 1. Document processing (simulated)
            document_content = """
            Singapore FinTech Consortium partners with DBS Digital Bank to develop innovative blockchain solutions 
            for cross-border payments. The Chief Innovation Officer announced the strategic partnership will leverage 
            artificial intelligence and machine learning technologies to enhance customer experience in Southeast Asia.
            """
            
            # 2. Chunking 
            from app.services.dynamic_chunk_sizing import DynamicChunkSizeCalculator
            chunk_calculator = DynamicChunkSizeCalculator()
            
            # Simple chunking for test
            chunk_size = 200
            chunks = []
            start = 0
            chunk_id = 1
            
            while start < len(document_content):
                end = min(start + chunk_size, len(document_content))
                chunk_text = document_content[start:end].strip()
                
                if chunk_text:
                    from app.document_handlers.base import ExtractedChunk
                    chunk = ExtractedChunk(
                        chunk_id=f"e2e_chunk_{chunk_id}",
                        text=chunk_text,
                        metadata={"page": 1, "section": f"test_section_{chunk_id}"}
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                
                start = end
            
            logger.info(f"📄 Created {len(chunks)} chunks for end-to-end test")
            
            # 3. Knowledge extraction and storage
            from app.services.knowledge_graph_service import get_knowledge_graph_service
            kg_service = get_knowledge_graph_service()
            
            total_entities_extracted = 0
            total_relationships_extracted = 0
            total_entities_stored = 0
            total_relationships_stored = 0
            
            extraction_times = []
            storage_results = []
            
            # Process each chunk through the complete pipeline
            for i, chunk in enumerate(chunks):
                logger.info(f"🔄 Processing chunk {i+1}/{len(chunks)}")
                
                # Extract knowledge
                start_time = time.time()
                extraction_result = await kg_service.extract_from_chunk(chunk, document_id="e2e_test_doc")
                extraction_time = time.time() - start_time
                extraction_times.append(extraction_time)
                
                total_entities_extracted += len(extraction_result.entities)
                total_relationships_extracted += len(extraction_result.relationships)
                
                # Store in Neo4j
                storage_result = await kg_service.store_in_neo4j(extraction_result, document_id="e2e_test_doc")
                storage_results.append(storage_result)
                
                if storage_result.get('success'):
                    total_entities_stored += storage_result.get('entities_stored', 0)
                    total_relationships_stored += storage_result.get('relationships_stored', 0)
            
            # Calculate pipeline metrics
            total_extraction_time = sum(extraction_times)
            avg_extraction_time = total_extraction_time / max(len(chunks), 1)
            successful_storage_operations = sum(1 for r in storage_results if r.get('success'))
            
            # Success criteria
            pipeline_success = (
                total_entities_extracted > 0 and
                total_relationships_extracted > 0 and
                total_entities_stored > 0 and
                successful_storage_operations > 0
            )
            
            logger.info(f"📊 End-to-end results: {total_entities_stored} entities, {total_relationships_stored} relationships stored")
            
            if pipeline_success:
                logger.info("✅ End-to-end pipeline integration PASSED")
            else:
                logger.warning("⚠️ End-to-end pipeline integration FAILED")
            
            return {
                'success': pipeline_success,
                'pipeline_metrics': {
                    'chunks_processed': len(chunks),
                    'total_extraction_time_ms': total_extraction_time * 1000,
                    'avg_extraction_time_ms': avg_extraction_time * 1000,
                    'successful_storage_operations': successful_storage_operations
                },
                'extraction_results': {
                    'entities_extracted': total_entities_extracted,
                    'relationships_extracted': total_relationships_extracted,
                    'entities_stored': total_entities_stored,
                    'relationships_stored': total_relationships_stored
                },
                'storage_success_rate': successful_storage_operations / max(len(chunks), 1) * 100,
                'sample_chunks': [
                    {
                        'id': c.chunk_id,
                        'size': len(c.text),
                        'preview': c.text[:100] + "..." if len(c.text) > 100 else c.text
                    }
                    for c in chunks[:2]  # Show first 2 chunks
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ End-to-end pipeline test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'details': {'exception_type': type(e).__name__}
            }
    
    def generate_final_report(self):
        """Generate comprehensive validation report"""
        logger.info("\n" + "=" * 70)
        logger.info("📋 FINAL KNOWLEDGE GRAPH PIPELINE VALIDATION REPORT")
        logger.info("=" * 70)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result and result.get('success', False))
        failed_tests = total_tests - passed_tests
        
        logger.info(f"📊 TEST SUMMARY: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        
        # Detailed results
        for test_name, result in self.test_results.items():
            if result is None:
                status = "❓ NOT RUN"
            elif result.get('success', False):
                status = "✅ PASSED"
            else:
                status = "❌ FAILED"
            
            logger.info(f"{status} {test_name.replace('_', ' ').title()}")
            
            if result and not result.get('success', False):
                error = result.get('error', 'Unknown error')
                logger.info(f"    Error: {error}")
        
        # Overall assessment
        if passed_tests == total_tests:
            logger.info("\n🎉 ALL TESTS PASSED - Knowledge Graph Pipeline is working correctly!")
            logger.info("✅ LLM connectivity: Fixed (host.docker.internal:11434)")
            logger.info("✅ Neo4j connectivity: Fixed (neo4j:7687)")
            logger.info("✅ Chunking strategy: Produces balanced chunks (5-8)")
            logger.info("✅ Entity extraction: Working properly")
            logger.info("✅ Relationship extraction: Working properly") 
            logger.info("✅ Relationship ratio: Enforced (≤4 per entity)")
            logger.info("✅ Neo4j storage: Working properly")
            logger.info("✅ End-to-end pipeline: Fully integrated")
        elif passed_tests >= total_tests * 0.75:
            logger.warning(f"\n⚠️ MOSTLY WORKING - {failed_tests} issues need attention:")
            for test_name, result in self.test_results.items():
                if result and not result.get('success', False):
                    logger.warning(f"   ❌ {test_name.replace('_', ' ').title()}: {result.get('error')}")
        else:
            logger.error(f"\n🚨 MAJOR ISSUES - {failed_tests} critical failures detected!")
            logger.error("Pipeline requires significant fixes before production use.")
        
        logger.info("\n" + "=" * 70)

async def main():
    """Main test execution"""
    validator = KnowledgeGraphPipelineValidator()
    
    try:
        results = await validator.run_all_tests()
        
        # Save results to file
        results_file = Path(__file__).parent / f"kg_pipeline_validation_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📄 Detailed results saved to: {results_file}")
        
        # Return summary
        passed = sum(1 for r in results.values() if r and r.get('success', False))
        total = len(results)
        
        return {
            'overall_success': passed == total,
            'tests_passed': passed,
            'total_tests': total,
            'success_rate': passed / total * 100,
            'results_file': str(results_file)
        }
        
    except Exception as e:
        logger.error(f"❌ Validation execution failed: {e}")
        logger.exception("Full exception details:")
        return {
            'overall_success': False,
            'error': str(e)
        }

if __name__ == "__main__":
    # Run the complete validation
    import asyncio
    result = asyncio.run(main())
    
    if result.get('overall_success'):
        print("\n🎉 Knowledge Graph Pipeline Validation: SUCCESS")
        sys.exit(0)
    else:
        print("\n❌ Knowledge Graph Pipeline Validation: FAILED")
        sys.exit(1)