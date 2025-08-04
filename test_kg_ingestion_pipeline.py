#!/usr/bin/env python3
"""
Comprehensive Test: Knowledge Graph Ingestion Pipeline

Tests the complete end-to-end knowledge graph ingestion process:
1. Document processing and chunking
2. LLM entity and relationship extraction
3. Two-step extraction prompt usage
4. Neo4j storage with budget controls
5. Anti-silo analysis execution
6. Relationship ratio monitoring
7. Configuration loading verification

This test will reveal what's working and what's broken in the ingestion system.
"""

import sys
import os
sys.path.append('/Users/kianwoonwong/Downloads/jarvis')

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

# Core imports
from app.core.knowledge_graph_settings_cache import get_knowledge_graph_settings
from app.services.knowledge_graph_service import get_knowledge_graph_service, GlobalRelationshipBudget
from app.services.neo4j_service import get_neo4j_service
from app.document_handlers.base import ExtractedChunk

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KnowledgeGraphIngestionTester:
    """Comprehensive tester for knowledge graph ingestion pipeline"""
    
    def __init__(self):
        self.test_results = {
            'config_loaded': False,
            'neo4j_available': False,
            'extraction_working': False,
            'storage_working': False,
            'budgets_enforced': False,
            'anti_silo_working': False,
            'errors': [],
            'warnings': []
        }
        
    async def run_complete_test(self):
        """Run complete end-to-end ingestion test"""
        logger.info("🚀 Starting Comprehensive Knowledge Graph Ingestion Test")
        logger.info("=" * 70)
        
        try:
            # Test 1: Configuration Loading
            await self.test_configuration_loading()
            
            # Test 2: Neo4j Connectivity
            await self.test_neo4j_connectivity()
            
            # Test 3: Document Processing & Extraction
            await self.test_document_extraction()
            
            # Test 4: Neo4j Storage with Budget Controls
            await self.test_neo4j_storage_with_budgets()
            
            # Test 5: Anti-silo Analysis
            await self.test_anti_silo_analysis()
            
            # Test 6: Relationship Ratio Monitoring
            await self.test_relationship_ratio_monitoring()
            
            # Test 7: Two-step Extraction Prompts
            await self.test_two_step_extraction_prompts()
            
            # Generate comprehensive report
            self.generate_test_report()
            
        except Exception as e:
            logger.error(f"❌ Critical test failure: {e}")
            self.test_results['errors'].append(f"Critical failure: {str(e)}")
            raise
    
    async def test_configuration_loading(self):
        """Test configuration loading and parsing"""
        logger.info("\n🔧 Testing Configuration Loading...")
        
        try:
            # Load KG settings
            kg_settings = get_knowledge_graph_settings()
            logger.info(f"✅ Knowledge graph settings loaded")
            
            # Check critical settings
            extraction_config = kg_settings.get('extraction', {})
            logger.info(f"📊 Extraction mode: {kg_settings.get('mode', 'NOT FOUND')}")
            logger.info(f"📊 Max entities per chunk: {kg_settings.get('max_entities_per_chunk', 'NOT FOUND')}")
            logger.info(f"📊 Max relationships per chunk: {kg_settings.get('max_relationships_per_chunk', 'NOT FOUND')}")
            logger.info(f"📊 Min entity confidence: {extraction_config.get('min_entity_confidence', 'NOT FOUND')}")
            logger.info(f"📊 Min relationship confidence: {extraction_config.get('min_relationship_confidence', 'NOT FOUND')}")
            logger.info(f"📊 Anti-silo enabled: {extraction_config.get('enable_anti_silo', 'NOT FOUND')}")
            
            # Check prompt templates - check for extraction_prompt first, then prompts dict
            extraction_prompt = kg_settings.get('extraction_prompt', '')
            prompts = kg_settings.get('prompts', {})
            entity_prompt = prompts.get('entity_extraction_template', '')
            relationship_prompt = prompts.get('relationship_extraction_template', '')
            
            logger.info(f"📝 Extraction prompt length: {len(extraction_prompt)} chars")
            logger.info(f"📝 Entity prompt length: {len(entity_prompt)} chars")
            logger.info(f"📝 Relationship prompt length: {len(relationship_prompt)} chars")
            
            # Check if we have proper extraction prompts
            has_extraction_prompt = len(extraction_prompt) > 100
            has_two_step_prompts = len(entity_prompt) > 100 and len(relationship_prompt) > 100
            
            if has_extraction_prompt or has_two_step_prompts:
                logger.info("✅ Extraction prompts are configured")
            else:
                logger.warning("⚠️  Extraction prompts may be missing or incomplete")
                self.test_results['warnings'].append("Extraction prompts incomplete")
            
            self.test_results['config_loaded'] = True
            
        except Exception as e:
            logger.error(f"❌ Configuration loading failed: {e}")
            self.test_results['errors'].append(f"Config loading: {str(e)}")
    
    async def test_neo4j_connectivity(self):
        """Test Neo4j connectivity and basic operations"""
        logger.info("\n🗄️  Testing Neo4j Connectivity...")
        
        try:
            neo4j_service = get_neo4j_service()
            
            if not neo4j_service.is_enabled():
                logger.error("❌ Neo4j service is not enabled")
                self.test_results['errors'].append("Neo4j not enabled")
                return
            
            # Test basic queries
            entity_count = neo4j_service.get_total_entity_count()
            relationship_count = neo4j_service.get_total_relationship_count()
            
            logger.info(f"✅ Neo4j connected successfully")
            logger.info(f"📊 Current entities: {entity_count}")
            logger.info(f"📊 Current relationships: {relationship_count}")
            
            # Calculate current ratio
            if entity_count > 0:
                current_ratio = relationship_count / entity_count
                logger.info(f"📊 Current ratio: {current_ratio:.2f} relationships per entity")
            else:
                logger.info("📊 No entities in database")
            
            self.test_results['neo4j_available'] = True
            
        except Exception as e:
            logger.error(f"❌ Neo4j connectivity test failed: {e}")
            self.test_results['errors'].append(f"Neo4j connectivity: {str(e)}")
    
    async def test_document_extraction(self):
        """Test LLM-based entity and relationship extraction"""
        logger.info("\n🧠 Testing Document Extraction...")
        
        try:
            # Create test document chunk
            test_text = """
            DBS Bank is evaluating OceanBase database technology for their core banking systems.
            The bank is considering migrating from their current PostgreSQL infrastructure to improve 
            performance and scalability. This digital transformation initiative is part of DBS's 
            broader strategy to modernize their technology stack in Singapore and across Asia.
            
            Ant Group, the parent company of Alipay, has extensive experience with OceanBase in 
            high-volume financial transactions. Their expertise could be valuable for DBS's 
            evaluation process.
            """
            
            test_chunk = ExtractedChunk(
                chunk_id="test_chunk_001",
                text=test_text,
                metadata={
                    'source': 'test_document.txt',
                    'page_number': 1,
                    'chunk_index': 0
                }
            )
            
            # Test extraction
            kg_service = get_knowledge_graph_service()
            extraction_result = await kg_service.extract_from_chunk(test_chunk, "test_doc_001")
            
            logger.info(f"✅ Extraction completed")
            logger.info(f"📊 Entities extracted: {len(extraction_result.entities)}")
            logger.info(f"📊 Relationships extracted: {len(extraction_result.relationships)}")
            logger.info(f"📊 Processing time: {extraction_result.processing_time_ms:.1f}ms")
            
            # Log entities
            if extraction_result.entities:
                logger.info("🏷️  Extracted Entities:")
                for entity in extraction_result.entities[:10]:  # Show first 10
                    logger.info(f"   - {entity.canonical_form} ({entity.label}) [conf: {entity.confidence:.2f}]")
            
            # Log relationships
            if extraction_result.relationships:
                logger.info("🔗 Extracted Relationships:")
                for rel in extraction_result.relationships[:10]:  # Show first 10
                    logger.info(f"   - {rel.source_entity} --[{rel.relationship_type}]--> {rel.target_entity} [conf: {rel.confidence:.2f}]")
            
            # Check warnings
            if extraction_result.warnings:
                logger.warning("⚠️  Extraction warnings:")
                for warning in extraction_result.warnings:
                    logger.warning(f"   - {warning}")
                    self.test_results['warnings'].append(f"Extraction: {warning}")
            
            if len(extraction_result.entities) > 0:
                self.test_results['extraction_working'] = True
            else:
                logger.error("❌ No entities extracted from test document")
                self.test_results['errors'].append("No entities extracted")
            
            return extraction_result
            
        except Exception as e:
            logger.error(f"❌ Document extraction test failed: {e}")
            self.test_results['errors'].append(f"Document extraction: {str(e)}")
            return None
    
    async def test_neo4j_storage_with_budgets(self):
        """Test Neo4j storage with budget enforcement"""
        logger.info("\n💾 Testing Neo4j Storage with Budget Controls...")
        
        try:
            # First, run extraction to get results
            extraction_result = await self.test_document_extraction()
            if not extraction_result:
                logger.error("❌ Cannot test storage without extraction results")
                return
            
            # Test budget initialization
            global_budget = GlobalRelationshipBudget()
            neo4j_service = get_neo4j_service()
            
            logger.info(f"💰 Budget limits:")
            logger.info(f"   - Global cap: {global_budget.max_global}")
            logger.info(f"   - Per session: {global_budget.max_per_session}")
            logger.info(f"   - Max ratio: {global_budget.max_ratio}")
            
            # Get pre-storage counts
            pre_entities = neo4j_service.get_total_entity_count()
            pre_relationships = neo4j_service.get_total_relationship_count()
            
            logger.info(f"📊 Pre-storage state: {pre_entities} entities, {pre_relationships} relationships")
            
            # Test storage
            kg_service = get_knowledge_graph_service()
            storage_result = await kg_service.store_in_neo4j(extraction_result, "test_doc_001")
            
            logger.info(f"✅ Storage completed")
            logger.info(f"📊 Result: {storage_result}")
            
            # Get post-storage counts
            post_entities = neo4j_service.get_total_entity_count()
            post_relationships = neo4j_service.get_total_relationship_count()
            
            entities_added = post_entities - pre_entities
            relationships_added = post_relationships - pre_relationships
            
            logger.info(f"📊 Post-storage state: {post_entities} entities (+{entities_added}), {post_relationships} relationships (+{relationships_added})")
            
            # Check budget enforcement
            if relationships_added <= global_budget.max_per_session:
                logger.info(f"✅ Budget enforced: {relationships_added} ≤ {global_budget.max_per_session} per session limit")
                self.test_results['budgets_enforced'] = True
            else:
                logger.warning(f"⚠️  Budget may not be enforced: {relationships_added} > {global_budget.max_per_session}")
                self.test_results['warnings'].append(f"Budget enforcement: {relationships_added} > {global_budget.max_per_session}")
            
            # Check ratio
            if post_entities > 0:
                new_ratio = post_relationships / post_entities
                logger.info(f"📊 New ratio: {new_ratio:.2f} relationships per entity")
                
                if new_ratio <= global_budget.max_ratio:
                    logger.info(f"✅ Ratio within limits: {new_ratio:.2f} ≤ {global_budget.max_ratio}")
                else:
                    logger.warning(f"⚠️  Ratio exceeds limit: {new_ratio:.2f} > {global_budget.max_ratio}")
                    self.test_results['warnings'].append(f"Ratio limit: {new_ratio:.2f} > {global_budget.max_ratio}")
            
            if storage_result.get('success'):
                self.test_results['storage_working'] = True
            else:
                logger.error(f"❌ Storage failed: {storage_result.get('error', 'Unknown error')}")
                self.test_results['errors'].append(f"Storage: {storage_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"❌ Neo4j storage test failed: {e}")
            self.test_results['errors'].append(f"Neo4j storage: {str(e)}")
    
    async def test_anti_silo_analysis(self):
        """Test anti-silo analysis functionality"""
        logger.info("\n🔗 Testing Anti-silo Analysis...")
        
        try:
            kg_service = get_knowledge_graph_service()
            neo4j_service = get_neo4j_service()
            
            # Get isolated nodes before anti-silo
            pre_isolated = neo4j_service.get_truly_isolated_nodes()
            logger.info(f"📊 Isolated nodes before anti-silo: {len(pre_isolated)}")
            
            if pre_isolated:
                isolated_names = [node.get('name', 'Unknown') for node in pre_isolated[:5]]
                logger.info(f"🔍 Sample isolated nodes: {isolated_names}")
            
            # Run limited anti-silo analysis
            anti_silo_result = await kg_service.run_limited_anti_silo_analysis()
            
            logger.info(f"✅ Anti-silo analysis completed")
            logger.info(f"📊 Result: {anti_silo_result}")
            
            if anti_silo_result.get('success'):
                logger.info(f"✅ Anti-silo analysis successful")
                logger.info(f"   - Initial silos: {anti_silo_result.get('initial_silo_count', 0)}")
                logger.info(f"   - Final silos: {anti_silo_result.get('final_silo_count', 0)}")
                logger.info(f"   - Connections made: {anti_silo_result.get('connections_made', 0)}")
                logger.info(f"   - Nodes removed: {anti_silo_result.get('nodes_removed', 0)}")
                
                self.test_results['anti_silo_working'] = True
            else:
                logger.error(f"❌ Anti-silo analysis failed: {anti_silo_result.get('error', 'Unknown error')}")
                self.test_results['errors'].append(f"Anti-silo: {anti_silo_result.get('error', 'Unknown error')}")
            
            # Get isolated nodes after anti-silo
            post_isolated = neo4j_service.get_truly_isolated_nodes()
            logger.info(f"📊 Isolated nodes after anti-silo: {len(post_isolated)}")
            
            reduction = len(pre_isolated) - len(post_isolated)
            if reduction > 0:
                logger.info(f"✅ Anti-silo reduced isolation by {reduction} nodes")
            elif reduction == 0:
                logger.info("📊 No change in isolated nodes (expected if none existed)")
            else:
                logger.warning(f"⚠️  Anti-silo increased isolated nodes by {abs(reduction)}")
                
        except Exception as e:
            logger.error(f"❌ Anti-silo analysis test failed: {e}")
            self.test_results['errors'].append(f"Anti-silo analysis: {str(e)}")
    
    async def test_relationship_ratio_monitoring(self):
        """Test relationship ratio monitoring and enforcement"""
        logger.info("\n📊 Testing Relationship Ratio Monitoring...")
        
        try:
            neo4j_service = get_neo4j_service()
            global_budget = GlobalRelationshipBudget()
            
            # Get current counts
            entity_count = neo4j_service.get_total_entity_count()
            relationship_count = neo4j_service.get_total_relationship_count()
            
            logger.info(f"📊 Current state:")
            logger.info(f"   - Entities: {entity_count}")
            logger.info(f"   - Relationships: {relationship_count}")
            
            if entity_count > 0:
                current_ratio = relationship_count / entity_count
                logger.info(f"   - Ratio: {current_ratio:.2f} relationships per entity")
                
                # Test ratio checking
                ratio_ok = not global_budget.check_ratio_limit(neo4j_service)
                
                if ratio_ok:
                    logger.info(f"✅ Ratio within acceptable limits (≤ {global_budget.max_ratio})")
                else:
                    logger.warning(f"⚠️  Ratio exceeds limits (> {global_budget.max_ratio})")
                    self.test_results['warnings'].append(f"Ratio monitoring: {current_ratio:.2f} > {global_budget.max_ratio}")
                
                # Test budget capacity
                test_request = 5
                allowed = global_budget.can_add_relationships(neo4j_service, test_request)
                logger.info(f"📊 Budget test: Requested {test_request}, allowed {allowed}")
                
                if allowed < test_request:
                    logger.info("✅ Budget enforcement is active")
                else:
                    logger.info("📊 Budget has capacity for new relationships")
            else:
                logger.info("📊 No entities in database - ratio monitoring not applicable")
            
        except Exception as e:
            logger.error(f"❌ Relationship ratio monitoring test failed: {e}")
            self.test_results['errors'].append(f"Ratio monitoring: {str(e)}")
    
    async def test_two_step_extraction_prompts(self):
        """Test if extraction prompts are being used"""
        logger.info("\n📝 Testing Extraction Prompts...")
        
        try:
            # Check if prompts are configured
            kg_settings = get_knowledge_graph_settings()
            
            # Check for main extraction prompt
            extraction_prompt = kg_settings.get('extraction_prompt', '')
            
            # Check for separate step prompts
            prompts = kg_settings.get('prompts', {})
            entity_template = prompts.get('entity_extraction_template', '')
            relationship_template = prompts.get('relationship_extraction_template', '')
            
            logger.info(f"📊 Prompt configuration:")
            logger.info(f"   - Main extraction prompt: {len(extraction_prompt)} characters")
            logger.info(f"   - Entity template: {len(entity_template)} characters")
            logger.info(f"   - Relationship template: {len(relationship_template)} characters")
            
            # Check which type of prompts we have
            has_main_prompt = len(extraction_prompt) > 100
            has_two_step_prompts = len(entity_template) > 100 and len(relationship_template) > 100
            
            if has_main_prompt:
                logger.info("✅ Main extraction prompt is configured")
                # Check for key indicators
                indicators = ['entities', 'relationships', 'extract', 'identify', 'classify']
                score = sum(1 for indicator in indicators if indicator.lower() in extraction_prompt.lower())
                logger.info(f"📊 Main prompt relevance: {score}/{len(indicators)}")
            
            if has_two_step_prompts:
                logger.info("✅ Two-step extraction prompts are configured")
                # Check for key indicators of two-step prompts
                entity_indicators = ['entities', 'extract', 'identify', 'classify']
                relationship_indicators = ['relationships', 'connections', 'links', 'between']
                
                entity_score = sum(1 for indicator in entity_indicators if indicator.lower() in entity_template.lower())
                relationship_score = sum(1 for indicator in relationship_indicators if indicator.lower() in relationship_template.lower())
                
                logger.info(f"📊 Two-step template analysis:")
                logger.info(f"   - Entity template relevance: {entity_score}/{len(entity_indicators)}")
                logger.info(f"   - Relationship template relevance: {relationship_score}/{len(relationship_indicators)}")
            
            if not has_main_prompt and not has_two_step_prompts:
                logger.warning("⚠️  No extraction prompts found")
                self.test_results['warnings'].append("No extraction prompts found")
                
            # Test if extraction actually uses prompts
            logger.info("📊 Note: Actual prompt usage verification requires LLM extraction monitoring")
            
        except Exception as e:
            logger.error(f"❌ Extraction prompt test failed: {e}")
            self.test_results['errors'].append(f"Extraction prompts: {str(e)}")
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("\n" + "=" * 70)
        logger.info("🎯 KNOWLEDGE GRAPH INGESTION TEST REPORT")
        logger.info("=" * 70)
        
        # Summary
        total_tests = len([k for k in self.test_results.keys() if k not in ['errors', 'warnings']])
        passed_tests = len([v for k, v in self.test_results.items() if k not in ['errors', 'warnings'] and v])
        
        logger.info(f"📊 OVERALL SUMMARY:")
        logger.info(f"   - Tests passed: {passed_tests}/{total_tests}")
        logger.info(f"   - Success rate: {(passed_tests/total_tests)*100:.1f}%")
        logger.info(f"   - Errors: {len(self.test_results['errors'])}")
        logger.info(f"   - Warnings: {len(self.test_results['warnings'])}")
        
        # Detailed results
        logger.info(f"\n📋 DETAILED RESULTS:")
        test_names = {
            'config_loaded': 'Configuration Loading',
            'neo4j_available': 'Neo4j Connectivity',
            'extraction_working': 'Document Extraction',
            'storage_working': 'Neo4j Storage',
            'budgets_enforced': 'Budget Enforcement',
            'anti_silo_working': 'Anti-silo Analysis'
        }
        
        for key, name in test_names.items():
            status = "✅ PASS" if self.test_results[key] else "❌ FAIL"
            logger.info(f"   - {name}: {status}")
        
        # Errors
        if self.test_results['errors']:
            logger.info(f"\n❌ ERRORS ENCOUNTERED:")
            for error in self.test_results['errors']:
                logger.info(f"   - {error}")
        
        # Warnings
        if self.test_results['warnings']:
            logger.info(f"\n⚠️  WARNINGS:")
            for warning in self.test_results['warnings']:
                logger.info(f"   - {warning}")
        
        # Recommendations
        logger.info(f"\n💡 RECOMMENDATIONS:")
        
        if not self.test_results['config_loaded']:
            logger.info("   - Fix knowledge graph configuration loading")
        
        if not self.test_results['neo4j_available']:
            logger.info("   - Ensure Neo4j is running and accessible")
        
        if not self.test_results['extraction_working']:
            logger.info("   - Debug LLM extraction process")
        
        if not self.test_results['storage_working']:
            logger.info("   - Fix Neo4j storage functionality")
        
        if not self.test_results['budgets_enforced']:
            logger.info("   - Review and fix budget enforcement logic")
        
        if not self.test_results['anti_silo_working']:
            logger.info("   - Debug anti-silo analysis process")
        
        if len(self.test_results['warnings']) > 0:
            logger.info("   - Address configuration warnings")
        
        # Final assessment
        if passed_tests == total_tests and len(self.test_results['errors']) == 0:
            logger.info(f"\n🎉 ASSESSMENT: Knowledge graph ingestion pipeline is FULLY FUNCTIONAL")
        elif passed_tests >= total_tests * 0.8:
            logger.info(f"\n⚠️  ASSESSMENT: Knowledge graph ingestion is MOSTLY WORKING with minor issues")
        elif passed_tests >= total_tests * 0.5:
            logger.info(f"\n🚨 ASSESSMENT: Knowledge graph ingestion has SIGNIFICANT ISSUES")
        else:
            logger.info(f"\n💥 ASSESSMENT: Knowledge graph ingestion is CRITICALLY BROKEN")

async def main():
    """Run comprehensive knowledge graph ingestion test"""
    tester = KnowledgeGraphIngestionTester()
    await tester.run_complete_test()

if __name__ == "__main__":
    asyncio.run(main())