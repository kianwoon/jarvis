#!/usr/bin/env python3
"""
Comprehensive Test Script for Enhanced Radiating System

This script verifies that the enhanced radiating system properly handles
the user's original query about essential AI implementation technologies,
demonstrating deep multi-hop exploration with 30+ entities and 50+ relationships.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys
import os

# Add the app directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.radiating.radiating_service import RadiatingService
from app.langchain.radiating_agent_system import RadiatingAgent, RadiatingAgentPool
from app.services.radiating.models.radiating_context import RadiatingContext, TraversalStrategy
from app.services.radiating.models.radiating_entity import RadiatingEntity
from app.services.radiating.extraction.universal_entity_extractor import UniversalEntityExtractor
from app.core.redis_client import get_redis_client
from app.core.db import get_db_session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Disable some verbose loggers
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)


class RadiatingSystemTester:
    """Test harness for the enhanced radiating system"""
    
    def __init__(self):
        self.service = RadiatingService()
        self.agent_pool = RadiatingAgentPool()
        self.entity_extractor = UniversalEntityExtractor()
        self.test_results = {
            'before_enhancement': {},
            'after_enhancement': {},
            'comparison': {}
        }
        
    async def test_basic_extraction(self, query: str) -> Dict[str, Any]:
        """Test basic entity extraction without enhancements"""
        logger.info("=" * 80)
        logger.info("TESTING BASIC EXTRACTION (Before Enhancement)")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        # Extract entities using basic method
        entities = await self.entity_extractor.extract_entities(query)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            'query': query,
            'entities_found': len(entities),
            'entity_types': {},
            'processing_time': processing_time,
            'sample_entities': []
        }
        
        # Count entity types
        for entity in entities:
            entity_type = entity.entity_type
            result['entity_types'][entity_type] = result['entity_types'].get(entity_type, 0) + 1
            
            # Add sample entities (first 5)
            if len(result['sample_entities']) < 5:
                result['sample_entities'].append({
                    'text': entity.text,
                    'type': entity.entity_type,
                    'confidence': entity.confidence
                })
        
        logger.info(f"✓ Found {len(entities)} entities")
        logger.info(f"  Entity types: {result['entity_types']}")
        logger.info(f"  Processing time: {processing_time:.2f}s")
        
        return result
    
    async def test_enhanced_extraction(self, query: str) -> Dict[str, Any]:
        """Test enhanced entity extraction with web search"""
        logger.info("=" * 80)
        logger.info("TESTING ENHANCED EXTRACTION (After Enhancement)")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        # Extract entities using enhanced method with web search
        entities = await self.entity_extractor.extract_entities_with_web_search(
            query,
            force_web_search=True
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            'query': query,
            'entities_found': len(entities),
            'entity_types': {},
            'processing_time': processing_time,
            'sample_entities': [],
            'web_search_used': True
        }
        
        # Count entity types and collect samples
        for entity in entities:
            entity_type = entity.entity_type
            result['entity_types'][entity_type] = result['entity_types'].get(entity_type, 0) + 1
            
            # Add sample entities (first 10 for enhanced)
            if len(result['sample_entities']) < 10:
                result['sample_entities'].append({
                    'text': entity.text,
                    'type': entity.entity_type,
                    'confidence': entity.confidence,
                    'metadata': entity.metadata.get('source', 'unknown')
                })
        
        logger.info(f"✓ Found {len(entities)} entities with web search enhancement")
        logger.info(f"  Entity types: {result['entity_types']}")
        logger.info(f"  Processing time: {processing_time:.2f}s")
        
        return result
    
    async def test_radiating_coverage(self, query: str, use_web_search: bool = False) -> Dict[str, Any]:
        """Test full radiating coverage with relationship discovery"""
        logger.info("=" * 80)
        logger.info(f"TESTING RADIATING COVERAGE ({'Enhanced' if use_web_search else 'Basic'})")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        # Execute radiating query
        filters = {'use_web_search': use_web_search} if use_web_search else None
        
        result = await self.service.execute_radiating_query(
            query=query,
            max_depth=3,
            strategy='adaptive',
            filters=filters,
            include_coverage=True
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        coverage_result = {
            'query': query,
            'status': result.get('status'),
            'processing_time': processing_time,
            'web_search_used': use_web_search,
            'coverage': result.get('coverage', {}),
            'entities': result.get('entities', []),
            'relationships': result.get('relationships', []),
            'metrics': {
                'total_entities': len(result.get('entities', [])),
                'total_relationships': len(result.get('relationships', [])),
                'max_depth_reached': result.get('coverage', {}).get('max_depth_reached', 0),
                'coverage_percentage': result.get('coverage', {}).get('coverage_percentage', 0),
                'explored_paths': result.get('coverage', {}).get('explored_paths', 0),
                'entity_types': result.get('coverage', {}).get('entity_types', {}),
                'relationship_types': result.get('coverage', {}).get('relationship_types', {})
            }
        }
        
        logger.info(f"✓ Radiating coverage completed")
        logger.info(f"  Total entities: {coverage_result['metrics']['total_entities']}")
        logger.info(f"  Total relationships: {coverage_result['metrics']['total_relationships']}")
        logger.info(f"  Max depth reached: {coverage_result['metrics']['max_depth_reached']}")
        logger.info(f"  Coverage percentage: {coverage_result['metrics']['coverage_percentage']}%")
        logger.info(f"  Processing time: {processing_time:.2f}s")
        
        return coverage_result
    
    async def test_agent_processing(self, query: str) -> Dict[str, Any]:
        """Test RadiatingAgent processing with full pipeline"""
        logger.info("=" * 80)
        logger.info("TESTING RADIATING AGENT PROCESSING")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        # Get an agent from the pool
        agent = await self.agent_pool.get_agent(
            radiating_config={'expansion_strategy': 'adaptive'}
        )
        
        # Process query with radiation
        context = {
            'max_depth': 3,
            'strategy': 'adaptive',
            'relevance_threshold': 0.1
        }
        
        entities_discovered = 0
        relationships_found = 0
        processing_messages = []
        
        try:
            async for chunk in agent.process_with_radiation(query, context, stream=True):
                if chunk['type'] == 'status':
                    processing_messages.append(chunk['message'])
                    logger.info(f"  Status: {chunk['message']}")
                elif chunk['type'] == 'metadata':
                    entities_discovered = chunk.get('entities_discovered', 0)
                    relationships_found = chunk.get('relationships_found', 0)
                elif chunk['type'] == 'error':
                    logger.error(f"  Error: {chunk['message']}")
        
        finally:
            await self.agent_pool.release_agent(agent.agent_id)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            'query': query,
            'entities_discovered': entities_discovered,
            'relationships_found': relationships_found,
            'processing_time': processing_time,
            'processing_messages': processing_messages,
            'agent_metrics': agent.metrics
        }
        
        logger.info(f"✓ Agent processing completed")
        logger.info(f"  Entities discovered: {entities_discovered}")
        logger.info(f"  Relationships found: {relationships_found}")
        logger.info(f"  Processing time: {processing_time:.2f}s")
        
        return result
    
    def print_detailed_comparison(self):
        """Print detailed comparison between before and after enhancement"""
        logger.info("\n")
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE TEST RESULTS COMPARISON")
        logger.info("=" * 80)
        
        # Entity Extraction Comparison
        logger.info("\n📊 ENTITY EXTRACTION COMPARISON:")
        logger.info("-" * 40)
        
        basic = self.test_results.get('basic_extraction', {})
        enhanced = self.test_results.get('enhanced_extraction', {})
        
        logger.info(f"Basic Extraction:")
        logger.info(f"  • Entities found: {basic.get('entities_found', 0)}")
        logger.info(f"  • Processing time: {basic.get('processing_time', 0):.2f}s")
        logger.info(f"  • Entity types: {basic.get('entity_types', {})}")
        
        logger.info(f"\nEnhanced Extraction (with Web Search):")
        logger.info(f"  • Entities found: {enhanced.get('entities_found', 0)}")
        logger.info(f"  • Processing time: {enhanced.get('processing_time', 0):.2f}s")
        logger.info(f"  • Entity types: {enhanced.get('entity_types', {})}")
        
        # Calculate improvement
        entity_improvement = (
            (enhanced.get('entities_found', 0) - basic.get('entities_found', 0)) / 
            max(basic.get('entities_found', 1), 1) * 100
        )
        logger.info(f"\n🚀 Improvement: {entity_improvement:.1f}% more entities discovered")
        
        # Radiating Coverage Comparison
        logger.info("\n📈 RADIATING COVERAGE COMPARISON:")
        logger.info("-" * 40)
        
        basic_coverage = self.test_results.get('basic_coverage', {})
        enhanced_coverage = self.test_results.get('enhanced_coverage', {})
        
        logger.info(f"Basic Radiating Coverage:")
        logger.info(f"  • Total entities: {basic_coverage.get('metrics', {}).get('total_entities', 0)}")
        logger.info(f"  • Total relationships: {basic_coverage.get('metrics', {}).get('total_relationships', 0)}")
        logger.info(f"  • Max depth: {basic_coverage.get('metrics', {}).get('max_depth_reached', 0)}")
        logger.info(f"  • Coverage: {basic_coverage.get('metrics', {}).get('coverage_percentage', 0)}%")
        
        logger.info(f"\nEnhanced Radiating Coverage:")
        logger.info(f"  • Total entities: {enhanced_coverage.get('metrics', {}).get('total_entities', 0)}")
        logger.info(f"  • Total relationships: {enhanced_coverage.get('metrics', {}).get('total_relationships', 0)}")
        logger.info(f"  • Max depth: {enhanced_coverage.get('metrics', {}).get('max_depth_reached', 0)}")
        logger.info(f"  • Coverage: {enhanced_coverage.get('metrics', {}).get('coverage_percentage', 0)}%")
        
        # Sample Entities
        logger.info("\n🔍 SAMPLE ENTITIES DISCOVERED:")
        logger.info("-" * 40)
        
        if enhanced.get('sample_entities'):
            logger.info("Enhanced extraction found modern technologies:")
            for i, entity in enumerate(enhanced['sample_entities'][:10], 1):
                logger.info(f"  {i}. {entity['text']} ({entity['type']}) - confidence: {entity['confidence']:.2f}")
        
        # Relationships
        if enhanced_coverage.get('relationships'):
            logger.info("\n🔗 SAMPLE RELATIONSHIPS DISCOVERED:")
            logger.info("-" * 40)
            for i, rel in enumerate(enhanced_coverage['relationships'][:5], 1):
                logger.info(f"  {i}. {rel.get('source_entity_id', 'Unknown')} "
                          f"--[{rel.get('relationship_type', 'relates')}]--> "
                          f"{rel.get('target_entity_id', 'Unknown')}")
        
        # Summary
        logger.info("\n✅ TEST VERIFICATION SUMMARY:")
        logger.info("-" * 40)
        
        entities_target = 30
        relationships_target = 50
        
        entities_achieved = enhanced_coverage.get('metrics', {}).get('total_entities', 0)
        relationships_achieved = enhanced_coverage.get('metrics', {}).get('total_relationships', 0)
        
        logger.info(f"Target: {entities_target}+ entities, {relationships_target}+ relationships")
        logger.info(f"Achieved: {entities_achieved} entities, {relationships_achieved} relationships")
        
        if entities_achieved >= entities_target and relationships_achieved >= relationships_target:
            logger.info("🎉 SUCCESS: System meets or exceeds target metrics!")
        else:
            logger.info(f"⚠️  Partial success: Entities {'✓' if entities_achieved >= entities_target else '✗'}, "
                      f"Relationships {'✓' if relationships_achieved >= relationships_target else '✗'}")
        
        # Performance metrics
        logger.info("\n⏱️  PERFORMANCE METRICS:")
        logger.info("-" * 40)
        total_time = sum([
            basic.get('processing_time', 0),
            enhanced.get('processing_time', 0),
            basic_coverage.get('processing_time', 0),
            enhanced_coverage.get('processing_time', 0)
        ])
        logger.info(f"Total test execution time: {total_time:.2f}s")
        logger.info(f"Average entity discovery rate: {entities_achieved / max(enhanced_coverage.get('processing_time', 1), 1):.1f} entities/second")
    
    async def run_comprehensive_test(self):
        """Run comprehensive test suite"""
        query = "what are the essential technologies of AI implementation, favor open source"
        
        logger.info("\n" + "=" * 80)
        logger.info("STARTING COMPREHENSIVE RADIATING SYSTEM TEST")
        logger.info(f"Test Query: '{query}'")
        logger.info("=" * 80)
        
        try:
            # Test 1: Basic entity extraction
            logger.info("\n📋 Test 1: Basic Entity Extraction")
            self.test_results['basic_extraction'] = await self.test_basic_extraction(query)
            
            # Test 2: Enhanced entity extraction with web search
            logger.info("\n📋 Test 2: Enhanced Entity Extraction")
            self.test_results['enhanced_extraction'] = await self.test_enhanced_extraction(query)
            
            # Test 3: Basic radiating coverage
            logger.info("\n📋 Test 3: Basic Radiating Coverage")
            self.test_results['basic_coverage'] = await self.test_radiating_coverage(query, use_web_search=False)
            
            # Test 4: Enhanced radiating coverage with web search
            logger.info("\n📋 Test 4: Enhanced Radiating Coverage")
            self.test_results['enhanced_coverage'] = await self.test_radiating_coverage(query, use_web_search=True)
            
            # Test 5: Agent processing
            logger.info("\n📋 Test 5: RadiatingAgent Processing")
            self.test_results['agent_processing'] = await self.test_agent_processing(query)
            
            # Print comprehensive comparison
            self.print_detailed_comparison()
            
            # Save results to file
            await self.save_results()
            
        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    async def save_results(self):
        """Save test results to file"""
        filename = f"radiating_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        logger.info(f"\n💾 Test results saved to: {filename}")


async def main():
    """Main test execution function"""
    tester = RadiatingSystemTester()
    await tester.run_comprehensive_test()


if __name__ == "__main__":
    # Run the test
    asyncio.run(main())