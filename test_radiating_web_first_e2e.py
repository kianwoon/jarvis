#!/usr/bin/env python3
"""
End-to-End Test for Web-First Radiating Coverage System

This test verifies that the radiating system correctly:
1. Triggers web search for technology queries
2. Discovers entities from web search results
3. Stores web entities in Neo4j with proper labels
4. Extracts relationships from search snippets
5. Demonstrates superior coverage vs LLM-only approach

Run with: python test_radiating_web_first_e2e.py
"""

import asyncio
import json
import logging
import sys
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import os

# Setup logging to see the flow
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_radiating_web_first.log')
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

@dataclass
class TestResult:
    """Container for test results"""
    test_name: str
    success: bool
    duration: float
    entities_found: int
    web_entities: int
    llm_entities: int
    relationships: int
    neo4j_stored: int
    message: str
    details: Dict[str, Any] = None


class RadiatingWebFirstTest:
    """Comprehensive test suite for web-first radiating system"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.radiating_service = None
        self.neo4j_service = None
        self.test_start_time = datetime.now()
        
    async def setup(self):
        """Initialize services and connections"""
        try:
            logger.info("🚀 Setting up test environment...")
            
            # Initialize services
            from app.services.radiating.radiating_service import RadiatingService
            from app.services.neo4j_service import Neo4jService
            from app.core.radiating_settings_cache import get_radiating_settings
            
            self.radiating_service = RadiatingService()
            
            # Try to initialize Neo4j (may fail if not running)
            try:
                self.neo4j_service = Neo4jService()
                # Verify Neo4j connection
                neo4j_status = self.neo4j_service.test_connection()
                if not neo4j_status.get('success'):
                    logger.warning(f"⚠️ Neo4j not available: {neo4j_status.get('error')}")
                    self.neo4j_service = None
                else:
                    logger.info(f"✅ Neo4j connected: {neo4j_status.get('database_info', {}).get('node_count', 0)} nodes")
            except Exception as e:
                logger.warning(f"⚠️ Neo4j initialization failed: {e}")
                self.neo4j_service = None
            
            # Get current settings
            settings = get_radiating_settings()
            logger.info(f"📋 Radiating settings: enabled={settings.get('enabled')}, "
                       f"max_depth={settings.get('max_depth')}, "
                       f"traversal_strategy={settings.get('traversal_strategy')}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False
    
    async def test_web_first_technology_query(self):
        """Test 1: Web-First Technology Query"""
        test_name = "Web-First Technology Query"
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 TEST 1: {test_name}")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Test query that should trigger web search
            query = "What are the latest open source LLM frameworks in 2025?"
            logger.info(f"📝 Query: {query}")
            
            logger.info("🔍 Starting radiating coverage with web-first approach...")
            
            # Use execute_radiating_query with proper parameters
            result = await self.radiating_service.execute_radiating_query(
                query=query,
                max_depth=2,
                strategy='best_first',
                filters={'use_web_search': True},  # Force web search
                include_coverage=True
            )
            
            # Analyze results
            web_entities = []
            llm_entities = []
            all_entities = []
            
            # Extract coverage information from result
            coverage = result.get('coverage', {}) if result else {}
            entities_data = coverage.get('entities', [])
            
            # Process entities
            for entity in entities_data:
                all_entities.append(entity)
                # Check source in metadata
                metadata = entity.get('metadata', {})
                source = metadata.get('source', 'unknown')
                if source == 'web_search' or metadata.get('extraction_method') == 'web_first':
                    web_entities.append(entity)
                else:
                    llm_entities.append(entity)
            
            # Count relationships
            relationships = coverage.get('relationships', [])
            
            duration = time.time() - start_time
            
            # Log results
            logger.info(f"\n📊 Results Summary:")
            logger.info(f"  Total Entities: {len(all_entities)}")
            logger.info(f"  Web-Sourced: {len(web_entities)}")
            logger.info(f"  LLM-Sourced: {len(llm_entities)}")
            logger.info(f"  Relationships: {len(relationships)}")
            logger.info(f"  Duration: {duration:.2f}s")
            
            # Display sample entities
            if web_entities:
                logger.info(f"\n🌐 Sample Web-Sourced Entities:")
                for entity in web_entities[:5]:
                    name = entity.get('name', entity.get('text', 'Unknown'))
                    etype = entity.get('type', entity.get('entity_type', 'Unknown'))
                    conf = entity.get('confidence', 0.0)
                    logger.info(f"  - {name} ({etype}) [conf: {conf:.2f}]")
            
            if llm_entities:
                logger.info(f"\n🤖 Sample LLM-Sourced Entities:")
                for entity in llm_entities[:5]:
                    name = entity.get('name', entity.get('text', 'Unknown'))
                    etype = entity.get('type', entity.get('entity_type', 'Unknown'))
                    conf = entity.get('confidence', 0.0)
                    logger.info(f"  - {name} ({etype}) [conf: {conf:.2f}]")
            
            # Store result
            result = TestResult(
                test_name=test_name,
                success=len(web_entities) > 0,
                duration=duration,
                entities_found=len(all_entities),
                web_entities=len(web_entities),
                llm_entities=len(llm_entities),
                relationships=len(relationships),
                neo4j_stored=0,  # Will check in Neo4j test
                message=f"Successfully discovered {len(web_entities)} web entities",
                details={
                    'query': query,
                    'sample_web_entities': [e.get('name', e.get('text', 'Unknown')) for e in web_entities[:10]] if web_entities else [],
                    'sample_llm_entities': [e.get('name', e.get('text', 'Unknown')) for e in llm_entities[:10]] if llm_entities else []
                }
            )
            self.results.append(result)
            
            # Store entities for comparison test
            self.web_first_entities = all_entities
            self.web_first_web_count = len(web_entities)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}", exc_info=True)
            result = TestResult(
                test_name=test_name,
                success=False,
                duration=time.time() - start_time,
                entities_found=0,
                web_entities=0,
                llm_entities=0,
                relationships=0,
                neo4j_stored=0,
                message=f"Test failed: {str(e)}"
            )
            self.results.append(result)
            return result
    
    async def test_llm_only_comparison(self):
        """Test 2: LLM-Only Comparison"""
        test_name = "LLM-Only Comparison"
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 TEST 2: {test_name}")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Same query but with web search disabled
            query = "What are the latest open source LLM frameworks in 2025?"
            logger.info(f"📝 Query: {query}")
            logger.info("🤖 Using LLM-only extraction (web search disabled)...")
            
            # Use entity extractor directly without web search
            from app.services.radiating.extraction.universal_entity_extractor import UniversalEntityExtractor
            
            extractor = UniversalEntityExtractor()
            
            # Extract with prefer_web_search=False to force LLM-only
            entities = await extractor.extract_entities(
                text=query,
                domain_hints=['AI', 'LLM', 'frameworks'],
                context="User is asking about LLM technologies",
                prefer_web_search=False  # Force LLM-only
            )
            
            duration = time.time() - start_time
            
            # Log results
            logger.info(f"\n📊 LLM-Only Results:")
            logger.info(f"  Total Entities: {len(entities)}")
            logger.info(f"  Duration: {duration:.2f}s")
            
            # Display entities
            if entities:
                logger.info(f"\n🤖 LLM Entities:")
                for entity in entities[:10]:
                    logger.info(f"  - {entity.text} ({entity.entity_type}) [conf: {entity.confidence:.2f}]")
            
            # Compare with web-first approach
            comparison_message = ""
            if hasattr(self, 'web_first_web_count'):
                improvement = ((self.web_first_web_count - len(entities)) / max(len(entities), 1)) * 100
                comparison_message = f"\n📈 Web-First Improvement: {improvement:.1f}% more entities"
                logger.info(comparison_message)
                logger.info(f"  Web-First found: {self.web_first_web_count} entities")
                logger.info(f"  LLM-Only found: {len(entities)} entities")
            
            # Store result
            result = TestResult(
                test_name=test_name,
                success=True,
                duration=duration,
                entities_found=len(entities),
                web_entities=0,
                llm_entities=len(entities),
                relationships=0,
                neo4j_stored=0,
                message=f"LLM-only found {len(entities)} entities. {comparison_message}",
                details={
                    'query': query,
                    'llm_entities': [e.text for e in entities[:10]] if entities else [],
                    'comparison': {
                        'web_first_count': getattr(self, 'web_first_web_count', 0),
                        'llm_only_count': len(entities),
                        'improvement_percent': improvement if hasattr(self, 'web_first_web_count') else 0
                    }
                }
            )
            self.results.append(result)
            return result
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}", exc_info=True)
            result = TestResult(
                test_name=test_name,
                success=False,
                duration=time.time() - start_time,
                entities_found=0,
                web_entities=0,
                llm_entities=0,
                relationships=0,
                neo4j_stored=0,
                message=f"Test failed: {str(e)}"
            )
            self.results.append(result)
            return result
    
    async def test_neo4j_persistence(self):
        """Test 3: Neo4j Persistence"""
        test_name = "Neo4j Persistence"
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 TEST 3: {test_name}")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            if not self.neo4j_service or not self.neo4j_service.is_enabled():
                logger.warning("⚠️ Neo4j not available, skipping persistence test")
                result = TestResult(
                    test_name=test_name,
                    success=False,
                    duration=0,
                    entities_found=0,
                    web_entities=0,
                    llm_entities=0,
                    relationships=0,
                    neo4j_stored=0,
                    message="Neo4j not available"
                )
                self.results.append(result)
                return result
            
            logger.info("🔍 Querying Neo4j for WEB_SOURCED entities...")
            
            # Query for web-sourced entities
            with self.neo4j_service.driver.session() as session:
                # Count WEB_SOURCED entities
                count_result = session.run("""
                    MATCH (e:Entity:WEB_SOURCED)
                    RETURN count(e) as count
                """)
                web_count = count_result.single()['count']
                
                # Get sample entities
                sample_result = session.run("""
                    MATCH (e:Entity:WEB_SOURCED)
                    RETURN e.name as name, 
                           e.type as type, 
                           e.confidence_score as confidence,
                           e.source_url as url,
                           e.discovery_timestamp as timestamp
                    ORDER BY e.discovery_timestamp DESC
                    LIMIT 10
                """)
                
                samples = []
                for record in sample_result:
                    samples.append({
                        'name': record['name'],
                        'type': record['type'],
                        'confidence': record['confidence'],
                        'url': record['url'],
                        'timestamp': record['timestamp']
                    })
                
                # Count relationships
                rel_result = session.run("""
                    MATCH (e1:Entity:WEB_SOURCED)-[r]->(e2:Entity)
                    RETURN count(r) as count
                """)
                rel_count = rel_result.single()['count']
            
            duration = time.time() - start_time
            
            # Log results
            logger.info(f"\n📊 Neo4j Storage Results:")
            logger.info(f"  WEB_SOURCED Entities: {web_count}")
            logger.info(f"  Relationships: {rel_count}")
            logger.info(f"  Duration: {duration:.2f}s")
            
            if samples:
                logger.info(f"\n🗄️ Sample Stored Entities:")
                for sample in samples[:5]:
                    logger.info(f"  - {sample['name']} ({sample['type']}) "
                               f"[conf: {sample['confidence']:.2f}]")
                    if sample['url']:
                        logger.info(f"    URL: {sample['url'][:60]}...")
            
            # Store result
            result = TestResult(
                test_name=test_name,
                success=web_count > 0,
                duration=duration,
                entities_found=web_count,
                web_entities=web_count,
                llm_entities=0,
                relationships=rel_count,
                neo4j_stored=web_count,
                message=f"Found {web_count} WEB_SOURCED entities in Neo4j",
                details={
                    'neo4j_web_count': web_count,
                    'neo4j_relationships': rel_count,
                    'sample_entities': samples[:5]
                }
            )
            self.results.append(result)
            return result
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}", exc_info=True)
            result = TestResult(
                test_name=test_name,
                success=False,
                duration=time.time() - start_time,
                entities_found=0,
                web_entities=0,
                llm_entities=0,
                relationships=0,
                neo4j_stored=0,
                message=f"Test failed: {str(e)}"
            )
            self.results.append(result)
            return result
    
    async def test_performance_and_caching(self):
        """Test 4: Performance and Caching"""
        test_name = "Performance and Caching"
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 TEST 4: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            query = "What are the best vector databases for RAG systems?"
            logger.info(f"📝 Query: {query}")
            
            # First run (cold cache)
            logger.info("\n🥶 First run (cold cache)...")
            start_time = time.time()
            
            result1 = await self.radiating_service.execute_radiating_query(
                query=query,
                max_depth=1,
                strategy='best_first',
                filters={'use_web_search': True},
                include_coverage=True
            )
            
            duration1 = time.time() - start_time
            coverage1 = result1.get('coverage', {}) if result1 else {}
            entities1 = len(coverage1.get('entities', []))
            
            logger.info(f"  Duration: {duration1:.2f}s")
            logger.info(f"  Entities: {entities1}")
            
            # Second run (warm cache)
            logger.info("\n🔥 Second run (warm cache)...")
            start_time = time.time()
            
            result2 = await self.radiating_service.execute_radiating_query(
                query=query,
                max_depth=1,
                strategy='best_first',
                filters={'use_web_search': True},
                include_coverage=True
            )
            
            duration2 = time.time() - start_time
            coverage2 = result2.get('coverage', {}) if result2 else {}
            entities2 = len(coverage2.get('entities', []))
            
            logger.info(f"  Duration: {duration2:.2f}s")
            logger.info(f"  Entities: {entities2}")
            
            # Calculate speedup
            speedup = (duration1 / duration2) if duration2 > 0 else 0
            cache_benefit = ((duration1 - duration2) / duration1 * 100) if duration1 > 0 else 0
            
            logger.info(f"\n📈 Performance Analysis:")
            logger.info(f"  Speedup: {speedup:.2f}x")
            logger.info(f"  Cache Benefit: {cache_benefit:.1f}% faster")
            logger.info(f"  Time Saved: {duration1 - duration2:.2f}s")
            
            # Store result
            result = TestResult(
                test_name=test_name,
                success=duration2 < duration1,
                duration=duration1 + duration2,
                entities_found=entities1,
                web_entities=0,
                llm_entities=0,
                relationships=0,
                neo4j_stored=0,
                message=f"Cache provided {speedup:.2f}x speedup ({cache_benefit:.1f}% faster)",
                details={
                    'query': query,
                    'cold_cache_duration': duration1,
                    'warm_cache_duration': duration2,
                    'speedup': speedup,
                    'cache_benefit_percent': cache_benefit
                }
            )
            self.results.append(result)
            return result
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}", exc_info=True)
            result = TestResult(
                test_name=test_name,
                success=False,
                duration=0,
                entities_found=0,
                web_entities=0,
                llm_entities=0,
                relationships=0,
                neo4j_stored=0,
                message=f"Test failed: {str(e)}"
            )
            self.results.append(result)
            return result
    
    async def cleanup(self, clear_test_data: bool = False):
        """Cleanup test resources"""
        try:
            logger.info("\n🧹 Cleaning up...")
            
            if clear_test_data and self.neo4j_service and self.neo4j_service.is_enabled():
                logger.info("🗑️ Clearing test entities from Neo4j...")
                
                # Option to clear test entities (be careful!)
                if input("Clear WEB_SOURCED test entities? (y/N): ").lower() == 'y':
                    with self.neo4j_service.driver.session() as session:
                        # Only clear entities created in the last hour
                        result = session.run("""
                            MATCH (e:Entity:WEB_SOURCED)
                            WHERE e.discovery_timestamp > datetime() - duration('PT1H')
                            DETACH DELETE e
                            RETURN count(e) as deleted
                        """)
                        deleted = result.single()['deleted']
                        logger.info(f"  Deleted {deleted} test entities")
            
            # Close connections
            if self.neo4j_service and self.neo4j_service.driver:
                self.neo4j_service.driver.close()
                logger.info("✅ Neo4j connection closed")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            return False
    
    def print_summary(self):
        """Print test results summary"""
        logger.info(f"\n{'='*80}")
        logger.info("📊 TEST RESULTS SUMMARY")
        logger.info(f"{'='*80}")
        
        total_duration = (datetime.now() - self.test_start_time).total_seconds()
        passed = sum(1 for r in self.results if r.success)
        failed = len(self.results) - passed
        
        # Overall statistics
        total_web_entities = sum(r.web_entities for r in self.results)
        total_llm_entities = sum(r.llm_entities for r in self.results)
        total_entities = sum(r.entities_found for r in self.results)
        total_relationships = sum(r.relationships for r in self.results)
        
        logger.info(f"\n📈 Overall Statistics:")
        logger.info(f"  Tests Run: {len(self.results)}")
        logger.info(f"  Passed: {passed} ✅")
        logger.info(f"  Failed: {failed} ❌")
        logger.info(f"  Total Duration: {total_duration:.2f}s")
        
        logger.info(f"\n🌐 Entity Discovery:")
        logger.info(f"  Total Entities: {total_entities}")
        logger.info(f"  Web-Sourced: {total_web_entities}")
        logger.info(f"  LLM-Sourced: {total_llm_entities}")
        logger.info(f"  Relationships: {total_relationships}")
        
        # Individual test results
        logger.info(f"\n📋 Individual Test Results:")
        for result in self.results:
            status = "✅" if result.success else "❌"
            logger.info(f"\n  {status} {result.test_name}")
            logger.info(f"     Duration: {result.duration:.2f}s")
            logger.info(f"     Message: {result.message}")
            if result.entities_found > 0:
                logger.info(f"     Entities: {result.entities_found} "
                           f"(Web: {result.web_entities}, LLM: {result.llm_entities})")
        
        # Key findings
        logger.info(f"\n🔍 Key Findings:")
        
        # Find web-first vs LLM-only comparison
        web_first_test = next((r for r in self.results if "Web-First" in r.test_name), None)
        llm_only_test = next((r for r in self.results if "LLM-Only" in r.test_name), None)
        
        if web_first_test and llm_only_test:
            web_count = web_first_test.entities_found
            llm_count = llm_only_test.entities_found
            improvement = ((web_count - llm_count) / max(llm_count, 1)) * 100
            logger.info(f"  📈 Web-First Advantage: {improvement:.1f}% more entities discovered")
            logger.info(f"     Web-First: {web_count} entities")
            logger.info(f"     LLM-Only: {llm_count} entities")
        
        # Find caching benefit
        perf_test = next((r for r in self.results if "Performance" in r.test_name), None)
        if perf_test and perf_test.details:
            speedup = perf_test.details.get('speedup', 0)
            cache_benefit = perf_test.details.get('cache_benefit_percent', 0)
            logger.info(f"  ⚡ Caching Benefit: {speedup:.2f}x speedup ({cache_benefit:.1f}% faster)")
        
        # Neo4j storage
        neo4j_test = next((r for r in self.results if "Neo4j" in r.test_name), None)
        if neo4j_test and neo4j_test.success:
            logger.info(f"  💾 Neo4j Storage: {neo4j_test.neo4j_stored} WEB_SOURCED entities persisted")
        
        logger.info(f"\n{'='*80}")
        logger.info("✨ Web-First Radiating Coverage Test Complete!")
        logger.info(f"{'='*80}\n")
        
        return passed == len(self.results)


async def main():
    """Main test runner"""
    logger.info("🚀 Starting Web-First Radiating Coverage End-to-End Test")
    logger.info(f"Timestamp: {datetime.now().isoformat()}\n")
    
    tester = RadiatingWebFirstTest()
    
    try:
        # Setup
        if not await tester.setup():
            logger.error("Setup failed, aborting tests")
            return False
        
        # Run tests
        await tester.test_web_first_technology_query()
        await asyncio.sleep(1)  # Small delay between tests
        
        await tester.test_llm_only_comparison()
        await asyncio.sleep(1)
        
        await tester.test_neo4j_persistence()
        await asyncio.sleep(1)
        
        await tester.test_performance_and_caching()
        
        # Print summary
        success = tester.print_summary()
        
        # Cleanup
        await tester.cleanup(clear_test_data=False)
        
        return success
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    # Run the async main function
    success = asyncio.run(main())
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)