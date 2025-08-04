#!/usr/bin/env python3
"""
Test Chunking Strategy Fixes for Business Documents

This script tests the implemented chunking strategy fixes that were designed to address
the poor entity extraction issue with the DBS Technology Strategy document.

The fixes being tested:
1. Modified dynamic_chunk_sizing.py to use balanced chunking for business documents 
   (5-6 chunks of 50K chars max instead of 1 mega chunk)
2. Updated knowledge_graph_settings_cache.py with business document detection and optimized settings
3. Enhanced graph_processor.py to detect business documents and apply appropriate settings

Expected outcomes:
- 60-100 entities extracted (vs. previous 14)
- Organization entities included (previously missing)
- Relationship ratio ≤4 per entity for smooth browser performance
- Multiple balanced chunks instead of single mega chunk
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

# Import document processing services
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.document_handlers.base import ExtractedChunk
from app.document_handlers.graph_processor import get_graph_document_processor
from app.services.dynamic_chunk_sizing import get_dynamic_chunk_sizer, get_optimal_chunk_config
from app.core.knowledge_graph_settings_cache import (
    get_knowledge_graph_settings, 
    detect_business_document, 
    get_business_optimized_settings
)
from app.services.knowledge_graph_service import get_knowledge_graph_service

class ChunkingStrategyTester:
    """Test the chunking strategy fixes for business documents"""
    
    def __init__(self):
        self.graph_processor = get_graph_document_processor()
        self.chunk_sizer = get_dynamic_chunk_sizer()
        self.kg_service = get_knowledge_graph_service()
        
        # Initialize text splitter
        kg_settings = get_knowledge_graph_settings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=kg_settings.get('chunk_size', 1000),
            chunk_overlap=kg_settings.get('chunk_overlap', 200),
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Test document path
        self.test_document = "/Users/kianwoonwong/Downloads/jarvis/DBS Technology Strategy (Confidential).pdf"
        
    def print_section(self, title: str):
        """Print formatted section header"""
        print(f"\n{'='*80}")
        print(f"{title}")
        print(f"{'='*80}")
    
    def print_subsection(self, title: str):
        """Print formatted subsection header"""
        print(f"\n{'-'*60}")
        print(f"{title}")
        print(f"{'-'*60}")
    
    async def _load_and_chunk_document(self, file_path: str) -> List[ExtractedChunk]:
        """Load document and create chunks using PyPDFLoader"""
        try:
            # Load document using PyPDFLoader
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Generate document ID for chunk metadata
            document_id = f"test_{Path(file_path).stem}_{int(time.time())}"
            
            # Split into chunks
            chunks = []
            for i, doc in enumerate(documents):
                text_chunks = self.text_splitter.split_text(doc.page_content)
                
                for j, chunk_text in enumerate(text_chunks):
                    chunk_id = f"{document_id}_page{i}_chunk{j}"
                    chunk = ExtractedChunk(
                        content=chunk_text,
                        metadata={
                            'chunk_id': chunk_id,
                            'document_id': document_id,
                            'page_number': i,
                            'chunk_index': j,
                            'source': file_path,
                            'total_chunks': len(text_chunks)
                        },
                        quality_score=1.0
                    )
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            print(f"❌ Failed to load and chunk document: {e}")
            return []
    
    async def test_business_document_detection(self) -> Dict[str, Any]:
        """Test business document detection functionality"""
        self.print_section("TESTING BUSINESS DOCUMENT DETECTION")
        
        # Test filename detection
        filename = "DBS Technology Strategy (Confidential).pdf"
        is_business_by_filename = detect_business_document(filename=filename)
        
        print(f"📄 Document: {filename}")
        print(f"🏢 Detected as business document: {is_business_by_filename}")
        
        # Read document content for content-based detection
        if Path(self.test_document).exists():
            try:
                # Extract text from PDF for content analysis
                chunks = await self._load_and_chunk_document(self.test_document)
                sample_content = ' '.join(chunk.content[:500] for chunk in chunks[:3])
                
                is_business_by_content = detect_business_document(content=sample_content)
                print(f"📝 Content-based detection: {is_business_by_content}")
                
                # Show business keywords found
                business_keywords = [
                    'strategy', 'technology', 'financial', 'annual report', 'business plan',
                    'quarterly report', 'dbs bank', 'subsidiary', 'organization', 'enterprise',
                    'digital transformation', 'technology roadmap', 'competitive analysis'
                ]
                
                found_keywords = []
                content_lower = sample_content.lower()
                for keyword in business_keywords:
                    if keyword in content_lower:
                        found_keywords.append(keyword)
                
                print(f"🔍 Keywords found: {found_keywords[:5]}...")  # Show first 5
                
                return {
                    'filename_detection': is_business_by_filename,
                    'content_detection': is_business_by_content,
                    'keywords_found': found_keywords,
                    'sample_content_length': len(sample_content)
                }
                
            except Exception as e:
                print(f"❌ Error processing document: {e}")
                return {'error': str(e)}
        else:
            print(f"❌ Document not found: {self.test_document}")
            return {'error': 'Document not found'}
    
    async def test_chunk_sizing_strategy(self, document_type: str = 'general') -> Dict[str, Any]:
        """Test the dynamic chunk sizing strategy"""
        self.print_section(f"TESTING CHUNK SIZING STRATEGY - {document_type.upper()}")
        
        # Get chunk configuration for different document types
        general_config = get_optimal_chunk_config('general')
        business_config = get_optimal_chunk_config('technology_strategy')
        
        print(f"🧠 Model: {self.chunk_sizer.model_name}")
        print(f"📊 Context limit: {self.chunk_sizer.context_limit:,} tokens")
        print(f"📏 Optimal chunk size: {self.chunk_sizer.optimal_chunk_size:,} characters")
        
        self.print_subsection("GENERAL DOCUMENT CONFIGURATION")
        print(f"   Strategy: {general_config['processing_strategy']}")
        print(f"   Max chunk size: {general_config['max_chunk_size']:,} chars")
        print(f"   Target chunks per doc: {general_config.get('target_chunks_per_document', 'N/A')}")
        print(f"   Max consolidation ratio: {general_config.get('max_consolidation_ratio', 'N/A')}:1")
        
        self.print_subsection("BUSINESS DOCUMENT CONFIGURATION")
        print(f"   Strategy: {business_config['processing_strategy']}")
        print(f"   Max chunk size: {business_config['max_chunk_size']:,} chars")
        print(f"   Target chunks per doc: {business_config.get('target_chunks_per_document', 'N/A')}")
        print(f"   Max consolidation ratio: {business_config.get('max_consolidation_ratio', 'N/A')}:1")
        
        # Key differences
        self.print_subsection("KEY DIFFERENCES FOR BUSINESS DOCUMENTS")
        if business_config['processing_strategy'] != general_config['processing_strategy']:
            print(f"✅ Strategy changed: {general_config['processing_strategy']} → {business_config['processing_strategy']}")
        
        business_target = business_config.get('target_chunks_per_document', 0)
        general_target = general_config.get('target_chunks_per_document', 0)
        if business_target != general_target:
            print(f"✅ Target chunks changed: {general_target} → {business_target}")
        
        business_overlap = business_config.get('chunk_overlap', 0)
        general_overlap = general_config.get('chunk_overlap', 0)
        if business_overlap != general_overlap:
            print(f"✅ Overlap changed: {general_overlap} → {business_overlap} chars")
        
        return {
            'general_config': general_config,
            'business_config': business_config,
            'model_info': {
                'name': self.chunk_sizer.model_name,
                'context_limit': self.chunk_sizer.context_limit,
                'optimal_chunk_size': self.chunk_sizer.optimal_chunk_size
            }
        }
    
    async def test_business_settings_optimization(self) -> Dict[str, Any]:
        """Test business document settings optimization"""
        self.print_section("TESTING BUSINESS SETTINGS OPTIMIZATION")
        
        # Get default and business-optimized settings
        default_settings = get_knowledge_graph_settings()
        business_settings = get_business_optimized_settings()
        
        print(f"📊 DEFAULT SETTINGS:")
        print(f"   Max entities per chunk: {default_settings.get('max_entities_per_chunk', 'N/A')}")
        print(f"   Max relationships per chunk: {default_settings.get('max_relationships_per_chunk', 'N/A')}")
        print(f"   Entity confidence threshold: {default_settings.get('extraction', {}).get('min_entity_confidence', 'N/A')}")
        print(f"   Relationship confidence threshold: {default_settings.get('extraction', {}).get('min_relationship_confidence', 'N/A')}")
        
        print(f"\n🏢 BUSINESS-OPTIMIZED SETTINGS:")
        print(f"   Max entities per chunk: {business_settings.get('max_entities_per_chunk', 'N/A')}")
        print(f"   Max relationships per chunk: {business_settings.get('max_relationships_per_chunk', 'N/A')}")
        print(f"   Entity confidence threshold: {business_settings.get('extraction', {}).get('min_entity_confidence', 'N/A')}")
        print(f"   Relationship confidence threshold: {business_settings.get('extraction', {}).get('min_relationship_confidence', 'N/A')}")
        
        # Highlight changes
        self.print_subsection("OPTIMIZATION CHANGES")
        default_entities = default_settings.get('max_entities_per_chunk', 0)
        business_entities = business_settings.get('max_entities_per_chunk', 0)
        if business_entities > default_entities:
            print(f"✅ Increased entity limit: {default_entities} → {business_entities}")
        
        default_rels = default_settings.get('max_relationships_per_chunk', 0)
        business_rels = business_settings.get('max_relationships_per_chunk', 0)
        if business_rels > default_rels:
            print(f"✅ Increased relationship limit: {default_rels} → {business_rels}")
        
        default_entity_conf = default_settings.get('extraction', {}).get('min_entity_confidence', 0)
        business_entity_conf = business_settings.get('extraction', {}).get('min_entity_confidence', 0)
        if business_entity_conf < default_entity_conf:
            print(f"✅ Lowered entity confidence threshold: {default_entity_conf} → {business_entity_conf}")
        
        # Check for business-specific extraction features
        business_extraction = business_settings.get('extraction', {})
        if business_extraction.get('business_document_mode'):
            print(f"✅ Business document mode enabled")
        if business_extraction.get('extract_all_organizations'):
            print(f"✅ Organization extraction enabled")
        if business_extraction.get('enable_multi_pass'):
            print(f"✅ Multi-pass extraction enabled")
            passes = business_extraction.get('extraction_passes', [])
            print(f"   Passes: {', '.join(passes)}")
        
        return {
            'default_settings': default_settings,
            'business_settings': business_settings,
            'optimizations_applied': True
        }
    
    async def process_dbs_document(self) -> Dict[str, Any]:
        """Process the DBS Technology Strategy document with chunking fixes"""
        self.print_section("PROCESSING DBS TECHNOLOGY STRATEGY DOCUMENT")
        
        if not Path(self.test_document).exists():
            error_msg = f"Document not found: {self.test_document}"
            print(f"❌ {error_msg}")
            return {'error': error_msg}
        
        start_time = time.time()
        
        try:
            # Step 1: Extract chunks from PDF
            print(f"📄 Processing PDF: {Path(self.test_document).name}")
            chunks = await self._load_and_chunk_document(self.test_document)
            
            print(f"📊 Initial extraction:")
            print(f"   Chunks extracted: {len(chunks)}")
            if chunks:
                total_chars = sum(len(chunk.content) for chunk in chunks)
                avg_chars = total_chars // len(chunks)
                min_chars = min(len(chunk.content) for chunk in chunks)
                max_chars = max(len(chunk.content) for chunk in chunks)
                
                print(f"   Total characters: {total_chars:,}")
                print(f"   Average chunk size: {avg_chars:,} chars")
                print(f"   Size range: {min_chars:,} - {max_chars:,} chars")
            
            # Step 2: Test business document detection on actual content
            sample_content = ' '.join(chunk.content[:1000] for chunk in chunks[:3])
            is_business = detect_business_document(filename=Path(self.test_document).name, content=sample_content)
            print(f"🏢 Business document detected: {is_business}")
            
            # Step 3: Process for knowledge graph extraction
            document_id = f"dbs_tech_strategy_{int(time.time())}"
            
            print(f"\n🧠 Processing for knowledge graph extraction...")
            print(f"   Document ID: {document_id}")
            print(f"   Using chunking fixes: {'✅ YES' if is_business else '❌ NO'}")
            
            # Process with graph processor (this will apply chunking fixes)
            graph_result = await self.graph_processor.process_document_for_graph(
                chunks=chunks,
                document_id=document_id,
                store_in_neo4j=False,  # Don't store for testing
                progressive_storage=False,
                filename=Path(self.test_document).name,
                document_content=sample_content
            )
            
            processing_time = time.time() - start_time
            
            # Step 4: Analyze results
            self.print_subsection("PROCESSING RESULTS")
            print(f"✅ Processing completed in {processing_time:.2f} seconds")
            print(f"📊 Results:")
            print(f"   Total chunks processed: {graph_result.processed_chunks}/{graph_result.total_chunks}")
            print(f"   Entities extracted: {graph_result.total_entities}")
            print(f"   Relationships extracted: {graph_result.total_relationships}")
            
            # Calculate relationship ratio
            if graph_result.total_entities > 0:
                ratio = graph_result.total_relationships / graph_result.total_entities
                print(f"   Relationship ratio: {ratio:.2f} per entity")
                print(f"   Target ratio achieved: {'✅ YES' if ratio <= 4.0 else '❌ NO'}")
            else:
                ratio = 0
                print(f"   Relationship ratio: N/A (no entities)")
            
            print(f"   Success: {'✅ YES' if graph_result.success else '❌ NO'}")
            if graph_result.errors:
                print(f"   Errors: {len(graph_result.errors)}")
                for error in graph_result.errors[:3]:
                    print(f"      - {error}")
            
            # Step 5: Analyze chunk optimization
            if graph_result.graph_data:
                self.print_subsection("CHUNK ANALYSIS")
                chunk_sizes = []
                entity_counts = []
                relationship_counts = []
                
                for extraction in graph_result.graph_data:
                    # Estimate chunk size from entity/relationship count
                    chunk_sizes.append(len(extraction.entities) + len(extraction.relationships))
                    entity_counts.append(len(extraction.entities))
                    relationship_counts.append(len(extraction.relationships))
                
                if chunk_sizes:
                    print(f"   Processed chunks: {len(chunk_sizes)}")
                    print(f"   Avg entities per chunk: {sum(entity_counts)/len(entity_counts):.1f}")
                    print(f"   Avg relationships per chunk: {sum(relationship_counts)/len(relationship_counts):.1f}")
                    print(f"   Entity distribution: min={min(entity_counts)}, max={max(entity_counts)}")
                    print(f"   Relationship distribution: min={min(relationship_counts)}, max={max(relationship_counts)}")
            
            # Step 6: Entity type analysis
            if graph_result.graph_data:
                self.print_subsection("ENTITY TYPE ANALYSIS")
                entity_types = {}
                organization_entities = []
                
                for extraction in graph_result.graph_data:
                    for entity in extraction.entities:
                        entity_type = entity.label.lower()
                        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
                        
                        # Check for organization entities
                        if 'organization' in entity_type or 'company' in entity_type or 'bank' in entity_type:
                            organization_entities.append(entity.text)
                
                print(f"   Entity types found: {len(entity_types)}")
                for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True)[:10]:
                    print(f"      - {entity_type}: {count}")
                
                print(f"   Organization entities: {len(organization_entities)}")
                for org in organization_entities[:5]:
                    print(f"      - {org}")
                if len(organization_entities) > 5:
                    print(f"      ... and {len(organization_entities) - 5} more")
            
            return {
                'success': graph_result.success,
                'processing_time_seconds': processing_time,
                'initial_chunks': len(chunks),
                'processed_chunks': graph_result.processed_chunks,
                'total_entities': graph_result.total_entities,
                'total_relationships': graph_result.total_relationships,
                'relationship_ratio': ratio,
                'target_ratio_achieved': ratio <= 4.0,
                'business_document_detected': is_business,
                'entity_types': len(entity_types) if 'entity_types' in locals() else 0,
                'organization_entities_found': len(organization_entities) if 'organization_entities' in locals() else 0,
                'errors': graph_result.errors
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Processing failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'processing_time_seconds': processing_time
            }
    
    async def compare_with_previous_results(self, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current results with previous known issues"""
        self.print_section("COMPARISON WITH PREVIOUS RESULTS")
        
        # Previous problematic results (baseline)
        previous_results = {
            'entities': 14,
            'organizations': 0,
            'chunks': 1,  # Single mega chunk
            'strategy': 'full_context_utilization'
        }
        
        current_entities = current_results.get('total_entities', 0)
        current_orgs = current_results.get('organization_entities_found', 0)
        current_chunks = current_results.get('processed_chunks', 0)
        
        print(f"📊 COMPARISON RESULTS:")
        print(f"   Entities: {previous_results['entities']} → {current_entities} ({'✅ IMPROVED' if current_entities > previous_results['entities'] else '❌ NO CHANGE'})")
        print(f"   Organizations: {previous_results['organizations']} → {current_orgs} ({'✅ IMPROVED' if current_orgs > previous_results['organizations'] else '❌ NO CHANGE'})")
        print(f"   Chunks processed: {previous_results['chunks']} → {current_chunks} ({'✅ IMPROVED' if current_chunks > previous_results['chunks'] else '❌ NO CHANGE'})")
        
        # Calculate improvement metrics
        entity_improvement = ((current_entities - previous_results['entities']) / previous_results['entities'] * 100) if previous_results['entities'] > 0 else 0
        org_improvement = current_orgs  # From 0 to current_orgs
        chunk_improvement = ((current_chunks - previous_results['chunks']) / previous_results['chunks'] * 100) if previous_results['chunks'] > 0 else 0
        
        print(f"\n📈 IMPROVEMENT METRICS:")
        print(f"   Entity extraction: {entity_improvement:+.1f}%")
        print(f"   Organization entities: +{org_improvement} (from 0)")
        print(f"   Chunk processing: {chunk_improvement:+.1f}%")
        
        # Success criteria
        success_criteria = {
            'entities_60_to_100': 60 <= current_entities <= 100,
            'organizations_found': current_orgs > 0,
            'relationship_ratio_ok': current_results.get('target_ratio_achieved', False),
            'multiple_chunks': current_chunks > 1
        }
        
        print(f"\n🎯 SUCCESS CRITERIA:")
        for criterion, met in success_criteria.items():
            status = '✅ MET' if met else '❌ NOT MET'
            print(f"   {criterion.replace('_', ' ').title()}: {status}")
        
        overall_success = all(success_criteria.values())
        print(f"\n🏆 OVERALL SUCCESS: {'✅ YES' if overall_success else '❌ NO'}")
        
        return {
            'previous_results': previous_results,
            'current_results': {
                'entities': current_entities,
                'organizations': current_orgs,
                'chunks': current_chunks
            },
            'improvements': {
                'entity_improvement_percent': entity_improvement,
                'organization_improvement': org_improvement,
                'chunk_improvement_percent': chunk_improvement
            },
            'success_criteria': success_criteria,
            'overall_success': overall_success
        }
    
    async def run_full_test(self) -> Dict[str, Any]:
        """Run complete chunking strategy fix validation"""
        print("🧪 CHUNKING STRATEGY FIXES VALIDATION TEST")
        print(f"📅 Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'test_document': self.test_document
        }
        
        try:
            # Test 1: Business document detection
            detection_results = await self.test_business_document_detection()
            test_results['business_detection'] = detection_results
            
            # Test 2: Chunk sizing strategy
            chunking_results = await self.test_chunk_sizing_strategy()
            test_results['chunking_strategy'] = chunking_results
            
            # Test 3: Business settings optimization
            settings_results = await self.test_business_settings_optimization()
            test_results['business_settings'] = settings_results
            
            # Test 4: Process DBS document
            processing_results = await self.process_dbs_document()
            test_results['document_processing'] = processing_results
            
            # Test 5: Compare with previous results
            comparison_results = await self.compare_with_previous_results(processing_results)
            test_results['comparison'] = comparison_results
            
            # Final summary
            self.print_section("FINAL SUMMARY")
            
            overall_success = (
                detection_results.get('filename_detection', False) and
                processing_results.get('success', False) and
                comparison_results.get('overall_success', False)
            )
            
            print(f"🎯 CHUNKING STRATEGY FIXES VALIDATION: {'✅ PASSED' if overall_success else '❌ FAILED'}")
            
            if overall_success:
                print(f"✅ All fixes working correctly:")
                print(f"   - Business document detection: Working")
                print(f"   - Balanced chunking strategy: Applied")
                print(f"   - Entity extraction improved: {processing_results.get('total_entities', 0)} entities")
                print(f"   - Organization entities found: {processing_results.get('organization_entities_found', 0)}")
                print(f"   - Relationship ratio controlled: {processing_results.get('relationship_ratio', 0):.2f} ≤ 4.0")
            else:
                print(f"❌ Issues found:")
                if not detection_results.get('filename_detection', False):
                    print(f"   - Business document detection failed")
                if not processing_results.get('success', False):
                    print(f"   - Document processing failed")
                if not comparison_results.get('overall_success', False):
                    print(f"   - Success criteria not met")
            
            test_results['overall_success'] = overall_success
            
            # Save results
            results_file = f"chunking_strategy_test_results_{int(time.time())}.json"
            with open(results_file, 'w') as f:
                json.dump(test_results, f, indent=2, default=str)
            
            print(f"\n📄 Full test results saved to: {results_file}")
            
            return test_results
            
        except Exception as e:
            error_msg = f"Test execution failed: {str(e)}"
            print(f"❌ {error_msg}")
            test_results['error'] = error_msg
            test_results['overall_success'] = False
            return test_results

async def main():
    """Main test execution"""
    tester = ChunkingStrategyTester()
    results = await tester.run_full_test()
    
    # Exit with appropriate code
    exit_code = 0 if results.get('overall_success', False) else 1
    print(f"\n🏁 Test completed with exit code: {exit_code}")
    return exit_code

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)