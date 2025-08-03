#!/usr/bin/env python3
"""
Comprehensive anti-silo test with problematic entities
Tests the enhanced alias detection and entity normalization
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent / "app"))

from services.knowledge_graph_service import KnowledgeGraphService

async def test_comprehensive_anti_silo():
    """Test anti-silo improvements with comprehensive test documents"""
    
    # Create test documents with the problematic entities mentioned by user
    test_docs = {
        "doc1.txt": """
        Apache Kafka is a distributed streaming platform that has become essential for real-time data processing. 
        Many companies in India are adopting Kafka for their data infrastructure needs. The technology originated 
        from LinkedIn and has since become an Apache Foundation project.
        
        PostgreSQL is another critical technology being widely adopted across Asia, particularly in countries 
        like Indonesia where digital transformation is accelerating. DBS Bank has been evaluating PostgreSQL 
        for their core banking systems.
        
        Temenos, the banking software company, has been expanding its presence in the East Asian markets, 
        including partnerships with financial institutions in Singapore and Hong Kong.
        """,
        
        "doc2.txt": """
        In the eastern regions of Asia, kafka deployment patterns have evolved significantly. Indonesia's 
        fintech sector relies heavily on streaming data platforms for real-time payment processing.
        
        Postgres databases are being used extensively by Indian startups and established companies alike. 
        The scalability and reliability of postgresql make it ideal for high-transaction environments.
        
        Financial institutions in the east are increasingly looking at Temenos solutions for digital 
        banking transformation. The company's core banking platform supports modern payment infrastructures.
        """,
        
        "doc3.txt": """
        Apache Kafka's adoption in India continues to grow as more organizations recognize its value for 
        building resilient data pipelines. Companies are using Kafka for everything from log aggregation 
        to real-time analytics.
        
        PostgreSQL has proven its worth in demanding environments across Indonesia and other Southeast 
        Asian markets. Its support for ACID transactions makes it particularly suitable for financial 
        applications.
        
        Temenos has established strong partnerships throughout Eastern Asia, helping banks modernize 
        their core systems. The platform's flexibility allows for rapid deployment of new financial 
        products and services.
        """
    }
    
    print("🧪 Starting comprehensive anti-silo test...")
    print("=" * 80)
    
    # Initialize knowledge graph service
    kg_service = KnowledgeGraphService()
    
    # Clear any existing test data
    try:
        await kg_service.clear_all_nodes()
        print("✅ Cleared existing test data")
    except Exception as e:
        print(f"⚠️  Could not clear existing data: {e}")
    
    # Process each document
    all_entities = []
    all_relationships = []
    
    for doc_name, content in test_docs.items():
        print(f"\n📄 Processing {doc_name}...")
        print("-" * 40)
        
        try:
            # Process document
            result = await kg_service.process_document(
                content=content,
                document_id=doc_name,
                metadata={"source": doc_name, "test_type": "anti_silo_comprehensive"}
            )
            
            if result.get('success'):
                entities = result.get('entities', [])
                relationships = result.get('relationships', [])
                
                print(f"   Extracted {len(entities)} entities")
                print(f"   Extracted {len(relationships)} relationships")
                
                # Show key entities for verification
                key_entities = [e for e in entities if any(term in e.get('name', '').lower() 
                                                          for term in ['kafka', 'postgres', 'temenos', 'india', 'indonesia', 'east'])]
                
                if key_entities:
                    print("   🎯 Key entities found:")
                    for entity in key_entities:
                        print(f"      - {entity.get('name')} ({entity.get('type')})")
                
                all_entities.extend(entities)
                all_relationships.extend(relationships)
                
            else:
                print(f"   ❌ Failed to process {doc_name}: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ Error processing {doc_name}: {str(e)}")
    
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    
    # Analyze entity variations and normalization
    entity_names = [e.get('name', '') for e in all_entities]
    
    # Check for the problematic entities mentioned by user
    problematic_entities = ['apache kafka', 'kafka', 'india', 'east', 'indonesia', 'postgresql', 'postgres', 'temenos']
    
    print(f"\n🔍 Entity Analysis (Total: {len(all_entities)})")
    print("-" * 40)
    
    found_variations = {}
    for prob_entity in problematic_entities:
        variations = [name for name in entity_names if prob_entity.lower() in name.lower() or name.lower() in prob_entity.lower()]
        if variations:
            found_variations[prob_entity] = list(set(variations))
    
    for entity, variations in found_variations.items():
        print(f"📋 {entity.upper()}:")
        for var in variations:
            print(f"   - {var}")
    
    # Check for anti-silo relationships
    print(f"\n🔗 Relationship Analysis (Total: {len(all_relationships)})")
    print("-" * 40)
    
    cross_doc_relationships = []
    for rel in all_relationships:
        source_meta = rel.get('source_metadata', {})
        target_meta = rel.get('target_metadata', {})
        
        if (source_meta.get('document_id') != target_meta.get('document_id') and 
            source_meta.get('document_id') and target_meta.get('document_id')):
            cross_doc_relationships.append(rel)
    
    print(f"🌉 Cross-document relationships: {len(cross_doc_relationships)}")
    
    if cross_doc_relationships:
        print("\n📝 Cross-document relationship examples:")
        for i, rel in enumerate(cross_doc_relationships[:5]):  # Show first 5
            source = rel.get('source', 'Unknown')
            target = rel.get('target', 'Unknown')
            rel_type = rel.get('type', 'Unknown')
            source_doc = rel.get('source_metadata', {}).get('document_id', 'Unknown')
            target_doc = rel.get('target_metadata', {}).get('document_id', 'Unknown')
            
            print(f"   {i+1}. {source} --[{rel_type}]--> {target}")
            print(f"      ({source_doc} → {target_doc})")
    
    # Query Neo4j for silo analysis
    try:
        print(f"\n🏝️  SILO ANALYSIS")
        print("-" * 40)
        
        # Get all nodes and their connection counts
        silo_query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]-(connected)
        WITH n, count(DISTINCT connected) as connection_count
        WHERE connection_count <= 1
        RETURN n.name as name, n.type as type, connection_count
        ORDER BY n.name
        """
        
        silo_results = await kg_service.execute_cypher(silo_query)
        
        if silo_results:
            silo_nodes = [result for result in silo_results if result.get('connection_count', 0) <= 1]
            
            print(f"🏝️  Found {len(silo_nodes)} potential silo nodes:")
            
            # Check specifically for the problematic entities
            problematic_silos = []
            for node in silo_nodes:
                node_name = node.get('name', '').lower()
                if any(prob.lower() in node_name or node_name in prob.lower() for prob in problematic_entities):
                    problematic_silos.append(node)
            
            if problematic_silos:
                print(f"   ⚠️  PROBLEMATIC SILOS ({len(problematic_silos)}):")
                for node in problematic_silos:
                    print(f"      - {node.get('name')} ({node.get('type')}) - {node.get('connection_count', 0)} connections")
            else:
                print(f"   ✅ No problematic entities found in silo nodes!")
            
            if len(silo_nodes) > len(problematic_silos):
                print(f"   📋 Other silo nodes ({len(silo_nodes) - len(problematic_silos)}):")
                other_silos = [node for node in silo_nodes if node not in problematic_silos]
                for node in other_silos[:10]:  # Show first 10
                    print(f"      - {node.get('name')} ({node.get('type')}) - {node.get('connection_count', 0)} connections")
        else:
            print("   ℹ️  No silo analysis results returned")
            
    except Exception as e:
        print(f"   ❌ Error in silo analysis: {str(e)}")
    
    # Final assessment
    print(f"\n" + "=" * 80)
    print("🎯 FINAL ASSESSMENT")
    print("=" * 80)
    
    # Check if we successfully linked the problematic entities
    success_metrics = {
        'entities_extracted': len(all_entities),
        'relationships_created': len(all_relationships),
        'cross_doc_relationships': len(cross_doc_relationships),
        'entity_variations_found': len(found_variations),
        'problematic_silos_remaining': len(problematic_silos) if 'problematic_silos' in locals() else 'Unknown'
    }
    
    print("📊 Success Metrics:")
    for metric, value in success_metrics.items():
        print(f"   - {metric.replace('_', ' ').title()}: {value}")
    
    # Determine overall success
    if (cross_doc_relationships and 
        len(found_variations) >= 4 and  # Found variations for most problematic entities
        (not 'problematic_silos' in locals() or len(problematic_silos) == 0)):
        print("\n✅ ANTI-SILO IMPROVEMENTS: SUCCESS!")
        print("   Enhanced alias detection and entity normalization working properly")
    else:
        print("\n⚠️  ANTI-SILO IMPROVEMENTS: NEEDS ATTENTION")
        print("   Some problematic entities may still be forming silos")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(test_comprehensive_anti_silo())