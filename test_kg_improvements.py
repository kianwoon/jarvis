#!/usr/bin/env python3
"""
Knowledge Graph Improvements Test Script

Tests the real-world functionality of our knowledge graph enhancements
to verify they work as claimed and measure actual improvements.
"""

import json
import requests
import time
from typing import Dict, Any

class KnowledgeGraphTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = {}
    
    def test_quality_assessment(self):
        """Test the quality assessment functionality"""
        print("🔍 Testing quality assessment...")
        
        response = requests.get(f"{self.base_url}/api/v1/knowledge-graph/quality-assessment")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Quality assessment working")
            print(f"   Overall score: {data['overall_quality_score']:.3f}")
            print(f"   Generic relationships: {data['relationship_quality']['generic_ratio']:.1%}")
            print(f"   Semantic relationships: {data['relationship_quality']['semantic_ratio']:.1%}")
            print(f"   Classification errors: {len(data['quality_issues']['classification_errors'])}")
            print(f"   Naming issues: {len(data['quality_issues']['naming_issues'])}")
            print(f"   Questionable relationships: {len(data['quality_issues']['questionable_relationships'])}")
            self.results['quality_assessment'] = data
            return True
        else:
            print(f"❌ Quality assessment failed: {response.status_code}")
            return False
    
    def test_graph_connectivity(self):
        """Test connectivity analysis"""
        print("\n🔗 Testing connectivity analysis...")
        
        response = requests.get(f"{self.base_url}/api/v1/knowledge-graph/connectivity-analysis")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Connectivity analysis working")
            print(f"   Connectivity ratio: {data['connectivity_ratio']:.1%}")
            print(f"   Average degree: {data['average_degree']:.2f}")
            print(f"   Isolated nodes: {data['isolated_nodes']}")
            self.results['connectivity'] = data
            return True
        else:
            print(f"❌ Connectivity analysis failed: {response.status_code}")
            return False
    
    def test_basic_graph_stats(self):
        """Test basic graph statistics"""
        print("\n📊 Testing basic graph stats...")
        
        response = requests.get(f"{self.base_url}/api/v1/knowledge-graph/stats")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Basic stats working")
            print(f"   Total entities: {data['total_entities']}")
            print(f"   Total relationships: {data['total_relationships']}")
            print(f"   Entity types: {len(data['entity_types'])}")
            print(f"   Relationship types: {len(data['relationship_types'])}")
            self.results['basic_stats'] = data
            return True
        else:
            print(f"❌ Basic stats failed: {response.status_code}")
            return False
    
    def test_health_check(self):
        """Test health check and feature enablement"""
        print("\n🏥 Testing health check...")
        
        response = requests.get(f"{self.base_url}/api/v1/knowledge-graph/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check working")
            print(f"   Status: {data['status']}")
            print(f"   LLM enhancement: {data['config']['llm_enhancement_enabled']}")
            print(f"   Cross-document linking: {data['config']['cross_document_linking']}")
            print(f"   Multi-chunk processing: {data['config']['multi_chunk_processing']}")
            self.results['health'] = data
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    
    def analyze_relationship_quality(self):
        """Analyze relationship quality in detail"""
        print("\n🔬 Analyzing relationship quality...")
        
        # Get sample of high-quality relationships
        query = {
            "query": "MATCH (a)-[r]->(b) WHERE r.confidence >= 0.7 RETURN a.name as source, type(r) as relationship, b.name as target, r.confidence as confidence ORDER BY r.confidence DESC LIMIT 10",
            "query_type": "cypher"
        }
        
        response = requests.post(
            f"{self.base_url}/api/v1/knowledge-graph/query",
            headers={"Content-Type": "application/json"},
            json=query
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ High-quality relationships found: {len(data['results'])}")
            for rel in data['results']:
                print(f"   {rel['source']} --[{rel['relationship']}]--> {rel['target']} (conf: {rel['confidence']})")
            self.results['high_quality_rels'] = data['results']
            return True
        else:
            print(f"❌ Relationship analysis failed: {response.status_code}")
            return False
    
    def check_entity_classification_fixes(self):
        """Check if our entity classification fixes worked"""
        print("\n🔧 Checking entity classification fixes...")
        
        # Check SQL entity
        sql_query = {
            "query": "MATCH (n {name: 'Sql'}) RETURN n.name, labels(n) as labels, n.type",
            "query_type": "cypher"
        }
        
        response = requests.post(
            f"{self.base_url}/api/v1/knowledge-graph/query",
            headers={"Content-Type": "application/json"},
            json=sql_query
        )
        
        if response.status_code == 200:
            data = response.json()
            if data['results']:
                result = data['results'][0]
                if 'CONCEPT' in result['labels']:
                    print(f"✅ SQL entity correctly classified as CONCEPT")
                else:
                    print(f"❌ SQL entity still misclassified: {result['labels']}")
            else:
                print("❌ SQL entity not found")
        
        # Check PostgreSQL entity
        pg_query = {
            "query": "MATCH (n {name: 'Postgresql'}) RETURN n.name, labels(n) as labels, n.type",
            "query_type": "cypher"
        }
        
        response = requests.post(
            f"{self.base_url}/api/v1/knowledge-graph/query",
            headers={"Content-Type": "application/json"},
            json=sql_query
        )
        
        if response.status_code == 200:
            data = response.json()
            if data['results']:
                result = data['results'][0]
                if 'CONCEPT' in result['labels']:
                    print(f"✅ PostgreSQL entity correctly classified as CONCEPT")
                else:
                    print(f"❌ PostgreSQL entity still misclassified: {result['labels']}")
    
    def run_all_tests(self):
        """Run all tests and provide summary"""
        print("🚀 Starting Knowledge Graph Improvements Test Suite")
        print("=" * 60)
        
        tests = [
            self.test_health_check,
            self.test_basic_graph_stats,
            self.test_graph_connectivity,
            self.test_quality_assessment,
            self.analyze_relationship_quality,
            self.check_entity_classification_fixes
        ]
        
        passed = 0
        for test in tests:
            try:
                if test():
                    passed += 1
            except Exception as e:
                print(f"❌ Test failed with exception: {e}")
        
        print("\n" + "=" * 60)
        print(f"📋 Test Summary: {passed}/{len(tests)} tests passed")
        
        if 'quality_assessment' in self.results:
            qa = self.results['quality_assessment']
            print(f"\n🎯 Key Quality Metrics:")
            print(f"   Overall Quality Score: {qa['overall_quality_score']:.3f}/1.0")
            print(f"   Connectivity Score: {qa['component_scores']['connectivity']:.3f}/1.0")
            print(f"   Relationship Quality: {qa['component_scores']['relationship_quality']:.3f}/1.0")
            print(f"   Entity Quality: {qa['component_scores']['entity_quality']:.3f}/1.0")
            
            # Calculate what percentage of improvements are working
            issues_found = (
                len(qa['quality_issues']['classification_errors']) +
                len(qa['quality_issues']['naming_issues']) +
                len(qa['quality_issues']['questionable_relationships'])
            )
            
            print(f"\n🔍 Issues Analysis:")
            print(f"   Total issues detected: {issues_found}")
            print(f"   Generic relationships: {qa['relationship_quality']['generic_ratio']:.1%}")
            print(f"   Semantic relationships: {qa['relationship_quality']['semantic_ratio']:.1%}")
            
            # Honest assessment
            if qa['relationship_quality']['semantic_ratio'] > 0.4:
                print(f"✅ Relationship quality is decent (>40% semantic)")
            else:
                print(f"⚠️  Relationship quality needs improvement (<40% semantic)")
            
            if qa['overall_quality_score'] > 0.8:
                print(f"✅ Overall graph quality is good (>0.8)")
            else:
                print(f"⚠️  Overall graph quality needs improvement (<0.8)")
        
        return passed == len(tests)

if __name__ == "__main__":
    tester = KnowledgeGraphTester()
    success = tester.run_all_tests()
    
    if success:
        print(f"\n🎉 All tests passed! Knowledge graph improvements are working.")
    else:
        print(f"\n⚠️  Some tests failed. Check the output above for details.")