#!/usr/bin/env python3
"""
Universal Radiating Coverage System - Example Usage

This script demonstrates various use cases of the radiating system:
- Basic entity traversal
- Query expansion
- Performance comparison
- Result visualization
"""

import asyncio
import json
import time
from typing import Dict, List, Any
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.radiating.radiating_service import RadiatingService
from app.services.radiating.models.radiating_context import RadiatingContext
from app.services.radiating.query_expansion.query_analyzer import QueryAnalyzer
from app.services.radiating.engine.radiating_traverser import RadiatingTraverser
from app.services.neo4j_service import Neo4jService
from app.core.radiating_settings_cache import get_radiating_settings

class RadiatingSystemDemo:
    """Demonstration of the Universal Radiating Coverage System"""
    
    def __init__(self):
        self.radiating_service = RadiatingService()
        self.query_analyzer = QueryAnalyzer()
        self.traverser = RadiatingTraverser()
        self.neo4j_service = Neo4jService()
        self.settings = get_radiating_settings()
        
    async def example_basic_traversal(self):
        """Example 1: Basic entity traversal"""
        print("\n" + "="*60)
        print("EXAMPLE 1: Basic Entity Traversal")
        print("="*60)
        
        # Starting entity
        start_entity = "Artificial Intelligence"
        depth = 3
        
        print(f"\nStarting from: '{start_entity}'")
        print(f"Traversal depth: {depth}")
        print("-"*40)
        
        start_time = time.time()
        
        # Create radiating context
        context = RadiatingContext(
            original_query=start_entity,
            depth_limit=depth,
            relevance_threshold=0.6
        )
        
        # Perform traversal
        results = await self.traverser.traverse(
            start_entity=start_entity,
            context=context
        )
        
        elapsed = time.time() - start_time
        
        # Display results
        print(f"\n✅ Traversal completed in {elapsed:.2f} seconds")
        print(f"📊 Results:")
        print(f"  - Entities discovered: {len(results.get('entities', []))}")
        print(f"  - Relationships found: {len(results.get('relationships', []))}")
        print(f"  - Coverage score: {results.get('coverage_score', 0):.2%}")
        
        # Show top entities by relevance
        entities = sorted(
            results.get('entities', []),
            key=lambda x: x.get('relevance_score', 0),
            reverse=True
        )[:5]
        
        print("\n🎯 Top 5 Related Entities:")
        for i, entity in enumerate(entities, 1):
            print(f"  {i}. {entity['name']} (Score: {entity['relevance_score']:.3f})")
        
        return results
    
    async def example_query_expansion(self):
        """Example 2: Query expansion demonstration"""
        print("\n" + "="*60)
        print("EXAMPLE 2: Query Expansion")
        print("="*60)
        
        queries = [
            "What is machine learning?",
            "Latest AI developments",
            "How do neural networks work?",
            "Natural language processing applications"
        ]
        
        for query in queries:
            print(f"\n📝 Original Query: '{query}'")
            print("-"*40)
            
            # Analyze and expand query
            analysis = await self.query_analyzer.analyze_query(query)
            expansions = await self.query_analyzer.expand_query(
                query=query,
                analysis=analysis
            )
            
            print(f"🔍 Query Analysis:")
            print(f"  - Intent: {analysis.get('intent', 'unknown')}")
            print(f"  - Domain: {analysis.get('domain', 'general')}")
            print(f"  - Temporal: {analysis.get('is_temporal', False)}")
            print(f"  - Confidence: {analysis.get('confidence', 0):.2%}")
            
            print(f"\n🌟 Expanded Queries ({len(expansions)}):")
            for i, exp in enumerate(expansions[:3], 1):
                print(f"  {i}. {exp['query']} (Weight: {exp['weight']:.2f})")
    
    async def example_performance_comparison(self):
        """Example 3: Performance comparison with and without radiating"""
        print("\n" + "="*60)
        print("EXAMPLE 3: Performance Comparison")
        print("="*60)
        
        test_query = "Explain the relationship between AI and machine learning"
        
        print(f"\n🔬 Test Query: '{test_query}'")
        print("-"*40)
        
        # Test WITHOUT radiating (standard search)
        print("\n1️⃣ Standard Search (without radiating):")
        start_time = time.time()
        
        # Simulate standard search
        standard_results = await self._standard_search(test_query)
        standard_time = time.time() - start_time
        
        print(f"  ⏱️ Time: {standard_time:.3f}s")
        print(f"  📊 Results: {len(standard_results)} items")
        print(f"  🎯 Relevance: Limited to direct matches")
        
        # Test WITH radiating
        print("\n2️⃣ Radiating Search (with coverage expansion):")
        start_time = time.time()
        
        radiating_results = await self.radiating_service.search(
            query=test_query,
            enable_radiating=True,
            depth=3
        )
        radiating_time = time.time() - start_time
        
        print(f"  ⏱️ Time: {radiating_time:.3f}s")
        print(f"  📊 Results: {len(radiating_results.get('results', []))} items")
        print(f"  🎯 Coverage: {radiating_results.get('coverage_score', 0):.2%}")
        print(f"  🌐 Entities explored: {radiating_results.get('entities_explored', 0)}")
        
        # Performance comparison
        print("\n📈 Performance Analysis:")
        time_ratio = radiating_time / standard_time if standard_time > 0 else 0
        result_ratio = len(radiating_results.get('results', [])) / max(len(standard_results), 1)
        
        print(f"  ⚡ Time overhead: {(time_ratio - 1) * 100:.1f}%")
        print(f"  📚 Result improvement: {(result_ratio - 1) * 100:.1f}%")
        print(f"  💡 Coverage gain: {radiating_results.get('coverage_score', 0):.2%}")
        
        if time_ratio < 2 and result_ratio > 1.5:
            print("\n✅ Recommendation: Radiating provides significant value!")
        elif result_ratio > 2:
            print("\n✅ Recommendation: Excellent coverage improvement!")
        else:
            print("\n⚠️ Consider adjusting radiating parameters for this query type")
    
    async def example_visual_representation(self):
        """Example 4: Visual representation of radiating coverage"""
        print("\n" + "="*60)
        print("EXAMPLE 4: Visual Representation")
        print("="*60)
        
        entity = "Machine Learning"
        
        print(f"\n🎨 Generating visual representation for: '{entity}'")
        print("-"*40)
        
        # Get radiating graph data
        graph_data = await self._get_graph_visualization(entity)
        
        # ASCII visualization
        print("\n📊 Graph Structure:")
        print("```")
        print(f"        [{entity}]")
        print("            |")
        print("    +-------+-------+")
        print("    |       |       |")
        
        # Level 1 entities
        level1 = graph_data.get('level1', [])[:3]
        for e in level1:
            print(f"  [{e['name'][:15]}]", end="")
        print("\n    |       |       |")
        
        # Level 2 entities (simplified)
        print("    ...     ...     ...")
        print("```")
        
        # Statistics
        print("\n📈 Graph Statistics:")
        print(f"  - Total nodes: {graph_data.get('total_nodes', 0)}")
        print(f"  - Total edges: {graph_data.get('total_edges', 0)}")
        print(f"  - Max depth reached: {graph_data.get('max_depth', 0)}")
        print(f"  - Average connectivity: {graph_data.get('avg_connectivity', 0):.2f}")
        
        # Export for visualization tools
        export_path = Path("radiating_graph.json")
        with open(export_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        print(f"\n💾 Graph data exported to: {export_path}")
        print("   (Can be imported into visualization tools like Gephi or D3.js)")
    
    async def example_domain_specific_search(self):
        """Example 5: Domain-specific radiating search"""
        print("\n" + "="*60)
        print("EXAMPLE 5: Domain-Specific Search")
        print("="*60)
        
        domains = {
            "Technology": "What are the latest breakthroughs in quantum computing?",
            "Healthcare": "How does AI improve medical diagnosis?",
            "Finance": "Applications of blockchain in banking",
            "Education": "Impact of AI on personalized learning"
        }
        
        for domain, query in domains.items():
            print(f"\n🏢 Domain: {domain}")
            print(f"📝 Query: '{query}'")
            print("-"*40)
            
            # Configure domain-specific settings
            from app.services.radiating.models.radiating_context import DomainContext
            domain_map = {
                'financial': DomainContext.FINANCIAL,
                'technology': DomainContext.TECHNOLOGY,
                'healthcare': DomainContext.HEALTHCARE,
                'general': DomainContext.GENERAL
            }
            context = RadiatingContext(
                original_query=query,
                query_domain=domain_map.get(domain.lower(), DomainContext.GENERAL),
                depth_limit=2,
                relevance_threshold=0.7
            )
            
            start_time = time.time()
            
            # Perform domain-aware search
            results = await self.radiating_service.search_with_context(
                query=query,
                context=context
            )
            
            elapsed = time.time() - start_time
            
            print(f"  ⏱️ Search time: {elapsed:.2f}s")
            print(f"  📊 Results found: {len(results.get('results', []))}")
            print(f"  🎯 Domain relevance: {results.get('domain_score', 0):.2%}")
            
            # Top domain-specific entities
            top_entities = results.get('domain_entities', [])[:3]
            if top_entities:
                print(f"  🌟 Key {domain} Entities:")
                for entity in top_entities:
                    print(f"     - {entity['name']} ({entity['relevance']:.2f})")
    
    async def example_continuous_learning(self):
        """Example 6: Continuous learning and feedback integration"""
        print("\n" + "="*60)
        print("EXAMPLE 6: Continuous Learning")
        print("="*60)
        
        query = "How do transformers work in NLP?"
        
        print(f"\n🎓 Query: '{query}'")
        print("-"*40)
        
        # Initial search
        print("\n1️⃣ Initial Search:")
        initial_results = await self.radiating_service.search(
            query=query,
            enable_radiating=True
        )
        
        print(f"  - Quality score: {initial_results.get('quality_score', 0):.2%}")
        print(f"  - Coverage: {initial_results.get('coverage_score', 0):.2%}")
        
        # Simulate user feedback
        feedback = {
            "query": query,
            "relevant_results": [0, 2, 4],  # Indices of relevant results
            "irrelevant_results": [1, 3],   # Indices of irrelevant results
            "missing_topics": ["attention mechanism", "positional encoding"],
            "rating": 4  # Out of 5
        }
        
        print("\n2️⃣ User Feedback:")
        print(f"  - Rating: {feedback['rating']}/5")
        print(f"  - Relevant results: {len(feedback['relevant_results'])}")
        print(f"  - Missing topics: {', '.join(feedback['missing_topics'])}")
        
        # Integrate feedback
        await self.radiating_service.integrate_feedback(feedback)
        
        # Improved search
        print("\n3️⃣ Improved Search (after feedback):")
        improved_results = await self.radiating_service.search(
            query=query,
            enable_radiating=True
        )
        
        print(f"  - Quality score: {improved_results.get('quality_score', 0):.2%}")
        print(f"  - Coverage: {improved_results.get('coverage_score', 0):.2%}")
        
        # Calculate improvement
        quality_improvement = (
            improved_results.get('quality_score', 0) - 
            initial_results.get('quality_score', 0)
        )
        
        print(f"\n📈 Improvement: +{quality_improvement:.2%}")
        
        if quality_improvement > 0:
            print("✅ System successfully learned from feedback!")
        else:
            print("ℹ️ More feedback needed for significant improvement")
    
    async def _standard_search(self, query: str) -> List[Dict]:
        """Simulate standard search without radiating"""
        # Simple keyword matching in Neo4j
        cypher_query = """
        MATCH (n:Entity)
        WHERE toLower(n.name) CONTAINS toLower($query)
           OR toLower(n.description) CONTAINS toLower($query)
        RETURN n
        LIMIT 10
        """
        
        results = self.neo4j_service.execute_query(
            cypher_query,
            parameters={"query": query}
        )
        
        return results if results else []
    
    async def _get_graph_visualization(self, entity: str) -> Dict:
        """Generate graph visualization data"""
        context = RadiatingContext(
            original_query=entity,
            depth_limit=3,
            relevance_threshold=0.5
        )
        
        # Get full graph structure
        graph_data = await self.traverser.get_graph_structure(
            start_entity=entity,
            context=context
        )
        
        # Calculate statistics
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])
        
        # Group nodes by level
        levels = {}
        for node in nodes:
            level = node.get('level', 0)
            if level not in levels:
                levels[level] = []
            levels[level].append(node)
        
        # Calculate connectivity
        node_degrees = {}
        for edge in edges:
            source = edge['source']
            target = edge['target']
            node_degrees[source] = node_degrees.get(source, 0) + 1
            node_degrees[target] = node_degrees.get(target, 0) + 1
        
        avg_connectivity = (
            sum(node_degrees.values()) / len(node_degrees)
            if node_degrees else 0
        )
        
        return {
            'nodes': nodes,
            'edges': edges,
            'levels': levels,
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'max_depth': max(levels.keys()) if levels else 0,
            'avg_connectivity': avg_connectivity,
            'level1': levels.get(1, [])
        }
    
    async def run_all_examples(self):
        """Run all demonstration examples"""
        print("\n" + "🚀 " + "="*56 + " 🚀")
        print("   UNIVERSAL RADIATING COVERAGE SYSTEM - DEMONSTRATION")
        print("🚀 " + "="*56 + " 🚀")
        
        examples = [
            ("Basic Traversal", self.example_basic_traversal),
            ("Query Expansion", self.example_query_expansion),
            ("Performance Comparison", self.example_performance_comparison),
            ("Visual Representation", self.example_visual_representation),
            ("Domain-Specific Search", self.example_domain_specific_search),
            ("Continuous Learning", self.example_continuous_learning)
        ]
        
        results_summary = []
        
        for name, example_func in examples:
            try:
                print(f"\n{'='*60}")
                print(f"Running: {name}")
                print('='*60)
                
                start_time = time.time()
                await example_func()
                elapsed = time.time() - start_time
                
                results_summary.append({
                    'example': name,
                    'status': 'Success',
                    'time': elapsed
                })
                
            except Exception as e:
                print(f"\n❌ Error in {name}: {str(e)}")
                results_summary.append({
                    'example': name,
                    'status': 'Failed',
                    'error': str(e)
                })
        
        # Print summary
        print("\n" + "="*60)
        print("DEMONSTRATION SUMMARY")
        print("="*60)
        
        for result in results_summary:
            status_icon = "✅" if result['status'] == 'Success' else "❌"
            print(f"{status_icon} {result['example']}: {result['status']}", end="")
            if 'time' in result:
                print(f" ({result['time']:.2f}s)")
            else:
                print(f" - {result.get('error', 'Unknown error')}")
        
        successful = sum(1 for r in results_summary if r['status'] == 'Success')
        total = len(results_summary)
        
        print(f"\n📊 Overall: {successful}/{total} examples completed successfully")
        
        if successful == total:
            print("🎉 All demonstrations completed successfully!")
        else:
            print(f"⚠️ {total - successful} demonstrations failed. Check logs for details.")

async def main():
    """Main entry point for the demonstration"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Universal Radiating Coverage System - Example Usage'
    )
    parser.add_argument(
        '--example',
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        help='Run specific example (1-6)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all examples'
    )
    
    args = parser.parse_args()
    
    demo = RadiatingSystemDemo()
    
    try:
        if args.all or (not args.example):
            await demo.run_all_examples()
        else:
            examples = {
                1: demo.example_basic_traversal,
                2: demo.example_query_expansion,
                3: demo.example_performance_comparison,
                4: demo.example_visual_representation,
                5: demo.example_domain_specific_search,
                6: demo.example_continuous_learning
            }
            
            example_func = examples.get(args.example)
            if example_func:
                print(f"\nRunning Example {args.example}")
                await example_func()
                print("\n✅ Example completed successfully!")
            else:
                print(f"❌ Invalid example number: {args.example}")
                
    except KeyboardInterrupt:
        print("\n\n⚠️ Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())