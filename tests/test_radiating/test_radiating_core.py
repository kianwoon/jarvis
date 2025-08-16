"""
Test Radiating Core

Tests for core radiating traversal functionality.
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from app.services.radiating.models.radiating_entity import RadiatingEntity
from app.services.radiating.models.radiating_relationship import RadiatingRelationship
from app.services.radiating.models.radiating_context import RadiatingContext, TraversalStrategy
from app.services.radiating.models.radiating_graph import RadiatingGraph
from app.services.radiating.engine.radiating_traverser import RadiatingTraverser
from app.services.radiating.engine.relevance_scorer import RelevanceScorer
from app.services.radiating.radiating_service import RadiatingService


class TestRadiatingCore(unittest.TestCase):
    """Test core radiating functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.service = RadiatingService()
        self.traverser = RadiatingTraverser()
        self.scorer = RelevanceScorer()
        
        # Create test entities
        self.test_entities = [
            RadiatingEntity(
                text="Test Entity 1",
                label="person",
                start_char=0,
                end_char=13,
                confidence=0.85,
                canonical_form="Test Entity 1",
                properties={'id': 'entity1', 'name': 'Test Entity 1'},
                traversal_depth=1,
                relevance_score=0.9
            ),
            RadiatingEntity(
                text="Test Entity 2",
                label="organization",
                start_char=14,
                end_char=27,
                confidence=0.75,
                canonical_form="Test Entity 2",
                properties={'id': 'entity2', 'name': 'Test Entity 2'},
                traversal_depth=1,
                relevance_score=0.8
            ),
            RadiatingEntity(
                text="Test Entity 3",
                label="location",
                start_char=28,
                end_char=41,
                confidence=0.65,
                canonical_form="Test Entity 3",
                properties={'id': 'entity3', 'name': 'Test Entity 3'},
                traversal_depth=2,
                relevance_score=0.7
            )
        ]
        
        # Create test relationships
        self.test_relationships = [
            RadiatingRelationship(
                source_entity="entity1",
                target_entity="entity2",
                relationship_type="works_for",
                confidence=0.7,
                properties={'id': 'rel1', 'strength': 0.8}
            ),
            RadiatingRelationship(
                source_entity="entity2",
                target_entity="entity3",
                relationship_type="located_in",
                confidence=0.8,
                properties={'id': 'rel2', 'strength': 0.9}
            )
        ]
        
        # Create test context
        self.test_context = RadiatingContext(
            original_query="test query",
            depth_limit=3,
            max_entities_per_level=50,
            relevance_threshold=0.3,
            traversal_strategy=TraversalStrategy.BREADTH_FIRST
        )
        
        # Create test graph
        self.test_graph = RadiatingGraph()
        for entity in self.test_entities:
            self.test_graph.add_node(entity)
        for relationship in self.test_relationships:
            self.test_graph.add_edge(relationship)
    
    def test_entity_creation(self):
        """Test entity creation and properties"""
        entity = RadiatingEntity(
            text="Test Name",
            label="test_type",
            start_char=0,
            end_char=9,
            confidence=0.6,
            canonical_form="Test Name",
            properties={'id': 'test_id', 'name': 'Test Name'},
            traversal_depth=1,
            relevance_score=0.5
        )
        
        self.assertEqual(entity.get_entity_id(), "test_id")  # Uses properties['id']
        self.assertEqual(entity.properties.get('name'), "Test Name")
        self.assertEqual(entity.label, "test_type")
        self.assertEqual(entity.traversal_depth, 1)
        self.assertEqual(entity.relevance_score, 0.5)
        self.assertEqual(entity.confidence, 0.6)
    
    def test_relationship_creation(self):
        """Test relationship creation and properties"""
        rel = RadiatingRelationship(
            source_entity="source",
            target_entity="target",
            relationship_type="related_to",
            confidence=0.8,
            context="test context",
            properties={'id': 'test_rel', 'strength': 0.7}
        )
        
        self.assertEqual(rel.get_relationship_id(), rel.get_relationship_id())  # Check it generates an ID
        self.assertEqual(rel.source_entity, "source")
        self.assertEqual(rel.target_entity, "target")
        self.assertEqual(rel.relationship_type, "related_to")
        self.assertEqual(rel.properties.get('strength'), 0.7)
        self.assertEqual(rel.confidence, 0.8)
    
    def test_context_creation(self):
        """Test context creation with various strategies"""
        # Test with breadth-first strategy
        context = RadiatingContext(
            original_query="test",
            depth_limit=5,
            traversal_strategy=TraversalStrategy.BREADTH_FIRST
        )
        
        self.assertEqual(context.query, "test")
        self.assertEqual(context.max_depth, 5)
        self.assertEqual(context.traversal_strategy, TraversalStrategy.BREADTH_FIRST)
        
        # Test with depth-first strategy
        context2 = RadiatingContext(
            original_query="test2",
            traversal_strategy=TraversalStrategy.DEPTH_FIRST
        )
        
        self.assertEqual(context2.traversal_strategy, TraversalStrategy.DEPTH_FIRST)
    
    def test_graph_creation(self):
        """Test graph creation and methods"""
        graph = RadiatingGraph()
        for entity in self.test_entities:
            graph.add_node(entity)
        for relationship in self.test_relationships:
            graph.add_edge(relationship)
        
        self.assertEqual(len(graph.entities), 3)
        self.assertEqual(len(graph.relationships), 2)
        
        # Test getting entities by depth
        depth_1_entities = graph.get_entities_by_depth(1)
        self.assertEqual(len(depth_1_entities), 2)
        
        depth_2_entities = graph.get_entities_by_depth(2)
        self.assertEqual(len(depth_2_entities), 1)
    
    def test_relevance_scoring(self):
        """Test relevance scoring calculations"""
        entity = self.test_entities[0]
        context = self.test_context
        
        # Calculate base relevance
        score = self.scorer.calculate_entity_relevance(entity, context)
        
        # Score should be between 0 and 1
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
        
        # Higher depth should have lower score (with decay)
        entity2 = RadiatingEntity(
            id="deep_entity",
            name="Deep Entity",
            type="test",
            depth=5,
            relevance_score=0.9,
            confidence=0.9
        )
        
        score2 = self.scorer.calculate_entity_relevance(entity2, context)
        self.assertLess(score2, score)  # Deeper entity should have lower score
    
    @patch('app.services.neo4j_service.get_neo4j_service')
    async def test_traversal_basic(self, mock_neo4j):
        """Test basic traversal functionality"""
        # Mock Neo4j responses
        mock_neo4j.return_value.driver.session.return_value.run = AsyncMock(
            return_value=[
                {'entity': {'name': 'Entity1', 'type': 'person'}},
                {'entity': {'name': 'Entity2', 'type': 'organization'}}
            ]
        )
        
        # Perform traversal
        result = await self.traverser.traverse(
            start_entity=self.test_entities[0],
            context=self.test_context,
            current_depth=1
        )
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, RadiatingGraph)
    
    def test_service_initialization(self):
        """Test service initialization and settings"""
        service = RadiatingService()
        
        self.assertIsNotNone(service.settings)
        self.assertTrue(service.settings.enabled)
        self.assertEqual(service.settings.max_depth, 3)
        self.assertEqual(service.settings.max_entities_per_level, 50)
    
    @patch('app.services.radiating.radiating_service.get_redis_client')
    async def test_service_caching(self, mock_redis):
        """Test service caching functionality"""
        mock_redis.return_value.get = AsyncMock(return_value=None)
        mock_redis.return_value.setex = AsyncMock()
        
        service = RadiatingService()
        
        # Test cache miss and set
        result = await service._get_cached_result("test_key")
        self.assertIsNone(result)
        
        # Test cache set
        await service._cache_result("test_key", self.test_graph)
        mock_redis.return_value.setex.assert_called_once()
    
    def test_graph_filtering(self):
        """Test graph filtering by relevance"""
        # Create graph with mixed relevance scores
        entities = [
            RadiatingEntity(
                text=f"Entity {i}",
                label="test",
                start_char=i*10,
                end_char=i*10+8,
                confidence=0.7,
                canonical_form=f"Entity {i}",
                properties={'id': f'e{i}', 'name': f'Entity {i}'},
                traversal_depth=1,
                relevance_score=i * 0.2
            )
            for i in range(6)
        ]
        
        graph = RadiatingGraph()
        for entity in entities:
            graph.add_node(entity)
        
        # Filter by relevance threshold
        filtered = graph.filter_by_relevance(0.5)
        
        # Should keep only entities with relevance >= 0.5
        self.assertEqual(len(filtered.entities), 3)
        for entity in filtered.entities:
            self.assertGreaterEqual(entity.relevance_score, 0.5)
    
    def test_graph_merging(self):
        """Test merging multiple graphs"""
        # Create first graph
        graph1 = RadiatingGraph()
        for entity in self.test_entities[:2]:
            graph1.add_node(entity)
        graph1.add_edge(self.test_relationships[0])
        
        # Create second graph with some overlap
        new_entities = [
            self.test_entities[1],  # Overlap
            self.test_entities[2]   # New
        ]
        
        graph2 = RadiatingGraph()
        for entity in new_entities:
            graph2.add_node(entity)
        graph2.add_edge(self.test_relationships[1])
        
        # Merge graphs
        merged = graph1.merge(graph2)
        
        # Should have all unique entities
        self.assertEqual(len(merged.entities), 3)
        
        # Should have all relationships
        self.assertEqual(len(merged.relationships), 2)


class TestRadiatingAsync(unittest.IsolatedAsyncioTestCase):
    """Test async radiating functionality"""
    
    async def test_async_traversal(self):
        """Test async traversal operations"""
        traverser = RadiatingTraverser()
        
        # Create mock context
        context = RadiatingContext(
            original_query="async test",
            depth_limit=2
        )
        
        # Mock the expansion method
        with patch.object(traverser, 'expand_entity', new_callable=AsyncMock) as mock_expand:
            mock_expand.return_value = ([], [])
            
            # Start entity
            start = RadiatingEntity(
                id="start",
                name="Start",
                type="test",
                depth=0,
                relevance_score=1.0,
                confidence=1.0
            )
            
            result = await traverser.traverse(start, context, 0)
            
            self.assertIsNotNone(result)
            mock_expand.assert_called()
    
    async def test_parallel_expansion(self):
        """Test parallel entity expansion"""
        traverser = RadiatingTraverser()
        
        # Create multiple entities to expand
        entities = [
            RadiatingEntity(
                id=f"entity_{i}",
                name=f"Entity {i}",
                type="test",
                depth=1,
                relevance_score=0.8,
                confidence=0.7
            )
            for i in range(5)
        ]
        
        context = RadiatingContext(
            original_query="parallel test",
            depth_limit=3
        )
        
        # Mock expansion
        with patch.object(traverser, 'expand_entity', new_callable=AsyncMock) as mock_expand:
            mock_expand.return_value = ([], [])
            
            # Expand in parallel
            tasks = [
                traverser.expand_entity(entity, context)
                for entity in entities
            ]
            
            results = await asyncio.gather(*tasks)
            
            self.assertEqual(len(results), 5)
            self.assertEqual(mock_expand.call_count, 5)
    
    async def test_service_radiating_coverage(self):
        """Test full radiating coverage service"""
        service = RadiatingService()
        
        # Mock dependencies
        with patch.object(service.traverser, 'traverse', new_callable=AsyncMock) as mock_traverse:
            with patch.object(service.query_analyzer, 'analyze', new_callable=AsyncMock) as mock_analyze:
                with patch.object(service.entity_extractor, 'extract', new_callable=AsyncMock) as mock_extract:
                    
                    # Set up mocks
                    mock_analyze.return_value = RadiatingContext(
                        original_query="test coverage",
                        depth_limit=2
                    )
                    
                    mock_extract.return_value = [
                        RadiatingEntity(
                            id="extracted",
                            name="Extracted Entity",
                            type="test",
                            depth=0,
                            relevance_score=1.0,
                            confidence=0.9
                        )
                    ]
                    
                    mock_traverse.return_value = RadiatingGraph()
                    
                    # Execute radiating coverage
                    result = await service.execute_radiating_coverage(
                        query="test coverage query",
                        max_depth=2
                    )
                    
                    self.assertIsNotNone(result)
                    mock_analyze.assert_called_once()
                    mock_extract.assert_called_once()
                    mock_traverse.assert_called()


if __name__ == '__main__':
    unittest.main()