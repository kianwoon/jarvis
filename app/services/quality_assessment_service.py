"""
Knowledge Graph Quality Assessment Service

Provides comprehensive quality metrics and analysis for knowledge graph
to measure improvements and identify issues for debugging and optimization.
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

from app.services.neo4j_service import get_neo4j_service

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for knowledge graph"""
    # Connectivity metrics
    total_entities: int
    total_relationships: int
    isolated_entities: int
    connectivity_ratio: float
    average_degree: float
    
    # Relationship quality metrics
    generic_relationship_ratio: float
    semantic_relationship_ratio: float
    high_confidence_relationships: int
    low_confidence_relationships: int
    
    # Entity quality metrics
    entity_type_distribution: Dict[str, int]
    entities_with_attributes: int
    cross_document_entities: int
    
    # Specific quality issues
    potential_naming_issues: List[Dict[str, Any]]
    questionable_relationships: List[Dict[str, Any]]
    classification_errors: List[Dict[str, Any]]
    
    # Overall scores
    connectivity_score: float  # 0-1
    relationship_quality_score: float  # 0-1
    entity_quality_score: float  # 0-1
    overall_quality_score: float  # 0-1

class KnowledgeGraphQualityAssessment:
    """Service for comprehensive knowledge graph quality assessment"""
    
    def __init__(self):
        self.neo4j_service = get_neo4j_service()
        
        # Define quality thresholds and patterns
        self.generic_relationship_types = {
            'MENTIONED_WITH', 'CONTEXTUALLY_RELATED', 'RELATED_TO', 
            'ASSOCIATED_WITH', 'DOCUMENT_RELATED'
        }
        
        self.semantic_relationship_types = {
            'LEADS', 'WORKS_FOR', 'OWNS', 'PARTNERS_WITH', 'USES', 
            'DEVELOPS', 'BASED_ON', 'LOCATED_IN', 'OPERATES_IN'
        }
        
        # Patterns for detecting naming issues
        self.naming_issue_patterns = [
            {'pattern': r'^[A-Z]{2,}$', 'issue': 'likely_abbreviation_as_entity'},
            {'pattern': r'[,;]', 'issue': 'multiple_entities_in_name'},
            {'pattern': r'\d+$', 'issue': 'number_as_entity_name'},
            {'pattern': r'^(and|or|the|a|an)$', 'issue': 'stop_word_as_entity'}
        ]
        
        # Expected entity types for common terms
        self.type_expectations = {
            'sql': 'CONCEPT',
            'database': 'CONCEPT', 
            'technology': 'CONCEPT',
            'platform': 'CONCEPT',
            'system': 'CONCEPT'
        }
    
    async def assess_graph_quality(self) -> QualityMetrics:
        """Perform comprehensive quality assessment of the knowledge graph"""
        if not self.neo4j_service.is_enabled():
            logger.error("Neo4j not enabled, cannot assess quality")
            return self._create_empty_metrics()
        
        try:
            logger.info("🔍 Starting comprehensive knowledge graph quality assessment...")
            
            # Get basic graph statistics
            basic_stats = await self._get_basic_statistics()
            
            # Analyze connectivity
            connectivity_metrics = await self._analyze_connectivity()
            
            # Assess relationship quality
            relationship_quality = await self._assess_relationship_quality()
            
            # Assess entity quality
            entity_quality = await self._assess_entity_quality()
            
            # Identify specific issues
            naming_issues = await self._identify_naming_issues()
            questionable_rels = await self._identify_questionable_relationships()
            classification_errors = await self._identify_classification_errors()
            
            # Calculate quality scores
            scores = self._calculate_quality_scores(
                connectivity_metrics, relationship_quality, entity_quality
            )
            
            metrics = QualityMetrics(
                # Basic stats
                total_entities=basic_stats['total_entities'],
                total_relationships=basic_stats['total_relationships'],
                isolated_entities=connectivity_metrics['isolated_count'],
                connectivity_ratio=connectivity_metrics['connectivity_ratio'],
                average_degree=connectivity_metrics['average_degree'],
                
                # Relationship quality
                generic_relationship_ratio=relationship_quality['generic_ratio'],
                semantic_relationship_ratio=relationship_quality['semantic_ratio'],
                high_confidence_relationships=relationship_quality['high_confidence_count'],
                low_confidence_relationships=relationship_quality['low_confidence_count'],
                
                # Entity quality
                entity_type_distribution=entity_quality['type_distribution'],
                entities_with_attributes=entity_quality['entities_with_attributes'],
                cross_document_entities=entity_quality['cross_document_count'],
                
                # Issues
                potential_naming_issues=naming_issues,
                questionable_relationships=questionable_rels,
                classification_errors=classification_errors,
                
                # Scores
                connectivity_score=scores['connectivity'],
                relationship_quality_score=scores['relationship_quality'],
                entity_quality_score=scores['entity_quality'],
                overall_quality_score=scores['overall']
            )
            
            logger.info(f"🔍 Quality assessment complete - Overall score: {scores['overall']:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return self._create_empty_metrics()
    
    async def _get_basic_statistics(self) -> Dict[str, Any]:
        """Get basic graph statistics"""
        entities_query = "MATCH (n) RETURN count(n) as total_entities"
        relationships_query = "MATCH ()-[r]->() RETURN count(r) as total_relationships"
        
        entity_result = self.neo4j_service.execute_cypher(entities_query)
        rel_result = self.neo4j_service.execute_cypher(relationships_query)
        
        return {
            'total_entities': entity_result[0]['total_entities'] if entity_result else 0,
            'total_relationships': rel_result[0]['total_relationships'] if rel_result else 0
        }
    
    async def _analyze_connectivity(self) -> Dict[str, Any]:
        """Analyze graph connectivity metrics"""
        # Get isolated nodes
        isolated_query = """
        MATCH (n)
        WHERE NOT (n)-[]-()
        RETURN count(n) as isolated_count
        """
        isolated_result = self.neo4j_service.execute_cypher(isolated_query)
        isolated_count = isolated_result[0]['isolated_count'] if isolated_result else 0
        
        # Get total nodes for ratio calculation
        total_query = "MATCH (n) RETURN count(n) as total"
        total_result = self.neo4j_service.execute_cypher(total_query)
        total_nodes = total_result[0]['total'] if total_result else 0
        
        # Calculate average degree
        degree_query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(r) as degree
        RETURN avg(degree) as avg_degree
        """
        degree_result = self.neo4j_service.execute_cypher(degree_query)
        avg_degree = degree_result[0]['avg_degree'] if degree_result else 0
        
        connectivity_ratio = (total_nodes - isolated_count) / total_nodes if total_nodes > 0 else 0
        
        return {
            'isolated_count': isolated_count,
            'total_nodes': total_nodes,
            'connectivity_ratio': connectivity_ratio,
            'average_degree': float(avg_degree) if avg_degree else 0.0
        }
    
    async def _assess_relationship_quality(self) -> Dict[str, Any]:
        """Assess quality of relationships"""
        # Get relationship type distribution
        rel_types_query = """
        MATCH ()-[r]->()
        RETURN type(r) as rel_type, count(r) as count, avg(r.confidence) as avg_confidence
        ORDER BY count DESC
        """
        rel_types_result = self.neo4j_service.execute_cypher(rel_types_query)
        
        total_relationships = sum(row['count'] for row in rel_types_result)
        generic_count = sum(
            row['count'] for row in rel_types_result 
            if row['rel_type'] in self.generic_relationship_types
        )
        semantic_count = sum(
            row['count'] for row in rel_types_result
            if row['rel_type'] in self.semantic_relationship_types
        )
        
        # Get confidence distribution
        confidence_query = """
        MATCH ()-[r]->()
        WHERE r.confidence IS NOT NULL
        RETURN 
            count(CASE WHEN r.confidence >= 0.7 THEN 1 END) as high_confidence,
            count(CASE WHEN r.confidence < 0.5 THEN 1 END) as low_confidence
        """
        confidence_result = self.neo4j_service.execute_cypher(confidence_query)
        confidence_stats = confidence_result[0] if confidence_result else {'high_confidence': 0, 'low_confidence': 0}
        
        return {
            'type_distribution': {row['rel_type']: row['count'] for row in rel_types_result},
            'generic_ratio': generic_count / total_relationships if total_relationships > 0 else 0,
            'semantic_ratio': semantic_count / total_relationships if total_relationships > 0 else 0,
            'high_confidence_count': confidence_stats['high_confidence'],
            'low_confidence_count': confidence_stats['low_confidence']
        }
    
    async def _assess_entity_quality(self) -> Dict[str, Any]:
        """Assess quality of entities"""
        # Get entity type distribution
        entity_types_query = """
        MATCH (n)
        RETURN labels(n)[0] as entity_type, count(n) as count
        ORDER BY count DESC
        """
        entity_types_result = self.neo4j_service.execute_cypher(entity_types_query)
        
        # Count entities with additional attributes
        attributes_query = """
        MATCH (n)
        WHERE size(keys(n)) > 5  // More than basic required properties
        RETURN count(n) as entities_with_attributes
        """
        attributes_result = self.neo4j_service.execute_cypher(attributes_query)
        entities_with_attributes = attributes_result[0]['entities_with_attributes'] if attributes_result else 0
        
        # Count cross-document entities
        cross_doc_query = """
        MATCH (n)
        WHERE n.document_count > 1 OR size(n.cross_document_references) > 1
        RETURN count(n) as cross_document_count
        """
        cross_doc_result = self.neo4j_service.execute_cypher(cross_doc_query)
        cross_document_count = cross_doc_result[0]['cross_document_count'] if cross_doc_result else 0
        
        return {
            'type_distribution': {row['entity_type']: row['count'] for row in entity_types_result},
            'entities_with_attributes': entities_with_attributes,
            'cross_document_count': cross_document_count
        }
    
    async def _identify_naming_issues(self) -> List[Dict[str, Any]]:
        """Identify potential naming issues with entities"""
        naming_issues = []
        
        # Get all entity names for analysis
        names_query = """
        MATCH (n)
        RETURN n.id as id, n.name as name, labels(n)[0] as type
        """
        names_result = self.neo4j_service.execute_cypher(names_query)
        
        for entity in names_result:
            name = entity['name']
            entity_type = entity['type']
            entity_id = entity['id']
            
            # Check for various naming issues
            import re
            
            # Multiple entities in one name (contains commas/semicolons)
            if ',' in name or ';' in name:
                naming_issues.append({
                    'entity_id': entity_id,
                    'name': name,
                    'type': entity_type,
                    'issue': 'multiple_entities_in_name',
                    'severity': 'high'
                })
            
            # Very short names that might be abbreviations
            if len(name) <= 2:
                naming_issues.append({
                    'entity_id': entity_id,
                    'name': name,
                    'type': entity_type,
                    'issue': 'very_short_name',
                    'severity': 'medium'
                })
            
            # All caps (might be abbreviation)
            if name.isupper() and len(name) > 2:
                naming_issues.append({
                    'entity_id': entity_id,
                    'name': name,
                    'type': entity_type,
                    'issue': 'all_caps_name',
                    'severity': 'low'
                })
        
        return naming_issues
    
    async def _identify_questionable_relationships(self) -> List[Dict[str, Any]]:
        """Identify relationships that seem questionable or nonsensical"""
        questionable = []
        
        # Look for relationships with very low confidence
        low_confidence_query = """
        MATCH (a)-[r]->(b)
        WHERE r.confidence < 0.4
        RETURN a.name as source, type(r) as rel_type, b.name as target, 
               r.confidence as confidence, r.context as context
        LIMIT 20
        """
        low_conf_result = self.neo4j_service.execute_cypher(low_confidence_query)
        
        for rel in low_conf_result:
            questionable.append({
                'source': rel['source'],
                'relationship': rel['rel_type'],
                'target': rel['target'],
                'confidence': rel['confidence'],
                'context': rel['context'],
                'issue': 'low_confidence',
                'severity': 'medium'
            })
        
        # Look for type mismatches (e.g., PERSON LOCATED_IN PERSON)
        type_mismatch_query = """
        MATCH (a)-[r:LOCATED_IN]->(b)
        WHERE labels(a)[0] = labels(b)[0] AND labels(a)[0] <> 'LOCATION'
        RETURN a.name as source, labels(a)[0] as source_type, b.name as target, labels(b)[0] as target_type
        LIMIT 10
        """
        type_mismatch_result = self.neo4j_service.execute_cypher(type_mismatch_query)
        
        for rel in type_mismatch_result:
            questionable.append({
                'source': rel['source'],
                'source_type': rel['source_type'],
                'relationship': 'LOCATED_IN',
                'target': rel['target'],
                'target_type': rel['target_type'],
                'issue': 'type_mismatch_for_location',
                'severity': 'high'
            })
        
        return questionable
    
    async def _identify_classification_errors(self) -> List[Dict[str, Any]]:
        """Identify potential entity classification errors"""
        classification_errors = []
        
        # Check for common misclassifications
        for term, expected_type in self.type_expectations.items():
            query = f"""
            MATCH (n)
            WHERE toLower(n.name) CONTAINS '{term}' AND labels(n)[0] <> '{expected_type}'
            RETURN n.id as id, n.name as name, labels(n)[0] as current_type
            LIMIT 5
            """
            
            result = self.neo4j_service.execute_cypher(query)
            for entity in result:
                classification_errors.append({
                    'entity_id': entity['id'],
                    'name': entity['name'],
                    'current_type': entity['current_type'],
                    'expected_type': expected_type,
                    'issue': f'should_be_{expected_type.lower()}',
                    'severity': 'medium'
                })
        
        return classification_errors
    
    def _calculate_quality_scores(self, connectivity_metrics: Dict, 
                                relationship_quality: Dict, entity_quality: Dict) -> Dict[str, float]:
        """Calculate overall quality scores with isolation penalty"""
        
        # Connectivity score (0-1) with heavy penalty for isolated nodes
        base_connectivity = connectivity_metrics['connectivity_ratio']
        isolation_penalty = connectivity_metrics['isolated_count'] / max(connectivity_metrics['total_nodes'], 1) * 0.5
        connectivity_score = min(1.0, base_connectivity - isolation_penalty + 
                               (connectivity_metrics['average_degree'] / 10))
        
        # Relationship quality score (0-1) 
        rel_quality_score = min(1.0, relationship_quality['semantic_ratio'] * 2)
        
        # Entity quality score (0-1) - based on distribution and attributes
        # Add penalty for isolated nodes in entity quality too
        base_entity_score = min(1.0, 
            len(entity_quality['type_distribution']) / 10 +  # Diversity bonus
            entity_quality['entities_with_attributes'] / max(sum(entity_quality['type_distribution'].values()), 1)
        )
        entity_quality_score = max(0.0, base_entity_score - isolation_penalty)
        
        # Overall score (weighted average) - isolation affects everything
        overall_score = (
            connectivity_score * 0.3 + 
            rel_quality_score * 0.4 + 
            entity_quality_score * 0.3
        )
        
        return {
            'connectivity': connectivity_score,
            'relationship_quality': rel_quality_score, 
            'entity_quality': entity_quality_score,
            'overall': overall_score
        }
    
    def _create_empty_metrics(self) -> QualityMetrics:
        """Create empty metrics for error cases"""
        return QualityMetrics(
            total_entities=0, total_relationships=0, isolated_entities=0,
            connectivity_ratio=0.0, average_degree=0.0,
            generic_relationship_ratio=0.0, semantic_relationship_ratio=0.0,
            high_confidence_relationships=0, low_confidence_relationships=0,
            entity_type_distribution={}, entities_with_attributes=0, cross_document_entities=0,
            potential_naming_issues=[], questionable_relationships=[], classification_errors=[],
            connectivity_score=0.0, relationship_quality_score=0.0, 
            entity_quality_score=0.0, overall_quality_score=0.0
        )

# Singleton instance
_quality_service: Optional[KnowledgeGraphQualityAssessment] = None

def get_quality_assessment_service() -> KnowledgeGraphQualityAssessment:
    """Get or create quality assessment service singleton"""
    global _quality_service
    if _quality_service is None:
        _quality_service = KnowledgeGraphQualityAssessment()
    return _quality_service