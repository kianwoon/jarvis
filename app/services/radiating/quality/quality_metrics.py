"""
Quality Metrics

Calculates precision, recall, coverage metrics, diversity scores,
and generates quality reports for the radiating system.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import precision_score, recall_score, f1_score
import asyncio
import json

from app.services.radiating.models.radiating_entity import RadiatingEntity
from app.services.radiating.models.radiating_relationship import RadiatingRelationship
from app.services.radiating.models.radiating_graph import RadiatingGraph
from app.core.redis_client import get_redis_client
from app.core.db import get_db_session

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of quality metrics"""
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    COVERAGE = "coverage"
    DIVERSITY = "diversity"
    NOVELTY = "novelty"
    COHERENCE = "coherence"
    COMPLETENESS = "completeness"


@dataclass
class QualityScore:
    """Overall quality score"""
    overall_score: float  # 0-1 scale
    precision: float
    recall: float
    f1: float
    coverage: float
    diversity: float
    novelty: float
    coherence: float
    completeness: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricReport:
    """Detailed metric report"""
    metric_type: MetricType
    value: float
    trend: str  # increasing, decreasing, stable
    comparison_period: str
    details: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CoverageAnalysis:
    """Coverage analysis results"""
    entity_coverage: float  # Percentage of relevant entities found
    relationship_coverage: float  # Percentage of relationships discovered
    depth_coverage: float  # How well depth levels are covered
    type_coverage: Dict[str, float]  # Coverage by entity type
    gaps: List[str]  # Identified coverage gaps


@dataclass
class DiversityAnalysis:
    """Diversity analysis results"""
    entity_diversity: float  # Shannon entropy of entity types
    relationship_diversity: float  # Shannon entropy of relationship types
    source_diversity: float  # Diversity of information sources
    temporal_diversity: float  # Diversity across time periods
    distribution: Dict[str, Dict[str, int]]  # Distribution details


class QualityMetrics:
    """
    Calculates and tracks quality metrics for the radiating system,
    providing insights into performance and areas for improvement.
    """
    
    # Metric weights for overall score
    METRIC_WEIGHTS = {
        'precision': 0.2,
        'recall': 0.2,
        'coverage': 0.15,
        'diversity': 0.15,
        'novelty': 0.1,
        'coherence': 0.1,
        'completeness': 0.1
    }
    
    # Quality thresholds
    QUALITY_THRESHOLDS = {
        'excellent': 0.9,
        'good': 0.75,
        'acceptable': 0.6,
        'poor': 0.4
    }
    
    # Metric targets
    METRIC_TARGETS = {
        MetricType.PRECISION: 0.85,
        MetricType.RECALL: 0.80,
        MetricType.F1_SCORE: 0.82,
        MetricType.COVERAGE: 0.75,
        MetricType.DIVERSITY: 0.70,
        MetricType.NOVELTY: 0.30,
        MetricType.COHERENCE: 0.85,
        MetricType.COMPLETENESS: 0.80
    }
    
    def __init__(self):
        """Initialize QualityMetrics"""
        self.redis_client = get_redis_client()
        
        # Metric history
        self.metric_history: Dict[MetricType, List[float]] = {
            metric: [] for metric in MetricType
        }
        
        # Ground truth cache for evaluation
        self.ground_truth_cache: Dict[str, Set[str]] = {}
        
        # Statistics
        self.stats = {
            'evaluations_performed': 0,
            'reports_generated': 0,
            'average_quality_score': 0.0,
            'best_quality_score': 0.0,
            'worst_quality_score': 1.0
        }
        
        # Start background tasks
        asyncio.create_task(self._metric_tracker())
    
    async def calculate_precision_recall(
        self,
        predicted_entities: List[RadiatingEntity],
        ground_truth_entities: Set[str],
        predicted_relationships: List[RadiatingRelationship],
        ground_truth_relationships: Set[Tuple[str, str, str]]
    ) -> Tuple[float, float, float]:
        """
        Calculate precision, recall, and F1 score
        
        Args:
            predicted_entities: Entities found by the system
            ground_truth_entities: Known relevant entities
            predicted_relationships: Relationships found
            ground_truth_relationships: Known relationships
            
        Returns:
            Tuple of (precision, recall, f1_score)
        """
        # Entity precision and recall
        predicted_entity_names = {e.name.lower() for e in predicted_entities}
        ground_truth_lower = {e.lower() for e in ground_truth_entities}
        
        entity_tp = len(predicted_entity_names & ground_truth_lower)
        entity_fp = len(predicted_entity_names - ground_truth_lower)
        entity_fn = len(ground_truth_lower - predicted_entity_names)
        
        entity_precision = entity_tp / (entity_tp + entity_fp) if (entity_tp + entity_fp) > 0 else 0
        entity_recall = entity_tp / (entity_tp + entity_fn) if (entity_tp + entity_fn) > 0 else 0
        
        # Relationship precision and recall
        predicted_rels = {
            (r.source_id, r.type, r.target_id)
            for r in predicted_relationships
        }
        
        rel_tp = len(predicted_rels & ground_truth_relationships)
        rel_fp = len(predicted_rels - ground_truth_relationships)
        rel_fn = len(ground_truth_relationships - predicted_rels)
        
        rel_precision = rel_tp / (rel_tp + rel_fp) if (rel_tp + rel_fp) > 0 else 0
        rel_recall = rel_tp / (rel_tp + rel_fn) if (rel_tp + rel_fn) > 0 else 0
        
        # Combine entity and relationship metrics
        precision = (entity_precision + rel_precision) / 2
        recall = (entity_recall + rel_recall) / 2
        
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Update history
        self.metric_history[MetricType.PRECISION].append(precision)
        self.metric_history[MetricType.RECALL].append(recall)
        self.metric_history[MetricType.F1_SCORE].append(f1)
        
        return precision, recall, f1
    
    async def calculate_coverage_metrics(
        self,
        graph: RadiatingGraph,
        context: Dict[str, Any]
    ) -> CoverageAnalysis:
        """
        Calculate coverage metrics
        
        Args:
            graph: Radiating graph
            context: Query context
            
        Returns:
            CoverageAnalysis results
        """
        # Entity coverage by depth
        depth_coverage = {}
        max_depth = context.get('max_depth', 3)
        
        for depth in range(1, max_depth + 1):
            entities_at_depth = [e for e in graph.entities if e.depth == depth]
            expected_entities = context.get(f'expected_at_depth_{depth}', 10)
            depth_coverage[depth] = min(1.0, len(entities_at_depth) / expected_entities)
        
        avg_depth_coverage = np.mean(list(depth_coverage.values())) if depth_coverage else 0
        
        # Type coverage
        type_coverage = {}
        entity_types = set(e.type for e in graph.entities)
        expected_types = set(context.get('expected_types', ['person', 'organization', 'location']))
        
        for etype in expected_types:
            type_entities = [e for e in graph.entities if e.type == etype]
            type_coverage[etype] = min(1.0, len(type_entities) / max(1, len(graph.entities) / len(expected_types)))
        
        # Relationship coverage
        relationship_types = set(r.type for r in graph.relationships)
        expected_rel_types = set(context.get('expected_relationship_types', ['related_to']))
        rel_coverage = len(relationship_types & expected_rel_types) / len(expected_rel_types) if expected_rel_types else 1.0
        
        # Overall entity coverage
        total_entities_found = len(graph.entities)
        expected_entities = context.get('expected_entities', 50)
        entity_coverage = min(1.0, total_entities_found / expected_entities)
        
        # Identify gaps
        gaps = []
        
        # Check for missing entity types
        missing_types = expected_types - entity_types
        if missing_types:
            gaps.append(f"Missing entity types: {', '.join(missing_types)}")
        
        # Check for sparse depths
        for depth, coverage in depth_coverage.items():
            if coverage < 0.5:
                gaps.append(f"Low coverage at depth {depth}: {coverage:.1%}")
        
        # Check for missing relationship types
        missing_rel_types = expected_rel_types - relationship_types
        if missing_rel_types:
            gaps.append(f"Missing relationship types: {', '.join(missing_rel_types)}")
        
        # Update history
        overall_coverage = (entity_coverage + rel_coverage + avg_depth_coverage) / 3
        self.metric_history[MetricType.COVERAGE].append(overall_coverage)
        
        return CoverageAnalysis(
            entity_coverage=entity_coverage,
            relationship_coverage=rel_coverage,
            depth_coverage=avg_depth_coverage,
            type_coverage=type_coverage,
            gaps=gaps
        )
    
    async def calculate_diversity_scores(
        self,
        graph: RadiatingGraph
    ) -> DiversityAnalysis:
        """
        Calculate diversity scores using entropy
        
        Args:
            graph: Radiating graph
            
        Returns:
            DiversityAnalysis results
        """
        # Entity type diversity
        entity_types = [e.type for e in graph.entities]
        entity_type_counts = {}
        for etype in entity_types:
            entity_type_counts[etype] = entity_type_counts.get(etype, 0) + 1
        
        if entity_type_counts:
            entity_probs = np.array(list(entity_type_counts.values())) / len(entity_types)
            entity_diversity = entropy(entity_probs) / np.log(len(entity_type_counts))  # Normalize
        else:
            entity_diversity = 0
        
        # Relationship type diversity
        rel_types = [r.type for r in graph.relationships]
        rel_type_counts = {}
        for rtype in rel_types:
            rel_type_counts[rtype] = rel_type_counts.get(rtype, 0) + 1
        
        if rel_type_counts:
            rel_probs = np.array(list(rel_type_counts.values())) / len(rel_types)
            rel_diversity = entropy(rel_probs) / np.log(len(rel_type_counts))  # Normalize
        else:
            rel_diversity = 0
        
        # Source diversity (based on entity metadata)
        sources = []
        for entity in graph.entities:
            if entity.metadata and 'source' in entity.metadata:
                sources.append(entity.metadata['source'])
        
        source_counts = {}
        for source in sources:
            source_counts[source] = source_counts.get(source, 0) + 1
        
        if source_counts:
            source_probs = np.array(list(source_counts.values())) / len(sources)
            source_diversity = entropy(source_probs) / np.log(len(source_counts))  # Normalize
        else:
            source_diversity = 0.5  # Default if no source info
        
        # Temporal diversity (based on timestamps if available)
        temporal_diversity = self._calculate_temporal_diversity(graph)
        
        # Overall diversity
        overall_diversity = np.mean([
            entity_diversity,
            rel_diversity,
            source_diversity,
            temporal_diversity
        ])
        
        # Update history
        self.metric_history[MetricType.DIVERSITY].append(overall_diversity)
        
        return DiversityAnalysis(
            entity_diversity=entity_diversity,
            relationship_diversity=rel_diversity,
            source_diversity=source_diversity,
            temporal_diversity=temporal_diversity,
            distribution={
                'entity_types': entity_type_counts,
                'relationship_types': rel_type_counts,
                'sources': source_counts
            }
        )
    
    def _calculate_temporal_diversity(self, graph: RadiatingGraph) -> float:
        """Calculate temporal diversity of entities"""
        timestamps = []
        
        for entity in graph.entities:
            if entity.metadata and 'timestamp' in entity.metadata:
                timestamps.append(entity.metadata['timestamp'])
        
        if len(timestamps) < 2:
            return 0.5  # Default if insufficient temporal data
        
        # Convert to numerical values (hours from earliest)
        timestamps.sort()
        earliest = timestamps[0]
        time_diffs = [(t - earliest).total_seconds() / 3600 for t in timestamps]
        
        # Calculate distribution across time buckets
        num_buckets = min(10, len(timestamps))
        hist, _ = np.histogram(time_diffs, bins=num_buckets)
        
        if hist.sum() > 0:
            probs = hist / hist.sum()
            # Remove zero probabilities
            probs = probs[probs > 0]
            temporal_diversity = entropy(probs) / np.log(len(probs)) if len(probs) > 1 else 0
        else:
            temporal_diversity = 0
        
        return temporal_diversity
    
    async def calculate_novelty_score(
        self,
        graph: RadiatingGraph,
        historical_entities: Set[str]
    ) -> float:
        """
        Calculate novelty score (percentage of new discoveries)
        
        Args:
            graph: Current radiating graph
            historical_entities: Previously known entities
            
        Returns:
            Novelty score (0-1)
        """
        current_entities = {e.name.lower() for e in graph.entities}
        new_entities = current_entities - historical_entities
        
        novelty = len(new_entities) / len(current_entities) if current_entities else 0
        
        # Update history
        self.metric_history[MetricType.NOVELTY].append(novelty)
        
        return novelty
    
    async def calculate_coherence_score(
        self,
        graph: RadiatingGraph
    ) -> float:
        """
        Calculate coherence score (how well connected the graph is)
        
        Args:
            graph: Radiating graph
            
        Returns:
            Coherence score (0-1)
        """
        if not graph.entities:
            return 0
        
        # Calculate connectivity ratio
        num_entities = len(graph.entities)
        num_relationships = len(graph.relationships)
        
        # Maximum possible relationships (fully connected)
        max_relationships = num_entities * (num_entities - 1) / 2
        
        if max_relationships > 0:
            connectivity = min(1.0, num_relationships / (max_relationships * 0.1))  # 10% connectivity is good
        else:
            connectivity = 0
        
        # Check for isolated entities
        connected_entities = set()
        for rel in graph.relationships:
            connected_entities.add(rel.source_id)
            connected_entities.add(rel.target_id)
        
        isolation_penalty = (num_entities - len(connected_entities)) / num_entities if num_entities > 0 else 0
        
        # Calculate average path length (simplified)
        avg_depth = np.mean([e.depth for e in graph.entities]) if graph.entities else 0
        depth_coherence = 1 / (1 + abs(avg_depth - 2))  # Depth 2 is optimal
        
        # Combine metrics
        coherence = (
            connectivity * 0.5 +
            (1 - isolation_penalty) * 0.3 +
            depth_coherence * 0.2
        )
        
        # Update history
        self.metric_history[MetricType.COHERENCE].append(coherence)
        
        return coherence
    
    async def calculate_completeness_score(
        self,
        graph: RadiatingGraph,
        min_entities: int = 10,
        min_relationships: int = 5
    ) -> float:
        """
        Calculate completeness score
        
        Args:
            graph: Radiating graph
            min_entities: Minimum expected entities
            min_relationships: Minimum expected relationships
            
        Returns:
            Completeness score (0-1)
        """
        # Check entity completeness
        entity_completeness = min(1.0, len(graph.entities) / min_entities)
        
        # Check relationship completeness
        rel_completeness = min(1.0, len(graph.relationships) / min_relationships)
        
        # Check metadata completeness
        entities_with_metadata = sum(
            1 for e in graph.entities
            if e.metadata and len(e.metadata) > 0
        )
        metadata_completeness = entities_with_metadata / len(graph.entities) if graph.entities else 0
        
        # Check confidence scores
        high_confidence_entities = sum(
            1 for e in graph.entities
            if e.confidence > 0.7
        )
        confidence_completeness = high_confidence_entities / len(graph.entities) if graph.entities else 0
        
        # Combine scores
        completeness = np.mean([
            entity_completeness,
            rel_completeness,
            metadata_completeness,
            confidence_completeness
        ])
        
        # Update history
        self.metric_history[MetricType.COMPLETENESS].append(completeness)
        
        return completeness
    
    async def calculate_quality_score(
        self,
        graph: RadiatingGraph,
        ground_truth: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> QualityScore:
        """
        Calculate overall quality score
        
        Args:
            graph: Radiating graph
            ground_truth: Optional ground truth for evaluation
            context: Optional context for evaluation
            
        Returns:
            QualityScore with all metrics
        """
        self.stats['evaluations_performed'] += 1
        
        # Calculate precision and recall if ground truth available
        if ground_truth:
            precision, recall, f1 = await self.calculate_precision_recall(
                graph.entities,
                set(ground_truth.get('entities', [])),
                graph.relationships,
                set(ground_truth.get('relationships', []))
            )
        else:
            # Use confidence scores as proxy
            precision = np.mean([e.confidence for e in graph.entities]) if graph.entities else 0
            recall = 0.7  # Default estimate
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate coverage
        if context:
            coverage_analysis = await self.calculate_coverage_metrics(graph, context)
            coverage = (
                coverage_analysis.entity_coverage +
                coverage_analysis.relationship_coverage +
                coverage_analysis.depth_coverage
            ) / 3
        else:
            coverage = len(graph.entities) / 50  # Assume 50 entities is good coverage
            coverage = min(1.0, coverage)
        
        # Calculate diversity
        diversity_analysis = await self.calculate_diversity_scores(graph)
        diversity = (
            diversity_analysis.entity_diversity +
            diversity_analysis.relationship_diversity +
            diversity_analysis.source_diversity
        ) / 3
        
        # Calculate novelty
        historical = self.ground_truth_cache.get('historical_entities', set())
        novelty = await self.calculate_novelty_score(graph, historical)
        
        # Calculate coherence
        coherence = await self.calculate_coherence_score(graph)
        
        # Calculate completeness
        completeness = await self.calculate_completeness_score(graph)
        
        # Calculate overall score
        overall_score = (
            precision * self.METRIC_WEIGHTS['precision'] +
            recall * self.METRIC_WEIGHTS['recall'] +
            coverage * self.METRIC_WEIGHTS['coverage'] +
            diversity * self.METRIC_WEIGHTS['diversity'] +
            novelty * self.METRIC_WEIGHTS['novelty'] +
            coherence * self.METRIC_WEIGHTS['coherence'] +
            completeness * self.METRIC_WEIGHTS['completeness']
        )
        
        # Update statistics
        self.stats['average_quality_score'] = (
            (self.stats['average_quality_score'] * (self.stats['evaluations_performed'] - 1) +
             overall_score) / self.stats['evaluations_performed']
        )
        self.stats['best_quality_score'] = max(self.stats['best_quality_score'], overall_score)
        self.stats['worst_quality_score'] = min(self.stats['worst_quality_score'], overall_score)
        
        quality_score = QualityScore(
            overall_score=overall_score,
            precision=precision,
            recall=recall,
            f1=f1,
            coverage=coverage,
            diversity=diversity,
            novelty=novelty,
            coherence=coherence,
            completeness=completeness,
            metadata={
                'entity_count': len(graph.entities),
                'relationship_count': len(graph.relationships),
                'max_depth': max(e.depth for e in graph.entities) if graph.entities else 0,
                'quality_level': self._determine_quality_level(overall_score)
            }
        )
        
        # Store score
        await self._store_quality_score(quality_score)
        
        return quality_score
    
    def _determine_quality_level(self, score: float) -> str:
        """Determine quality level from score"""
        if score >= self.QUALITY_THRESHOLDS['excellent']:
            return 'excellent'
        elif score >= self.QUALITY_THRESHOLDS['good']:
            return 'good'
        elif score >= self.QUALITY_THRESHOLDS['acceptable']:
            return 'acceptable'
        elif score >= self.QUALITY_THRESHOLDS['poor']:
            return 'poor'
        else:
            return 'very_poor'
    
    async def _store_quality_score(self, score: QualityScore):
        """Store quality score for tracking"""
        try:
            # Store in Redis
            score_key = f"radiating:quality:score:{score.timestamp.timestamp()}"
            await self.redis_client.setex(
                score_key,
                86400 * 7,  # 7 days
                json.dumps({
                    'overall_score': score.overall_score,
                    'precision': score.precision,
                    'recall': score.recall,
                    'f1': score.f1,
                    'coverage': score.coverage,
                    'diversity': score.diversity,
                    'novelty': score.novelty,
                    'coherence': score.coherence,
                    'completeness': score.completeness,
                    'timestamp': score.timestamp.isoformat(),
                    'metadata': score.metadata
                })
            )
            
            # Add to score list
            await self.redis_client.lpush(
                "radiating:quality:scores",
                score_key
            )
            
            # Trim to keep only recent scores
            await self.redis_client.ltrim("radiating:quality:scores", 0, 999)
        
        except Exception as e:
            logger.error(f"Error storing quality score: {e}")
    
    async def generate_quality_report(
        self,
        period: timedelta = timedelta(days=1)
    ) -> List[MetricReport]:
        """
        Generate quality report for a time period
        
        Args:
            period: Time period for report
            
        Returns:
            List of metric reports
        """
        self.stats['reports_generated'] += 1
        reports = []
        
        for metric_type in MetricType:
            # Get recent history
            history = self.metric_history[metric_type][-100:]  # Last 100 measurements
            
            if not history:
                continue
            
            # Calculate statistics
            current_value = history[-1] if history else 0
            avg_value = np.mean(history)
            
            # Determine trend
            if len(history) > 10:
                recent = np.mean(history[-5:])
                older = np.mean(history[-10:-5])
                
                if recent > older * 1.05:
                    trend = "increasing"
                elif recent < older * 0.95:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "insufficient_data"
            
            # Generate recommendations
            recommendations = self._generate_metric_recommendations(
                metric_type,
                current_value,
                trend
            )
            
            # Create report
            report = MetricReport(
                metric_type=metric_type,
                value=current_value,
                trend=trend,
                comparison_period=f"{period.days} days",
                details={
                    'average': avg_value,
                    'min': min(history) if history else 0,
                    'max': max(history) if history else 0,
                    'std': np.std(history) if len(history) > 1 else 0,
                    'measurements': len(history),
                    'target': self.METRIC_TARGETS.get(metric_type, 0.7)
                },
                recommendations=recommendations
            )
            
            reports.append(report)
        
        return reports
    
    def _generate_metric_recommendations(
        self,
        metric_type: MetricType,
        value: float,
        trend: str
    ) -> List[str]:
        """Generate recommendations for metric improvement"""
        recommendations = []
        target = self.METRIC_TARGETS.get(metric_type, 0.7)
        
        if value < target:
            if metric_type == MetricType.PRECISION:
                recommendations.append("Increase confidence thresholds")
                recommendations.append("Improve entity validation")
                recommendations.append("Filter low-quality results")
            
            elif metric_type == MetricType.RECALL:
                recommendations.append("Expand search depth")
                recommendations.append("Include more entity types")
                recommendations.append("Lower filtering thresholds")
            
            elif metric_type == MetricType.COVERAGE:
                recommendations.append("Increase traversal depth")
                recommendations.append("Expand relationship types")
                recommendations.append("Include more data sources")
            
            elif metric_type == MetricType.DIVERSITY:
                recommendations.append("Diversify search strategies")
                recommendations.append("Include varied entity types")
                recommendations.append("Use multiple data sources")
            
            elif metric_type == MetricType.COHERENCE:
                recommendations.append("Improve relationship extraction")
                recommendations.append("Reduce isolated entities")
                recommendations.append("Strengthen graph connectivity")
            
            elif metric_type == MetricType.COMPLETENESS:
                recommendations.append("Enrich entity metadata")
                recommendations.append("Improve confidence scoring")
                recommendations.append("Expand result set")
        
        # Add trend-based recommendations
        if trend == "decreasing":
            recommendations.append(f"Investigate recent changes affecting {metric_type.value}")
            recommendations.append("Review and rollback recent configuration changes if needed")
        
        return recommendations
    
    async def _metric_tracker(self):
        """Background task to track metrics over time"""
        while True:
            try:
                await asyncio.sleep(300)  # Track every 5 minutes
                
                # Clean old history
                max_history = 1000
                for metric_type in MetricType:
                    if len(self.metric_history[metric_type]) > max_history:
                        self.metric_history[metric_type] = (
                            self.metric_history[metric_type][-max_history:]
                        )
                
                # Log current statistics
                logger.debug(
                    f"Quality Metrics - "
                    f"Evaluations: {self.stats['evaluations_performed']}, "
                    f"Avg Score: {self.stats['average_quality_score']:.2f}, "
                    f"Best: {self.stats['best_quality_score']:.2f}, "
                    f"Worst: {self.stats['worst_quality_score']:.2f}"
                )
                
            except Exception as e:
                logger.error(f"Error in metric tracker: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get quality metrics statistics"""
        current_metrics = {}
        
        for metric_type in MetricType:
            history = self.metric_history[metric_type]
            if history:
                current_metrics[metric_type.value] = {
                    'current': history[-1],
                    'average': np.mean(history),
                    'target': self.METRIC_TARGETS.get(metric_type, 0.7),
                    'meeting_target': history[-1] >= self.METRIC_TARGETS.get(metric_type, 0.7)
                }
        
        return {
            'evaluations_performed': self.stats['evaluations_performed'],
            'reports_generated': self.stats['reports_generated'],
            'average_quality_score': self.stats['average_quality_score'],
            'best_quality_score': self.stats['best_quality_score'],
            'worst_quality_score': self.stats['worst_quality_score'],
            'current_metrics': current_metrics,
            'quality_level': self._determine_quality_level(
                self.stats['average_quality_score']
            )
        }