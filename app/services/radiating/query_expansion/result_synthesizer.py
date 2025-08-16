"""
Result Synthesizer for Radiating Coverage System

Combines radiating results into coherent responses, ranks and filters by relevance,
and formats output for different use cases.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

from app.core.radiating_settings_cache import get_radiating_settings

logger = logging.getLogger(__name__)

class OutputFormat(Enum):
    """Supported output formats for synthesized results"""
    SUMMARY = "summary"  # Concise summary
    DETAILED = "detailed"  # Full details with relationships
    GRAPH = "graph"  # Graph-structured data
    NARRATIVE = "narrative"  # Natural language narrative
    JSON = "json"  # Structured JSON

@dataclass
class RadiatingResult:
    """Individual result from radiating coverage"""
    entity: Dict[str, Any]  # Entity information
    relationships: List[Dict[str, Any]]  # Related entities and relationships
    relevance_score: float
    confidence: float
    depth: int  # How many hops from origin
    path: List[str]  # Path taken to reach this result
    source: str  # Source of information (neo4j, vector, llm)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class SynthesizedResult:
    """Final synthesized result from radiating coverage"""
    query: str
    summary: str
    key_findings: List[Dict[str, Any]]
    entity_graph: Dict[str, Any]
    relationships_found: List[Dict[str, Any]]
    coverage_metrics: Dict[str, Any]
    confidence: float
    format: OutputFormat
    raw_results: Optional[List[RadiatingResult]] = None

class ResultSynthesizer:
    """
    Synthesizes radiating coverage results into coherent, ranked responses.
    """
    
    def __init__(self):
        self.settings = get_radiating_settings()
        self.synthesis_config = self.settings.get('synthesis', {})
        self.relevance_threshold = self.settings.get('relevance_threshold', 0.7)
        
    async def synthesize(
        self,
        query: str,
        results: List[RadiatingResult],
        output_format: OutputFormat = OutputFormat.SUMMARY
    ) -> SynthesizedResult:
        """
        Synthesize radiating results into a coherent response.
        
        Args:
            query: Original query
            results: List of radiating results
            output_format: Desired output format
            
        Returns:
            SynthesizedResult with formatted output
        """
        
        # Filter results by relevance
        filtered_results = self._filter_results(results)
        
        # Rank results
        ranked_results = self._rank_results(filtered_results)
        
        # Merge duplicate or overlapping results
        if self.synthesis_config.get('enable_result_merging', True):
            merged_results = self._merge_results(ranked_results)
        else:
            merged_results = ranked_results
        
        # Extract key findings
        key_findings = self._extract_key_findings(merged_results)
        
        # Build entity graph
        entity_graph = self._build_entity_graph(merged_results)
        
        # Extract relationships
        relationships = self._extract_relationships(merged_results)
        
        # Generate summary based on format
        if output_format == OutputFormat.SUMMARY:
            summary = await self._generate_summary(query, key_findings)
        elif output_format == OutputFormat.DETAILED:
            summary = await self._generate_detailed_summary(query, merged_results)
        elif output_format == OutputFormat.NARRATIVE:
            summary = await self._generate_narrative(query, merged_results)
        elif output_format == OutputFormat.GRAPH:
            summary = self._format_as_graph(entity_graph, relationships)
        else:  # JSON
            summary = json.dumps({
                'entities': [self._entity_to_dict(r.entity) for r in merged_results],
                'relationships': relationships
            }, indent=2)
        
        # Calculate coverage metrics
        coverage_metrics = self._calculate_coverage_metrics(merged_results, results)
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(merged_results)
        
        return SynthesizedResult(
            query=query,
            summary=summary,
            key_findings=key_findings,
            entity_graph=entity_graph,
            relationships_found=relationships,
            coverage_metrics=coverage_metrics,
            confidence=confidence,
            format=output_format,
            raw_results=merged_results if self.synthesis_config.get('include_raw_results', False) else None
        )
    
    def _filter_results(self, results: List[RadiatingResult]) -> List[RadiatingResult]:
        """Filter results based on relevance and confidence thresholds"""
        
        if not self.synthesis_config.get('enable_result_filtering', True):
            return results
        
        filtered = []
        
        for result in results:
            # Check relevance threshold
            if result.relevance_score < self.relevance_threshold:
                continue
            
            # Check confidence threshold if enabled
            if self.synthesis_config.get('filter_low_confidence', True):
                min_confidence = self.synthesis_config.get('min_confidence', 0.5)
                if result.confidence < min_confidence:
                    continue
            
            filtered.append(result)
        
        logger.debug(f"Filtered {len(results)} results to {len(filtered)}")
        return filtered
    
    def _rank_results(self, results: List[RadiatingResult]) -> List[RadiatingResult]:
        """Rank results based on configured ranking algorithm"""
        
        if not self.synthesis_config.get('enable_result_ranking', True):
            return results
        
        ranking_algorithm = self.synthesis_config.get('ranking_algorithm', 'combined')
        
        if ranking_algorithm == 'relevance':
            # Sort by relevance score
            ranked = sorted(results, key=lambda x: x.relevance_score, reverse=True)
        elif ranking_algorithm == 'confidence':
            # Sort by confidence
            ranked = sorted(results, key=lambda x: x.confidence, reverse=True)
        elif ranking_algorithm == 'combined':
            # Combined score: weighted average of relevance and confidence
            ranked = sorted(
                results,
                key=lambda x: (0.6 * x.relevance_score + 0.4 * x.confidence),
                reverse=True
            )
        else:
            # Default to relevance
            ranked = sorted(results, key=lambda x: x.relevance_score, reverse=True)
        
        return ranked
    
    def _merge_results(self, results: List[RadiatingResult]) -> List[RadiatingResult]:
        """Merge duplicate or highly similar results"""
        
        if not results:
            return []
        
        merge_strategy = self.synthesis_config.get('merge_strategy', 'weighted')
        merged = []
        seen_entities = {}
        
        for result in results:
            entity_key = self._get_entity_key(result.entity)
            
            if entity_key in seen_entities:
                # Merge with existing result
                existing = seen_entities[entity_key]
                
                if merge_strategy == 'weighted':
                    # Weighted average based on confidence
                    total_confidence = existing.confidence + result.confidence
                    existing.relevance_score = (
                        existing.relevance_score * existing.confidence +
                        result.relevance_score * result.confidence
                    ) / total_confidence
                    existing.confidence = total_confidence / 2
                    
                    # Merge relationships
                    existing.relationships.extend(result.relationships)
                    
                    # Update metadata
                    existing.metadata['merge_count'] = existing.metadata.get('merge_count', 1) + 1
                    
                elif merge_strategy == 'highest':
                    # Keep the result with highest score
                    if result.relevance_score > existing.relevance_score:
                        seen_entities[entity_key] = result
                        
            else:
                seen_entities[entity_key] = result
                merged.append(result)
        
        return merged
    
    def _extract_key_findings(self, results: List[RadiatingResult]) -> List[Dict[str, Any]]:
        """Extract the most important findings from results"""
        
        key_findings = []
        
        # Take top N results based on relevance
        top_n = min(10, len(results))
        top_results = results[:top_n]
        
        for result in top_results:
            finding = {
                'entity': self._entity_to_dict(result.entity),
                'relevance': result.relevance_score,
                'confidence': result.confidence,
                'depth': result.depth,
                'key_relationships': self._get_key_relationships(result.relationships),
                'source': result.source
            }
            key_findings.append(finding)
        
        return key_findings
    
    def _build_entity_graph(self, results: List[RadiatingResult]) -> Dict[str, Any]:
        """Build a graph structure from the results"""
        
        nodes = []
        edges = []
        node_ids = set()
        
        for result in results:
            # Add main entity as node
            entity_id = self._get_entity_key(result.entity)
            if entity_id not in node_ids:
                nodes.append({
                    'id': entity_id,
                    'label': result.entity.get('text', ''),
                    'type': result.entity.get('type', 'Unknown'),
                    'relevance': result.relevance_score,
                    'depth': result.depth
                })
                node_ids.add(entity_id)
            
            # Add relationships as edges
            for rel in result.relationships:
                target_entity = rel.get('target_entity', {})
                target_id = self._get_entity_key(target_entity)
                
                # Add target node if not exists
                if target_id not in node_ids:
                    nodes.append({
                        'id': target_id,
                        'label': target_entity.get('text', ''),
                        'type': target_entity.get('type', 'Unknown'),
                        'relevance': rel.get('relevance', 0.5),
                        'depth': result.depth + 1
                    })
                    node_ids.add(target_id)
                
                # Add edge
                edges.append({
                    'source': entity_id,
                    'target': target_id,
                    'relationship': rel.get('type', 'RELATED_TO'),
                    'weight': rel.get('weight', 1.0)
                })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'node_count': len(nodes),
            'edge_count': len(edges)
        }
    
    def _extract_relationships(self, results: List[RadiatingResult]) -> List[Dict[str, Any]]:
        """Extract all unique relationships from results"""
        
        relationships = []
        seen_relationships = set()
        
        for result in results:
            source_id = self._get_entity_key(result.entity)
            
            for rel in result.relationships:
                target_entity = rel.get('target_entity', {})
                target_id = self._get_entity_key(target_entity)
                rel_type = rel.get('type', 'RELATED_TO')
                
                # Create unique relationship key
                rel_key = f"{source_id}-{rel_type}-{target_id}"
                
                if rel_key not in seen_relationships:
                    relationships.append({
                        'source': result.entity,
                        'target': target_entity,
                        'type': rel_type,
                        'confidence': rel.get('confidence', 0.5),
                        'metadata': rel.get('metadata', {})
                    })
                    seen_relationships.add(rel_key)
        
        return relationships
    
    async def _generate_summary(self, query: str, key_findings: List[Dict]) -> str:
        """Generate a concise summary of findings"""
        
        if not key_findings:
            return f"No significant findings for query: {query}"
        
        summary_parts = [f"For the query '{query}', the following key findings were discovered:"]
        
        for i, finding in enumerate(key_findings[:5], 1):
            entity = finding['entity']
            summary_parts.append(
                f"{i}. {entity['text']} ({entity['type']}) - "
                f"Relevance: {finding['relevance']:.2f}, "
                f"with {len(finding['key_relationships'])} key relationships"
            )
        
        if len(key_findings) > 5:
            summary_parts.append(f"... and {len(key_findings) - 5} more findings.")
        
        return "\n".join(summary_parts)
    
    async def _generate_detailed_summary(
        self, 
        query: str, 
        results: List[RadiatingResult]
    ) -> str:
        """Generate a detailed summary with full information"""
        
        sections = [f"Detailed Analysis for: {query}\n{'='*50}"]
        
        for result in results[:10]:
            entity = result.entity
            sections.append(f"\n{entity['text']} ({entity['type']})")
            sections.append(f"  Relevance: {result.relevance_score:.2f}")
            sections.append(f"  Confidence: {result.confidence:.2f}")
            sections.append(f"  Depth: {result.depth}")
            sections.append(f"  Source: {result.source}")
            
            if result.relationships:
                sections.append("  Relationships:")
                for rel in result.relationships[:5]:
                    target = rel.get('target_entity', {})
                    sections.append(
                        f"    - {rel.get('type', 'RELATED_TO')} -> "
                        f"{target.get('text', 'Unknown')} ({target.get('type', 'Unknown')})"
                    )
        
        return "\n".join(sections)
    
    async def _generate_narrative(self, query: str, results: List[RadiatingResult]) -> str:
        """Generate a natural language narrative from results"""
        
        if not results:
            return f"No information was found related to '{query}'."
        
        # Group results by depth for narrative structure
        by_depth = {}
        for result in results:
            depth = result.depth
            if depth not in by_depth:
                by_depth[depth] = []
            by_depth[depth].append(result)
        
        narrative_parts = [f"Exploring '{query}' reveals an interconnected network of information."]
        
        # Direct findings (depth 0)
        if 0 in by_depth:
            direct = by_depth[0]
            entities = [r.entity['text'] for r in direct[:3]]
            narrative_parts.append(
                f"Directly related are: {', '.join(entities)}."
            )
        
        # Secondary connections (depth 1)
        if 1 in by_depth:
            secondary = by_depth[1]
            narrative_parts.append(
                f"These connect to {len(secondary)} additional entities, "
                f"expanding the knowledge network."
            )
        
        # Deeper connections
        max_depth = max(by_depth.keys()) if by_depth else 0
        if max_depth > 1:
            narrative_parts.append(
                f"The exploration extends {max_depth} levels deep, "
                f"uncovering {len(results)} total connections."
            )
        
        return " ".join(narrative_parts)
    
    def _format_as_graph(
        self, 
        entity_graph: Dict[str, Any], 
        relationships: List[Dict[str, Any]]
    ) -> str:
        """Format results as a graph structure"""
        
        return json.dumps({
            'graph': entity_graph,
            'relationships': relationships,
            'statistics': {
                'total_nodes': entity_graph['node_count'],
                'total_edges': entity_graph['edge_count'],
                'unique_relationships': len(relationships)
            }
        }, indent=2)
    
    def _calculate_coverage_metrics(
        self, 
        filtered_results: List[RadiatingResult],
        all_results: List[RadiatingResult]
    ) -> Dict[str, Any]:
        """Calculate metrics about coverage quality"""
        
        if not all_results:
            return {
                'total_discovered': 0,
                'relevant_discovered': 0,
                'coverage_ratio': 0.0,
                'average_depth': 0,
                'max_depth': 0
            }
        
        depths = [r.depth for r in filtered_results]
        
        return {
            'total_discovered': len(all_results),
            'relevant_discovered': len(filtered_results),
            'coverage_ratio': len(filtered_results) / len(all_results),
            'average_depth': sum(depths) / len(depths) if depths else 0,
            'max_depth': max(depths) if depths else 0,
            'sources_used': list(set(r.source for r in filtered_results)),
            'filtering_rate': 1 - (len(filtered_results) / len(all_results))
        }
    
    def _calculate_confidence(self, results: List[RadiatingResult]) -> float:
        """Calculate overall confidence in the synthesized results"""
        
        if not results:
            return 0.0
        
        # Weighted average based on relevance
        total_weight = sum(r.relevance_score for r in results)
        if total_weight == 0:
            return 0.0
        
        weighted_confidence = sum(
            r.confidence * r.relevance_score for r in results
        ) / total_weight
        
        return min(weighted_confidence, 1.0)
    
    def _get_entity_key(self, entity: Dict[str, Any]) -> str:
        """Generate a unique key for an entity"""
        text = entity.get('text', '').lower().strip()
        entity_type = entity.get('type', 'unknown').lower()
        return f"{entity_type}:{text}"
    
    def _entity_to_dict(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Convert entity to dictionary format"""
        return {
            'text': entity.get('text', ''),
            'type': entity.get('type', 'Unknown'),
            'confidence': entity.get('confidence', 0.0),
            'metadata': entity.get('metadata', {})
        }
    
    def _get_key_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract the most important relationships"""
        
        # Sort by confidence/weight and take top 5
        sorted_rels = sorted(
            relationships,
            key=lambda x: x.get('confidence', x.get('weight', 0.5)),
            reverse=True
        )
        
        key_rels = []
        for rel in sorted_rels[:5]:
            key_rels.append({
                'type': rel.get('type', 'RELATED_TO'),
                'target': rel.get('target_entity', {}).get('text', 'Unknown'),
                'confidence': rel.get('confidence', 0.5)
            })
        
        return key_rels