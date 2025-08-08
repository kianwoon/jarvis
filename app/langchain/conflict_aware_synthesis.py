"""
Conflict-Aware Synthesis Module
Enhanced synthesis with pre-emptive conflict prevention and intelligent resolution
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class ConflictAwareSynthesizer:
    """
    Advanced synthesis system with conflict prevention and resolution capabilities
    """
    
    def __init__(self):
        # Enhanced conflict patterns with semantic understanding
        self.semantic_patterns = {
            'product_existence': {
                'entities': {
                    'openai': ['openai', 'open ai', 'chatgpt', 'gpt'],
                    'anthropic': ['anthropic', 'claude'],
                    'google': ['google', 'gemini', 'bard', 'deepmind'],
                    'meta': ['meta', 'facebook', 'llama']
                },
                'products': {
                    'models': ['gpt-5', 'chatgpt-5', 'gpt 5', 'claude-4', 'gemini-2', 'llama-4'],
                    'features': ['vision', 'voice', 'api', 'plugin', 'tool']
                },
                'states': {
                    'exists': ['exists', 'available', 'released', 'launched', 'is out', 'now available'],
                    'not_exists': ['does not exist', 'doesn\'t exist', 'not exist', 'myth', 'no official']
                }
            },
            'temporal_claims': {
                'time_markers': {
                    'current': ['now', 'currently', 'today', 'as of today', 'latest', 'recent'],
                    'past': ['previously', 'formerly', 'used to', 'was', 'were'],
                    'future': ['will', 'going to', 'planned', 'upcoming', 'soon']
                },
                'dates': {
                    'specific': r'\b(20\d{2})\b',
                    'relative': r'\b(yesterday|today|tomorrow|last\s+\w+|next\s+\w+)\b'
                }
            },
            'quantitative_claims': {
                'metrics': {
                    'performance': ['faster', 'slower', 'better', 'worse', 'improved'],
                    'scale': ['billion', 'million', 'thousand', 'parameters', 'users'],
                    'percentage': r'\b\d+(?:\.\d+)?%\b'
                }
            }
        }
        
        # Conflict resolution strategies
        self.resolution_strategies = {
            'existence_conflict': self._resolve_existence_conflict,
            'temporal_conflict': self._resolve_temporal_conflict,
            'quantitative_conflict': self._resolve_quantitative_conflict,
            'version_conflict': self._resolve_version_conflict
        }
    
    async def synthesize_with_conflict_prevention(
        self,
        question: str,
        info_sources: List[Dict[str, Any]],
        conversation_history: Optional[str] = None,
        enable_prevention: bool = True
    ) -> Dict[str, Any]:
        """
        Synthesize response with pre-emptive conflict prevention
        
        Args:
            question: User question
            info_sources: Prioritized information sources
            conversation_history: Previous conversation context
            enable_prevention: Whether to enable conflict prevention
            
        Returns:
            Synthesis result with conflict analysis
        """
        
        # Step 1: Pre-process sources for conflict detection
        conflict_analysis = await self._analyze_source_conflicts(info_sources)
        
        # Step 2: Apply conflict prevention if enabled
        if enable_prevention and conflict_analysis['has_conflicts']:
            info_sources = await self._apply_conflict_prevention(
                info_sources, 
                conflict_analysis
            )
        
        # Step 3: Build synthesis with conflict awareness
        synthesis_result = self._build_conflict_aware_synthesis(
            question,
            info_sources,
            conflict_analysis,
            conversation_history
        )
        
        # Step 4: Generate conflict report
        conflict_report = self._generate_conflict_report(conflict_analysis)
        
        return {
            'synthesis': synthesis_result,
            'conflict_analysis': conflict_analysis,
            'conflict_report': conflict_report,
            'sources_modified': enable_prevention and conflict_analysis['has_conflicts'],
            'prevention_applied': enable_prevention
        }
    
    async def _analyze_source_conflicts(
        self, 
        info_sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze sources for potential conflicts"""
        
        conflicts = []
        conflict_pairs = []
        
        # Compare each source pair
        for i, source1 in enumerate(info_sources):
            for j, source2 in enumerate(info_sources[i+1:], i+1):
                conflict = self._detect_conflict_between_sources(source1, source2)
                if conflict:
                    conflict_pairs.append({
                        'source1_idx': i,
                        'source2_idx': j,
                        'source1_label': source1.get('label', f'Source {i}'),
                        'source2_label': source2.get('label', f'Source {j}'),
                        'conflict_type': conflict['type'],
                        'confidence': conflict['confidence'],
                        'details': conflict.get('details', {})
                    })
                    conflicts.append(conflict)
        
        # Identify conflict clusters (multiple sources in conflict)
        clusters = self._identify_conflict_clusters(conflict_pairs)
        
        return {
            'has_conflicts': len(conflicts) > 0,
            'total_conflicts': len(conflicts),
            'conflict_pairs': conflict_pairs,
            'conflict_clusters': clusters,
            'dominant_conflict_type': self._get_dominant_conflict_type(conflicts),
            'resolution_needed': len(clusters) > 0
        }
    
    def _detect_conflict_between_sources(
        self, 
        source1: Dict[str, Any], 
        source2: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Detect conflicts between two sources"""
        
        content1 = source1.get('content', '').lower()
        content2 = source2.get('content', '').lower()
        
        # Check for existence conflicts
        existence_conflict = self._check_existence_conflict(content1, content2)
        if existence_conflict:
            return {
                'type': 'existence_conflict',
                'confidence': existence_conflict['confidence'],
                'details': existence_conflict
            }
        
        # Check for temporal conflicts
        temporal_conflict = self._check_temporal_conflict(content1, content2)
        if temporal_conflict:
            return {
                'type': 'temporal_conflict',
                'confidence': temporal_conflict['confidence'],
                'details': temporal_conflict
            }
        
        # Check for quantitative conflicts
        quant_conflict = self._check_quantitative_conflict(content1, content2)
        if quant_conflict:
            return {
                'type': 'quantitative_conflict',
                'confidence': quant_conflict['confidence'],
                'details': quant_conflict
            }
        
        return None
    
    def _check_existence_conflict(self, content1: str, content2: str) -> Optional[Dict[str, Any]]:
        """Check for existence vs non-existence conflicts"""
        
        patterns = self.semantic_patterns['product_existence']
        
        # Find entity and product mentions
        for entity_key, entity_patterns in patterns['entities'].items():
            entity_in_1 = any(e in content1 for e in entity_patterns)
            entity_in_2 = any(e in content2 for e in entity_patterns)
            
            if entity_in_1 and entity_in_2:
                # Check for conflicting existence states
                exists_in_1 = any(e in content1 for e in patterns['states']['exists'])
                not_exists_in_1 = any(e in content1 for e in patterns['states']['not_exists'])
                exists_in_2 = any(e in content2 for e in patterns['states']['exists'])
                not_exists_in_2 = any(e in content2 for e in patterns['states']['not_exists'])
                
                if (exists_in_1 and not_exists_in_2) or (not_exists_in_1 and exists_in_2):
                    # Calculate confidence based on explicitness
                    confidence = 0.9 if ('does not exist' in content1 or 'does not exist' in content2) else 0.7
                    
                    return {
                        'entity': entity_key,
                        'state_in_source1': 'exists' if exists_in_1 else 'not_exists',
                        'state_in_source2': 'exists' if exists_in_2 else 'not_exists',
                        'confidence': confidence
                    }
        
        return None
    
    def _check_temporal_conflict(self, content1: str, content2: str) -> Optional[Dict[str, Any]]:
        """Check for temporal/chronological conflicts"""
        
        patterns = self.semantic_patterns['temporal_claims']
        
        # Extract temporal markers
        time1 = self._extract_temporal_markers(content1, patterns)
        time2 = self._extract_temporal_markers(content2, patterns)
        
        if time1 and time2:
            # Check for conflicting time claims about same entity
            if time1['type'] != time2['type']:
                # e.g., one says "currently" other says "previously"
                if (time1['type'] == 'current' and time2['type'] == 'past') or \
                   (time1['type'] == 'past' and time2['type'] == 'current'):
                    return {
                        'time_in_source1': time1,
                        'time_in_source2': time2,
                        'confidence': 0.8
                    }
        
        return None
    
    def _check_quantitative_conflict(self, content1: str, content2: str) -> Optional[Dict[str, Any]]:
        \"\"\"Check for conflicting numerical claims\"\"\"
        
        patterns = self.semantic_patterns['quantitative_claims']
        
        # Extract numbers with context
        numbers1 = self._extract_numbers_with_context(content1)
        numbers2 = self._extract_numbers_with_context(content2)
        
        # Compare similar metrics
        for num1 in numbers1:
            for num2 in numbers2:
                if self._are_same_metric(num1['context'], num2['context']):
                    # Check if numbers differ significantly
                    try:
                        val1 = float(num1['value'])
                        val2 = float(num2['value'])
                        if abs(val1 - val2) / max(val1, val2) > 0.2:  # >20% difference
                            return {
                                'metric': num1['context'],
                                'value_in_source1': val1,
                                'value_in_source2': val2,
                                'difference_percent': abs(val1 - val2) / max(val1, val2) * 100,
                                'confidence': 0.7
                            }
                    except:
                        pass
        
        return None
    
    def _extract_temporal_markers(self, content: str, patterns: dict) -> Optional[Dict[str, str]]:
        \"\"\"Extract temporal markers from content\"\"\"
        
        for time_type, markers in patterns['time_markers'].items():
            if any(marker in content for marker in markers):
                return {'type': time_type, 'markers': [m for m in markers if m in content]}
        
        return None
    
    def _extract_numbers_with_context(self, content: str, window: int = 20) -> List[Dict[str, Any]]:
        \"\"\"Extract numbers with surrounding context\"\"\"
        
        number_pattern = r'\\b(\\d+(?:\\.\\d+)?(?:k|m|b|%)?)\\b'
        results = []
        
        for match in re.finditer(number_pattern, content):
            start = max(0, match.start() - window)
            end = min(len(content), match.end() + window)
            context = content[start:end]
            
            results.append({
                'value': match.group(1),
                'context': context,
                'position': match.start()
            })
        
        return results
    
    def _are_same_metric(self, context1: str, context2: str) -> bool:
        \"\"\"Check if two contexts refer to the same metric\"\"\"
        
        # Simple heuristic: check for common keywords
        keywords = ['users', 'parameters', 'accuracy', 'speed', 'cost', 'revenue', 'downloads']
        
        for keyword in keywords:
            if keyword in context1.lower() and keyword in context2.lower():
                return True
        
        return False
    
    def _identify_conflict_clusters(
        self, 
        conflict_pairs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        \"\"\"Identify clusters of conflicting sources\"\"\"
        
        clusters = []
        processed = set()
        
        for pair in conflict_pairs:
            if pair['source1_idx'] in processed and pair['source2_idx'] in processed:
                continue
            
            # Find all sources connected to this conflict
            cluster_indices = {pair['source1_idx'], pair['source2_idx']}
            
            # Expand cluster
            changed = True
            while changed:
                changed = False
                for other_pair in conflict_pairs:
                    if other_pair['source1_idx'] in cluster_indices or \
                       other_pair['source2_idx'] in cluster_indices:
                        before_size = len(cluster_indices)
                        cluster_indices.add(other_pair['source1_idx'])
                        cluster_indices.add(other_pair['source2_idx'])
                        if len(cluster_indices) > before_size:
                            changed = True
            
            processed.update(cluster_indices)
            
            clusters.append({
                'source_indices': list(cluster_indices),
                'size': len(cluster_indices),
                'conflict_types': list(set(p['conflict_type'] for p in conflict_pairs 
                                          if p['source1_idx'] in cluster_indices or 
                                             p['source2_idx'] in cluster_indices))
            })
        
        return clusters
    
    def _get_dominant_conflict_type(self, conflicts: List[Dict[str, Any]]) -> Optional[str]:
        \"\"\"Get the most common conflict type\"\"\"
        
        if not conflicts:
            return None
        
        type_counts = {}
        for conflict in conflicts:
            conflict_type = conflict['type']
            type_counts[conflict_type] = type_counts.get(conflict_type, 0) + 1
        
        return max(type_counts.items(), key=lambda x: x[1])[0]
    
    async def _apply_conflict_prevention(
        self, 
        info_sources: List[Dict[str, Any]], 
        conflict_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        \"\"\"Apply conflict prevention strategies to sources\"\"\"
        
        modified_sources = info_sources.copy()
        
        # Handle each conflict cluster
        for cluster in conflict_analysis['conflict_clusters']:
            # Get sources in cluster
            cluster_sources = [modified_sources[i] for i in cluster['source_indices']]
            
            # Sort by priority and freshness
            cluster_sources.sort(key=lambda x: (x.get('priority', 0), x.get('freshness_score', 0)), reverse=True)
            
            # Apply resolution strategy based on dominant conflict type
            for conflict_type in cluster['conflict_types']:
                if conflict_type in self.resolution_strategies:
                    resolution_func = self.resolution_strategies[conflict_type]
                    cluster_sources = resolution_func(cluster_sources)
            
            # Update sources with resolved versions
            for idx, source_idx in enumerate(cluster['source_indices']):
                if idx < len(cluster_sources):
                    modified_sources[source_idx] = cluster_sources[idx]
        
        return modified_sources
    
    def _resolve_existence_conflict(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        \"\"\"Resolve existence conflicts between sources\"\"\"
        
        # Highest priority source wins for existence claims
        if sources:
            winner = sources[0]  # Already sorted by priority
            
            # Mark other sources as superseded
            for i in range(1, len(sources)):
                sources[i]['metadata'] = sources[i].get('metadata', {})
                sources[i]['metadata']['superseded'] = True
                sources[i]['metadata']['superseded_by'] = winner.get('label', 'higher_priority_source')
                sources[i]['metadata']['conflict_type'] = 'existence'
        
        return sources
    
    def _resolve_temporal_conflict(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        \"\"\"Resolve temporal conflicts between sources\"\"\"
        
        # Most recent source wins for temporal claims
        if sources:
            # Sort by timestamp if available
            sources_with_time = [(s, self._extract_timestamp(s)) for s in sources]
            sources_with_time.sort(key=lambda x: x[1] if x[1] else 0, reverse=True)
            
            winner = sources_with_time[0][0]
            
            # Mark older sources as outdated
            for source, _ in sources_with_time[1:]:
                source['metadata'] = source.get('metadata', {})
                source['metadata']['outdated'] = True
                source['metadata']['updated_by'] = winner.get('label', 'newer_source')
                source['metadata']['conflict_type'] = 'temporal'
        
        return [s[0] for s in sources_with_time]
    
    def _resolve_quantitative_conflict(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        \"\"\"Resolve quantitative conflicts between sources\"\"\"
        
        # For quantitative conflicts, add reconciliation note
        if sources:
            # Find the range of values
            all_numbers = []
            for source in sources:
                numbers = self._extract_numbers_with_context(source.get('content', ''))
                all_numbers.extend([n['value'] for n in numbers])
            
            if all_numbers:
                # Add reconciliation metadata
                for source in sources:
                    source['metadata'] = source.get('metadata', {})
                    source['metadata']['quantitative_conflict'] = True
                    source['metadata']['value_range'] = f\"Values vary across sources: {', '.join(all_numbers[:3])}\"\n                    source['metadata']['conflict_type'] = 'quantitative'
        
        return sources
    
    def _resolve_version_conflict(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        \"\"\"Resolve version conflicts between sources\"\"\"
        
        # Latest version wins
        if sources:
            # Extract versions
            sources_with_versions = []
            for source in sources:
                version = self._extract_version(source.get('content', ''))
                sources_with_versions.append((source, version))
            
            # Sort by version (if parseable)
            sources_with_versions.sort(key=lambda x: x[1] if x[1] else '', reverse=True)
            
            if sources_with_versions[0][1]:  # If we found versions
                winner = sources_with_versions[0][0]
                
                # Mark older versions as outdated
                for source, version in sources_with_versions[1:]:
                    if version:
                        source['metadata'] = source.get('metadata', {})
                        source['metadata']['version_outdated'] = True
                        source['metadata']['newer_version'] = sources_with_versions[0][1]
                        source['metadata']['conflict_type'] = 'version'
            
            return [s[0] for s in sources_with_versions]
        
        return sources
    
    def _extract_timestamp(self, source: Dict[str, Any]) -> Optional[float]:
        \"\"\"Extract timestamp from source\"\"\"
        
        # Try multiple timestamp fields
        for field in ['timestamp', 'created_at', 'updated_at']:
            if field in source:
                try:
                    if isinstance(source[field], (int, float)):
                        return source[field]
                    elif isinstance(source[field], str):
                        return datetime.fromisoformat(source[field]).timestamp()
                except:
                    pass
        
        return None
    
    def _extract_version(self, content: str) -> Optional[str]:
        \"\"\"Extract version number from content\"\"\"
        
        version_pattern = r'\\b(?:version|v\\.?)\\s*(\\d+(?:\\.\\d+)*)\\b'
        match = re.search(version_pattern, content.lower())
        return match.group(1) if match else None
    
    def _build_conflict_aware_synthesis(
        self,
        question: str,
        info_sources: List[Dict[str, Any]],
        conflict_analysis: Dict[str, Any],
        conversation_history: Optional[str]
    ) -> str:
        \"\"\"Build synthesis with conflict awareness\"\"\"
        
        synthesis_parts = []
        
        # Add conflict warning if needed
        if conflict_analysis['has_conflicts']:
            synthesis_parts.append(
                f\"⚠️ CONFLICT RESOLUTION APPLIED: Detected {conflict_analysis['total_conflicts']} \"
                f\"conflicts across sources. Using highest-priority and most recent information.\"
            )
        
        # Group sources by conflict status
        clean_sources = []
        conflicted_sources = []
        
        for source in info_sources:
            if source.get('metadata', {}).get('superseded') or \
               source.get('metadata', {}).get('outdated'):
                conflicted_sources.append(source)
            else:
                clean_sources.append(source)
        
        # Build main synthesis from clean sources
        if clean_sources:
            synthesis_parts.append(\"\\n📊 VERIFIED INFORMATION:\")
            for source in clean_sources[:3]:  # Top 3 clean sources
                synthesis_parts.append(f\"• {source.get('label', 'Source')}: {source.get('content', '')[:200]}...\")
        
        # Add conflicted information with warnings
        if conflicted_sources and conflict_analysis.get('resolution_needed'):
            synthesis_parts.append(\"\\n⚠️ CONFLICTING INFORMATION (Lower Priority):\")
            for source in conflicted_sources[:2]:  # Show top 2 conflicts
                conflict_type = source.get('metadata', {}).get('conflict_type', 'unknown')
                synthesis_parts.append(
                    f\"• [{conflict_type.upper()}] {source.get('label', 'Source')}: \"
                    f\"{source.get('content', '')[:150]}...\"
                )
        
        # Add conversation context if no major conflicts
        if conversation_history and not conflict_analysis.get('resolution_needed'):
            synthesis_parts.append(f\"\\n📝 CONVERSATION CONTEXT:\\n{conversation_history}\")
        
        return \"\\n\".join(synthesis_parts)
    
    def _generate_conflict_report(self, conflict_analysis: Dict[str, Any]) -> str:
        \"\"\"Generate human-readable conflict report\"\"\"
        
        if not conflict_analysis['has_conflicts']:
            return \"No conflicts detected between information sources.\"\n        \n        report_parts = [\n            f\"🔍 CONFLICT ANALYSIS REPORT\",\n            f\"Total Conflicts: {conflict_analysis['total_conflicts']}\",\n            f\"Dominant Type: {conflict_analysis.get('dominant_conflict_type', 'mixed')}\",\n            \"\"\n        ]\n        \n        # Detail conflict pairs\n        if conflict_analysis['conflict_pairs']:\n            report_parts.append(\"Conflict Pairs:\")\n            for pair in conflict_analysis['conflict_pairs'][:5]:  # Show top 5\n                report_parts.append(\n                    f\"  • {pair['source1_label']} ↔ {pair['source2_label']}: \"\n                    f\"{pair['conflict_type']} (confidence: {pair['confidence']:.2f})\"\n                )\n        \n        # Detail conflict clusters\n        if conflict_analysis['conflict_clusters']:\n            report_parts.append(\"\\nConflict Clusters:\")\n            for i, cluster in enumerate(conflict_analysis['conflict_clusters']):\n                report_parts.append(\n                    f\"  Cluster {i+1}: {cluster['size']} sources with \"\n                    f\"{', '.join(cluster['conflict_types'])} conflicts\"\n                )\n        \n        return \"\\n\".join(report_parts)\n\n# Global instance\nconflict_aware_synthesizer = ConflictAwareSynthesizer()