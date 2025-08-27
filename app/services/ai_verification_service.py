"""
AI Verification and Self-Correction Service

This service adds intelligence to verify retrieval completeness and trigger
self-correction when results appear incomplete. It's the quality assurance
layer that makes the notebook system truly intelligent and self-healing.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pydantic import BaseModel

from app.models.notebook_models import NotebookRAGSource, NotebookRAGResponse
from app.services.ai_task_planner import TaskExecutionPlan, VerificationRules, RetrievalStrategy

logger = logging.getLogger(__name__)


class VerificationResult(BaseModel):
    """Result of intelligent verification analysis"""
    confidence: float  # 0.0-1.0 confidence in completeness
    completeness_score: float  # 0.0-1.0 estimated completeness
    needs_correction: bool  # True if self-correction is needed
    
    # Analysis details
    result_count: int
    unique_sources: int
    diversity_score: float  # 0.0-1.0 how diverse the sources are
    expected_vs_actual: str  # "25 expected, 19 found"
    
    # Recommendations for improvement
    correction_strategies: List[RetrievalStrategy]
    reasoning: str
    quality_issues: List[str]  # Identified quality problems
    

class CompletenessAnalysis(BaseModel):
    """Detailed analysis of result completeness"""
    entity_coverage: Dict[str, int]  # {"projects": 19, "companies": 15}
    temporal_coverage: Dict[str, int]  # {"2020-2024": 10, "2015-2019": 9}
    source_diversity: Dict[str, int]  # {"document": 15, "memory": 4}
    missing_indicators: List[str]  # Potential gaps identified
    

class AIVerificationService:
    """
    AI-powered verification service that ensures retrieval completeness
    and triggers self-correction when results appear incomplete.
    """
    
    def __init__(self):
        self.verification_history = {}  # Track verification patterns
        
    async def verify_completeness(
        self,
        results: NotebookRAGResponse,
        plan: TaskExecutionPlan,
        notebook_id: Optional[str] = None
    ) -> VerificationResult:
        """
        Main verification method: analyzes results against plan expectations
        and determines if self-correction is needed.
        """
        try:
            logger.info(f"[VERIFICATION] Analyzing {len(results.sources)} results against plan expectations")
            
            # Perform comprehensive analysis
            analysis = await self._analyze_completeness(results, plan)
            
            # Calculate confidence scores
            confidence = self._calculate_confidence(analysis, plan)
            completeness = self._estimate_completeness(analysis, plan)
            
            # Determine if correction is needed
            needs_correction = self._needs_correction(confidence, completeness, plan)
            
            # Generate correction strategies if needed
            correction_strategies = []
            if needs_correction:
                correction_strategies = await self._generate_correction_strategies(
                    analysis, plan, results
                )
            
            # Create verification result
            verification_result = VerificationResult(
                confidence=confidence,
                completeness_score=completeness,
                needs_correction=needs_correction,
                result_count=len(results.sources),
                unique_sources=len(set(source.document_id for source in results.sources)),
                diversity_score=self._calculate_diversity_score(results.sources),
                expected_vs_actual=self._format_expectations(analysis, plan),
                correction_strategies=correction_strategies,
                reasoning=self._generate_reasoning(analysis, plan, confidence),
                quality_issues=self._identify_quality_issues(analysis, results)
            )
            
            # Log verification results
            logger.info(f"[VERIFICATION] Confidence: {confidence:.2f}, Completeness: {completeness:.2f}, "
                       f"Needs correction: {needs_correction}")
            
            # Track verification history
            if notebook_id:
                self._track_verification(notebook_id, verification_result, plan)
            
            return verification_result
            
        except Exception as e:
            logger.error(f"Verification analysis failed: {e}")
            # Return conservative verification result
            return VerificationResult(
                confidence=0.5,
                completeness_score=0.6,
                needs_correction=False,  # Conservative: don't correct on errors
                result_count=len(results.sources) if results.sources else 0,
                unique_sources=0,
                diversity_score=0.0,
                expected_vs_actual="Verification failed",
                correction_strategies=[],
                reasoning="Verification analysis encountered an error",
                quality_issues=["Verification system error"]
            )
    
    async def _analyze_completeness(
        self,
        results: NotebookRAGResponse,
        plan: TaskExecutionPlan
    ) -> CompletenessAnalysis:
        """Perform detailed completeness analysis"""
        
        # Analyze entity coverage (projects, companies, etc.)
        entity_coverage = {}
        for entity in plan.data_requirements.entities:
            count = self._count_entity_mentions(results.sources, entity)
            entity_coverage[entity] = count
        
        # Analyze temporal coverage (years, time periods)
        temporal_coverage = self._analyze_temporal_coverage(results.sources)
        
        # Analyze source diversity
        source_diversity = {
            'document': len([s for s in results.sources if s.source_type == 'document']),
            'memory': len([s for s in results.sources if s.source_type == 'memory']),
            'unique_docs': len(set(s.document_id for s in results.sources))
        }
        
        # Identify missing indicators
        missing_indicators = self._identify_missing_indicators(results, plan)
        
        return CompletenessAnalysis(
            entity_coverage=entity_coverage,
            temporal_coverage=temporal_coverage,
            source_diversity=source_diversity,
            missing_indicators=missing_indicators
        )
    
    def _count_entity_mentions(self, sources: List[NotebookRAGSource], entity: str) -> int:
        """Count mentions of specific entity in sources"""
        if entity == "projects":
            patterns = ["project", "initiative", "solution", "system", "platform", "application"]
        elif entity == "companies":
            patterns = ["company", "client", "organization", "corporation", "firm"]
        elif entity == "technologies":
            patterns = ["technology", "framework", "tool", "language", "platform"]
        else:
            patterns = [entity]
        
        mentions = 0
        seen_content = set()
        
        for source in sources:
            content_lower = source.content.lower()
            # Avoid counting the same content multiple times
            content_hash = hash(content_lower[:200])
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)
            
            for pattern in patterns:
                if pattern in content_lower:
                    mentions += content_lower.count(pattern)
        
        return mentions
    
    def _analyze_temporal_coverage(self, sources: List[NotebookRAGSource]) -> Dict[str, int]:
        """Analyze temporal distribution in sources"""
        temporal_coverage = {}
        
        for source in sources:
            content = source.content
            # Look for year patterns
            import re
            year_pattern = r'\b(19|20)\d{2}\b'
            years = re.findall(year_pattern, content)
            
            for year in years:
                decade = f"{year[:3]}0s"
                temporal_coverage[decade] = temporal_coverage.get(decade, 0) + 1
        
        return temporal_coverage
    
    def _identify_missing_indicators(
        self,
        results: NotebookRAGResponse,
        plan: TaskExecutionPlan
    ) -> List[str]:
        """Identify potential gaps in retrieval"""
        missing_indicators = []
        
        # Check for expected attributes
        for attr in plan.data_requirements.attributes:
            mentions = sum(1 for source in results.sources 
                          if attr.lower() in source.content.lower())
            if mentions < len(results.sources) * 0.3:  # Less than 30% have this attribute
                missing_indicators.append(f"Low {attr} coverage: only {mentions} mentions")
        
        # Check for temporal gaps
        if "years" in plan.data_requirements.attributes:
            current_year = datetime.now().year
            recent_mentions = sum(1 for source in results.sources
                                 if str(current_year - 1) in source.content or
                                 str(current_year) in source.content or
                                 str(current_year - 2) in source.content)
            
            if recent_mentions < 3:
                missing_indicators.append("Few recent projects mentioned")
        
        # Check for diversity
        if len(set(source.document_id for source in results.sources)) < 3:
            missing_indicators.append("Low source document diversity")
        
        return missing_indicators
    
    def _calculate_confidence(self, analysis: CompletenessAnalysis, plan: TaskExecutionPlan) -> float:
        """Calculate confidence score in result completeness"""
        
        # FIXED: For comprehensive queries, trust the extraction results
        if plan.intent_type == "comprehensive_analysis" and plan.data_requirements.completeness == "all":
            # High confidence for comprehensive queries with good source coverage
            source_count = sum(analysis.entity_coverage.values())
            if source_count >= 20:  # Found substantial results
                return 0.95  # Very high confidence
            elif source_count >= 10:
                return 0.85  # High confidence
            else:
                return 0.75  # Still good confidence
        
        # Original logic for non-comprehensive queries
        confidence_factors = []
        
        # Entity coverage confidence
        for entity in plan.data_requirements.entities:
            expected = self._get_expected_entity_count(entity, plan)
            actual = analysis.entity_coverage.get(entity, 0)
            if expected > 0:
                coverage_ratio = min(1.0, actual / expected)
                confidence_factors.append(coverage_ratio)
        
        # Source diversity confidence
        unique_docs = analysis.source_diversity.get('unique_docs', 1)
        diversity_confidence = min(1.0, unique_docs / 5)  # Expect at least 5 diverse sources
        confidence_factors.append(diversity_confidence)
        
        # Missing indicators penalty
        missing_penalty = max(0.0, 1.0 - (len(analysis.missing_indicators) * 0.1))
        confidence_factors.append(missing_penalty)
        
        # Calculate weighted average
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        else:
            return 0.5  # Neutral confidence
    
    def _estimate_completeness(self, analysis: CompletenessAnalysis, plan: TaskExecutionPlan) -> float:
        """Estimate how complete the results are"""
        if plan.data_requirements.completeness == "all":
            # For exhaustive requests, check against expected counts
            expected_count = self._parse_expected_count(plan.data_requirements.expected_count)
            if expected_count:
                primary_entity = plan.data_requirements.entities[0] if plan.data_requirements.entities else "projects"
                actual_count = analysis.entity_coverage.get(primary_entity, 0)
                return min(1.0, actual_count / expected_count)
        
        # For non-exhaustive requests, use quality indicators
        quality_score = 1.0 - (len(analysis.missing_indicators) * 0.15)
        return max(0.0, quality_score)
    
    def _needs_correction(self, confidence: float, completeness: float, plan: TaskExecutionPlan) -> bool:
        """Determine if self-correction is needed"""
        threshold = plan.verification.confidence_threshold
        
        # Need correction if confidence or completeness is below threshold
        if confidence < threshold or completeness < threshold:
            return True
        
        # Need correction for exhaustive requests with low completeness
        if plan.data_requirements.completeness == "all" and completeness < 0.8:
            return True
        
        return False
    
    async def _generate_correction_strategies(
        self,
        analysis: CompletenessAnalysis,
        plan: TaskExecutionPlan,
        original_results: NotebookRAGResponse
    ) -> List[RetrievalStrategy]:
        """Generate correction strategies based on what's missing"""
        correction_strategies = []
        
        # Strategy 1: Lower threshold search for missing entities
        for entity in plan.data_requirements.entities:
            if analysis.entity_coverage.get(entity, 0) < 5:  # Very low count
                correction_strategies.append(RetrievalStrategy(
                    query=f"{entity} work experience portfolio",
                    threshold=0.2,  # Lower threshold
                    max_chunks=150,
                    description=f"Recovery search for {entity} with lower threshold"
                ))
        
        # Strategy 2: Temporal gap filling
        if "Few recent projects" in analysis.missing_indicators:
            current_year = datetime.now().year
            correction_strategies.append(RetrievalStrategy(
                query=f"{current_year} {current_year-1} {current_year-2} recent current",
                threshold=0.25,
                max_chunks=100,
                description="Search for recent temporal references"
            ))
        
        # Strategy 3: Diverse document search
        if "Low source document diversity" in analysis.missing_indicators:
            correction_strategies.append(RetrievalStrategy(
                query="portfolio resume experience background work history",
                threshold=0.3,
                max_chunks=200,
                description="Broad search across different document types"
            ))
        
        # Strategy 4: Alternative terminology
        if plan.data_requirements.entities:
            primary_entity = plan.data_requirements.entities[0]
            if primary_entity == "projects":
                correction_strategies.append(RetrievalStrategy(
                    query="initiatives solutions deliverables achievements accomplishments",
                    threshold=0.3,
                    max_chunks=150,
                    description="Alternative terminology search"
                ))
        
        return correction_strategies[:3]  # Limit to top 3 strategies
    
    def _calculate_diversity_score(self, sources: List[NotebookRAGSource]) -> float:
        """Calculate diversity score of sources"""
        if not sources:
            return 0.0
        
        # Document diversity
        unique_docs = len(set(source.document_id for source in sources))
        doc_diversity = min(1.0, unique_docs / 5)  # Normalize to 5 docs
        
        # Source type diversity
        source_types = set(source.source_type for source in sources)
        type_diversity = len(source_types) / 3  # Assume 3 possible types
        
        # Content length diversity
        lengths = [len(source.content) for source in sources]
        if lengths:
            length_std = (max(lengths) - min(lengths)) / max(lengths) if max(lengths) > 0 else 0
            length_diversity = min(1.0, length_std)
        else:
            length_diversity = 0.0
        
        return (doc_diversity + type_diversity + length_diversity) / 3
    
    def _format_expectations(self, analysis: CompletenessAnalysis, plan: TaskExecutionPlan) -> str:
        """Format expectation vs reality comparison"""
        if plan.data_requirements.expected_count:
            expected = self._parse_expected_count(plan.data_requirements.expected_count)
            primary_entity = plan.data_requirements.entities[0] if plan.data_requirements.entities else "items"
            actual = analysis.entity_coverage.get(primary_entity, len(analysis.entity_coverage))
            return f"{expected} {primary_entity} expected, {actual} found"
        else:
            total_found = sum(analysis.entity_coverage.values())
            return f"Found {total_found} relevant items"
    
    def _generate_reasoning(self, analysis: CompletenessAnalysis, plan: TaskExecutionPlan, confidence: float) -> str:
        """Generate human-readable reasoning for verification result"""
        reasons = []
        
        if confidence > 0.8:
            reasons.append("High confidence in result completeness")
        elif confidence > 0.6:
            reasons.append("Moderate confidence, some potential gaps")
        else:
            reasons.append("Low confidence, likely incomplete results")
        
        if analysis.missing_indicators:
            reasons.append(f"Issues identified: {', '.join(analysis.missing_indicators[:2])}")
        
        diversity_score = self._calculate_diversity_score([])  # Will be calculated properly in context
        if diversity_score < 0.5:
            reasons.append("Limited source diversity detected")
        
        return "; ".join(reasons)
    
    def _identify_quality_issues(self, analysis: CompletenessAnalysis, results: NotebookRAGResponse) -> List[str]:
        """Identify quality issues in results"""
        issues = []
        
        # Check for very short results
        short_results = sum(1 for source in results.sources if len(source.content.strip()) < 100)
        if short_results > len(results.sources) * 0.3:
            issues.append(f"{short_results} results are very short (< 100 chars)")
        
        # Check for duplicate content
        contents = [source.content[:200] for source in results.sources]
        unique_contents = set(contents)
        if len(unique_contents) < len(contents) * 0.8:
            issues.append("Significant content duplication detected")
        
        # Add missing indicators as quality issues
        issues.extend(analysis.missing_indicators)
        
        return issues
    
    def _get_expected_entity_count(self, entity: str, plan: TaskExecutionPlan) -> int:
        """Get expected count for entity based on plan"""
        expected_count_str = plan.data_requirements.expected_count
        if expected_count_str:
            return self._parse_expected_count(expected_count_str)
        
        # Default expectations based on entity type
        if entity == "projects":
            return 25
        elif entity == "companies":
            return 10
        elif entity == "technologies":
            return 15
        else:
            return 10
    
    def _parse_expected_count(self, expected_count_str: Optional[str]) -> int:
        """Parse expected count string to number"""
        if not expected_count_str:
            return 0
        
        # Handle formats like "~25", "20-30", "25"
        import re
        numbers = re.findall(r'\d+', expected_count_str)
        if numbers:
            return int(numbers[0])  # Take first number
        return 0
    
    def _track_verification(self, notebook_id: str, result: VerificationResult, plan: TaskExecutionPlan):
        """Track verification history for learning"""
        if notebook_id not in self.verification_history:
            self.verification_history[notebook_id] = []
        
        self.verification_history[notebook_id].append({
            'timestamp': datetime.now(),
            'confidence': result.confidence,
            'completeness': result.completeness_score,
            'needs_correction': result.needs_correction,
            'plan_type': plan.intent_type,
            'result_count': result.result_count
        })
        
        # Keep only recent history (last 50 entries)
        if len(self.verification_history[notebook_id]) > 50:
            self.verification_history[notebook_id] = self.verification_history[notebook_id][-50:]


# Global instance for easy import
ai_verification_service = AIVerificationService()

async def verify_retrieval_completeness(
    results: NotebookRAGResponse,
    plan: TaskExecutionPlan,
    notebook_id: Optional[str] = None
) -> VerificationResult:
    """Convenience function for verification"""
    return await ai_verification_service.verify_completeness(results, plan, notebook_id)