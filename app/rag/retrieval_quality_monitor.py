"""
Retrieval Quality Monitor for Self-Reflective RAG

This module monitors and evaluates the quality of document retrieval
to identify when additional or different retrieval strategies are needed.
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class RetrievalIssue(Enum):
    """Types of retrieval quality issues"""
    LOW_RELEVANCE = "low_relevance"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"
    HIGH_REDUNDANCY = "high_redundancy"
    TOPIC_DRIFT = "topic_drift"
    MISSING_CONTEXT = "missing_context"
    OUTDATED_INFORMATION = "outdated_information"


@dataclass
class DocumentRelevance:
    """Relevance assessment for a single document"""
    doc_id: str
    relevance_score: float
    key_matches: List[str]
    missing_aspects: List[str]
    confidence: float
    metadata: Dict


@dataclass
class RetrievalAssessment:
    """Complete assessment of retrieval quality"""
    overall_quality: float
    coverage_score: float
    relevance_distribution: Dict[str, float]
    identified_issues: List[RetrievalIssue]
    missing_information: List[str]
    redundant_documents: List[Tuple[str, str]]  # Pairs of similar docs
    improvement_suggestions: List[str]
    needs_reretrieval: bool
    alternative_queries: List[str]
    metadata: Dict


class RetrievalQualityMonitor:
    """
    Monitors the quality of retrieved documents to enable
    self-reflection and iterative improvement in retrieval.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the monitor with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            "min_relevance_threshold": 0.7,
            "min_coverage_threshold": 0.8,
            "redundancy_threshold": 0.85,
            "reretrieval_threshold": 0.6,
            "max_documents": 10,
            "diversity_weight": 0.2
        }
    
    def assess_retrieval_quality(
        self,
        query: str,
        retrieved_documents: List[Dict],
        scores: Optional[List[float]] = None
    ) -> RetrievalAssessment:
        """
        Assess the quality of retrieved documents
        
        Args:
            query: Original search query
            retrieved_documents: List of retrieved documents with content and metadata
            scores: Optional relevance scores from the retrieval system
            
        Returns:
            RetrievalAssessment with detailed quality metrics
        """
        # Extract query aspects
        query_aspects = self._extract_query_aspects(query)
        
        # Assess individual document relevance
        doc_relevances = []
        for i, doc in enumerate(retrieved_documents):
            relevance = self._assess_document_relevance(
                doc, query, query_aspects,
                scores[i] if scores and i < len(scores) else None
            )
            doc_relevances.append(relevance)
        
        # Calculate coverage
        coverage_score, missing_info = self._calculate_coverage(
            query_aspects, doc_relevances
        )
        
        # Identify redundancy
        redundant_pairs = self._identify_redundancy(retrieved_documents)
        
        # Calculate relevance distribution
        relevance_dist = self._calculate_relevance_distribution(doc_relevances)
        
        # Identify issues
        issues = self._identify_retrieval_issues(
            doc_relevances, coverage_score, redundant_pairs, relevance_dist
        )
        
        # Calculate overall quality
        overall_quality = self._calculate_overall_quality(
            doc_relevances, coverage_score, len(redundant_pairs)
        )
        
        # Determine if re-retrieval is needed
        needs_reretrieval = self._needs_reretrieval(
            overall_quality, coverage_score, issues
        )
        
        # Generate improvement suggestions
        suggestions = self._generate_improvement_suggestions(
            issues, coverage_score, doc_relevances
        )
        
        # Generate alternative queries
        alt_queries = self._generate_alternative_queries(
            query, missing_info, issues
        )
        
        return RetrievalAssessment(
            overall_quality=overall_quality,
            coverage_score=coverage_score,
            relevance_distribution=relevance_dist,
            identified_issues=issues,
            missing_information=missing_info,
            redundant_documents=redundant_pairs,
            improvement_suggestions=suggestions,
            needs_reretrieval=needs_reretrieval,
            alternative_queries=alt_queries,
            metadata={
                "total_documents": len(retrieved_documents),
                "high_relevance_count": sum(
                    1 for r in doc_relevances 
                    if r.relevance_score >= self.config["min_relevance_threshold"]
                ),
                "query_aspects": len(query_aspects)
            }
        )
    
    def _extract_query_aspects(self, query: str) -> List[str]:
        """Extract key aspects from the query"""
        aspects = []
        
        # Extract keywords (simplified - in production use NLP)
        words = query.lower().split()
        
        # Remove stop words
        stop_words = {
            'the', 'is', 'are', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'what', 'how', 'when', 'where', 'why', 'who'
        }
        
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        aspects.extend(keywords)
        
        # Extract phrases (simplified)
        if 'and' in query:
            parts = query.split('and')
            aspects.extend([p.strip() for p in parts if len(p.strip()) > 5])
        
        return list(set(aspects))
    
    def _assess_document_relevance(
        self,
        document: Dict,
        query: str,
        query_aspects: List[str],
        retrieval_score: Optional[float]
    ) -> DocumentRelevance:
        """Assess relevance of a single document"""
        content = document.get('content', '').lower()
        doc_id = document.get('id', str(hash(content)))
        
        # Find matching aspects
        key_matches = []
        missing_aspects = []
        
        for aspect in query_aspects:
            if aspect.lower() in content:
                key_matches.append(aspect)
            else:
                missing_aspects.append(aspect)
        
        # Calculate relevance score
        if query_aspects:
            aspect_coverage = len(key_matches) / len(query_aspects)
        else:
            aspect_coverage = 0.5
        
        # Consider retrieval score if available
        if retrieval_score is not None:
            relevance_score = 0.6 * aspect_coverage + 0.4 * retrieval_score
        else:
            relevance_score = aspect_coverage
        
        # Calculate confidence
        confidence = self._calculate_relevance_confidence(
            key_matches, missing_aspects, content
        )
        
        return DocumentRelevance(
            doc_id=doc_id,
            relevance_score=relevance_score,
            key_matches=key_matches,
            missing_aspects=missing_aspects,
            confidence=confidence,
            metadata={
                "content_length": len(content),
                "source": document.get('source', 'unknown')
            }
        )
    
    def _calculate_coverage(
        self, query_aspects: List[str], doc_relevances: List[DocumentRelevance]
    ) -> Tuple[float, List[str]]:
        """Calculate how well documents cover query aspects"""
        if not query_aspects:
            return 1.0, []
        
        # Collect all matched aspects across documents
        all_matched = set()
        for relevance in doc_relevances:
            all_matched.update(relevance.key_matches)
        
        # Calculate coverage
        coverage = len(all_matched) / len(query_aspects)
        
        # Identify missing information
        missing = [
            aspect for aspect in query_aspects
            if aspect not in all_matched
        ]
        
        return coverage, missing
    
    def _identify_redundancy(self, documents: List[Dict]) -> List[Tuple[str, str]]:
        """Identify redundant document pairs"""
        redundant_pairs = []
        
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                similarity = self._calculate_similarity(
                    documents[i].get('content', ''),
                    documents[j].get('content', '')
                )
                
                if similarity > self.config["redundancy_threshold"]:
                    redundant_pairs.append((
                        documents[i].get('id', str(i)),
                        documents[j].get('id', str(j))
                    ))
        
        return redundant_pairs
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (simplified)"""
        # Tokenize
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Jaccard similarity
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _calculate_relevance_distribution(
        self, doc_relevances: List[DocumentRelevance]
    ) -> Dict[str, float]:
        """Calculate distribution statistics of relevance scores"""
        scores = [r.relevance_score for r in doc_relevances]
        
        if not scores:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0
            }
        
        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "median": float(np.median(scores))
        }
    
    def _identify_retrieval_issues(
        self,
        doc_relevances: List[DocumentRelevance],
        coverage_score: float,
        redundant_pairs: List[Tuple[str, str]],
        relevance_dist: Dict[str, float]
    ) -> List[RetrievalIssue]:
        """Identify specific retrieval quality issues"""
        issues = []
        
        # Check relevance
        if relevance_dist["mean"] < self.config["min_relevance_threshold"]:
            issues.append(RetrievalIssue.LOW_RELEVANCE)
        
        # Check coverage
        if coverage_score < self.config["min_coverage_threshold"]:
            issues.append(RetrievalIssue.INSUFFICIENT_COVERAGE)
        
        # Check redundancy
        if len(redundant_pairs) > len(doc_relevances) * 0.2:
            issues.append(RetrievalIssue.HIGH_REDUNDANCY)
        
        # Check topic drift (high variance in relevance)
        if relevance_dist["std"] > 0.3:
            issues.append(RetrievalIssue.TOPIC_DRIFT)
        
        # Check for missing context (all docs have low confidence)
        avg_confidence = np.mean([r.confidence for r in doc_relevances])
        if avg_confidence < 0.5:
            issues.append(RetrievalIssue.MISSING_CONTEXT)
        
        return issues
    
    def _calculate_overall_quality(
        self,
        doc_relevances: List[DocumentRelevance],
        coverage_score: float,
        redundant_count: int
    ) -> float:
        """Calculate overall retrieval quality score"""
        if not doc_relevances:
            return 0.0
        
        # Average relevance
        avg_relevance = np.mean([r.relevance_score for r in doc_relevances])
        
        # Diversity penalty for redundancy
        diversity_score = 1.0 - (redundant_count / max(len(doc_relevances), 1))
        
        # Weighted combination
        quality = (
            0.4 * avg_relevance +
            0.4 * coverage_score +
            0.2 * diversity_score
        )
        
        return float(quality)
    
    def _needs_reretrieval(
        self,
        overall_quality: float,
        coverage_score: float,
        issues: List[RetrievalIssue]
    ) -> bool:
        """Determine if re-retrieval is needed"""
        # Check quality threshold
        if overall_quality < self.config["reretrieval_threshold"]:
            return True
        
        # Check critical issues
        critical_issues = {
            RetrievalIssue.INSUFFICIENT_COVERAGE,
            RetrievalIssue.LOW_RELEVANCE,
            RetrievalIssue.MISSING_CONTEXT
        }
        
        if any(issue in critical_issues for issue in issues):
            return True
        
        return False
    
    def _generate_improvement_suggestions(
        self,
        issues: List[RetrievalIssue],
        coverage_score: float,
        doc_relevances: List[DocumentRelevance]
    ) -> List[str]:
        """Generate suggestions for improving retrieval"""
        suggestions = []
        
        if RetrievalIssue.LOW_RELEVANCE in issues:
            suggestions.append("Refine search query with more specific terms")
            suggestions.append("Consider using different search strategies")
        
        if RetrievalIssue.INSUFFICIENT_COVERAGE in issues:
            suggestions.append("Expand search to include related terms")
            suggestions.append("Try breaking down complex queries")
        
        if RetrievalIssue.HIGH_REDUNDANCY in issues:
            suggestions.append("Increase diversity in retrieval results")
            suggestions.append("Filter out duplicate content")
        
        if RetrievalIssue.TOPIC_DRIFT in issues:
            suggestions.append("Use more precise search terms")
            suggestions.append("Apply stricter relevance filtering")
        
        if RetrievalIssue.MISSING_CONTEXT in issues:
            suggestions.append("Search for background information")
            suggestions.append("Include context terms in query")
        
        return suggestions
    
    def _generate_alternative_queries(
        self,
        original_query: str,
        missing_info: List[str],
        issues: List[RetrievalIssue]
    ) -> List[str]:
        """Generate alternative queries for better retrieval"""
        alternatives = []
        
        # Add missing aspects
        if missing_info:
            for info in missing_info[:2]:  # Limit to 2
                alternatives.append(f"{original_query} {info}")
        
        # Simplify if coverage is low
        if RetrievalIssue.INSUFFICIENT_COVERAGE in issues:
            # Extract core terms
            words = original_query.split()
            if len(words) > 3:
                core_query = ' '.join(words[:3])
                alternatives.append(core_query)
        
        # Add context if missing
        if RetrievalIssue.MISSING_CONTEXT in issues:
            alternatives.append(f"background context {original_query}")
            alternatives.append(f"overview of {original_query}")
        
        # Rephrase for better relevance
        if RetrievalIssue.LOW_RELEVANCE in issues:
            alternatives.append(f"specific information about {original_query}")
            alternatives.append(f"detailed {original_query}")
        
        return alternatives[:4]  # Limit to 4 alternatives
    
    def _calculate_relevance_confidence(
        self, key_matches: List[str], missing_aspects: List[str], content: str
    ) -> float:
        """Calculate confidence in relevance assessment"""
        if not key_matches and not missing_aspects:
            return 0.5
        
        # Base confidence on match ratio
        total_aspects = len(key_matches) + len(missing_aspects)
        match_ratio = len(key_matches) / total_aspects
        
        # Adjust for content length (longer content might have more confidence)
        length_factor = min(1.0, len(content) / 1000)
        
        confidence = 0.7 * match_ratio + 0.3 * length_factor
        
        return confidence
    
    def compare_retrieval_strategies(
        self,
        query: str,
        strategy_results: Dict[str, List[Dict]]
    ) -> Dict[str, RetrievalAssessment]:
        """
        Compare multiple retrieval strategies
        
        Args:
            query: Search query
            strategy_results: Dict mapping strategy names to retrieved documents
            
        Returns:
            Dict mapping strategy names to their assessments
        """
        assessments = {}
        
        for strategy_name, documents in strategy_results.items():
            assessment = self.assess_retrieval_quality(query, documents)
            assessments[strategy_name] = assessment
        
        return assessments
    
    def recommend_best_strategy(
        self, assessments: Dict[str, RetrievalAssessment]
    ) -> Tuple[str, RetrievalAssessment]:
        """
        Recommend the best retrieval strategy based on assessments
        
        Args:
            assessments: Dict of strategy assessments
            
        Returns:
            Tuple of (best_strategy_name, best_assessment)
        """
        if not assessments:
            raise ValueError("No assessments provided")
        
        # Score each strategy
        strategy_scores = {}
        for name, assessment in assessments.items():
            # Weighted scoring
            score = (
                0.4 * assessment.overall_quality +
                0.4 * assessment.coverage_score +
                0.2 * (1.0 - len(assessment.identified_issues) / 10)
            )
            strategy_scores[name] = score
        
        # Find best strategy
        best_strategy = max(strategy_scores, key=strategy_scores.get)
        
        return best_strategy, assessments[best_strategy]