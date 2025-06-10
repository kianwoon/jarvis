"""
Response Quality Evaluator for Self-Reflective RAG

This module evaluates the quality of generated responses to enable
self-reflection and iterative improvement in the RAG pipeline.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class QualityDimension(Enum):
    """Dimensions of response quality to evaluate"""
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    ACCURACY = "accuracy"
    COHERENCE = "coherence"
    SPECIFICITY = "specificity"
    CONFIDENCE = "confidence"


@dataclass
class QualityScore:
    """Quality score for a specific dimension"""
    dimension: QualityDimension
    score: float  # 0.0 to 1.0
    reasoning: str
    issues: List[str]
    suggestions: List[str]


@dataclass
class ResponseEvaluation:
    """Complete evaluation of a response"""
    overall_score: float
    dimension_scores: Dict[QualityDimension, QualityScore]
    needs_refinement: bool
    refinement_suggestions: List[str]
    missing_aspects: List[str]
    confidence_level: float
    metadata: Dict[str, Any]


class ResponseQualityEvaluator:
    """
    Evaluates the quality of RAG responses across multiple dimensions
    to enable self-reflection and iterative improvement.
    """
    
    def __init__(self, llm_client=None, config: Optional[Dict] = None):
        """
        Initialize the evaluator with configuration
        
        Args:
            llm_client: LLM client for evaluation tasks
            config: Configuration dictionary
        """
        self.llm_client = llm_client
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            "min_acceptable_score": 0.7,
            "refinement_threshold": 0.8,
            "max_refinement_iterations": 3,
            "evaluation_prompts": {
                "completeness": "Evaluate if the response fully addresses all aspects of the query.",
                "relevance": "Assess if the response stays on topic and directly answers the question.",
                "accuracy": "Check if the information provided is factually correct based on the context.",
                "coherence": "Evaluate the logical flow and clarity of the response.",
                "specificity": "Assess if the response provides specific, detailed information rather than vague statements.",
                "confidence": "Evaluate the confidence level of the response and identify any uncertainties."
            }
        }
    
    async def evaluate_response(
        self,
        query: str,
        response: str,
        context_documents: List[Dict],
        conversation_history: Optional[List[Dict]] = None
    ) -> ResponseEvaluation:
        """
        Evaluate the quality of a response across multiple dimensions
        
        Args:
            query: Original user query
            response: Generated response
            context_documents: Retrieved documents used for generation
            conversation_history: Previous conversation turns
            
        Returns:
            ResponseEvaluation object with detailed quality assessment
        """
        dimension_scores = {}
        
        # Evaluate each quality dimension
        for dimension in QualityDimension:
            score = await self._evaluate_dimension(
                dimension, query, response, context_documents, conversation_history
            )
            dimension_scores[dimension] = score
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(dimension_scores)
        
        # Determine if refinement is needed
        needs_refinement = self._needs_refinement(overall_score, dimension_scores)
        
        # Generate refinement suggestions
        refinement_suggestions = self._generate_refinement_suggestions(
            dimension_scores, query, response
        )
        
        # Identify missing aspects
        missing_aspects = self._identify_missing_aspects(
            query, response, dimension_scores
        )
        
        # Calculate confidence level
        confidence_level = self._calculate_confidence(dimension_scores)
        
        return ResponseEvaluation(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            needs_refinement=needs_refinement,
            refinement_suggestions=refinement_suggestions,
            missing_aspects=missing_aspects,
            confidence_level=confidence_level,
            metadata={
                "evaluated_dimensions": len(dimension_scores),
                "below_threshold_dimensions": sum(
                    1 for score in dimension_scores.values()
                    if score.score < self.config["min_acceptable_score"]
                )
            }
        )
    
    async def _evaluate_dimension(
        self,
        dimension: QualityDimension,
        query: str,
        response: str,
        context_documents: List[Dict],
        conversation_history: Optional[List[Dict]]
    ) -> QualityScore:
        """Evaluate a specific quality dimension"""
        
        if dimension == QualityDimension.COMPLETENESS:
            return self._evaluate_completeness(query, response)
        elif dimension == QualityDimension.RELEVANCE:
            return self._evaluate_relevance(query, response, context_documents)
        elif dimension == QualityDimension.ACCURACY:
            return self._evaluate_accuracy(response, context_documents)
        elif dimension == QualityDimension.COHERENCE:
            return self._evaluate_coherence(response)
        elif dimension == QualityDimension.SPECIFICITY:
            return self._evaluate_specificity(response)
        elif dimension == QualityDimension.CONFIDENCE:
            return self._evaluate_confidence(response)
        else:
            raise ValueError(f"Unknown dimension: {dimension}")
    
    def _evaluate_completeness(self, query: str, response: str) -> QualityScore:
        """Evaluate if response addresses all aspects of the query"""
        score = 1.0
        issues = []
        suggestions = []
        
        # Extract key aspects from query
        query_aspects = self._extract_query_aspects(query)
        
        # Check if each aspect is addressed
        addressed_aspects = []
        for aspect in query_aspects:
            if self._is_aspect_addressed(aspect, response):
                addressed_aspects.append(aspect)
            else:
                score -= 0.2
                issues.append(f"Missing aspect: {aspect}")
                suggestions.append(f"Address the aspect: {aspect}")
        
        # Check for partial answers
        if self._contains_hedging_language(response):
            score -= 0.1
            issues.append("Response contains hedging or uncertain language")
            suggestions.append("Provide more definitive answers where possible")
        
        score = max(0.0, score)
        
        return QualityScore(
            dimension=QualityDimension.COMPLETENESS,
            score=score,
            reasoning=f"Addressed {len(addressed_aspects)}/{len(query_aspects)} query aspects",
            issues=issues,
            suggestions=suggestions
        )
    
    def _evaluate_relevance(
        self, query: str, response: str, context_documents: List[Dict]
    ) -> QualityScore:
        """Evaluate response relevance to query and context"""
        score = 1.0
        issues = []
        suggestions = []
        
        # Check topic alignment
        query_keywords = self._extract_keywords(query)
        response_keywords = self._extract_keywords(response)
        
        keyword_overlap = len(
            set(query_keywords) & set(response_keywords)
        ) / max(len(query_keywords), 1)
        
        if keyword_overlap < 0.3:
            score -= 0.3
            issues.append("Low keyword overlap with query")
            suggestions.append("Focus more on the specific terms in the query")
        
        # Check for off-topic content
        if self._contains_off_topic_content(response, query):
            score -= 0.2
            issues.append("Response contains off-topic information")
            suggestions.append("Remove tangential information")
        
        score = max(0.0, score)
        
        return QualityScore(
            dimension=QualityDimension.RELEVANCE,
            score=score,
            reasoning=f"Keyword overlap: {keyword_overlap:.2f}",
            issues=issues,
            suggestions=suggestions
        )
    
    def _evaluate_accuracy(
        self, response: str, context_documents: List[Dict]
    ) -> QualityScore:
        """Evaluate factual accuracy against context documents"""
        score = 1.0
        issues = []
        suggestions = []
        
        # Extract claims from response
        claims = self._extract_claims(response)
        
        # Verify each claim against context
        unverified_claims = []
        for claim in claims:
            if not self._verify_claim(claim, context_documents):
                unverified_claims.append(claim)
                score -= 0.15
                issues.append(f"Unverified claim: {claim}")
        
        if unverified_claims:
            suggestions.append("Ensure all claims are supported by retrieved documents")
        
        # Check for potential hallucinations
        if self._detect_potential_hallucination(response, context_documents):
            score -= 0.3
            issues.append("Response may contain hallucinated information")
            suggestions.append("Stick closely to information in the context documents")
        
        score = max(0.0, score)
        
        return QualityScore(
            dimension=QualityDimension.ACCURACY,
            score=score,
            reasoning=f"Verified {len(claims) - len(unverified_claims)}/{len(claims)} claims",
            issues=issues,
            suggestions=suggestions
        )
    
    def _evaluate_coherence(self, response: str) -> QualityScore:
        """Evaluate logical flow and clarity"""
        score = 1.0
        issues = []
        suggestions = []
        
        # Check sentence structure
        sentences = response.split('.')
        
        # Check for very short or very long sentences
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        if avg_sentence_length < 5:
            score -= 0.2
            issues.append("Sentences are too short and choppy")
            suggestions.append("Combine related ideas into more flowing sentences")
        elif avg_sentence_length > 30:
            score -= 0.2
            issues.append("Sentences are too long and complex")
            suggestions.append("Break down complex sentences for clarity")
        
        # Check for logical connectors
        connectors = ['therefore', 'however', 'moreover', 'furthermore', 'additionally']
        has_connectors = any(connector in response.lower() for connector in connectors)
        
        if not has_connectors and len(sentences) > 3:
            score -= 0.1
            issues.append("Lacks logical connectors between ideas")
            suggestions.append("Use transitional phrases to connect ideas")
        
        score = max(0.0, score)
        
        return QualityScore(
            dimension=QualityDimension.COHERENCE,
            score=score,
            reasoning=f"Average sentence length: {avg_sentence_length:.1f} words",
            issues=issues,
            suggestions=suggestions
        )
    
    def _evaluate_specificity(self, response: str) -> QualityScore:
        """Evaluate level of specific detail vs vague statements"""
        score = 1.0
        issues = []
        suggestions = []
        
        # Check for vague terms
        vague_terms = [
            'some', 'many', 'few', 'various', 'certain', 'several',
            'might', 'could', 'possibly', 'maybe', 'perhaps'
        ]
        
        vague_count = sum(
            1 for term in vague_terms 
            if term in response.lower().split()
        )
        
        if vague_count > 3:
            score -= 0.3
            issues.append(f"Contains {vague_count} vague terms")
            suggestions.append("Replace vague terms with specific information")
        
        # Check for specific examples or numbers
        has_numbers = bool(re.search(r'\d+', response))
        has_examples = 'example' in response.lower() or 'for instance' in response.lower()
        
        if not has_numbers and not has_examples and len(response) > 100:
            score -= 0.2
            issues.append("Lacks specific examples or quantitative information")
            suggestions.append("Add concrete examples or specific data points")
        
        score = max(0.0, score)
        
        return QualityScore(
            dimension=QualityDimension.SPECIFICITY,
            score=score,
            reasoning=f"Vague terms: {vague_count}, Has specifics: {has_numbers or has_examples}",
            issues=issues,
            suggestions=suggestions
        )
    
    def _evaluate_confidence(self, response: str) -> QualityScore:
        """Evaluate confidence level and uncertainty handling"""
        score = 1.0
        issues = []
        suggestions = []
        
        # Check for uncertainty markers
        uncertainty_markers = [
            'i think', 'i believe', 'probably', 'likely', 'uncertain',
            'not sure', 'may be', 'might be', 'could be'
        ]
        
        uncertainty_count = sum(
            1 for marker in uncertainty_markers
            if marker in response.lower()
        )
        
        if uncertainty_count > 2:
            score -= 0.2
            issues.append(f"High uncertainty ({uncertainty_count} markers)")
            suggestions.append("Verify uncertain information or clarify limitations")
        
        # Check for appropriate caveats
        has_caveats = any(
            phrase in response.lower()
            for phrase in ['however', 'note that', 'keep in mind', 'important to']
        )
        
        if not has_caveats and len(response) > 200:
            score -= 0.1
            issues.append("Lacks appropriate caveats or limitations")
            suggestions.append("Add relevant caveats where appropriate")
        
        score = max(0.0, score)
        
        return QualityScore(
            dimension=QualityDimension.CONFIDENCE,
            score=score,
            reasoning=f"Uncertainty markers: {uncertainty_count}",
            issues=issues,
            suggestions=suggestions
        )
    
    def _calculate_overall_score(
        self, dimension_scores: Dict[QualityDimension, QualityScore]
    ) -> float:
        """Calculate weighted overall score"""
        weights = {
            QualityDimension.COMPLETENESS: 0.25,
            QualityDimension.RELEVANCE: 0.25,
            QualityDimension.ACCURACY: 0.20,
            QualityDimension.COHERENCE: 0.10,
            QualityDimension.SPECIFICITY: 0.10,
            QualityDimension.CONFIDENCE: 0.10
        }
        
        weighted_sum = sum(
            dimension_scores[dim].score * weights.get(dim, 0.1)
            for dim in dimension_scores
        )
        
        return weighted_sum
    
    def _needs_refinement(
        self, overall_score: float, dimension_scores: Dict[QualityDimension, QualityScore]
    ) -> bool:
        """Determine if response needs refinement"""
        # Check overall threshold
        if overall_score < self.config["refinement_threshold"]:
            return True
        
        # Check critical dimensions
        critical_dimensions = [
            QualityDimension.COMPLETENESS,
            QualityDimension.ACCURACY,
            QualityDimension.RELEVANCE
        ]
        
        for dim in critical_dimensions:
            if dim in dimension_scores and dimension_scores[dim].score < self.config["min_acceptable_score"]:
                return True
        
        return False
    
    def _generate_refinement_suggestions(
        self,
        dimension_scores: Dict[QualityDimension, QualityScore],
        query: str,
        response: str
    ) -> List[str]:
        """Generate specific suggestions for refinement"""
        suggestions = []
        
        # Collect suggestions from low-scoring dimensions
        for dim, score in dimension_scores.items():
            if score.score < self.config["refinement_threshold"]:
                suggestions.extend(score.suggestions)
        
        # Add query-specific suggestions
        if not suggestions:
            suggestions.append("Consider expanding on the main points")
        
        # Prioritize suggestions
        return suggestions[:5]  # Return top 5 suggestions
    
    def _identify_missing_aspects(
        self,
        query: str,
        response: str,
        dimension_scores: Dict[QualityDimension, QualityScore]
    ) -> List[str]:
        """Identify aspects of the query not addressed"""
        missing = []
        
        # Get from completeness evaluation
        if QualityDimension.COMPLETENESS in dimension_scores:
            completeness_score = dimension_scores[QualityDimension.COMPLETENESS]
            for issue in completeness_score.issues:
                if "Missing aspect:" in issue:
                    missing.append(issue.replace("Missing aspect:", "").strip())
        
        return missing
    
    def _calculate_confidence(
        self, dimension_scores: Dict[QualityDimension, QualityScore]
    ) -> float:
        """Calculate overall confidence level"""
        # Use confidence dimension score if available
        if QualityDimension.CONFIDENCE in dimension_scores:
            base_confidence = dimension_scores[QualityDimension.CONFIDENCE].score
        else:
            base_confidence = 0.5
        
        # Adjust based on other dimensions
        accuracy_score = dimension_scores.get(QualityDimension.ACCURACY, QualityScore(
            dimension=QualityDimension.ACCURACY, score=0.5, reasoning="", issues=[], suggestions=[]
        )).score
        
        completeness_score = dimension_scores.get(QualityDimension.COMPLETENESS, QualityScore(
            dimension=QualityDimension.COMPLETENESS, score=0.5, reasoning="", issues=[], suggestions=[]
        )).score
        
        # Weighted confidence calculation
        confidence = (base_confidence * 0.4 + accuracy_score * 0.4 + completeness_score * 0.2)
        
        return min(1.0, max(0.0, confidence))
    
    # Helper methods
    def _extract_query_aspects(self, query: str) -> List[str]:
        """Extract key aspects from the query"""
        aspects = []
        
        # Look for question words and their associated phrases
        question_patterns = [
            r'what\s+(\w+\s+\w+)',
            r'how\s+(\w+\s+\w+)',
            r'why\s+(\w+\s+\w+)',
            r'when\s+(\w+\s+\w+)',
            r'where\s+(\w+\s+\w+)',
            r'who\s+(\w+\s+\w+)'
        ]
        
        for pattern in question_patterns:
            matches = re.findall(pattern, query.lower())
            aspects.extend(matches)
        
        # Look for "and" separated items
        if ' and ' in query:
            parts = query.split(' and ')
            aspects.extend([part.strip() for part in parts if len(part.strip()) > 3])
        
        return list(set(aspects))  # Remove duplicates
    
    def _is_aspect_addressed(self, aspect: str, response: str) -> bool:
        """Check if an aspect is addressed in the response"""
        aspect_words = aspect.lower().split()
        response_lower = response.lower()
        
        # Check if key words from aspect appear in response
        matches = sum(1 for word in aspect_words if word in response_lower)
        
        return matches >= len(aspect_words) * 0.5
    
    def _contains_hedging_language(self, text: str) -> bool:
        """Check for hedging or uncertain language"""
        hedging_phrases = [
            'might be', 'could be', 'possibly', 'perhaps', 'maybe',
            'it seems', 'appears to', 'likely', 'probably'
        ]
        
        return any(phrase in text.lower() for phrase in hedging_phrases)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Remove common words
        stop_words = {'the', 'is', 'are', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        
        words = text.lower().split()
        keywords = [
            word.strip('.,!?;:')
            for word in words
            if len(word) > 3 and word not in stop_words
        ]
        
        return keywords
    
    def _contains_off_topic_content(self, response: str, query: str) -> bool:
        """Detect if response contains off-topic content"""
        # This is a simplified check - in production, use more sophisticated methods
        query_keywords = set(self._extract_keywords(query))
        response_sentences = response.split('.')
        
        off_topic_sentences = 0
        for sentence in response_sentences:
            sentence_keywords = set(self._extract_keywords(sentence))
            if not sentence_keywords & query_keywords and len(sentence) > 20:
                off_topic_sentences += 1
        
        return off_topic_sentences > len(response_sentences) * 0.3
    
    def _extract_claims(self, response: str) -> List[str]:
        """Extract factual claims from response"""
        claims = []
        
        # Look for definitive statements
        patterns = [
            r'(?:is|are|was|were)\s+([^.]+)',
            r'(?:has|have|had)\s+([^.]+)',
            r'(?:can|could|will|would)\s+([^.]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response)
            claims.extend([match.strip() for match in matches if len(match) > 10])
        
        return claims[:10]  # Limit to first 10 claims
    
    def _verify_claim(self, claim: str, context_documents: List[Dict]) -> bool:
        """Verify if a claim is supported by context documents"""
        claim_keywords = set(self._extract_keywords(claim))
        
        for doc in context_documents:
            doc_text = doc.get('content', '').lower()
            doc_keywords = set(self._extract_keywords(doc_text))
            
            # Check keyword overlap
            if len(claim_keywords & doc_keywords) >= len(claim_keywords) * 0.5:
                return True
        
        return False
    
    def _detect_potential_hallucination(
        self, response: str, context_documents: List[Dict]
    ) -> bool:
        """Detect potential hallucinations in response"""
        # Extract specific facts from response
        specific_patterns = [
            r'\d+%',  # Percentages
            r'\d{4}',  # Years
            r'"\w+"',  # Quoted terms
            r'[A-Z]\w+\s+[A-Z]\w+'  # Proper nouns
        ]
        
        response_specifics = []
        for pattern in specific_patterns:
            response_specifics.extend(re.findall(pattern, response))
        
        # Check if specifics appear in context
        all_context = ' '.join(doc.get('content', '') for doc in context_documents)
        
        unverified_specifics = 0
        for specific in response_specifics:
            if specific not in all_context:
                unverified_specifics += 1
        
        return unverified_specifics > len(response_specifics) * 0.3