"""
Temporal Relevance Engine

Main orchestrator for temporal relevance scoring that combines:
- Query classification for temporal sensitivity
- Temporal decay functions for freshness scoring
- Source authority for credibility scoring
- Intent detection for user needs
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from app.core.temporal_query_classifier import (
    TemporalQueryClassifier, 
    TemporalClassification,
    TemporalSensitivity,
    get_temporal_classifier
)
from app.core.temporal_decay_functions import (
    calculate_temporal_score,
    DomainDecayProfiles
)
from app.core.temporal_source_authority import (
    TemporalSourceAuthorityScorer,
    SourceAuthority,
    get_source_scorer
)

logger = logging.getLogger(__name__)


@dataclass
class TemporalRelevanceScore:
    """Complete temporal relevance assessment for a document"""
    document_id: str
    url: str
    title: str
    snippet: str
    age_days: float
    temporal_score: float  # How fresh/relevant based on age
    authority_score: float  # How credible the source is
    combined_score: float  # Overall relevance score
    should_include: bool  # Whether to include in results
    rank_boost: float  # Boost factor for ranking
    metadata: Dict[str, Any]


class TemporalRelevanceEngine:
    """
    Main engine for temporal relevance scoring and filtering
    
    This engine:
    1. Analyzes queries for temporal sensitivity
    2. Scores documents based on age and source authority
    3. Filters out outdated information
    4. Provides ranking boosts for relevant content
    """
    
    def __init__(self):
        """Initialize the temporal relevance engine"""
        self.query_classifier = get_temporal_classifier()
        self.source_scorer = get_source_scorer()
        self.current_date = datetime.now()
        
        # Thresholds for filtering
        self.min_temporal_score = 0.1  # Minimum temporal score to include
        self.min_authority_score = 0.2  # Minimum authority score to include
        self.min_combined_score = 0.15  # Minimum combined score to include
        
        # Intent detection patterns
        self.intent_patterns = {
            "current": [
                r"\b(current|latest|now|today|recent|updated|modern|2025)\b",
                r"\b(what is|what are|how much does|how many)\b",
                r"\b(this year|this month|this week)\b"
            ],
            "historical": [
                r"\b(history|historical|past|originally|previously)\b",
                r"\b(was|were|had|did|used to)\b",
                r"\b(timeline|evolution|changes over time)\b",
                r"\bin\s+(19\d{2}|20[012]\d)\b"  # "in 2023" etc
            ],
            "comparison": [
                r"\b(vs|versus|compared to|difference between)\b",
                r"\b(better than|worse than|improvement)\b"
            ]
        }
    
    def analyze_query(self, query: str) -> TemporalClassification:
        """
        Analyze a query for temporal sensitivity and intent
        
        Args:
            query: The search query
            
        Returns:
            TemporalClassification with sensitivity and metadata
        """
        classification = self.query_classifier.classify(query)
        
        # Override intent detection if needed
        detected_intent = self._detect_intent(query)
        if detected_intent and detected_intent != classification.intent:
            logger.info(f"Overriding intent from {classification.intent} to {detected_intent}")
            classification.intent = detected_intent
        
        logger.info(f"Query classification: sensitivity={classification.sensitivity.value}, "
                   f"domain={classification.domain}, intent={classification.intent}, "
                   f"max_age={classification.max_age_days} days")
        
        return classification
    
    def score_document(
        self,
        document: Dict[str, Any],
        classification: TemporalClassification,
        document_date: Optional[datetime] = None
    ) -> TemporalRelevanceScore:
        """
        Score a single document for temporal relevance
        
        Args:
            document: Document with url, title, snippet, date
            classification: Query classification
            document_date: Optional parsed document date
            
        Returns:
            TemporalRelevanceScore with all scoring details
        """
        url = document.get("url", "")
        title = document.get("title", "")
        snippet = document.get("snippet", "")
        
        # Calculate document age
        if document_date:
            age_days = (self.current_date - document_date).days
        else:
            # Try to extract date from document
            extracted_date = self._extract_date_from_document(document)
            if extracted_date:
                age_days = (self.current_date - extracted_date).days
            else:
                # Default to assuming it's recent if no date found
                age_days = 30
        
        # Ensure non-negative age
        age_days = max(0, age_days)
        
        # Calculate temporal decay score
        temporal_score = calculate_temporal_score(
            age_days=age_days,
            domain=classification.domain
        )
        
        # Calculate source authority score
        source_authority = self.source_scorer.score_source(
            url=url,
            title=title,
            snippet=snippet
        )
        
        # Combine scores based on intent
        combined_score = self.source_scorer.get_combined_score(
            temporal_score=temporal_score,
            authority_score=source_authority.authority_score,
            intent=classification.intent
        )
        
        # Determine if document should be included
        should_include = self._should_include_document(
            temporal_score=temporal_score,
            authority_score=source_authority.authority_score,
            combined_score=combined_score,
            age_days=age_days,
            classification=classification
        )
        
        # Calculate rank boost
        rank_boost = self._calculate_rank_boost(
            temporal_score=temporal_score,
            authority_score=source_authority.authority_score,
            classification=classification,
            is_primary_source=source_authority.is_primary_source
        )
        
        return TemporalRelevanceScore(
            document_id=document.get("id", str(hash(url))),
            url=url,
            title=title,
            snippet=snippet,
            age_days=age_days,
            temporal_score=temporal_score,
            authority_score=source_authority.authority_score,
            combined_score=combined_score,
            should_include=should_include,
            rank_boost=rank_boost,
            metadata={
                "source_type": source_authority.source_type,
                "is_primary_source": source_authority.is_primary_source,
                "domain": source_authority.domain,
                "query_domain": classification.domain,
                "query_intent": classification.intent,
                "sensitivity": classification.sensitivity.value
            }
        )
    
    def filter_and_rank_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
        max_results: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Filter and rank search results based on temporal relevance
        
        Args:
            results: List of search results
            query: The original query
            max_results: Maximum number of results to return
            
        Returns:
            Tuple of (filtered_results, metadata)
        """
        # Analyze query
        classification = self.analyze_query(query)
        
        # Score all documents
        scored_documents = []
        for result in results:
            score = self.score_document(result, classification)
            if score.should_include:
                # Add scoring metadata to result
                result["temporal_relevance"] = {
                    "age_days": score.age_days,
                    "temporal_score": score.temporal_score,
                    "authority_score": score.authority_score,
                    "combined_score": score.combined_score,
                    "rank_boost": score.rank_boost
                }
                result["relevance_score"] = score.combined_score
                scored_documents.append((score, result))
        
        # Sort by combined score with rank boost
        scored_documents.sort(
            key=lambda x: x[0].combined_score * x[0].rank_boost,
            reverse=True
        )
        
        # Extract results
        filtered_results = [doc for _, doc in scored_documents]
        
        # Limit results if specified
        if max_results:
            filtered_results = filtered_results[:max_results]
        
        # Build metadata
        metadata = {
            "total_results": len(results),
            "filtered_results": len(filtered_results),
            "query_classification": {
                "sensitivity": classification.sensitivity.value,
                "domain": classification.domain,
                "intent": classification.intent,
                "max_age_days": classification.max_age_days
            },
            "filtering_stats": {
                "removed_outdated": len(results) - len(filtered_results),
                "average_age_days": sum(s.age_days for s, _ in scored_documents) / len(scored_documents) if scored_documents else 0,
                "average_temporal_score": sum(s.temporal_score for s, _ in scored_documents) / len(scored_documents) if scored_documents else 0,
                "average_authority_score": sum(s.authority_score for s, _ in scored_documents) / len(scored_documents) if scored_documents else 0
            }
        }
        
        logger.info(f"Temporal filtering: {len(results)} -> {len(filtered_results)} results "
                   f"(removed {len(results) - len(filtered_results)} outdated)")
        
        return filtered_results, metadata
    
    def _detect_intent(self, query: str) -> Optional[str]:
        """Detect user intent from query"""
        query_lower = query.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        
        return None
    
    def _extract_date_from_document(self, document: Dict[str, Any]) -> Optional[datetime]:
        """Extract date from document metadata or content"""
        # Check for explicit date field
        if "date" in document:
            return self._parse_date_string(document["date"])
        
        if "published_date" in document:
            return self._parse_date_string(document["published_date"])
        
        if "days_old" in document and document["days_old"] is not None:
            return self.current_date - timedelta(days=document["days_old"])
        
        # Try to extract from snippet
        snippet = document.get("snippet", "")
        date_patterns = [
            r"(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})",
            r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})",
            r"(\d{4})-(\d{1,2})-(\d{1,2})",
            r"(\d{1,2})/(\d{1,2})/(\d{4})"
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, snippet)
            if match:
                try:
                    date_str = match.group(0)
                    return self._parse_date_string(date_str)
                except:
                    continue
        
        return None
    
    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """Parse various date string formats"""
        if not date_str:
            return None
        
        # Common date formats
        formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d/%m/%Y",
            "%B %d, %Y",
            "%b %d, %Y",
            "%d %B %Y",
            "%d %b %Y",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(str(date_str), fmt)
            except:
                continue
        
        # Try parsing month names
        months = {
            "jan": 1, "january": 1, "feb": 2, "february": 2,
            "mar": 3, "march": 3, "apr": 4, "april": 4,
            "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
            "aug": 8, "august": 8, "sep": 9, "september": 9,
            "oct": 10, "october": 10, "nov": 11, "november": 11,
            "dec": 12, "december": 12
        }
        
        # Try to extract month day year
        pattern = r"(\w+)\s+(\d{1,2}),?\s+(\d{4})"
        match = re.search(pattern, str(date_str), re.IGNORECASE)
        if match:
            month_str, day, year = match.groups()
            month = months.get(month_str.lower())
            if month:
                try:
                    return datetime(int(year), month, int(day))
                except:
                    pass
        
        return None
    
    def _should_include_document(
        self,
        temporal_score: float,
        authority_score: float,
        combined_score: float,
        age_days: float,
        classification: TemporalClassification
    ) -> bool:
        """Determine if a document should be included in results"""
        # Never include if too old for the query type
        if age_days > classification.max_age_days:
            return False
        
        # Historical queries include everything
        if classification.intent == "historical":
            return True
        
        # For current information queries, be strict
        if classification.intent == "current":
            if classification.sensitivity in [TemporalSensitivity.VERY_HIGH, TemporalSensitivity.HIGH]:
                # Very strict for volatile information
                return temporal_score >= 0.3 and combined_score >= 0.25
            else:
                # Moderate strictness
                return temporal_score >= 0.2 and combined_score >= 0.2
        
        # For high authority sources, be more lenient
        if authority_score >= 0.8:
            return combined_score >= self.min_combined_score * 0.5
        
        # Standard filtering
        return (
            temporal_score >= self.min_temporal_score and
            authority_score >= self.min_authority_score and
            combined_score >= self.min_combined_score
        )
    
    def _calculate_rank_boost(
        self,
        temporal_score: float,
        authority_score: float,
        classification: TemporalClassification,
        is_primary_source: bool
    ) -> float:
        """Calculate ranking boost factor"""
        boost = 1.0
        
        # Boost for high temporal relevance
        if temporal_score >= 0.8:
            boost *= 1.3
        elif temporal_score >= 0.6:
            boost *= 1.15
        
        # Boost for high authority
        if authority_score >= 0.9:
            boost *= 1.25
        elif authority_score >= 0.7:
            boost *= 1.1
        
        # Boost for primary sources
        if is_primary_source:
            boost *= 1.2
        
        # Boost based on query sensitivity
        if classification.sensitivity == TemporalSensitivity.VERY_HIGH:
            # For very time-sensitive queries, heavily boost recent content
            if temporal_score >= 0.7:
                boost *= 1.5
        
        return boost


# Singleton instance
_relevance_engine = None


def get_relevance_engine() -> TemporalRelevanceEngine:
    """Get singleton instance of temporal relevance engine"""
    global _relevance_engine
    if _relevance_engine is None:
        _relevance_engine = TemporalRelevanceEngine()
    return _relevance_engine