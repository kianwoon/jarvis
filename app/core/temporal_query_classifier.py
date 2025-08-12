"""
Temporal Query Classifier

Analyzes queries to determine their temporal sensitivity and how quickly
information becomes outdated for different types of queries.
"""

import re
import logging
from enum import Enum
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class TemporalSensitivity(Enum):
    """Temporal sensitivity levels for different query types"""
    VERY_HIGH = "real-time"      # Hours to days (news, stock prices, current events)
    HIGH = "volatile"             # Days to weeks (product updates, pricing, features)  
    MEDIUM = "seasonal"           # Weeks to months (documentation, guides, policies)
    LOW = "stable"               # Months to years (historical facts, scientific principles)
    HISTORICAL = "past-focused"   # User explicitly wants historical information


@dataclass
class TemporalClassification:
    """Result of temporal query classification"""
    sensitivity: TemporalSensitivity
    confidence: float
    domain: str
    intent: str  # current, historical, comparison, neutral
    decay_days: int  # Suggested number of days before information becomes stale
    max_age_days: int  # Maximum acceptable age in days
    keywords_matched: List[str]


class TemporalQueryClassifier:
    """Classifies queries based on their temporal sensitivity"""
    
    def __init__(self):
        """Initialize the temporal query classifier with patterns and rules"""
        
        # Domain patterns and their default sensitivity
        self.domain_patterns = {
            "technology": {
                "patterns": [
                    r'\b(software|app|api|sdk|version|update|release|bug|feature)\b',
                    r'\b(chatgpt|gpt-\d|openai|claude|llm|ai model)\b',
                    r'\b(subscription|pricing|tier|plan|limit|usage|cost)\b'
                ],
                "sensitivity": TemporalSensitivity.HIGH,
                "decay_days": 90,
                "max_age_days": 180
            },
            "finance": {
                "patterns": [
                    r'\b(stock|price|market|trading|investment|crypto|bitcoin)\b',
                    r'\b(earnings|revenue|profit|financial|quarterly|annual)\b',
                    r'\b(rate|interest|inflation|economy|gdp)\b'
                ],
                "sensitivity": TemporalSensitivity.VERY_HIGH,
                "decay_days": 7,
                "max_age_days": 30
            },
            "news": {
                "patterns": [
                    r'\b(news|latest|breaking|update|announcement|report)\b',
                    r'\b(today|yesterday|this week|recent)\b',
                    r'\b(event|happening|occurred|incident)\b'
                ],
                "sensitivity": TemporalSensitivity.VERY_HIGH,
                "decay_days": 3,
                "max_age_days": 14
            },
            "science": {
                "patterns": [
                    r'\b(research|study|paper|journal|scientific|experiment)\b',
                    r'\b(theory|hypothesis|discovery|finding|conclusion)\b',
                    r'\b(physics|chemistry|biology|medicine|psychology)\b'
                ],
                "sensitivity": TemporalSensitivity.LOW,
                "decay_days": 730,
                "max_age_days": 3650
            },
            "history": {
                "patterns": [
                    r'\b(history|historical|past|ancient|century|decade)\b',
                    r'\b(war|revolution|empire|civilization|era)\b',
                    r'\b(was|were|had been|used to)\b.*\b(originally|previously|formerly)\b'
                ],
                "sensitivity": TemporalSensitivity.HISTORICAL,
                "decay_days": 36500,  # 100 years
                "max_age_days": 365000  # Effectively no limit
            },
            "general": {
                "patterns": [r'.*'],  # Catch-all
                "sensitivity": TemporalSensitivity.MEDIUM,
                "decay_days": 180,
                "max_age_days": 365
            }
        }
        
        # Intent patterns
        self.intent_patterns = {
            "current": {
                "patterns": [
                    r'\b(current|latest|now|today|recent|new|updated|modern)\b',
                    r'\b(what is|what are|how much|how many)\b',
                    r'\b(2025|this year|this month|this week)\b'
                ],
                "boost_sensitivity": True
            },
            "historical": {
                "patterns": [
                    r'\b(was|were|had|did|used to|originally|previously)\b',
                    r'\b(history|historical|past|ago|back in|during)\b',
                    r'\b(19\d{2}|20[012]\d)\b',  # Years before 2025
                    r'\bin\s+(19\d{2}|20[012]\d)\b'  # "in 2023" etc
                ],
                "boost_sensitivity": False,
                "override_sensitivity": TemporalSensitivity.HISTORICAL
            },
            "comparison": {
                "patterns": [
                    r'\b(vs|versus|compared to|difference between|better than)\b',
                    r'\b(evolution|timeline|changes|progression)\b'
                ],
                "boost_sensitivity": False
            }
        }
        
        # Volatility modifiers - terms that increase temporal sensitivity
        self.volatility_modifiers = {
            "high_volatility": [
                "price", "cost", "rate", "limit", "usage", "quota",
                "subscription", "plan", "tier", "feature", "update",
                "release", "version", "beta", "preview"
            ],
            "low_volatility": [
                "concept", "theory", "principle", "law", "definition",
                "meaning", "explanation", "tutorial", "guide", "basic"
            ]
        }
    
    def classify(self, query: str) -> TemporalClassification:
        """
        Classify a query's temporal sensitivity
        
        Args:
            query: The search query to classify
            
        Returns:
            TemporalClassification with sensitivity level and metadata
        """
        query_lower = query.lower()
        
        # Detect domain
        domain, domain_config = self._detect_domain(query_lower)
        
        # Detect intent
        intent = self._detect_intent(query_lower)
        
        # Get base sensitivity from domain
        sensitivity = domain_config["sensitivity"]
        decay_days = domain_config["decay_days"]
        max_age_days = domain_config["max_age_days"]
        
        # Adjust based on intent
        if intent == "historical":
            sensitivity = TemporalSensitivity.HISTORICAL
            decay_days = 36500
            max_age_days = 365000
        elif intent == "current":
            # Boost sensitivity for current information requests
            if sensitivity == TemporalSensitivity.MEDIUM:
                sensitivity = TemporalSensitivity.HIGH
                decay_days = min(decay_days, 90)
                max_age_days = min(max_age_days, 180)
            elif sensitivity == TemporalSensitivity.LOW:
                sensitivity = TemporalSensitivity.MEDIUM
                decay_days = min(decay_days, 180)
                max_age_days = min(max_age_days, 365)
        
        # Check for volatility modifiers
        volatility_adjustment = self._check_volatility_modifiers(query_lower)
        if volatility_adjustment > 0 and sensitivity not in [TemporalSensitivity.HISTORICAL, TemporalSensitivity.VERY_HIGH]:
            # Increase sensitivity
            if sensitivity == TemporalSensitivity.LOW:
                sensitivity = TemporalSensitivity.MEDIUM
            elif sensitivity == TemporalSensitivity.MEDIUM:
                sensitivity = TemporalSensitivity.HIGH
            decay_days = int(decay_days * 0.5)  # Halve the decay period
            max_age_days = int(max_age_days * 0.5)
        elif volatility_adjustment < 0 and sensitivity not in [TemporalSensitivity.HISTORICAL]:
            # Decrease sensitivity
            if sensitivity == TemporalSensitivity.HIGH:
                sensitivity = TemporalSensitivity.MEDIUM
            elif sensitivity == TemporalSensitivity.VERY_HIGH:
                sensitivity = TemporalSensitivity.HIGH
            decay_days = int(decay_days * 1.5)
            max_age_days = int(max_age_days * 1.5)
        
        # Calculate confidence
        confidence = self._calculate_confidence(query_lower, domain, intent)
        
        # Get matched keywords for transparency
        keywords_matched = self._get_matched_keywords(query_lower)
        
        return TemporalClassification(
            sensitivity=sensitivity,
            confidence=confidence,
            domain=domain,
            intent=intent,
            decay_days=decay_days,
            max_age_days=max_age_days,
            keywords_matched=keywords_matched
        )
    
    def _detect_domain(self, query: str) -> Tuple[str, Dict]:
        """Detect the domain of the query"""
        best_match = "general"
        best_score = 0
        
        for domain, config in self.domain_patterns.items():
            if domain == "general":
                continue
            
            score = sum(1 for pattern in config["patterns"] 
                       if re.search(pattern, query, re.IGNORECASE))
            
            if score > best_score:
                best_score = score
                best_match = domain
        
        return best_match, self.domain_patterns[best_match]
    
    def _detect_intent(self, query: str) -> str:
        """Detect the temporal intent of the query"""
        for intent, config in self.intent_patterns.items():
            for pattern in config["patterns"]:
                if re.search(pattern, query, re.IGNORECASE):
                    return intent
        return "neutral"
    
    def _check_volatility_modifiers(self, query: str) -> int:
        """
        Check for volatility modifiers in the query
        Returns: positive for high volatility, negative for low volatility, 0 for neutral
        """
        high_count = sum(1 for term in self.volatility_modifiers["high_volatility"] 
                         if term in query)
        low_count = sum(1 for term in self.volatility_modifiers["low_volatility"] 
                        if term in query)
        
        return high_count - low_count
    
    def _calculate_confidence(self, query: str, domain: str, intent: str) -> float:
        """Calculate confidence in the classification"""
        confidence = 0.5  # Base confidence
        
        # Domain match increases confidence
        if domain != "general":
            confidence += 0.2
        
        # Clear intent increases confidence
        if intent in ["current", "historical"]:
            confidence += 0.2
        
        # Volatility modifiers increase confidence
        volatility = abs(self._check_volatility_modifiers(query))
        if volatility > 0:
            confidence += min(0.1 * volatility, 0.3)
        
        return min(confidence, 1.0)
    
    def _get_matched_keywords(self, query: str) -> List[str]:
        """Get list of keywords that influenced the classification"""
        matched = []
        
        # Check domain keywords
        for domain, config in self.domain_patterns.items():
            for pattern in config["patterns"]:
                matches = re.findall(pattern, query, re.IGNORECASE)
                matched.extend(matches)
        
        # Check intent keywords
        for intent, config in self.intent_patterns.items():
            for pattern in config["patterns"]:
                matches = re.findall(pattern, query, re.IGNORECASE)
                matched.extend(matches)
        
        # Remove duplicates and return
        return list(set(matched))[:10]  # Limit to 10 keywords


# Singleton instance
_temporal_classifier = None


def get_temporal_classifier() -> TemporalQueryClassifier:
    """Get singleton instance of temporal query classifier"""
    global _temporal_classifier
    if _temporal_classifier is None:
        _temporal_classifier = TemporalQueryClassifier()
    return _temporal_classifier