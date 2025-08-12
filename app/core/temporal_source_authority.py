"""
Temporal Source Authority Scorer

Evaluates the credibility and authority of information sources to balance
recency with reliability.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass 
class SourceAuthority:
    """Source authority assessment"""
    domain: str
    url: str
    authority_score: float  # 0-1 score for source credibility
    source_type: str  # official, academic, news, forum, blog, unknown
    is_primary_source: bool  # Is this an original/primary source
    confidence: float  # Confidence in the assessment


class TemporalSourceAuthorityScorer:
    """Scores sources based on their credibility and authority"""
    
    def __init__(self):
        """Initialize the source authority scorer with patterns and rules"""
        
        # Domain authority scores
        self.domain_authority = {
            # Official sources - highest authority
            "openai.com": 1.0,
            "help.openai.com": 1.0,
            "platform.openai.com": 1.0,
            "anthropic.com": 1.0,
            "microsoft.com": 0.95,
            "google.com": 0.95,
            "aws.amazon.com": 0.95,
            "azure.microsoft.com": 0.95,
            
            # Academic and research
            "arxiv.org": 0.9,
            "scholar.google.com": 0.9,
            "nature.com": 0.9,
            "science.org": 0.9,
            "ieee.org": 0.85,
            "acm.org": 0.85,
            
            # Tech documentation
            "github.com": 0.8,
            "stackoverflow.com": 0.75,
            "docs.python.org": 0.9,
            "developer.mozilla.org": 0.85,
            
            # News and media
            "techcrunch.com": 0.7,
            "theverge.com": 0.7,
            "wired.com": 0.7,
            "arstechnica.com": 0.7,
            "reuters.com": 0.75,
            "bloomberg.com": 0.75,
            
            # Forums and communities
            "reddit.com": 0.4,
            "news.ycombinator.com": 0.5,
            "community.openai.com": 0.6,  # Official community gets higher score
            "discord.com": 0.3,
            "twitter.com": 0.3,
            "x.com": 0.3,
            
            # Blogs and personal sites
            "medium.com": 0.4,
            "substack.com": 0.4,
            "wordpress.com": 0.3,
            "blogspot.com": 0.3
        }
        
        # Source type patterns
        self.source_patterns = {
            "official": {
                "patterns": [
                    r"official", r"documentation", r"docs\.", r"help\.",
                    r"support\.", r"developer\.", r"api\.", r"platform\."
                ],
                "url_keywords": ["official", "docs", "documentation", "help", "support"],
                "base_score": 0.9
            },
            "academic": {
                "patterns": [
                    r"\.edu", r"journal", r"research", r"paper", r"study",
                    r"arxiv", r"pubmed", r"scholar", r"academic"
                ],
                "url_keywords": ["edu", "research", "journal", "paper"],
                "base_score": 0.85
            },
            "news": {
                "patterns": [
                    r"news", r"article", r"report", r"press", r"media",
                    r"techcrunch", r"verge", r"wired", r"reuters"
                ],
                "url_keywords": ["news", "article", "report"],
                "base_score": 0.65
            },
            "forum": {
                "patterns": [
                    r"reddit", r"forum", r"community", r"discussion",
                    r"stackoverflow", r"discord", r"slack"
                ],
                "url_keywords": ["reddit", "forum", "community", "discuss"],
                "base_score": 0.4
            },
            "blog": {
                "patterns": [
                    r"blog", r"medium", r"substack", r"wordpress",
                    r"personal", r"opinion", r"thoughts"
                ],
                "url_keywords": ["blog", "medium", "substack"],
                "base_score": 0.35
            }
        }
        
        # Keywords that indicate primary sources
        self.primary_source_indicators = [
            "announcement", "release", "launches", "introduces",
            "official", "we are", "our", "today we"
        ]
        
        # Keywords that indicate secondary sources
        self.secondary_source_indicators = [
            "report", "according to", "sources say", "reportedly",
            "analysis", "review", "opinion", "thoughts on"
        ]
    
    def score_source(self, url: str, title: str = "", snippet: str = "") -> SourceAuthority:
        """
        Score a source's authority and credibility
        
        Args:
            url: The URL of the source
            title: Optional title of the content
            snippet: Optional snippet/excerpt of the content
            
        Returns:
            SourceAuthority with scoring details
        """
        # Parse URL
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www prefix
            domain = domain.replace("www.", "")
        except:
            domain = ""
            logger.warning(f"Could not parse URL: {url}")
        
        # Get base domain authority
        authority_score = self._get_domain_authority(domain)
        
        # Detect source type
        source_type = self._detect_source_type(url, domain, title, snippet)
        
        # Adjust score based on source type
        type_config = self.source_patterns.get(source_type, {"base_score": 0.5})
        if authority_score == 0.5:  # Default/unknown domain
            authority_score = type_config["base_score"]
        
        # Check if primary source
        is_primary = self._is_primary_source(url, title, snippet)
        if is_primary:
            authority_score = min(1.0, authority_score * 1.1)  # 10% boost for primary sources
        
        # Adjust for specific patterns
        authority_score = self._apply_pattern_adjustments(authority_score, url, title, snippet)
        
        # Calculate confidence
        confidence = self._calculate_confidence(domain, source_type, is_primary)
        
        return SourceAuthority(
            domain=domain,
            url=url,
            authority_score=min(1.0, max(0.0, authority_score)),
            source_type=source_type,
            is_primary_source=is_primary,
            confidence=confidence
        )
    
    def _get_domain_authority(self, domain: str) -> float:
        """Get base authority score for a domain"""
        # Check exact match
        if domain in self.domain_authority:
            return self.domain_authority[domain]
        
        # Check partial matches (e.g., subdomain.openai.com)
        for auth_domain, score in self.domain_authority.items():
            if auth_domain in domain or domain.endswith(f".{auth_domain}"):
                return score * 0.95  # Slight reduction for subdomains
        
        # Default score
        return 0.5
    
    def _detect_source_type(self, url: str, domain: str, title: str, snippet: str) -> str:
        """Detect the type of source"""
        combined_text = f"{url} {domain} {title} {snippet}".lower()
        
        best_match = "unknown"
        best_score = 0
        
        for source_type, config in self.source_patterns.items():
            score = 0
            
            # Check patterns
            for pattern in config["patterns"]:
                if re.search(pattern, combined_text):
                    score += 1
            
            # Check URL keywords
            for keyword in config["url_keywords"]:
                if keyword in url.lower():
                    score += 2  # URL keywords are stronger indicators
            
            if score > best_score:
                best_score = score
                best_match = source_type
        
        return best_match
    
    def _is_primary_source(self, url: str, title: str, snippet: str) -> bool:
        """Check if this appears to be a primary source"""
        combined_text = f"{title} {snippet}".lower()
        
        # Check for primary source indicators
        primary_count = sum(1 for indicator in self.primary_source_indicators 
                           if indicator in combined_text)
        
        # Check for secondary source indicators
        secondary_count = sum(1 for indicator in self.secondary_source_indicators 
                             if indicator in combined_text)
        
        # Primary source if more primary than secondary indicators
        return primary_count > secondary_count
    
    def _apply_pattern_adjustments(self, base_score: float, url: str, title: str, snippet: str) -> float:
        """Apply specific pattern-based adjustments"""
        score = base_score
        combined_text = f"{url} {title} {snippet}".lower()
        
        # Boost for specific quality indicators
        quality_indicators = [
            ("official documentation", 0.1),
            ("api reference", 0.1),
            ("technical specification", 0.05),
            ("peer reviewed", 0.15),
            ("whitepaper", 0.1),
            ("research paper", 0.1)
        ]
        
        for indicator, boost in quality_indicators:
            if indicator in combined_text:
                score += boost
        
        # Penalties for low-quality indicators
        penalty_indicators = [
            ("opinion", -0.1),
            ("rumor", -0.2),
            ("unconfirmed", -0.15),
            ("speculation", -0.15),
            ("my thoughts", -0.1),
            ("i think", -0.05)
        ]
        
        for indicator, penalty in penalty_indicators:
            if indicator in combined_text:
                score += penalty
        
        return score
    
    def _calculate_confidence(self, domain: str, source_type: str, is_primary: bool) -> float:
        """Calculate confidence in the authority assessment"""
        confidence = 0.5  # Base confidence
        
        # Known domain increases confidence
        if domain in self.domain_authority:
            confidence += 0.3
        
        # Clear source type increases confidence
        if source_type != "unknown":
            confidence += 0.1
        
        # Primary source determination increases confidence
        if is_primary:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def get_combined_score(
        self, 
        temporal_score: float, 
        authority_score: float,
        intent: str = "neutral"
    ) -> float:
        """
        Calculate combined temporal-authority score
        
        Args:
            temporal_score: Temporal relevance score (0-1)
            authority_score: Source authority score (0-1)
            intent: User intent (current, historical, neutral)
            
        Returns:
            Combined score between 0 and 1
        """
        if intent == "current":
            # For current information, weight temporal more heavily
            combined = temporal_score * 0.7 + authority_score * 0.3
        elif intent == "historical":
            # For historical information, authority matters more than recency
            combined = temporal_score * 0.2 + authority_score * 0.8
        else:
            # Balanced weighting for neutral intent
            combined = temporal_score * 0.5 + authority_score * 0.5
        
        return min(1.0, max(0.0, combined))


# Singleton instance
_source_scorer = None


def get_source_scorer() -> TemporalSourceAuthorityScorer:
    """Get singleton instance of source authority scorer"""
    global _source_scorer
    if _source_scorer is None:
        _source_scorer = TemporalSourceAuthorityScorer()
    return _source_scorer