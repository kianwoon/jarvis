"""
Temporal Decay Functions

Implements various decay functions to calculate how information relevance
decreases over time for different domains and content types.
"""

import math
import logging
from typing import Callable, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DecayConfig:
    """Configuration for a decay function"""
    function_type: str  # exponential, linear, stepped, gaussian, logarithmic
    half_life_days: float  # Days until 50% relevance
    cutoff_days: float  # Days until 0% relevance
    minimum_score: float = 0.0  # Minimum score (floor)
    steepness: float = 1.0  # Controls decay curve steepness


class TemporalDecayFunctions:
    """Collection of temporal decay functions for different use cases"""
    
    @staticmethod
    def exponential_decay(age_days: float, config: DecayConfig) -> float:
        """
        Exponential decay function - fast initial decay, slowing over time
        Good for: news, social media, volatile information
        
        Args:
            age_days: Age of the document in days
            config: Decay configuration
            
        Returns:
            Relevance score between 0 and 1
        """
        if age_days <= 0:
            return 1.0
        if age_days >= config.cutoff_days:
            return config.minimum_score
        
        # Calculate decay constant from half-life
        decay_constant = math.log(2) / config.half_life_days
        
        # Apply exponential decay with steepness modifier
        score = math.exp(-decay_constant * config.steepness * age_days)
        
        return max(score, config.minimum_score)
    
    @staticmethod
    def linear_decay(age_days: float, config: DecayConfig) -> float:
        """
        Linear decay function - constant rate of decay
        Good for: documentation, guides, stable information
        
        Args:
            age_days: Age of the document in days
            config: Decay configuration
            
        Returns:
            Relevance score between 0 and 1
        """
        if age_days <= 0:
            return 1.0
        if age_days >= config.cutoff_days:
            return config.minimum_score
        
        # Linear interpolation from 1 to minimum_score
        score = 1.0 - (age_days / config.cutoff_days) * (1.0 - config.minimum_score)
        
        return max(score, config.minimum_score)
    
    @staticmethod
    def stepped_decay(age_days: float, config: DecayConfig) -> float:
        """
        Stepped decay function - discrete steps of relevance
        Good for: versioned content, regulatory documents, policies
        
        Args:
            age_days: Age of the document in days
            config: Decay configuration
            
        Returns:
            Relevance score between 0 and 1
        """
        if age_days <= 0:
            return 1.0
        
        # Define steps based on half-life
        steps = [
            (0, 1.0),
            (config.half_life_days * 0.5, 0.8),
            (config.half_life_days, 0.5),
            (config.half_life_days * 1.5, 0.3),
            (config.half_life_days * 2, 0.1),
            (config.cutoff_days, config.minimum_score)
        ]
        
        # Find appropriate step
        for i in range(len(steps) - 1):
            if steps[i][0] <= age_days < steps[i + 1][0]:
                return steps[i][1]
        
        return config.minimum_score
    
    @staticmethod
    def gaussian_decay(age_days: float, config: DecayConfig) -> float:
        """
        Gaussian (bell curve) decay - peaks at a certain age then decays
        Good for: event-based content, seasonal information
        
        Args:
            age_days: Age of the document in days
            config: Decay configuration
            
        Returns:
            Relevance score between 0 and 1
        """
        if age_days < 0:
            return 0.0
        if age_days >= config.cutoff_days:
            return config.minimum_score
        
        # For Gaussian, we'll treat half_life as the peak point
        # and cutoff as 3 standard deviations
        mean = 0  # Peak relevance at creation
        std_dev = config.cutoff_days / 3
        
        # Calculate Gaussian score
        score = math.exp(-0.5 * ((age_days - mean) / std_dev) ** 2)
        
        return max(score * config.steepness, config.minimum_score)
    
    @staticmethod
    def logarithmic_decay(age_days: float, config: DecayConfig) -> float:
        """
        Logarithmic decay - slow initial decay, faster later
        Good for: academic papers, research, foundational knowledge
        
        Args:
            age_days: Age of the document in days
            config: Decay configuration
            
        Returns:
            Relevance score between 0 and 1
        """
        if age_days <= 0:
            return 1.0
        if age_days >= config.cutoff_days:
            return config.minimum_score
        
        # Logarithmic decay with adjustment for half-life
        # Score at half_life should be 0.5
        x = age_days / config.half_life_days
        
        if x < 1:
            # Before half-life: slow decay
            score = 1.0 - 0.5 * math.log(1 + x) / math.log(2)
        else:
            # After half-life: faster decay
            score = 0.5 * (1.0 - math.log(x) / math.log(config.cutoff_days / config.half_life_days))
        
        return max(score, config.minimum_score)
    
    @staticmethod
    def hybrid_decay(age_days: float, config: DecayConfig) -> float:
        """
        Hybrid decay - combines exponential and linear for balanced decay
        Good for: general purpose, mixed content types
        
        Args:
            age_days: Age of the document in days
            config: Decay configuration
            
        Returns:
            Relevance score between 0 and 1
        """
        # Combine exponential and linear with weights
        exp_score = TemporalDecayFunctions.exponential_decay(age_days, config)
        lin_score = TemporalDecayFunctions.linear_decay(age_days, config)
        
        # Weight exponential more for early decay, linear for later
        if age_days < config.half_life_days:
            weight = 0.7  # More exponential
        else:
            weight = 0.3  # More linear
        
        score = weight * exp_score + (1 - weight) * lin_score
        
        return max(score, config.minimum_score)


class DomainDecayProfiles:
    """Predefined decay profiles for different domains"""
    
    PROFILES = {
        "news": DecayConfig(
            function_type="exponential",
            half_life_days=3,
            cutoff_days=30,
            minimum_score=0.0,
            steepness=1.5
        ),
        "tech_product": DecayConfig(
            function_type="exponential",
            half_life_days=60,
            cutoff_days=180,
            minimum_score=0.0,
            steepness=1.2
        ),
        "tech_subscription": DecayConfig(
            function_type="exponential",
            half_life_days=90,
            cutoff_days=180,
            minimum_score=0.0,
            steepness=1.0
        ),
        "documentation": DecayConfig(
            function_type="linear",
            half_life_days=180,
            cutoff_days=365,
            minimum_score=0.1,
            steepness=1.0
        ),
        "academic": DecayConfig(
            function_type="logarithmic",
            half_life_days=730,
            cutoff_days=3650,
            minimum_score=0.2,
            steepness=1.0
        ),
        "legal": DecayConfig(
            function_type="stepped",
            half_life_days=365,
            cutoff_days=1825,
            minimum_score=0.1,
            steepness=1.0
        ),
        "historical": DecayConfig(
            function_type="linear",
            half_life_days=36500,
            cutoff_days=365000,
            minimum_score=0.5,
            steepness=0.1
        ),
        "general": DecayConfig(
            function_type="hybrid",
            half_life_days=120,
            cutoff_days=365,
            minimum_score=0.05,
            steepness=1.0
        )
    }
    
    @classmethod
    def get_profile(cls, domain: str) -> DecayConfig:
        """Get decay profile for a domain"""
        return cls.PROFILES.get(domain, cls.PROFILES["general"])
    
    @classmethod
    def get_decay_function(cls, domain: str) -> Callable[[float, DecayConfig], float]:
        """Get the appropriate decay function for a domain"""
        profile = cls.get_profile(domain)
        
        function_map = {
            "exponential": TemporalDecayFunctions.exponential_decay,
            "linear": TemporalDecayFunctions.linear_decay,
            "stepped": TemporalDecayFunctions.stepped_decay,
            "gaussian": TemporalDecayFunctions.gaussian_decay,
            "logarithmic": TemporalDecayFunctions.logarithmic_decay,
            "hybrid": TemporalDecayFunctions.hybrid_decay
        }
        
        return function_map.get(profile.function_type, TemporalDecayFunctions.hybrid_decay)


def calculate_temporal_score(
    age_days: float,
    domain: str = "general",
    custom_config: Optional[DecayConfig] = None
) -> float:
    """
    Calculate temporal relevance score for a document
    
    Args:
        age_days: Age of the document in days
        domain: Domain of the content
        custom_config: Optional custom decay configuration
        
    Returns:
        Temporal relevance score between 0 and 1
    """
    # Use custom config if provided, otherwise get domain profile
    config = custom_config or DomainDecayProfiles.get_profile(domain)
    
    # Get appropriate decay function
    decay_function = DomainDecayProfiles.get_decay_function(domain)
    
    # Calculate and return score
    score = decay_function(age_days, config)
    
    logger.debug(f"Temporal score for {domain} at {age_days} days: {score:.3f}")
    
    return score