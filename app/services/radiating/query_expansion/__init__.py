"""
Query Expansion System for Radiating Coverage

This module provides intelligent query expansion capabilities to enhance
the radiating coverage system's ability to discover relevant information.
"""

from .query_analyzer import QueryAnalyzer
from .expansion_strategy import (
    ExpansionStrategy,
    SemanticExpansionStrategy,
    HierarchicalExpansionStrategy,
    AdaptiveExpansionStrategy
)
from .context_preserver import ContextPreserver
from .result_synthesizer import ResultSynthesizer

__all__ = [
    'QueryAnalyzer',
    'ExpansionStrategy',
    'SemanticExpansionStrategy',
    'HierarchicalExpansionStrategy',
    'AdaptiveExpansionStrategy',
    'ContextPreserver',
    'ResultSynthesizer'
]