"""
Standalone Agent-Based RAG Module

This module provides an LLM-orchestrated RAG system with intelligent collection routing,
recursive query refinement, and multi-strategy search execution.

Main Components:
- RAGOrchestrator: Main orchestration engine
- LLMRouter: LLM-based collection selection and routing
- CollectionToolRegistry: Dynamic collection tool management
- ExecutionPlanner: Multi-step execution planning
- QueryEngine: Recursive query refinement
- ResultFusion: Intelligent result combination

Usage:
    from app.rag_agent import StandaloneRAGInterface
    
    rag = StandaloneRAGInterface()
    response = await rag.query("What's our policy on data retention?")
"""

from .interfaces.rag_interface import StandaloneRAGInterface
from .core.rag_orchestrator import RAGOrchestrator
from .routers.llm_router import LLMRouter
from .routers.tool_registry import CollectionToolRegistry

__version__ = "1.0.0"
__all__ = [
    "StandaloneRAGInterface",
    "RAGOrchestrator", 
    "LLMRouter",
    "CollectionToolRegistry"
]