"""
Data models and types for the RAG agent system
"""

from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class ExecutionStrategy(Enum):
    """Types of execution strategies"""
    SINGLE_COLLECTION = "single_collection"
    CROSS_REFERENCE = "cross_reference" 
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    ITERATIVE_REFINEMENT = "iterative_refinement"
    PARALLEL_SEARCH = "parallel_search"


class StepType(Enum):
    """Types of execution steps"""
    PRIMARY_SEARCH = "primary_search"
    VALIDATION_SEARCH = "validation_search"
    FOCUSED_SEARCH = "focused_search"
    GAP_ANALYSIS = "gap_analysis"
    SYNTHESIS = "synthesis"
    INITIAL_BROAD_SEARCH = "initial_broad_search"


class FusionMethod(Enum):
    """Result fusion methods"""
    RELEVANCE_WEIGHTED = "relevance_weighted"
    COLLECTION_AUTHORITY = "collection_authority" 
    TEMPORAL_PRIORITY = "temporal_priority"
    CROSS_VALIDATION = "cross_validation"


@dataclass
class Source:
    """Source information for retrieved content"""
    collection_name: str
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    page: Optional[int] = None
    section: Optional[str] = None


@dataclass
class CollectionResult:
    """Results from searching a specific collection"""
    collection_name: str
    sources: List[Source]
    relevance_score: float
    search_strategy: str
    query_used: str
    execution_time_ms: int
    
    
@dataclass
class StepResult:
    """Results from an execution step"""
    step_id: str
    step_type: StepType
    collection_results: List[CollectionResult]
    success_score: float
    refinement_history: List[Dict] = field(default_factory=list)
    final_query: str = ""
    execution_time_ms: int = 0


@dataclass
class ExecutionStep:
    """Definition of an execution step"""
    step_id: str
    step_type: StepType
    collections: Union[List[str], str]  # List of collections or "auto_select"
    query_refinement: str
    success_criteria: str
    depends_on: List[int] = field(default_factory=list)
    max_iterations: int = 3
    parallel_execution: bool = False


@dataclass
class ExecutionPlan:
    """Complete execution plan for a query"""
    plan_id: str
    steps: List[ExecutionStep]
    strategy_type: ExecutionStrategy
    max_iterations: int
    estimated_time_ms: int = 0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class RoutingDecision:
    """LLM routing decision"""
    selected_collections: List[str]
    query_refinements: Dict[str, str]  # collection -> refined query
    execution_strategy: ExecutionStrategy
    confidence_score: float
    reasoning: str
    tool_calls: List[Dict] = field(default_factory=list)


@dataclass
class KnowledgeGap:
    """Identified knowledge gap in results"""
    type: str  # "missing_fact", "insufficient_detail", "limited_perspective"
    topic: str
    severity: str  # "high", "medium", "low"
    suggestion: str


@dataclass
class QualityAnalysis:
    """Quality analysis of search results"""
    meets_criteria: bool
    success_score: float
    gaps: List[KnowledgeGap]
    confidence: float
    recommendations: List[str]


@dataclass
class RAGOptions:
    """Configuration options for RAG processing"""
    max_iterations: int = 3
    stream: bool = False
    include_sources: bool = True
    include_execution_trace: bool = False
    confidence_threshold: float = 0.6
    max_results_per_collection: int = 10
    execution_timeout_ms: int = 30000


@dataclass
class ExecutionTrace:
    """Trace of execution steps for debugging"""
    plan_id: str
    steps_executed: List[StepResult]
    total_time_ms: int
    collections_searched: List[str]
    query_refinements: List[str]
    final_strategy: ExecutionStrategy


@dataclass
class RAGResponse:
    """Final response from RAG system"""
    content: str
    sources: List[Source]
    confidence_score: float
    execution_trace: Optional[ExecutionTrace] = None
    collections_searched: List[str] = field(default_factory=list)
    query_refinements: List[str] = field(default_factory=list)
    processing_time_ms: int = 0
    fusion_method: Optional[FusionMethod] = None


@dataclass
class RAGStreamChunk:
    """Streaming chunk for real-time responses"""
    content: str
    sources: List[Source] = field(default_factory=list)
    step_info: Optional[Dict] = None
    is_final: bool = False
    chunk_type: str = "content"  # "content", "sources", "metadata"


@dataclass
class SearchContext:
    """Context for search operations"""
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    domain: str = "general"
    urgency_level: str = "normal"
    required_accuracy: str = "high"
    conversation_history: List[Dict] = field(default_factory=list)
    user_permissions: List[str] = field(default_factory=list)


@dataclass
class IndexStrategy:
    """Vector index configuration"""
    index_type: str  # "IVF_FLAT", "IVF_SQ8", "HNSW"
    params: Dict[str, Any]
    collection_size_threshold: int = 0
    performance_target: str = "balanced"  # "speed", "accuracy", "balanced"