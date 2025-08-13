"""
Pydantic schemas for Chat Overflow Handler
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class OverflowConfig(BaseModel):
    """Configuration for overflow handling stored in database"""
    overflow_threshold_tokens: int = Field(default=8000, description="Token threshold to trigger overflow handling")
    chunk_size_tokens: int = Field(default=2000, description="Size of each chunk in tokens")
    chunk_overlap_tokens: int = Field(default=200, description="Overlap between chunks in tokens")
    l1_ttl_hours: int = Field(default=24, description="TTL for L1 hot storage in hours")
    l2_ttl_days: int = Field(default=7, description="TTL for L2 warm storage in days")
    max_overflow_context_ratio: float = Field(default=0.3, description="Maximum ratio of context for overflow content")
    retrieval_top_k: int = Field(default=5, description="Number of top chunks to retrieve")
    enable_semantic_search: bool = Field(default=True, description="Enable semantic search for chunk retrieval")
    enable_keyword_extraction: bool = Field(default=True, description="Enable keyword extraction from chunks")
    auto_promote_to_l1: bool = Field(default=True, description="Automatically promote frequently accessed chunks to L1")
    promotion_threshold_accesses: int = Field(default=3, description="Number of accesses before promoting to L1")

class OverflowChunk(BaseModel):
    """Schema for an overflow chunk"""
    chunk_id: str
    conversation_id: str
    position: int
    content: str
    embedding: Optional[List[float]] = None
    summary: str
    keywords: List[str]
    token_count: int
    created_at: datetime
    access_count: int = Field(default=0)
    last_accessed: Optional[datetime] = None
    storage_layer: str = Field(default="L2", pattern="^(L1|L2|L3)$")

class OverflowMetadata(BaseModel):
    """Metadata about overflow content for a conversation"""
    conversation_id: str
    total_chunks: int
    total_tokens: int
    storage_layers: Dict[str, int] = Field(default_factory=dict, description="Chunk count per storage layer")
    created_at: datetime
    expires_at: datetime
    last_accessed: Optional[datetime] = None
    access_count: int = Field(default=0)
    
class OverflowRequest(BaseModel):
    """Request to handle overflow content"""
    conversation_id: str
    content: str
    auto_chunk: bool = Field(default=True, description="Automatically chunk the content")
    storage_layer: str = Field(default="L2", pattern="^(L1|L2)$")
    custom_config: Optional[OverflowConfig] = None

class OverflowRetrievalRequest(BaseModel):
    """Request to retrieve relevant overflow chunks"""
    conversation_id: str
    query: str
    top_k: Optional[int] = None
    include_metadata: bool = Field(default=False)
    storage_layers: List[str] = Field(default=["L1", "L2"])

class OverflowResponse(BaseModel):
    """Response from overflow handling"""
    success: bool
    conversation_id: str
    chunks_created: int
    total_tokens: int
    storage_layer: str
    message: str
    metadata: Optional[OverflowMetadata] = None

class OverflowRetrievalResponse(BaseModel):
    """Response from overflow retrieval"""
    conversation_id: str
    chunks: List[OverflowChunk]
    total_chunks_available: int
    query_similarity_scores: Optional[Dict[str, float]] = None
    metadata: Optional[OverflowMetadata] = None

class OverflowSummary(BaseModel):
    """Summary of overflow content for a conversation"""
    conversation_id: str
    has_overflow: bool
    total_chunks: int
    total_tokens: int
    storage_distribution: Dict[str, int]
    ttl_remaining_hours: Optional[float] = None
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    most_relevant_topics: List[str] = Field(default_factory=list)

class OverflowConfigUpdate(BaseModel):
    """Update request for overflow configuration"""
    overflow_threshold_tokens: Optional[int] = None
    chunk_size_tokens: Optional[int] = None
    chunk_overlap_tokens: Optional[int] = None
    l1_ttl_hours: Optional[int] = None
    l2_ttl_days: Optional[int] = None
    max_overflow_context_ratio: Optional[float] = None
    retrieval_top_k: Optional[int] = None
    enable_semantic_search: Optional[bool] = None
    enable_keyword_extraction: Optional[bool] = None
    auto_promote_to_l1: Optional[bool] = None
    promotion_threshold_accesses: Optional[int] = None