"""
Pydantic models for Notebook API contracts.
Defines request/response models for notebook operations.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum
import uuid

class NotebookStatus(str, Enum):
    """Notebook status."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    ERROR = "error"

class DocumentType(str, Enum):
    """Document type in notebook."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"
    UNKNOWN = "unknown"

class NotebookCreateRequest(BaseModel):
    """Request model for notebook creation."""
    name: str = Field(..., min_length=1, max_length=255, description="Notebook name")
    description: Optional[str] = Field(None, description="Notebook description")
    user_id: Optional[str] = Field(None, description="User ID")
    source_filter: Optional[Dict[str, Any]] = Field(None, description="Source filtering configuration")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Notebook name cannot be empty")
        return v.strip()

class NotebookUpdateRequest(BaseModel):
    """Request model for notebook updates."""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Updated notebook name")
    description: Optional[str] = Field(None, description="Updated description")
    user_id: Optional[str] = Field(None, description="Updated user ID")
    source_filter: Optional[Dict[str, Any]] = Field(None, description="Updated source filtering")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")
    
    @validator('name')
    def validate_name(cls, v):
        if v is not None and (not v or not v.strip()):
            raise ValueError("Notebook name cannot be empty")
        return v.strip() if v else v

class NotebookDocumentAddRequest(BaseModel):
    """Request model for adding documents to notebook."""
    document_id: str = Field(..., description="Document ID to add")
    document_name: Optional[str] = Field(None, description="Document display name")
    document_type: Optional[DocumentType] = Field(None, description="Document type")
    milvus_collection: Optional[str] = Field(None, description="Milvus collection name")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")
    
    @validator('document_id')
    def validate_document_id(cls, v):
        if not v or not v.strip():
            raise ValueError("Document ID cannot be empty")
        return v.strip()

class NotebookDocumentBulkRequest(BaseModel):
    """Request model for bulk document operations."""
    action: str = Field(..., description="Action to perform")
    document_ids: List[str] = Field(..., min_items=1, description="Document IDs to act on")
    
    @validator('action')
    def validate_action(cls, v):
        allowed_actions = {'add', 'remove'}
        if v not in allowed_actions:
            raise ValueError(f"Invalid action. Allowed: {', '.join(allowed_actions)}")
        return v
    
    @validator('document_ids')
    def validate_document_ids(cls, v):
        if not v:
            raise ValueError("Document IDs cannot be empty")
        for doc_id in v:
            if not doc_id or not doc_id.strip():
                raise ValueError("Document ID cannot be empty")
        return [doc_id.strip() for doc_id in v]

class NotebookRAGRequest(BaseModel):
    """Request model for RAG queries against notebook."""
    query: str = Field(..., min_length=1, description="Query string")
    top_k: int = Field(5, ge=1, le=50, description="Number of results to return")
    include_metadata: bool = Field(True, description="Include source metadata")
    collection_filter: Optional[List[str]] = Field(None, description="Filter by specific collections")
    
    @validator('query')
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

class NotebookConversationRequest(BaseModel):
    """Request model for starting notebook conversation."""
    conversation_id: str = Field(..., description="Conversation ID")
    
    @validator('conversation_id')
    def validate_conversation_id(cls, v):
        if not v or not v.strip():
            raise ValueError("Conversation ID cannot be empty")
        return v.strip()

# Response Models

class NotebookDocumentResponse(BaseModel):
    """Response model for notebook documents."""
    id: str = Field(..., description="Document record ID")
    notebook_id: str = Field(..., description="Parent notebook ID")
    document_id: str = Field(..., description="Document ID")
    document_name: Optional[str] = Field(None, description="Document display name")
    document_type: Optional[DocumentType] = Field(None, description="Document type")
    milvus_collection: Optional[str] = Field(None, description="Milvus collection name")
    added_at: datetime = Field(..., description="When document was added")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")

class NotebookConversationResponse(BaseModel):
    """Response model for notebook conversations."""
    id: str = Field(..., description="Conversation record ID")
    notebook_id: str = Field(..., description="Parent notebook ID")
    conversation_id: str = Field(..., description="Conversation ID")
    started_at: datetime = Field(..., description="When conversation started")
    last_activity: datetime = Field(..., description="Last activity timestamp")

class NotebookResponse(BaseModel):
    """Response model for notebook operations."""
    id: str = Field(..., description="Unique notebook ID")
    name: str = Field(..., description="Notebook name")
    description: Optional[str] = Field(None, description="Notebook description")
    user_id: Optional[str] = Field(None, description="User ID")
    source_filter: Optional[Dict[str, Any]] = Field(None, description="Source filtering configuration")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    document_count: Optional[int] = Field(None, description="Number of documents in notebook")
    conversation_count: Optional[int] = Field(None, description="Number of conversations")

class NotebookDetailResponse(NotebookResponse):
    """Detailed notebook response with documents and conversations."""
    documents: List[NotebookDocumentResponse] = Field(..., description="Documents in notebook")
    conversations: List[NotebookConversationResponse] = Field(..., description="Active conversations")

class NotebookListResponse(BaseModel):
    """Response model for listing notebooks."""
    notebooks: List[NotebookResponse] = Field(..., description="List of notebooks")
    total_count: int = Field(..., description="Total number of notebooks")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")

class NotebookRAGSource(BaseModel):
    """Source information from notebook RAG query."""
    content: str = Field(..., description="Content of the source")
    metadata: Dict[str, Any] = Field(..., description="Source metadata")
    score: float = Field(..., description="Relevance score")
    document_id: str = Field(..., description="Source document ID")
    document_name: Optional[str] = Field(None, description="Source document name")
    collection: Optional[str] = Field(None, description="Milvus collection name")

class NotebookRAGResponse(BaseModel):
    """Response model for notebook RAG queries."""
    notebook_id: str = Field(..., description="Notebook ID")
    query: str = Field(..., description="Original query")
    sources: List[NotebookRAGSource] = Field(..., description="Query results")
    total_sources: int = Field(..., description="Total number of sources found")
    queried_documents: int = Field(..., description="Number of documents queried")
    collections_searched: List[str] = Field(..., description="Collections that were searched")

class NotebookDocumentBulkResponse(BaseModel):
    """Response model for bulk document operations."""
    success: bool = Field(..., description="Whether bulk operation was successful")
    action: str = Field(..., description="Action performed")
    processed_count: int = Field(..., description="Number of documents processed")
    failed_count: int = Field(..., description="Number of documents that failed")
    failed_ids: List[str] = Field(..., description="IDs of documents that failed")
    message: str = Field(..., description="Result message")

class NotebookStatsResponse(BaseModel):
    """Response model for notebook statistics."""
    notebook_id: str = Field(..., description="Notebook ID")
    total_documents: int = Field(..., description="Total number of documents")
    documents_by_type: Dict[str, int] = Field(..., description="Documents grouped by type")
    total_conversations: int = Field(..., description="Total number of conversations")
    active_conversations: int = Field(..., description="Number of active conversations")
    collections_used: List[str] = Field(..., description="Milvus collections in use")
    last_activity: Optional[datetime] = Field(None, description="Last activity timestamp")
    created_at: datetime = Field(..., description="Notebook creation time")

# Error response models
class NotebookError(BaseModel):
    """Standard error response for notebook operations."""
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    notebook_id: Optional[str] = Field(None, description="Related notebook ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")

class NotebookValidationError(BaseModel):
    """Validation error response."""
    error: str = Field(..., description="General error message")
    validation_errors: List[Dict[str, Any]] = Field(..., description="Detailed validation errors")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")

class NotebookOperationResponse(BaseModel):
    """Standard operation response."""
    success: bool = Field(..., description="Whether operation was successful")
    message: str = Field(..., description="Operation result message")
    data: Optional[Any] = Field(None, description="Optional result data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

# Document Admin Models

class DocumentDeleteRequest(BaseModel):
    """Request model for permanent document deletion."""
    document_ids: List[str] = Field(..., min_items=1, description="Document IDs to delete permanently")
    remove_from_notebooks: bool = Field(True, description="Whether to remove from all notebooks")
    confirm_permanent_deletion: bool = Field(..., description="Confirmation that user understands this is permanent")
    
    @validator('document_ids')
    def validate_document_ids(cls, v):
        if not v:
            raise ValueError("Document IDs cannot be empty")
        for doc_id in v:
            if not doc_id or not doc_id.strip():
                raise ValueError("Document ID cannot be empty")
        return [doc_id.strip() for doc_id in v]
    
    @validator('confirm_permanent_deletion')
    def validate_confirmation(cls, v):
        if not v:
            raise ValueError("Must confirm permanent deletion")
        return v

class DocumentUsageInfo(BaseModel):
    """Information about document usage before deletion."""
    document_id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Original filename")
    file_type: Optional[str] = Field(None, description="File type")
    file_size_bytes: Optional[int] = Field(None, description="File size in bytes")
    milvus_collection: Optional[str] = Field(None, description="Milvus collection name")
    notebook_count: int = Field(..., description="Number of notebooks using this document")
    notebooks_using: List[Dict[str, Any]] = Field(..., description="List of notebooks using this document")
    cross_references: int = Field(..., description="Number of cross-references")
    deletion_impact: Dict[str, Any] = Field(..., description="Impact of deletion")

class DocumentDeletionSummary(BaseModel):
    """Summary of document deletion operation."""
    document_id: str = Field(..., description="Document ID that was deleted")
    started_at: str = Field(..., description="When deletion started")
    completed_at: Optional[str] = Field(None, description="When deletion completed")
    success: bool = Field(..., description="Whether deletion was successful")
    milvus_deleted: bool = Field(..., description="Whether deleted from Milvus")
    database_deleted: bool = Field(..., description="Whether deleted from database")
    notebooks_removed: int = Field(..., description="Number of notebooks document was removed from")
    neo4j_deleted: bool = Field(..., description="Whether deleted from Neo4j")
    cache_cleared: bool = Field(..., description="Whether cache was cleared")
    errors: List[str] = Field(..., description="List of errors encountered")

class DocumentDeleteResponse(BaseModel):
    """Response model for document deletion operations."""
    success: bool = Field(..., description="Overall success of operation")
    message: str = Field(..., description="Summary message")
    total_requested: Optional[int] = Field(None, description="Total documents requested for deletion")
    successful_deletions: Optional[int] = Field(None, description="Number of successful deletions")
    failed_deletions: Optional[int] = Field(None, description="Number of failed deletions")
    deletion_details: List[DocumentDeletionSummary] = Field(..., description="Detailed deletion results")
    overall_errors: List[str] = Field(default_factory=list, description="Overall operation errors")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

# Chat Models

class NotebookChatRequest(BaseModel):
    """Request model for notebook chat."""
    message: str = Field(..., description="Chat message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")  
    include_context: Optional[bool] = Field(True, description="Whether to include notebook context")
    max_sources: Optional[int] = Field(5, description="Maximum number of sources to include")
    
    class Config:
        # Allow extra fields to be ignored instead of causing validation errors
        extra = "ignore"
        # Validate assignment to catch issues
        validate_assignment = True
    
    @validator('message', pre=True)
    def validate_message(cls, v):
        # Handle None or non-string values
        if v is None:
            raise ValueError("Chat message is required")
        # Convert to string if needed
        if not isinstance(v, str):
            v = str(v)
        # Check for empty after stripping
        if not v.strip():
            raise ValueError("Chat message cannot be empty")
        return v.strip()
        
    @validator('conversation_id', pre=True)
    def validate_conversation_id(cls, v):
        if v is None:
            return None
        return str(v) if v else None
        
    @validator('include_context', pre=True)
    def validate_include_context(cls, v):
        if v is None:
            return True
        # Handle string boolean values
        if isinstance(v, str):
            return v.lower() in ('true', '1', 'yes', 'on')
        return bool(v)
        
    @validator('max_sources', pre=True) 
    def validate_max_sources(cls, v):
        if v is None:
            return 5
        try:
            val = int(v) if not isinstance(v, int) else v
            return max(1, min(20, val))  # Clamp between 1 and 20
        except (ValueError, TypeError):
            return 5

class NotebookChatResponse(BaseModel):
    """Streaming response model for notebook chat."""
    answer: str = Field(..., description="Chat response")
    sources: List[NotebookRAGSource] = Field(default_factory=list, description="Sources used")
    notebook_id: str = Field(..., description="Notebook ID")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")