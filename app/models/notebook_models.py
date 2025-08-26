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
    source_type: Optional[str] = Field("document", description="Source type: 'document' or 'memory'")

class ProjectData(BaseModel):
    """Structured project data extracted from content."""
    name: str = Field(..., description="Project name")
    company: Optional[str] = Field(None, description="Company name (N/A if not found)")
    year: Optional[str] = Field(None, description="Project year or year range (N/A if not found)")
    description: str = Field(..., description="Project description")
    source_chunk_id: Optional[str] = Field(None, description="Source chunk ID where project was found")
    confidence_score: float = Field(0.0, description="Extraction confidence score (0.0-1.0)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional extracted metadata")

class NotebookRAGResponse(BaseModel):
    """Response model for notebook RAG queries."""
    notebook_id: str = Field(..., description="Notebook ID")
    query: str = Field(..., description="Original query")
    sources: List[NotebookRAGSource] = Field(..., description="Query results")
    total_sources: int = Field(..., description="Total number of sources found")
    queried_documents: int = Field(..., description="Number of documents queried")
    collections_searched: List[str] = Field(..., description="Collections that were searched")
    extracted_projects: Optional[List[ProjectData]] = Field(None, description="Extracted structured project data")

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
    max_sources: Optional[int] = Field(15, description="Maximum number of sources to include")
    
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
            return 15
        try:
            val = int(v) if not isinstance(v, int) else v
            return max(1, min(20, val))  # Clamp between 1 and 20
        except (ValueError, TypeError):
            return 15

class NotebookChatResponse(BaseModel):
    """Streaming response model for notebook chat."""
    answer: str = Field(..., description="Chat response")
    sources: List[NotebookRAGSource] = Field(default_factory=list, description="Sources used")
    notebook_id: str = Field(..., description="Notebook ID")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

# Memory Models

class MemoryCreateRequest(BaseModel):
    """Request model for creating a memory."""
    name: str = Field(..., min_length=1, max_length=255, description="Memory name")
    description: Optional[str] = Field(None, description="Memory description")
    content: str = Field(..., min_length=1, description="Memory content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Memory name cannot be empty")
        return v.strip()
    
    @validator('content')
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError("Memory content cannot be empty")
        return v.strip()

class MemoryUpdateRequest(BaseModel):
    """Request model for updating a memory."""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Updated memory name")
    description: Optional[str] = Field(None, description="Updated description")
    content: Optional[str] = Field(None, min_length=1, description="Updated memory content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")
    
    @validator('name')
    def validate_name(cls, v):
        if v is not None and (not v or not v.strip()):
            raise ValueError("Memory name cannot be empty")
        return v.strip() if v else v
    
    @validator('content')
    def validate_content(cls, v):
        if v is not None and (not v or not v.strip()):
            raise ValueError("Memory content cannot be empty")
        return v.strip() if v else v

class MemoryResponse(BaseModel):
    """Response model for memory operations."""
    id: str = Field(..., description="Memory record ID")
    notebook_id: str = Field(..., description="Parent notebook ID")
    memory_id: str = Field(..., description="Unique memory ID")
    name: str = Field(..., description="Memory name")
    description: Optional[str] = Field(None, description="Memory description")
    content: str = Field(..., description="Memory content")
    milvus_collection: Optional[str] = Field(None, description="Milvus collection name")
    chunk_count: int = Field(..., description="Number of chunks created")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class MemoryListResponse(BaseModel):
    """Response model for listing memories."""
    memories: List[MemoryResponse] = Field(..., description="List of memories")
    total_count: int = Field(..., description="Total number of memories")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")

# Chunk Editing Models

class ContentType(str, Enum):
    """Content type for chunk editing."""
    DOCUMENT = "document"
    MEMORY = "memory"

class ChunkUpdateRequest(BaseModel):
    """Request model for updating a chunk."""
    content: str = Field(..., min_length=1, description="Updated chunk content")
    re_embed: bool = Field(True, description="Whether to re-embed the chunk after update")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the edit")
    
    @validator('content')
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError("Chunk content cannot be empty")
        return v.strip()

class ChunkEditHistory(BaseModel):
    """Model for chunk edit history."""
    id: str = Field(..., description="Edit record ID")
    chunk_id: str = Field(..., description="Chunk ID")
    original_content: str = Field(..., description="Original content")
    edited_content: str = Field(..., description="Edited content")
    edited_by: Optional[str] = Field(None, description="User who made the edit")
    edited_at: datetime = Field(..., description="When the edit was made")
    re_embedded: bool = Field(..., description="Whether the chunk was re-embedded")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Edit metadata")

class ChunkResponse(BaseModel):
    """Response model for chunk operations."""
    chunk_id: str = Field(..., description="Unique chunk ID")
    document_id: str = Field(..., description="Parent document or memory ID")
    content_type: ContentType = Field(..., description="Type of content (document or memory)")
    content: str = Field(..., description="Chunk content")
    vector: Optional[List[float]] = Field(None, description="Embedding vector")
    metadata: Dict[str, Any] = Field(..., description="Chunk metadata")
    edit_history: List[ChunkEditHistory] = Field(default_factory=list, description="Edit history")
    last_edited: Optional[datetime] = Field(None, description="Last edit timestamp")

class ChunkListResponse(BaseModel):
    """Response model for listing chunks."""
    chunks: List[ChunkResponse] = Field(..., description="List of chunks")
    total_count: int = Field(..., description="Total number of chunks")
    document_id: str = Field(..., description="Parent document or memory ID")
    content_type: ContentType = Field(..., description="Content type")
    edited_chunks_count: int = Field(..., description="Number of chunks that have been edited")

class BulkChunkReEmbedRequest(BaseModel):
    """Request model for bulk re-embedding chunks."""
    chunk_ids: List[str] = Field(..., min_items=1, description="Chunk IDs to re-embed")
    
    @validator('chunk_ids')
    def validate_chunk_ids(cls, v):
        if not v:
            raise ValueError("Chunk IDs cannot be empty")
        for chunk_id in v:
            if not chunk_id or not chunk_id.strip():
                raise ValueError("Chunk ID cannot be empty")
        return [chunk_id.strip() for chunk_id in v]

class ChunkOperationResponse(BaseModel):
    """Response model for chunk operations."""
    success: bool = Field(..., description="Whether operation was successful")
    chunk_id: str = Field(..., description="Chunk ID")
    message: str = Field(..., description="Operation result message")
    re_embedded: bool = Field(False, description="Whether chunk was re-embedded")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

class BulkChunkOperationResponse(BaseModel):
    """Response model for bulk chunk operations."""
    success: bool = Field(..., description="Whether overall operation was successful")
    total_requested: int = Field(..., description="Total chunks requested")
    successful_operations: int = Field(..., description="Number of successful operations")
    failed_operations: int = Field(..., description="Number of failed operations")
    operation_details: List[ChunkOperationResponse] = Field(..., description="Detailed operation results")
    message: str = Field(..., description="Overall operation message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")