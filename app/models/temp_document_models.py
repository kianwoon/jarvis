"""
Pydantic models for temporary document API contracts.
Defines request/response models for temporary document operations.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum

class DocumentStatus(str, Enum):
    """Document processing status."""
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"
    EXPIRED = "expired"

class TempDocumentUploadRequest(BaseModel):
    """Request model for temporary document upload."""
    conversation_id: str = Field(..., description="Associated conversation ID")
    filename: str = Field(..., description="Original filename")
    ttl_hours: int = Field(2, ge=1, le=24, description="Time-to-live in hours (1-24)")
    auto_include: bool = Field(True, description="Automatically include in chat context")
    
    @validator('conversation_id')
    def validate_conversation_id(cls, v):
        if not v or not v.strip():
            raise ValueError("Conversation ID cannot be empty")
        return v.strip()
    
    @validator('filename')
    def validate_filename(cls, v):
        if not v or not v.strip():
            raise ValueError("Filename cannot be empty")
        supported_extensions = {'.pdf', '.docx', '.xlsx', '.pptx', '.txt'}
        if not any(v.lower().endswith(ext) for ext in supported_extensions):
            raise ValueError(f"Unsupported file type. Supported: {', '.join(supported_extensions)}")
        return v.strip()

class TempDocumentMetadata(BaseModel):
    """Metadata for a temporary document."""
    file_size: int = Field(..., description="File size in bytes")
    chunk_count: int = Field(..., description="Number of chunks extracted")
    avg_quality_score: float = Field(..., description="Average quality score of chunks")
    mime_type: Optional[str] = Field(None, description="MIME type of the file")

class TempDocumentResponse(BaseModel):
    """Response model for temporary document operations."""
    temp_doc_id: str = Field(..., description="Unique temporary document ID")
    conversation_id: str = Field(..., description="Associated conversation ID")
    filename: str = Field(..., description="Original filename")
    collection_name: str = Field(..., description="LlamaIndex collection name")
    upload_timestamp: datetime = Field(..., description="Upload timestamp")
    expiry_timestamp: datetime = Field(..., description="Expiry timestamp")
    is_included: bool = Field(..., description="Whether included in chat context")
    status: DocumentStatus = Field(..., description="Current processing status")
    metadata: TempDocumentMetadata = Field(..., description="Document metadata")

class TempDocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    success: bool = Field(..., description="Whether upload was successful")
    temp_doc_id: str = Field(..., description="Temporary document ID")
    message: Optional[str] = Field(None, description="Success or error message")
    metadata: Optional[TempDocumentResponse] = Field(None, description="Document metadata if successful")

class TempDocumentListResponse(BaseModel):
    """Response model for listing temporary documents."""
    conversation_id: str = Field(..., description="Conversation ID")
    documents: List[TempDocumentResponse] = Field(..., description="List of temporary documents")
    total_count: int = Field(..., description="Total number of documents")
    active_count: int = Field(..., description="Number of active (included) documents")

class TempDocumentPreferencesRequest(BaseModel):
    """Request model for updating document preferences."""
    is_included: Optional[bool] = Field(None, description="Whether to include in chat context")
    ttl_hours: Optional[int] = Field(None, ge=1, le=24, description="New TTL in hours")

class ConversationPreferencesRequest(BaseModel):
    """Request model for conversation-level preferences."""
    auto_include_new_docs: bool = Field(True, description="Auto-include new documents")
    default_ttl_hours: int = Field(2, ge=1, le=24, description="Default TTL for new documents")
    include_temp_docs_in_chat: bool = Field(True, description="Include temp docs in chat by default")
    max_documents_per_conversation: int = Field(5, ge=1, le=10, description="Max documents per conversation")

class ConversationPreferencesResponse(BaseModel):
    """Response model for conversation preferences."""
    conversation_id: str = Field(..., description="Conversation ID")
    auto_include_new_docs: bool = Field(..., description="Auto-include new documents")
    default_ttl_hours: int = Field(..., description="Default TTL for new documents")
    include_temp_docs_in_chat: bool = Field(..., description="Include temp docs in chat by default")
    max_documents_per_conversation: int = Field(..., description="Max documents per conversation")

class TempDocumentQueryRequest(BaseModel):
    """Request model for querying temporary documents."""
    conversation_id: str = Field(..., description="Conversation ID")
    query: str = Field(..., min_length=1, description="Query string")
    include_all_docs: bool = Field(False, description="Include all docs or only active ones")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")
    
    @validator('query')
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

class TempDocumentQuerySource(BaseModel):
    """Source information from document query."""
    content: str = Field(..., description="Content of the source")
    metadata: Dict[str, Any] = Field(..., description="Source metadata")
    score: float = Field(..., description="Relevance score")
    temp_doc_id: str = Field(..., description="Source document ID")
    filename: str = Field(..., description="Source filename")

class TempDocumentQueryResponse(BaseModel):
    """Response model for document queries."""
    conversation_id: str = Field(..., description="Conversation ID")
    query: str = Field(..., description="Original query")
    sources: List[TempDocumentQuerySource] = Field(..., description="Query results")
    total_sources: int = Field(..., description="Total number of sources")
    queried_documents: int = Field(..., description="Number of documents queried")

class ChatContextRequest(BaseModel):
    """Request model for chat with temporary document context."""
    message: str = Field(..., min_length=1, description="Chat message")
    conversation_id: str = Field(..., description="Conversation ID")
    include_temp_docs: Optional[bool] = Field(None, description="Override temp doc inclusion")
    active_temp_doc_ids: Optional[List[str]] = Field(None, description="Specific docs to include")
    
    @validator('message')
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()

class TempDocumentContextInfo(BaseModel):
    """Information about temporary document context."""
    has_temp_docs: bool = Field(..., description="Whether conversation has temp docs")
    total_docs: int = Field(..., description="Total number of temp docs")
    active_docs: int = Field(..., description="Number of active temp docs")
    active_doc_ids: List[str] = Field(..., description="List of active document IDs")
    last_updated: Optional[datetime] = Field(None, description="Last context update time")

class ChatWithTempDocsResponse(BaseModel):
    """Response model for chat with temporary document context."""
    message: str = Field(..., description="Chat response")
    conversation_id: str = Field(..., description="Conversation ID")
    temp_doc_context: Optional[TempDocumentContextInfo] = Field(None, description="Document context info")
    sources: Optional[List[TempDocumentQuerySource]] = Field(None, description="Sources used in response")
    routing_type: Optional[str] = Field(None, description="How the query was routed")

class TempDocumentCleanupResponse(BaseModel):
    """Response model for cleanup operations."""
    success: bool = Field(..., description="Whether cleanup was successful")
    cleaned_count: int = Field(..., description="Number of documents cleaned up")
    conversation_id: Optional[str] = Field(None, description="Conversation ID if applicable")
    message: str = Field(..., description="Cleanup result message")

class TempDocumentStatusResponse(BaseModel):
    """Response model for document status check."""
    temp_doc_id: str = Field(..., description="Temporary document ID")
    status: DocumentStatus = Field(..., description="Current status")
    timestamp: datetime = Field(..., description="Status timestamp")
    message: Optional[str] = Field(None, description="Status message")
    progress: Optional[float] = Field(None, ge=0, le=1, description="Processing progress (0-1)")

class TempDocumentBulkActionRequest(BaseModel):
    """Request model for bulk actions on temporary documents."""
    conversation_id: str = Field(..., description="Conversation ID")
    action: str = Field(..., description="Action to perform")
    temp_doc_ids: List[str] = Field(..., min_items=1, description="Document IDs to act on")
    
    @validator('action')
    def validate_action(cls, v):
        allowed_actions = {'delete', 'include', 'exclude', 'extend_ttl'}
        if v not in allowed_actions:
            raise ValueError(f"Invalid action. Allowed: {', '.join(allowed_actions)}")
        return v

class TempDocumentBulkActionResponse(BaseModel):
    """Response model for bulk actions."""
    success: bool = Field(..., description="Whether bulk action was successful")
    action: str = Field(..., description="Action performed")
    processed_count: int = Field(..., description="Number of documents processed")
    failed_count: int = Field(..., description="Number of documents that failed")
    failed_ids: List[str] = Field(..., description="IDs of documents that failed")
    message: str = Field(..., description="Result message")

class TempDocumentStatsResponse(BaseModel):
    """Response model for temporary document statistics."""
    conversation_id: str = Field(..., description="Conversation ID")
    total_documents: int = Field(..., description="Total number of documents")
    active_documents: int = Field(..., description="Number of active documents")
    total_chunks: int = Field(..., description="Total number of chunks")
    total_size_bytes: int = Field(..., description="Total size in bytes")
    avg_quality_score: float = Field(..., description="Average quality score")
    oldest_document: Optional[datetime] = Field(None, description="Oldest document timestamp")
    newest_document: Optional[datetime] = Field(None, description="Newest document timestamp")

# Error response models
class TempDocumentError(BaseModel):
    """Standard error response for temporary document operations."""
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    temp_doc_id: Optional[str] = Field(None, description="Related document ID")
    conversation_id: Optional[str] = Field(None, description="Related conversation ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")

class ValidationError(BaseModel):
    """Validation error details."""
    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Validation error message")
    invalid_value: Any = Field(..., description="Invalid value provided")

class TempDocumentValidationError(BaseModel):
    """Validation error response."""
    error: str = Field(..., description="General error message")
    validation_errors: List[ValidationError] = Field(..., description="Detailed validation errors")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")