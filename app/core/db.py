from sqlalchemy import create_engine, Column, Integer, String, JSON, TIMESTAMP, text, Boolean, ForeignKey, func, Float, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from app.core.config import get_settings
import os


settings = get_settings()

# PostgreSQL ONLY - NO SQLITE!
DATABASE_URL = (
    f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
    f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
)

# Enhanced connection pooling for automation workflows with 20+ agents
engine = create_engine(
    DATABASE_URL,
    pool_size=20,           # Max 20 persistent connections in pool
    max_overflow=30,        # Allow 30 additional connections beyond pool_size
    pool_pre_ping=True,     # Validate connections before use
    pool_recycle=3600,      # Recycle connections every hour
    pool_timeout=30,        # Wait 30 seconds for connection from pool
    echo=False,             # Set to True for SQL debugging
    connect_args={
        "application_name": "jarvis_automation",
        "options": "-c statement_timeout=60000"  # 60 second statement timeout
    }
)
# Test connection
try:
    with engine.connect() as conn:
        pass
    print("Successfully connected to PostgreSQL database")
except Exception as e:
    print(f"PostgreSQL connection failed: {str(e)}")
    raise Exception("PostgreSQL connection required - NO SQLITE FALLBACK!")
    
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# PostgreSQL timestamp default
server_default_now = text('now()')

class Settings(Base):
    __tablename__ = "settings"

    id = Column(Integer, primary_key=True, index=True)
    category = Column(String(50), unique=True, nullable=False, index=True)
    settings = Column(JSON, nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now, onupdate=server_default_now)

class MCPTool(Base):
    __tablename__ = "mcp_tools"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(String, nullable=True)
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False, server_default="POST")
    parameters = Column(JSON, nullable=True)
    headers = Column(JSON, nullable=True)
    is_active = Column(Boolean, nullable=False, server_default=text("true"))
    is_manual = Column(Boolean, nullable=False, server_default=text("false"))  # Track manually added tools
    manifest_id = Column(Integer, ForeignKey('mcp_manifests.id'), nullable=True)  # Legacy column for backward compatibility
    server_id = Column(Integer, ForeignKey('mcp_servers.id'), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now, onupdate=server_default_now)

    # Relationships
    manifest = relationship("MCPManifest", back_populates="tools")  # Legacy relationship
    server = relationship("MCPServer", back_populates="tools")

class MCPServer(Base):
    __tablename__ = "mcp_servers"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    config_type = Column(String(20), nullable=False)  # 'manifest', 'command', or 'remote_http'
    
    # Manifest-based configuration
    manifest_url = Column(String(255), nullable=True)
    hostname = Column(String(255), nullable=True)
    api_key = Column(String(255), nullable=True)
    oauth_credentials = Column(JSON, nullable=True)  # OAuth2 credentials
    
    # Command-based configuration
    command = Column(String(500), nullable=True)
    args = Column(JSON, nullable=True)  # Array of command arguments
    env = Column(JSON, nullable=True)   # Environment variables dict
    working_directory = Column(String(500), nullable=True)
    
    # Remote HTTP/SSE MCP Server Configuration
    remote_config = Column(JSON, nullable=True)  # Remote server configuration
    
    # Process management
    process_id = Column(Integer, nullable=True)  # PID when running
    is_running = Column(Boolean, default=False)
    restart_policy = Column(String(20), default="on-failure")  # 'always', 'on-failure', 'never'
    max_restarts = Column(Integer, default=3)
    restart_count = Column(Integer, default=0)
    
    # Enhanced Error Handling Configuration
    enhanced_error_handling_config = Column(JSON, nullable=True)  # Enhanced error handling settings
    auth_refresh_config = Column(JSON, nullable=True)  # Authentication refresh configuration
    
    # Common fields
    is_active = Column(Boolean, nullable=False, server_default=text("true"))
    last_health_check = Column(TIMESTAMP(timezone=True), nullable=True)
    health_status = Column(String(20), default="unknown")  # 'healthy', 'unhealthy', 'unknown'
    
    created_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now, onupdate=server_default_now)

    # Relationship with MCPTool
    tools = relationship("MCPTool", back_populates="server")

class MCPManifest(Base):
    __tablename__ = "mcp_manifests"
    id = Column(Integer, primary_key=True, index=True)
    url = Column(String(255), unique=True, nullable=False, index=True)
    hostname = Column(String(255), nullable=True)
    api_key = Column(String(255), nullable=True)
    content = Column(JSON, nullable=False)
    server_id = Column(Integer, ForeignKey('mcp_servers.id'), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now, onupdate=server_default_now)

    # Relationships
    server = relationship("MCPServer", backref="manifests")
    tools = relationship("MCPTool", back_populates="manifest")  # Legacy relationship

class LangGraphAgent(Base):
    __tablename__ = "langgraph_agents"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False)
    role = Column(String(255), nullable=False)
    system_prompt = Column(String, nullable=False)
    tools = Column(JSON, default=list)
    description = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    config = Column(JSON, default=dict)  # Agent-specific configuration
    created_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now, onupdate=server_default_now)

class CollectionRegistry(Base):
    __tablename__ = "collection_registry"
    
    id = Column(Integer, primary_key=True, index=True)
    collection_name = Column(String(100), unique=True, nullable=False, index=True)
    collection_type = Column(String(50), nullable=False)
    description = Column(String, nullable=True)
    metadata_schema = Column(JSON, nullable=True)
    search_config = Column(JSON, nullable=True)
    access_config = Column(JSON, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now, onupdate=server_default_now)

class CollectionStatistics(Base):
    __tablename__ = "collection_statistics"
    
    id = Column(Integer, primary_key=True, index=True)
    collection_name = Column(String(100), ForeignKey('collection_registry.collection_name'), unique=True, nullable=False)
    document_count = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)
    storage_size_mb = Column(Float, default=0.0)
    avg_search_latency_ms = Column(Integer, nullable=True)
    last_updated = Column(TIMESTAMP(timezone=True), server_default=server_default_now)
    
    # Relationship
    collection = relationship("CollectionRegistry", backref="statistics")

class UserCollectionAccess(Base):
    __tablename__ = "user_collection_access"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100), nullable=False)
    collection_name = Column(String(100), ForeignKey('collection_registry.collection_name'), nullable=False)
    permission_level = Column(String(20), default='read')
    granted_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now)
    
    # Relationship
    collection = relationship("CollectionRegistry", backref="user_access")

# AI Automation Models
class AutomationWorkflow(Base):
    """Automation workflow definition using Langflow"""
    __tablename__ = "automation_workflows"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(String, nullable=True)
    langflow_config = Column(JSON, nullable=False)  # Langflow flow configuration
    trigger_config = Column(JSON, nullable=True)    # Trigger configuration (webhook, schedule, etc.)
    is_active = Column(Boolean, nullable=False, server_default=text("true"))
    created_by = Column(String(255), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now, onupdate=server_default_now)
    
    # Unique constraint on workflow name per user to prevent duplicates
    __table_args__ = (
        UniqueConstraint('name', 'created_by', name='uq_workflow_name_per_user'),
    )
    
    # Relationships
    executions = relationship("AutomationExecution", back_populates="workflow", cascade="all, delete-orphan")

class AutomationExecution(Base):
    """Automation workflow execution tracking"""
    __tablename__ = "automation_executions"
    
    id = Column(Integer, primary_key=True, index=True)
    workflow_id = Column(Integer, ForeignKey('automation_workflows.id'), nullable=False, index=True)
    execution_id = Column(String(255), nullable=False, unique=True, index=True)  # UUID for tracking
    status = Column(String(50), nullable=False, index=True)  # 'running', 'completed', 'failed', 'cancelled'
    input_data = Column(JSON, nullable=True)     # Input parameters
    output_data = Column(JSON, nullable=True)    # Final results
    execution_log = Column(JSON, nullable=True)  # Step-by-step execution log
    workflow_state = Column(JSON, nullable=True) # Workflow state and checkpoints
    error_message = Column(String, nullable=True) # Error details if failed
    started_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now)
    completed_at = Column(TIMESTAMP(timezone=True), nullable=True)
    
    # Relationship
    workflow = relationship("AutomationWorkflow", back_populates="executions")

class AutomationTrigger(Base):
    """Automation triggers (webhooks, schedules, events)"""
    __tablename__ = "automation_triggers"
    
    id = Column(Integer, primary_key=True, index=True)
    workflow_id = Column(Integer, ForeignKey('automation_workflows.id'), nullable=False, index=True)
    trigger_type = Column(String(50), nullable=False)  # 'webhook', 'schedule', 'event', 'manual'
    trigger_config = Column(JSON, nullable=False)      # Type-specific configuration
    is_active = Column(Boolean, nullable=False, server_default=text("true"))
    last_triggered = Column(TIMESTAMP(timezone=True), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now, onupdate=server_default_now)
    
    # Relationship
    workflow = relationship("AutomationWorkflow")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

from contextlib import contextmanager

# Knowledge Graph Enhancement Models

class KnowledgeGraphDocument(Base):
    """Enhanced document tracking for unified knowledge graph processing"""
    __tablename__ = "knowledge_graph_documents"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(String(255), unique=True, nullable=False, index=True)
    filename = Column(String(500), nullable=False)
    file_hash = Column(String(64), nullable=False, index=True)
    file_size_bytes = Column(Integer, nullable=True)
    file_type = Column(String(50), nullable=True)
    
    # Processing configuration
    milvus_collection = Column(String(100), nullable=True)
    neo4j_graph_id = Column(String(255), nullable=True)
    processing_mode = Column(String(50), default='unified')  # 'unified', 'milvus-only', 'neo4j-only'
    
    # Processing status and metrics
    processing_status = Column(String(50), default='pending', index=True)  # 'pending', 'processing', 'completed', 'failed', 'partial'
    entities_extracted = Column(Integer, default=0)
    relationships_extracted = Column(Integer, default=0)
    chunks_processed = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)
    processing_time_ms = Column(Integer, nullable=True)
    extraction_confidence = Column(Float, nullable=True)
    
    # Error tracking
    error_message = Column(String, nullable=True)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    # Metadata and provenance
    upload_metadata = Column(JSON, nullable=True)
    processing_config = Column(JSON, nullable=True)
    quality_scores = Column(JSON, nullable=True)
    
    created_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now, onupdate=server_default_now)
    processing_started_at = Column(TIMESTAMP(timezone=True), nullable=True)
    processing_completed_at = Column(TIMESTAMP(timezone=True), nullable=True)
    
    # Relationships
    quality_metrics = relationship("ExtractionQualityMetric", back_populates="document", cascade="all, delete-orphan")

class ExtractionQualityMetric(Base):
    """Quality metrics and validation for knowledge graph extraction"""
    __tablename__ = "extraction_quality_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(String(255), ForeignKey('knowledge_graph_documents.document_id'), nullable=False, index=True)
    chunk_id = Column(String(255), nullable=True, index=True)
    
    # Extraction results
    entities_discovered = Column(Integer, default=0)
    relationships_discovered = Column(Integer, default=0)
    entities_validated = Column(Integer, default=0)
    relationships_validated = Column(Integer, default=0)
    
    # Quality scores
    confidence_scores = Column(JSON, nullable=True)  # {entity_confidence: [], relationship_confidence: []}
    validation_scores = Column(JSON, nullable=True)  # Quality validation results
    
    # Processing details
    llm_model_used = Column(String(100), nullable=True)
    processing_method = Column(String(50), nullable=True)  # 'llm_enhanced', 'traditional', 'hybrid'
    processing_time_ms = Column(Integer, nullable=True)
    
    # Validation and errors
    validation_errors = Column(JSON, nullable=True)
    extraction_warnings = Column(JSON, nullable=True)
    
    # Cross-reference tracking
    milvus_chunk_ids = Column(JSON, nullable=True)  # Array of related Milvus chunk IDs
    neo4j_entity_ids = Column(JSON, nullable=True)  # Array of related Neo4j entity IDs
    
    created_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now)
    
    # Relationship
    document = relationship("KnowledgeGraphDocument", back_populates="quality_metrics")

class GraphSchemaEvolution(Base):
    """Track evolution of knowledge graph schema over time"""
    __tablename__ = "graph_schema_evolution"
    
    id = Column(Integer, primary_key=True, index=True)
    schema_version = Column(String(20), nullable=False, index=True)
    
    # Schema definitions
    entity_types = Column(JSON, nullable=False)  # {type: {description, examples, confidence_threshold}}
    relationship_types = Column(JSON, nullable=False)  # {type: {description, inverse, examples, confidence_threshold}}
    
    # Configuration snapshots
    confidence_thresholds = Column(JSON, nullable=True)
    extraction_config = Column(JSON, nullable=True)
    
    # Change tracking
    change_description = Column(String, nullable=True)
    change_type = Column(String(50), nullable=True)  # 'manual', 'automatic', 'user_approval', 'system_optimization'
    changes_summary = Column(JSON, nullable=True)  # {added: [], modified: [], removed: []}
    
    # Impact metrics
    documents_affected = Column(Integer, default=0)
    entities_reclassified = Column(Integer, default=0)
    relationships_reclassified = Column(Integer, default=0)
    
    # Provenance
    created_by = Column(String(100), nullable=True)
    trigger_event = Column(String(200), nullable=True)  # What triggered this schema change
    
    created_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now)
    
    # Add index for efficient querying
    __table_args__ = (
        UniqueConstraint('schema_version', name='uq_schema_version'),
    )

class DocumentCrossReference(Base):
    """Cross-reference mapping between Milvus chunks and Neo4j entities"""
    __tablename__ = "document_cross_references"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(String(255), ForeignKey('knowledge_graph_documents.document_id'), nullable=False, index=True)
    
    # Milvus references
    milvus_collection = Column(String(100), nullable=False)
    milvus_chunk_id = Column(String(255), nullable=False, index=True)
    chunk_text_preview = Column(String(500), nullable=True)  # First 500 chars for reference
    
    # Neo4j references
    neo4j_entity_id = Column(String(255), nullable=False, index=True)
    entity_name = Column(String(255), nullable=False)
    entity_type = Column(String(100), nullable=False)
    
    # Relationship metadata
    confidence_score = Column(Float, nullable=True)
    relationship_type = Column(String(100), nullable=True)  # How chunk relates to entity
    context_window = Column(JSON, nullable=True)  # Character positions in chunk
    
    # Quality and validation
    validation_status = Column(String(50), default='pending')  # 'pending', 'validated', 'rejected'
    manual_review = Column(Boolean, default=False)
    review_notes = Column(String, nullable=True)
    
    created_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now, onupdate=server_default_now)
    
    # Compound index for efficient lookups
    __table_args__ = (
        UniqueConstraint('milvus_chunk_id', 'neo4j_entity_id', name='uq_chunk_entity_mapping'),
    )

# IDC (Intelligent Document Comparison) Models
class IDCReferenceDocument(Base):
    """Reference documents with extracted markdown for validation purposes"""
    __tablename__ = "idc_reference_documents"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    document_type = Column(String(50), nullable=False)
    category = Column(String(100), nullable=True)
    
    # Original document info
    original_filename = Column(String(255), nullable=True)
    file_hash = Column(String(64), nullable=False, index=True)
    file_size_bytes = Column(Integer, nullable=True)
    
    # Extracted content (LLM-generated markdown)
    extracted_markdown = Column(String, nullable=False)
    extraction_metadata = Column(JSON, nullable=True)
    
    # Model configuration for extraction
    extraction_model = Column(String(100), nullable=True)
    extraction_config = Column(JSON, nullable=True)
    extraction_confidence = Column(Float, nullable=True)
    
    # Default validation settings
    default_validation_config = Column(JSON, nullable=True)
    recommended_extraction_modes = Column(JSON, nullable=True)  # ['paragraph', 'sentence', 'qa_pairs']
    
    # Processing metrics
    processing_time_ms = Column(Integer, nullable=True)
    version = Column(Integer, default=1)
    is_active = Column(Boolean, nullable=False, server_default=text("true"))
    
    # Metadata
    created_by = Column(String(255), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now, onupdate=server_default_now)
    
    # Relationships
    validation_sessions = relationship("IDCValidationSession", back_populates="reference_document", cascade="all, delete-orphan")
    templates = relationship("IDCTemplate", back_populates="reference_document", cascade="all, delete-orphan")

class IDCValidationSession(Base):
    """Validation sessions with granular processing tracking"""
    __tablename__ = "idc_validation_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    reference_document_id = Column(Integer, ForeignKey('idc_reference_documents.id'), nullable=False)
    
    # Input document info
    input_filename = Column(String(255), nullable=True)
    input_file_hash = Column(String(64), nullable=True)
    input_file_size_bytes = Column(Integer, nullable=True)
    
    # Granular extraction configuration
    extraction_mode = Column(String(50), nullable=False)  # 'sentence', 'paragraph', 'qa_pairs', 'section'
    extraction_config = Column(JSON, nullable=True)
    
    # Processing configuration
    validation_model = Column(String(100), nullable=True)
    max_context_usage = Column(Float, default=0.35)  # Conservative limit
    
    # Systematic processing tracking
    total_units_extracted = Column(Integer, default=0)
    units_processed = Column(Integer, default=0)
    units_failed = Column(Integer, default=0)
    
    # Extracted content
    extracted_units = Column(JSON, nullable=True)  # Array of extracted units with metadata
    
    # Validation results
    validation_results = Column(JSON, nullable=True)  # Detailed unit-by-unit results
    overall_score = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    completeness_score = Column(Float, nullable=True)
    
    # Processing metrics
    status = Column(String(50), default='pending', index=True)
    processing_start_time = Column(TIMESTAMP(timezone=True), nullable=True)
    processing_end_time = Column(TIMESTAMP(timezone=True), nullable=True)
    total_processing_time_ms = Column(Integer, nullable=True)
    average_context_usage = Column(Float, nullable=True)
    max_context_usage_recorded = Column(Float, nullable=True)
    
    # Error tracking
    error_message = Column(String, nullable=True)
    failed_units = Column(JSON, nullable=True)  # Units that failed processing
    
    created_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now)
    
    # Relationships
    reference_document = relationship("IDCReferenceDocument", back_populates="validation_sessions")
    unit_results = relationship("IDCUnitValidationResult", back_populates="validation_session", cascade="all, delete-orphan")

class IDCUnitValidationResult(Base):
    """Detailed validation results for each extracted unit"""
    __tablename__ = "idc_unit_validation_results"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), ForeignKey('idc_validation_sessions.session_id'), nullable=False, index=True)
    
    # Unit information
    unit_index = Column(Integer, nullable=False)
    unit_type = Column(String(50), nullable=True)  # 'sentence', 'paragraph', 'qa_pair', 'section'
    unit_content = Column(String, nullable=False)
    unit_metadata = Column(JSON, nullable=True)  # position, length, etc.
    
    # Validation details
    validation_score = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    validation_feedback = Column(String, nullable=True)
    
    # Reference matching
    matched_reference_sections = Column(JSON, nullable=True)
    similarity_scores = Column(JSON, nullable=True)
    
    # Processing metrics
    context_tokens_used = Column(Integer, nullable=True)
    context_usage_percentage = Column(Float, nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    llm_model_used = Column(String(100), nullable=True)
    
    # Quality indicators
    requires_human_review = Column(Boolean, default=False)
    quality_flags = Column(JSON, nullable=True)  # low_confidence, context_overflow, etc.
    
    created_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now)
    
    # Relationships
    validation_session = relationship("IDCValidationSession", back_populates="unit_results")

class IDCTemplate(Base):
    """Pre-configured validation templates for common use cases"""
    __tablename__ = "idc_templates"
    
    id = Column(Integer, primary_key=True, index=True)
    template_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(String, nullable=True)
    template_type = Column(String(50), nullable=False)  # 'contract_review', 'exam_grading', 'resume_screening'
    
    # Template configuration
    reference_document_id = Column(Integer, ForeignKey('idc_reference_documents.id'), nullable=True)
    default_extraction_mode = Column(String(50), nullable=False)
    validation_config = Column(JSON, nullable=False)
    
    # Usage tracking
    usage_count = Column(Integer, default=0)
    success_rate = Column(Float, nullable=True)
    
    # Metadata
    is_public = Column(Boolean, default=False)
    created_by = Column(String(255), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now, onupdate=server_default_now)
    
    # Relationships
    reference_document = relationship("IDCReferenceDocument", back_populates="templates")

@contextmanager
def get_db_session():
    """Context manager for database sessions with automatic cleanup"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 