from sqlalchemy import create_engine, Column, Integer, String, JSON, TIMESTAMP, text, Boolean, ForeignKey, func, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from app.core.config import get_settings
import os

settings = get_settings()

# Use SQLite for local development if PostgreSQL connection fails
try:
    DATABASE_URL = (
        f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )
    engine = create_engine(DATABASE_URL)
    # Test connection
    with engine.connect() as conn:
        pass
    print("Successfully connected to PostgreSQL database")
    is_sqlite = False
except Exception as e:
    print(f"PostgreSQL connection failed: {str(e)}")
    print("Falling back to SQLite database")
    # Use SQLite as fallback
    DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "sqlite.db")
    DATABASE_URL = f"sqlite:///{DB_PATH}"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    is_sqlite = True

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Use appropriate timestamp defaults based on database
server_default_now = func.current_timestamp() if is_sqlite else text('now()')

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
    is_active = Column(Boolean, nullable=False, server_default=text("1") if is_sqlite else text("true"))
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
    config_type = Column(String(20), nullable=False)  # 'manifest' or 'command'
    
    # Manifest-based configuration
    manifest_url = Column(String(255), nullable=True)
    hostname = Column(String(255), nullable=True)
    api_key = Column(String(255), nullable=True)
    
    # Command-based configuration
    command = Column(String(500), nullable=True)
    args = Column(JSON, nullable=True)  # Array of command arguments
    env = Column(JSON, nullable=True)   # Environment variables dict
    working_directory = Column(String(500), nullable=True)
    
    # Process management
    process_id = Column(Integer, nullable=True)  # PID when running
    is_running = Column(Boolean, default=False)
    restart_policy = Column(String(20), default="on-failure")  # 'always', 'on-failure', 'never'
    max_restarts = Column(Integer, default=3)
    restart_count = Column(Integer, default=0)
    
    # Common fields
    is_active = Column(Boolean, nullable=False, server_default=text("1") if is_sqlite else text("true"))
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

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 