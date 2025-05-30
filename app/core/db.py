from sqlalchemy import create_engine, Column, Integer, String, JSON, TIMESTAMP, text, Boolean, ForeignKey, func
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
    manifest_id = Column(Integer, ForeignKey('mcp_manifests.id'), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now, onupdate=server_default_now)

    # Relationship with MCPManifest
    manifest = relationship("MCPManifest", back_populates="tools")

class MCPManifest(Base):
    __tablename__ = "mcp_manifests"
    id = Column(Integer, primary_key=True, index=True)
    url = Column(String(255), unique=True, nullable=False, index=True)
    hostname = Column(String(255), nullable=True)
    api_key = Column(String(255), nullable=True)
    content = Column(JSON, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=server_default_now, onupdate=server_default_now)

    # Relationship with MCPTool
    tools = relationship("MCPTool", back_populates="manifest")

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

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 