from sqlalchemy import create_engine, Column, Integer, String, JSON, TIMESTAMP, text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from app.core.config import get_settings

settings = get_settings()

DATABASE_URL = (
    f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
    f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Settings(Base):
    __tablename__ = "settings"

    id = Column(Integer, primary_key=True, index=True)
    category = Column(String(50), unique=True, nullable=False, index=True)
    settings = Column(JSON, nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'), onupdate=text('now()'))

class MCPTool(Base):
    __tablename__ = "mcp_tools"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(String, nullable=True)
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False, server_default="POST")
    parameters = Column(JSON, nullable=True)
    headers = Column(JSON, nullable=True)
    is_active = Column(Boolean, nullable=False, server_default="true")
    manifest_id = Column(Integer, ForeignKey('mcp_manifests.id'), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'))
    updated_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'), onupdate=text('now()'))

    # Relationship with MCPManifest
    manifest = relationship("MCPManifest", back_populates="tools")

class MCPManifest(Base):
    __tablename__ = "mcp_manifests"
    id = Column(Integer, primary_key=True, index=True)
    url = Column(String(255), unique=True, nullable=False, index=True)
    content = Column(JSON, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'))
    updated_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'), onupdate=text('now()'))

    # Relationship with MCPTool
    tools = relationship("MCPTool", back_populates="manifest") 