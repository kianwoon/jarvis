from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "llm_platform"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Database
    POSTGRES_USER: Optional[str] = "postgres"
    POSTGRES_PASSWORD: Optional[str] = "postgres"
    POSTGRES_DB: Optional[str] = "llm_platform"
    POSTGRES_HOST: Optional[str] = "postgres"
    POSTGRES_PORT: Optional[int] = 5432
    
    # Qdrant
    QDRANT_HOST: Optional[str] = "qdrant"
    QDRANT_PORT: Optional[int] = 6333
    
    # Neo4j
    NEO4J_HOST: Optional[str] = "neo4j"
    NEO4J_PORT: Optional[int] = 7687
    NEO4J_HTTP_PORT: Optional[int] = 7474
    NEO4J_USER: Optional[str] = "neo4j"
    NEO4J_PASSWORD: Optional[str] = "jarvis_neo4j_password"
    NEO4J_DATABASE: Optional[str] = "neo4j"
    NEO4J_URI: Optional[str] = "bolt://neo4j:7687"
    
    # LLM Configuration
    LLM_MODEL: Optional[str] = "llama2"
    LLM_API_KEY: Optional[str] = None
    LLM_BASE_URL: Optional[str] = None
    OLLAMA_BASE_URL: Optional[str] = "http://localhost:11434"
    
    # Memory Service
    MEMORY_SERVICE_URL: Optional[str] = "http://memory:8000"
    
    # Redis
    REDIS_HOST: Optional[str] = "redis"
    REDIS_PORT: Optional[int] = 6379
    REDIS_DB: Optional[int] = 0
    REDIS_PASSWORD: Optional[str] = None
    
    # Security
    SECRET_KEY: Optional[str] = "dev-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings() 