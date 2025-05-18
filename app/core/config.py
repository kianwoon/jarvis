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
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    
    # Qdrant
    QDRANT_HOST: str
    QDRANT_PORT: int
    
    # LLM Configuration
    LLM_MODEL: str
    LLM_API_KEY: Optional[str] = None
    LLM_BASE_URL: Optional[str] = None
    
    # Memory Service
    MEMORY_SERVICE_URL: str
    
    # Security
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings() 