"""
RAG Tool Configuration
======================

Configuration management for RAG MCP tool registration.
Eliminates hardcoding by loading from settings/environment.
"""

import logging
import os
from typing import Dict, Any, List, Optional
from app.core.db import SessionLocal, Settings as SettingsModel

logger = logging.getLogger(__name__)

class RAGToolConfig:
    """Configuration provider for RAG tool registration"""
    
    def __init__(self):
        self._config = None
        self._load_config()
    
    def _load_config(self):
        """Load RAG tool configuration from database settings"""
        try:
            db = SessionLocal()
            try:
                # Get RAG tool settings from database
                settings_row = db.query(SettingsModel).filter(
                    SettingsModel.category == 'rag_tool'
                ).first()
                
                if settings_row and settings_row.settings:
                    self._config = settings_row.settings
                    logger.info("Loaded RAG tool configuration from database")
                else:
                    self._config = self._get_default_config()
                    logger.info("Using default RAG tool configuration")
                    
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Failed to load RAG tool config: {e}")
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration with environment variable support"""
        return {
            "tool_name": os.getenv("RAG_TOOL_NAME", "rag_knowledge_search"),
            "description": os.getenv(
                "RAG_TOOL_DESCRIPTION", 
                "Intelligent document retrieval for in-house info using vector and keyword search with automatic collection selection"
            ),
            "endpoint": os.getenv("RAG_TOOL_ENDPOINT", "internal://rag_mcp_service"),
            "method": os.getenv("RAG_TOOL_METHOD", "POST"),
            "is_active": os.getenv("RAG_TOOL_ACTIVE", "true").lower() == "true",
            "is_manual": os.getenv("RAG_TOOL_MANUAL", "true").lower() == "true",
            "parameters": {
                "max_documents": {
                    "minimum": int(os.getenv("RAG_TOOL_MIN_DOCS", "1")),
                    "maximum": int(os.getenv("RAG_TOOL_MAX_DOCS", "50"))
                },
                "include_content_default": os.getenv("RAG_TOOL_INCLUDE_CONTENT", "true").lower() == "true"
            }
        }
    
    def get_tool_name(self) -> str:
        """Get the tool name"""
        return self._config.get("tool_name", "rag_knowledge_search")
    
    def get_description(self) -> str:
        """Get the tool description"""
        return self._config.get("description", "RAG document search tool")
    
    def get_endpoint(self) -> str:
        """Get the tool endpoint"""
        return self._config.get("endpoint", "internal://rag_mcp_service")
    
    def get_method(self) -> str:
        """Get the HTTP method"""
        return self._config.get("method", "POST")
    
    def is_active(self) -> bool:
        """Check if tool should be active"""
        return self._config.get("is_active", True)
    
    def is_manual(self) -> bool:
        """Check if tool is manually registered"""
        return self._config.get("is_manual", True)
    
    def get_max_documents_min(self) -> int:
        """Get minimum documents limit"""
        return self._config.get("parameters", {}).get("max_documents", {}).get("minimum", 1)
    
    def get_max_documents_max(self) -> int:
        """Get maximum documents limit"""
        return self._config.get("parameters", {}).get("max_documents", {}).get("maximum", 50)
    
    def get_include_content_default(self) -> bool:
        """Get default value for include_content parameter"""
        return self._config.get("parameters", {}).get("include_content_default", True)
    
    def get_tool_definition(self, collections_desc: str) -> Dict[str, Any]:
        """Get complete tool definition for registration"""
        return {
            "name": self.get_tool_name(),
            "description": self.get_description(),
            "endpoint": self.get_endpoint(),
            "method": self.get_method(),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query or question"
                    },
                    "collections": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": collections_desc
                    },
                    "max_documents": {
                        "type": "integer",
                        "description": "Maximum number of documents to return (optional - defaults to 8 from RAG config. Only specify if you need a different value)",
                        "minimum": self.get_max_documents_min(),
                        "maximum": self.get_max_documents_max()
                    },
                    "include_content": {
                        "type": "boolean",
                        "description": "Whether to include full document content or just metadata",
                        "default": self.get_include_content_default()
                    }
                },
                "required": ["query"]
            },
            "is_active": self.is_active(),
            "is_manual": self.is_manual()
        }
    
    def reload_config(self):
        """Reload configuration from database"""
        self._load_config()
        logger.info("RAG tool configuration reloaded")

# Global instance
_rag_tool_config = None

def get_rag_tool_config() -> RAGToolConfig:
    """Get the global RAG tool configuration instance"""
    global _rag_tool_config
    if _rag_tool_config is None:
        _rag_tool_config = RAGToolConfig()
    return _rag_tool_config

def reload_rag_tool_config():
    """Reload the global RAG tool configuration"""
    global _rag_tool_config
    if _rag_tool_config:
        _rag_tool_config.reload_config()
    else:
        _rag_tool_config = RAGToolConfig()