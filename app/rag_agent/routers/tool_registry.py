"""
Collection Tool Registry for LLM-based routing

This module converts collections into LLM-callable tools and manages
the dynamic registry of available collections for intelligent routing.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from app.rag_agent.utils.types import SearchContext
from app.core.collection_registry_cache import get_all_collections, get_collection_config

logger = logging.getLogger(__name__)


class CollectionTool:
    """Represents a collection as an LLM-callable tool"""
    
    def __init__(self, collection_config: Dict):
        self.name = collection_config["collection_name"]
        self.description = collection_config["description"]
        self.collection_type = collection_config["collection_type"]
        self.search_config = collection_config["search_config"]
        self.access_config = collection_config.get("access_config", {})
        self.statistics = collection_config.get("statistics", {})
        self.metadata_schema = collection_config.get("metadata_schema", {})
        
        # Cache tool schema
        self._tool_schema = None
        self._last_updated = datetime.now()
        
    def to_tool_schema(self) -> Dict[str, Any]:
        """Convert collection to LLM tool schema for function calling"""
        
        # Use cached schema if recent
        if (self._tool_schema and 
            datetime.now() - self._last_updated < timedelta(minutes=30)):
            return self._tool_schema
        
        use_cases = self._get_use_cases()
        performance_hints = self._get_performance_hints()
        
        self._tool_schema = {
            "type": "function",
            "function": {
                "name": f"search_{self.name}",
                "description": self._build_description(use_cases, performance_hints),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": f"Search query optimized for {self.collection_type} content. "
                                         f"Use specific terminology relevant to: {use_cases}"
                        },
                        "strategy": {
                            "type": "string",
                            "enum": ["semantic", "keyword", "hybrid", "auto"],
                            "default": self.search_config.get("strategy", "auto"),
                            "description": "Search strategy: semantic for conceptual, keyword for exact terms, hybrid for both, auto for system choice"
                        },
                        "max_results": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 50,
                            "default": self.search_config.get("max_results", 10),
                            "description": "Maximum number of results to return"
                        },
                        "similarity_threshold": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": self.search_config.get("similarity_threshold", 0.7),
                            "description": "Minimum similarity score for results (0.0-1.0)"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
        
        self._last_updated = datetime.now()
        return self._tool_schema
    
    def _build_description(self, use_cases: str, performance_hints: str) -> str:
        """Build comprehensive tool description for LLM"""
        
        doc_count = self.statistics.get('document_count', 0)
        chunk_count = self.statistics.get('total_chunks', 0)
        
        description = f"""Search the '{self.name}' collection containing {self.collection_type} content.

CONTENT: {self.description}

BEST FOR: {use_cases}

COLLECTION STATS:
- Documents: {doc_count:,}
- Searchable chunks: {chunk_count:,}
- Update frequency: {self._get_update_frequency()}

{performance_hints}

Use this tool when the user's query relates to {self.collection_type} topics, especially: {use_cases}"""
        
        return description
    
    def _get_use_cases(self) -> str:
        """Generate specific use cases based on collection type"""
        
        # Banking-specific use cases
        banking_use_cases = {
            "regulatory_compliance": "regulatory requirements, compliance policies, Basel III, Dodd-Frank, SOX, KYC/AML procedures, audit requirements, legal guidelines",
            "product_documentation": "banking products, account types, loan programs, credit cards, investment products, fees, rates, product specifications, customer guides",
            "risk_management": "credit risk assessments, market risk analysis, operational risk policies, risk mitigation strategies, stress testing, risk appetite",
            "customer_support": "customer service procedures, troubleshooting guides, FAQ responses, complaint handling, account servicing, customer onboarding",
            "audit_reports": "internal audit findings, external audit reports, SOX compliance audits, regulatory examinations, control assessments",
            "training_materials": "employee training content, compliance training, onboarding materials, certification requirements, procedure manuals"
        }
        
        # General use cases
        general_use_cases = {
            "technical_docs": "API documentation, system architecture, technical specifications, developer guides, integration instructions",
            "policies_procedures": "company policies, standard operating procedures, governance documents, process guidelines",
            "meeting_notes": "meeting minutes, action items, decisions, project updates, team communications",
            "contracts_legal": "legal agreements, contracts, terms and conditions, legal documentation, regulatory filings",
            "hr_documents": "employee handbook, benefits information, performance management, organizational policies",
            "finance_accounting": "financial procedures, accounting policies, budget documents, financial reports"
        }
        
        # Combine and return appropriate use cases
        all_use_cases = {**banking_use_cases, **general_use_cases}
        return all_use_cases.get(self.collection_type, "general knowledge queries and information retrieval")
    
    def _get_performance_hints(self) -> str:
        """Get performance and usage hints"""
        
        hints = []
        
        # Size-based hints
        chunk_count = self.statistics.get('total_chunks', 0)
        if chunk_count > 100000:
            hints.append("LARGE COLLECTION: Use specific queries for better performance")
        elif chunk_count < 1000:
            hints.append("SMALL COLLECTION: Broader queries may be more effective")
        
        # Strategy hints based on collection type
        if self.collection_type in ["regulatory_compliance", "policies_procedures"]:
            hints.append("TIP: Use exact terminology and regulatory references for best results")
        elif self.collection_type in ["technical_docs", "api_documentation"]:
            hints.append("TIP: Include specific technical terms, function names, or API endpoints")
        elif self.collection_type in ["customer_support", "training_materials"]:
            hints.append("TIP: Use natural language questions as customers would ask them")
        
        # Search config hints
        if self.search_config.get("enable_bm25", False):
            hints.append("FEATURE: Supports both semantic and keyword search")
        
        return "\n".join(hints) if hints else "Use specific, relevant terminology for best results."
    
    def _get_update_frequency(self) -> str:
        """Estimate update frequency based on collection type"""
        
        frequency_map = {
            "regulatory_compliance": "monthly (regulatory changes)",
            "product_documentation": "weekly (product updates)",
            "risk_management": "monthly (policy reviews)",
            "customer_support": "weekly (procedure updates)",
            "audit_reports": "quarterly (audit cycles)",
            "training_materials": "monthly (content updates)",
            "technical_docs": "weekly (system updates)",
            "policies_procedures": "quarterly (policy reviews)",
            "meeting_notes": "daily (new meetings)",
            "contracts_legal": "as needed (new agreements)"
        }
        
        return frequency_map.get(self.collection_type, "as needed")
    
    def is_accessible(self, user_context: Optional[SearchContext] = None) -> bool:
        """Check if collection is accessible based on user context"""
        
        # If not restricted, everyone has access
        if not self.access_config.get("restricted", False):
            return True
        
        # If no user context provided, assume no access to restricted collections
        if not user_context or not user_context.user_id:
            return False
        
        # Check user permissions
        allowed_users = self.access_config.get("allowed_users", [])
        user_permissions = user_context.user_permissions or []
        
        # Check direct user access or permission-based access
        return (user_context.user_id in allowed_users or 
                any(perm in allowed_users for perm in user_permissions))


class CollectionToolRegistry:
    """Manages collection tools for LLM routing"""
    
    def __init__(self):
        self.tools: Dict[str, CollectionTool] = {}
        self._last_refresh = None
        self._refresh_interval = timedelta(minutes=15)  # Refresh every 15 minutes
        
        # Load initial collections
        self._load_collections()
    
    def _load_collections(self):
        """Load collections from registry cache"""
        try:
            collections = get_all_collections()
            logger.info(f"Loading {len(collections)} collections into tool registry")
            
            new_tools = {}
            for collection_config in collections:
                tool = CollectionTool(collection_config)
                new_tools[tool.name] = tool
                logger.debug(f"Loaded collection tool: {tool.name} ({tool.collection_type})")
            
            self.tools = new_tools
            self._last_refresh = datetime.now()
            
        except Exception as e:
            logger.error(f"Failed to load collections into tool registry: {e}")
            # Keep existing tools if refresh fails
    
    def refresh_if_needed(self):
        """Refresh collections if needed"""
        if (not self._last_refresh or 
            datetime.now() - self._last_refresh > self._refresh_interval):
            logger.info("Refreshing collection tool registry")
            self._load_collections()
    
    def get_available_tools(
        self, 
        user_context: Optional[SearchContext] = None,
        collection_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get available tools as LLM tool schemas
        
        Args:
            user_context: User context for access control
            collection_types: Filter by collection types
            
        Returns:
            List of tool schemas for LLM function calling
        """
        self.refresh_if_needed()
        
        available_tools = []
        
        for tool in self.tools.values():
            # Check access permissions
            if not tool.is_accessible(user_context):
                continue
            
            # Filter by collection types if specified
            if collection_types and tool.collection_type not in collection_types:
                continue
            
            try:
                tool_schema = tool.to_tool_schema()
                available_tools.append(tool_schema)
            except Exception as e:
                logger.error(f"Failed to generate tool schema for {tool.name}: {e}")
        
        logger.info(f"Generated {len(available_tools)} available tool schemas")
        return available_tools
    
    def get_tool(self, collection_name: str) -> Optional[CollectionTool]:
        """Get specific collection tool"""
        self.refresh_if_needed()
        return self.tools.get(collection_name)
    
    def get_tools_by_type(self, collection_type: str) -> List[CollectionTool]:
        """Get all tools of a specific collection type"""
        self.refresh_if_needed()
        return [tool for tool in self.tools.values() 
                if tool.collection_type == collection_type]
    
    def get_collection_names(self) -> List[str]:
        """Get all available collection names"""
        self.refresh_if_needed()
        return list(self.tools.keys())
    
    def validate_collection_access(
        self, 
        collection_names: List[str], 
        user_context: Optional[SearchContext] = None
    ) -> List[str]:
        """
        Validate and filter collection names based on access permissions
        
        Args:
            collection_names: List of collection names to validate
            user_context: User context for access control
            
        Returns:
            List of accessible collection names
        """
        accessible_collections = []
        
        for collection_name in collection_names:
            tool = self.get_tool(collection_name)
            if tool and tool.is_accessible(user_context):
                accessible_collections.append(collection_name)
            else:
                logger.warning(f"Collection {collection_name} not accessible for user context")
        
        return accessible_collections
    
    def get_collection_suggestions(
        self, 
        query_keywords: List[str],
        user_context: Optional[SearchContext] = None
    ) -> List[str]:
        """
        Suggest relevant collections based on query keywords
        
        Args:
            query_keywords: Keywords extracted from user query
            user_context: User context for filtering
            
        Returns:
            List of suggested collection names
        """
        suggestions = []
        keyword_scores = {}
        
        for tool in self.tools.values():
            if not tool.is_accessible(user_context):
                continue
            
            score = 0
            use_cases = tool._get_use_cases().lower()
            description = tool.description.lower()
            
            # Score based on keyword matches
            for keyword in query_keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in use_cases:
                    score += 3
                elif keyword_lower in description:
                    score += 2
                elif keyword_lower in tool.collection_type:
                    score += 1
            
            if score > 0:
                keyword_scores[tool.name] = score
        
        # Sort by score and return top suggestions
        sorted_suggestions = sorted(keyword_scores.items(), 
                                  key=lambda x: x[1], reverse=True)
        
        return [name for name, score in sorted_suggestions[:5]]


# Global registry instance
_tool_registry = None

def get_collection_tool_registry() -> CollectionToolRegistry:
    """Get or create the global collection tool registry"""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = CollectionToolRegistry()
    return _tool_registry