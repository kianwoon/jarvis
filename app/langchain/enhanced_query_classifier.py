"""
Enhanced Query Classifier with Hybrid Query Support
Supports configurable patterns and multiple classifications with confidence scores
"""
import re
import yaml
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of queries for routing"""
    RAG = "rag"
    TOOL = "tool"
    LLM = "llm"
    CODE = "code"
    MULTI_AGENT = "multi_agent"
    
    # Hybrid types
    TOOL_RAG = "tool_rag"
    TOOL_LLM = "tool_llm"
    RAG_LLM = "rag_llm"
    TOOL_RAG_LLM = "tool_rag_llm"
    
    @classmethod
    def is_hybrid(cls, query_type: 'QueryType') -> bool:
        """Check if a query type is hybrid"""
        return query_type in [cls.TOOL_RAG, cls.TOOL_LLM, cls.RAG_LLM, cls.TOOL_RAG_LLM]
    
    @classmethod
    def get_components(cls, query_type: 'QueryType') -> List['QueryType']:
        """Get component types for hybrid queries"""
        if query_type == cls.TOOL_RAG:
            return [cls.TOOL, cls.RAG]
        elif query_type == cls.TOOL_LLM:
            return [cls.TOOL, cls.LLM]
        elif query_type == cls.RAG_LLM:
            return [cls.RAG, cls.LLM]
        elif query_type == cls.TOOL_RAG_LLM:
            return [cls.TOOL, cls.RAG, cls.LLM]
        else:
            return [query_type]

@dataclass
class ClassificationResult:
    """Result of query classification"""
    query_type: QueryType
    confidence: float
    metadata: Dict[str, any]
    suggested_tools: List[str] = None
    suggested_agents: List[str] = None
    matched_patterns: List[Tuple[str, str]] = None
    
    def __post_init__(self):
        if self.suggested_tools is None:
            self.suggested_tools = []
        if self.suggested_agents is None:
            self.suggested_agents = []
        if self.matched_patterns is None:
            self.matched_patterns = []

class EnhancedQueryClassifier:
    """Enhanced classifier with configurable patterns and hybrid query support"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or str(Path(__file__).parent / "query_patterns_config.yaml")
        self.config = self._load_config()
        self._load_dynamic_settings()
        self._load_mcp_tools()
        self._load_rag_collections()
        self.compiled_patterns = self._compile_patterns()
        
    def _load_config(self) -> Dict:
        """Load configuration from Redis cache or YAML file"""
        try:
            # Try Redis cache first
            from app.core.enhanced_query_classifier_cache import get_enhanced_classifier_config
            config = get_enhanced_classifier_config()
            logger.info(f"Loaded query patterns config from cache/file")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            # Return default config if everything fails
            return self._get_default_config()
    
    def _load_dynamic_settings(self):
        """Load dynamic settings from Query Classifier Redis cache/database"""
        try:
            # Load LLM-based classifier settings from Query Classifier configuration
            from app.core.query_classifier_settings_cache import get_query_classifier_settings
            dynamic_settings = get_query_classifier_settings()
            
            # Override config settings with Query Classifier Redis-cached settings
            if 'settings' not in self.config:
                self.config['settings'] = {}
            
            self.config['settings'].update(dynamic_settings)
            # logger.info(f"Loaded Query Classifier settings: {dynamic_settings}")
            
            # Check if LLM-based classification is enabled
            # Use new enable_llm_classification flag and require model to be configured
            enable_llm = dynamic_settings.get('enable_llm_classification', False)
            # Try both old and new schema model field names
            llm_model = dynamic_settings.get('llm_model', '').strip() or dynamic_settings.get('model', '').strip()
            # Try both old and new prompt field names - system_prompt is preferred
            llm_system_prompt = dynamic_settings.get('system_prompt', '').strip() or dynamic_settings.get('llm_system_prompt', '').strip()
            
            if enable_llm and llm_model and llm_system_prompt:
                self.use_llm_classification = True
                logger.info(f"LLM-based query classification enabled with model: {llm_model}")
            else:
                self.use_llm_classification = False
                if not enable_llm:
                    logger.info("LLM-based query classification disabled in settings")
                elif not llm_model:
                    logger.info("LLM-based query classification disabled - no model configured")
                elif not llm_system_prompt:
                    logger.info("LLM-based query classification disabled - no system_prompt or llm_system_prompt configured")
                
        except ImportError as e:
            logger.warning(f"Query Classifier settings cache not available yet: {e}")
            self.use_llm_classification = False
        except Exception as e:
            logger.warning(f"Failed to load Query Classifier settings, using defaults: {e}")
            self.use_llm_classification = False
    
    def reload_settings(self):
        """Reload dynamic settings from cache/database for hot reloading"""
        logger.info("Reloading query classifier settings due to cache update")
        self._load_dynamic_settings()
    
    def _load_mcp_tools(self):
        """Load available MCP tools for better classification"""
        try:
            from app.core.mcp_tools_cache import get_enabled_mcp_tools
            self.available_mcp_tools = get_enabled_mcp_tools()
            self.mcp_tool_names = set(self.available_mcp_tools.keys()) if self.available_mcp_tools else set()
            logger.info(f"Loaded {len(self.mcp_tool_names)} MCP tools: {list(self.mcp_tool_names)}")
            # logger.info(f"[MCP TOOLS DEBUG] Sample tool data: {list(self.available_mcp_tools.items())[:2] if self.available_mcp_tools else 'No tools available'}")
            
        except Exception as e:
            logger.warning(f"Failed to load MCP tools: {e}")
            self.available_mcp_tools = {}
            self.mcp_tool_names = set()
    
    def _load_rag_collections(self):
        """Load available RAG collections from cache for intelligent routing"""
        try:
            from app.core.collection_registry_cache import get_all_collections
            collections = get_all_collections()
            
            # Build collection metadata for classification
            self.rag_collections = {}
            self.collection_keywords = set()
            
            for collection in collections:
                name = collection.get('collection_name', '')
                description = collection.get('description', '')
                collection_type = collection.get('collection_type', '')
                stats = collection.get('statistics', {})
                
                # Get document count from statistics
                doc_count = stats.get('document_count', 0) if stats else 0
                
                # Extract keywords from collection names and descriptions
                collection_info = {
                    'name': name,
                    'description': description,
                    'type': collection_type,
                    'document_count': doc_count,
                    'keywords': self._extract_collection_keywords(name, description)
                }
                
                self.rag_collections[name] = collection_info
                self.collection_keywords.update(collection_info['keywords'])
            
            logger.info(f"Loaded {len(self.rag_collections)} RAG collections with {len(self.collection_keywords)} keywords")
            
            # Log collection summary for debugging
            for name, info in self.rag_collections.items():
                logger.debug(f"Collection '{name}': {info['document_count']} docs, keywords: {info['keywords']}")
                
        except Exception as e:
            logger.warning(f"Failed to load RAG collections: {e}")
            self.rag_collections = {}
            self.collection_keywords = set()
    
    def _extract_collection_keywords(self, name: str, description: str) -> Set[str]:
        """Extract keywords from collection name and description for matching"""
        keywords = set()
        
        # Extract from collection name
        name_words = re.findall(r'\b\w+\b', name.lower())
        keywords.update(name_words)
        
        # Extract from description
        if description:
            desc_words = re.findall(r'\b\w+\b', description.lower())
            # Only include meaningful words (exclude common words)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
            meaningful_words = [word for word in desc_words if len(word) > 2 and word not in stop_words]
            keywords.update(meaningful_words)
        
        return keywords
    
    def _apply_temporal_detection(self, query_lower: str, scores: Dict, metadata: Dict):
        """Apply temporal query detection and auto-suggest datetime tools"""
        try:
            from app.core.temporal_context_manager import get_temporal_context_manager
            temporal_manager = get_temporal_context_manager()
            
            # Detect if this is a time-related query
            is_time_related, matched_keywords, confidence = temporal_manager.detect_time_related_query(query_lower)
            
            if is_time_related and confidence > 0.5:
                # Find datetime tools in available MCP tools
                datetime_tools = [t for t in self.mcp_tool_names if 'datetime' in t.lower() or 'time' in t.lower() or t == 'get_datetime']
                
                if datetime_tools:
                    # Boost TOOL score for temporal queries
                    temporal_boost = min(0.8, confidence * 0.9)  # Cap at 0.8, scale by confidence
                    scores[QueryType.TOOL] += temporal_boost
                    
                    # Add to metadata
                    metadata[QueryType.TOOL]["matched_patterns"].append(
                        ("temporal_detection", f"Detected time query (confidence: {confidence:.2f})")
                    )
                    metadata[QueryType.TOOL]["pattern_groups"].add("temporal_query")
                    metadata[QueryType.TOOL]["suggested_tools"].update(datetime_tools)
                    metadata[QueryType.TOOL]["matched_keywords"] = matched_keywords
                    metadata[QueryType.TOOL]["temporal_confidence"] = confidence
                    
                    logger.info(f"Temporal query detected: '{query_lower}' (confidence: {confidence:.2f}, tools: {datetime_tools})")
                else:
                    logger.warning("Temporal query detected but no datetime tools available")
                    
        except Exception as e:
            logger.debug(f"Temporal detection failed: {e}")

    def _apply_mcp_tool_patterns(self, query_lower: str, scores: Dict, metadata: Dict):
        """Apply MCP tool-specific patterns for better classification"""
        
        # First apply temporal detection (high priority for time queries)
        self._apply_temporal_detection(query_lower, scores, metadata)
        
        # Get tool patterns from configuration
        tool_patterns = self.config.get("mcp_tool_patterns", {})
        
        for tool_category, config in tool_patterns.items():
            # Dynamically populate available_tools for any category with empty tools from MCP cache
            if not config['available_tools']:
                # For datetime category, filter to datetime-related tools only
                if tool_category == 'datetime':
                    # Filter to datetime-related tools from MCP cache
                    datetime_tools = [t for t in self.mcp_tool_names if 'datetime' in t or 'date' in t or 'time' in t]
                    config['available_tools'] = datetime_tools
                    logger.debug(f"Dynamically populated {tool_category} with {len(datetime_tools)} tools: {datetime_tools}")
                else:
                    # Use all available MCP tools for other categories
                    config['available_tools'] = list(self.mcp_tool_names)
                    logger.debug(f"Dynamically populated {tool_category} tools from MCP cache: {len(self.mcp_tool_names)} tools")
            
            # Only apply if we have the relevant tools available
            if any(tool in self.mcp_tool_names for tool in config['available_tools']):
                for pattern_str in config['patterns']:
                    if re.search(pattern_str, query_lower):
                        # High confidence boost for tool queries
                        scores[QueryType.TOOL] += config['confidence']
                        metadata[QueryType.TOOL]["matched_patterns"].append((tool_category, pattern_str))
                        metadata[QueryType.TOOL]["pattern_groups"].add(f"mcp_{tool_category}")
                        
                        # Add available tools to suggestions
                        available_tools = [t for t in config['available_tools'] if t in self.mcp_tool_names]
                        metadata[QueryType.TOOL]["suggested_tools"].update(available_tools)
                        
                        logger.debug(f"MCP tool pattern matched: {tool_category} for query: {query_lower}")
    
    def _apply_rag_collection_matching(self, query_lower: str, scores: Dict, metadata: Dict):
        """Apply RAG collection matching for intelligent routing"""
        if not self.rag_collections:
            # No RAG collections available - boost TOOL score for all queries
            settings = self.config.get("settings", {})
            no_rag_boost = float(settings.get("no_rag_tool_boost", 0.3))
            scores[QueryType.TOOL] += no_rag_boost
            metadata[QueryType.TOOL]["matched_patterns"].append(("no_rag_collections", "fallback_to_tools"))
            metadata[QueryType.TOOL]["pattern_groups"].add("no_rag_fallback")
            logger.debug(f"No RAG collections available - boosting TOOL score for query: {query_lower}")
            return
        
        # Extract query keywords
        query_keywords = set(re.findall(r'\b\w+\b', query_lower))
        
        # Check for collection keyword matches
        rag_matches = []
        for collection_name, collection_info in self.rag_collections.items():
            # Calculate keyword overlap
            keyword_overlap = query_keywords.intersection(collection_info['keywords'])
            
            if keyword_overlap:
                overlap_score = len(keyword_overlap) / max(len(query_keywords), 1)
                rag_matches.append({
                    'collection': collection_name,
                    'overlap_score': overlap_score,
                    'matched_keywords': keyword_overlap,
                    'document_count': collection_info['document_count']
                })
        
        if rag_matches:
            # Sort by overlap score and document count
            rag_matches.sort(key=lambda x: (x['overlap_score'], x['document_count']), reverse=True)
            best_match = rag_matches[0]
            
            # Boost RAG score based on collection relevance
            settings = self.config.get("settings", {})
            rag_boost_cap = float(settings.get("rag_boost_cap", 0.6))
            rag_boost_multiplier = float(settings.get("rag_boost_multiplier", 0.8))
            rag_boost = min(rag_boost_cap, best_match['overlap_score'] * rag_boost_multiplier)
            scores[QueryType.RAG] += rag_boost
            
            metadata[QueryType.RAG]["matched_patterns"].append((
                "collection_match", 
                f"{best_match['collection']} (keywords: {', '.join(best_match['matched_keywords'])})"
            ))
            metadata[QueryType.RAG]["pattern_groups"].add("collection_keywords")
            metadata[QueryType.RAG]["suggested_collections"] = [match['collection'] for match in rag_matches[:3]]
            
            logger.debug(f"RAG collection match for query '{query_lower}': {best_match['collection']} "
                        f"(score: {rag_boost:.2f}, keywords: {best_match['matched_keywords']})")
        else:
            # No collection matches - prefer fresh web search for up-to-date information
            # Instead of hardcoding indicators, apply general freshness bias
            
            # Check for temporal keywords that suggest need for current info
            temporal_keywords = {'current', 'latest', 'recent', 'today', 'now', 'compare', 'vs', 'versus', 'difference'}
            has_temporal = bool(query_keywords.intersection(temporal_keywords))
            
            # Check for comparison keywords that benefit from fresh data
            comparison_keywords = {'compare', 'vs', 'versus', 'difference', 'better', 'best', 'advantage', 'disadvantage'}
            has_comparison = bool(query_keywords.intersection(comparison_keywords))
            
            # Default bias toward fresh web search when no RAG collections match
            settings = self.config.get("settings", {})
            freshness_boost = float(settings.get("freshness_base_boost", 0.4))
            
            if has_temporal:
                temporal_boost = float(settings.get("temporal_boost", 0.3))
                freshness_boost += temporal_boost
                metadata[QueryType.TOOL]["pattern_groups"].add("temporal_freshness")
                
            if has_comparison:
                comparison_boost = float(settings.get("comparison_boost", 0.2))
                freshness_boost += comparison_boost
                metadata[QueryType.TOOL]["pattern_groups"].add("comparison_freshness")
            
            scores[QueryType.TOOL] += freshness_boost
            metadata[QueryType.TOOL]["matched_patterns"].append((
                "freshness_bias", 
                f"No relevant collections - prefer fresh web search (boost: {freshness_boost:.2f})"
            ))
            
            logger.debug(f"Freshness bias applied for '{query_lower}' - boosting TOOL score by {freshness_boost:.2f} "
                        f"(temporal: {has_temporal}, comparison: {has_comparison})")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration if file not found"""
        return {
            "tool_patterns": {},
            "rag_patterns": {},
            "code_patterns": {},
            "multi_agent_patterns": {},
            "direct_llm_patterns": {},
            "hybrid_indicators": {},
            "mcp_tool_patterns": {},
            "settings": {
                "min_confidence_threshold": 0.1,
                "max_classifications": 3,
                "enable_hybrid_detection": True,
                "confidence_decay_factor": 0.8,
                "pattern_combination_bonus": 0.15,
                "no_rag_tool_boost": 0.3,
                "rag_boost_cap": 0.6,
                "rag_boost_multiplier": 0.8,
                "freshness_base_boost": 0.4,
                "temporal_boost": 0.3,
                "comparison_boost": 0.2,
                "hybrid_component_multiplier": 0.5,
                "strong_results_threshold": 0.2,
                "hybrid_confidence_multiplier": 0.6,
                "three_way_hybrid_multiplier": 0.5,
                "fallback_default_confidence": 0.5,
                "fallback_tool_confidence": 0.6
            }
        }
    
    def _compile_patterns(self) -> Dict[str, List[Tuple[re.Pattern, Dict]]]:
        """Compile regex patterns for efficiency"""
        compiled = {}
        
        # Map pattern categories to query types
        category_mapping = {
            "tool_patterns": QueryType.TOOL,
            "rag_patterns": QueryType.RAG,
            "code_patterns": QueryType.CODE,
            "multi_agent_patterns": QueryType.MULTI_AGENT,
            "direct_llm_patterns": QueryType.LLM
        }
        
        for category, query_type in category_mapping.items():
            compiled[query_type.value] = []
            category_patterns = self.config.get(category, {})
            
            for pattern_group, group_config in category_patterns.items():
                patterns = group_config.get("patterns", [])
                for pattern in patterns:
                    try:
                        compiled_pattern = re.compile(pattern, re.IGNORECASE)
                        compiled[query_type.value].append((
                            compiled_pattern,
                            {
                                "group": pattern_group,
                                "confidence_boost": group_config.get("confidence_boost", 0.25),
                                "suggested_tools": group_config.get("suggested_tools", []),
                                "suggested_agents": group_config.get("suggested_agents", [])
                            }
                        ))
                    except re.error as e:
                        logger.error(f"Invalid regex pattern '{pattern}': {e}")
        
        # Compile hybrid indicators
        compiled["hybrid"] = []
        for hybrid_type, hybrid_config in self.config.get("hybrid_indicators", {}).items():
            patterns = hybrid_config.get("patterns", [])
            for pattern in patterns:
                try:
                    compiled_pattern = re.compile(pattern, re.IGNORECASE)
                    compiled["hybrid"].append((
                        compiled_pattern,
                        {
                            "hybrid_type": hybrid_type,
                            "primary_types": hybrid_config.get("primary_types", []),
                            "confidence_threshold": hybrid_config.get("confidence_threshold", 0.5)
                        }
                    ))
                except re.error as e:
                    logger.error(f"Invalid hybrid pattern '{pattern}': {e}")
        
        return compiled
    
    async def classify(self, query: str, trace=None) -> List[ClassificationResult]:
        """
        Classify a query using LLM-based classification only
        
        Returns:
            List of ClassificationResult objects sorted by confidence
        """
        # Refresh settings to ensure we have the latest configuration
        try:
            self.reload_settings()
        except Exception as e:
            logger.warning(f"Failed to reload settings, using cached: {e}")
        
        # Use LLM-based classification only - no pattern fallback
        if hasattr(self, 'use_llm_classification') and self.use_llm_classification:
            return await self._llm_classify(query, trace=trace)
        else:
            logger.error("LLM classification not enabled - cannot classify queries without LLM")
            return await self._retry_llm_classification(query, "llm_not_enabled")
    
    async def _llm_classify(self, query: str, trace=None) -> List[ClassificationResult]:
        """Use LLM-based classification with your configured settings"""
        # Create classification span for tracing
        classification_span = None
        tracer = None
        if trace:
            try:
                from app.core.langfuse_integration import get_tracer
                tracer = get_tracer()
                if tracer.is_enabled():
                    classification_span = tracer.create_span(
                        trace,
                        name="query-classification",
                        metadata={
                            "operation": "query_classification",
                            "classifier_type": "llm",
                            "query_length": len(query)
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to create classification span: {e}")
        
        try:
            from app.core.llm_settings_cache import get_llm_settings, get_query_classifier_full_config
            from app.core.query_classifier_settings_cache import get_query_classifier_settings
            
            # Get query classifier specific settings - DO NOT use general LLM settings
            from app.core.timeout_settings_cache import get_query_classification_timeout
            classifier_specific_settings = get_query_classifier_settings()
            
            # Use query classifier specific model configuration
            llm_model = classifier_specific_settings.get('model', '').strip()
            max_tokens = int(classifier_specific_settings.get('max_tokens', 10))
            temperature = float(classifier_specific_settings.get('temperature', 0.1))
            
            # Debug logging for configuration values
            logger.info(f"LLM Classifier using model: {llm_model}")
            logger.info(f"LLM Classifier config - max_tokens: {max_tokens}, temperature: {temperature}")
            
            # Validate that model is configured
            if not llm_model:
                logger.error("Query classifier model is not configured - check query_classifier.model setting")
                return "TOOL", 0.5, "No model configured for query classifier"
            
            # Use classifier-specific settings for classification behavior
            min_confidence = float(classifier_specific_settings.get('min_confidence_threshold', 0.1))
            max_classifications = int(classifier_specific_settings.get('max_classifications', 3))
            timeout_seconds = get_query_classification_timeout()  # Use centralized timeout config
            
            # Get system prompt from classifier-specific settings
            system_prompt = classifier_specific_settings.get('system_prompt', '').strip()
            
            if system_prompt:
                # Build detailed tool and collection information
                tools_info = self._build_tools_info()
                collections_info = self._build_collections_info()
                
                # Debug logging to verify what data is being fed to classifier
                
                # Process template placeholders in system prompt
                processed_system_prompt = system_prompt.format(
                    rag_collection=collections_info,
                    mcp_tools=tools_info
                )
                
                # Build messages for chat endpoint
                messages = [
                    {"role": "system", "content": processed_system_prompt},
                    {"role": "user", "content": f'**Query to classify:** "{query}"\n\nAnswer:'}
                ]
                
                # Debug log the messages
                logger.info(f"LLM Classifier using chat endpoint with {len(messages)} messages")
                logger.info(f"LLM Classifier system message length: {len(messages[0]['content'])} chars")
                logger.info(f"LLM Classifier system message preview:\n{messages[0]['content'][:300]}...")  # More context
                logger.info(f"LLM Classifier tools summary: {len(self.mcp_tool_names)} tools, {len(self.rag_collections)} collections")
                
                # Log tools and collections count for monitoring
                if self.mcp_tool_names:
                    logger.info(f"Available tools: {', '.join(list(self.mcp_tool_names)[:5])}{'...' if len(self.mcp_tool_names) > 5 else ''}")
                if self.rag_collections:
                    logger.info(f"Available collections: {', '.join(list(self.rag_collections.keys())[:3])}{'...' if len(self.rag_collections) > 3 else ''}")
            else:
                # No system prompt configured - cannot classify
                logger.error("No system prompt configured for query classifier")
                return await self._retry_llm_classification(query, "no_system_prompt")

            # Use Query Classifier LLM settings to make the classification call
            from app.llm.ollama import OllamaLLM
            from app.llm.base import LLMConfig
            from app.core.llm_settings_cache import get_llm_settings
            
            # Get main LLM settings for model_server only
            main_llm_settings = get_llm_settings()
            
            # Use Query Classifier specific configuration - no hardcoding
            llm_config = LLMConfig(
                model_name=llm_model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.95  # Use default top_p for classification
            )
            
            # Use query classifier model server from settings ONLY - no hardcoded fallbacks
            import os
            # Get full query classifier config which includes model_server
            from app.core.llm_settings_cache import get_query_classifier_full_config
            query_classifier_full_config = get_query_classifier_full_config()
            model_server = query_classifier_full_config.get('model_server', '').strip()
            
            if not model_server:
                # Fallback to main LLM model_server if query classifier doesn't have one
                from app.core.llm_settings_cache import get_main_llm_full_config
                main_llm_full_config = get_main_llm_full_config()
                model_server = main_llm_full_config.get('model_server', '').strip()
                
            if not model_server:
                # Use environment variable as last resort (not hardcoded)
                model_server = os.environ.get("OLLAMA_BASE_URL", "")
                
            if not model_server:
                logger.error("No model server configured in settings or environment. Please configure model_server in LLM settings.")
                return await self._retry_llm_classification(query, "no_model_server")
            
            # Apply Docker environment detection to the model_server URL
            is_docker = os.path.exists('/root') or os.environ.get('DOCKER_ENVIRONMENT') or os.path.exists('/.dockerenv')
            if 'localhost' in model_server and is_docker:
                model_server = model_server.replace('localhost', 'host.docker.internal')
                logger.info(f"Docker environment detected, converted URL to: {model_server}")
            
            logger.info(f"LLM Classifier using model server: {model_server}")
            logger.info(f"LLM Classifier using model: {llm_config.model_name}")
            
            llm = OllamaLLM(llm_config, base_url=model_server)
            
            # Create LLM generation span for classification
            generation_span = None
            if classification_span and tracer:
                try:
                    generation_span = tracer.create_llm_generation_span(
                        classification_span,
                        model=llm_config.model_name,
                        prompt=str(messages),  # Log messages for tracing
                        operation="classification"
                    )
                except Exception as e:
                    logger.warning(f"Failed to create classification generation span: {e}")
            
            # Add timeout wrapper for LLM classification using centralized timeout config
            try:
                response = await asyncio.wait_for(
                    llm.chat(messages),
                    timeout=timeout_seconds
                )
                response_text = response.text
            except asyncio.TimeoutError:
                logger.error(f"LLM classification timed out after {timeout_seconds} seconds")
                return await self._retry_llm_classification(query, "llm_timeout")
            
            # End generation span with result
            if generation_span and tracer:
                try:
                    # Estimate usage based on messages content
                    messages_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
                    usage = tracer.estimate_token_usage(messages_text, response_text)
                    tracer.end_span_with_result(generation_span, {
                        "raw_response": response_text[:500],
                        "usage": usage
                    }, True)
                except Exception as e:
                    logger.warning(f"Failed to end classification generation span: {e}")
            
            logger.info(f"LLM Classifier raw response: '{response_text}'")
            
            # Dynamic model behavior detection and response processing
            clean_response = response_text
            try:
                from app.llm.response_analyzer import detect_model_thinking_behavior
                
                # Detect model behavior from the response (for caching and future use)
                is_thinking, detection_confidence = detect_model_thinking_behavior(response_text, llm_model)
                logger.info(f"[CLASSIFIER DETECTION] Model: {llm_model}")
                logger.info(f"[CLASSIFIER DETECTION] Detected thinking behavior: {is_thinking} (confidence: {detection_confidence:.2f})")
                
                # Process response based on detected behavior
                if is_thinking and detection_confidence > 0.8:
                    # Remove thinking tags for thinking models
                    import re
                    clean_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL | re.IGNORECASE).strip()
                    logger.info(f"[CLASSIFIER PROCESSING] Removed thinking tags from response")
                    
                    # Log recommendation for configuration
                    logger.info(f"[CLASSIFIER RECOMMENDATION] Model {llm_model} appears to be a thinking model. Consider setting query_classifier.mode to 'non-thinking' for better classification performance.")
                else:
                    # Non-thinking model or low confidence - use response as-is
                    clean_response = response_text.strip()
                    logger.info(f"[CLASSIFIER PROCESSING] Using response as-is (non-thinking model)")
                
                # Store behavior profile for future use (the analyzer handles caching)
                logger.info(f"[CLASSIFIER CACHING] Behavior profile cached for model: {llm_model}")
                
            except Exception as e:
                logger.warning(f"[CLASSIFIER PROCESSING] Dynamic detection failed: {e}, using fallback processing")
                # Fallback: try to remove thinking tags anyway
                import re
                clean_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
            
            
            # Enhanced parsing for TYPE|CONFIDENCE format with model-specific handling
            classification_found = False
            
            # Try primary parsing: look for TYPE|CONFIDENCE format
            if '|' in clean_response:
                parts = clean_response.strip().split('|')
                query_type_str = parts[0].strip().upper()
                try:
                    confidence_str = parts[1].strip()
                    # Handle percentage format (e.g., "90%" -> 0.9)
                    if confidence_str.endswith('%'):
                        confidence = float(confidence_str[:-1]) / 100.0
                    else:
                        confidence = float(confidence_str)
                    # Clamp confidence to valid range
                    confidence = max(0.0, min(1.0, confidence))
                    logger.info(f"Parsed confidence: '{parts[1].strip()}' -> {confidence}")
                    classification_found = True
                except (ValueError, IndexError) as e:
                    logger.error(f"LLM provided invalid confidence in response: '{clean_response}', error: {e}")
            
            # Fallback parsing: look for just the classification type on first line
            if not classification_found:
                logger.warning(f"Standard TYPE|CONFIDENCE format not found, trying fallback parsing")
                
                # Try to extract classification from first line or word
                first_line = clean_response.split('\n')[0].strip().upper()
                
                # Check if first line contains a valid classification type
                valid_types = ['TOOL', 'TOOLS', 'LLM', 'WEB_SEARCH', 'MULTI_AGENT', 'SIMPLE_ANSWER']
                for valid_type in valid_types:
                    if valid_type in first_line:
                        query_type_str = valid_type
                        confidence = 0.7  # Default confidence for fallback parsing
                        classification_found = True
                        logger.info(f"Fallback parsing found: {query_type_str} with default confidence {confidence}")
                        break
            
            # If still no classification found, return error
            if not classification_found:
                logger.error(f"Could not parse classification from response: '{clean_response}'")
                return await self._retry_llm_classification(query, "unparseable_response")
            
            # Handle combinations (e.g., "TOOLS+WEB_SEARCH")
            if '+' in query_type_str:
                # For now, take the first type as primary
                primary_type = query_type_str.split('+')[0].strip()
                query_type_str = primary_type
            
            # Map to QueryType enum (handle both old and new type names)
            type_mapping = {
                'TOOL': QueryType.TOOL,
                'TOOLS': QueryType.TOOL,
                'WEB_SEARCH': QueryType.TOOL,  # Map WEB_SEARCH to TOOL for now
                'RAG': QueryType.RAG,
                'KNOWLEDGE': QueryType.RAG,  # Map KNOWLEDGE to RAG (common LLM response)
                'SEARCH': QueryType.RAG,     # Map SEARCH to RAG (common LLM response)
                'LLM': QueryType.LLM,
                'MULTI_AGENT': QueryType.MULTI_AGENT
            }
            
            # Check if response is a collection name (should be mapped to RAG)
            if query_type_str not in type_mapping:
                # If it's not a standard type, check if it matches a RAG collection
                if query_type_str.lower() in [col.lower() for col in self.rag_collections.keys()]:
                    query_type = QueryType.RAG
                    logger.info(f"LLM classified query for RAG collection: {query_type_str}")
                else:
                    # Default to LLM for unknown responses
                    query_type = QueryType.LLM
                    logger.warning(f"Unknown classification type '{query_type_str}', defaulting to LLM")
            else:
                query_type = type_mapping[query_type_str]
            
            # Always accept valid classifications, even with low confidence
            # The confidence value is still useful for routing decisions
            # If LLM classified as TOOL, use LLM to suggest specific tools
            suggested_tools = []
            if query_type == QueryType.TOOL:
                suggested_tools = await self._llm_suggest_tools(query)
            
            result = ClassificationResult(
                query_type=query_type,
                confidence=confidence,
                metadata={
                    "available_tools": self._build_tools_info(),
                    "available_collections": self._build_collections_info(),
                    "llm_response": response_text,
                    "low_confidence": confidence < min_confidence
                },
                suggested_tools=suggested_tools
            )
            
            logger.info(f"Query classified as: {query_type.value}|{confidence:.2f}")
            
            # End classification span with success
            if classification_span and tracer:
                try:
                    tracer.end_span_with_result(classification_span, {
                        "classification_type": query_type.value,
                        "confidence": confidence,
                        "suggested_tools": suggested_tools
                    }, True)
                except Exception as e:
                    logger.warning(f"Failed to end classification span: {e}")
            
            return [result]
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}", exc_info=True)
            
            # End classification span with error
            if classification_span and tracer:
                try:
                    tracer.end_span_with_result(classification_span, {"error": str(e)}, False, str(e))
                except Exception as e:
                    logger.warning(f"Failed to end classification span: {e}")
            
            return await self._retry_llm_classification(query, "exception")
    
    async def _retry_llm_classification(self, query: str, reason: str) -> List[ClassificationResult]:
        """Retry LLM classification with a simpler prompt or return error result"""
        logger.warning(f"LLM classification failed ({reason}), creating error result")
        
        # Return a default tool classification with low confidence to indicate uncertainty
        from app.core.query_classifier_settings_cache import get_query_classifier_settings
        classifier_specific_settings = get_query_classifier_settings()
        min_confidence = float(classifier_specific_settings.get('min_confidence_threshold', 0.1))
        
        result = ClassificationResult(
            query_type=QueryType.TOOL,
            confidence=min_confidence,
            metadata={
                "llm_error": True,
                "error_reason": reason,
                "fallback_classification": True,
                "available_tools": self._build_tools_info(),
                "available_collections": self._build_collections_info()
            },
            suggested_tools=[]
        )
        
        # Log fallback classification at debug level to avoid confusion
        logger.debug(f"Fallback classification: {result.query_type.value}|{result.confidence:.2f} (reason: {reason})")
        return [result]
    
    def _build_tools_info(self) -> str:
        """Build simple tool information in format: tool_name : description"""
        if not self.available_mcp_tools:
            return "No MCP tools are currently available."
        
        tool_list = []
        for tool_name, tool_info in self.available_mcp_tools.items():
            # Extract tool description if available from manifest
            description = "Available for use"
            if isinstance(tool_info, dict):
                manifest = tool_info.get('manifest', {})
                if manifest and 'tools' in manifest:
                    for tool_def in manifest['tools']:
                        if tool_def.get('name') == tool_name:
                            description = tool_def.get('description', description)
                            break
            
            # Enhance descriptions with capability details
            enhanced_description = self._enhance_tool_description(tool_name, description)
            
            # Simple format: tool_name : description
            tool_list.append(f"{tool_name} : {enhanced_description}")
        
        # Sort alphabetically for consistent presentation
        tool_list.sort()
        return "\n".join(tool_list)
    
    def _build_collections_info(self) -> str:
        """Build simple collection information in format: collection_name : description"""
        if not self.rag_collections:
            return "No RAG collections are currently available."
        
        collections_list = []
        for collection_name, collection_info in self.rag_collections.items():
            description = collection_info.get('description', 'No description available')
            # Simple format: collection_name : description
            collections_list.append(f"{collection_name} : {description}")
        
        # Sort alphabetically for consistent presentation
        collections_list.sort()
        return "\n".join(collections_list)
    
    def _enhance_tool_description(self, tool_name: str, base_description: str) -> str:
        """Enhance tool descriptions with specific capabilities for better LLM classification"""
        tool_lower = tool_name.lower()
        
        # Trust the MCP-provided description if it exists and is meaningful
        if base_description and base_description != "Available for use":
            return base_description
        
        # Only use generic categorization as a last resort when no description is provided
        # All hardcoded tool descriptions have been removed to use MCP manifest descriptions instead
        
        # Generic categorization based on common patterns in tool names
        if any(keyword in tool_lower for keyword in ['search', 'find', 'lookup', 'query']):
            return 'Search and retrieval tool. Use for finding information or data.'
        elif any(keyword in tool_lower for keyword in ['create', 'add', 'make', 'generate', 'build']):
            return 'Creation and generation tool. Use for creating new content or resources.'
        elif any(keyword in tool_lower for keyword in ['update', 'modify', 'edit', 'change', 'set']):
            return 'Modification tool. Use for updating or changing existing content.'
        elif any(keyword in tool_lower for keyword in ['delete', 'remove', 'clear', 'clean']):
            return 'Deletion and cleanup tool. Use for removing or cleaning data.'
        elif any(keyword in tool_lower for keyword in ['send', 'notify', 'alert', 'message']):
            return 'Communication tool. Use for sending messages, notifications, or alerts.'
        elif any(keyword in tool_lower for keyword in ['read', 'get', 'fetch', 'retrieve', 'view']):
            return 'Data retrieval tool. Use for accessing and viewing information.'
        elif any(keyword in tool_lower for keyword in ['manage', 'admin', 'control', 'configure']):
            return 'Management tool. Use for administrative and configuration tasks.'
        elif any(keyword in tool_lower for keyword in ['monitor', 'watch', 'track', 'observe']):
            return 'Monitoring tool. Use for tracking and observing system status.'
        elif any(keyword in tool_lower for keyword in ['analyze', 'check', 'test', 'validate']):
            return 'Analysis tool. Use for testing, validation, and analysis tasks.'
        elif any(keyword in tool_lower for keyword in ['convert', 'transform', 'format', 'process']):
            return 'Data processing tool. Use for converting and transforming data.'
        else:
            return f'Utility tool for {tool_name.replace("_", " ").title()} operations. Available for specialized tasks.'
    
    def _detect_hybrid_patterns(self, query_lower: str) -> List[Tuple[str, List[str], float]]:
        """Detect hybrid query patterns"""
        hybrid_detected = []
        
        for pattern, config in self.compiled_patterns.get("hybrid", []):
            if pattern.search(query_lower):
                hybrid_detected.append((
                    config["hybrid_type"],
                    config["primary_types"],
                    config["confidence_threshold"]
                ))
        
        return hybrid_detected
    
    def _create_hybrid_classifications(self, 
                                     results: List[ClassificationResult], 
                                     settings: Dict) -> List[ClassificationResult]:
        """Create hybrid classifications based on component scores"""
        hybrid_results = []
        
        # Only create hybrids if multiple strong signals exist
        strong_threshold = float(settings.get("strong_results_threshold", 0.2))
        strong_results = [r for r in results if r.confidence > strong_threshold]
        
        if len(strong_results) < 2:
            return hybrid_results
        
        # Check for specific hybrid combinations
        type_map = {r.query_type: r for r in strong_results}
        
        # Tool + RAG hybrid
        if QueryType.TOOL in type_map and QueryType.RAG in type_map:
            tool_result = type_map[QueryType.TOOL]
            rag_result = type_map[QueryType.RAG]
            
            hybrid_multiplier = float(settings.get("hybrid_confidence_multiplier", 0.6))
            hybrid_confidence = (tool_result.confidence + rag_result.confidence) * hybrid_multiplier
            
            if hybrid_confidence > float(settings.get("min_confidence_threshold", 0.1)):
                hybrid_results.append(ClassificationResult(
                    query_type=QueryType.TOOL_RAG,
                    confidence=hybrid_confidence,
                    metadata={
                        "component_confidences": {
                            "tool": tool_result.confidence,
                            "rag": rag_result.confidence
                        },
                        "is_hybrid": True
                    },
                    suggested_tools=tool_result.suggested_tools + rag_result.suggested_tools,
                    suggested_agents=tool_result.suggested_agents + rag_result.suggested_agents
                ))
        
        # Tool + LLM hybrid
        if QueryType.TOOL in type_map and QueryType.LLM in type_map:
            tool_result = type_map[QueryType.TOOL]
            llm_result = type_map[QueryType.LLM]
            
            hybrid_multiplier = float(settings.get("hybrid_confidence_multiplier", 0.6))
            hybrid_confidence = (tool_result.confidence + llm_result.confidence) * hybrid_multiplier
            
            if hybrid_confidence > float(settings.get("min_confidence_threshold", 0.1)):
                hybrid_results.append(ClassificationResult(
                    query_type=QueryType.TOOL_LLM,
                    confidence=hybrid_confidence,
                    metadata={
                        "component_confidences": {
                            "tool": tool_result.confidence,
                            "llm": llm_result.confidence
                        },
                        "is_hybrid": True
                    },
                    suggested_tools=tool_result.suggested_tools,
                    suggested_agents=llm_result.suggested_agents
                ))
        
        # RAG + LLM hybrid
        if QueryType.RAG in type_map and QueryType.LLM in type_map:
            rag_result = type_map[QueryType.RAG]
            llm_result = type_map[QueryType.LLM]
            
            hybrid_multiplier = float(settings.get("hybrid_confidence_multiplier", 0.6))
            hybrid_confidence = (rag_result.confidence + llm_result.confidence) * hybrid_multiplier
            
            if hybrid_confidence > float(settings.get("min_confidence_threshold", 0.1)):
                hybrid_results.append(ClassificationResult(
                    query_type=QueryType.RAG_LLM,
                    confidence=hybrid_confidence,
                    metadata={
                        "component_confidences": {
                            "rag": rag_result.confidence,
                            "llm": llm_result.confidence
                        },
                        "is_hybrid": True
                    },
                    suggested_tools=rag_result.suggested_tools,
                    suggested_agents=rag_result.suggested_agents + llm_result.suggested_agents
                ))
        
        # Tool + RAG + LLM hybrid (all three)
        if QueryType.TOOL in type_map and QueryType.RAG in type_map and QueryType.LLM in type_map:
            tool_result = type_map[QueryType.TOOL]
            rag_result = type_map[QueryType.RAG]
            llm_result = type_map[QueryType.LLM]
            
            three_way_multiplier = float(settings.get("three_way_hybrid_multiplier", 0.5))
            hybrid_confidence = (tool_result.confidence + rag_result.confidence + llm_result.confidence) * three_way_multiplier
            
            if hybrid_confidence > float(settings.get("min_confidence_threshold", 0.1)):
                hybrid_results.append(ClassificationResult(
                    query_type=QueryType.TOOL_RAG_LLM,
                    confidence=hybrid_confidence,
                    metadata={
                        "component_confidences": {
                            "tool": tool_result.confidence,
                            "rag": rag_result.confidence,
                            "llm": llm_result.confidence
                        },
                        "is_hybrid": True
                    },
                    suggested_tools=tool_result.suggested_tools + rag_result.suggested_tools,
                    suggested_agents=tool_result.suggested_agents + rag_result.suggested_agents + llm_result.suggested_agents
                ))
        
        return hybrid_results
    
    async def get_routing_recommendation(self, query: str, trace=None) -> Dict[str, any]:
        """
        Get complete routing recommendation for a query with support for hybrid routing
        
        Returns:
            Dict with routing information including multiple classifications
        """
        classifications = await self.classify(query, trace=trace)
        settings = self.config.get("settings", {})
        min_confidence = float(settings.get("min_confidence_threshold", 0.1))
        
        if not classifications:
            # Default fallback to tool search if no classification
            fallback_confidence = float(settings.get("fallback_default_confidence", 0.5))
            primary_classification = ClassificationResult(
                query_type=QueryType.TOOL,
                confidence=fallback_confidence,
                metadata={"fallback_reason": "no_classification", "suggested_action": "web_search"}
            )
        else:
            primary_classification = classifications[0]
            
            # Implement fallback strategy: if confidence too low, fallback to tools
            if primary_classification.confidence < min_confidence:
                logger.info(f"Confidence {primary_classification.confidence:.2f} below threshold {min_confidence}, falling back to tools")
                fallback_tool_confidence = float(settings.get("fallback_tool_confidence", 0.6))
                primary_classification = ClassificationResult(
                    query_type=QueryType.TOOL,
                    confidence=fallback_tool_confidence,
                    metadata={
                        "fallback_reason": "low_confidence", 
                        "original_type": primary_classification.query_type.value,
                        "original_confidence": primary_classification.confidence,
                        "suggested_action": "web_search"
                    }
                )
        
        recommendation = {
            "primary_type": primary_classification.query_type.value,
            "confidence": primary_classification.confidence,
            "is_hybrid": QueryType.is_hybrid(primary_classification.query_type),
            "classifications": [
                {
                    "type": c.query_type.value,
                    "confidence": c.confidence,
                    "metadata": c.metadata,
                    "suggested_tools": c.suggested_tools,
                    "suggested_agents": c.suggested_agents
                }
                for c in classifications
            ],
            "routing": self._build_routing_strategy(primary_classification, classifications),
            "metadata": {
                "query_length": len(query.split()),
                "total_classifications": len(classifications)
            }
        }
        
        return recommendation
    
    def _build_routing_strategy(self, 
                              primary: ClassificationResult, 
                              _all_classifications: List[ClassificationResult]) -> Dict:
        """Build routing strategy based on classifications"""
        strategy = {
            "handlers": [],
            "execution_mode": "sequential",  # or "parallel" for hybrid
            "use_streaming": True,
            "fallback_handler": "llm"
        }
        
        if QueryType.is_hybrid(primary.query_type):
            # Hybrid query - need multiple handlers
            components = QueryType.get_components(primary.query_type)
            strategy["execution_mode"] = "parallel"
            
            for component in components:
                handler = self._get_handler_for_type(component)
                strategy["handlers"].append({
                    "type": component.value,
                    "handler": handler,
                    "weight": primary.metadata.get("component_confidences", {}).get(component.value, 0.5)
                })
        else:
            # Single type query
            strategy["handlers"].append({
                "type": primary.query_type.value,
                "handler": self._get_handler_for_type(primary.query_type),
                "weight": 1.0
            })
        
        # Add suggested tools and agents
        if primary.suggested_tools:
            strategy["suggested_tools"] = primary.suggested_tools
        if primary.suggested_agents:
            strategy["suggested_agents"] = primary.suggested_agents
        
        return strategy
    
    def _get_handler_for_type(self, query_type: QueryType) -> str:
        """Map query type to handler name"""
        handlers = {
            QueryType.RAG: "rag",
            QueryType.TOOL: "tool_handler",
            QueryType.LLM: "llm",
            QueryType.CODE: "code_agent",
            QueryType.MULTI_AGENT: "multi_agent"
        }
        return handlers.get(query_type, "llm")
    
    def reload_config(self):
        """Reload configuration from file"""
        self.config = self._load_config()
        self.compiled_patterns = self._compile_patterns()
        logger.info("Reloaded query patterns configuration")
    
    async def _llm_suggest_tools(self, query: str) -> List[str]:
        """Use LLM to suggest specific tools for a query based on available MCP tools with temporal awareness"""
        if not self.mcp_tool_names:
            return []
        
        # First check for temporal queries and auto-suggest datetime tools
        try:
            from app.core.temporal_context_manager import get_temporal_context_manager
            temporal_manager = get_temporal_context_manager()
            
            is_time_related, matched_keywords, confidence = temporal_manager.detect_time_related_query(query.lower())
            
            if is_time_related and confidence > 0.6:
                # Find datetime tools and prioritize them
                datetime_tools = [t for t in self.mcp_tool_names if 'datetime' in t.lower() or 'time' in t.lower() or t == 'get_datetime']
                if datetime_tools:
                    logger.info(f"Auto-suggesting datetime tools for temporal query: {datetime_tools} (confidence: {confidence:.2f})")
                    return datetime_tools[:1]  # Return the first datetime tool found
                    
        except Exception as e:
            logger.debug(f"Temporal tool suggestion failed, proceeding with LLM suggestion: {e}")
        
        try:
            from app.core.llm_settings_cache import get_llm_settings, get_second_llm_full_config
            from app.core.query_classifier_settings_cache import get_query_classifier_settings
            
            # Use second_llm for tool suggestion instead of query classifier LLM
            main_llm_settings = get_llm_settings()
            second_llm_config = get_second_llm_full_config(main_llm_settings)
            classifier_specific_settings = get_query_classifier_settings()
            
            # Create enhanced prompt with tool descriptions for better selection
            tool_info = self._build_tools_info()
            
            # Debug logging to check what's being fed
            
            # Use configurable tool suggestion prompt
            tool_suggestion_template = classifier_specific_settings.get('tool_suggestion_prompt', 
                'Given the following query and available tools, select the most relevant tool:\n\nQuery: {query}\n\nAvailable MCP Tools:\n{tool_info}\n\nReturn ONLY the exact tool name that best matches the query.')
            
            prompt = tool_suggestion_template.format(
                query=query,
                tool_info=tool_info
            )
            
            # Use second_llm configuration instead of query classifier config
            from app.llm.ollama import OllamaLLM
            from app.llm.base import LLMConfig
            
            llm_config = LLMConfig(
                model_name=second_llm_config.get('model', ''),
                temperature=0.0,  # Use temperature 0 for deterministic tool selection
                max_tokens=int(second_llm_config.get('max_tokens', 4000)),  # Use second_llm max_tokens
                top_p=0.95
            )
            
            # Use model server from settings ONLY - no hardcoded fallbacks
            import os
            model_server = second_llm_config.get('model_server', '').strip()
            
            if not model_server:
                # Fallback to main LLM model_server if second LLM doesn't have one
                from app.core.llm_settings_cache import get_main_llm_full_config
                main_llm_full_config = get_main_llm_full_config()
                model_server = main_llm_full_config.get('model_server', '').strip()
                
            if not model_server:
                # Use environment variable as last resort (not hardcoded)
                model_server = os.environ.get("OLLAMA_BASE_URL", "")
                
            if not model_server:
                logger.error("No model server configured in settings or environment")
                raise ValueError("Model server must be configured in LLM settings")
            
            # Apply Docker environment detection to the settings-based URL
            is_docker = os.path.exists('/root') or os.environ.get('DOCKER_ENVIRONMENT') or os.path.exists('/.dockerenv')
            if 'localhost' in model_server and is_docker:
                model_server = model_server.replace('localhost', 'host.docker.internal')
                logger.info(f"Docker environment detected, converted URL to: {model_server}")
            
            llm = OllamaLLM(llm_config, base_url=model_server)
            
            # Get system prompt from second_llm config
            system_prompt = second_llm_config.get('system_prompt', '').strip()
            
            # Add timeout wrapper for LLM tool suggestion using centralized timeout config
            from app.core.timeout_settings_cache import get_timeout_value
            tool_suggestion_timeout = get_timeout_value("llm_ai", "agent_processing_timeout", 30)
            try:
                response = await asyncio.wait_for(
                    llm.generate(prompt, system_prompt=system_prompt) if system_prompt else llm.generate(prompt),
                    timeout=tool_suggestion_timeout
                )
                response_text = response.text.strip()
            except asyncio.TimeoutError:
                logger.error(f"LLM tool suggestion timed out after {tool_suggestion_timeout} seconds")
                return []
            
            # Clean the response - remove any thinking tags, explanations, etc.
            clean_response = response_text
            
            # Remove thinking tags if present
            if '<think>' in clean_response:
                clean_response = re.sub(r'<think>.*?</think>', '', clean_response, flags=re.DOTALL).strip()
            
            # Remove any extra formatting
            clean_response = re.sub(r'^[^\w]*', '', clean_response)  # Remove leading non-word chars
            clean_response = clean_response.split('\n')[0].strip()  # Take first line only
            clean_response = clean_response.split(' ')[0].strip()   # Take first word only
            
            logger.info(f"LLM tool suggestion for '{query}': {clean_response}")
            
            # Parse LLM response to extract tool names
            suggested_tools = []
            if clean_response and clean_response.lower() != "none":
                # Check if the cleaned response is a valid tool name
                if clean_response in self.mcp_tool_names:
                    suggested_tools.append(clean_response)
                else:
                    # Try to find partial matches for common typos
                    for tool_name in self.mcp_tool_names:
                        if tool_name.lower() in clean_response.lower() or clean_response.lower() in tool_name.lower():
                            suggested_tools.append(tool_name)
                            break
            
            logger.info(f"Parsed suggested tools: {suggested_tools}")
            return suggested_tools
            
        except Exception as e:
            logger.error(f"Failed to get LLM tool suggestions: {e}", exc_info=True)
            return []

    async def detect_explicit_search_intent(self, query: str) -> bool:
        """
        Detect if user has explicit search intent using configuration-based patterns only
        Returns True if user explicitly wants to search (should bypass skip_knowledge_search)
        """
        try:
            # Get explicit search patterns from configuration only
            from app.core.query_classifier_settings_cache import get_query_classifier_settings
            classifier_settings = get_query_classifier_settings()
            
            # Use only patterns from configuration - no fallback defaults
            explicit_patterns = classifier_settings.get("explicit_search_patterns", [])
            
            if not explicit_patterns:
                logger.debug("No explicit search patterns configured - defaulting to False")
                return False
            
            query_lower = query.lower()
            for pattern in explicit_patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    logger.info(f"Explicit search intent detected with pattern: {pattern}")
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Failed to detect explicit search intent: {e}")
            return False
