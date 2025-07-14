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
            logger.info(f"Loaded Query Classifier settings: {dynamic_settings}")
            
            # Check if LLM-based classification is enabled
            # Use new enable_llm_classification flag and require model to be configured
            enable_llm = dynamic_settings.get('enable_llm_classification', False)
            # Try both old and new schema model field names
            llm_model = dynamic_settings.get('llm_model', '').strip() or dynamic_settings.get('model', '').strip()
            llm_system_prompt = dynamic_settings.get('llm_system_prompt', '').strip()
            
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
                    logger.info("LLM-based query classification disabled - no system prompt configured")
                
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
            logger.info(f"[MCP TOOLS DEBUG] Sample tool data: {list(self.available_mcp_tools.items())[:2] if self.available_mcp_tools else 'No tools available'}")
            
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
            logger.info(f"[RAG COLLECTIONS DEBUG] Sample collection data: {list(self.rag_collections.items())[:2] if self.rag_collections else 'No collections available'}")
            
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
    
    def _apply_mcp_tool_patterns(self, query_lower: str, scores: Dict, metadata: Dict):
        """Apply MCP tool-specific patterns for better classification"""
        
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
            
            # Get full query classifier configuration from LLM cache
            llm_settings = get_llm_settings()
            classifier_config = get_query_classifier_full_config(llm_settings)
            
            # Get additional classifier-specific settings from query classifier cache
            classifier_specific_settings = get_query_classifier_settings()
            
            # Use LLM cache config for model parameters
            max_tokens = int(classifier_config.get('max_tokens', 100))  # Use LLM cache max_tokens
            llm_model = classifier_config.get('model', '')
            temperature = float(classifier_config.get('temperature', 0.1))
            
            # Use classifier-specific settings for classification behavior
            min_confidence = float(classifier_specific_settings.get('min_confidence_threshold', 0.1))
            max_classifications = int(classifier_specific_settings.get('max_classifications', 3))
            timeout_seconds = int(classifier_specific_settings.get('llm_timeout_seconds', 5))
            
            # Get system prompt from classifier-specific settings
            system_prompt = classifier_config.get('system_prompt')
            
            if system_prompt:
                # Build detailed tool and collection information
                tools_info = self._build_tools_info()
                collections_info = self._build_collections_info()
                
                # Debug logging to verify what data is being fed to classifier
                logger.info(f"[MAIN CLASSIFIER DEBUG] Tools info length: {len(tools_info)} chars")
                logger.info(f"[MAIN CLASSIFIER DEBUG] Tools info preview: {tools_info[:300]}...")
                logger.info(f"[MAIN CLASSIFIER DEBUG] Collections info length: {len(collections_info)} chars")
                logger.info(f"[MAIN CLASSIFIER DEBUG] Collections info preview: {collections_info[:300]}...")
                logger.info(f"[MAIN CLASSIFIER DEBUG] Available MCP tools count: {len(self.mcp_tool_names)}")
                logger.info(f"[MAIN CLASSIFIER DEBUG] Available RAG collections count: {len(self.rag_collections)}")
                
                # Process template placeholders in system prompt
                processed_prompt = system_prompt.format(
                    rag_collection=collections_info,
                    mcp_tools=tools_info
                )
                
                # Assemble comprehensive prompt with query
                prompt = f"""{processed_prompt}

**Query to classify:** "{query}"

Answer:"""
                
                # Debug log the prompt with structured sections
                logger.info(f"LLM Classifier prompt length: {len(prompt)} chars")
                logger.info(f"LLM Classifier prompt preview:\n{prompt[:300]}...")  # More context
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
            
            # Use same approach as main system - check env var first, then settings, then default
            import os
            model_server = os.environ.get("OLLAMA_BASE_URL")
            if not model_server:
                # Try main LLM settings if env var not set
                model_server = main_llm_settings.get('model_server', '').strip()
                if not model_server:
                    # Use same default as main system
                    model_server = "http://ollama:11434"
            
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
                        prompt=prompt,
                        operation="classification"
                    )
                except Exception as e:
                    logger.warning(f"Failed to create classification generation span: {e}")
            
            # Add timeout wrapper for LLM classification using Query Classifier settings
            classifier_timeout = timeout_seconds + 5  # Add 5 second buffer over configured timeout
            try:
                response = await asyncio.wait_for(
                    llm.generate(prompt),
                    timeout=classifier_timeout
                )
                response_text = response.text
            except asyncio.TimeoutError:
                logger.error(f"LLM classification timed out after {classifier_timeout} seconds")
                return await self._retry_llm_classification(query, "llm_timeout")
            
            # End generation span with result
            if generation_span and tracer:
                try:
                    usage = tracer.estimate_token_usage(prompt, response_text)
                    tracer.end_span_with_result(generation_span, {
                        "raw_response": response_text[:500],
                        "usage": usage
                    }, True)
                except Exception as e:
                    logger.warning(f"Failed to end classification generation span: {e}")
            
            logger.info(f"LLM Classifier raw response: '{response_text}'")
            
            # Clean response - remove thinking tags and extract classification
            import re
            clean_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
            logger.info(f"[CLASSIFIER DEBUG] Raw response length: {len(response_text)}")
            logger.info(f"[CLASSIFIER DEBUG] Response after cleaning: '{clean_response}'")
            
            # Parse LLM response - require TYPE|CONFIDENCE format
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
                except (ValueError, IndexError) as e:
                    logger.error(f"LLM provided invalid confidence in response: '{clean_response}', error: {e}")
                    return await self._retry_llm_classification(query, "invalid_confidence")
            else:
                logger.error(f"LLM did not follow required TYPE|CONFIDENCE format: '{clean_response}', no '|' found")
                return await self._retry_llm_classification(query, "missing_confidence")
            
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
        
        # Provide rich, capability-specific descriptions based on tool names
        enhanced_descriptions = {
            'google_search': 'Search the web for current information, news, and real-time data. Use for "latest", "recent", "current", "new", or any time-sensitive queries.',
            'get_datetime': 'Get current date and time information. Use for queries asking "what time is it", "what\'s the date", "current time/date".',
            'send_email': 'Send emails to recipients. Use when user wants to send, compose, or email someone.',
            'read_email': 'Read and retrieve email messages. Use when user wants to check, read, or view emails.',
            'create_jira_issue': 'Create new JIRA tickets/issues. Use when user wants to create, log, or report bugs/tasks.',
            'search_jira': 'Search for existing JIRA issues and tickets. Use when user wants to find, lookup, or check JIRA items.',
            'calendar_search': 'Search calendar events and appointments. Use when user asks about meetings, appointments, or calendar events.',
            'create_calendar_event': 'Create new calendar appointments. Use when user wants to schedule, book, or create meetings.',
            'file_search': 'Search through files and documents. Use when user wants to find specific files or content within files.',
            'weather': 'Get weather information and forecasts. Use for weather-related queries.',
            'calculator': 'Perform mathematical calculations. Use for math problems, computations, and numerical queries.',
            'note_taking': 'Create and manage notes. Use when user wants to save, create, or manage notes/memos.',
            'translation': 'Translate text between languages. Use for translation requests.',
            'currency_converter': 'Convert between different currencies. Use for currency exchange rate queries.',
            'stock_info': 'Get stock prices and financial market data. Use for stock quotes, market information.',
            'news_feed': 'Get news articles and current events. Use for news-related queries.',
            'task_manager': 'Manage tasks and to-do items. Use when user wants to create, update, or manage tasks.',
            'password_manager': 'Manage passwords and credentials securely. Use for password-related requests.',
            'system_info': 'Get system information and status. Use for system monitoring, performance queries.',
            'url_shortener': 'Shorten long URLs. Use when user wants to create short links.',
            'qr_code': 'Generate QR codes. Use when user wants to create QR codes for text, URLs, etc.',
            'screenshot': 'Take screenshots of the screen. Use when user wants to capture screen content.',
            'text_to_speech': 'Convert text to speech audio. Use for text-to-speech requests.',
            'image_analysis': 'Analyze and describe images. Use when user wants to understand image content.',
            'pdf_reader': 'Read and extract text from PDF files. Use for PDF document queries.',
            'code_formatter': 'Format and beautify code. Use for code formatting requests.',
            'json_validator': 'Validate and format JSON data. Use for JSON-related queries.',
            'base64_encoder': 'Encode/decode base64 data. Use for base64 encoding/decoding.',
            'hash_generator': 'Generate hash values (MD5, SHA, etc.). Use for hash generation requests.',
            'regex_tester': 'Test and validate regular expressions. Use for regex-related queries.',
            'color_picker': 'Work with colors and color codes. Use for color-related queries.',
            'unit_converter': 'Convert between different units of measurement. Use for unit conversion requests.',
            'timezone_converter': 'Convert times between different timezones. Use for timezone-related queries.',
            'ip_lookup': 'Lookup IP address information and geolocation. Use for IP-related queries.',
            'domain_checker': 'Check domain availability and information. Use for domain-related queries.',
            'port_scanner': 'Scan network ports and services. Use for network diagnostic queries.',
            'ssl_checker': 'Check SSL certificate information. Use for SSL/certificate queries.',
            'backup_manager': 'Manage file backups and restore operations. Use for backup-related requests.',
            'log_analyzer': 'Analyze log files and extract insights. Use for log analysis queries.',
            'database_query': 'Query databases and retrieve data. Use for database-related requests.',
            'api_tester': 'Test API endpoints and responses. Use for API testing queries.',
            'ssh_client': 'Connect to remote servers via SSH. Use for remote server access.',
            'ftp_client': 'Transfer files via FTP/SFTP. Use for file transfer requests.',
            'git_manager': 'Manage Git repositories and version control. Use for Git-related operations.',
            'docker_manager': 'Manage Docker containers and images. Use for Docker-related queries.',
            'kubernetes_client': 'Interact with Kubernetes clusters. Use for Kubernetes operations.',
            'cloud_storage': 'Manage cloud storage (AWS S3, Google Drive, etc.). Use for cloud file operations.',
            'monitoring_alerts': 'Set up and manage system alerts. Use for monitoring and alerting.',
            'performance_profiler': 'Profile application performance. Use for performance analysis.',
            'security_scanner': 'Scan for security vulnerabilities. Use for security assessment queries.',
            'compliance_checker': 'Check compliance with standards and policies. Use for compliance queries.',
            'data_anonymizer': 'Anonymize sensitive data. Use for data privacy requests.',
            'report_generator': 'Generate reports and documentation. Use for report creation.',
            'workflow_automation': 'Automate workflows and processes. Use for automation requests.',
            'integration_manager': 'Manage system integrations. Use for integration-related queries.',
            'analytics_dashboard': 'View analytics and metrics. Use for analytics queries.',
            'user_management': 'Manage users and permissions. Use for user administration.',
            'audit_logger': 'Log and track system activities. Use for audit and compliance.',
            'configuration_manager': 'Manage system configurations. Use for config-related requests.',
            'deployment_manager': 'Manage application deployments. Use for deployment operations.',
            'load_balancer': 'Manage load balancing and traffic distribution. Use for load balancing.',
            'cache_manager': 'Manage caching systems. Use for cache-related queries.',
            'message_queue': 'Manage message queues and async processing. Use for messaging.',
            'scheduler': 'Schedule and manage recurring tasks. Use for scheduling requests.',
            'notification_service': 'Send notifications and alerts. Use for notification requests.',
            'search_engine': 'Perform advanced search operations. Use for complex search queries.',
            'recommendation_engine': 'Generate recommendations and suggestions. Use for recommendation queries.',
            'machine_learning': 'Apply ML models and predictions. Use for AI/ML requests.',
            'data_visualization': 'Create charts and visualizations. Use for data visualization.',
            'export_import': 'Export/import data in various formats. Use for data export/import.',
            'template_engine': 'Generate content from templates. Use for template-based generation.',
            'validator': 'Validate data and formats. Use for validation requests.',
            'sanitizer': 'Clean and sanitize data. Use for data cleaning.',
            'transformer': 'Transform data between formats. Use for data transformation.',
            'aggregator': 'Aggregate and summarize data. Use for data aggregation.',
            'comparison_tool': 'Compare files, data, or configurations. Use for comparison requests.',
            'merge_tool': 'Merge files or data sources. Use for merging operations.',
            'diff_viewer': 'View differences between files or data. Use for diff operations.',
            'version_control': 'Track and manage versions. Use for versioning requests.',
            'rollback_manager': 'Rollback changes and deployments. Use for rollback operations.',
            'feature_toggle': 'Manage feature flags and toggles. Use for feature management.',
            'experiment_manager': 'Manage A/B tests and experiments. Use for experimentation.',
            'feedback_collector': 'Collect and manage user feedback. Use for feedback requests.',
            'survey_tool': 'Create and manage surveys. Use for survey-related queries.',
            'poll_creator': 'Create polls and voting systems. Use for polling requests.',
            'form_builder': 'Build and manage forms. Use for form creation.',
            'document_generator': 'Generate documents and reports. Use for document creation.',
            'signature_tool': 'Manage digital signatures. Use for signature requests.',
            'encryption_tool': 'Encrypt and decrypt data. Use for encryption/decryption.',
            'compression_tool': 'Compress and decompress files. Use for compression operations.',
            'archiver': 'Create and extract archives. Use for archiving operations.',
            'thumbnail_generator': 'Generate image thumbnails. Use for thumbnail creation.',
            'watermark_tool': 'Add watermarks to images/documents. Use for watermarking.',
            'ocr_scanner': 'Extract text from images using OCR. Use for text extraction from images.',
            'barcode_scanner': 'Scan and generate barcodes. Use for barcode operations.',
            'voice_recognition': 'Convert speech to text. Use for voice recognition.',
            'language_detector': 'Detect the language of text. Use for language detection.',
            'sentiment_analyzer': 'Analyze sentiment in text. Use for sentiment analysis.',
            'keyword_extractor': 'Extract keywords from text. Use for keyword extraction.',
            'summarizer': 'Summarize long text content. Use for text summarization.',
            'plagiarism_checker': 'Check for plagiarism in text. Use for plagiarism detection.',
            'grammar_checker': 'Check and correct grammar. Use for grammar checking.',
            'spell_checker': 'Check and correct spelling. Use for spell checking.',
            'readability_analyzer': 'Analyze text readability. Use for readability assessment.',
            'word_counter': 'Count words, characters, and paragraphs. Use for text statistics.',
            'text_cleaner': 'Clean and format text. Use for text cleaning.',
            'case_converter': 'Convert text case (upper, lower, title). Use for case conversion.',
            'text_splitter': 'Split text into segments. Use for text segmentation.',
            'text_merger': 'Merge multiple text sources. Use for text merging.',
            'pattern_finder': 'Find patterns in text or data. Use for pattern matching.',
            'anomaly_detector': 'Detect anomalies in data. Use for anomaly detection.',
            'trend_analyzer': 'Analyze trends in data. Use for trend analysis.',
            'forecaster': 'Forecast future values. Use for prediction queries.',
            'clustering_tool': 'Group similar data points. Use for clustering analysis.',
            'classifier': 'Classify data into categories. Use for classification tasks.',
            'recommendation_filter': 'Filter and rank recommendations. Use for filtering.',
            'personalization_engine': 'Personalize content and experiences. Use for personalization.',
            'ab_tester': 'Run A/B tests and experiments. Use for testing queries.',
            'performance_monitor': 'Monitor system performance. Use for performance monitoring.',
            'health_checker': 'Check system health and status. Use for health checks.',
            'uptime_monitor': 'Monitor service uptime. Use for uptime monitoring.',
            'error_tracker': 'Track and manage errors. Use for error tracking.',
            'debug_helper': 'Debug applications and issues. Use for debugging.',
            'profiler': 'Profile code and system performance. Use for profiling.',
            'benchmark_tool': 'Benchmark performance. Use for benchmarking.',
            'load_tester': 'Test system load and capacity. Use for load testing.',
            'stress_tester': 'Stress test systems. Use for stress testing.',
            'security_audit': 'Audit security configurations. Use for security audits.',
            'vulnerability_scanner': 'Scan for vulnerabilities. Use for vulnerability assessment.',
            'penetration_tester': 'Perform penetration testing. Use for pen testing.',
            'firewall_manager': 'Manage firewall rules. Use for firewall configuration.',
            'access_controller': 'Control access and permissions. Use for access management.',
            'identity_manager': 'Manage identities and authentication. Use for identity management.',
            'session_manager': 'Manage user sessions. Use for session management.',
            'token_manager': 'Manage authentication tokens. Use for token operations.',
            'certificate_manager': 'Manage SSL/TLS certificates. Use for certificate management.',
            'key_manager': 'Manage encryption keys. Use for key management.',
            'secret_manager': 'Manage secrets and credentials. Use for secret management.',
            'vault': 'Secure storage for sensitive data. Use for secure storage.',
            'backup_scheduler': 'Schedule automated backups. Use for backup scheduling.',
            'disaster_recovery': 'Manage disaster recovery procedures. Use for DR operations.',
            'failover_manager': 'Manage system failover. Use for failover operations.',
            'cluster_manager': 'Manage server clusters. Use for cluster management.',
            'auto_scaler': 'Automatically scale resources. Use for auto-scaling.',
            'resource_optimizer': 'Optimize resource usage. Use for resource optimization.',
            'cost_analyzer': 'Analyze and optimize costs. Use for cost analysis.',
            'usage_tracker': 'Track resource usage. Use for usage monitoring.',
            'billing_manager': 'Manage billing and invoicing. Use for billing operations.',
            'subscription_manager': 'Manage subscriptions. Use for subscription management.',
            'license_manager': 'Manage software licenses. Use for license tracking.',
            'compliance_monitor': 'Monitor compliance status. Use for compliance monitoring.',
            'policy_enforcer': 'Enforce organizational policies. Use for policy enforcement.',
            'governance_tool': 'Manage IT governance. Use for governance operations.',
            'risk_assessor': 'Assess and manage risks. Use for risk assessment.',
            'incident_manager': 'Manage security incidents. Use for incident response.',
            'change_manager': 'Manage system changes. Use for change management.',
            'release_manager': 'Manage software releases. Use for release management.',
            'environment_manager': 'Manage deployment environments. Use for environment management.',
            'pipeline_manager': 'Manage CI/CD pipelines. Use for pipeline operations.',
            'artifact_manager': 'Manage build artifacts. Use for artifact management.',
            'dependency_manager': 'Manage software dependencies. Use for dependency tracking.',
            'package_manager': 'Manage software packages. Use for package management.',
            'update_manager': 'Manage system updates. Use for update operations.',
            'patch_manager': 'Manage security patches. Use for patch management.',
            'inventory_manager': 'Track IT inventory. Use for inventory management.',
            'asset_tracker': 'Track organizational assets. Use for asset tracking.',
            'lifecycle_manager': 'Manage asset lifecycles. Use for lifecycle management.',
            'maintenance_scheduler': 'Schedule system maintenance. Use for maintenance planning.',
            'service_desk': 'Manage service requests. Use for service desk operations.',
            'helpdesk': 'Provide technical support. Use for helpdesk queries.',
            'knowledge_base': 'Manage knowledge articles. Use for knowledge management.',
            'documentation_tool': 'Create and manage documentation. Use for documentation.',
            'training_manager': 'Manage training programs. Use for training operations.',
            'certification_tracker': 'Track certifications. Use for certification management.',
            'skill_assessor': 'Assess technical skills. Use for skill assessment.',
            'performance_evaluator': 'Evaluate performance. Use for performance evaluation.',
            'goal_tracker': 'Track goals and objectives. Use for goal management.',
            'project_manager': 'Manage projects. Use for project management.',
            'task_scheduler': 'Schedule and assign tasks. Use for task scheduling.',
            'time_tracker': 'Track time and attendance. Use for time tracking.',
            'resource_planner': 'Plan resource allocation. Use for resource planning.',
            'capacity_planner': 'Plan system capacity. Use for capacity planning.',
            'budget_manager': 'Manage budgets and expenses. Use for budget management.',
            'procurement_tool': 'Manage procurement processes. Use for procurement.',
            'vendor_manager': 'Manage vendor relationships. Use for vendor management.',
            'contract_manager': 'Manage contracts and agreements. Use for contract management.',
            'approval_workflow': 'Manage approval processes. Use for approval workflows.',
            'escalation_manager': 'Manage escalation procedures. Use for escalations.',
            'notification_router': 'Route notifications intelligently. Use for notification routing.',
            'alert_manager': 'Manage system alerts. Use for alert management.',
            'dashboard_builder': 'Build custom dashboards. Use for dashboard creation.',
            'widget_manager': 'Manage dashboard widgets. Use for widget operations.',
            'theme_manager': 'Manage UI themes. Use for theme customization.',
            'layout_manager': 'Manage UI layouts. Use for layout customization.',
            'menu_builder': 'Build custom menus. Use for menu creation.',
            'form_validator': 'Validate form inputs. Use for form validation.',
            'data_mapper': 'Map data between systems. Use for data mapping.',
            'field_mapper': 'Map fields between formats. Use for field mapping.',
            'schema_validator': 'Validate data schemas. Use for schema validation.',
            'format_converter': 'Convert between data formats. Use for format conversion.',
            'protocol_handler': 'Handle different protocols. Use for protocol operations.',
            'connector': 'Connect to external systems. Use for system integration.',
            'adapter': 'Adapt between different interfaces. Use for system adaptation.',
            'bridge': 'Bridge different systems. Use for system bridging.',
            'proxy': 'Proxy requests to other systems. Use for proxy operations.',
            'gateway': 'Gateway for API access. Use for API gateway operations.',
            'middleware': 'Process requests between systems. Use for middleware operations.',
            'interceptor': 'Intercept and process requests. Use for request interception.',
            'filter': 'Filter data and requests. Use for filtering operations.',
            'transformer_pipeline': 'Transform data through pipelines. Use for data transformation.',
            'processor': 'Process data and requests. Use for data processing.',
            'handler': 'Handle specific operations. Use for operation handling.',
            'resolver': 'Resolve dependencies and references. Use for resolution operations.',
            'locator': 'Locate resources and services. Use for resource location.',
            'registry': 'Register and discover services. Use for service registry.',
            'catalog': 'Catalog resources and metadata. Use for resource cataloging.',
            'indexer': 'Index data for searching. Use for data indexing.',
            'crawler': 'Crawl and extract data. Use for data crawling.',
            'scraper': 'Scrape web content. Use for web scraping.',
            'extractor': 'Extract data from sources. Use for data extraction.',
            'parser': 'Parse structured data. Use for data parsing.',
            'interpreter': 'Interpret commands and scripts. Use for interpretation.',
            'compiler': 'Compile code and scripts. Use for compilation.',
            'transpiler': 'Transpile between languages. Use for transpilation.',
            'minifier': 'Minify code and assets. Use for minification.',
            'obfuscator': 'Obfuscate code. Use for code obfuscation.',
            'optimizer': 'Optimize code and resources. Use for optimization.',
            'bundler': 'Bundle assets and dependencies. Use for bundling.',
            'packager': 'Package applications. Use for packaging.',
            'installer': 'Install software and packages. Use for installation.',
            'uninstaller': 'Uninstall software. Use for uninstallation.',
            'updater': 'Update software and packages. Use for updates.',
            'patcher': 'Apply patches and fixes. Use for patching.',
            'rollback': 'Rollback to previous versions. Use for rollback operations.',
            'migrator': 'Migrate data and systems. Use for migration.',
            'importer': 'Import data from external sources. Use for data import.',
            'exporter': 'Export data to external formats. Use for data export.',
            'synchronizer': 'Synchronize data between systems. Use for synchronization.',
            'replicator': 'Replicate data and configurations. Use for replication.',
            'distributor': 'Distribute content and updates. Use for distribution.',
            'publisher': 'Publish content and releases. Use for publishing.',
            'subscriber': 'Subscribe to events and updates. Use for subscription.',
            'broadcaster': 'Broadcast messages and events. Use for broadcasting.',
            'multicaster': 'Multicast to multiple recipients. Use for multicasting.',
            'router': 'Route messages and requests. Use for routing.',
            'balancer': 'Balance load across resources. Use for load balancing.',
            'scheduler_queue': 'Queue and schedule tasks. Use for task queuing.',
            'worker': 'Process background tasks. Use for background processing.',
            'daemon': 'Run background services. Use for daemon operations.',
            'service': 'Provide specific services. Use for service operations.',
            'agent': 'Autonomous task execution. Use for agent-based operations.',
            'bot': 'Automated interactions. Use for bot operations.',
            'assistant': 'AI-powered assistance. Use for AI assistance.',
            'helper': 'General purpose utilities. Use for utility operations.',
            'utility': 'Utility functions. Use for utility operations.',
            'tool': 'General purpose tools. Use for various operations.',
        }
        
        # Look for exact matches first, then partial matches
        if tool_lower in enhanced_descriptions:
            return enhanced_descriptions[tool_lower]
        
        # Look for partial matches in tool name
        for pattern, description in enhanced_descriptions.items():
            if pattern in tool_lower:
                return description
        
        # If no match found, provide a generic but informative description
        if base_description and base_description != "Available for use":
            return base_description
        
        # Last resort: categorize by common patterns
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
        """Use LLM to suggest specific tools for a query based on available MCP tools"""
        if not self.mcp_tool_names:
            return []
        
        try:
            from app.core.llm_settings_cache import get_llm_settings, get_query_classifier_full_config
            from app.core.query_classifier_settings_cache import get_query_classifier_settings
            
            # Get full query classifier configuration from LLM cache
            main_llm_settings = get_llm_settings()
            classifier_config = get_query_classifier_full_config(main_llm_settings)
            classifier_specific_settings = get_query_classifier_settings()
            
            # Create enhanced prompt with tool descriptions for better selection
            tool_info = self._build_tools_info()
            
            # Debug logging to check what's being fed
            logger.info(f"[TOOL SUGGESTION DEBUG] Tool info length: {len(tool_info)} chars")
            logger.info(f"[TOOL SUGGESTION DEBUG] Tool info preview: {tool_info[:200]}...")
            logger.info(f"[TOOL SUGGESTION DEBUG] Available tool names: {list(self.mcp_tool_names)[:5]}...")
            
            # Use configurable tool suggestion prompt
            tool_suggestion_template = classifier_specific_settings.get('tool_suggestion_prompt', 
                'Given the following query and available tools, select the most relevant tool:\n\nQuery: {query}\n\nAvailable MCP Tools:\n{tool_info}\n\nReturn ONLY the exact tool name that best matches the query.')
            
            prompt = tool_suggestion_template.format(
                query=query,
                tool_info=tool_info
            )
            
            # Use same LLM configuration as Query Classifier
            from app.llm.ollama import OllamaLLM
            from app.llm.base import LLMConfig
            
            llm_config = LLMConfig(
                model_name=classifier_config.get('model', ''),
                temperature=0.0,  # Use temperature 0 for deterministic tool selection
                max_tokens=int(classifier_config.get('max_tokens', 100)),  # Use LLM cache max_tokens
                top_p=0.95
            )
            
            # Use same model server detection as main classifier
            import os
            model_server = os.environ.get("OLLAMA_BASE_URL")
            if not model_server:
                model_server = main_llm_settings.get('model_server', '').strip()
                if not model_server:
                    model_server = "http://ollama:11434"
            
            llm = OllamaLLM(llm_config, base_url=model_server)
            
            # Add timeout wrapper for LLM tool suggestion
            try:
                response = await asyncio.wait_for(
                    llm.generate(prompt),
                    timeout=10.0  # Shorter timeout for tool suggestions
                )
                response_text = response.text.strip()
            except asyncio.TimeoutError:
                logger.error("LLM tool suggestion timed out after 10 seconds")
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
