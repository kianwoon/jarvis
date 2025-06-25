"""
Enhanced Query Classifier with Hybrid Query Support
Supports configurable patterns and multiple classifications with confidence scores
"""
import re
import yaml
import logging
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
        """Load dynamic settings from Redis cache/database"""
        try:
            # Load LLM-based classifier settings from your configured Redis cache
            from app.core.enhanced_query_classifier_cache import get_enhanced_classifier_settings
            dynamic_settings = get_enhanced_classifier_settings()
            
            # Override config settings with your Redis-cached settings
            if 'settings' not in self.config:
                self.config['settings'] = {}
            
            self.config['settings'].update(dynamic_settings)
            logger.info(f"Loaded dynamic query classifier settings: {dynamic_settings}")
            
            # Check if LLM-based classification is configured
            # Require both max_tokens AND system_prompt to enable LLM classification
            if dynamic_settings.get('classifier_max_tokens') and dynamic_settings.get('system_prompt'):
                self.use_llm_classification = True
                logger.info("LLM-based query classification enabled from settings")
            else:
                self.use_llm_classification = False
                if dynamic_settings.get('classifier_max_tokens') and not dynamic_settings.get('system_prompt'):
                    logger.info("Pattern-based query classification enabled (no system prompt configured)")
                else:
                    logger.info("Pattern-based query classification enabled")
                
        except ImportError as e:
            logger.warning(f"Enhanced classifier cache not available yet: {e}")
            self.use_llm_classification = False
        except Exception as e:
            logger.warning(f"Failed to load dynamic settings, using defaults: {e}")
            self.use_llm_classification = False
    
    def _load_mcp_tools(self):
        """Load available MCP tools for better classification"""
        try:
            from app.core.mcp_tools_cache import get_enabled_mcp_tools
            self.available_mcp_tools = get_enabled_mcp_tools()
            self.mcp_tool_names = set(self.available_mcp_tools.keys()) if self.available_mcp_tools else set()
            logger.info(f"Loaded {len(self.mcp_tool_names)} MCP tools: {list(self.mcp_tool_names)}")
            
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
                
                # Extract keywords from collection names and descriptions
                collection_info = {
                    'name': name,
                    'description': description,
                    'type': collection_type,
                    'document_count': stats.get('document_count', 0),
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
            from app.core.llm_settings_cache import get_llm_settings
            llm_settings = get_llm_settings()
            classifier_config = llm_settings.get('query_classifier', {})
            
            # Use your configured settings
            max_tokens = int(classifier_config.get('classifier_max_tokens', 10))
            min_confidence = float(classifier_config.get('min_confidence_threshold', 0.1))
            max_classifications = int(classifier_config.get('max_classifications', 3))  # Reserved for future use
            
            # Get system prompt from configuration
            system_prompt = classifier_config.get('system_prompt')
            
            if system_prompt:
                # Build detailed tool and collection information
                tools_info = self._build_tools_info()
                collections_info = self._build_collections_info()
                
                # Assemble prompt structure as specified
                prompt = f"""{system_prompt}

**Available Tools:**
{tools_info}

**Available RAG Collections:**
{collections_info}

Query: "{query}"

Respond with the classification type and confidence score in this exact format: TYPE|CONFIDENCE
Where TYPE is one of: tool, rag, llm, multi_agent
Where CONFIDENCE is a decimal between 0.0 and 1.0 indicating your certainty

Example: tool|0.85"""
                
                # Debug log the prompt
                logger.info(f"LLM Classifier prompt length: {len(prompt)} chars")
                logger.info(f"LLM Classifier prompt preview:\n{prompt[:200]}...")  # First 200 chars
                
                # Log the last part to see /no_think
                logger.info(f"LLM Classifier prompt end: ...{prompt[-100:]}")  # Last 100 chars
            else:
                # No system prompt configured - cannot classify
                logger.error("No system prompt configured for query classifier")
                return await self._pattern_classify_fallback(query)

            # Use your LLM settings to make the classification call
            from app.llm.ollama import OllamaLLM
            from app.llm.base import LLMConfig
            
            # Use LLM configuration from your settings - no hardcoding
            # Use non_thinking_mode for classifier to avoid thinking tags
            non_thinking_mode = llm_settings.get('non_thinking_mode')
            llm_config = LLMConfig(
                model_name=llm_settings.get('model'),
                temperature=float(non_thinking_mode.get('temperature')),
                max_tokens=max_tokens,
                top_p=float(non_thinking_mode.get('top_p'))
            )
            
            # Use same approach as main system - check env var first, then settings, then default
            import os
            model_server = os.environ.get("OLLAMA_BASE_URL")
            if not model_server:
                # Try settings if env var not set
                model_server = llm_settings.get('model_server', '').strip()
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
            
            response = await llm.generate(prompt)
            response_text = response.text
            
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
            clean_response = response_text
            logger.info(f"[CLASSIFIER DEBUG] Raw response length: {len(response_text)}, has_think_tags: {'<think>' in clean_response and '</think>' in clean_response}")
            if '<think>' in clean_response and '</think>' in clean_response:
                import re
                clean_response = re.sub(r'<think>.*?</think>', '', clean_response, flags=re.DOTALL).strip()
                logger.info(f"[CLASSIFIER DEBUG] After cleaning: '{clean_response}'")
            
            # Parse LLM response - require TYPE|CONFIDENCE format
            if '|' in clean_response:
                parts = clean_response.strip().split('|')
                query_type_str = parts[0].strip().upper()
                try:
                    confidence = float(parts[1].strip())
                    # Clamp confidence to valid range
                    confidence = max(0.0, min(1.0, confidence))
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
                'LLM': QueryType.LLM,
                'MULTI_AGENT': QueryType.MULTI_AGENT
            }
            
            query_type = type_mapping.get(query_type_str, QueryType.LLM)
            
            if confidence >= min_confidence:
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
                        "llm_response": response_text
                    },
                    suggested_tools=suggested_tools
                )
                
                logger.info(f"LLM classification: {query_type.value} (confidence: {confidence:.2f})")
                
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
            
            # Fallback if parsing fails
            logger.error(f"Failed to parse LLM classification response: {response_text}")
            
            # End classification span with error
            if classification_span and tracer:
                try:
                    tracer.end_span_with_result(classification_span, {"error": "parse_failed"}, False, "Failed to parse LLM response")
                except Exception as e:
                    logger.warning(f"Failed to end classification span: {e}")
            
            return await self._retry_llm_classification(query, "parse_failed")
            
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
        from app.core.llm_settings_cache import get_llm_settings
        llm_settings = get_llm_settings()
        classifier_config = llm_settings.get('query_classifier', {})
        min_confidence = float(classifier_config.get('min_confidence_threshold', 0.1))
        
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
        
        logger.info(f"Fallback classification: {result.query_type.value} (confidence: {result.confidence:.2f}) due to {reason}")
        return [result]
    
    def _build_tools_info(self) -> str:
        """Build detailed information about available MCP tools"""
        if not self.available_mcp_tools:
            return "No MCP tools are currently available."
        
        tools_list = []
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
            
            tools_list.append(f"- {tool_name}: {description}")
        
        return "\n".join(tools_list) if tools_list else "No tools available."
    
    def _build_collections_info(self) -> str:
        """Build detailed information about available RAG collections"""
        if not self.rag_collections:
            return "No RAG collections are currently available."
        
        collections_list = []
        for collection_name, collection_info in self.rag_collections.items():
            description = collection_info.get('description', 'No description available')
            doc_count = collection_info.get('document_count', 0)
            collection_type = collection_info.get('type', 'Unknown')
            
            collections_list.append(
                f"- {collection_name}: {description} (Type: {collection_type}, Documents: {doc_count})"
            )
        
        return "\n".join(collections_list) if collections_list else "No collections available."
    
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
            from app.core.llm_settings_cache import get_llm_settings
            llm_settings = get_llm_settings()
            
            # Create simple, direct prompt
            prompt = f"""Query: {query}

Tools: {', '.join(self.mcp_tool_names)}

Output only the most relevant tool name (no explanations):/NO_THINK"""

            # Use same LLM configuration as classification
            from app.llm.ollama import OllamaLLM
            from app.llm.base import LLMConfig
            
            non_thinking_mode = llm_settings.get('non_thinking_mode', {})
            llm_config = LLMConfig(
                model_name=llm_settings.get('model'),
                temperature=float(non_thinking_mode.get('temperature', 0.1)),
                max_tokens=50,  # Short response needed - just tool names
                top_p=float(non_thinking_mode.get('top_p', 0.9))
            )
            
            # Use same model server detection as main classifier
            import os
            model_server = os.environ.get("OLLAMA_BASE_URL")
            if not model_server:
                model_server = llm_settings.get('model_server', '').strip()
                if not model_server:
                    model_server = "http://ollama:11434"
            
            llm = OllamaLLM(llm_config, base_url=model_server)
            response = await llm.generate(prompt)
            response_text = response.text.strip()
            
            # Clean response - remove thinking tags
            clean_response = response_text
            if '<think>' in clean_response and '</think>' in clean_response:
                clean_response = re.sub(r'<think>.*?</think>', '', clean_response, flags=re.DOTALL).strip()
            
            logger.info(f"LLM tool suggestion for '{query}': {clean_response}")
            
            # Parse LLM response to extract tool names
            suggested_tools = []
            if clean_response.lower() != "none":
                lines = clean_response.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    # Remove bullet points, numbers, etc.
                    line = re.sub(r'^[-*•\d+.)\s]+', '', line).strip()
                    # Only include if it's a valid tool name
                    if line in self.mcp_tool_names:
                        suggested_tools.append(line)
            
            logger.info(f"Parsed suggested tools: {suggested_tools}")
            return suggested_tools
            
        except Exception as e:
            logger.error(f"Failed to get LLM tool suggestions: {e}", exc_info=True)
            return []