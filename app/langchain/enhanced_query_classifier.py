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
        self.compiled_patterns = self._compile_patterns()
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded query patterns config from {self.config_path}")
                return config
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            # Return default config if file not found
            return self._get_default_config()
    
    def _load_dynamic_settings(self):
        """Load dynamic settings from cache/database"""
        try:
            # Lazy import to avoid circular dependencies during startup
            from app.core.query_classifier_settings_cache import get_query_classifier_settings
            dynamic_settings = get_query_classifier_settings()
            
            # Override config settings with dynamic settings
            if 'settings' not in self.config:
                self.config['settings'] = {}
            
            self.config['settings'].update(dynamic_settings)
            logger.info(f"Loaded dynamic query classifier settings: {dynamic_settings}")
        except ImportError as e:
            logger.warning(f"Query classifier settings cache not available yet: {e}")
            # Use defaults from config
        except Exception as e:
            logger.warning(f"Failed to load dynamic settings, using defaults: {e}")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration if file not found"""
        return {
            "tool_patterns": {},
            "rag_patterns": {},
            "code_patterns": {},
            "multi_agent_patterns": {},
            "direct_llm_patterns": {},
            "hybrid_indicators": {},
            "settings": {
                "min_confidence_threshold": 0.1,
                "max_classifications": 3,
                "enable_hybrid_detection": True,
                "confidence_decay_factor": 0.8,
                "pattern_combination_bonus": 0.15
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
    
    def classify(self, query: str) -> List[ClassificationResult]:
        """
        Classify a query and return multiple classifications with confidence scores
        
        Returns:
            List of ClassificationResult objects sorted by confidence
        """
        query_lower = query.lower()
        settings = self.config.get("settings", {})
        
        # Initialize scores for each query type
        scores = {qt: 0.0 for qt in QueryType}
        metadata = {qt: {
            "matched_patterns": [],
            "suggested_tools": set(),
            "suggested_agents": set(),
            "pattern_groups": set()
        } for qt in QueryType}
        
        # Check for hybrid patterns first if enabled
        if settings.get("enable_hybrid_detection", True):
            hybrid_detected = self._detect_hybrid_patterns(query_lower)
            if hybrid_detected:
                # Boost component types for detected hybrid patterns
                for hybrid_type, components, confidence in hybrid_detected:
                    for component in components:
                        component_type = QueryType[component]
                        scores[component_type] += confidence * 0.5
                        metadata[component_type]["hybrid_indicator"] = True
        
        # Process regular patterns
        for query_type_str, patterns in self.compiled_patterns.items():
            if query_type_str == "hybrid":
                continue
                
            query_type = QueryType(query_type_str)
            
            for pattern, config in patterns:
                if pattern.search(query_lower):
                    # Apply confidence boost
                    scores[query_type] += config["confidence_boost"]
                    
                    # Track metadata
                    metadata[query_type]["matched_patterns"].append((config["group"], pattern.pattern))
                    metadata[query_type]["pattern_groups"].add(config["group"])
                    
                    # Add suggested tools/agents
                    if config.get("suggested_tools"):
                        metadata[query_type]["suggested_tools"].update(config["suggested_tools"])
                    if config.get("suggested_agents"):
                        metadata[query_type]["suggested_agents"].update(config["suggested_agents"])
        
        # Apply pattern combination bonus
        combination_bonus = settings.get("pattern_combination_bonus", 0.15)
        for query_type, meta in metadata.items():
            if len(meta["pattern_groups"]) > 1:
                scores[query_type] += combination_bonus * (len(meta["pattern_groups"]) - 1)
        
        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            for query_type in scores:
                scores[query_type] /= total_score
        
        # Build classification results
        results = []
        for query_type, score in scores.items():
            if score >= settings.get("min_confidence_threshold", 0.1):
                result = ClassificationResult(
                    query_type=query_type,
                    confidence=score,
                    metadata={
                        "query_length": len(query.split()),
                        "has_question_words": any(word in query_lower.split() for word in ['what', 'who', 'when', 'where', 'why', 'how']),
                        "pattern_groups": list(metadata[query_type]["pattern_groups"]),
                        "is_hybrid_component": metadata[query_type].get("hybrid_indicator", False)
                    },
                    suggested_tools=list(metadata[query_type]["suggested_tools"]),
                    suggested_agents=list(metadata[query_type]["suggested_agents"]),
                    matched_patterns=metadata[query_type]["matched_patterns"]
                )
                results.append(result)
        
        # Detect and add hybrid types based on component scores
        hybrid_results = self._create_hybrid_classifications(results, settings)
        results.extend(hybrid_results)
        
        # Sort by confidence and return top N
        results.sort(key=lambda x: x.confidence, reverse=True)
        max_classifications = settings.get("max_classifications", 3)
        
        # Log classification results
        logger.info(f"Query classifications for '{query[:50]}...':")
        for i, result in enumerate(results[:max_classifications]):
            logger.info(f"  {i+1}. {result.query_type.value} (confidence: {result.confidence:.2f})")
        
        return results[:max_classifications]
    
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
        strong_results = [r for r in results if r.confidence > 0.2]
        
        if len(strong_results) < 2:
            return hybrid_results
        
        # Check for specific hybrid combinations
        type_map = {r.query_type: r for r in strong_results}
        
        # Tool + RAG hybrid
        if QueryType.TOOL in type_map and QueryType.RAG in type_map:
            tool_result = type_map[QueryType.TOOL]
            rag_result = type_map[QueryType.RAG]
            
            hybrid_confidence = (tool_result.confidence + rag_result.confidence) * 0.6
            
            if hybrid_confidence > settings.get("min_confidence_threshold", 0.1):
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
            
            hybrid_confidence = (tool_result.confidence + llm_result.confidence) * 0.6
            
            if hybrid_confidence > settings.get("min_confidence_threshold", 0.1):
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
            
            hybrid_confidence = (rag_result.confidence + llm_result.confidence) * 0.6
            
            if hybrid_confidence > settings.get("min_confidence_threshold", 0.1):
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
            
            hybrid_confidence = (tool_result.confidence + rag_result.confidence + llm_result.confidence) * 0.5
            
            if hybrid_confidence > settings.get("min_confidence_threshold", 0.1):
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
    
    def get_routing_recommendation(self, query: str) -> Dict[str, any]:
        """
        Get complete routing recommendation for a query with support for hybrid routing
        
        Returns:
            Dict with routing information including multiple classifications
        """
        classifications = self.classify(query)
        
        if not classifications:
            # Default to LLM if no strong classification
            primary_classification = ClassificationResult(
                query_type=QueryType.LLM,
                confidence=0.5,
                metadata={"default": True}
            )
        else:
            primary_classification = classifications[0]
        
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
                              all_classifications: List[ClassificationResult]) -> Dict:
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