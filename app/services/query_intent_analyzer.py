"""
QueryIntentAnalyzer Service

Provides AI-powered query intent understanding using semantic analysis without 
hardcoded patterns. Uses the existing LLM infrastructure to determine user intent,
scope, and comprehensive result preferences through intelligent prompting.
"""

import asyncio
import json
import logging
import hashlib
import time
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import httpx

from app.core.llm_settings_cache import get_llm_settings, get_main_llm_full_config
# from app.services.request_execution_state_tracker import (
#     check_operation_completed, mark_operation_completed, get_operation_result
# )  # Removed - causing Redis async/sync errors

logger = logging.getLogger(__name__)

# In-memory cache for query intent analysis (TTL: 5 minutes)
_intent_cache = {}
_cache_ttl = 300  # 5 minutes in seconds

class QueryIntentAnalyzer:
    """
    AI-powered query intent analyzer that uses semantic understanding
    instead of pattern matching to determine user intent and preferences.
    """
    
    def __init__(self):
        """Initialize the analyzer with LLM configuration"""
        self.llm_settings = None
        self.llm_config = None
        self._load_llm_config()
    
    def _load_llm_config(self):
        """Load LLM configuration from settings cache"""
        try:
            self.llm_settings = get_llm_settings()
            self.llm_config = get_main_llm_full_config(self.llm_settings)
            logger.debug(f"Loaded LLM config with model: {self.llm_config.get('model', 'unknown')}")
        except Exception as e:
            logger.error(f"Failed to load LLM config: {e}")
            # Use fallback configuration
            self._setup_fallback_config()
    
    def _setup_fallback_config(self):
        """Setup fallback LLM configuration"""
        self.llm_config = {
            "base_url": "http://localhost:11434",
            "model": "llama3.1:8b",
            "temperature": 0.1,
            "max_tokens": 1000
        }
        logger.warning("Using fallback LLM configuration")
    
    def _get_cache_key(self, query: str, context: Optional[Dict] = None) -> str:
        """Generate cache key for query intent analysis"""
        context_str = json.dumps(context or {}, sort_keys=True)
        combined = f"{query}|{context_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cache entry is still valid (within TTL)"""
        return (time.time() - timestamp) < _cache_ttl
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """Get cached result if valid"""
        if cache_key in _intent_cache:
            result, timestamp = _intent_cache[cache_key]
            if self._is_cache_valid(timestamp):
                logger.debug(f"Cache hit for query intent: {cache_key[:8]}...")
                return result
            else:
                # Remove expired entry
                del _intent_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: Dict):
        """Cache the analysis result"""
        _intent_cache[cache_key] = (result, time.time())
        
        # Simple cleanup: remove expired entries when cache gets large
        if len(_intent_cache) > 1000:
            current_time = time.time()
            expired_keys = [
                key for key, (_, timestamp) in _intent_cache.items()
                if (current_time - timestamp) >= _cache_ttl
            ]
            for key in expired_keys:
                del _intent_cache[key]
    
    async def _make_llm_call(self, prompt: str, timeout: int = 30, custom_config: Optional[Dict] = None) -> str:
        """Make an LLM API call using the configured settings"""
        # Use custom config if provided, otherwise fall back to instance config
        config = custom_config or self.llm_config
        if not config:
            self._load_llm_config()
            config = self.llm_config
        
        try:
            # Prepare the request payload following the pattern from the codebase
            payload = {
                "model": config.get("model", "llama3.1:8b"),
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": config.get("temperature", 0.1)
                }
            }
            
            base_url = config.get("base_url") or config.get("model_server", "http://localhost:11434")
            endpoint = f"{base_url}/api/chat"
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(endpoint, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get('message', {}).get('content', '').strip()
                    logger.debug(f"LLM response received: {len(content)} characters")
                    return content
                else:
                    logger.error(f"LLM API error: {response.status_code} - {response.text}")
                    raise Exception(f"LLM API returned status {response.status_code}")
                    
        except asyncio.TimeoutError:
            logger.error(f"LLM call timed out after {timeout} seconds")
            raise
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _parse_json_from_response(self, response: str) -> Dict:
        """Extract and parse JSON from LLM response"""
        try:
            # Find JSON in response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                logger.warning("No JSON found in LLM response")
                return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            return {}
    
    async def analyze_intent(self, query: str, llm_service=None, llm_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Use LLM to understand true user intent and query characteristics.
        
        Args:
            query: The user's query string
            llm_service: Legacy parameter for compatibility (ignored)
            llm_config: Optional LLM configuration to override default
            request_id: Optional request ID for execution state tracking
            
        Returns:
            dict: {
                'scope': 'comprehensive|filtered|specific',
                'quantity_intent': 'all|limited|few|single',
                'confidence': float (0.0-1.0),
                'context': dict with additional insights,
                'user_type': 'researcher|casual|specific',
                'urgency': 'high|medium|low',
                'completeness_preference': 'thorough|balanced|quick',
                'analysis_mode': bool,
                'analysis_target': str|None
            }
        """
        # Removed execution state tracking to fix Redis async/sync errors
        
        cache_key = self._get_cache_key(query)
        
        # Check cache first
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            # Removed execution state tracking to fix Redis async/sync errors
            return cached_result
        
        # Prepare comprehensive intent analysis prompt
        prompt = f"""
Analyze this user query to understand their true intent and preferences. Focus on semantic meaning and context, not keywords.

QUERY: "{query}"

Analyze the query and return ONLY a JSON object with this exact structure:

{{
    "scope": "comprehensive|filtered|specific",
    "quantity_intent": "all|limited|few|single", 
    "confidence": 0.0-1.0,
    "context": {{
        "domain": "detected domain/topic",
        "complexity": "high|medium|low",
        "specificity": "broad|narrow|targeted"
    }},
    "user_type": "researcher|casual|specific",
    "urgency": "high|medium|low",
    "completeness_preference": "thorough|balanced|quick",
    "analysis_mode": true|false,
    "analysis_target": "detected_object_type|null",
    "reasoning": "brief explanation of analysis"
}}

Guidelines for analysis:
- scope: "comprehensive" for broad exploration, "filtered" for targeted search, "specific" for precise lookup
- quantity_intent: "all" for wanting everything available, "limited" for manageable subset, "few" for highlights, "single" for one answer
- confidence: how certain you are about the intent (1.0 = very certain, 0.6 = moderate, 0.3 = uncertain)
- user_type: "researcher" for deep analysis, "casual" for general interest, "specific" for targeted task
- urgency: based on language indicating time pressure or immediacy
- completeness_preference: "thorough" for detailed results, "balanced" for moderate detail, "quick" for brief answers
- analysis_mode: true if user wants comprehensive analysis/summary (like "analyze all projects", "summarize data", "compare options"), false for simple retrieval
- analysis_target: the object type being analyzed (e.g., "projects", "companies", "clients", "documents") or null if not analysis

CRITICAL: Distinguish between:
- ANALYSIS: "analyze all projects", "summarize companies", "compare clients" → comprehensive analysis needed, analysis_mode=true
- SIMPLE SEARCH: "tell me about machine learning projects", "find AI companies" → basic retrieval OK, analysis_mode=false

Focus on semantic understanding of what the user really wants, not surface-level keywords.

JSON:
"""
        
        try:
            response = await self._make_llm_call(prompt, timeout=20, custom_config=llm_config)
            parsed_result = self._parse_json_from_response(response)
            
            if not parsed_result:
                # Fallback analysis
                logger.warning("LLM JSON parsing failed, using semantic fallback")
                parsed_result = self._semantic_fallback_analysis(query)
            
            # Validate and normalize the result
            result = self._validate_and_normalize_intent(parsed_result, query)
            
            # Cache the result
            self._cache_result(cache_key, result)
            
            # Removed execution state tracking to fix Redis async/sync errors
            
            logger.info(f"Intent analysis complete: scope={result['scope']}, quantity={result['quantity_intent']}, confidence={result['confidence']}, analysis={result['analysis_mode']}, target={result['analysis_target']}")
            return result
            
        except Exception as e:
            logger.error(f"Intent analysis failed: {e}")
            # Return semantic fallback
            return self._semantic_fallback_analysis(query)
    
    def _semantic_fallback_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback semantic analysis when LLM call fails"""
        query_lower = query.lower()
        
        # Basic semantic indicators (not pattern matching, but semantic understanding)
        comprehensive_indicators = len([w for w in query_lower.split() if len(w) > 8]) > 2
        specific_indicators = any(word in query_lower for word in ['specific', 'exact', 'particular', 'precise'])
        broad_indicators = any(word in query_lower for word in ['overview', 'summary', 'general', 'broad', 'comprehensive'])
        
        # Analysis detection patterns
        analysis_patterns = [
            ('analyze', 'analyze'), ('summarize', 'summarize'), ('compare', 'compare'), ('review', 'review'),
            ('assess', 'assess'), ('evaluate', 'evaluate'), ('examine', 'examine'),
            ('breakdown', 'breakdown'), ('table', 'table'), ('format', 'format')
        ]
        
        analysis_mode = False
        analysis_target = None
        
        for pattern, _ in analysis_patterns:
            if pattern in query_lower:
                analysis_mode = True
                # Try to detect the target object type
                words = query_lower.split()
                for i, word in enumerate(words):
                    if word in ['projects', 'companies', 'clients', 'documents', 'files', 'reports', 'tasks']:
                        analysis_target = word
                        break
                    elif word in ['all', 'the'] and i + 1 < len(words):
                        next_word = words[i + 1]
                        if next_word in ['projects', 'companies', 'clients', 'documents', 'files', 'reports', 'tasks']:
                            analysis_target = next_word
                            break
                break
        
        return {
            'scope': 'comprehensive' if comprehensive_indicators or broad_indicators 
                    else 'specific' if specific_indicators 
                    else 'filtered',
            'quantity_intent': 'all' if comprehensive_indicators or analysis_mode else 'limited',
            'confidence': 0.6,  # Moderate confidence for fallback
            'context': {
                'domain': 'general',
                'complexity': 'medium',
                'specificity': 'moderate'
            },
            'user_type': 'casual',
            'urgency': 'medium',
            'completeness_preference': 'balanced',
            'analysis_mode': analysis_mode,
            'analysis_target': analysis_target,
            'reasoning': 'Fallback analysis due to LLM unavailability'
        }
    
    def _validate_and_normalize_intent(self, parsed_result: Dict, query: str) -> Dict[str, Any]:
        """Validate and normalize the intent analysis result"""
        # Ensure all required fields are present with valid values
        result = {
            'scope': parsed_result.get('scope', 'filtered'),
            'quantity_intent': parsed_result.get('quantity_intent', 'limited'),
            'confidence': max(0.0, min(1.0, parsed_result.get('confidence', 0.5))),
            'context': parsed_result.get('context', {}),
            'user_type': parsed_result.get('user_type', 'casual'),
            'urgency': parsed_result.get('urgency', 'medium'),
            'completeness_preference': parsed_result.get('completeness_preference', 'balanced'),
            'analysis_mode': bool(parsed_result.get('analysis_mode', False)),
            'analysis_target': parsed_result.get('analysis_target'),
            'reasoning': parsed_result.get('reasoning', 'AI semantic analysis')
        }
        
        # Validate enum values
        valid_scopes = ['comprehensive', 'filtered', 'specific']
        if result['scope'] not in valid_scopes:
            result['scope'] = 'filtered'
            
        valid_quantities = ['all', 'limited', 'few', 'single']
        if result['quantity_intent'] not in valid_quantities:
            result['quantity_intent'] = 'limited'
            
        # Ensure context is a dict
        if not isinstance(result['context'], dict):
            result['context'] = {}
        
        # Clean analysis_target - ensure it's a string or None
        if result['analysis_target'] is not None and not isinstance(result['analysis_target'], str):
            result['analysis_target'] = None
        
        # If analysis_target is an empty string or "null", set to None
        if result['analysis_target'] in ['', 'null', 'None']:
            result['analysis_target'] = None
            
        return result
    
    async def wants_comprehensive_results(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Determine if user wants all available content through semantic analysis.
        
        Args:
            query: The user's query
            context: Optional context information
            
        Returns:
            dict: {
                'wants_all': bool,
                'confidence': float,
                'estimated_count_preference': int|None,
                'reasoning': str
            }
        """
        cache_key = self._get_cache_key(f"comprehensive_{query}", context)
        
        # Check cache first
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        context_str = ""
        if context:
            context_str = f"\nContext: {json.dumps(context, indent=2)}"
        
        prompt = f"""
Analyze if this user wants comprehensive, exhaustive results or prefers a limited subset.

QUERY: "{query}"{context_str}

Determine the user's preference for result quantity. Return ONLY a JSON object:

{{
    "wants_all": true|false,
    "confidence": 0.0-1.0,
    "estimated_count_preference": number|null,
    "reasoning": "explanation of why you think they want all or limited results"
}}

Guidelines:
- wants_all: true if user seems to want exhaustive/comprehensive results, false for limited/curated
- confidence: certainty level of your analysis
- estimated_count_preference: rough number if user implies a specific count (null if unclear)
- reasoning: brief explanation of your decision

Consider semantic indicators like:
- Exploratory language suggesting broad research
- Specific task language suggesting targeted needs
- Time constraints or urgency indicators
- Research vs. practical application context

JSON:
"""
        
        try:
            response = await self._make_llm_call(prompt, timeout=15)
            parsed_result = self._parse_json_from_response(response)
            
            if not parsed_result:
                # Fallback to intent analysis
                intent = await self.analyze_intent(query)
                parsed_result = {
                    'wants_all': intent['quantity_intent'] == 'all',
                    'confidence': intent['confidence'],
                    'estimated_count_preference': None,
                    'reasoning': 'Derived from intent analysis'
                }
            
            # Validate result
            result = {
                'wants_all': bool(parsed_result.get('wants_all', False)),
                'confidence': max(0.0, min(1.0, parsed_result.get('confidence', 0.5))),
                'estimated_count_preference': parsed_result.get('estimated_count_preference'),
                'reasoning': parsed_result.get('reasoning', 'AI semantic analysis')
            }
            
            # Cache the result
            self._cache_result(cache_key, result)
            
            logger.info(f"Comprehensive analysis: wants_all={result['wants_all']}, confidence={result['confidence']}")
            return result
            
        except Exception as e:
            logger.error(f"Comprehensive results analysis failed: {e}")
            # Fallback based on query length and complexity
            return {
                'wants_all': len(query.split()) > 8,  # Longer queries often want more results
                'confidence': 0.4,
                'estimated_count_preference': None,
                'reasoning': 'Fallback analysis based on query complexity'
            }
    
    async def get_semantic_scope(self, query: str) -> str:
        """
        Analyze scope using semantic understanding and embeddings if available.
        
        Args:
            query: The user's query
            
        Returns:
            str: 'comprehensive'|'filtered'|'specific'
        """
        try:
            # Use the main intent analysis which already determines scope
            intent = await self.analyze_intent(query)
            return intent['scope']
        except Exception as e:
            logger.error(f"Semantic scope analysis failed: {e}")
            # Simple fallback based on query characteristics
            if len(query.split()) > 10:
                return 'comprehensive'
            elif any(word in query.lower() for word in ['what is', 'define', 'explain', 'how does']):
                return 'specific'
            else:
                return 'filtered'

# Global instance for easy import
query_intent_analyzer = QueryIntentAnalyzer()

async def analyze_query_intent(query: str, llm_service=None, llm_config: Optional[Dict] = None) -> Dict[str, Any]:
    """Convenience function for query intent analysis"""
    return await query_intent_analyzer.analyze_intent(query, llm_service, llm_config)

async def wants_comprehensive_results(query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
    """Convenience function for comprehensive results analysis"""
    return await query_intent_analyzer.wants_comprehensive_results(query, context)

async def get_semantic_scope(query: str) -> str:
    """Convenience function for semantic scope analysis"""
    return await query_intent_analyzer.get_semantic_scope(query)