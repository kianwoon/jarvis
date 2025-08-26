"""
AI Task Planner Service

The intelligent brain of the notebook system that understands user intent,
separates data requirements from presentation needs, and creates comprehensive
execution plans for consistent, complete results.

This transforms the notebook from mechanical RAG into intelligent AI assistant.
"""

import asyncio
import json
import logging
import hashlib
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pydantic import BaseModel
import httpx

from app.core.notebook_llm_settings_cache import get_notebook_llm_settings

logger = logging.getLogger(__name__)

# Cache for task plans (TTL: 10 minutes)
_plan_cache = {}
_cache_ttl = 600  # 10 minutes


class RetrievalStrategy(BaseModel):
    """Single retrieval strategy within a plan"""
    query: str
    threshold: float = 0.3
    max_chunks: int = 200
    description: str


class DataRequirements(BaseModel):
    """What data the user needs"""
    entities: List[str]  # ["projects", "clients", "technologies"]
    attributes: List[str]  # ["name", "company", "years", "description"]
    completeness: str  # "all" | "relevant" | "top_k"
    expected_count: Optional[str] = None  # "~25" | "10-20" | None


class PresentationRequirements(BaseModel):
    """How to present the data"""
    format: str  # "table" | "list" | "narrative" | "bullet_points"
    sorting: Optional[Dict[str, str]] = None  # {"field": "years", "order": "asc"}
    fields_to_show: List[str]  # ["name", "company", "years"]
    include_details: bool = True


class VerificationRules(BaseModel):
    """Rules for verifying completeness"""
    min_expected_results: int
    require_diverse_sources: bool = True
    check_for_duplicates: bool = True
    confidence_threshold: float = 0.8


class TaskExecutionPlan(BaseModel):
    """Comprehensive plan for intelligent task execution"""
    intent_type: str  # "exhaustive_enumeration" | "targeted_search" | "exploration"
    confidence: float  # How confident we are in this plan
    
    data_requirements: DataRequirements
    retrieval_strategies: List[RetrievalStrategy]
    presentation: PresentationRequirements
    verification: VerificationRules
    
    reasoning: str  # Why this plan was chosen
    

class AITaskPlanner:
    """
    AI-powered task planner that understands user intent and creates
    intelligent execution plans for consistent, complete results.
    """
    
    def __init__(self):
        self.llm_settings = None
        self.llm_config = None
        self._load_llm_config()
    
    def _load_llm_config(self):
        """Load LLM configuration for plan generation"""
        try:
            self.llm_settings = get_notebook_llm_settings()
            self.llm_config = self.llm_settings.get('notebook_llm', {})
            logger.debug(f"Loaded LLM config for task planning: {self.llm_config.get('model', 'unknown')}")
        except Exception as e:
            logger.error(f"Failed to load LLM config: {e}")
            self._setup_fallback_config()
    
    def _setup_fallback_config(self):
        """Setup fallback LLM configuration"""
        self.llm_config = {
            "base_url": "http://host.docker.internal:11434",
            "model": "qwen3:30b-a3b-instruct-2507-q4_K_M",
            "temperature": 0.1,
            "max_tokens": 2000
        }
        logger.warning("Using fallback LLM configuration for task planning")
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for task plan"""
        return hashlib.md5(query.encode()).hexdigest()
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cache entry is still valid"""
        return (time.time() - timestamp) < _cache_ttl
    
    def _get_cached_plan(self, cache_key: str) -> Optional[TaskExecutionPlan]:
        """Get cached plan if valid"""
        if cache_key in _plan_cache:
            plan, timestamp = _plan_cache[cache_key]
            if self._is_cache_valid(timestamp):
                logger.debug(f"Cache hit for task plan: {cache_key[:8]}...")
                return plan
            else:
                del _plan_cache[cache_key]
        return None
    
    def _cache_plan(self, cache_key: str, plan: TaskExecutionPlan):
        """Cache the task plan"""
        _plan_cache[cache_key] = (plan, time.time())
        
        # Simple cleanup
        if len(_plan_cache) > 100:
            current_time = time.time()
            expired_keys = [
                key for key, (_, timestamp) in _plan_cache.items()
                if (current_time - timestamp) >= _cache_ttl
            ]
            for key in expired_keys:
                del _plan_cache[key]
    
    async def _make_llm_call(self, prompt: str, timeout: int = 30) -> str:
        """Make LLM API call for plan generation"""
        if not self.llm_config:
            self._load_llm_config()
        
        try:
            payload = {
                "model": self.llm_config.get("model", "qwen3:30b-a3b-instruct-2507-q4_K_M"),
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": self.llm_config.get("temperature", 0.1)
                }
            }
            
            base_url = self.llm_config.get("base_url", "http://host.docker.internal:11434")
            endpoint = f"{base_url}/api/chat"
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(endpoint, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get('message', {}).get('content', '').strip()
                    return content
                else:
                    raise Exception(f"LLM API returned status {response.status_code}")
                    
        except Exception as e:
            logger.error(f"LLM call for task planning failed: {e}")
            raise
    
    def _parse_json_from_response(self, response: str) -> Dict:
        """Extract and parse JSON from LLM response"""
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                logger.warning("No JSON found in LLM task planning response")
                return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from task planning response: {e}")
            return {}
    
    async def understand_and_plan(self, query: str) -> TaskExecutionPlan:
        """
        The main intelligence: understand user intent and create execution plan.
        
        This is where the AI truly shows intelligence by understanding what the user
        wants regardless of how they phrase it, and creating a comprehensive plan
        for consistent, complete results.
        """
        cache_key = self._get_cache_key(query)
        
        # Check cache first
        cached_plan = self._get_cached_plan(cache_key)
        if cached_plan:
            return cached_plan
        
        # Generate plan using AI
        try:
            plan = await self._generate_plan_with_ai(query)
            self._cache_plan(cache_key, plan)
            logger.info(f"AI Task Plan generated: {plan.intent_type} for {plan.data_requirements.entities}")
            return plan
            
        except Exception as e:
            logger.warning(f"AI plan generation failed ({str(e)}), using intelligent fallback")
            # Use intelligent fallback that still provides comprehensive planning
            fallback_plan = self._generate_fallback_plan(query)
            # Mark as AI-assisted fallback with higher confidence
            fallback_plan.confidence = 0.8  # Higher confidence for intelligent fallback
            fallback_plan.reasoning = f"Intelligent fallback plan due to LLM unavailability: {str(e)[:100]}"
            
            # Cache the fallback plan
            self._cache_plan(cache_key, fallback_plan)
            logger.info(f"Intelligent Fallback Plan generated: {fallback_plan.intent_type} for {fallback_plan.data_requirements.entities}")
            return fallback_plan
    
    async def _generate_plan_with_ai(self, query: str) -> TaskExecutionPlan:
        """Generate task plan using AI intelligence"""
        
        prompt = f"""
You are an intelligent task planner for a notebook AI system. Your job is to understand what the user REALLY wants and create a comprehensive execution plan.

USER QUERY: "{query}"

Analyze this query and understand:
1. What is the core intent? (Are they asking for exhaustive enumeration, targeted search, or exploration?)
2. What data do they need? (entities, attributes, completeness requirements)
3. How should it be presented? (format, sorting, fields to show)
4. What retrieval strategies would ensure complete results?

Key Intelligence:
- "list all projects" and "list all projects order by years" ask for THE SAME DATA, just different presentation
- For exhaustive requests, use multiple broad retrieval strategies to ensure completeness
- Separate DATA needs from PRESENTATION needs
- Plan for ~25 projects in typical notebook (adjust expectations)

Return ONLY a JSON object with this structure:

{{
    "intent_type": "exhaustive_enumeration|targeted_search|exploration",
    "confidence": 0.0-1.0,
    "data_requirements": {{
        "entities": ["projects", "companies", etc],
        "attributes": ["name", "company", "years", "description"],
        "completeness": "all|relevant|top_k",
        "expected_count": "~25|10-20|null"
    }},
    "retrieval_strategies": [
        {{
            "query": "projects initiatives solutions",
            "threshold": 0.3,
            "max_chunks": 200,
            "description": "Broad entity search"
        }},
        {{
            "query": "developed built implemented delivered",
            "threshold": 0.3,
            "max_chunks": 200,
            "description": "Action-based search"
        }},
        {{
            "query": "company client organization timeline years",
            "threshold": 0.3,
            "max_chunks": 150,
            "description": "Context and metadata search"
        }}
    ],
    "presentation": {{
        "format": "table|list|narrative|bullet_points",
        "sorting": {{"field": "years", "order": "asc"}} or null,
        "fields_to_show": ["name", "company", "years"],
        "include_details": true|false
    }},
    "verification": {{
        "min_expected_results": 20,
        "require_diverse_sources": true,
        "check_for_duplicates": true,
        "confidence_threshold": 0.8
    }},
    "reasoning": "Brief explanation of why this plan was chosen"
}}

Focus on creating plans that will return CONSISTENT results regardless of query phrasing variations.

JSON:
"""
        
        response = await self._make_llm_call(prompt, timeout=25)
        parsed_result = self._parse_json_from_response(response)
        
        if not parsed_result:
            raise Exception("Failed to parse AI plan response")
        
        return self._validate_and_create_plan(parsed_result, query)
    
    def _validate_and_create_plan(self, parsed_result: Dict, query: str) -> TaskExecutionPlan:
        """Validate AI response and create TaskExecutionPlan"""
        try:
            # Validate and normalize data requirements
            data_req = DataRequirements(
                entities=parsed_result.get('data_requirements', {}).get('entities', ['projects']),
                attributes=parsed_result.get('data_requirements', {}).get('attributes', ['name', 'description']),
                completeness=parsed_result.get('data_requirements', {}).get('completeness', 'relevant'),
                expected_count=parsed_result.get('data_requirements', {}).get('expected_count')
            )
            
            # Validate and normalize retrieval strategies
            strategies = []
            for strategy_data in parsed_result.get('retrieval_strategies', []):
                strategy = RetrievalStrategy(
                    query=strategy_data.get('query', 'projects work'),
                    threshold=strategy_data.get('threshold', 0.3),
                    max_chunks=strategy_data.get('max_chunks', 200),
                    description=strategy_data.get('description', 'General search')
                )
                strategies.append(strategy)
            
            # If no strategies provided, create default ones
            if not strategies:
                strategies = self._create_default_strategies(query)
            
            # Validate presentation requirements
            presentation = PresentationRequirements(
                format=parsed_result.get('presentation', {}).get('format', 'list'),
                sorting=parsed_result.get('presentation', {}).get('sorting'),
                fields_to_show=parsed_result.get('presentation', {}).get('fields_to_show', ['name']),
                include_details=parsed_result.get('presentation', {}).get('include_details', True)
            )
            
            # Validate verification rules
            verification = VerificationRules(
                min_expected_results=parsed_result.get('verification', {}).get('min_expected_results', 5),
                require_diverse_sources=parsed_result.get('verification', {}).get('require_diverse_sources', True),
                check_for_duplicates=parsed_result.get('verification', {}).get('check_for_duplicates', True),
                confidence_threshold=parsed_result.get('verification', {}).get('confidence_threshold', 0.8)
            )
            
            return TaskExecutionPlan(
                intent_type=parsed_result.get('intent_type', 'targeted_search'),
                confidence=max(0.0, min(1.0, parsed_result.get('confidence', 0.7))),
                data_requirements=data_req,
                retrieval_strategies=strategies,
                presentation=presentation,
                verification=verification,
                reasoning=parsed_result.get('reasoning', 'AI-generated execution plan')
            )
            
        except Exception as e:
            logger.error(f"Failed to validate AI plan: {e}")
            return self._generate_fallback_plan(query)
    
    def _create_default_strategies(self, query: str) -> List[RetrievalStrategy]:
        """Create default retrieval strategies when AI doesn't provide them"""
        return [
            RetrievalStrategy(
                query="projects work initiatives solutions systems",
                threshold=0.3,
                max_chunks=200,
                description="Broad entity search"
            ),
            RetrievalStrategy(
                query="developed built implemented created delivered",
                threshold=0.3,
                max_chunks=200,
                description="Action-based search"
            ),
            RetrievalStrategy(
                query="company client organization years timeline",
                threshold=0.35,
                max_chunks=150,
                description="Context search"
            )
        ]
    
    def _generate_fallback_plan(self, query: str) -> TaskExecutionPlan:
        """Generate fallback plan when AI fails"""
        query_lower = query.lower()
        
        # Detect enumeration patterns
        enumeration_indicators = [
            'all', 'every', 'complete', 'list', 'show', 'enumerate', 
            'find all', 'get all', 'display all'
        ]
        
        is_enumeration = any(indicator in query_lower for indicator in enumeration_indicators)
        
        # Detect entities being requested
        entities = ['projects']  # Default
        if any(term in query_lower for term in ['company', 'client', 'organization']):
            entities.append('companies')
        if any(term in query_lower for term in ['technology', 'skill', 'tool']):
            entities.append('technologies')
            
        # Detect presentation format
        format_type = 'list'
        if 'table' in query_lower:
            format_type = 'table'
        elif 'bullet' in query_lower or 'point' in query_lower:
            format_type = 'bullet_points'
            
        # Detect sorting
        sorting = None
        if 'order by year' in query_lower or 'sort by year' in query_lower:
            sorting = {"field": "years", "order": "asc"}
        elif 'order by' in query_lower or 'sort by' in query_lower:
            sorting = {"field": "name", "order": "asc"}
            
        return TaskExecutionPlan(
            intent_type="exhaustive_enumeration" if is_enumeration else "targeted_search",
            confidence=0.6,  # Moderate confidence for fallback
            data_requirements=DataRequirements(
                entities=entities,
                attributes=['name', 'company', 'years', 'description'],
                completeness="all" if is_enumeration else "relevant",
                expected_count="~25" if is_enumeration else None
            ),
            retrieval_strategies=self._create_default_strategies(query),
            presentation=PresentationRequirements(
                format=format_type,
                sorting=sorting,
                fields_to_show=['name', 'company', 'years'] if 'company' in query_lower else ['name'],
                include_details=True
            ),
            verification=VerificationRules(
                min_expected_results=20 if is_enumeration else 5,
                require_diverse_sources=True,
                check_for_duplicates=True,
                confidence_threshold=0.7
            ),
            reasoning="Fallback plan generated due to AI planning failure"
        )


# Global instance for easy import
ai_task_planner = AITaskPlanner()

async def create_execution_plan(query: str) -> TaskExecutionPlan:
    """Convenience function for creating task execution plans"""
    return await ai_task_planner.understand_and_plan(query)