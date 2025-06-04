"""
Query Classifier for routing queries to appropriate handlers
"""
import re
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of queries for routing"""
    RAG_SEARCH = "rag_search"          # Knowledge retrieval from documents
    TOOL_USE = "tool_use"              # Requires external tools/APIs
    DIRECT_LLM = "direct_llm"          # General conversation/reasoning
    CODE_GENERATION = "code_generation" # Programming/code related
    MULTI_AGENT = "multi_agent"        # Complex tasks requiring multiple agents
    
class QueryClassifier:
    """Classify queries to determine appropriate processing route"""
    
    def __init__(self):
        # Tool-related keywords and patterns
        self.tool_patterns = [
            r'\b(search|browse|web|internet|google|website|url|http)\b',
            r'\b(weather|temperature|forecast|climate)\b',
            r'\b(calculate|compute|math|arithmetic)\b',
            r'\b(api|endpoint|request|fetch|call)\b',
            r'\b(database|query|sql|mongo|postgres)\b',
            r'\b(file|read|write|save|load|download|upload)\b',
            r'\b(email|send|notify|message)\b',
            r'\b(translate|translation|language)\b',
            r'\b(current|today|now|latest|real-time|live)\b',
        ]
        
        # RAG/knowledge search patterns
        self.rag_patterns = [
            r'\b(what|who|when|where|why|how)\b.*\b(is|are|was|were|does|do|did)\b',
            r'\b(tell me about|explain|describe|what do you know about)\b',
            r'\b(find|search|look for|locate|retrieve)\b.*\b(document|file|information|data|info|details|about)\b',
            r'\b(find out|look up|search for|get)\b.*\b(info|information|details|data)\b',
            r'\b(according to|based on|from the|in the)\b.*\b(document|file|data|information)\b',
            r'\b(summarize|summary|overview|brief)\b',
            r'\b(documentation|manual|guide|reference)\b',
            r'\b(policy|procedure|guideline|standard)\b',
            r'\b(outage|outages|issue|issues|problem|problems|incident|incidents)\b.*\b(info|information|details|report)\b',
        ]
        
        # Code generation patterns
        self.code_patterns = [
            r'\b(code|program|script|function|class|method)\b',
            r'\b(python|javascript|java|c\+\+|typescript|golang|rust)\b',
            r'\b(implement|create|write|develop|build)\b.*\b(code|function|program|app)\b',
            r'\b(debug|fix|error|bug|issue)\b.*\b(code|program|script)\b',
            r'\b(algorithm|data structure|design pattern)\b',
            r'\b(api|library|framework|package)\b',
        ]
        
        # Multi-agent patterns (complex tasks)
        self.multi_agent_patterns = [
            r'\b(analyze|research|investigate)\b.*\b(and|then|also|plus)\b.*\b(create|write|develop|build)\b',
            r'\b(multiple|several|various|different)\b.*\b(tasks|steps|things|aspects)\b',
            r'\b(comprehensive|complete|full|entire|whole)\b.*\b(analysis|report|solution)\b',
            r'\b(step by step|step-by-step|iterative|sequential)\b',
            r'\b(complex|complicated|sophisticated|advanced)\b',
        ]
        
    def classify(self, query: str) -> Tuple[QueryType, float, Dict[str, any]]:
        """
        Classify a query and return the type with confidence score
        
        Returns:
            Tuple of (QueryType, confidence_score, metadata)
        """
        query_lower = query.lower()
        scores = {
            QueryType.TOOL_USE: 0.0,
            QueryType.RAG_SEARCH: 0.0,
            QueryType.CODE_GENERATION: 0.0,
            QueryType.MULTI_AGENT: 0.0,
            QueryType.DIRECT_LLM: 0.1,  # Base score for direct LLM
        }
        
        metadata = {
            "matched_patterns": [],
            "query_length": len(query.split()),
            "has_question_words": False,
        }
        
        # Check for tool use patterns
        for pattern in self.tool_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                scores[QueryType.TOOL_USE] += 0.3
                metadata["matched_patterns"].append(("tool", pattern))
                
        # Check for RAG patterns
        for pattern in self.rag_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                scores[QueryType.RAG_SEARCH] += 0.25
                metadata["matched_patterns"].append(("rag", pattern))
                
        # Check for question words
        question_words = ['what', 'who', 'when', 'where', 'why', 'how', 'which', 'whose']
        if any(word in query_lower.split() for word in question_words):
            metadata["has_question_words"] = True
            scores[QueryType.RAG_SEARCH] += 0.1
            
        # Check for code patterns
        for pattern in self.code_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                scores[QueryType.CODE_GENERATION] += 0.3
                metadata["matched_patterns"].append(("code", pattern))
                
        # Check for multi-agent patterns
        for pattern in self.multi_agent_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                scores[QueryType.MULTI_AGENT] += 0.35
                metadata["matched_patterns"].append(("multi_agent", pattern))
                
        # Adjust scores based on query characteristics
        if metadata["query_length"] > 20:  # Long queries might need multi-agent
            scores[QueryType.MULTI_AGENT] += 0.1
            
        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            for query_type in scores:
                scores[query_type] /= total_score
                
        # Get the highest scoring type
        best_type = max(scores.items(), key=lambda x: x[1])
        
        # Log classification decision
        logger.info(f"Query classification: {best_type[0].value} (confidence: {best_type[1]:.2f})")
        logger.debug(f"All scores: {[(t.value, f'{s:.2f}') for t, s in scores.items()]}")
        logger.debug(f"Metadata: {metadata}")
        
        return best_type[0], best_type[1], metadata
    
    def should_use_rag(self, query_type: QueryType, confidence: float) -> bool:
        """Determine if RAG should be used based on query type and confidence"""
        # Use RAG for search queries or when confidence is low
        if query_type == QueryType.RAG_SEARCH:
            return True
        elif query_type == QueryType.DIRECT_LLM and confidence < 0.5:
            # Low confidence direct LLM might benefit from RAG
            return True
        elif query_type in [QueryType.MULTI_AGENT, QueryType.CODE_GENERATION]:
            # These might need RAG for context
            return confidence < 0.7
        else:
            return False
            
    def get_routing_recommendation(self, query: str) -> Dict[str, any]:
        """
        Get complete routing recommendation for a query
        
        Returns:
            Dict with routing information and metadata
        """
        query_type, confidence, metadata = self.classify(query)
        
        recommendation = {
            "query_type": query_type.value,
            "confidence": confidence,
            "use_rag": self.should_use_rag(query_type, confidence),
            "metadata": metadata,
            "routing": {
                "primary_handler": self._get_primary_handler(query_type),
                "fallback_handler": "direct_llm",
                "use_streaming": True,
            }
        }
        
        # Add specific recommendations based on query type
        if query_type == QueryType.TOOL_USE:
            recommendation["routing"]["suggested_tools"] = self._suggest_tools(query)
        elif query_type == QueryType.MULTI_AGENT:
            recommendation["routing"]["suggested_agents"] = self._suggest_agents(query)
            
        return recommendation
        
    def _get_primary_handler(self, query_type: QueryType) -> str:
        """Map query type to primary handler"""
        handlers = {
            QueryType.RAG_SEARCH: "rag",
            QueryType.TOOL_USE: "tool_handler",
            QueryType.DIRECT_LLM: "llm",
            QueryType.CODE_GENERATION: "code_agent",
            QueryType.MULTI_AGENT: "multi_agent",
        }
        return handlers.get(query_type, "llm")
        
    def _suggest_tools(self, query: str) -> List[str]:
        """Suggest tools based on query content"""
        tools = []
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['search', 'web', 'internet', 'browse']):
            tools.append("web_search")
        if any(word in query_lower for word in ['weather', 'temperature', 'forecast']):
            tools.append("weather_api")
        if any(word in query_lower for word in ['calculate', 'compute', 'math']):
            tools.append("calculator")
        if any(word in query_lower for word in ['file', 'read', 'write', 'save']):
            tools.append("file_system")
            
        return tools
        
    def _suggest_agents(self, query: str) -> List[str]:
        """Suggest agents based on query content"""
        agents = []
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['analyze', 'research', 'investigate']):
            agents.append("research_analyst")
        if any(word in query_lower for word in ['code', 'program', 'implement']):
            agents.append("code_developer")
        if any(word in query_lower for word in ['write', 'create', 'document']):
            agents.append("content_writer")
        if any(word in query_lower for word in ['plan', 'strategy', 'organize']):
            agents.append("project_planner")
            
        return agents