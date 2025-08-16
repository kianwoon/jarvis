"""
Web Search Integration for Radiating System

This module integrates web search capabilities into the radiating entity extraction system
to discover the latest AI/LLM technologies beyond the LLM's knowledge cutoff.

Uses MCP (Model Context Protocol) tools for web search functionality.
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class WebSearchIntegration:
    """
    Integrates web search capabilities to augment entity extraction
    with real-time information about latest AI/LLM technologies.
    """
    
    def __init__(self):
        """Initialize the web search integration."""
        self.search_queries_cache = {}
        self.max_search_results = 10
        
        # Technology-focused search queries
        self.technology_search_templates = [
            "latest open source LLM frameworks {year}",
            "new AI tools {year}",
            "emerging LLM technologies {year}",
            "best vector databases {year}",
            "top RAG frameworks {year}",
            "new prompt engineering tools {year}",
            "latest AI agent frameworks {year}",
            "newest open source LLMs {year}",
            "cutting edge AI inference engines {year}",
            "modern LLM orchestration tools {year}"
        ]
        
        # Keywords that trigger web search
        self.web_search_triggers = [
            'latest', 'current', 'recent', 'new', 'newest',
            'emerging', 'cutting edge', 'modern', 'state of the art',
            '2024', '2025', 'this year', 'today', 'now'
        ]
    
    def should_use_web_search(self, query: str) -> bool:
        """
        Determine if web search would be valuable for this query.
        
        Args:
            query: The user's query text
            
        Returns:
            True if web search should be used
        """
        query_lower = query.lower()
        
        # Check for temporal triggers
        has_temporal = any(trigger in query_lower for trigger in self.web_search_triggers)
        
        # Check for technology exploration
        is_tech_query = any(term in query_lower for term in [
            'technology', 'technologies', 'tool', 'tools',
            'framework', 'frameworks', 'library', 'libraries',
            'platform', 'platforms', 'system', 'systems',
            'llm', 'ai', 'ml', 'rag', 'vector', 'agent'
        ])
        
        # Check for discovery intent
        has_discovery_intent = any(term in query_lower for term in [
            'what are', 'which', 'list', 'show', 'find',
            'discover', 'explore', 'available', 'options'
        ])
        
        return has_temporal or (is_tech_query and has_discovery_intent)
    
    async def search_for_technologies(self, query: str, focus_areas: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Perform web searches to discover latest technologies.
        
        Args:
            query: The original user query
            focus_areas: Optional list of focus areas to search
            
        Returns:
            List of search results with extracted entities
        """
        try:
            # Get current year for search queries
            current_year = datetime.now().year
            
            # Generate search queries based on the user's query
            search_queries = self._generate_search_queries(query, focus_areas, current_year)
            
            # Execute searches using MCP tools
            all_results = []
            for search_query in search_queries[:3]:  # Limit to 3 searches
                results = await self._execute_web_search(search_query)
                all_results.extend(results)
            
            # Extract entities from search results
            entities = await self._extract_entities_from_results(all_results)
            
            return entities
            
        except Exception as e:
            logger.error(f"Web search for technologies failed: {e}")
            return []
    
    def _generate_search_queries(self, user_query: str, focus_areas: Optional[List[str]], year: int) -> List[str]:
        """
        Generate targeted search queries based on user input.
        
        Args:
            user_query: The original user query
            focus_areas: Optional focus areas
            year: Current year for temporal queries
            
        Returns:
            List of search queries to execute
        """
        queries = []
        
        # Parse user query for specific interests
        query_lower = user_query.lower()
        
        # Add specific technology searches based on user query
        if 'llm' in query_lower or 'language model' in query_lower:
            queries.append(f"latest open source LLM frameworks {year}")
            queries.append(f"new LLM inference engines {year}")
        
        if 'rag' in query_lower or 'retrieval' in query_lower:
            queries.append(f"best RAG frameworks {year}")
            queries.append(f"latest vector databases for RAG {year}")
        
        if 'agent' in query_lower:
            queries.append(f"new AI agent frameworks {year}")
            queries.append(f"multi-agent orchestration tools {year}")
        
        if 'open source' in query_lower:
            queries.append(f"latest open source AI tools {year}")
            queries.append(f"new open source LLMs {year}")
        
        # Add general technology searches if no specific focus
        if not queries:
            queries.extend([
                f"emerging AI technologies {year}",
                f"latest LLM tools and frameworks {year}",
                f"new machine learning platforms {year}"
            ])
        
        # Add focus area searches if provided
        if focus_areas:
            for area in focus_areas[:2]:  # Limit focus areas
                queries.append(f"latest {area} technologies {year}")
        
        return queries
    
    async def _execute_web_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a web search using MCP tools.
        
        Args:
            query: Search query to execute
            
        Returns:
            List of search results
        """
        try:
            # Import MCP tool execution
            from app.core.enhanced_tool_executor import call_mcp_tool_enhanced_async
            from app.core.mcp_tools_cache import get_enabled_mcp_tools
            
            # Check if web search tools are available
            enabled_tools = get_enabled_mcp_tools()
            
            # Try different search tool names (support multiple search providers)
            search_tool_names = ['web_search', 'google_search', 'search_web', 'internet_search']
            search_tool = None
            
            for tool_name in search_tool_names:
                if tool_name in enabled_tools:
                    search_tool = tool_name
                    break
            
            if not search_tool:
                logger.warning("No web search tool available in MCP tools")
                return []
            
            logger.info(f"Executing web search with tool '{search_tool}': {query}")
            
            # Execute the search
            result = await call_mcp_tool_enhanced_async(
                search_tool,
                {
                    "query": query,
                    "num_results": self.max_search_results
                }
            )
            
            # Parse the results
            if "error" in result:
                logger.error(f"Web search error: {result['error']}")
                return []
            
            # Extract search results from the response
            search_results = self._parse_search_results(result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to execute web search for '{query}': {e}")
            return []
    
    def _parse_search_results(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse search results from MCP tool response.
        
        Args:
            result: Raw result from MCP tool
            
        Returns:
            Parsed list of search results
        """
        try:
            parsed_results = []
            
            # Handle different response formats
            if 'content' in result:
                content = result['content']
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and 'text' in item:
                            # Parse text content for results
                            text = item['text']
                            results = self._extract_results_from_text(text)
                            parsed_results.extend(results)
                elif isinstance(content, str):
                    results = self._extract_results_from_text(content)
                    parsed_results.extend(results)
            
            elif 'results' in result:
                # Direct results format
                for item in result['results']:
                    parsed_results.append({
                        'title': item.get('title', ''),
                        'snippet': item.get('snippet', ''),
                        'url': item.get('link', item.get('url', ''))
                    })
            
            return parsed_results
            
        except Exception as e:
            logger.error(f"Failed to parse search results: {e}")
            return []
    
    def _extract_results_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract structured results from text response.
        
        Args:
            text: Text containing search results
            
        Returns:
            List of extracted results
        """
        results = []
        
        # Split by common result separators
        sections = re.split(r'\n\n|\*\*.*?\*\*', text)
        
        for section in sections:
            if not section.strip():
                continue
            
            # Try to extract title, snippet, and URL
            lines = section.strip().split('\n')
            if lines:
                result = {
                    'title': '',
                    'snippet': '',
                    'url': ''
                }
                
                # Extract URL (look for http/https links)
                url_match = re.search(r'https?://[^\s]+', section)
                if url_match:
                    result['url'] = url_match.group(0)
                
                # First line is often the title
                if lines[0]:
                    result['title'] = lines[0].strip('*[]')
                
                # Rest is snippet
                if len(lines) > 1:
                    result['snippet'] = ' '.join(lines[1:])
                elif lines[0]:
                    result['snippet'] = lines[0]
                
                if result['title'] or result['snippet']:
                    results.append(result)
        
        return results
    
    async def _extract_entities_from_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract technology entities from search results.
        
        Args:
            search_results: List of search results
            
        Returns:
            List of extracted entities
        """
        entities = []
        seen_entities = set()  # Avoid duplicates
        
        # Technology patterns to extract
        tech_patterns = [
            # Frameworks and libraries with version numbers
            r'\b([A-Z][a-zA-Z]+(?:[A-Z][a-zA-Z]+)*)\s*(?:v?[\d.]+)?\b',
            # Technologies in quotes
            r'"([^"]+)"',
            r"'([^']+)'",
            # Technologies after keywords
            r'(?:framework|library|tool|platform|engine|database|model):\s*([A-Za-z][A-Za-z0-9\s\-_.]+)',
            # Common technology name patterns
            r'\b([A-Z][a-z]+[A-Z][a-zA-Z]*)\b',  # CamelCase
            r'\b([A-Z]{2,}[a-z]*)\b',  # Acronyms like LLM, RAG
            r'\b([a-z]+\.js)\b',  # JavaScript libraries
            r'\b([A-Za-z]+DB)\b',  # Databases
            r'\b([A-Za-z]+AI)\b',  # AI tools
        ]
        
        for result in search_results:
            text = f"{result.get('title', '')} {result.get('snippet', '')}"
            
            # Extract using patterns
            for pattern in tech_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    entity_text = match.strip()
                    
                    # Filter out common words and short matches
                    if (len(entity_text) > 2 and 
                        entity_text not in seen_entities and
                        not self._is_common_word(entity_text)):
                        
                        # Determine entity type based on context
                        entity_type = self._classify_technology(entity_text, text)
                        
                        if entity_type:
                            entities.append({
                                'text': entity_text,
                                'type': entity_type,
                                'source': 'web_search',
                                'url': result.get('url', ''),
                                'confidence': 0.7,  # Web search entities get moderate confidence
                                'context': result.get('snippet', '')[:200]
                            })
                            seen_entities.add(entity_text)
        
        # Sort by confidence and limit
        entities.sort(key=lambda x: x['confidence'], reverse=True)
        
        return entities[:20]  # Return top 20 web-discovered entities
    
    def _is_common_word(self, word: str) -> bool:
        """Check if a word is too common to be a technology entity."""
        common_words = {
            'the', 'and', 'for', 'with', 'this', 'that', 'from',
            'have', 'will', 'can', 'are', 'was', 'were', 'been',
            'their', 'your', 'our', 'its', 'these', 'those',
            'what', 'which', 'when', 'where', 'who', 'how',
            'new', 'latest', 'best', 'top', 'most', 'more'
        }
        return word.lower() in common_words
    
    def _classify_technology(self, entity_text: str, context: str) -> Optional[str]:
        """
        Classify a technology entity based on its name and context.
        
        Args:
            entity_text: The entity text
            context: The context where it was found
            
        Returns:
            Entity type or None if not a technology
        """
        entity_lower = entity_text.lower()
        context_lower = context.lower()
        
        # Classification rules
        if 'llm' in entity_lower or 'language model' in context_lower:
            return 'LLMFramework'
        elif 'vector' in context_lower and 'database' in context_lower:
            return 'VectorDatabase'
        elif 'rag' in entity_lower or 'retrieval' in context_lower:
            return 'RAGFramework'
        elif 'agent' in context_lower:
            return 'AgentFramework'
        elif 'inference' in context_lower or 'serving' in context_lower:
            return 'InferenceEngine'
        elif 'embed' in context_lower:
            return 'EmbeddingModel'
        elif 'database' in context_lower or 'db' in entity_lower:
            return 'Database'
        elif 'framework' in context_lower:
            return 'Framework'
        elif 'library' in context_lower or '.js' in entity_lower or '.py' in entity_lower:
            return 'Library'
        elif 'tool' in context_lower:
            return 'Tool'
        elif 'platform' in context_lower:
            return 'Platform'
        elif 'api' in entity_lower or 'api' in context_lower:
            return 'API'
        elif 'model' in context_lower:
            return 'Model'
        else:
            # Check if it looks like a technology name
            if re.match(r'^[A-Z]', entity_text) and len(entity_text) > 3:
                return 'Technology'
        
        return None
    
    async def merge_entities(
        self,
        llm_entities: List[Any],
        web_entities: List[Dict[str, Any]]
    ) -> List[Any]:
        """
        Merge LLM-extracted entities with web-discovered entities.
        
        Args:
            llm_entities: Entities extracted by the LLM
            web_entities: Entities discovered from web search
            
        Returns:
            Merged list of entities, prioritizing newer information
        """
        # Convert LLM entities to dict for easier comparison
        llm_entity_texts = {e.text.lower(): e for e in llm_entities}
        
        merged = list(llm_entities)  # Start with LLM entities
        
        # Add web entities that are not duplicates
        for web_entity in web_entities:
            entity_text_lower = web_entity['text'].lower()
            
            if entity_text_lower not in llm_entity_texts:
                # Create entity object compatible with ExtractedEntity
                from dataclasses import dataclass
                
                @dataclass
                class WebExtractedEntity:
                    text: str
                    entity_type: str
                    confidence: float
                    context: str
                    metadata: Dict[str, Any]
                    entity_id: Optional[str] = None
                    
                    def __post_init__(self):
                        if not self.entity_id:
                            import hashlib
                            hash_input = f"{self.text}_{self.entity_type}"
                            self.entity_id = hashlib.md5(hash_input.encode()).hexdigest()[:16]
                
                new_entity = WebExtractedEntity(
                    text=web_entity['text'],
                    entity_type=web_entity['type'],
                    confidence=web_entity['confidence'],
                    context=web_entity['context'],
                    metadata={
                        'source': 'web_search',
                        'url': web_entity.get('url', ''),
                        'extraction_method': 'web_search_discovery'
                    }
                )
                merged.append(new_entity)
            else:
                # Boost confidence of LLM entity if also found in web search
                llm_entity = llm_entity_texts[entity_text_lower]
                llm_entity.confidence = min(1.0, llm_entity.confidence + 0.1)
                llm_entity.metadata['verified_by_web'] = True
        
        # Sort by confidence
        merged.sort(key=lambda x: x.confidence, reverse=True)
        
        return merged