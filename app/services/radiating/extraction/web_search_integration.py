"""
Web Search Integration for Radiating System

This module integrates web search capabilities into the radiating entity extraction system
to discover the latest information on ANY topic, making web search the PRIMARY source
for maximum freshness and accuracy.

The internet is ALWAYS the most up-to-date source for:
- Current events and news
- People and their activities  
- Company information
- Product releases
- Research and discoveries
- Market data
- Sports, entertainment
- Politics and policy
- Scientific developments
- Technology updates
- ANY real-world information

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
    with real-time information about ANY topic or entity.
    Web search is the PRIMARY source for all queries to ensure maximum freshness.
    """
    
    def __init__(self):
        """Initialize the web search integration."""
        self.search_queries_cache = {}
        self.max_search_results = 20
        
        # Universal search query templates for any domain
        self.universal_search_templates = [
            # General news and updates
            "latest news about {topic} {year}",
            "current {topic} updates {year}",
            "recent developments in {topic}",
            "{topic} news today",
            "what's new with {topic} {year}",
            "breaking news {topic}",
            "{topic} headlines {year}",
            
            # People and organizations
            "{person} latest news {year}",
            "{company} recent announcements {year}",
            "{organization} current activities",
            
            # Products and releases
            "new {product} releases {year}",
            "latest {product} updates",
            "{product} features {year}",
            
            # Research and discoveries
            "recent research on {topic} {year}",
            "latest studies about {topic}",
            "new discoveries in {field} {year}",
            
            # Events and happenings
            "{event} latest information {year}",
            "current {event} status",
            "upcoming {event} details",
            
            # Technology (kept for compatibility)
            "latest {technology} frameworks {year}",
            "new {technology} tools {year}",
            "emerging {technology} trends {year}",
            
            # Market and business
            "{market} trends {year}",
            "{industry} news today",
            "latest {sector} developments",
            
            # General exploration
            "everything about {topic} {year}",
            "comprehensive guide to {topic}",
            "{topic} overview {year}"
        ]
        
        # Keywords that trigger web search
        self.web_search_triggers = [
            'latest', 'current', 'recent', 'new', 'newest',
            'emerging', 'cutting edge', 'modern', 'state of the art',
            '2024', '2025', 'this year', 'today', 'now'
        ]
    
    def should_use_web_search(self, query: str) -> bool:
        """
        Determine if web search should be used for this query.
        Web search is now the PRIMARY source for ALL queries by default.
        
        Args:
            query: The user's query text
            
        Returns:
            True if web search should be used (True by default for maximum freshness)
        """
        query_lower = query.lower()
        
        # Only skip web search for very specific local/personal queries
        skip_web_search_patterns = [
            # Personal/local file queries
            'my file', 'my document', 'my code', 'my project',
            'local file', 'local database', 'localhost',
            # Code analysis queries
            'analyze this code', 'review this function', 'debug this',
            'explain this code', 'what does this code do',
            # Abstract/philosophical queries
            'meaning of life', 'philosophical', 'hypothetical',
            # Basic math/logic
            'calculate', 'compute', 'solve equation',
        ]
        
        # Check if query is purely local/personal
        is_local_query = any(pattern in query_lower for pattern in skip_web_search_patterns)
        
        if is_local_query:
            # Even for local queries, use web if temporal keywords present
            has_temporal = any(trigger in query_lower for trigger in self.web_search_triggers)
            if has_temporal:
                return True
            return False
        
        # DEFAULT: Always use web search for maximum freshness
        # The internet is the most current source for:
        # - News and current events
        # - People, companies, organizations
        # - Products and releases  
        # - Research and discoveries
        # - Market data and trends
        # - Technology updates
        # - ANY real-world information
        return True
    
    async def search_for_entities(self, query: str, focus_areas: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Perform web searches to discover latest information on ANY topic.
        
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
            for search_query in search_queries[:10]:  # Allow up to 10 searches for comprehensive coverage
                results = await self._execute_web_search(search_query)
                all_results.extend(results)
            
            # Extract entities from search results
            entities = await self._extract_entities_from_results(all_results)
            
            return entities
            
        except Exception as e:
            logger.error(f"Web search for entities failed: {e}")
            return []
    
    def _extract_key_topics(self, query: str) -> List[str]:
        """
        Extract key topics/entities from a query string.
        
        Args:
            query: The user query to extract topics from
            
        Returns:
            List of key topics/entities
        """
        # Remove common words and extract meaningful topics
        stop_words = {
            'the', 'and', 'for', 'with', 'this', 'that', 'from', 'about',
            'have', 'will', 'can', 'are', 'was', 'were', 'been', 'what',
            'their', 'your', 'our', 'its', 'these', 'those', 'how',
            'which', 'when', 'where', 'who', 'why', 'show', 'tell',
            'give', 'find', 'search', 'looking', 'need', 'want', 'please'
        }
        
        # Simple topic extraction (can be enhanced with NLP)
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9\-_.]+\b', query)
        topics = []
        
        for word in words:
            if word.lower() not in stop_words and len(word) > 2:
                topics.append(word)
        
        # Also extract multi-word phrases (proper nouns, compound terms)
        # Look for capitalized phrases
        cap_phrases = re.findall(r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b', query)
        topics.extend(cap_phrases)
        
        # Extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', query) + re.findall(r"'([^']+)'", query)
        topics.extend(quoted)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_topics = []
        for topic in topics:
            if topic.lower() not in seen:
                unique_topics.append(topic)
                seen.add(topic.lower())
        
        return unique_topics[:5]  # Return top 5 topics
    
    def _generate_search_queries(self, user_query: str, focus_areas: Optional[List[str]], year: int) -> List[str]:
        """
        Generate targeted search queries based on user input for ANY domain.
        
        Args:
            user_query: The original user query
            focus_areas: Optional focus areas
            year: Current year for temporal queries
            
        Returns:
            List of search queries to execute
        """
        queries = []
        query_lower = user_query.lower()
        
        # Extract key topics/entities from the query
        key_topics = self._extract_key_topics(user_query)
        
        # Always start with the user's exact query for best relevance
        queries.append(user_query)
        
        # Add temporal variations of the query
        queries.extend([
            f"{user_query} {year}",
            f"latest {user_query}",
            f"current {user_query}",
            f"{user_query} news",
            f"{user_query} updates {year}"
        ])
        
        # Generate domain-specific queries based on detected patterns
        
        # People queries
        person_patterns = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'  # Simple name pattern
        if re.search(person_patterns, user_query):
            for match in re.findall(person_patterns, user_query):
                queries.extend([
                    f"{match} latest news {year}",
                    f"what is {match} doing now",
                    f"{match} recent activities {year}"
                ])
        
        # Company/Organization queries
        if any(term in query_lower for term in ['company', 'corporation', 'inc', 'ltd', 'org']):
            for topic in key_topics:
                queries.extend([
                    f"{topic} company news {year}",
                    f"{topic} latest announcements",
                    f"{topic} quarterly results {year}"
                ])
        
        # Product queries
        if any(term in query_lower for term in ['product', 'release', 'launch', 'version']):
            for topic in key_topics:
                queries.extend([
                    f"{topic} new releases {year}",
                    f"{topic} product updates",
                    f"latest {topic} features {year}"
                ])
        
        # Event queries
        if any(term in query_lower for term in ['event', 'conference', 'summit', 'meeting']):
            for topic in key_topics:
                queries.extend([
                    f"{topic} events {year}",
                    f"{topic} conference updates",
                    f"upcoming {topic} events"
                ])
        
        # Research/Science queries
        if any(term in query_lower for term in ['research', 'study', 'science', 'discovery']):
            for topic in key_topics:
                queries.extend([
                    f"latest research on {topic} {year}",
                    f"new studies about {topic}",
                    f"{topic} scientific discoveries {year}"
                ])
        
        # Technology queries (keep existing tech support)
        if any(term in query_lower for term in ['technology', 'tech', 'software', 'ai', 'ml', 'llm']):
            for topic in key_topics:
                queries.extend([
                    f"latest {topic} technology {year}",
                    f"new {topic} tools {year}",
                    f"{topic} frameworks {year}"
                ])
        
        # Market/Business queries
        if any(term in query_lower for term in ['market', 'business', 'industry', 'stock', 'finance']):
            for topic in key_topics:
                queries.extend([
                    f"{topic} market trends {year}",
                    f"{topic} industry news",
                    f"{topic} business updates {year}"
                ])
        
        # Sports/Entertainment queries
        if any(term in query_lower for term in ['sport', 'game', 'match', 'movie', 'music', 'entertainment']):
            for topic in key_topics:
                queries.extend([
                    f"{topic} latest news {year}",
                    f"{topic} updates today",
                    f"{topic} recent results"
                ])
        
        # Politics/Policy queries
        if any(term in query_lower for term in ['politics', 'policy', 'government', 'election', 'legislation']):
            for topic in key_topics:
                queries.extend([
                    f"{topic} political news {year}",
                    f"{topic} policy updates",
                    f"{topic} government announcements {year}"
                ])
        
        # Add focus area searches if provided
        if focus_areas:
            for area in focus_areas[:3]:  # Allow more focus areas
                queries.extend([
                    f"latest {area} news {year}",
                    f"current {area} updates",
                    f"{area} developments {year}"
                ])
        
        # If we don't have many queries yet, add general exploration queries
        if len(queries) < 5:
            for topic in key_topics[:2]:
                queries.extend([
                    f"everything about {topic} {year}",
                    f"comprehensive {topic} information",
                    f"{topic} complete overview {year}"
                ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            if q.lower() not in seen:
                unique_queries.append(q)
                seen.add(q.lower())
        
        return unique_queries[:15]  # Increase limit for comprehensive coverage
    
    # Backward compatibility alias
    async def search_for_technologies(self, query: str, focus_areas: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Backward compatibility alias for search_for_entities."""
        return await self.search_for_entities(query, focus_areas)
    
    def generate_query_variations(self, base_query: str, max_variations: int = 5) -> List[str]:
        """
        Generate multiple search query variations for comprehensive coverage across ANY domain.
        
        Args:
            base_query: The base query to generate variations from
            max_variations: Maximum number of variations to generate
            
        Returns:
            List of query variations
        """
        variations = []
        current_year = datetime.now().year
        query_lower = base_query.lower()
        
        # Universal temporal qualifiers (works for ANY domain)
        temporal_variations = [
            f"latest {base_query} {current_year}",
            f"current {base_query} {current_year}",
            f"recent {base_query} updates",
            f"{base_query} news today",
            f"new {base_query} {current_year}",
            f"{base_query} trends {current_year}",
            f"breaking {base_query} news",
            f"{base_query} latest developments",
            f"what's new with {base_query}",
            f"{base_query} {current_year} updates"
        ]
        variations.extend(temporal_variations[:max_variations])
        
        # General exploration queries (works for any topic)
        exploration_variations = [
            f"everything about {base_query}",
            f"{base_query} comprehensive guide",
            f"{base_query} complete information",
            f"all about {base_query} {current_year}",
            f"{base_query} overview {current_year}",
            f"understanding {base_query}",
            f"{base_query} explained {current_year}"
        ]
        variations.extend(exploration_variations[:3])
        
        # Comparison queries (universal)
        if 'vs' not in query_lower and 'versus' not in query_lower:
            variations.extend([
                f"{base_query} alternatives {current_year}",
                f"best {base_query} options",
                f"{base_query} comparison {current_year}",
                f"top {base_query} choices",
                f"{base_query} reviews {current_year}"
            ])
        
        # Add domain-agnostic research queries
        variations.extend([
            f"{base_query} research {current_year}",
            f"{base_query} studies {current_year}",
            f"{base_query} analysis {current_year}",
            f"{base_query} report {current_year}",
            f"{base_query} data {current_year}"
        ])
        
        # Implementation/How-to queries (universal)
        variations.extend([
            f"how to {base_query}",
            f"{base_query} guide {current_year}",
            f"{base_query} tutorial {current_year}",
            f"{base_query} tips {current_year}",
            f"{base_query} best practices {current_year}"
        ])
        
        # Future/Prediction queries
        variations.extend([
            f"{base_query} future trends",
            f"{base_query} predictions {current_year + 1}",
            f"{base_query} outlook {current_year}",
            f"future of {base_query}"
        ])
        
        # Location-based variations (if applicable)
        variations.extend([
            f"{base_query} near me",
            f"{base_query} worldwide {current_year}",
            f"global {base_query} trends {current_year}"
        ])
        
        # Keep technology-specific variations if detected
        if any(tech_term in query_lower for tech_term in ['ai', 'ml', 'llm', 'software', 'tech', 'app', 'platform']):
            variations.extend([
                f"open source {base_query}",
                f"{base_query} github repositories",
                f"{base_query} implementation",
                f"{base_query} documentation"
            ])
        
        # Remove duplicates and limit
        seen = set()
        unique_variations = []
        for v in variations:
            v_lower = v.lower().strip()
            base_lower = base_query.lower().strip()
            if v_lower not in seen and v_lower != base_lower:
                unique_variations.append(v)
                seen.add(v_lower)
                if len(unique_variations) >= max_variations * 2:  # Allow more variations
                    break
        
        return unique_variations[:max_variations]
    
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
                            
                            # Enhanced error detection for API failures
                            error_indicators = [
                                'Error:', 'error:', 'ERROR:',
                                '400', '401', '403', '404', '429', '500', '502', '503',
                                'API error', 'api error', 'API Error',
                                'Failed to', 'failed to', 'Could not', 'could not',
                                'Unable to', 'unable to',
                                'Invalid', 'invalid',
                                'Rate limit', 'rate limit',
                                'Quota exceeded', 'quota exceeded',
                                'Authentication failed', 'authentication failed',
                                'Service unavailable', 'service unavailable',
                                'Bad request', 'bad request',
                                'No results found', 'no results found'
                            ]
                            
                            # Check if this is an error message
                            if any(indicator in text for indicator in error_indicators):
                                logger.warning(f"Search API error or failure detected: {text[:200]}...")
                                # Return empty results for errors to prevent entity extraction from error messages
                                return []
                            
                            results = self._extract_results_from_text(text)
                            parsed_results.extend(results)
                elif isinstance(content, str):
                    # Also check for errors in string content
                    error_indicators = [
                        'Error:', 'error:', 'ERROR:',
                        '400', '401', '403', '404', '429', '500', '502', '503',
                        'API error', 'Failed to', 'Unable to'
                    ]
                    if any(indicator in content for indicator in error_indicators):
                        logger.warning(f"Search API error detected in content: {content[:200]}...")
                        return []
                    
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
        Extract entities of ANY type from search results.
        
        Args:
            search_results: List of search results
            
        Returns:
            List of extracted entities
        """
        entities = []
        seen_entities = set()  # Avoid duplicates
        
        # Blacklist of entity names that shouldn't be extracted (error-related)
        entity_blacklist = {
            'api', 'error', 'google search', 'google', 'search',
            'api error', 'search error', 'failed', 'failure',
            'invalid', 'unable', 'could not', 'bad request',
            'not found', 'unauthorized', 'forbidden', 'rate limit',
            'quota', 'authentication', 'service', 'unavailable'
        }
        
        # Skip if we got error results
        if search_results and len(search_results) == 1:
            first_result = search_results[0]
            if 'Error' in first_result.get('title', '') or 'error' in first_result.get('snippet', ''):
                logger.warning("Skipping entity extraction from error results")
                return []
        
        # Universal entity patterns to extract
        entity_patterns = [
            # People names (capitalized words, typically 2-3 words)
            r'\b([A-Z][a-z]+(?: [A-Z][a-z]+){1,2})\b',
            # Organizations/Companies (often with Inc, Corp, Ltd, etc.)
            r'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*(?:\s+(?:Inc|Corp|Corporation|Ltd|LLC|Company|Co|Group|Foundation|Institute|University|College|Agency|Department|Ministry|Commission|Bureau|Authority))?)\b',
            # Locations (cities, countries, places)
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:,\s*[A-Z][a-z]+)?)\b',
            # Products/Brands (often single word or compound)
            r'\b([A-Z][a-zA-Z]+(?:[A-Z][a-zA-Z]+)*)\s*(?:v?[\d.]+)?\b',
            # Events (conferences, summits, etc.)
            r'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*(?:\s+(?:Conference|Summit|Meeting|Forum|Symposium|Convention|Expo|Festival|Championship|Games|Awards|Ceremony))?)\b',
            # Quoted entities (anything important in quotes)
            r'"([^"]+)"',
            r"'([^']+)'",
            # Entities after keywords
            r'(?:CEO|founder|president|director|manager|leader|head|chief)(?:\s+of)?\s+([A-Za-z][A-Za-z0-9\s\-_.]+)',
            r'(?:company|organization|institution|agency|department):\s*([A-Za-z][A-Za-z0-9\s\-_.]+)',
            # Technology patterns (kept for compatibility)
            r'(?:framework|library|tool|platform|engine|database|model):\s*([A-Za-z][A-Za-z0-9\s\-_.]+)',
            # Common name patterns
            r'\b([A-Z][a-z]+[A-Z][a-zA-Z]*)\b',  # CamelCase
            r'\b([A-Z]{2,}[a-z]*)\b',  # Acronyms
            r'\b([a-z]+\.(?:js|py|go|rs|java|cpp))\b',  # Programming libraries
            r'\b([A-Za-z]+(?:DB|SQL|API|SDK|OS|UI|UX|ML|AI|IoT))\b',  # Tech acronyms
        ]
        
        for result in search_results:
            text = f"{result.get('title', '')} {result.get('snippet', '')}"
            
            # Extract using patterns
            for pattern in entity_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    entity_text = match.strip()
                    entity_text_lower = entity_text.lower()
                    
                    # Filter out common words, short matches, and blacklisted entities
                    if (len(entity_text) > 2 and 
                        entity_text not in seen_entities and
                        entity_text_lower not in entity_blacklist and
                        not self._is_common_word(entity_text)):
                        
                        # Determine entity type based on context
                        entity_type = self._classify_entity(entity_text, text)
                        
                        if entity_type:
                            # Additional check: skip if it's a low-confidence error-related entity
                            if entity_text_lower in {'api', 'error', 'google', 'search'} and entity_type == 'Entity':
                                logger.debug(f"Filtering out potential error entity: {entity_text} (type: {entity_type})")
                                continue
                            
                            entities.append({
                                'text': entity_text,
                                'type': entity_type,
                                'source': 'web_search',
                                'url': result.get('url', ''),
                                'confidence': self.calculate_entity_confidence(
                                    {'text': entity_text, 'type': entity_type},
                                    search_results.index(result),
                                    1  # Initial frequency count
                                ),
                                'context': result.get('snippet', '')[:200]
                            })
                            seen_entities.add(entity_text)
        
        # Sort by confidence and limit
        entities.sort(key=lambda x: x['confidence'], reverse=True)
        
        return entities[:50]  # Return top 50 web-discovered entities for comprehensive coverage
    
    def _is_common_word(self, word: str) -> bool:
        """Check if a word is too common to be a meaningful entity."""
        common_words = {
            'the', 'and', 'for', 'with', 'this', 'that', 'from',
            'have', 'will', 'can', 'are', 'was', 'were', 'been',
            'their', 'your', 'our', 'its', 'these', 'those',
            'what', 'which', 'when', 'where', 'who', 'how',
            'new', 'latest', 'best', 'top', 'most', 'more',
            'some', 'many', 'few', 'all', 'any', 'every',
            'about', 'over', 'under', 'between', 'through'
        }
        return word.lower() in common_words
    
    def _classify_entity(self, entity_text: str, context: str) -> Optional[str]:
        """
        Classify an entity based on its name and context across ANY domain.
        
        Args:
            entity_text: The entity text
            context: The context where it was found
            
        Returns:
            Entity type or None if not a valid entity
        """
        entity_lower = entity_text.lower()
        context_lower = context.lower()
        
        # Person classification
        if any(title in context_lower for title in ['ceo', 'founder', 'president', 'director', 'manager', 'author', 'professor', 'dr.', 'mr.', 'ms.', 'mrs.']):
            return 'Person'
        elif re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?$', entity_text):  # Name pattern
            if any(person_context in context_lower for person_context in ['said', 'announced', 'wrote', 'created', 'founded', 'leads', 'manages']):
                return 'Person'
        
        # Organization/Company classification
        if any(org_suffix in entity_lower for org_suffix in ['inc', 'corp', 'ltd', 'llc', 'company', 'foundation', 'institute', 'university', 'college']):
            return 'Organization'
        elif any(org_context in context_lower for org_context in ['company', 'corporation', 'organization', 'firm', 'agency', 'department', 'ministry']):
            return 'Organization'
        
        # Location classification
        if any(loc_context in context_lower for loc_context in ['city', 'country', 'state', 'province', 'region', 'capital', 'located in', 'based in']):
            return 'Location'
        elif re.match(r'^[A-Z][a-z]+(?:,\s*[A-Z][a-z]+)?$', entity_text):  # City, State pattern
            return 'Location'
        
        # Event classification
        if any(event_suffix in entity_lower for event_suffix in ['conference', 'summit', 'forum', 'symposium', 'convention', 'expo', 'festival', 'championship', 'games', 'awards']):
            return 'Event'
        elif any(event_context in context_lower for event_context in ['event', 'conference', 'meeting', 'gathering', 'ceremony']):
            return 'Event'
        
        # Product classification
        if any(product_context in context_lower for product_context in ['product', 'release', 'launch', 'version', 'model', 'brand', 'device', 'gadget']):
            return 'Product'
        elif re.match(r'^[A-Z][a-zA-Z]+\s*v?[\d.]+$', entity_text):  # Product with version
            return 'Product'
        
        # Research/Study classification
        if any(research_context in context_lower for research_context in ['research', 'study', 'paper', 'journal', 'publication', 'findings', 'discovery']):
            return 'Research'
        
        # Technology classification (kept for compatibility)
        if 'llm' in entity_lower or 'language model' in context_lower:
            return 'LLMFramework'
        elif 'vector' in context_lower and 'database' in context_lower:
            return 'VectorDatabase'
        elif 'rag' in entity_lower or 'retrieval' in context_lower:
            return 'RAGFramework'
        elif 'agent' in context_lower and any(tech in context_lower for tech in ['ai', 'software', 'system']):
            return 'AgentFramework'
        elif 'database' in context_lower or 'db' in entity_lower:
            return 'Database'
        elif 'framework' in context_lower:
            return 'Framework'
        elif 'library' in context_lower or '.js' in entity_lower or '.py' in entity_lower:
            return 'Library'
        elif 'api' in entity_lower or 'sdk' in entity_lower:
            return 'API'
        elif any(tech_term in entity_lower for tech_term in ['ai', 'ml', 'software', 'platform', 'tool', 'app']):
            return 'Technology'
        
        # Concept/Topic classification (catch-all for important terms)
        elif re.match(r'^[A-Z]', entity_text) and len(entity_text) > 3:
            # Check if it's likely a meaningful concept
            if any(concept_context in context_lower for concept_context in ['concept', 'idea', 'theory', 'principle', 'method', 'approach', 'strategy']):
                return 'Concept'
            # Default to general entity if capitalized and substantial
            return 'Entity'
        
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
    
    async def store_entities_in_neo4j(self, entities: List[Dict], search_metadata: Dict):
        """
        Persist web-discovered entities to Neo4j with WEB_SOURCED label.
        
        Args:
            entities: List of entities to store
            search_metadata: Metadata about the search that discovered these entities
        """
        try:
            from app.services.radiating.storage.radiating_neo4j_service import RadiatingNeo4jService
            
            neo4j_service = RadiatingNeo4jService()
            if not neo4j_service.is_enabled():
                logger.warning("Neo4j is not enabled, skipping entity storage")
                return
            
            timestamp = datetime.now().isoformat()
            
            for entity in entities:
                # Create entity with WEB_SOURCED label
                entity_data = {
                    'text': entity.get('text', ''),
                    'type': entity.get('type', 'Technology'),
                    'discovery_timestamp': timestamp,
                    'source_url': entity.get('url', ''),
                    'confidence_score': entity.get('confidence', 0.5),
                    'discovery_source': 'web_search',
                    'search_query': search_metadata.get('query', ''),
                    'context': entity.get('context', ''),
                    'metadata': json.dumps({
                        'extraction_method': 'web_search_discovery',
                        'search_rank': entities.index(entity),
                        'search_metadata': search_metadata
                    })
                }
                
                # Store in Neo4j with WEB_SOURCED label
                with neo4j_service.driver.session() as session:
                    session.run("""
                        MERGE (e:Entity {name: $text})
                        SET e:WEB_SOURCED,
                            e.type = $type,
                            e.discovery_timestamp = $discovery_timestamp,
                            e.source_url = $source_url,
                            e.confidence_score = $confidence_score,
                            e.discovery_source = $discovery_source,
                            e.search_query = $search_query,
                            e.context = $context,
                            e.metadata = $metadata,
                            e.last_updated = timestamp()
                        RETURN e
                    """, entity_data)
            
            logger.info(f"Stored {len(entities)} web-discovered entities in Neo4j")
            
        except Exception as e:
            logger.error(f"Failed to store entities in Neo4j: {e}")
    
    def extract_relationships_from_snippets(self, search_results: List[Dict]) -> List[Dict]:
        """
        Parse search snippets to identify entity relationships.
        
        Args:
            search_results: List of search results with snippets
            
        Returns:
            List of extracted relationships
        """
        relationships = []
        
        # Relationship patterns to look for
        relationship_patterns = [
            # "X is based on Y" pattern
            (r'([A-Za-z][A-Za-z0-9\s\-_.]+)\s+(?:is|are)\s+based\s+on\s+([A-Za-z][A-Za-z0-9\s\-_.]+)', 'BASED_ON'),
            # "X uses Y" pattern
            (r'([A-Za-z][A-Za-z0-9\s\-_.]+)\s+(?:use|uses|using|utilize|utilizes)\s+([A-Za-z][A-Za-z0-9\s\-_.]+)', 'USES'),
            # "X alternative to Y" pattern
            (r'([A-Za-z][A-Za-z0-9\s\-_.]+)\s+(?:alternative|alternatives)\s+to\s+([A-Za-z][A-Za-z0-9\s\-_.]+)', 'ALTERNATIVE_TO'),
            # "X built on Y" pattern
            (r'([A-Za-z][A-Za-z0-9\s\-_.]+)\s+(?:built|developed)\s+(?:on|with)\s+([A-Za-z][A-Za-z0-9\s\-_.]+)', 'BUILT_ON'),
            # "X integrates with Y" pattern
            (r'([A-Za-z][A-Za-z0-9\s\-_.]+)\s+(?:integrates?|integration)\s+with\s+([A-Za-z][A-Za-z0-9\s\-_.]+)', 'INTEGRATES_WITH'),
            # "X supports Y" pattern
            (r'([A-Za-z][A-Za-z0-9\s\-_.]+)\s+supports?\s+([A-Za-z][A-Za-z0-9\s\-_.]+)', 'SUPPORTS'),
            # "X compatible with Y" pattern
            (r'([A-Za-z][A-Za-z0-9\s\-_.]+)\s+compatible\s+with\s+([A-Za-z][A-Za-z0-9\s\-_.]+)', 'COMPATIBLE_WITH'),
            # "X extends Y" pattern
            (r'([A-Za-z][A-Za-z0-9\s\-_.]+)\s+extends?\s+([A-Za-z][A-Za-z0-9\s\-_.]+)', 'EXTENDS'),
            # "X replaces Y" pattern
            (r'([A-Za-z][A-Za-z0-9\s\-_.]+)\s+replaces?\s+([A-Za-z][A-Za-z0-9\s\-_.]+)', 'REPLACES'),
            # "X powered by Y" pattern
            (r'([A-Za-z][A-Za-z0-9\s\-_.]+)\s+powered\s+by\s+([A-Za-z][A-Za-z0-9\s\-_.]+)', 'POWERED_BY'),
            # "X vs Y" comparison pattern
            (r'([A-Za-z][A-Za-z0-9\s\-_.]+)\s+(?:vs\.?|versus)\s+([A-Za-z][A-Za-z0-9\s\-_.]+)', 'COMPARES_TO'),
            # "X fork of Y" pattern
            (r'([A-Za-z][A-Za-z0-9\s\-_.]+)\s+(?:fork|forked)\s+(?:of|from)\s+([A-Za-z][A-Za-z0-9\s\-_.]+)', 'FORK_OF')
        ]
        
        seen_relationships = set()
        
        for result in search_results:
            text = f"{result.get('title', '')} {result.get('snippet', '')}"
            
            for pattern, rel_type in relationship_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    source = match.group(1).strip()
                    target = match.group(2).strip()
                    
                    # Filter out common words and validate entities
                    if (len(source) > 2 and len(target) > 2 and
                        not self._is_common_word(source) and
                        not self._is_common_word(target)):
                        
                        # Create unique key to avoid duplicates
                        rel_key = f"{source.lower()}_{rel_type}_{target.lower()}"
                        if rel_key not in seen_relationships:
                            relationships.append({
                                'source': source,
                                'target': target,
                                'type': rel_type,
                                'confidence': 0.6,  # Moderate confidence for snippet-extracted relationships
                                'context': match.group(0),
                                'source_url': result.get('url', ''),
                                'extraction_method': 'snippet_pattern_matching'
                            })
                            seen_relationships.add(rel_key)
        
        return relationships
    
    def calculate_entity_confidence(self, entity: Dict, search_rank: int, frequency: int) -> float:
        """
        Score entities based on search ranking and frequency.
        
        Args:
            entity: The entity dictionary
            search_rank: Position in search results (0-based)
            frequency: Number of times entity appears
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence starts at 0.5
        confidence = 0.5
        
        # Boost based on search ranking (higher rank = higher confidence)
        # First result gets +0.3, second +0.25, third +0.2, etc.
        rank_boost = max(0, 0.3 - (search_rank * 0.05))
        confidence += rank_boost
        
        # Boost based on frequency (more mentions = higher confidence)
        # Each additional mention adds +0.05, max +0.2
        frequency_boost = min(0.2, (frequency - 1) * 0.05)
        confidence += frequency_boost
        
        # Boost for specific entity types that are more reliable
        entity_type = entity.get('type', '')
        if entity_type in ['LLMFramework', 'VectorDatabase', 'RAGFramework', 'AgentFramework']:
            confidence += 0.1
        
        # Cap at 1.0
        return min(1.0, confidence)