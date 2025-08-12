"""
Search Query Optimizer

Intelligently optimizes user queries for search tools using LLM-based query enhancement.
Follows established codebase patterns for LLM calls, configuration, and caching.
"""

import logging
import re
from typing import Optional, List, Dict, Any, Set

logger = logging.getLogger(__name__)


class SearchQueryOptimizer:
    """
    LLM-based search query optimizer that transforms conversational queries
    into optimized search terms using existing system patterns.
    """
    
    def __init__(self):
        """Initialize the search query optimizer"""
        self.optimization_cache = {}  # Cache successful optimizations
        # Patterns for identifying entities that should be preserved
        self.entity_patterns = [
            r'\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\b',  # Proper nouns
            r'\b(?:GPT|AI|ML|API|SDK|PRO|PLUS|Plus|Pro|Premium|Enterprise|Team)\b',  # Common tech/product terms
            r'\b\d+(?:\.\d+)?\b',  # Version numbers
            r'"[^"]+"',  # Quoted terms
            r"'[^']+'",  # Single quoted terms
            r'\b[A-Z]{2,}\b',  # Acronyms
        ]
    
    def extract_protected_entities(self, query: str) -> List[str]:
        """Extract entities that should not be modified during optimization"""
        entities = set()
        
        # Extract based on patterns
        for pattern in self.entity_patterns:
            try:
                matches = re.findall(pattern, query)
                entities.update(matches)
            except Exception as e:
                logger.debug(f"Pattern matching failed for {pattern}: {e}")
        
        # Extract capitalized multi-word terms (like "ChatGPT PRO")
        words = query.split()
        for i, word in enumerate(words):
            if len(word) > 1 and word[0].isupper():
                # Check for multi-word proper nouns
                phrase = [word]
                for j in range(i+1, min(i+4, len(words))):
                    if words[j][0].isupper() or words[j].lower() in ['pro', 'plus', 'premium', 'enterprise', 'team']:
                        phrase.append(words[j])
                    else:
                        break
                if len(phrase) > 1:
                    entities.add(' '.join(phrase))
                elif word not in ['What', 'How', 'When', 'Where', 'Why', 'Who', 'Which', 'Can', 'Does', 'Is', 'Are']:
                    entities.add(word)
        
        return list(entities)
    
    def calculate_optimization_confidence(self, query: str) -> float:
        """Calculate confidence score for whether to optimize this query"""
        confidence = 1.0
        
        # Reduce confidence for queries with many entities
        entities = self.extract_protected_entities(query)
        if len(entities) > 2:
            confidence *= 0.5
        elif len(entities) > 1:
            confidence *= 0.7
        
        # Reduce confidence for technical queries
        tech_indicators = ['API', 'SDK', 'config', 'error', 'bug', 'version', 'documentation', 'install']
        if any(ind.lower() in query.lower() for ind in tech_indicators):
            confidence *= 0.6
        
        # Reduce confidence for specific product/tier queries
        if re.search(r'\b\w+\s+(PRO|Pro|Plus|PLUS|Premium|Enterprise|Team|Free)\b', query, re.I):
            confidence *= 0.5
        
        # Reduce confidence for comparison queries
        if any(word in query.lower() for word in ['difference', 'vs', 'versus', 'compare', 'between']):
            confidence *= 0.4
        
        return max(0.0, min(1.0, confidence))
    
    def validate_optimization(self, original: str, optimized: str) -> bool:
        """Validate that optimization preserves critical information"""
        # Extract entities from both
        original_entities = set(self.extract_protected_entities(original))
        
        # Check if critical entities were removed or modified
        for entity in original_entities:
            # Check exact match first
            if entity not in optimized:
                # Check case-insensitive match for non-product terms
                if entity.lower() not in optimized.lower():
                    # Special check for product terms that must match exactly
                    if any(term in entity for term in ['PRO', 'Plus', 'Premium', 'Enterprise', 'Team']):
                        logger.warning(f"Optimization removed/changed critical product term: {entity}")
                        return False
                    # For other entities, warn but don't fail
                    logger.debug(f"Optimization removed entity: {entity}")
        
        # Check word overlap to ensure query hasn't changed too much
        original_words = set(original.lower().split())
        optimized_words = set(optimized.lower().split())
        if original_words:
            overlap = len(original_words & optimized_words) / len(original_words)
            if overlap < 0.3:
                logger.warning(f"Optimization changed too much: {overlap:.2f} word overlap")
                return False
        
        return True
    
    def light_optimization(self, query: str, entities: List[str]) -> str:
        """Perform light optimization that removes filler words and adds temporal context"""
        from datetime import datetime
        
        optimized = query
        
        # Remove question mark at the end for search
        optimized = optimized.rstrip('?')
        
        # Remove filler words but preserve entities
        filler_phrases = [
            'can you tell me', 'could you tell me', 'please tell me',
            'i want to know', 'i need to know', 'i would like to know',
            'what is the', 'what are the', 'how do i', 'how can i',
            'please help me', 'can you help me'
        ]
        
        lower_optimized = optimized.lower()
        for filler in filler_phrases:
            if filler in lower_optimized:
                # Only remove if it doesn't contain any protected entities
                if not any(entity.lower() in filler for entity in entities):
                    # Use regex to preserve case in the rest of the query
                    pattern = re.compile(re.escape(filler), re.IGNORECASE)
                    optimized = pattern.sub('', optimized).strip()
                    lower_optimized = optimized.lower()
        
        # Clean up extra spaces
        optimized = ' '.join(optimized.split())
        
        # Add temporal context for current information queries
        query_lower = optimized.lower()
        current_year = str(datetime.now().year)
        
        # Check if query asks for current/latest information
        needs_year = any(term in query_lower for term in [
            'current', 'latest', 'now', 'today', 'recent', 'new'
        ])
        
        # Check if it's about subscriptions/pricing/limits (often changes over time)
        is_time_sensitive = any(term in query_lower for term in [
            'subscription', 'pricing', 'price', 'cost', 'limit', 'usage', 'tier', 'plan'
        ])
        
        # Add current year if appropriate and not already present
        if (needs_year or is_time_sensitive) and current_year not in query_lower:
            # Don't add year if query explicitly mentions a different year
            has_other_year = any(str(year) in query_lower for year in range(2020, 2030))
            if not has_other_year:
                optimized = f"{optimized} {current_year}"
                logger.debug(f"Added current year to query: {optimized}")
        
        return optimized
    
    async def optimize_query(self, user_query: str) -> Dict[str, Any]:
        """
        Transform user's conversational question into an optimized search query with metadata
        
        Args:
            user_query: The original user question
            
        Returns:
            Dict containing original query, optimized query, confidence, method, and preserved entities
        """
        result = {
            'original': user_query,
            'optimized': user_query,
            'confidence': 0.0,
            'method': 'none',
            'entities_preserved': []
        }
        
        if not user_query or not user_query.strip():
            return result
            
        # Calculate confidence and extract entities first
        confidence = self.calculate_optimization_confidence(user_query)
        entities = self.extract_protected_entities(user_query)
        
        result['confidence'] = confidence
        result['entities_preserved'] = entities
        
        # Skip optimization if confidence is too low
        if confidence < 0.3:
            logger.debug(f"Skipping optimization due to low confidence: {confidence:.2f}")
            result['method'] = 'skipped_low_confidence'
            result['optimized'] = user_query
            return result
        
        # Check cache for high-confidence optimizations
        cache_key = user_query.strip().lower()
        if confidence > 0.7 and cache_key in self.optimization_cache:
            logger.debug(f"Using cached optimization for: {user_query}")
            cached = self.optimization_cache[cache_key]
            result['optimized'] = cached
            result['method'] = 'cached'
            return result
        
        try:
            # Get search optimization configuration
            from app.core.llm_settings_cache import get_search_optimization_full_config
            config = get_search_optimization_full_config()
            
            # Check if optimization is enabled
            if not config.get('enable_search_optimization', True):
                logger.debug("Search query optimization disabled, returning original query")
                return user_query
            
            # Build enhanced optimization prompt with entity preservation
            optimization_prompt = self.build_enhanced_prompt(user_query, entities, config)
            timeout_seconds = config.get('optimization_timeout', 8)
            
            # Call LLM for optimization using existing patterns
            optimized_query = await self._call_llm_for_optimization(
                optimization_prompt, 
                timeout_seconds,
                config
            )
            
            if optimized_query and optimized_query.strip():
                # Clean and validate the optimized query
                cleaned_query = self._clean_optimized_query(optimized_query)
                if cleaned_query and len(cleaned_query.strip()) > 0:
                    # Validate that optimization preserves intent
                    if self.validate_optimization(user_query, cleaned_query):
                        # Cache the successful optimization for high confidence
                        if confidence > 0.7:
                            self.optimization_cache[cache_key] = cleaned_query
                        logger.info(f"Search query optimized: '{user_query}' → '{cleaned_query}'")
                        result['optimized'] = cleaned_query
                        result['method'] = 'llm_validated'
                        return result
                    else:
                        # Optimization failed validation, use light optimization
                        logger.debug("LLM optimization failed validation, using light optimization")
                        light_optimized = self.light_optimization(user_query, entities)
                        result['optimized'] = light_optimized
                        result['method'] = 'light_fallback'
                        return result
            
            # Fallback to light optimization if LLM failed
            logger.debug("LLM optimization failed, using light optimization")
            light_optimized = self.light_optimization(user_query, entities)
            result['optimized'] = light_optimized
            result['method'] = 'light_only'
            return result
            
        except Exception as e:
            logger.warning(f"Search query optimization failed: {e}, using light optimization")
            # Try light optimization as final fallback
            try:
                light_optimized = self.light_optimization(user_query, entities)
                result['optimized'] = light_optimized
                result['method'] = 'error_fallback'
            except:
                result['optimized'] = user_query
                result['method'] = 'error'
            return result
    
    async def _call_llm_for_optimization(self, prompt: str, timeout_seconds: int, config: dict) -> Optional[str]:
        """
        Call LLM for search query optimization using search_optimization configuration
        
        Args:
            prompt: The optimization prompt
            timeout_seconds: Timeout for the LLM call
            config: Search optimization configuration dictionary
            
        Returns:
            Optimized query string or None if failed
        """
        try:
            import asyncio
            
            # Use direct Ollama LLM call to avoid circular imports
            from app.llm.ollama import JarvisLLM
            
            # Use search_optimization config directly
            if not config or not config.get('model'):
                logger.error("Search optimization configuration not available or missing model")
                return None
            
            logger.debug(f"Using LLM model for search optimization: {config.get('model')}")
            
            # Get model server from search_optimization config
            base_url = config.get('model_server', '')
            
            if not base_url:
                logger.error("No model server configured in search_optimization settings")
                raise ValueError("Model server must be configured in search_optimization settings")
            
            # Convert localhost to host.docker.internal for Docker containers (following existing pattern)
            if "localhost" in base_url:
                import os
                is_docker = os.path.exists('/root') or os.environ.get('DOCKER_ENVIRONMENT')
                if is_docker:
                    base_url = base_url.replace("localhost", "host.docker.internal")
                    logger.debug(f"Docker environment detected, using Docker URL: {base_url}")
            
            # Make LLM call with timeout - create instance just before use to minimize resource lifetime
            try:
                jarvis_llm = JarvisLLM(
                    mode=config.get('mode', 'non-thinking'),  # Use search optimization mode
                    max_tokens=config.get('max_tokens', 50),
                    base_url=base_url
                )
                
                response = await asyncio.wait_for(
                    jarvis_llm.llm.generate(prompt),
                    timeout=timeout_seconds
                )
                
                # Extract text from LLMResponse object
                response_text = response.text if hasattr(response, 'text') else str(response)
                logger.debug(f"LLM search optimization response: '{response_text[:100]}...'")
                return response_text
                
            except asyncio.TimeoutError:
                logger.warning(f"Search query optimization timed out after {timeout_seconds} seconds")
                return None
            finally:
                # Clear reference to allow garbage collection of any resources
                jarvis_llm = None
                
        except Exception as e:
            logger.error(f"Failed to call LLM for search query optimization: {e}")
            return None
    
    def build_enhanced_prompt(self, query: str, entities: List[str], config: dict) -> str:
        """Build an enhanced optimization prompt that preserves entities"""
        # Get base prompt template
        prompt_template = config.get('optimization_prompt', '')
        
        # If no template, create a robust default
        if not prompt_template:
            prompt_template = """Optimize this search query while preserving ALL proper nouns, product names, version numbers, and technical terms EXACTLY as written.

CRITICAL RULES:
1. NEVER change product names (e.g., "ChatGPT PRO" must stay "ChatGPT PRO", not "ChatGPT Plus")
2. NEVER change version numbers or technical specifications
3. NEVER substitute similar-sounding products or services
4. Preserve ALL capitalized terms and proper nouns exactly as written
5. Only optimize grammar and remove unnecessary filler words

Query: {query}

Return ONLY the optimized query:"""
        
        # If we have entities, enhance the prompt with specific preservation instructions
        if entities:
            entities_str = ', '.join([f'"{e}"' for e in entities])
            enhanced_prompt = f"""Optimize this search query while preserving user intent and critical terms.

ORIGINAL QUERY: {query}

CRITICAL TERMS TO PRESERVE EXACTLY: {entities_str}

RULES:
1. NEVER change the critical terms listed above
2. NEVER change product names (e.g., "ChatGPT PRO" stays "ChatGPT PRO", not "ChatGPT Plus")
3. NEVER substitute similar products (e.g., "GPT-4" stays "GPT-4", not "ChatGPT")
4. Only optimize by removing filler words and improving grammar
5. Keep the search intent exactly the same

Return ONLY the optimized query, nothing else:"""
            return enhanced_prompt
        else:
            # Use the template with the query
            return prompt_template.format(query=query)
    
    def _clean_optimized_query(self, raw_query: str) -> str:
        """
        Clean and validate the optimized query from LLM response
        
        Args:
            raw_query: Raw response from LLM
            
        Returns:
            Cleaned and validated query string
        """
        if not raw_query:
            return ""
        
        # Remove any thinking tags if present (following existing patterns)
        import re
        cleaned = raw_query
        if '<think>' in cleaned:
            cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL).strip()
        
        # Remove any explanation text - take first line only
        lines = cleaned.split('\n')
        if lines:
            cleaned = lines[0].strip()
        
        # Remove quotes if the entire query is quoted
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1].strip()
        elif cleaned.startswith("'") and cleaned.endswith("'"):
            cleaned = cleaned[1:-1].strip()
        
        # Remove common prefixes that LLM might add
        prefixes_to_remove = [
            "optimized query:",
            "search query:",
            "search:",
            "query:",
            "result:",
            "answer:"
        ]
        
        cleaned_lower = cleaned.lower()
        for prefix in prefixes_to_remove:
            if cleaned_lower.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break
        
        # Basic validation - ensure it's not too short or empty
        if len(cleaned.strip()) < 2:
            logger.debug(f"Optimized query too short: '{cleaned}'")
            return ""
        
        # Limit length to reasonable search query size
        if len(cleaned) > 200:
            cleaned = cleaned[:200].strip()
            logger.debug("Truncated long optimized query to 200 chars")
        
        return cleaned
    
    def clear_cache(self):
        """Clear the optimization cache"""
        self.optimization_cache.clear()
        logger.info("Search query optimization cache cleared")


# Global instance following singleton pattern used elsewhere in codebase
_search_query_optimizer = None


def get_search_query_optimizer() -> SearchQueryOptimizer:
    """Get singleton instance of SearchQueryOptimizer"""
    global _search_query_optimizer
    if _search_query_optimizer is None:
        _search_query_optimizer = SearchQueryOptimizer()
    return _search_query_optimizer


async def optimize_search_query(user_query: str) -> str:
    """
    Convenience function to optimize a search query
    
    Args:
        user_query: The original user question
        
    Returns:
        Optimized search query, falls back to original if optimization fails
    """
    optimizer = get_search_query_optimizer()
    result = await optimizer.optimize_query(user_query)
    # For backward compatibility, return just the optimized query string
    if isinstance(result, dict):
        return result.get('optimized', user_query)
    return result