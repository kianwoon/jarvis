"""
Search Query Optimizer

Intelligently optimizes user queries for search tools using LLM-based query enhancement.
Follows established codebase patterns for LLM calls, configuration, and caching.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SearchQueryOptimizer:
    """
    LLM-based search query optimizer that transforms conversational queries
    into optimized search terms using existing system patterns.
    """
    
    def __init__(self):
        """Initialize the search query optimizer"""
        self.optimization_cache = {}  # Cache successful optimizations
    
    async def optimize_query(self, user_query: str) -> str:
        """
        Transform user's conversational question into an optimized search query
        
        Args:
            user_query: The original user question
            
        Returns:
            Optimized search query string, falls back to original if optimization fails
        """
        if not user_query or not user_query.strip():
            return user_query
            
        # Check cache first
        cache_key = user_query.strip().lower()
        if cache_key in self.optimization_cache:
            logger.debug(f"Using cached optimization for: {user_query}")
            return self.optimization_cache[cache_key]
        
        try:
            # Get search optimization configuration
            from app.core.llm_settings_cache import get_search_optimization_config
            config = get_search_optimization_config()
            
            # Check if optimization is enabled
            if not config.get('enable_search_optimization', True):
                logger.debug("Search query optimization disabled, returning original query")
                return user_query
            
            # Get optimization prompt and timeout
            optimization_prompt_template = config.get('optimization_prompt', '')
            if not optimization_prompt_template:
                logger.warning("No search optimization prompt configured, returning original query")
                return user_query
            
            timeout_seconds = config.get('optimization_timeout', 8)
            
            # Build the optimization prompt
            optimization_prompt = optimization_prompt_template.format(query=user_query)
            
            # Call LLM for optimization using existing patterns
            optimized_query = await self._call_llm_for_optimization(
                optimization_prompt, 
                timeout_seconds
            )
            
            if optimized_query and optimized_query.strip():
                # Clean and validate the optimized query
                cleaned_query = self._clean_optimized_query(optimized_query)
                if cleaned_query and len(cleaned_query.strip()) > 0:
                    # Cache the successful optimization
                    self.optimization_cache[cache_key] = cleaned_query
                    logger.info(f"Search query optimized: '{user_query}' → '{cleaned_query}'")
                    return cleaned_query
            
            # Fallback to original query if optimization failed
            logger.debug("Search query optimization produced no result, using original query")
            return user_query
            
        except Exception as e:
            logger.warning(f"Search query optimization failed: {e}, using original query")
            return user_query
    
    async def _call_llm_for_optimization(self, prompt: str, timeout_seconds: int) -> Optional[str]:
        """
        Call LLM for search query optimization using existing patterns
        
        Args:
            prompt: The optimization prompt
            timeout_seconds: Timeout for the LLM call
            
        Returns:
            Optimized query string or None if failed
        """
        try:
            import asyncio
            
            # Use direct Ollama LLM call to avoid circular imports
            from app.llm.ollama import JarvisLLM
            from app.core.llm_settings_cache import get_llm_settings, get_second_llm_full_config
            
            # Get LLM settings using established pattern
            llm_settings = get_llm_settings()
            
            # Use second_llm config for optimization (following tool planner pattern)
            llm_config = get_second_llm_full_config(llm_settings)
            
            if not llm_config or not llm_config.get('model'):
                logger.error("LLM configuration not available for search optimization")
                return None
            
            logger.debug(f"Using LLM model for search optimization: {llm_config.get('model')}")
            
            # Create JarvisLLM instance with second_llm config
            base_url = llm_config.get('model_server', 'http://localhost:11434')
            
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
                    mode='non-thinking',  # Use non-thinking mode for structured optimization
                    max_tokens=llm_config.get('max_tokens'),
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
    return await optimizer.optimize_query(user_query)