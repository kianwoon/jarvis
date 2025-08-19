"""
Enhanced Search Result Formatter for Rich Google Search Metadata

This module provides utilities to format Google Search results with rich metadata
including HTML snippets, formatted URLs, thumbnails, and publication dates.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from html import unescape
import re

logger = logging.getLogger(__name__)


class EnhancedSearchResultFormatter:
    """Formats Google Search results with rich metadata for better display and processing"""
    
    @staticmethod
    def extract_publication_date(result: Dict[str, Any]) -> Optional[str]:
        """
        Extract publication date from various metadata fields
        
        Priority order:
        1. pagemap.metatags.article:published_time
        2. pagemap.metatags.datePublished
        3. pagemap.metatags.og:updated_time
        4. pagemap.newsarticle.datePublished
        5. Date patterns in snippet
        """
        try:
            # Check pagemap metadata
            pagemap = result.get('pagemap', {})
            
            # Check metatags
            metatags = pagemap.get('metatags', [])
            if metatags and isinstance(metatags, list):
                meta = metatags[0] if metatags else {}
                
                # Try various date fields
                date_fields = [
                    'article:published_time',
                    'datePublished',
                    'og:updated_time',
                    'article:modified_time',
                    'publishedDate',
                    'date'
                ]
                
                for field in date_fields:
                    if field in meta:
                        return meta[field]
            
            # Check news article metadata
            newsarticle = pagemap.get('newsarticle', [])
            if newsarticle and isinstance(newsarticle, list):
                article = newsarticle[0] if newsarticle else {}
                if 'datePublished' in article:
                    return article['datePublished']
            
            # Try to extract date from snippet using regex
            snippet = result.get('snippet', '')
            # Common date patterns: "Jan 15, 2025", "2025-01-15", "15/01/2025"
            date_patterns = [
                r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b',
                r'\b(\d{4}-\d{2}-\d{2})\b',
                r'\b(\d{1,2}/\d{1,2}/\d{4})\b'
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, snippet, re.IGNORECASE)
                if match:
                    return match.group(1)
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting publication date: {e}")
            return None
    
    @staticmethod
    def extract_thumbnail(result: Dict[str, Any]) -> Optional[str]:
        """Extract thumbnail URL from pagemap if available"""
        try:
            pagemap = result.get('pagemap', {})
            
            # Check for CSE thumbnail
            cse_thumbnail = pagemap.get('cse_thumbnail', [])
            if cse_thumbnail and isinstance(cse_thumbnail, list):
                thumb = cse_thumbnail[0] if cse_thumbnail else {}
                if 'src' in thumb:
                    return thumb['src']
            
            # Check metatags for og:image
            metatags = pagemap.get('metatags', [])
            if metatags and isinstance(metatags, list):
                meta = metatags[0] if metatags else {}
                if 'og:image' in meta:
                    return meta['og:image']
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting thumbnail: {e}")
            return None
    
    @staticmethod
    def format_html_snippet(result: Dict[str, Any]) -> str:
        """
        Format HTML snippet with proper text extraction and highlighting preservation
        """
        html_snippet = result.get('htmlSnippet', '')
        regular_snippet = result.get('snippet', '')
        
        if html_snippet:
            # Convert HTML entities and preserve search term highlighting
            # <b> tags indicate search term matches
            formatted = html_snippet.replace('<b>', '**').replace('</b>', '**')
            # Remove other HTML tags but keep the text
            formatted = re.sub(r'<[^>]+>', '', formatted)
            # Unescape HTML entities
            formatted = unescape(formatted)
            return formatted
        
        return regular_snippet
    
    @staticmethod
    def format_for_llm(results: List[Dict[str, Any]]) -> str:
        """
        Format search results for LLM synthesis with rich metadata
        
        Args:
            results: List of Google search result dictionaries
            
        Returns:
            Formatted string for LLM processing
        """
        if not results:
            return "No search results found."
        
        formatted_results = []
        formatter = EnhancedSearchResultFormatter()
        
        for i, result in enumerate(results, 1):
            # Extract all metadata
            title = result.get('title', 'Untitled')
            link = result.get('link', '')
            display_link = result.get('displayLink', '')
            formatted_url = result.get('formattedUrl', link)
            snippet = formatter.format_html_snippet(result)
            pub_date = formatter.extract_publication_date(result)
            thumbnail = formatter.extract_thumbnail(result)
            
            # Build formatted result
            result_text = f"**Result {i}:**\n"
            result_text += f"**Title:** {title}\n"
            
            # Use display link for cleaner presentation
            if display_link:
                result_text += f"**Source:** {display_link}\n"
            
            result_text += f"**URL:** {formatted_url}\n"
            
            # Add publication date if available
            if pub_date:
                result_text += f"**Published:** {pub_date}\n"
            
            # Add snippet with potential highlighting
            result_text += f"**Summary:** {snippet}\n"
            
            # Note if thumbnail is available (for frontend use)
            if thumbnail:
                result_text += f"**[Has thumbnail image]**\n"
            
            # Add metadata from pagemap if relevant
            pagemap = result.get('pagemap', {})
            metatags = pagemap.get('metatags', [])
            if metatags and isinstance(metatags, list):
                meta = metatags[0] if metatags else {}
                
                # Add author if available
                author = meta.get('author') or meta.get('article:author')
                if author:
                    result_text += f"**Author:** {author}\n"
                
                # Add description if different from snippet
                description = meta.get('description') or meta.get('og:description')
                if description and description != snippet and len(description) < 500:
                    result_text += f"**Description:** {description}\n"
            
            formatted_results.append(result_text)
        
        return "\n---\n".join(formatted_results)
    
    @staticmethod
    def format_for_display(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format search results for frontend display with enhanced metadata
        
        Args:
            results: List of Google search result dictionaries
            
        Returns:
            List of enhanced result dictionaries for display
        """
        formatted_results = []
        formatter = EnhancedSearchResultFormatter()
        
        for result in results:
            enhanced_result = {
                'title': result.get('title', 'Untitled'),
                'link': result.get('link', ''),
                'displayLink': result.get('displayLink', ''),
                'formattedUrl': result.get('formattedUrl', result.get('link', '')),
                'snippet': formatter.format_html_snippet(result),
                'originalSnippet': result.get('snippet', ''),
                'htmlSnippet': result.get('htmlSnippet', ''),
                'publicationDate': formatter.extract_publication_date(result),
                'thumbnail': formatter.extract_thumbnail(result),
                'metadata': {}
            }
            
            # Extract additional metadata
            pagemap = result.get('pagemap', {})
            metatags = pagemap.get('metatags', [])
            if metatags and isinstance(metatags, list):
                meta = metatags[0] if metatags else {}
                
                enhanced_result['metadata'] = {
                    'author': meta.get('author') or meta.get('article:author'),
                    'description': meta.get('description') or meta.get('og:description'),
                    'type': meta.get('og:type'),
                    'siteName': meta.get('og:site_name'),
                    'keywords': meta.get('keywords')
                }
            
            # Add temporal relevance placeholder
            enhanced_result['temporalRelevance'] = {
                'hasDate': bool(enhanced_result['publicationDate']),
                'dateString': enhanced_result['publicationDate']
            }
            
            formatted_results.append(enhanced_result)
        
        return formatted_results
    
    @staticmethod
    def extract_dates_for_temporal_scoring(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract and parse dates from search results for temporal relevance scoring
        
        Args:
            results: List of Google search result dictionaries
            
        Returns:
            List of results with parsed date information
        """
        formatter = EnhancedSearchResultFormatter()
        results_with_dates = []
        
        for result in results:
            date_str = formatter.extract_publication_date(result)
            parsed_date = None
            age_days = None
            
            if date_str:
                # Try to parse the date string
                date_formats = [
                    '%Y-%m-%dT%H:%M:%S%z',  # ISO format with timezone
                    '%Y-%m-%dT%H:%M:%SZ',   # ISO format UTC
                    '%Y-%m-%d',              # Simple date
                    '%B %d, %Y',             # January 15, 2025
                    '%b %d, %Y',             # Jan 15, 2025
                    '%d/%m/%Y',              # 15/01/2025
                    '%m/%d/%Y',              # 01/15/2025
                ]
                
                for fmt in date_formats:
                    try:
                        parsed_date = datetime.strptime(date_str.replace(',', ''), fmt)
                        if not parsed_date.tzinfo:
                            parsed_date = parsed_date.replace(tzinfo=timezone.utc)
                        break
                    except:
                        continue
                
                # Calculate age in days
                if parsed_date:
                    now = datetime.now(timezone.utc)
                    age_days = (now - parsed_date).days
            
            results_with_dates.append({
                **result,
                'extracted_date': date_str,
                'parsed_date': parsed_date.isoformat() if parsed_date else None,
                'age_days': age_days
            })
        
        return results_with_dates


def format_search_results_for_llm(tool_name: str, tool_result: Any) -> str:
    """
    Main entry point for formatting search results from MCP tools
    
    Args:
        tool_name: Name of the tool (e.g., 'google_search')
        tool_result: Raw result from the MCP tool
        
    Returns:
        Formatted string for LLM synthesis
    """
    formatter = EnhancedSearchResultFormatter()
    
    try:
        # Handle different result formats
        if isinstance(tool_result, dict):
            # Check for content field (MCP response format)
            if 'content' in tool_result:
                content = tool_result.get('content', [])
                if content and isinstance(content, list):
                    for item in content:
                        if item.get('type') == 'text':
                            text_content = item.get('text', '')
                            # Try to parse as JSON
                            if text_content.startswith('[') or text_content.startswith('{'):
                                try:
                                    results = json.loads(text_content)
                                    if isinstance(results, list):
                                        return formatter.format_for_llm(results)
                                except json.JSONDecodeError:
                                    pass
                            return text_content
            
            # Check for direct results array
            elif 'results' in tool_result:
                results = tool_result.get('results', [])
                if isinstance(results, list):
                    return formatter.format_for_llm(results)
            
            # Check if it's a direct array of results
            elif isinstance(tool_result, list):
                return formatter.format_for_llm(tool_result)
        
        # Fallback to JSON representation
        return json.dumps(tool_result, indent=2) if isinstance(tool_result, (dict, list)) else str(tool_result)
        
    except Exception as e:
        logger.error(f"Error formatting search results: {e}")
        return str(tool_result)