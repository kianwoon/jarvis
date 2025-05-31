"""
Focused query analyzer for entity-centric RAG search
"""
import re
from typing import Dict, List, Tuple, Optional

class FocusedQueryAnalyzer:
    """Analyzes queries to identify core entities and their aspects"""
    
    def __init__(self):
        # Patterns for entity-aspect extraction
        self.entity_aspect_patterns = [
            # "how's/how is X Y look like" -> entity: X, aspect: Y
            (r"how'?s?\s+(\w+)\s+([\w\s]+?)\s+(?:look|looking|looks)", 
             lambda m: {'entity': m.group(1), 'aspect': m.group(2).strip()}),
            
            # "what is X's Y" -> entity: X, aspect: Y
            (r"what\s+is\s+(\w+)'?s?\s+([\w\s]+?)(?:\?|$)", 
             lambda m: {'entity': m.group(1), 'aspect': m.group(2).strip()}),
            
            # "tell me about X's Y" -> entity: X, aspect: Y
            (r"tell\s+(?:me\s+)?about\s+(\w+)'?s?\s+([\w\s]+?)(?:\?|$)", 
             lambda m: {'entity': m.group(1), 'aspect': m.group(2).strip()}),
            
            # "X Y information/details/data" -> entity: X, aspect: Y
            (r"(\w+)\s+([\w\s]+?)\s+(?:information|details|data)(?:\?|$)", 
             lambda m: {'entity': m.group(1), 'aspect': m.group(2).strip()}),
            
            # "describe X Y" -> entity: X, aspect: Y
            (r"describe\s+(\w+)\s+([\w\s]+?)(?:\?|$)", 
             lambda m: {'entity': m.group(1), 'aspect': m.group(2).strip()}),
            
            # "X architecture/structure/organization" -> entity: X, aspect: Y
            (r"(\w+)\s+(architecture|structure|organization|organisation|system|platform|model)(?:\s|$)", 
             lambda m: {'entity': m.group(1), 'aspect': m.group(2)}),
        ]
        
        # Aspect keywords that indicate specific information types
        self.aspect_keywords = {
            'organization': ['organization', 'organisation', 'structure', 'hierarchy', 'departments', 'divisions'],
            'financial': ['revenue', 'profit', 'earnings', 'financial', 'income', 'sales'],
            'leadership': ['ceo', 'leadership', 'management', 'executives', 'board', 'directors'],
            'products': ['products', 'services', 'offerings', 'portfolio'],
            'history': ['history', 'founded', 'established', 'timeline', 'evolution'],
            'technology': ['technology', 'tech', 'platform', 'architecture', 'system'],
            'strategy': ['strategy', 'vision', 'mission', 'goals', 'plan'],
            'partnership': ['partnership', 'partners', 'collaboration', 'alliance', 'joint']
        }
        
    def analyze(self, question: str) -> Dict[str, any]:
        """
        Analyze query to extract core entity and aspects
        Returns: {
            'core_entity': str,
            'aspects': List[str],
            'aspect_category': str,
            'search_strategy': str,
            'confidence': float
        }
        """
        question_lower = question.lower().strip()
        
        # Try entity-aspect patterns
        for pattern, extractor in self.entity_aspect_patterns:
            match = re.search(pattern, question_lower, re.IGNORECASE)
            if match:
                try:
                    result = extractor(match)
                    entity = result['entity']
                    aspect = result['aspect']
                    
                    # Categorize the aspect
                    aspect_category = self._categorize_aspect(aspect)
                    
                    return {
                        'core_entity': entity,
                        'aspects': [aspect],
                        'aspect_category': aspect_category,
                        'search_strategy': 'entity_focused',
                        'confidence': 0.9,
                        'original_query': question
                    }
                except:
                    continue
        
        # Fallback: Try to identify just the entity
        entity = self._extract_entity_fallback(question)
        if entity:
            return {
                'core_entity': entity,
                'aspects': [],
                'aspect_category': 'general',
                'search_strategy': 'entity_broad',
                'confidence': 0.6,
                'original_query': question
            }
        
        # No clear entity found
        return {
            'core_entity': None,
            'aspects': self._extract_general_topics(question),
            'aspect_category': 'general',
            'search_strategy': 'general',
            'confidence': 0.3,
            'original_query': question
        }
    
    def _categorize_aspect(self, aspect: str) -> str:
        """Categorize the aspect into predefined categories"""
        aspect_lower = aspect.lower()
        
        for category, keywords in self.aspect_keywords.items():
            if any(keyword in aspect_lower for keyword in keywords):
                return category
        
        return 'general'
    
    def _extract_entity_fallback(self, question: str) -> Optional[str]:
        """Try to extract entity using simpler patterns"""
        # Look for capitalized words not at sentence start
        words = question.split()
        for i, word in enumerate(words):
            if i > 0 and word[0].isupper() and word.lower() not in {'i', 'the', 'what', 'how', 'when', 'where', 'why'}:
                return word.lower()
        
        # Look for words before certain keywords
        patterns = [
            r"about\s+(\w+)",
            r"(?:what|how|when|where)\s+(?:is|are|was|were)\s+(\w+)",
            r"(\w+)\s+(?:company|corporation|organization|firm)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question.lower())
            if match:
                return match.group(1)
        
        return None
    
    def _extract_general_topics(self, question: str) -> List[str]:
        """Extract general topic words when no clear entity is found"""
        stop_words = {
            'the', 'is', 'are', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'a', 'an', 'what', 'how', 'when', 'where', 'why',
            'can', 'could', 'would', 'should', 'tell', 'show', 'find', 'get'
        }
        
        words = re.findall(r'\w+', question.lower())
        topics = []
        
        for word in words:
            if len(word) > 3 and word not in stop_words:
                topics.append(word)
        
        return topics[:3]  # Limit to top 3 topics
    
    def get_search_params(self, analysis: Dict[str, any]) -> Dict[str, any]:
        """
        Convert analysis into search parameters
        """
        if analysis['search_strategy'] == 'entity_focused':
            return {
                'primary_terms': [analysis['core_entity']],
                'secondary_terms': analysis['aspects'],
                'boost_primary': True,
                'search_in_source': True,
                'min_relevance': 0.7
            }
        elif analysis['search_strategy'] == 'entity_broad':
            return {
                'primary_terms': [analysis['core_entity']],
                'secondary_terms': [],
                'boost_primary': True,
                'search_in_source': True,
                'min_relevance': 0.5
            }
        else:
            return {
                'primary_terms': analysis['aspects'],
                'secondary_terms': [],
                'boost_primary': False,
                'search_in_source': False,
                'min_relevance': 0.3
            }