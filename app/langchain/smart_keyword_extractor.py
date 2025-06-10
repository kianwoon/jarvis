"""
Smart keyword extraction for enhanced RAG search
"""
import re
from typing import List, Dict, Set, Tuple
from collections import Counter

try:
    import spacy
except ImportError:
    spacy = None

class SmartKeywordExtractor:
    def __init__(self):
        self.nlp = None
        if spacy is not None:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                # Model not downloaded, will use rule-based extraction
                pass
            
        # Common question patterns that indicate the focus
        self.question_patterns = [
            (r"what\s+is\s+(\w+)", 1),  # "what is X"
            (r"how\s+is\s+(\w+)", 1),   # "how is X"
            (r"how'?s?\s+(\w+)", 1),    # "how's X" or "hows X"
            (r"tell\s+me\s+about\s+(\w+)", 1),  # "tell me about X"
            (r"information\s+about\s+(\w+)", 1),  # "information about X"
            (r"(\w+)\s+structure", 1),   # "X structure"
            (r"(\w+)\s+organization", 1), # "X organization"
            (r"(\w+)\s+partnership", 1),  # "X partnership"
        ]
        
    def extract_key_terms(self, question: str) -> Dict[str, List[str]]:
        """
        Extract key terms using multiple strategies
        """
        key_terms = {
            'entities': [],      # Named entities (orgs, people, etc.)
            'noun_phrases': [],  # Important noun phrases
            'pattern_matches': [], # Terms extracted from question patterns
            'focal_terms': []    # The main focus of the question
        }
        
        # Strategy 1: Pattern matching for question focus
        key_terms['pattern_matches'] = self._extract_from_patterns(question)
        
        # Strategy 2: NLP-based extraction if available
        if self.nlp:
            doc = self.nlp(question)
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PERSON', 'PRODUCT', 'GPE', 'LOC']:
                    key_terms['entities'].append(ent.text.lower())
            
            # Extract noun phrases
            for chunk in doc.noun_chunks:
                # Filter out generic phrases
                if self._is_meaningful_phrase(chunk.text):
                    key_terms['noun_phrases'].append(chunk.text.lower())
        
        # Strategy 3: Rule-based extraction (fallback or complement)
        rule_based_terms = self._extract_rule_based(question)
        for term in rule_based_terms:
            if term not in key_terms['entities']:
                key_terms['focal_terms'].append(term)
        
        # Deduplicate across categories
        key_terms = self._deduplicate_terms(key_terms)
        
        return key_terms
    
    def _extract_from_patterns(self, question: str) -> List[str]:
        """Extract terms based on question patterns"""
        matches = []
        question_lower = question.lower()
        
        for pattern, group_num in self.question_patterns:
            match = re.search(pattern, question_lower)
            if match:
                term = match.group(group_num)
                if len(term) > 2:  # Skip very short terms
                    matches.append(term)
        
        return matches
    
    def _extract_rule_based(self, question: str) -> List[str]:
        """Rule-based extraction for when NLP is not available"""
        # Remove common words and punctuation
        stop_words = {
            'the', 'is', 'are', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'a', 'an', 'what', 'how', 'when', 'where', 'why',
            'can', 'could', 'would', 'should', 'give', 'tell', 'show', 'find',
            'look', 'like', 'about', 'information', 'me'
        }
        
        # Clean and tokenize
        words = re.findall(r'\w+', question.lower())
        
        # Find potential key terms
        key_terms = []
        for i, word in enumerate(words):
            # Skip stop words and short words
            if word in stop_words or len(word) <= 2:
                continue
                
            # Check if it might be a proper noun (capitalized in original)
            original_words = re.findall(r'\w+', question)
            if i < len(original_words) and original_words[i][0].isupper():
                key_terms.append(word)
            # Check if it's a potentially important term (specific criteria)
            elif self._is_specific_term(word, words):
                key_terms.append(word)
        
        return key_terms
    
    def _is_meaningful_phrase(self, phrase: str) -> bool:
        """Check if a noun phrase is meaningful (not too generic)"""
        generic_phrases = {
            'information', 'data', 'details', 'thing', 'things',
            'way', 'ways', 'type', 'types', 'kind', 'kinds'
        }
        
        phrase_lower = phrase.lower().strip()
        
        # Skip single generic words
        if phrase_lower in generic_phrases:
            return False
            
        # Skip phrases that are too short
        if len(phrase_lower) < 3:
            return False
            
        # Accept phrases with specific terms
        if any(term in phrase_lower for term in ['structure', 'organization', 'partnership', 'system']):
            return True
            
        return True
    
    def _is_specific_term(self, word: str, context: List[str]) -> bool:
        """Determine if a word is likely a specific/important term"""
        # Terms that often indicate importance when they appear
        important_indicators = {
            'structure', 'organization', 'organisation', 'partnership',
            'system', 'platform', 'service', 'product', 'company',
            'corporation', 'business', 'model', 'framework'
        }
        
        # If the word itself is an important indicator
        if word in important_indicators:
            return True
            
        # If the word appears before/after an important indicator
        word_index = context.index(word) if word in context else -1
        if word_index > 0:
            if context[word_index - 1] in important_indicators:
                return True
        if word_index < len(context) - 1 and word_index >= 0:
            if context[word_index + 1] in important_indicators:
                return True
                
        return False
    
    def _deduplicate_terms(self, key_terms: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Remove duplicates while preserving category information"""
        seen = set()
        deduped = {}
        
        for category, terms in key_terms.items():
            deduped[category] = []
            for term in terms:
                if term not in seen:
                    seen.add(term)
                    deduped[category].append(term)
                    
        return deduped
    
    def get_all_key_terms(self, key_terms: Dict[str, List[str]]) -> List[str]:
        """Get all unique key terms as a flat list"""
        all_terms = []
        for terms in key_terms.values():
            all_terms.extend(terms)
        return list(set(all_terms))
    
    def prioritize_terms(self, key_terms: Dict[str, List[str]]) -> List[str]:
        """Return terms in priority order"""
        # Priority: entities > pattern_matches > focal_terms > noun_phrases
        prioritized = []
        prioritized.extend(key_terms.get('entities', []))
        prioritized.extend(key_terms.get('pattern_matches', []))
        prioritized.extend(key_terms.get('focal_terms', []))
        prioritized.extend(key_terms.get('noun_phrases', []))
        
        # Remove duplicates while preserving order
        seen = set()
        result = []
        for term in prioritized:
            if term not in seen:
                seen.add(term)
                result.append(term)
                
        return result
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text (compatibility method)"""
        key_terms = self.extract_key_terms(text)
        return self.prioritize_terms(key_terms)