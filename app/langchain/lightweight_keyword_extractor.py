"""
Lightweight keyword extraction without external NLP dependencies
"""
import re
from typing import List, Dict, Set

class LightweightKeywordExtractor:
    def __init__(self):
        # Question patterns that indicate focus
        self.question_patterns = [
            # Pattern, group number to extract
            (r"what\s+is\s+([a-zA-Z][\w\s]*?)(?:\s+(?:and|or|in|at|for|structure|organization)|\?|$)", 1),
            (r"how'?s?\s+([a-zA-Z][\w\s]*?)(?:\s+(?:look|doing|structured|organized)|\?|$)", 1),
            (r"tell\s+me\s+about\s+([a-zA-Z][\w\s]*?)(?:\s+(?:and|or|in|at)|\?|$)", 1),
            (r"(?:information|details?|data)\s+(?:about|on|for)\s+([a-zA-Z][\w\s]*?)(?:\s+(?:and|or)|\?|$)", 1),
            (r"([a-zA-Z][\w\s]*?)\s+(?:structure|organization|partnership|system|platform)", 1),
            (r"(?:describe|explain|show)\s+([a-zA-Z][\w\s]*?)(?:\s+(?:and|or)|\?|$)", 1),
        ]
        
        # Domain-specific important terms
        self.important_keywords = {
            'structure', 'organization', 'organisation', 'partnership', 'system',
            'platform', 'architecture', 'framework', 'model', 'process',
            'hierarchy', 'department', 'division', 'team', 'management',
            'collaboration', 'agreement', 'contract', 'alliance', 'joint',
            'venture', 'cooperation', 'relationship', 'business', 'strategic'
        }
        
        # Extended stop words
        self.stop_words = {
            'the', 'is', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'must', 'shall', 'can', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under',
            'again', 'further', 'then', 'once', 'a', 'an', 'the', 'as', 'i', 'me',
            'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
            'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
            'these', 'those', 'am', 'some', 'any', 'all', 'each', 'every', 'either',
            'neither', 'one', 'many', 'few', 'most', 'other', 'another', 'such',
            'no', 'nor', 'not', 'only', 'same', 'so', 'than', 'too', 'very',
            'just', 'there', 'where', 'when', 'why', 'how', 'both', 'each', 'more',
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'can', 'will', 'just', 'should', 'now', 'get', 'got',
            'give', 'gave', 'given', 'take', 'took', 'taken', 'make', 'made',
            'know', 'knew', 'known', 'think', 'thought', 'see', 'saw', 'seen',
            'come', 'came', 'go', 'went', 'gone', 'say', 'said', 'tell', 'told',
            'show', 'showed', 'shown', 'find', 'found', 'look', 'looked', 'like'
        }
        
    def extract_key_terms(self, question: str) -> Dict[str, List[str]]:
        """Extract key terms from question"""
        key_terms = {
            'pattern_matches': [],
            'potential_entities': [],
            'important_phrases': [],
            'content_terms': []
        }
        
        # Extract from patterns
        for pattern, group_num in self.question_patterns:
            matches = re.finditer(pattern, question, re.IGNORECASE)
            for match in matches:
                term = match.group(group_num).strip().lower()
                # Clean up the term
                term = re.sub(r'\s+', ' ', term)  # Normalize whitespace
                term = term.rstrip('?.,!;:')  # Remove trailing punctuation
                if len(term) > 2 and term not in self.stop_words:
                    key_terms['pattern_matches'].append(term)
        
        # Extract potential entities (capitalized words + known company patterns)
        words = question.split()
        
        # Flexible company/entity patterns (case-insensitive) - works for any company
        company_patterns = [
            # Common company suffixes
            r'\b([a-z]{3,})\s+(corp|inc|ltd|llc|company|systems|technologies|group|enterprise|solutions|services)\b',
            r'\b([a-z]+soft|[a-z]+corp|[a-z]+inc|[a-z]+ltd|[a-z]+llc|[a-z]+tech)\b',
            # Partnership/business relationship patterns
            r'\b(partnership|collaboration|alliance)\s+(?:between|with)\s+([a-z]{3,})\s+and\s+([a-z]{3,})\b',
            r'\b([a-z]{3,})\s+(?:partnership|collaboration|alliance)\s+with\s+([a-z]{3,})\b',
            # Multi-word company names (e.g., "beyond soft", "ten cent")
            r'\b([a-z]{3,})\s+([a-z]{3,})\s+(?:corp|inc|ltd|company|group|technologies)\b',
            # Simple proper nouns that could be companies (3+ chars, not common words)
            r'\b([a-z]{3,})\b(?=\s+(?:and|with|between|corp|inc|ltd|company|group|technologies))'
        ]
        
        # Extract company names using patterns
        question_lower = question.lower()
        for pattern in company_patterns:
            matches = re.finditer(pattern, question_lower)
            for match in matches:
                # Extract all capture groups (could be multiple companies)
                for i in range(1, len(match.groups()) + 1):
                    try:
                        entity = match.group(i)
                        if entity and len(entity) > 2 and entity not in self.stop_words:
                            # Skip common words that aren't companies
                            common_words = {'between', 'with', 'and', 'partnership', 'collaboration', 'alliance', 'company', 'corp', 'inc', 'ltd', 'llc', 'group', 'systems', 'technologies'}
                            if entity.lower() not in common_words:
                                key_terms['potential_entities'].append(entity.lower())
                    except:
                        continue
        
        # Original capitalized word detection
        for i, word in enumerate(words):
            # Skip first word (often capitalized as sentence start)
            if i == 0:
                continue
            
            # Check if word is capitalized and not a stop word
            if word[0].isupper() and word.lower() not in self.stop_words:
                clean_word = re.sub(r'[^\w]', '', word).lower()
                if len(clean_word) > 2:
                    key_terms['potential_entities'].append(clean_word)
        
        # Extract important phrases (2-3 word combinations + partnership patterns)
        words_lower = [re.sub(r'[^\w]', '', w).lower() for w in words]
        
        # Look for "Company1 and Company2" or "partnership between Company1 and Company2" patterns
        partnership_patterns = [
            r'\b(partnership|collaboration|alliance)\s+between\s+(\w+)\s+and\s+(\w+)\b',
            r'\b(\w+)\s+and\s+(\w+)\s+(partnership|collaboration|alliance)\b',
            r'\b(\w+)\s+(partnership|collaboration|alliance)\s+with\s+(\w+)\b'
        ]
        
        for pattern in partnership_patterns:
            matches = re.finditer(pattern, question_lower)
            for match in matches:
                # Extract all non-stop words from the match
                for group_text in match.groups():
                    if group_text and len(group_text) > 2 and group_text not in self.stop_words:
                        key_terms['important_phrases'].append(group_text)
        
        # Original phrase extraction
        for i in range(len(words_lower) - 1):
            # Two-word phrases
            if words_lower[i] not in self.stop_words:
                phrase = f"{words_lower[i]} {words_lower[i+1]}"
                if words_lower[i+1] in self.important_keywords:
                    key_terms['important_phrases'].append(phrase)
                elif i < len(words_lower) - 2:
                    # Three-word phrases
                    phrase_3 = f"{words_lower[i]} {words_lower[i+1]} {words_lower[i+2]}"
                    if words_lower[i+2] in self.important_keywords:
                        key_terms['important_phrases'].append(phrase_3)
        
        # Extract content terms (non-stop words)
        for word in words_lower:
            if (len(word) > 2 and 
                word not in self.stop_words and 
                word not in key_terms['pattern_matches'] and
                word not in key_terms['potential_entities']):
                key_terms['content_terms'].append(word)
        
        # Deduplicate within each category
        for category in key_terms:
            key_terms[category] = list(dict.fromkeys(key_terms[category]))
        
        return key_terms
    
    def get_search_terms(self, key_terms: Dict[str, List[str]], max_terms: int = 5) -> List[str]:
        """Get prioritized search terms"""
        search_terms = []
        
        # Priority order
        search_terms.extend(key_terms.get('pattern_matches', []))
        search_terms.extend(key_terms.get('potential_entities', []))
        search_terms.extend(key_terms.get('important_phrases', []))
        
        # Add content terms if we need more
        if len(search_terms) < max_terms:
            search_terms.extend(key_terms.get('content_terms', []))
        
        # Deduplicate and limit
        seen = set()
        final_terms = []
        for term in search_terms:
            if term not in seen and len(final_terms) < max_terms:
                seen.add(term)
                final_terms.append(term)
        
        return final_terms