#!/usr/bin/env python3
"""
Test script for entity normalization improvements
"""
import re

def is_valid_entity(entity_text: str, entity_type: str) -> bool:
    """Validate if an extracted entity is valid"""
    entity_text = entity_text.strip()
    
    # Basic length validation
    if len(entity_text) < 2 or len(entity_text) > 50:
        return False
        
    # Convert to lowercase for pattern matching
    text_lower = entity_text.lower()
    
    # Invalid patterns - more precise to avoid rejecting valid compound entities
    invalid_patterns = [
        r'^(this|that|these|those|it|they|we|you|i|me|my|your|their)(\s|$)',  # Pronouns at start
        r'^(the|a|an|and|or|but|in|on|at|to|for|of|with|by|from)(\s|$)',   # Articles/prepositions at start
        r'^(submit|identify|include|present|highlight|align|maintain|submitting|identifying|including|presenting|highlighting|aligning|maintaining)(\s|$)',  # Action verbs at start
        r'^(immediately|ready|likely|aligning|including|perhaps|even)(\s|$)',  # Adverbs at start
        r'^(this might include|this includes|as well as)(\s|$)',
        r'^[\d\W]+$',  # Numbers and special chars only
        r'^\s*$',     # Empty or whitespace
        r'^(and|or|but|in|on|at|to|for|of|with|by|from)$',  # Single word articles/prepositions
    ]
    
    # Check against invalid patterns
    for pattern in invalid_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return False
    
    # Check for overly generic terms
    generic_terms = {
        'thing', 'stuff', 'item', 'part', 'piece', 'aspect', 'element', 'component',
        'factor', 'issue', 'problem', 'solution', 'approach', 'method', 'technique',
        'process', 'procedure', 'step', 'action', 'activity', 'task', 'work',
        'result', 'outcome', 'impact', 'effect', 'benefit', 'advantage',
        'plan', 'strategy', 'proposal', 'document', 'content', 'information'
    }
    
    if text_lower in generic_terms:
        return False
        
    # Check if it's a complete phrase vs. action/fragment
    if any(word in text_lower for word in ['submit the', 'identify the', 'include the', 'present the', 'highlight the']):
        return False
        
    # Check for sentence-like structures
    if len(text_lower.split()) > 6:  # Too many words for a single entity (reduced from 8)
        return False
        
    # Reject entities that look like sentence fragments or actions
    sentence_indicators = [
        r'\b(earned|aims|strengthen|gained?|ensure|examine|overhauling|adopting|handling|align)\b',
        r'\b(reputation|infrastructure|advantages|compliance|environment|motivations|technologies)\b',
        r'\b(as|a|an|the)\s+\w+\s+\w+',  # "as a digital", "the bank's"
        r'\b(with|in|on|at|to|for|from)\s+\w+',  # preposition + word
        r'\b(is|was|are|were|has|have|had)\b',  # linking verbs
    ]
    
    for pattern in sentence_indicators:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return False
        
    return True

def normalize_entity_name(entity_text: str) -> str:
    """Normalize entity name handling possessives, special chars, and proper capitalization"""
    # Remove possessive suffixes
    normalized = re.sub(r"'s$|'s$", "", entity_text)
    
    # Handle special characters and formatting
    normalized = re.sub(r"[\u2019\u2018]", "'", normalized)  # Smart quotes to regular quotes
    normalized = re.sub(r"\s+", " ", normalized)  # Multiple spaces to single
    normalized = normalized.strip()
    
    # Smart capitalization - preserve known abbreviations and company names
    words = normalized.split()
    result_words = []
    
    for word in words:
        # Preserve all-caps words that are likely abbreviations (2-5 chars)
        if word.isupper() and 2 <= len(word) <= 5:
            result_words.append(word)
        # Preserve mixed case for likely company names or tech terms
        elif any(char.isupper() for char in word[1:]) and len(word) > 2:
            result_words.append(word)
        else:
            result_words.append(word.capitalize())
    
    return " ".join(result_words)

def clean_entity_reference(entity_text: str) -> str:
    """Clean entity reference from relationship extraction"""
    # Remove common prefixes and suffixes that shouldn't be part of entity names
    entity_text = entity_text.strip()
    
    # Remove leading articles and determiners
    entity_text = re.sub(r'^(the|a|an|some|this|that|these|those)\s+', '', entity_text, flags=re.IGNORECASE)
    
    # Remove trailing prepositions and articles that got captured
    entity_text = re.sub(r'\s+(in|on|at|to|for|of|with|by|from|the|a|an)$', '', entity_text, flags=re.IGNORECASE)
    
    # Handle possessives
    entity_text = re.sub(r"'s$|'s$", "", entity_text)
    
    # Clean up extra spaces
    entity_text = re.sub(r'\s+', ' ', entity_text).strip()
    
    return entity_text

if __name__ == "__main__":
    # Test cases from the log
    test_cases = [
        "Ant Group's",
        "DBS Bank", 
        "SOFAStack middleware",
        "OceanBase database",
        "edge database",
        "Its interest"
    ]

    print("Testing entity normalization:")
    for case in test_cases:
        normalized = normalize_entity_name(case)
        print(f"  {case} -> {normalized}")
        
    # Test entity cleaning
    print("\nTesting entity reference cleaning:")
    dirty_cases = [
        "the DBS Bank",
        "Ant Group's technology",
        "a distributed SQL database", 
        "OceanBase database system",
        "Introduction",
        "DBS Bank → earned a reputation as a digital trailblazer among banks",
        "Its interest → cutting",
        "edge database → middleware solutions"
    ]

    for case in dirty_cases:
        cleaned = clean_entity_reference(case)
        print(f"  '{case}' -> '{cleaned}'")
        
    # Test the specific problematic relationships from the log
    print("\nTesting problematic relationship entities:")
    problematic = [
        "DBS Bank → earned a reputation as a digital trailblazer among banks",
        "Its interest → cutting", 
        "edge database → middleware solutions",
        "including Ant Group's OceanBase database → SOFAStack middleware",
        "DBS aims to strengthen its technology infrastructure → line with its digital transformation goals"
    ]
    
    for rel in problematic:
        if " → " in rel:
            source, target = rel.split(" → ", 1)
            source_clean = clean_entity_reference(source)
            target_clean = clean_entity_reference(target)
            
            source_valid = is_valid_entity(source_clean, "UNKNOWN")
            target_valid = is_valid_entity(target_clean, "UNKNOWN")
            
            print(f"  Source: '{source}' -> '{source_clean}' (Valid: {source_valid})")
            print(f"  Target: '{target}' -> '{target_clean}' (Valid: {target_valid})")
            print()
            
    # Test specific entities that should be valid
    print("Testing valid entities:")
    valid_test = [
        "DBS Bank",
        "Ant Group", 
        "SOFAStack",
        "OceanBase",
        "Tencent",
        "middleware solutions",
        "database technology",
        "cutting edge",
    ]
    
    for entity in valid_test:
        valid = is_valid_entity(entity, "UNKNOWN")
        print(f"  '{entity}' -> Valid: {valid}")
        
    # Test invalid entities
    print("\nTesting invalid entities:")
    invalid_test = [
        "earned a reputation as a digital trailblazer among banks",
        "Its interest",
        "DBS aims to strengthen its technology infrastructure",
        "line with its digital transformation goals",
        "cutting",
        "the bank's broader IT strategy",
        "this includes",
        "as well as",
    ]
    
    for entity in invalid_test:
        valid = is_valid_entity(entity, "UNKNOWN")
        print(f"  '{entity}' -> Valid: {valid}")