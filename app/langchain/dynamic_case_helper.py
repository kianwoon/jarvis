"""
Dynamic case-insensitive search helper without any hardcoding
"""
from typing import List, Set
import re

def generate_case_variations(term: str) -> List[str]:
    """
    Generate common case variations for a search term dynamically
    """
    variations = set()
    
    # Basic variations
    variations.add(term.lower())
    variations.add(term.upper())
    variations.add(term.title())
    variations.add(term.capitalize())
    
    # Handle camelCase/PascalCase detection
    # Check if term has mixed case (e.g., "OpenAI", "LinkedIn")
    if term != term.lower() and term != term.upper():
        # It's already mixed case, preserve it
        variations.add(term)
    
    # Detect word boundaries and create variations
    # Split on uppercase letters that follow lowercase letters
    words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)', term)
    if len(words) > 1:
        # Create variations with different capitalizations
        variations.add(''.join(words))  # allwords
        variations.add(' '.join(words))  # all words
        variations.add(''.join(w.capitalize() for w in words))  # PascalCase
        variations.add(words[0].lower() + ''.join(w.capitalize() for w in words[1:]))  # camelCase
    
    # Handle potential acronyms or special patterns
    # If term has numbers or special patterns, preserve original
    if re.search(r'\d', term) or re.search(r'[^a-zA-Z\s]', term):
        variations.add(term)
    
    # Convert set to list and return
    return list(variations)

def build_case_insensitive_condition(field: str, term: str, operator: str = "or") -> str:
    """
    Build a case-insensitive search condition for Milvus
    
    Args:
        field: The field to search in (e.g., 'content', 'source', 'section')
        term: The search term
        operator: 'or' or 'and' to join conditions
        
    Returns:
        A Milvus expression string
    """
    variations = generate_case_variations(term)
    conditions = [f'{field} like "%{var}%"' for var in variations]
    
    if len(conditions) == 1:
        return conditions[0]
    
    return f'({" {operator} ".join(conditions)})'

def build_multi_field_case_insensitive_expr(term: str, fields: List[str] = None) -> str:
    """
    Build case-insensitive search across multiple fields
    
    Args:
        term: The search term
        fields: List of fields to search in (default: ['content', 'source', 'section'])
        
    Returns:
        A Milvus expression string
    """
    if fields is None:
        fields = ['content', 'source', 'section']
    
    field_expressions = []
    for field in fields:
        field_expr = build_case_insensitive_condition(field, term, "or")
        field_expressions.append(field_expr)
    
    return " or ".join(field_expressions)

def smart_case_detection(text: str) -> List[str]:
    """
    Intelligently detect and generate case variations based on the text pattern
    """
    variations = set()
    
    # Always include basic variations
    variations.add(text.lower())
    variations.add(text.upper())
    
    # Detect if it's likely a proper noun or brand name
    # by checking if it starts with capital in a sentence context
    if text and text[0].isupper():
        variations.add(text)  # Keep original
        variations.add(text.title())  # Ensure title case
        
    # Detect compound words or phrases
    # Split by common delimiters
    delimiters = [' ', '-', '_', '.']
    for delimiter in delimiters:
        if delimiter in text:
            parts = text.split(delimiter)
            # Generate variations with different cases for each part
            variations.add(delimiter.join(p.lower() for p in parts))
            variations.add(delimiter.join(p.upper() for p in parts))
            variations.add(delimiter.join(p.title() for p in parts))
            
            # Also try without delimiter
            variations.add(''.join(p.lower() for p in parts))
            variations.add(''.join(p.title() for p in parts))
    
    return list(variations)