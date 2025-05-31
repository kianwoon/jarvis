"""
Helper functions for case-insensitive search in Milvus
"""
from typing import List, Set

def generate_case_variations(term: str) -> List[str]:
    """
    Generate common case variations for a search term
    """
    variations = [
        term.lower(),           # lowercase
        term.upper(),           # UPPERCASE
        term.title(),           # Title Case
        term.capitalize(),      # First letter cap
    ]
    
    # Handle special cases for known entities
    special_cases = {
        'alibaba': ['Alibaba', 'ALIBABA', 'alibaba', 'AliBaba'],
        'openai': ['OpenAI', 'openai', 'OPENAI', 'OpenAi'],
        'chatgpt': ['ChatGPT', 'chatgpt', 'CHATGPT', 'chatGPT'],
        'linkedin': ['LinkedIn', 'linkedin', 'LINKEDIN', 'Linkedin'],
        'youtube': ['YouTube', 'youtube', 'YOUTUBE', 'Youtube'],
        'github': ['GitHub', 'github', 'GITHUB', 'Github'],
        'facebook': ['Facebook', 'facebook', 'FACEBOOK', 'FaceBook'],
        'microsoft': ['Microsoft', 'microsoft', 'MICROSOFT'],
        'macos': ['macOS', 'MacOS', 'MACOS', 'macos'],
        'ios': ['iOS', 'IOS', 'ios'],
    }
    
    # Add special case variations if applicable
    term_lower = term.lower()
    if term_lower in special_cases:
        variations.extend(special_cases[term_lower])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_variations = []
    for v in variations:
        if v not in seen:
            seen.add(v)
            unique_variations.append(v)
    
    return unique_variations

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