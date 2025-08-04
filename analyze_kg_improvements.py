#!/usr/bin/env python3
"""
Analyze and provide specific improvements for meaningful entity extraction
"""

import re
from typing import List, Dict, Tuple

# Sample text for analysis
SAMPLE_TEXT = """
DBS Bank's Digital Transformation Journey in Singapore

In Q2 2024, DBS Bank CEO Piyush Gupta announced a comprehensive $2.5 billion digital transformation 
initiative targeting the Southeast Asian market. The bank partnered with Microsoft Azure and AWS 
to implement cloud-native architecture across their retail banking division.

Key executives involved include:
- CTO David Gledhill overseeing technology implementation
- CFO Sok Hui Chng managing the $2.5B budget allocation
- CDO Paul Cobban leading digital customer experience
"""

def analyze_current_prompt():
    """Analyze issues with current extraction prompt"""
    
    print("=" * 80)
    print("CURRENT EXTRACTION PROMPT ANALYSIS")
    print("=" * 80)
    
    current_prompt_issues = """
Current Knowledge Extraction Prompt Issues:

1. GENERIC ENTITY INSTRUCTIONS:
   - "Extract full names, complete technologies, organizations"
   - No specific guidance on what makes an entity "meaningful"
   - Examples include low-value entities like "Below", "They", "CEO"
   
2. LACK OF BUSINESS CONTEXT:
   - No emphasis on business relevance or strategic importance
   - Missing guidance on financial metrics, KPIs, executive roles
   - No industry-specific patterns or domain knowledge

3. WEAK FILTERING:
   - Only filters basic stop words: 'the', 'and', 'or', 'but'
   - No quality scoring or business relevance assessment
   - Accepts entities with confidence >= 0.3 (too low)

4. QUANTITY OVER QUALITY:
   - Targets "15-25 entities" and "20-30 concepts" per chunk
   - Encourages extraction of everything remotely entity-like
   - No prioritization of high-value entities
"""
    
    print(current_prompt_issues)

def generate_improved_prompt():
    """Generate improved extraction prompt for meaningful entities"""
    
    print("\n" + "=" * 80)
    print("IMPROVED EXTRACTION PROMPT")
    print("=" * 80)
    
    improved_prompt = '''You are an expert business intelligence analyst extracting HIGH-VALUE entities for executive decision-making.

{context_info}
{domain_guidance}

ENTITY EXTRACTION PRIORITIES:

TIER 1 - CRITICAL BUSINESS ENTITIES (Extract ALL):
• Named Organizations: Full company names with context
  ✅ "DBS Bank", "Microsoft Azure", "Monetary Authority of Singapore"
  ❌ "bank", "company", "authority"
  
• Key Personnel: Names WITH roles/titles
  ✅ "CEO Piyush Gupta", "David Gledhill (CTO)", "CFO Sok Hui Chng"
  ❌ "CEO", "manager", "Gupta"
  
• Financial Metrics: Specific amounts, percentages, targets
  ✅ "$2.5 billion investment", "30% cost reduction", "Q2 2024"
  ❌ "investment", "reduction", "quarter"

TIER 2 - STRATEGIC ENTITIES (High Priority):
• Products/Services: Specific offerings, platforms
  ✅ "POSB DigiBank app", "PayNow system", "Kubernetes platform"
  ❌ "app", "system", "platform"
  
• Business Initiatives: Named programs, strategies
  ✅ "digital transformation initiative", "cloud migration program"
  ❌ "initiative", "program", "strategy"

QUALITY CRITERIA:
1. Completeness: Full names, not fragments
2. Specificity: Concrete, not generic
3. Business Value: Would appear in board presentation
4. Verifiability: Can be tracked/measured

FILTERING RULES:
REJECT these low-value entities:
- Single common words: "bank", "digital", "system"
- Generic roles without names: "CEO", "CTO", "manager"
- Vague concepts: "transformation", "technology", "solution"
- Partial names: "Piyush" (without surname), "DBS" (without "Bank")

TEXT: {text}

Extract entities that a competitor would pay to know about.
Quality > Quantity: 20 meaningful entities > 100 generic ones.

OUTPUT FORMAT:
{{
    "entities": [
        {{
            "text": "exact phrase from document",
            "canonical_form": "standardized name",
            "type": "ORGANIZATION|PERSON|FINANCIAL|PRODUCT|INITIATIVE",
            "business_impact": "why this matters",
            "confidence": 0.95
        }}
    ],
    "quality_score": 0.85
}}'''
    
    print(improved_prompt)
    
    return improved_prompt

def demonstrate_entity_filtering():
    """Show improved entity filtering logic"""
    
    print("\n" + "=" * 80)
    print("ENHANCED ENTITY FILTERING LOGIC")
    print("=" * 80)
    
    filtering_code = '''
def score_entity_business_value(entity_text: str, entity_type: str) -> float:
    """Score entity based on business value and meaningfulness"""
    
    score = 0.5  # Base score
    text_lower = entity_text.lower()
    
    # HIGH VALUE INDICATORS (+0.2 to +0.3 each)
    
    # Financial metrics
    if re.match(r'\$[\d.]+\s*(billion|million|M|B)', entity_text):
        score += 0.3
    if re.match(r'\d+%', entity_text):
        score += 0.2
    if re.match(r'[QF]Y?\d{4}|Q[1-4]\s*\d{4}', entity_text):
        score += 0.2
        
    # Named entities with context
    if len(entity_text.split()) >= 2:  # Multi-word entities
        score += 0.15
    if any(title in text_lower for title in ['ceo', 'cto', 'cfo', 'president', 'director']):
        if any(c.isupper() for c in entity_text):  # Has proper names
            score += 0.25
            
    # Specific organizations/products
    org_indicators = ['bank', 'corp', 'ltd', 'inc', 'holdings', 'financial']
    if any(ind in text_lower for ind in org_indicators) and len(entity_text.split()) >= 2:
        score += 0.2
        
    # PENALTIES (-0.3 to -0.5)
    
    # Generic single words
    if len(entity_text.split()) == 1 and len(entity_text) < 5:
        score -= 0.3
        
    # Common generic terms
    generic_terms = ['system', 'platform', 'solution', 'service', 'technology', 
                     'initiative', 'program', 'strategy', 'digital', 'business']
    if text_lower in generic_terms:
        score -= 0.4
        
    # Role without name
    if text_lower in ['ceo', 'cto', 'cfo', 'manager', 'director', 'executive']:
        score -= 0.5
        
    return max(0.0, min(1.0, score))

def filter_meaningful_entities(entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
    """Filter entities based on business value"""
    
    filtered = []
    
    for entity in entities:
        # Calculate business value score
        score = score_entity_business_value(entity.text, entity.type)
        
        # Apply minimum threshold
        if score >= 0.7:  # High threshold for quality
            entity.confidence = score  # Update confidence with business value
            filtered.append(entity)
        else:
            logger.debug(f"Filtered out low-value entity: {entity.text} (score: {score:.2f})")
            
    return filtered
'''
    
    print(filtering_code)

def show_extraction_examples():
    """Show examples of good vs bad extraction"""
    
    print("\n" + "=" * 80)
    print("EXTRACTION EXAMPLES: GOOD VS BAD")
    print("=" * 80)
    
    examples = [
        ("✅ GOOD EXTRACTIONS", [
            ("DBS Bank", "ORGANIZATION", "Primary subject of transformation"),
            ("CEO Piyush Gupta", "PERSON", "Key decision maker"),
            ("$2.5 billion", "FINANCIAL", "Investment amount"),
            ("Q2 2024", "TEMPORAL", "Specific timeline"),
            ("Microsoft Azure", "ORGANIZATION", "Technology partner"),
            ("POSB DigiBank", "PRODUCT", "Specific product launch"),
            ("30% cost reduction", "METRIC", "Measurable target"),
            ("Southeast Asian market", "LOCATION", "Target market"),
        ]),
        
        ("❌ BAD EXTRACTIONS", [
            ("bank", "ORGANIZATION", "Too generic"),
            ("CEO", "PERSON", "Role without name"),
            ("digital", "CONCEPT", "Too vague"),
            ("system", "TECHNOLOGY", "Not specific"),
            ("they", "MISC", "Pronoun, not entity"),
            ("below", "MISC", "Common word"),
            ("initiative", "CONCEPT", "Generic term"),
            ("technology", "CONCEPT", "Too broad"),
        ])
    ]
    
    for category, items in examples:
        print(f"\n{category}:")
        for text, type_, reason in items:
            print(f"  {text:25} {type_:15} {reason}")

def generate_implementation_steps():
    """Generate specific implementation steps"""
    
    print("\n" + "=" * 80)
    print("IMPLEMENTATION STEPS")
    print("=" * 80)
    
    steps = """
STEP 1: Update Extraction Prompt in Database
----------------------------------------
UPDATE settings 
SET settings = jsonb_set(
    settings,
    '{prompts,2,prompt_template}',
    to_jsonb(NEW_OPTIMIZED_PROMPT)
)
WHERE category = 'knowledge_graph';

STEP 2: Enhance Entity Filtering in llm_knowledge_extractor.py
------------------------------------------------------------
# In _enhance_entities_with_hierarchy() method:

# Add business value scoring
business_score = self._score_entity_business_value(entity_name, entity_type)

# Apply stricter filtering
if business_score < 0.7:
    logger.debug(f"Filtering low-value entity: {entity_name} (score: {business_score:.2f})")
    continue

# Update confidence with business relevance
confidence = min(confidence, business_score)

STEP 3: Expand Stop Words List
-----------------------------
BUSINESS_STOP_WORDS = [
    # Current stop words
    'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    
    # Add business generics
    'company', 'business', 'system', 'platform', 'solution', 'service',
    'technology', 'digital', 'transformation', 'initiative', 'program',
    'strategy', 'process', 'approach', 'model', 'framework',
    
    # Add role generics
    'ceo', 'cto', 'cfo', 'manager', 'director', 'executive', 'leader',
    'team', 'group', 'department', 'division', 'unit'
]

STEP 4: Add Pattern Matching for High-Value Entities
-------------------------------------------------
BUSINESS_PATTERNS = {
    'FINANCIAL': [
        r'\$[\d,]+\.?\d*\s*(billion|million|B|M)\b',
        r'\d+\.?\d*%\s*(growth|reduction|increase|decrease)',
        r'[QF]Y\s?\d{4}',
        r'Q[1-4]\s+\d{4}'
    ],
    'EXECUTIVE': [
        r'(CEO|CTO|CFO|President|VP|Director)\s+[A-Z][a-z]+\s+[A-Z][a-z]+',
        r'[A-Z][a-z]+\s+[A-Z][a-z]+,?\s+(CEO|CTO|CFO|President)'
    ],
    'ORGANIZATION': [
        r'[A-Z][A-Za-z]+\s+(Bank|Corp|Corporation|Ltd|Inc|Holdings|Financial)',
        r'[A-Z]{3,}\s+[A-Z][a-z]+',  # "DBS Bank", "AWS Cloud"
    ]
}

STEP 5: Test and Validate
------------------------
1. Run on sample business documents
2. Measure quality score distribution
3. Validate against expected entities
4. Check for false positives/negatives
5. Adjust thresholds based on results
"""
    
    print(steps)

def main():
    """Run the analysis"""
    
    print("KNOWLEDGE GRAPH ENTITY EXTRACTION OPTIMIZATION ANALYSIS")
    print("=" * 80)
    
    # Analyze current issues
    analyze_current_prompt()
    
    # Generate improved prompt
    improved_prompt = generate_improved_prompt()
    
    # Show filtering logic
    demonstrate_entity_filtering()
    
    # Show examples
    show_extraction_examples()
    
    # Implementation steps
    generate_implementation_steps()
    
    print("\n" + "=" * 80)
    print("SUMMARY OF IMPROVEMENTS")
    print("=" * 80)
    print("""
KEY IMPROVEMENTS NEEDED:

1. PROMPT ENGINEERING:
   - Focus on business value, not just any entity
   - Provide clear examples of high vs low value entities
   - Set quality expectations explicitly

2. ENTITY SCORING:
   - Implement business value scoring (0.0-1.0)
   - Filter entities below 0.7 threshold
   - Prioritize completeness and specificity

3. PATTERN MATCHING:
   - Add regex patterns for financial metrics
   - Recognize executive name patterns
   - Identify full organization names

4. STOP WORD EXPANSION:
   - Add business-generic terms to filter
   - Keep context-specific exceptions
   - Filter role titles without names

5. QUALITY METRICS:
   - Track high-value entity percentage
   - Monitor extraction precision
   - Measure business relevance score

Expected Results:
- 70%+ reduction in low-value entities
- 90%+ of extracted entities are business-relevant
- Improved downstream relationship extraction
- Better knowledge graph quality overall
""")

if __name__ == "__main__":
    main()