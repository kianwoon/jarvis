#!/usr/bin/env python3
"""Check what prompts are being used"""

from app.core.radiating_settings_cache import get_prompt

# Get the prompts
comprehensive = get_prompt('entity_extraction', 'extraction_comprehensive', 'FALLBACK')
regular = get_prompt('entity_extraction', 'extraction_regular', 'FALLBACK')

print("Comprehensive extraction prompt:")
print("-" * 60)
print(comprehensive)
print("\n" + "=" * 60 + "\n")

print("Regular extraction prompt:")
print("-" * 60)
print(regular)
