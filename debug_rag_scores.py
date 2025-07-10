#!/usr/bin/env python3
"""
Debug script to trace RAG score propagation
"""

print("=== Debugging RAG Score Propagation ===\n")

print("1. FLOW ANALYSIS:")
print("   Milvus Search → Returns (doc, score) tuples")
print("   ↓")
print("   filtered_and_ranked → Contains (doc, score, similarity, keyword_relevance, combined_score)")
print("   ↓")
print("   filtered_docs_with_scores → We now preserve (doc, score) pairs")
print("   ↓")
print("   sources array → Now includes 'score' field")
print("   ↓")
print("   rag_sources → Should contain score field")
print("   ↓")
print("   documents array → Uses source_info.get('score', ...)")
print("   ↓")
print("   Frontend → Displays as percentage\n")

print("2. KEY CODE LOCATIONS:")
print("   - service.py:2250 - Score preservation in filtered_docs_with_scores")
print("   - service.py:2279 - Score added to source_info")
print("   - service.py:3583 - Score retrieved for frontend\n")

print("3. WHAT TO CHECK:")
print("   Run a query and look for these debug messages:")
print("   - '[DEBUG] handle_rag_query: Sources with scores:'")
print("   - '[DEBUG] rag_answer: Converting X rag_sources to documents'")
print("   - Check if scores show as 'None' or actual values\n")

print("4. POTENTIAL ISSUES:")
print("   - If scores show as 'None' in handle_rag_query:")
print("     → Problem is in score extraction from Milvus")
print("   - If scores show values in handle_rag_query but None in rag_answer:")
print("     → Problem is in passing sources between functions")
print("   - If scores show in debug but still 80% in UI:")
print("     → Problem might be in frontend or response format\n")

print("=== End Debug Info ===")