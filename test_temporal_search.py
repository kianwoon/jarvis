#!/usr/bin/env python3
"""Test temporal filtering and relevance scoring for search results"""

import asyncio
import json
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_temporal_google_search():
    """Test the enhanced Google Search with temporal filtering"""
    print("=" * 80)
    print("Testing Google Search with Temporal Filtering and Relevance")
    print(f"Current Date: {datetime.now().strftime('%B %d, %Y')}")
    print("=" * 80)
    
    try:
        from app.core.unified_mcp_service import UnifiedMCPService
        
        service = UnifiedMCPService()
        
        # Test queries
        test_queries = [
            {
                "name": "ChatGPT Pro subscription (time-sensitive)",
                "query": "what are the usage limits of ChatGPT PRO subscription?",
                "expected": "Should apply date restriction and temporal scoring"
            },
            {
                "name": "Current ChatGPT Pro information",
                "query": "current ChatGPT Pro subscription features",
                "expected": "Should restrict to recent results (6 months)"
            },
            {
                "name": "Latest pricing information",
                "query": "latest ChatGPT pricing tiers 2025",
                "expected": "Should prioritize 2025 content"
            }
        ]
        
        for test in test_queries:
            print(f"\n{'='*60}")
            print(f"Test: {test['name']}")
            print(f"Query: {test['query']}")
            print(f"Expected: {test['expected']}")
            print("-" * 60)
            
            # Execute search
            result = await service._direct_google_search({
                "query": test['query'],
                "num_results": 5
            })
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                content = result.get("content", [{}])[0].get("text", "")
                
                # Check for temporal warnings
                if "⚠️ Warning:" in content:
                    print("✅ Temporal warning added for outdated content")
                    warning_start = content.find("⚠️ Warning:")
                    warning_end = content.find("\n\n", warning_start + 1)
                    if warning_end > 0:
                        print(f"   {content[warning_start:warning_end]}")
                elif "⚠️ Note:" in content:
                    print("✅ Temporal/disambiguation note added")
                    note_start = content.find("⚠️ Note:")
                    note_end = content.find("\n\n", note_start + 1)
                    if note_end > 0:
                        print(f"   {content[note_start:note_end]}")
                
                # Check for recency labels
                if "[Current]" in content:
                    print("✅ Found results labeled as [Current] (< 30 days old)")
                if "[Recent]" in content:
                    print("✅ Found results labeled as [Recent] (30-90 days old)")
                if "[Outdated]" in content:
                    print("⚠️  Found results labeled as [Outdated] (> 1 year old)")
                
                # Extract first result to check
                lines = content.split("\n")
                print(f"\nFirst line: {lines[0]}")
                
                # Show first result with recency label
                result_blocks = content.split("\n\n")[1:2]  # Get first result
                if result_blocks:
                    first_result = result_blocks[0]
                    print(f"\nFirst result preview:")
                    print(f"  {first_result[:200]}...")
        
        # Clean up
        await service.close()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

async def test_query_optimization_with_year():
    """Test query optimization with automatic year addition"""
    print("\n" + "=" * 80)
    print("Testing Query Optimization with Temporal Context")
    print("=" * 80)
    
    try:
        from app.langchain.search_query_optimizer import get_search_query_optimizer
        
        optimizer = get_search_query_optimizer()
        
        test_queries = [
            {
                "query": "what are the usage limit of chatgpt pro subscription?",
                "description": "Subscription query without year"
            },
            {
                "query": "current ChatGPT Pro features",
                "description": "Query with 'current' keyword"
            },
            {
                "query": "ChatGPT pricing",
                "description": "Pricing query without temporal context"
            },
            {
                "query": "ChatGPT Pro limits 2024",
                "description": "Query with explicit year (should not add 2025)"
            }
        ]
        
        current_year = str(datetime.now().year)
        
        for test in test_queries:
            print(f"\n{test['description']}:")
            print(f"  Original: {test['query']}")
            
            result = await optimizer.optimize_query(test['query'])
            
            print(f"  Optimized: {result['optimized']}")
            print(f"  Method: {result['method']}")
            
            # Check if year was added
            if current_year in result['optimized'] and current_year not in test['query']:
                print(f"  ✅ Added current year ({current_year}) to query")
            elif current_year not in result['optimized'] and any(term in test['query'].lower() for term in ['current', 'latest', 'subscription', 'pricing', 'limit']):
                print(f"  ⚠️  Year not added despite time-sensitive terms")
            else:
                print(f"  ℹ️  No year addition needed")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

async def test_complete_flow():
    """Test the complete flow: optimization + search + temporal filtering"""
    print("\n" + "=" * 80)
    print("Testing Complete Flow with Temporal Enhancements")
    print("=" * 80)
    
    try:
        from app.langchain.search_query_optimizer import get_search_query_optimizer
        from app.core.unified_mcp_service import UnifiedMCPService
        
        optimizer = get_search_query_optimizer()
        service = UnifiedMCPService()
        
        # Original query
        original = "what are the usage limit of chatgpt pro subscription?"
        print(f"Original query: {original}")
        
        # Step 1: Optimize
        opt_result = await optimizer.optimize_query(original)
        print(f"\n1. Optimization:")
        print(f"   Result: {opt_result['optimized']}")
        print(f"   Method: {opt_result['method']}")
        
        # Step 2: Search with optimized query
        print(f"\n2. Executing search...")
        search_result = await service._direct_google_search({
            "query": opt_result['optimized'],
            "num_results": 3
        })
        
        if "error" not in search_result:
            content = search_result.get("content", [{}])[0].get("text", "")
            
            # Analyze results
            print(f"\n3. Results analysis:")
            
            # Check temporal indicators
            if "2025" in opt_result['optimized']:
                print("   ✅ Year 2025 added to query")
            
            if "[Current]" in content or "[Recent]" in content:
                print("   ✅ Recent results found and labeled")
            
            if "⚠️" in content:
                print("   ✅ Temporal/product warnings present")
            
            # Count date mentions
            import re
            dates_2025 = len(re.findall(r'2025', content))
            dates_2024 = len(re.findall(r'2024', content))
            dates_2023 = len(re.findall(r'2023', content))
            
            print(f"\n4. Date distribution in results:")
            print(f"   2025 mentions: {dates_2025}")
            print(f"   2024 mentions: {dates_2024}")
            print(f"   2023 mentions: {dates_2023}")
            
            if dates_2025 > dates_2024 + dates_2023:
                print("   ✅ Successfully prioritizing 2025 content")
            elif dates_2024 > 0 or dates_2023 > 0:
                print("   ⚠️  Still showing older content (may need stronger filtering)")
        
        # Clean up
        await service.close()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting temporal search enhancement tests...\n")
    
    # Run all tests
    asyncio.run(test_temporal_google_search())
    asyncio.run(test_query_optimization_with_year())
    asyncio.run(test_complete_flow())
    
    print("\n" + "=" * 80)
    print("Temporal search tests completed!")
    print("=" * 80)