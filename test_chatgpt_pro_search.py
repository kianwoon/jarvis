#!/usr/bin/env python3
"""Test enhanced search with product-aware filtering for ChatGPT Pro queries"""

import asyncio
import json
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_direct_google_search():
    """Test the enhanced _direct_google_search with product filtering"""
    print("=" * 80)
    print("Testing Enhanced Google Search with Product-Aware Filtering")
    print("=" * 80)
    
    try:
        from app.core.unified_mcp_service import UnifiedMCPService
        
        service = UnifiedMCPService()
        
        # Test queries
        test_queries = [
            {
                "name": "ChatGPT Pro subscription",
                "query": "what are the usage limits of ChatGPT PRO subscription?",
                "expected_focus": "Pro ($200/month)"
            },
            {
                "name": "ChatGPT Plus subscription",
                "query": "what are the usage limits of ChatGPT Plus subscription?",
                "expected_focus": "Plus ($20/month)"
            },
            {
                "name": "ChatGPT Pro vs Plus comparison",
                "query": "difference between ChatGPT Pro and ChatGPT Plus subscriptions",
                "expected_focus": "Both products"
            }
        ]
        
        for test in test_queries:
            print(f"\n{'='*60}")
            print(f"Test: {test['name']}")
            print(f"Query: {test['query']}")
            print(f"Expected focus: {test['expected_focus']}")
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
                
                # Check for disambiguation note
                if "⚠️ Note:" in content:
                    print("✅ Disambiguation note added")
                    # Extract and print the note
                    note_start = content.find("⚠️ Note:")
                    note_end = content.find("\n\n", note_start + 1)
                    if note_end > 0:
                        print(f"   {content[note_start:note_end]}")
                
                # Check result filtering
                lines = content.split("\n")
                print(f"\nResults returned: {lines[0]}")
                
                # Count mentions of different products
                content_lower = content.lower()
                pro_mentions = content_lower.count("pro")
                plus_mentions = content_lower.count("plus")
                price_200_mentions = content_lower.count("$200") + content_lower.count("200/month")
                price_20_mentions = content_lower.count("$20") + content_lower.count("20/month")
                
                print(f"\nContent analysis:")
                print(f"  Pro mentions: {pro_mentions}")
                print(f"  Plus mentions: {plus_mentions}")
                print(f"  $200 price mentions: {price_200_mentions}")
                print(f"  $20 price mentions: {price_20_mentions}")
                
                # Show first few results
                print(f"\nFirst results:")
                result_blocks = content.split("\n\n")[1:4]  # Skip header, get first 3
                for i, block in enumerate(result_blocks, 1):
                    title = block.split("\n")[0] if "\n" in block else block
                    print(f"  {i}. {title[:80]}...")
        
        # Clean up
        await service.close()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

async def test_search_optimization_integration():
    """Test the complete flow: optimization + search + filtering"""
    print("\n" + "=" * 80)
    print("Testing Complete Search Flow with Optimization")
    print("=" * 80)
    
    try:
        from app.langchain.search_query_optimizer import get_search_query_optimizer
        from app.core.unified_mcp_service import UnifiedMCPService
        
        optimizer = get_search_query_optimizer()
        service = UnifiedMCPService()
        
        # Original user query
        original_query = "what are the usage limit of chatgpt pro subscription?"
        
        print(f"Original query: {original_query}")
        
        # Step 1: Optimize query
        optimization_result = await optimizer.optimize_query(original_query)
        print(f"\nOptimization result:")
        print(f"  Optimized: {optimization_result['optimized']}")
        print(f"  Confidence: {optimization_result['confidence']:.2f}")
        print(f"  Method: {optimization_result['method']}")
        print(f"  Entities preserved: {optimization_result['entities_preserved']}")
        
        # Verify PRO was preserved
        if 'pro' in optimization_result['optimized'].lower():
            print("  ✅ 'PRO' preserved in optimization")
        else:
            print("  ❌ 'PRO' was lost in optimization!")
        
        # Step 2: Execute search with optimized query
        print(f"\nExecuting search with optimized query...")
        search_result = await service._direct_google_search({
            "query": optimization_result['optimized'],
            "num_results": 5
        })
        
        if "error" in search_result:
            print(f"❌ Search error: {search_result['error']}")
        else:
            content = search_result.get("content", [{}])[0].get("text", "")
            
            # Check for Pro-focused results
            content_lower = content.lower()
            if "$200" in content_lower or "200/month" in content_lower:
                print("✅ Search results include Pro pricing ($200/month)")
            else:
                print("⚠️  No Pro pricing found in results")
            
            if "⚠️ Note:" in content:
                print("✅ Disambiguation note present")
            
            # Show summary
            lines = content.split("\n")
            print(f"\nSearch summary: {lines[0]}")
        
        # Clean up
        await service.close()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting enhanced search tests...\n")
    
    # Run tests
    asyncio.run(test_direct_google_search())
    asyncio.run(test_search_optimization_integration())
    
    print("\n" + "=" * 80)
    print("Tests completed!")
    print("=" * 80)