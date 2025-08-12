#!/usr/bin/env python3
"""
Simple test to verify ChatGPT search results without complex imports
"""

import asyncio
import aiohttp
import os
import json
from datetime import datetime

async def test_google_search_direct():
    """Test Google search API directly"""
    
    # Get API credentials
    api_key = os.getenv("GOOGLE_SEARCH_API_KEY", "AIzaSyA2U7MBpH7cNDykiZ_OlGsdJJlXumsMps4")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID", "d77ac8c3d3e124c3c")
    
    query = "ChatGPT Plus Pro subscription usage limits messages per hour GPT-4o"
    
    print(f"\n🔍 Testing Google Search with query: {query}")
    print(f"Current system date: {datetime.now()}")
    
    search_url = "https://www.googleapis.com/customsearch/v1"
    search_params = {
        "key": api_key,
        "cx": search_engine_id,
        "q": query,
        "num": 5
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(search_url, params=search_params) as response:
            if response.status == 200:
                data = await response.json()
                
                print("\n📊 Raw Google Search API Response:")
                print(json.dumps(data, indent=2))
                
                print("\n📝 Extracted Search Results:")
                for i, item in enumerate(data.get("items", []), 1):
                    print(f"\n{i}. {item.get('title', 'No title')}")
                    print(f"   {item.get('snippet', 'No snippet')}")
                    print(f"   {item.get('link', 'No link')}")
                    
                    # Check for suspicious content
                    content = f"{item.get('title', '')} {item.get('snippet', '')}".lower()
                    if 'o3' in content or 'o4-mini' in content:
                        print(f"   ⚠️ WARNING: Found suspicious model name!")
                    if '100 messages' in content or 'messages/week' in content:
                        print(f"   💡 Found usage limit information!")
            else:
                print(f"❌ Google Search API error: {response.status}")
                print(await response.text())

async def test_correct_date_search():
    """Test search with correct date constraints"""
    
    api_key = os.getenv("GOOGLE_SEARCH_API_KEY", "AIzaSyA2U7MBpH7cNDykiZ_OlGsdJJlXumsMps4")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID", "d77ac8c3d3e124c3c")
    
    # Force search to 2024 results
    query = 'ChatGPT Plus subscription "40 messages" OR "80 messages" OR "messages every 3 hours" 2024'
    
    print(f"\n🔍 Testing with 2024-specific query: {query}")
    
    search_url = "https://www.googleapis.com/customsearch/v1"
    search_params = {
        "key": api_key,
        "cx": search_engine_id,
        "q": query,
        "num": 5,
        "dateRestrict": "y1"  # Restrict to past year
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(search_url, params=search_params) as response:
            if response.status == 200:
                data = await response.json()
                
                print("\n📝 Search Results (2024 focus):")
                for i, item in enumerate(data.get("items", []), 1):
                    print(f"\n{i}. {item.get('title', 'No title')}")
                    print(f"   {item.get('snippet', 'No snippet')[:150]}...")
                    
                    # Look for actual ChatGPT limits
                    snippet = item.get('snippet', '').lower()
                    if '40 messages' in snippet or '80 messages' in snippet:
                        print(f"   ✅ Found valid ChatGPT usage limit info!")

async def main():
    print("=" * 60)
    print("ChatGPT Search Issue Investigation")
    print("=" * 60)
    
    # Test 1: Direct Google search
    await test_google_search_direct()
    
    # Test 2: Search with date constraints
    await test_correct_date_search()
    
    print("\n" + "=" * 60)
    print("Investigation Complete!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())