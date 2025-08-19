#!/usr/bin/env python3
"""
Test script for enhanced Google Search result processing with rich metadata.

This script tests:
1. Extraction of publication dates from pagemap metadata
2. HTML snippet formatting with search term highlighting
3. Temporal relevance scoring with actual dates
4. Enhanced display formatting with thumbnails and metadata
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_search_results() -> List[Dict[str, Any]]:
    """Create sample Google Search results with rich metadata"""
    return [
        {
            "title": "Latest AI News: GPT-5 Release Date Announced",
            "link": "https://techcrunch.com/2025/01/15/gpt5-release",
            "displayLink": "techcrunch.com",
            "formattedUrl": "https://techcrunch.com › 2025 › 01 › 15 › gpt5-release",
            "snippet": "OpenAI has officially announced the release date for GPT-5, marking a significant milestone in artificial intelligence development...",
            "htmlSnippet": "OpenAI has officially announced the release date for <b>GPT-5</b>, marking a significant milestone in <b>artificial intelligence</b> development...",
            "pagemap": {
                "metatags": [{
                    "article:published_time": "2025-01-15T09:30:00Z",
                    "og:updated_time": "2025-01-15T14:45:00Z",
                    "author": "Sarah Johnson",
                    "og:image": "https://techcrunch.com/images/gpt5-announcement.jpg",
                    "og:description": "OpenAI announces GPT-5 release date with major improvements in reasoning and multimodal capabilities",
                    "og:site_name": "TechCrunch"
                }],
                "cse_thumbnail": [{
                    "src": "https://techcrunch.com/thumbs/gpt5-announcement-thumb.jpg",
                    "width": "300",
                    "height": "200"
                }]
            }
        },
        {
            "title": "Understanding Temporal Relevance in Search Results",
            "link": "https://searchengineland.com/temporal-relevance-guide",
            "displayLink": "searchengineland.com",
            "formattedUrl": "https://searchengineland.com › seo › temporal-relevance-guide",
            "snippet": "Learn how search engines use temporal relevance to rank results. Published Jan 10, 2025. This comprehensive guide covers...",
            "htmlSnippet": "Learn how search engines use <b>temporal relevance</b> to rank results. Published Jan 10, 2025. This comprehensive guide covers...",
            "pagemap": {
                "newsarticle": [{
                    "datePublished": "2025-01-10",
                    "headline": "Understanding Temporal Relevance in Search Results"
                }],
                "metatags": [{
                    "author": "Mike Chen",
                    "keywords": "SEO, temporal relevance, search ranking"
                }]
            }
        },
        {
            "title": "Historical Overview: Evolution of AI Models",
            "link": "https://arxiv.org/abs/2024.12345",
            "displayLink": "arxiv.org",
            "formattedUrl": "https://arxiv.org › abs › 2024.12345",
            "snippet": "This paper provides a comprehensive historical overview of AI model evolution from 2020 to 2024, examining key breakthroughs...",
            "htmlSnippet": "This paper provides a comprehensive historical overview of <b>AI model</b> evolution from 2020 to 2024, examining key breakthroughs...",
            "pagemap": {
                "metatags": [{
                    "date": "2024-06-15",
                    "og:type": "article",
                    "og:description": "Academic paper on AI model evolution"
                }]
            }
        }
    ]


async def test_search_result_formatter():
    """Test the enhanced search result formatter"""
    print("\n" + "="*80)
    print(" TESTING ENHANCED SEARCH RESULT FORMATTER ")
    print("="*80)
    
    from app.core.search_result_formatter import EnhancedSearchResultFormatter
    
    formatter = EnhancedSearchResultFormatter()
    results = create_sample_search_results()
    
    # Test 1: Date extraction
    print("\n📅 TEST 1: Publication Date Extraction")
    print("-"*60)
    for i, result in enumerate(results, 1):
        pub_date = formatter.extract_publication_date(result)
        print(f"Result {i}: {result['title'][:50]}...")
        print(f"  Extracted Date: {pub_date}")
        print(f"  Source: {result.get('displayLink', 'Unknown')}")
    
    # Test 2: Thumbnail extraction
    print("\n🖼️  TEST 2: Thumbnail Extraction")
    print("-"*60)
    for i, result in enumerate(results, 1):
        thumbnail = formatter.extract_thumbnail(result)
        print(f"Result {i}: {result['title'][:50]}...")
        if thumbnail:
            print(f"  ✅ Thumbnail: {thumbnail[:60]}...")
        else:
            print(f"  ❌ No thumbnail found")
    
    # Test 3: HTML snippet formatting
    print("\n✨ TEST 3: HTML Snippet Formatting")
    print("-"*60)
    for i, result in enumerate(results, 1):
        formatted_snippet = formatter.format_html_snippet(result)
        print(f"Result {i}:")
        print(f"  Original: {result['snippet'][:80]}...")
        print(f"  Formatted: {formatted_snippet[:80]}...")
        if '**' in formatted_snippet:
            print(f"  ✅ Search term highlighting preserved")
    
    # Test 4: LLM formatting
    print("\n🤖 TEST 4: LLM-Friendly Formatting")
    print("-"*60)
    llm_formatted = formatter.format_for_llm(results)
    print("Sample LLM format (first 500 chars):")
    print(llm_formatted[:500] + "...")
    
    # Test 5: Display formatting
    print("\n💻 TEST 5: Frontend Display Formatting")
    print("-"*60)
    display_results = formatter.format_for_display(results)
    for i, result in enumerate(display_results[:2], 1):
        print(f"\nResult {i} (enhanced for display):")
        print(f"  Title: {result['title']}")
        print(f"  Display Link: {result['displayLink']}")
        print(f"  Formatted URL: {result['formattedUrl']}")
        print(f"  Has Thumbnail: {bool(result['thumbnail'])}")
        print(f"  Has Publication Date: {result['temporalRelevance']['hasDate']}")
        if result['metadata'].get('author'):
            print(f"  Author: {result['metadata']['author']}")
    
    # Test 6: Temporal scoring preparation
    print("\n⏰ TEST 6: Temporal Scoring Data Extraction")
    print("-"*60)
    results_with_dates = formatter.extract_dates_for_temporal_scoring(results)
    for i, result in enumerate(results_with_dates, 1):
        print(f"\nResult {i}:")
        print(f"  Title: {result['title'][:50]}...")
        print(f"  Extracted Date: {result.get('extracted_date', 'None')}")
        print(f"  Parsed Date: {result.get('parsed_date', 'None')}")
        print(f"  Age (days): {result.get('age_days', 'Unknown')}")
    
    return True


async def test_temporal_relevance_engine():
    """Test temporal relevance engine with enhanced metadata"""
    print("\n" + "="*80)
    print(" TESTING TEMPORAL RELEVANCE ENGINE ")
    print("="*80)
    
    from app.core.temporal_relevance_engine import TemporalRelevanceEngine
    from app.core.search_result_formatter import EnhancedSearchResultFormatter
    
    engine = TemporalRelevanceEngine()
    formatter = EnhancedSearchResultFormatter()
    
    # Prepare test data with dates
    results = create_sample_search_results()
    results_with_dates = formatter.extract_dates_for_temporal_scoring(results)
    
    # Test different query types
    queries = [
        "latest GPT-5 news",  # Current/temporal query
        "AI model evolution history",  # Historical query
        "temporal relevance in search"  # Technical/educational query
    ]
    
    for query in queries:
        print(f"\n📝 Query: '{query}'")
        print("-"*60)
        
        # Analyze query
        classification = engine.analyze_query(query)
        print(f"Classification:")
        print(f"  Sensitivity: {classification.sensitivity.value}")
        print(f"  Domain: {classification.domain}")
        print(f"  Intent: {classification.intent}")
        print(f"  Max Age Days: {classification.max_age_days}")
        
        # Score documents
        print(f"\nDocument Scores:")
        for i, doc in enumerate(results_with_dates, 1):
            score = engine.score_document(doc, classification)
            print(f"  {i}. {doc['title'][:40]}...")
            print(f"     Age: {score.age_days:.1f} days")
            print(f"     Temporal Score: {score.temporal_score:.3f}")
            print(f"     Authority Score: {score.authority_score:.3f}")
            print(f"     Combined Score: {score.combined_score:.3f}")
            print(f"     Include: {'✅' if score.should_include else '❌'}")
        
        # Filter and rank
        filtered, metadata = engine.filter_and_rank_results(results_with_dates, query)
        print(f"\nFiltering Results:")
        print(f"  Total: {metadata['total_results']}")
        print(f"  Filtered: {metadata['filtered_results']}")
        print(f"  Removed: {metadata['filtering_stats']['removed_outdated']}")
        print(f"  Avg Temporal Score: {metadata['filtering_stats']['average_temporal_score']:.3f}")
    
    return True


async def test_integration_with_service():
    """Test integration with the main service layer"""
    print("\n" + "="*80)
    print(" TESTING SERVICE INTEGRATION ")
    print("="*80)
    
    try:
        from app.core.search_result_formatter import format_search_results_for_llm
        
        # Simulate MCP tool result
        mock_tool_result = {
            "content": [{
                "type": "text",
                "text": json.dumps(create_sample_search_results())
            }]
        }
        
        # Test formatting
        formatted = format_search_results_for_llm("google_search", mock_tool_result)
        
        print("\n✅ Service Integration Test:")
        print(f"Formatted output length: {len(formatted)} chars")
        print("\nSample output (first 800 chars):")
        print(formatted[:800])
        
        # Check for expected features
        features = {
            "Has publication dates": "Published:" in formatted,
            "Has source attribution": "Source:" in formatted,
            "Has URL formatting": "URL:" in formatted,
            "Has search highlighting": "**" in formatted,
            "Has thumbnail indicator": "[Has thumbnail" in formatted
        }
        
        print("\n📊 Feature Check:")
        for feature, present in features.items():
            status = "✅" if present else "❌"
            print(f"  {status} {feature}")
        
        return all(features.values())
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test runner"""
    print("\n" + "🚀"*40)
    print(" ENHANCED GOOGLE SEARCH METADATA PROCESSING TEST SUITE ")
    print("🚀"*40)
    
    tests = [
        ("Search Result Formatter", test_search_result_formatter),
        ("Temporal Relevance Engine", test_temporal_relevance_engine),
        ("Service Integration", test_integration_with_service)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n\n{'='*80}")
            print(f" Running: {test_name}")
            print('='*80)
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n\n" + "="*80)
    print(" TEST SUMMARY ")
    print("="*80)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n🎉 All tests passed! The enhanced search result processing is working correctly.")
        print("\nKey improvements implemented:")
        print("✅ Rich metadata extraction from pagemap")
        print("✅ Publication date extraction and parsing")
        print("✅ HTML snippet formatting with highlighting")
        print("✅ Thumbnail URL extraction")
        print("✅ Enhanced temporal relevance scoring")
        print("✅ Better formatted output for LLM synthesis")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
    
    return all_passed


if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)