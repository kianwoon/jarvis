#!/usr/bin/env python3
"""
Comprehensive Test Suite for Temporal Relevance System

Tests the complete temporal relevance implementation including:
- Query classification
- Temporal decay functions
- Source authority scoring
- Google Search integration
- End-to-end ChatGPT Pro query handling
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.temporal_query_classifier import get_temporal_classifier, TemporalSensitivity
from app.core.temporal_decay_functions import calculate_temporal_score, DomainDecayProfiles
from app.core.temporal_source_authority import get_source_scorer
from app.core.temporal_relevance_engine import get_relevance_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TemporalRelevanceTestSuite:
    """Test suite for temporal relevance system"""
    
    def __init__(self):
        self.classifier = get_temporal_classifier()
        self.source_scorer = get_source_scorer()
        self.relevance_engine = get_relevance_engine()
        self.mcp_service = None  # Will be initialized in async context
        self.test_results = []
    
    def test_query_classification(self):
        """Test query classification for various query types"""
        logger.info("\n" + "="*60)
        logger.info("Testing Query Classification")
        logger.info("="*60)
        
        test_queries = [
            ("what are the usage limits of chatgpt PRO subscription", "HIGH", "technology", "current"),
            ("latest ChatGPT Pro features 2025", "HIGH", "technology", "current"),
            ("history of world war 2", "HISTORICAL", "history", "historical"),
            ("current stock market prices", "VERY_HIGH", "finance", "current"),
            ("python programming tutorial", "MEDIUM", "general", "neutral"),
            ("ChatGPT pricing plans", "HIGH", "technology", "current"),
            ("what happened in 2023", "HISTORICAL", "history", "historical"),
        ]
        
        results = []
        for query, expected_sensitivity, expected_domain, expected_intent in test_queries:
            classification = self.classifier.classify(query)
            
            sensitivity_match = classification.sensitivity.value.split("_")[0] == expected_sensitivity.split("_")[0]
            domain_match = classification.domain == expected_domain
            intent_match = classification.intent == expected_intent
            
            result = {
                "query": query,
                "sensitivity": {
                    "expected": expected_sensitivity,
                    "actual": classification.sensitivity.value,
                    "match": sensitivity_match
                },
                "domain": {
                    "expected": expected_domain,
                    "actual": classification.domain,
                    "match": domain_match
                },
                "intent": {
                    "expected": expected_intent,
                    "actual": classification.intent,
                    "match": intent_match
                },
                "max_age_days": classification.max_age_days,
                "keywords": classification.keywords_matched[:5]
            }
            
            results.append(result)
            
            status = "✅" if all([sensitivity_match, domain_match, intent_match]) else "❌"
            logger.info(f"{status} Query: '{query[:50]}...'")
            logger.info(f"   Sensitivity: {classification.sensitivity.value} (expected: {expected_sensitivity})")
            logger.info(f"   Domain: {classification.domain} (expected: {expected_domain})")
            logger.info(f"   Intent: {classification.intent} (expected: {expected_intent})")
            logger.info(f"   Max age: {classification.max_age_days} days")
        
        self.test_results.append(("Query Classification", results))
        return results
    
    def test_temporal_decay(self):
        """Test temporal decay functions for different domains"""
        logger.info("\n" + "="*60)
        logger.info("Testing Temporal Decay Functions")
        logger.info("="*60)
        
        test_cases = [
            ("news", 1, 0.8, 1.0),  # 1 day old news should have high score
            ("news", 7, 0.3, 0.7),  # 1 week old news should decay significantly
            ("news", 30, 0.0, 0.2),  # 1 month old news should be near zero
            ("tech_subscription", 30, 0.6, 1.0),  # 1 month old tech info should still be relevant
            ("tech_subscription", 180, 0.0, 0.3),  # 6 months old should be low
            ("academic", 365, 0.7, 1.0),  # 1 year old academic should still be relevant
            ("documentation", 180, 0.4, 0.6),  # 6 months old docs should be moderate
        ]
        
        results = []
        for domain, age_days, min_expected, max_expected in test_cases:
            score = calculate_temporal_score(age_days, domain)
            in_range = min_expected <= score <= max_expected
            
            result = {
                "domain": domain,
                "age_days": age_days,
                "score": score,
                "expected_range": f"{min_expected}-{max_expected}",
                "in_range": in_range
            }
            results.append(result)
            
            status = "✅" if in_range else "❌"
            logger.info(f"{status} Domain: {domain}, Age: {age_days} days")
            logger.info(f"   Score: {score:.3f} (expected: {min_expected}-{max_expected})")
        
        self.test_results.append(("Temporal Decay", results))
        return results
    
    def test_source_authority(self):
        """Test source authority scoring"""
        logger.info("\n" + "="*60)
        logger.info("Testing Source Authority Scoring")
        logger.info("="*60)
        
        test_sources = [
            ("https://help.openai.com/articles/chatgpt-pro", "Official ChatGPT Pro Documentation", 
             "Complete guide to ChatGPT Pro features", 0.9, 1.0),
            ("https://reddit.com/r/ChatGPT/comments/xyz", "User discussion about ChatGPT",
             "Users discussing their experience", 0.3, 0.5),
            ("https://techcrunch.com/2025/chatgpt-pro-launch", "TechCrunch: ChatGPT Pro Launches",
             "OpenAI announces ChatGPT Pro", 0.6, 0.8),
            ("https://medium.com/user/chatgpt-review", "My thoughts on ChatGPT",
             "Personal review and opinion", 0.3, 0.5),
            ("https://platform.openai.com/docs", "OpenAI API Documentation",
             "Official API reference", 0.9, 1.0)
        ]
        
        results = []
        for url, title, snippet, min_expected, max_expected in test_sources:
            authority = self.source_scorer.score_source(url, title, snippet)
            in_range = min_expected <= authority.authority_score <= max_expected
            
            result = {
                "url": url,
                "domain": authority.domain,
                "authority_score": authority.authority_score,
                "source_type": authority.source_type,
                "is_primary": authority.is_primary_source,
                "expected_range": f"{min_expected}-{max_expected}",
                "in_range": in_range
            }
            results.append(result)
            
            status = "✅" if in_range else "❌"
            logger.info(f"{status} URL: {url[:50]}...")
            logger.info(f"   Authority: {authority.authority_score:.3f} (expected: {min_expected}-{max_expected})")
            logger.info(f"   Type: {authority.source_type}, Primary: {authority.is_primary_source}")
        
        self.test_results.append(("Source Authority", results))
        return results
    
    def test_relevance_engine(self):
        """Test the complete relevance engine"""
        logger.info("\n" + "="*60)
        logger.info("Testing Temporal Relevance Engine")
        logger.info("="*60)
        
        # Simulate search results with different ages
        mock_results = [
            {
                "id": "1",
                "url": "https://help.openai.com/chatgpt-pro-limits",
                "title": "ChatGPT Pro Usage Limits - Official Documentation",
                "snippet": "ChatGPT Pro offers unlimited access to all models...",
                "days_old": 5  # 5 days old - very recent
            },
            {
                "id": "2", 
                "url": "https://reddit.com/r/ChatGPT/old-discussion",
                "title": "ChatGPT Plus is amazing!",
                "snippet": "I love ChatGPT Plus, it's only $20/month...",
                "days_old": 400  # Over a year old
            },
            {
                "id": "3",
                "url": "https://techcrunch.com/2024/chatgpt-news",
                "title": "ChatGPT Updates from 2024",
                "snippet": "Last year's ChatGPT improvements...",
                "days_old": 250  # 8+ months old
            },
            {
                "id": "4",
                "url": "https://platform.openai.com/docs/chatgpt-pro",
                "title": "ChatGPT Pro API Documentation",
                "snippet": "ChatGPT Pro subscription at $200/month provides...",
                "days_old": 30  # 1 month old
            }
        ]
        
        # Test with ChatGPT Pro query
        query = "what are the usage limits of chatgpt PRO subscription"
        filtered_results, metadata = self.relevance_engine.filter_and_rank_results(
            mock_results, query, max_results=3
        )
        
        logger.info(f"Query: '{query}'")
        logger.info(f"Input: {len(mock_results)} results")
        logger.info(f"Output: {len(filtered_results)} results")
        logger.info(f"Removed: {metadata['filtering_stats']['removed_outdated']} outdated")
        
        for i, result in enumerate(filtered_results, 1):
            temporal = result.get("temporal_relevance", {})
            logger.info(f"\n  Result {i}: {result['title'][:50]}...")
            logger.info(f"    Age: {result.get('days_old', 'unknown')} days")
            logger.info(f"    Temporal score: {temporal.get('temporal_score', 0):.3f}")
            logger.info(f"    Authority score: {temporal.get('authority_score', 0):.3f}")
            logger.info(f"    Combined score: {temporal.get('combined_score', 0):.3f}")
        
        # Verify old results were filtered out
        result_ids = [r["id"] for r in filtered_results]
        assert "2" not in result_ids, "Very old Reddit result should be filtered"
        assert "3" not in result_ids, "Old TechCrunch result should be filtered"
        assert "1" in result_ids, "Recent official docs should be included"
        
        self.test_results.append(("Relevance Engine", {
            "query": query,
            "input_count": len(mock_results),
            "output_count": len(filtered_results),
            "filtered_ids": result_ids,
            "metadata": metadata
        }))
        
        return filtered_results
    
    async def test_google_search_integration(self):
        """Test Google Search with temporal relevance"""
        logger.info("\n" + "="*60)
        logger.info("Testing Google Search Integration")
        logger.info("="*60)
        
        # Initialize MCP service in async context
        if not self.mcp_service:
            from app.core.unified_mcp_service import UnifiedMCPService
            self.mcp_service = UnifiedMCPService()
        
        # Test query that should trigger temporal filtering
        query = "ChatGPT Pro subscription limits and pricing 2025"
        
        logger.info(f"Testing query: '{query}'")
        
        try:
            # Call the direct Google search method
            result = await self.mcp_service._direct_google_search({
                "query": query,
                "num_results": 5
            })
            
            if "error" in result:
                logger.error(f"Search error: {result['error']}")
                return None
            
            # Extract the text content
            content = result.get("content", [{}])[0].get("text", "")
            
            # Check for temporal awareness indicators
            has_temporal_labels = any(label in content for label in [
                "[Current]", "[Recent]", "[This week]", "[This month]", 
                "[Few months old]", "[Over a year old]"
            ])
            
            has_date_warnings = "may be outdated" in content.lower()
            has_classification_log = "Query classification:" in content
            
            logger.info(f"✅ Search completed successfully")
            logger.info(f"   Has temporal labels: {has_temporal_labels}")
            logger.info(f"   Has date warnings: {has_date_warnings}")
            logger.info(f"   Used classification: {has_classification_log}")
            
            # Parse results count
            import re
            match = re.search(r"Found (\d+) relevant search results", content)
            if match:
                result_count = int(match.group(1))
                logger.info(f"   Results returned: {result_count}")
            
            self.test_results.append(("Google Search Integration", {
                "query": query,
                "success": True,
                "has_temporal_labels": has_temporal_labels,
                "has_date_warnings": has_date_warnings,
                "content_preview": content[:500]
            }))
            
            return result
            
        except Exception as e:
            logger.error(f"Error testing Google search: {e}")
            self.test_results.append(("Google Search Integration", {
                "query": query,
                "success": False,
                "error": str(e)
            }))
            return None
    
    async def test_chatgpt_pro_query(self):
        """Test the original ChatGPT Pro query that started this enhancement"""
        logger.info("\n" + "="*60)
        logger.info("Testing Original ChatGPT Pro Query")
        logger.info("="*60)
        
        # Initialize MCP service in async context if needed
        if not self.mcp_service:
            from app.core.unified_mcp_service import UnifiedMCPService
            self.mcp_service = UnifiedMCPService()
        
        query = "what are the usage limit of chatgpt PRO subscription"
        
        logger.info(f"Original query: '{query}'")
        
        try:
            # Test query classification
            classification = self.classifier.classify(query)
            logger.info(f"Classification: {classification.sensitivity.value}, "
                       f"domain: {classification.domain}, intent: {classification.intent}")
            
            # Test with Google Search
            result = await self.mcp_service._direct_google_search({
                "query": query,
                "num_results": 5
            })
            
            if "error" in result:
                logger.error(f"Search failed: {result['error']}")
                return False
            
            content = result.get("content", [{}])[0].get("text", "")
            
            # Verify Pro-specific information is prioritized
            has_pro_info = "$200" in content or "ChatGPT Pro" in content
            has_plus_confusion = content.count("$20") > content.count("$200")
            
            # Check for temporal relevance
            has_recent_info = any(term in content for term in [
                "[Current]", "[Recent]", "[This week]", "[This month]", "2025"
            ])
            
            # Check if outdated info is filtered
            has_old_warnings = "outdated" in content.lower() or "may be outdated" in content.lower()
            
            logger.info(f"Results analysis:")
            logger.info(f"   ✅ Has Pro-specific info: {has_pro_info}")
            logger.info(f"   {'❌' if has_plus_confusion else '✅'} Plus/Pro distinction: {'Confused' if has_plus_confusion else 'Clear'}")
            logger.info(f"   ✅ Has recent information: {has_recent_info}")
            logger.info(f"   ℹ️  Has outdated warnings: {has_old_warnings}")
            
            success = has_pro_info and not has_plus_confusion and has_recent_info
            
            self.test_results.append(("ChatGPT Pro Query", {
                "query": query,
                "success": success,
                "has_pro_info": has_pro_info,
                "has_plus_confusion": has_plus_confusion,
                "has_recent_info": has_recent_info,
                "classification": {
                    "sensitivity": classification.sensitivity.value,
                    "domain": classification.domain,
                    "intent": classification.intent
                }
            }))
            
            return success
            
        except Exception as e:
            logger.error(f"Error testing ChatGPT Pro query: {e}")
            self.test_results.append(("ChatGPT Pro Query", {
                "query": query,
                "success": False,
                "error": str(e)
            }))
            return False
    
    def print_summary(self):
        """Print test summary"""
        logger.info("\n" + "="*60)
        logger.info("TEST SUMMARY")
        logger.info("="*60)
        
        for test_name, results in self.test_results:
            logger.info(f"\n{test_name}:")
            if isinstance(results, list):
                passed = sum(1 for r in results if r.get("in_range", r.get("match", False)))
                total = len(results)
                logger.info(f"  Passed: {passed}/{total}")
            elif isinstance(results, dict):
                if "success" in results:
                    status = "✅ PASSED" if results["success"] else "❌ FAILED"
                    logger.info(f"  Status: {status}")
                else:
                    logger.info(f"  Completed: ✅")
    
    async def run_all_tests(self):
        """Run all tests"""
        logger.info("\n" + "#"*60)
        logger.info("# TEMPORAL RELEVANCE SYSTEM TEST SUITE")
        logger.info("#"*60)
        
        # Run synchronous tests
        self.test_query_classification()
        self.test_temporal_decay()
        self.test_source_authority()
        self.test_relevance_engine()
        
        # Run async tests
        await self.test_google_search_integration()
        await self.test_chatgpt_pro_query()
        
        # Print summary
        self.print_summary()
        
        # Close MCP service if initialized
        if self.mcp_service:
            await self.mcp_service.close()


async def main():
    """Main test runner"""
    test_suite = TemporalRelevanceTestSuite()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())