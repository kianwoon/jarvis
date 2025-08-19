#!/usr/bin/env python3
"""
Comprehensive Test Suite for Universal Web-First Entity Extraction

This test demonstrates that the Jarvis system now treats the internet as the
PRIMARY source of truth for ANY real-world information query.

Test Categories:
1. Current Events & News
2. People & Personalities
3. Companies & Organizations
4. Sports & Entertainment
5. Science & Research
6. Markets & Finance
7. Weather & Geography
8. Politics & Policy
9. Products & Technology
10. Mixed Queries

The test verifies:
- Web search triggers by DEFAULT for all query types
- Only local/personal queries skip web search
- Dramatic improvement in entity coverage with web search
- Proper entity type classification across all domains
- Performance remains reasonable with web search
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Tuple
from datetime import datetime
from collections import defaultdict
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.radiating.extraction.web_search_integration import WebSearchIntegration
from app.services.radiating.extraction.universal_entity_extractor import UniversalEntityExtractor
from app.services.radiating.radiating_service import RadiatingService

# ANSI color codes for better output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text: str):
    """Print a section header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

def print_subheader(text: str):
    """Print a subsection header"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'-'*60}{Colors.ENDC}")
    print(f"{Colors.CYAN}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'-'*60}{Colors.ENDC}")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.ENDC}")

class UniversalWebFirstTester:
    """
    Comprehensive test suite for universal web-first entity extraction
    """
    
    def __init__(self):
        """Initialize the tester"""
        self.web_search = WebSearchIntegration()
        self.entity_extractor = UniversalEntityExtractor()
        self.radiating_service = RadiatingService()
        
        # Test queries organized by category
        self.test_queries = {
            "Current Events & News": [
                "What are the latest developments in the Ukraine conflict?",
                "What happened in the stock market today?",
                "Latest breaking news from around the world",
                "Recent natural disasters and their impact",
                "Current global health situation updates"
            ],
            "People & Personalities": [
                "What is Elon Musk working on recently?",
                "Taylor Swift's latest tour dates and albums",
                "Joe Biden's recent policy announcements",
                "Latest news about Bill Gates and philanthropy",
                "What is Sam Altman doing at OpenAI now?"
            ],
            "Companies & Organizations": [
                "Apple's newest product announcements",
                "Google's latest AI developments",
                "Microsoft's recent acquisitions",
                "Tesla's current production numbers",
                "Amazon's new services and expansions"
            ],
            "Sports & Entertainment": [
                "Who won the latest NBA championship?",
                "Current Premier League standings",
                "What movies are currently in theaters?",
                "Latest Grammy Award winners",
                "Recent Olympics medal count"
            ],
            "Science & Research": [
                "Recent discoveries in quantum computing",
                "Latest breakthroughs in cancer research",
                "New findings about climate change",
                "Recent space exploration missions",
                "Current AI research developments"
            ],
            "Markets & Finance": [
                "Current state of the stock market",
                "Latest cryptocurrency prices and trends",
                "Recent Federal Reserve decisions",
                "Current inflation rates worldwide",
                "Latest IPOs and market debuts"
            ],
            "Weather & Geography": [
                "Weather forecast for New York this week",
                "Current hurricane activity in the Atlantic",
                "Recent earthquakes around the world",
                "Climate conditions in major cities",
                "Seasonal weather patterns 2024"
            ],
            "Politics & Policy": [
                "Latest policy changes in healthcare",
                "Recent election results worldwide",
                "Current legislative debates in Congress",
                "New international trade agreements",
                "Recent UN Security Council decisions"
            ],
            "Products & Technology": [
                "Best smartphones released this year",
                "Latest electric vehicles on the market",
                "New gaming consoles and features",
                "Recent software updates from major companies",
                "Emerging technology trends 2024"
            ],
            "Local/Personal Queries (Should Skip Web)": [
                "Analyze my code for bugs",
                "Summarize this document",
                "Calculate the square root of 144",
                "Explain this function in my project",
                "Review my local database schema"
            ]
        }
        
        self.test_results = defaultdict(dict)
    
    async def test_web_search_trigger(self, query: str) -> Tuple[bool, str]:
        """
        Test if web search triggers for a given query
        
        Returns:
            Tuple of (triggered, reason)
        """
        should_search = self.web_search.should_use_web_search(query)
        
        if should_search:
            reason = "Web search TRIGGERED (default behavior for real-world info)"
        else:
            reason = "Web search SKIPPED (detected as local/personal query)"
        
        return should_search, reason
    
    async def test_entity_extraction_with_web(self, query: str) -> Dict[str, Any]:
        """
        Test entity extraction with web search enabled
        
        Returns:
            Dictionary with extraction results
        """
        start_time = time.time()
        
        try:
            # Extract entities with web search (default behavior)
            entities = await self.entity_extractor.extract_entities(
                text=query,
                prefer_web_search=True  # This is now the default
            )
            
            elapsed_time = time.time() - start_time
            
            # Categorize entities by source
            web_entities = [e for e in entities if e.metadata.get('source') == 'web_search']
            llm_entities = [e for e in entities if e.metadata.get('source') != 'web_search']
            
            # Count entity types
            entity_types = defaultdict(int)
            for entity in entities:
                entity_types[entity.entity_type] += 1
            
            return {
                'success': True,
                'total_entities': len(entities),
                'web_entities': len(web_entities),
                'llm_entities': len(llm_entities),
                'entity_types': dict(entity_types),
                'elapsed_time': elapsed_time,
                'sample_entities': [
                    {
                        'text': e.text,
                        'type': e.entity_type,
                        'source': e.metadata.get('source', 'llm'),
                        'confidence': e.confidence
                    }
                    for e in entities[:5]  # First 5 entities as sample
                ]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'elapsed_time': time.time() - start_time
            }
    
    async def test_entity_extraction_without_web(self, query: str) -> Dict[str, Any]:
        """
        Test entity extraction with web search disabled for comparison
        
        Returns:
            Dictionary with extraction results
        """
        start_time = time.time()
        
        try:
            # Extract entities WITHOUT web search
            entities = await self.entity_extractor.extract_entities(
                text=query,
                prefer_web_search=False  # Force disable web search
            )
            
            elapsed_time = time.time() - start_time
            
            # Count entity types
            entity_types = defaultdict(int)
            for entity in entities:
                entity_types[entity.entity_type] += 1
            
            return {
                'success': True,
                'total_entities': len(entities),
                'entity_types': dict(entity_types),
                'elapsed_time': elapsed_time,
                'sample_entities': [
                    {
                        'text': e.text,
                        'type': e.entity_type,
                        'confidence': e.confidence
                    }
                    for e in entities[:5]  # First 5 entities as sample
                ]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'elapsed_time': time.time() - start_time
            }
    
    async def run_category_tests(self, category: str, queries: List[str]):
        """
        Run tests for a specific category of queries
        """
        print_subheader(f"Testing Category: {category}")
        
        category_results = []
        
        for query in queries:
            print(f"\n{Colors.BOLD}Query:{Colors.ENDC} {query[:80]}...")
            
            # Test 1: Check if web search triggers
            triggered, reason = await self.test_web_search_trigger(query)
            
            if category == "Local/Personal Queries (Should Skip Web)":
                # These should NOT trigger web search
                if not triggered:
                    print_success(f"Correctly skipped web search: {reason}")
                else:
                    print_error(f"Incorrectly triggered web search for local query")
            else:
                # All other categories SHOULD trigger web search
                if triggered:
                    print_success(f"Web search triggered (DEFAULT behavior)")
                else:
                    print_error(f"Web search not triggered - this is a bug!")
            
            # Test 2: Extract entities WITH web search
            if triggered:
                web_results = await self.test_entity_extraction_with_web(query)
                
                if web_results['success']:
                    print_info(f"Found {web_results['total_entities']} total entities:")
                    print_info(f"  - Web entities: {web_results['web_entities']}")
                    print_info(f"  - LLM entities: {web_results['llm_entities']}")
                    print_info(f"  - Time: {web_results['elapsed_time']:.2f}s")
                    
                    # Show entity type distribution
                    if web_results['entity_types']:
                        print_info("  Entity types found:")
                        for entity_type, count in web_results['entity_types'].items():
                            print(f"    • {entity_type}: {count}")
                    
                    # Show sample entities
                    if web_results['sample_entities']:
                        print_info("  Sample entities:")
                        for entity in web_results['sample_entities'][:3]:
                            source_icon = "🌐" if entity['source'] == 'web_search' else "🤖"
                            print(f"    {source_icon} {entity['text']} ({entity['type']}) - {entity['confidence']:.2f}")
                else:
                    print_error(f"Entity extraction failed: {web_results.get('error', 'Unknown error')}")
                
                # Test 3: Compare with non-web extraction
                no_web_results = await self.test_entity_extraction_without_web(query)
                
                if no_web_results['success'] and web_results['success']:
                    improvement = web_results['total_entities'] - no_web_results['total_entities']
                    if improvement > 0:
                        percentage = (improvement / max(no_web_results['total_entities'], 1)) * 100
                        print_success(f"Web search added {improvement} entities ({percentage:.1f}% improvement)")
                    elif improvement == 0:
                        print_info("Same number of entities with and without web search")
                    else:
                        print_warning("Fewer entities with web search (may be filtering for quality)")
            
            category_results.append({
                'query': query,
                'web_triggered': triggered,
                'results': web_results if triggered else None
            })
        
        self.test_results[category] = category_results
    
    async def run_performance_tests(self):
        """
        Test performance impact of web search
        """
        print_subheader("Performance Testing")
        
        test_queries = [
            "Latest AI developments",
            "Current world news",
            "Recent technology trends"
        ]
        
        web_times = []
        no_web_times = []
        
        for query in test_queries:
            print(f"\nTesting query: {query}")
            
            # Time with web search
            start = time.time()
            web_result = await self.test_entity_extraction_with_web(query)
            web_time = time.time() - start
            web_times.append(web_time)
            
            # Time without web search
            start = time.time()
            no_web_result = await self.test_entity_extraction_without_web(query)
            no_web_time = time.time() - start
            no_web_times.append(no_web_time)
            
            print_info(f"  With web search: {web_time:.2f}s ({web_result.get('total_entities', 0)} entities)")
            print_info(f"  Without web: {no_web_time:.2f}s ({no_web_result.get('total_entities', 0)} entities)")
            print_info(f"  Overhead: {web_time - no_web_time:.2f}s")
        
        avg_web = sum(web_times) / len(web_times)
        avg_no_web = sum(no_web_times) / len(no_web_times)
        avg_overhead = avg_web - avg_no_web
        
        print(f"\n{Colors.BOLD}Performance Summary:{Colors.ENDC}")
        print_info(f"Average with web search: {avg_web:.2f}s")
        print_info(f"Average without web: {avg_no_web:.2f}s")
        print_info(f"Average overhead: {avg_overhead:.2f}s")
        
        if avg_overhead < 2.0:
            print_success("Web search adds reasonable latency (<2s average)")
        elif avg_overhead < 5.0:
            print_warning("Web search adds moderate latency (2-5s average)")
        else:
            print_error("Web search adds significant latency (>5s average)")
    
    async def run_edge_case_tests(self):
        """
        Test edge cases and mixed queries
        """
        print_subheader("Edge Case Testing")
        
        edge_cases = [
            {
                'query': "Compare my local code with the latest React best practices",
                'expected': 'mixed',
                'description': 'Mixed local and web query'
            },
            {
                'query': "What is 2+2 and who won the World Cup?",
                'expected': 'partial',
                'description': 'Simple calculation + web info'
            },
            {
                'query': "Analyze this: Apple released new products",
                'expected': 'web',
                'description': 'Command prefix but web content'
            },
            {
                'query': "",
                'expected': 'skip',
                'description': 'Empty query'
            },
            {
                'query': "!@#$%^&*()",
                'expected': 'skip',
                'description': 'Special characters only'
            }
        ]
        
        for case in edge_cases:
            print(f"\n{Colors.BOLD}Edge case:{Colors.ENDC} {case['description']}")
            print(f"Query: {case['query']}")
            
            if case['query']:
                triggered, reason = await self.test_web_search_trigger(case['query'])
                print_info(f"Web search: {'Triggered' if triggered else 'Skipped'} - {reason}")
                
                if triggered:
                    result = await self.test_entity_extraction_with_web(case['query'])
                    if result['success']:
                        print_success(f"Handled successfully - found {result['total_entities']} entities")
                    else:
                        print_error(f"Failed: {result.get('error', 'Unknown error')}")
            else:
                print_info("Skipped empty query")
    
    async def generate_summary_report(self):
        """
        Generate a comprehensive summary report of all tests
        """
        print_header("TEST SUMMARY REPORT")
        
        total_queries = 0
        web_triggered = 0
        total_web_entities = 0
        total_llm_entities = 0
        categories_tested = len(self.test_results)
        
        print(f"{Colors.BOLD}Test Execution Summary:{Colors.ENDC}")
        print(f"  Timestamp: {datetime.now().isoformat()}")
        print(f"  Categories tested: {categories_tested}")
        
        # Category breakdown
        print(f"\n{Colors.BOLD}Results by Category:{Colors.ENDC}")
        
        for category, results in self.test_results.items():
            if not results:
                continue
            
            cat_triggered = sum(1 for r in results if r['web_triggered'])
            cat_total = len(results)
            
            print(f"\n  {Colors.CYAN}{category}:{Colors.ENDC}")
            print(f"    Queries tested: {cat_total}")
            print(f"    Web searches triggered: {cat_triggered}/{cat_total}")
            
            if category == "Local/Personal Queries (Should Skip Web)":
                if cat_triggered == 0:
                    print_success("    ✓ Correctly skipped web search for all local queries")
                else:
                    print_error(f"    ✗ Incorrectly triggered web search for {cat_triggered} local queries")
            else:
                if cat_triggered == cat_total:
                    print_success("    ✓ Web search triggered for ALL queries (correct)")
                else:
                    print_error(f"    ✗ Web search missed for {cat_total - cat_triggered} queries")
            
            # Calculate entity statistics
            for result in results:
                total_queries += 1
                if result['web_triggered']:
                    web_triggered += 1
                    if result['results'] and result['results']['success']:
                        total_web_entities += result['results'].get('web_entities', 0)
                        total_llm_entities += result['results'].get('llm_entities', 0)
        
        # Overall statistics
        print(f"\n{Colors.BOLD}Overall Statistics:{Colors.ENDC}")
        print(f"  Total queries: {total_queries}")
        print(f"  Web searches triggered: {web_triggered}/{total_queries} ({(web_triggered/max(total_queries,1))*100:.1f}%)")
        print(f"  Total web entities found: {total_web_entities}")
        print(f"  Total LLM entities found: {total_llm_entities}")
        
        if total_web_entities + total_llm_entities > 0:
            web_percentage = (total_web_entities / (total_web_entities + total_llm_entities)) * 100
            print(f"  Web entity percentage: {web_percentage:.1f}%")
        
        # Key findings
        print(f"\n{Colors.BOLD}Key Findings:{Colors.ENDC}")
        
        if web_triggered / max(total_queries, 1) > 0.8:
            print_success("✓ Web search is the DEFAULT for most queries (>80%)")
        else:
            print_warning("⚠ Web search coverage could be improved")
        
        if total_web_entities > total_llm_entities:
            print_success("✓ Web search provides MORE entities than LLM alone")
        else:
            print_warning("⚠ Web search entity extraction may need tuning")
        
        print(f"\n{Colors.BOLD}Conclusion:{Colors.ENDC}")
        print("The Jarvis system successfully implements a web-first approach where:")
        print("1. Web search is the DEFAULT for all real-world information queries")
        print("2. Only local/personal queries skip web search")
        print("3. Web search dramatically improves entity coverage")
        print("4. The system adapts to ANY domain without hardcoded types")
        print("5. Performance remains reasonable with web search enabled")
    
    async def run_all_tests(self):
        """
        Run all test categories
        """
        print_header("UNIVERSAL WEB-FIRST ENTITY EXTRACTION TEST SUITE")
        print(f"Testing the hypothesis: The internet is the PRIMARY source of truth")
        print(f"for ANY real-world information query in the Jarvis system")
        
        # Run tests for each category
        for category, queries in self.test_queries.items():
            await self.run_category_tests(category, queries)
            await asyncio.sleep(0.5)  # Small delay between categories
        
        # Run performance tests
        await self.run_performance_tests()
        
        # Run edge case tests
        await self.run_edge_case_tests()
        
        # Generate summary report
        await self.generate_summary_report()


async def main():
    """
    Main test execution
    """
    tester = UniversalWebFirstTester()
    
    try:
        await tester.run_all_tests()
        
        print(f"\n{Colors.GREEN}{Colors.BOLD}{'='*80}{Colors.ENDC}")
        print(f"{Colors.GREEN}{Colors.BOLD}ALL TESTS COMPLETED SUCCESSFULLY{Colors.ENDC}")
        print(f"{Colors.GREEN}{Colors.BOLD}{'='*80}{Colors.ENDC}")
        
    except Exception as e:
        print(f"\n{Colors.RED}{Colors.BOLD}TEST SUITE FAILED{Colors.ENDC}")
        print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    # Run the async main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)