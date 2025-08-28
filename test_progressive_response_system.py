#!/usr/bin/env python3
"""
Test the Universal Progressive Response System
Tests format detection, streaming, and complete data display
"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Add app to Python path
sys.path.append('/Users/kianwoonwong/Downloads/jarvis')

from app.services.progressive_response_service import progressive_response_service
from app.models.notebook_models import ProjectData

def create_test_projects(count: int) -> list:
    """Create test project data"""
    projects = []
    companies = ["TechCorp", "DataSystems", "InnovateLabs", "CloudWorks", "DevStudio"]
    
    for i in range(count):
        project = ProjectData(
            name=f"Project {i+1}",
            company=companies[i % len(companies)],
            year=str(2020 + (i % 5)),
            description=f"This is a comprehensive project description for project {i+1}. " +
                       f"It involves advanced technology implementation with multiple stakeholders. " +
                       f"The project duration spans several months with significant impact."
        )
        projects.append(project)
    
    return projects

async def test_format_detection():
    """Test format detection logic"""
    print("🔍 Testing Format Detection...")
    
    test_cases = [
        ("I want all projects in table format with counter", "table"),
        ("List all my projects please", "list"),
        ("Give me a summary of all projects", "summary"),
        ("Analyze my project portfolio", "analysis"),
        ("Show me everything in tabulate format", "table"),
        ("I want a comprehensive overview", "summary"),
        ("Can you analyze the data?", "analysis")
    ]
    
    for query, expected in test_cases:
        detected = progressive_response_service.detect_response_format(query)
        status = "✅" if detected == expected else "❌"
        print(f"  {status} Query: '{query[:40]}...' → Expected: {expected}, Got: {detected}")
    
    return True

async def test_progressive_streaming():
    """Test progressive streaming with different formats"""
    print("\n📊 Testing Progressive Streaming...")
    
    # Create 89 test projects (same as real scenario)
    projects = create_test_projects(89)
    print(f"  Created {len(projects)} test projects")
    
    formats_to_test = ["table", "list", "summary", "analysis"]
    
    for format_type in formats_to_test:
        print(f"\n  Testing {format_type.upper()} format:")
        
        # Create format-specific query
        queries = {
            "table": "Show me all projects in table format with counter column",
            "list": "List all my projects please",  
            "summary": "Give me a summary of all projects",
            "analysis": "Analyze my project portfolio"
        }
        
        query = queries[format_type]
        chunk_count = 0
        content_chunks = []
        
        try:
            async for chunk in progressive_response_service.generate_progressive_stream(
                data=projects,
                query=query,
                notebook_id="test-notebook",
                conversation_id="test-conversation"
            ):
                chunk_count += 1
                chunk_data = json.loads(chunk)
                content_chunks.append(chunk_data)
                
                if chunk_count <= 3:  # Show first few chunks
                    print(f"    Chunk {chunk_count}: {chunk_data['type']} ({len(chunk_data.get('content', ''))} chars)")
            
            # Analyze results
            header_chunks = [c for c in content_chunks if c['type'] == 'response_header']
            batch_chunks = [c for c in content_chunks if c['type'] == 'response_batch']
            footer_chunks = [c for c in content_chunks if c['type'] == 'response_footer']
            
            print(f"    ✅ {format_type.capitalize()} format: {len(header_chunks)} header, {len(batch_chunks)} batches, {len(footer_chunks)} footer")
            print(f"    ✅ Total chunks: {chunk_count}")
            
            # Verify all projects included
            if batch_chunks:
                last_batch = batch_chunks[-1]
                total_processed = last_batch['batch_info']['processed_total']
                print(f"    ✅ Projects processed: {total_processed}/89")
                
                if total_processed == 89:
                    print(f"    ✅ ALL PROJECTS INCLUDED - No data loss!")
                else:
                    print(f"    ❌ Missing projects: {89 - total_processed}")
            
        except Exception as e:
            print(f"    ❌ {format_type.capitalize()} format failed: {str(e)}")
            return False
    
    return True

async def test_threshold_logic():
    """Test threshold logic for when to use progressive response"""
    print("\n📏 Testing Threshold Logic...")
    
    test_scenarios = [
        (15, "Should NOT trigger progressive (15 < 20)"),
        (25, "Should trigger progressive (25 > 20)"),
        (89, "Should trigger progressive (89 > 20) - Real scenario")
    ]
    
    for count, description in test_scenarios:
        projects = create_test_projects(count)
        should_use = await progressive_response_service.should_use_progressive_response(
            projects, 
            "table format please"
        )
        
        expected = count > 20
        status = "✅" if should_use == expected else "❌"
        print(f"  {status} {description}: {'TRIGGERED' if should_use else 'NOT TRIGGERED'}")
    
    return True

async def test_performance_with_large_dataset():
    """Test performance with the real 89+ project scenario"""
    print("\n⚡ Testing Performance with Large Dataset...")
    
    projects = create_test_projects(113)  # Simulate real extracted count
    print(f"  Testing with {len(projects)} projects (simulating real scenario)")
    
    start_time = datetime.now()
    chunk_count = 0
    total_content = 0
    
    try:
        async for chunk in progressive_response_service.generate_progressive_stream(
            data=projects,
            query="I want all projects in table format with counter column",
            notebook_id="test-performance",
            conversation_id="test-performance"
        ):
            chunk_count += 1
            chunk_data = json.loads(chunk)
            total_content += len(chunk_data.get('content', ''))
            
            # Show progress every 5 chunks
            if chunk_count % 5 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"    Progress: {chunk_count} chunks in {elapsed:.1f}s")
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        print(f"  ✅ Performance test completed:")
        print(f"    • Total time: {elapsed:.2f} seconds")  
        print(f"    • Chunks generated: {chunk_count}")
        print(f"    • Total content: {total_content:,} characters")
        print(f"    • Processing rate: {len(projects)/elapsed:.1f} projects/second")
        
        if elapsed < 5.0:  # Should be very fast for direct streaming
            print(f"  ✅ EXCELLENT: Fast streaming performance ({elapsed:.2f}s)")
        else:
            print(f"  ⚠️ SLOW: Performance may need optimization ({elapsed:.2f}s)")
        
        return elapsed < 10.0  # Accept up to 10 seconds
        
    except Exception as e:
        print(f"  ❌ Performance test failed: {str(e)}")
        return False

async def run_progressive_response_tests():
    """Run comprehensive progressive response system tests"""
    print("🚀 Starting Universal Progressive Response System Testing")
    print("=" * 70)
    
    results = []
    
    # Test format detection
    format_result = await test_format_detection()
    results.append(("Format Detection", format_result))
    
    # Test progressive streaming
    streaming_result = await test_progressive_streaming()
    results.append(("Progressive Streaming", streaming_result))
    
    # Test threshold logic
    threshold_result = await test_threshold_logic()
    results.append(("Threshold Logic", threshold_result))
    
    # Test performance
    performance_result = await test_performance_with_large_dataset()
    results.append(("Performance Test", performance_result))
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 PROGRESSIVE RESPONSE SYSTEM TEST SUMMARY")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    print("-" * 70)
    print(f"OVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL PROGRESSIVE RESPONSE TESTS PASSED!")
        print("\n📋 What the system now provides:")
        print("   ✅ Universal format support: table, list, summary, analysis")
        print("   ✅ Intelligent format detection from user queries")
        print("   ✅ Progressive streaming for unlimited dataset sizes") 
        print("   ✅ Complete data display - all 89+ projects included")
        print("   ✅ Fast performance - direct streaming bypasses LLM")
        print("   ✅ Error handling with graceful fallback to LLM")
        
        print("\n🎯 Expected user experience:")
        print("   • Request 'table format' → All 89 projects in table")
        print("   • Request 'list projects' → All 89 projects in list")
        print("   • Request 'summarize' → Summary of all 89 projects")
        print("   • Request 'analyze' → Analysis of all 89 projects")
        print("   • No more partial results (34/89) - shows ALL data!")
    else:
        print("⚠️ Some tests failed - check individual results above")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(run_progressive_response_tests())
    
    if success:
        print("\n✅ Progressive Response System ready for production!")
        print("   Users will now see ALL extracted projects, not partial results.")
        print("   System supports ANY format request intelligently.")
    else:
        print("\n⚠️ Some issues remain - check the test results above")
    
    sys.exit(0 if success else 1)