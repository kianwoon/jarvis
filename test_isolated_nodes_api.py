#!/usr/bin/env python3
"""
Test the isolated nodes API endpoint to understand the discrepancy
"""

import requests
import json

def test_isolated_nodes():
    """Test the isolated nodes endpoint"""
    url = "http://localhost:8000/api/v1/knowledge-graph/isolated-nodes"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            
            print("\n" + "="*80)
            print("🔍 ISOLATED NODES API TEST")
            print("="*80)
            
            # Print summary
            summary = data.get('summary', {})
            print("\n📊 ISOLATION DETECTION METHODS:")
            print(f"  Method 1 (deduplication-aware): {summary.get('method1_count', 0)} nodes")
            print(f"  Method 2 (simple check):        {summary.get('method2_count', 0)} nodes")
            print(f"  Method 3 (direct count):        {summary.get('method3_count', 0)} nodes")
            print(f"  Method 4 (no relationships):    {summary.get('method4_count', 0)} nodes")
            print(f"\n  Methods agree: {summary.get('methods_agree', False)}")
            print(f"  Total unique isolated: {summary.get('total_unique_isolated', 0)}")
            
            # If methods disagree, show details
            if not summary.get('methods_agree', False):
                print("\n⚠️  CRITICAL: Isolation detection methods DISAGREE!")
                print("This explains why anti-silo reports 0 but user sees isolated nodes.")
                
                # Show discrepancies
                discrepancies = data.get('discrepancy_analysis', {})
                if discrepancies:
                    print("\n🔴 Discrepancy Analysis:")
                    for node_id, info in list(discrepancies.items())[:5]:  # Show first 5
                        node_info = info.get('node_info', {})
                        print(f"\n  Node: {node_info.get('name', 'Unknown')} (ID: {node_id})")
                        print(f"    - In Method 1: {info.get('in_method1', False)}")
                        print(f"    - In Method 2: {info.get('in_method2', False)}")
                        print(f"    - In Method 3: {info.get('in_method3', False)}")
                        print(f"    - In Method 4: {info.get('in_method4', False)}")
            
            # Show sample isolated nodes from each method
            methods = data.get('isolation_methods', {})
            for method_name, method_data in methods.items():
                if method_data['count'] > 0:
                    print(f"\n📍 {method_name} - Sample nodes:")
                    for node in method_data['nodes'][:3]:
                        print(f"    - {node.get('name', 'Unknown')} (type: {node.get('type', 'Unknown')})")
            
            print("\n" + "="*80)
            print("🎯 DIAGNOSIS")
            print("="*80)
            
            if summary.get('method1_count', 0) == 0 and summary.get('method2_count', 0) > 0:
                print("\n❗ CRITICAL BUG CONFIRMED:")
                print("   The anti-silo system uses Method 1 (deduplication-aware)")
                print("   But Method 1 is NOT detecting isolated nodes correctly!")
                print("   Meanwhile, the simple check (Method 2) finds isolated nodes.")
                print("\n   This is why the system reports '0 isolated' but user sees them.")
                print("\n   ROOT CAUSE: The deduplication-aware query is flawed.")
                
        else:
            print(f"❌ Error: API returned status {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing API: {e}")

def test_silo_analysis():
    """Test the silo analysis endpoint"""
    url = "http://localhost:8000/api/v1/knowledge-graph/silo-analysis"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            analysis = data.get('analysis', {})
            summary = analysis.get('summary', {})
            
            print("\n📊 SILO ANALYSIS:")
            print(f"  Isolated nodes (0 connections): {summary.get('isolated_count', 0)}")
            print(f"  Weakly connected (1-2 connections): {summary.get('weakly_connected_count', 0)}")
            print(f"  Total potential silos: {summary.get('total_entities_analyzed', 0)}")
            
    except Exception as e:
        print(f"❌ Error testing silo analysis: {e}")

if __name__ == "__main__":
    test_isolated_nodes()
    test_silo_analysis()