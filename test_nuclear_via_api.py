#!/usr/bin/env python3
"""Test nuclear anti-silo via API endpoint"""

import requests
import json
import time

def check_silo_nodes():
    """Check current silo nodes via API"""
    try:
        # Use knowledge graph endpoint to check connectivity
        response = requests.get('http://localhost:8000/api/v1/knowledge-graph/stats')
        
        if response.status_code == 200:
            stats = response.json()
            print(f"📊 Graph Statistics:")
            print(f"   Total nodes: {stats.get('total_nodes', 0)}")
            print(f"   Total relationships: {stats.get('total_relationships', 0)}")
            print(f"   Isolated nodes: {stats.get('isolated_nodes', 0)}")
            print(f"   Connectivity: {stats.get('connectivity_percentage', 0):.1f}%")
            return stats.get('isolated_nodes', 0)
        else:
            print(f"❌ Failed to get stats: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error checking silo nodes: {e}")
        return None

def trigger_knowledge_graph_processing():
    """Trigger knowledge graph processing via document upload"""
    try:
        # Create a test document to trigger processing
        test_content = """DBS Bank is evaluating OceanBase database technology from Alibaba Group. 
        The bank is also considering SOFAStack middleware platform from Ant Group. 
        These technologies are part of Singapore's digital banking transformation initiative.
        Oracle Database and Microsoft SQL Server are competing solutions in this space."""
        
        # Upload document via API
        files = {'file': ('test_nuclear.txt', test_content, 'text/plain')}
        data = {'enable_kg': 'true'}
        
        print("📄 Uploading test document to trigger KG processing...")
        response = requests.post('http://localhost:8000/api/v1/upload-documents', files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Document uploaded: {result.get('message', 'Success')}")
            
            # Wait for processing
            print("⏳ Waiting for KG processing to complete...")
            time.sleep(5)
            
            return True
        else:
            print(f"❌ Upload failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error triggering processing: {e}")
        return False

def test_nuclear_anti_silo():
    """Test the nuclear anti-silo functionality"""
    print("🧪 NUCLEAR ANTI-SILO API TEST")
    print("=" * 60)
    
    # Check initial state
    print("\n📊 BEFORE Processing:")
    print("-" * 30)
    silo_count_before = check_silo_nodes()
    
    if silo_count_before is None:
        print("❌ Could not check initial state")
        return False
    
    if silo_count_before == 0:
        print("✅ No silo nodes found - graph already fully connected!")
        return True
    
    print(f"🎯 Target: Eliminate {silo_count_before} silo nodes")
    
    # Trigger processing with nuclear option enabled
    print("\n☢️  TRIGGERING PROCESSING WITH NUCLEAR OPTION:")
    print("-" * 50)
    
    success = trigger_knowledge_graph_processing()
    if not success:
        print("❌ Failed to trigger processing")
        return False
    
    # Check final state
    print("\n📊 AFTER Processing:")
    print("-" * 30)
    silo_count_after = check_silo_nodes()
    
    if silo_count_after is None:
        print("❌ Could not check final state")
        return False
    
    # Results
    print(f"\n📈 NUCLEAR TEST RESULTS:")
    print("=" * 40)
    print(f"   Before: {silo_count_before} silo nodes")
    print(f"   After:  {silo_count_after} silo nodes")
    print(f"   Eliminated: {silo_count_before - silo_count_after} silo nodes")
    
    if silo_count_after == 0:
        print("🎉 SUCCESS: All silo nodes eliminated!")
        print("   Nuclear anti-silo processing is working correctly!")
        return True
    elif silo_count_after < silo_count_before:
        print(f"⚠️  PARTIAL SUCCESS: Reduced silo nodes by {silo_count_before - silo_count_after}")
        print(f"   {silo_count_after} silo nodes still remain")
        print("   Nuclear processing is working but may need adjustment")
        return False
    else:
        print("❌ FAILED: No improvement in silo node count")
        print("   Nuclear processing may not be working correctly")
        return False

if __name__ == "__main__":
    try:
        success = test_nuclear_anti_silo()
        
        if success:
            print("\n✅ Nuclear anti-silo test PASSED!")
        else:
            print("\n❌ Nuclear anti-silo test FAILED!")
            print("\n🔧 Debugging suggestions:")
            print("   1. Check if nuclear option is enabled in settings")
            print("   2. Review server logs for processing details")
            print("   3. Verify Neo4j connection and data")
            print("   4. Check if anti-silo thresholds are appropriate")
        
        exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        exit(1)