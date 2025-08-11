#!/usr/bin/env python3
"""
Test IDC Integration
Tests the full IDC workflow from document upload to validation
"""

import requests
import json
import time
import io
from pathlib import Path

# Test configuration
BASE_URL = "http://localhost:8000/api/v1/idc"

def test_reference_upload():
    """Test reference document upload"""
    print("🔄 Testing reference document upload...")
    
    # Create a sample reference document
    reference_content = """
# Company Policy Manual

## Time Management Policy
All employees must arrive at work by 9:00 AM and depart no earlier than 5:00 PM.
Lunch breaks are limited to 1 hour and must be taken between 12:00 PM and 2:00 PM.

## Dress Code Policy
Business casual attire is required for all employees.
Jeans are permitted on Fridays only.

## Communication Policy
All work-related communication must be professional and respectful.
Personal calls should be limited and taken during breaks.
"""
    
    # Prepare file for upload
    files = {
        'file': ('company_policy.txt', io.StringIO(reference_content), 'text/plain')
    }
    
    data = {
        'name': 'Company Policy Manual',
        'document_type': 'policy',
        'category': 'hr_policies',
        'recommended_modes': 'paragraph,sentence'
    }
    
    try:
        response = requests.post(f"{BASE_URL}/reference/upload", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Reference document uploaded successfully!")
            print(f"   Document ID: {result['document_id']}")
            print(f"   Extraction confidence: {result['extraction_confidence']:.2f}")
            print(f"   Recommended modes: {result['recommended_extraction_modes']}")
            print(f"   Processing time: {result['processing_time_ms']}ms")
            return result['document_id']
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return None

def test_validation_workflow(reference_id):
    """Test granular validation workflow"""
    print("\n🔄 Testing granular validation workflow...")
    
    # Create a sample input document to validate
    input_content = """
I always arrive at the office by 8:30 AM to start my day early.
I usually take my lunch break at 11:30 AM for about 45 minutes.
I prefer wearing jeans and a t-shirt to work as it's more comfortable.
On Fridays, I like to dress up in business formal attire.
I keep all my work communications professional and courteous.
I make personal calls only during my designated break times.
"""
    
    # Prepare input file
    files = {
        'file': ('employee_practices.txt', io.StringIO(input_content), 'text/plain')
    }
    
    data = {
        'reference_id': reference_id,
        'extraction_mode': 'sentence',
        'max_context_usage': 0.35,
        'preserve_context': True,
        'quality_threshold': 0.8
    }
    
    try:
        # Start validation
        response = requests.post(f"{BASE_URL}/validate/granular", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            session_id = result['session_id']
            print("✅ Validation started successfully!")
            print(f"   Session ID: {session_id}")
            print(f"   Total units to validate: {result['extraction_summary']['total_units']}")
            print(f"   Estimated processing time: {result['estimated_processing_time_seconds']}s")
            
            # Monitor progress
            print("\n🔄 Monitoring validation progress...")
            return monitor_validation_progress(session_id)
            
        else:
            print(f"❌ Validation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return None

def monitor_validation_progress(session_id):
    """Monitor validation progress until completion"""
    max_wait_time = 300  # 5 minutes max
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        try:
            response = requests.get(f"{BASE_URL}/validate/{session_id}/progress")
            
            if response.status_code == 200:
                progress = response.json()['progress']
                status = progress.get('status', 'unknown')
                completed = progress.get('completed_units', 0)
                total = progress.get('total_units', 0)
                percentage = progress.get('progress_percentage', 0)
                
                print(f"   Progress: {completed}/{total} units ({percentage:.1f}%) - Status: {status}")
                
                if status in ['completed', 'failed']:
                    if status == 'completed':
                        print("✅ Validation completed successfully!")
                        return get_validation_results(session_id)
                    else:
                        print("❌ Validation failed!")
                        return None
                        
            time.sleep(3)  # Wait 3 seconds before checking again
            
        except Exception as e:
            print(f"❌ Progress monitoring error: {e}")
            break
    
    print("⏰ Validation timeout - taking too long")
    return None

def get_validation_results(session_id):
    """Get detailed validation results"""
    print("\n🔄 Retrieving validation results...")
    
    try:
        response = requests.get(f"{BASE_URL}/validate/{session_id}/results/detailed")
        
        if response.status_code == 200:
            results = response.json()
            overall = results['overall_results']
            
            print("✅ Validation results retrieved!")
            print(f"   Overall score: {overall['overall_score']:.2f}")
            print(f"   Confidence score: {overall['confidence_score']:.2f}")
            print(f"   Completeness: {overall['completeness_score']:.2f}")
            print(f"   Units processed: {overall['units_processed']}/{overall['total_units']}")
            
            # Show sample unit results
            unit_results = results.get('unit_results', [])[:3]  # First 3 units
            if unit_results:
                print("\n   Sample unit results:")
                for unit in unit_results:
                    print(f"     Unit {unit['unit_index']}: Score {unit['validation_score']:.2f}")
                    print(f"       Content: {unit['unit_content'][:100]}...")
                    print(f"       Requires review: {unit['requires_human_review']}")
            
            return results
            
        else:
            print(f"❌ Failed to get results: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Results retrieval error: {e}")
        return None

def test_configuration():
    """Test configuration endpoint"""
    print("\n🔄 Testing IDC configuration...")
    
    try:
        response = requests.get(f"{BASE_URL}/configuration")
        
        if response.status_code == 200:
            config = response.json()['configuration']
            print("✅ Configuration retrieved successfully!")
            print(f"   Available models: {len(config['available_models'])}")
            print(f"   Default extraction model: {config['default_extraction_model']}")
            print(f"   Supported extraction modes: {config['supported_extraction_modes']}")
            print(f"   Ollama URL: {config['ollama_base_url']}")
            return True
        else:
            print(f"❌ Configuration failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def main():
    """Run complete IDC integration test"""
    print("🚀 Starting IDC Integration Test")
    print("=" * 50)
    
    # Test configuration
    if not test_configuration():
        print("❌ Configuration test failed - stopping")
        return
    
    # Test reference upload
    reference_id = test_reference_upload()
    if not reference_id:
        print("❌ Reference upload failed - stopping")
        return
    
    # Test validation workflow
    validation_results = test_validation_workflow(reference_id)
    if validation_results:
        print("\n✅ IDC Integration Test PASSED!")
        print("All IDC components are working correctly.")
    else:
        print("\n❌ IDC Integration Test FAILED!")
        print("Validation workflow encountered issues.")
    
    print("\n" + "=" * 50)
    print("🏁 IDC Integration Test Complete")

if __name__ == "__main__":
    main()