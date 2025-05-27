#!/usr/bin/env python3
"""
Test script for PDF upload progress functionality
"""
import requests
import json
import time
from pathlib import Path

API_BASE_URL = "http://localhost:8000/api/v1"

def test_upload_progress():
    """Test PDF upload with progress tracking"""
    print("PDF Upload Progress Test")
    print("=" * 50)
    
    # Find a test PDF file
    test_pdf = None
    for pdf_path in [
        "test.pdf",
        "sample.pdf",
        "/tmp/test.pdf",
        Path.home() / "Downloads" / "test.pdf"
    ]:
        if Path(pdf_path).exists():
            test_pdf = pdf_path
            break
    
    if not test_pdf:
        print("❌ No test PDF found. Please place a test.pdf file in the current directory.")
        return
    
    print(f"📄 Using test file: {test_pdf}")
    print(f"📊 File size: {Path(test_pdf).stat().st_size / 1024 / 1024:.2f} MB")
    
    # Upload with progress tracking
    with open(test_pdf, 'rb') as f:
        files = {'file': ('test.pdf', f, 'application/pdf')}
        
        print("\n🚀 Starting upload...")
        response = requests.post(
            f"{API_BASE_URL}/documents/upload_pdf_progress",
            files=files,
            stream=True
        )
        
        if response.status_code != 200:
            print(f"❌ Upload failed: {response.status_code} - {response.text}")
            return
        
        # Process SSE stream
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])
                        
                        # Display progress
                        progress = data['progress_percent']
                        step_name = data['step_name']
                        current = data['current_step']
                        total = data['total_steps']
                        
                        # Progress bar
                        bar_length = 40
                        filled = int(bar_length * progress / 100)
                        bar = '█' * filled + '░' * (bar_length - filled)
                        
                        print(f"\r[{bar}] {progress}% - Step {current}/{total}: {step_name}", end='', flush=True)
                        
                        # Show details on new line for important info
                        if 'total_chunks' in data['details']:
                            print(f"\n  📦 Total chunks: {data['details']['total_chunks']}")
                        if 'duplicates_found' in data['details']:
                            print(f"  🔍 Duplicates found: {data['details']['duplicates_found']}")
                        if 'embedding_progress' in data['details']:
                            print(f"  🧠 Embeddings: {data['details']['embedding_progress']}/{data['details']['total_embeddings']}")
                        
                        # Check for completion
                        if data.get('error'):
                            print(f"\n\n❌ Error: {data['error']}")
                            break
                        elif current == total:
                            print("\n\n✅ Upload complete!")
                            details = data['details']
                            if details.get('status') == 'success':
                                print(f"  📊 Chunks inserted: {details.get('unique_chunks_inserted', 0)}")
                                print(f"  🔍 Duplicates filtered: {details.get('duplicates_filtered', 0)}")
                                print(f"  💾 Collection: {details.get('collection', 'N/A')}")
                                print(f"  🔑 File ID: {details.get('file_id', 'N/A')}")
                            elif details.get('status') == 'skipped':
                                print(f"  ⚠️  Skipped: {details.get('reason', 'Unknown reason')}")
                            break
                            
                    except json.JSONDecodeError:
                        print(f"\n❌ Failed to parse: {line}")

def test_upload_fallback():
    """Test fallback to original upload endpoint"""
    print("\n\nTesting Original Upload Endpoint")
    print("=" * 50)
    
    test_pdf = "test.pdf"
    if not Path(test_pdf).exists():
        print("❌ No test.pdf found")
        return
    
    with open(test_pdf, 'rb') as f:
        files = {'file': ('test.pdf', f, 'application/pdf')}
        
        print("📤 Uploading...")
        response = requests.post(
            f"{API_BASE_URL}/documents/upload_pdf",
            files=files
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Upload successful!")
            print(f"  📊 Total chunks: {data.get('total_chunks', 0)}")
            print(f"  💾 Unique chunks: {data.get('unique_chunks', 0)}")
            print(f"  🔍 Duplicates: {data.get('duplicates_filtered', 0)}")
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(response.text)

if __name__ == "__main__":
    try:
        test_upload_progress()
        test_upload_fallback()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted")
    except Exception as e:
        print(f"\n\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()