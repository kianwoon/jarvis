#!/usr/bin/env python3
"""
Test the progress generator directly to check for errors
"""
import asyncio
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, '/Users/kianwoonwong/Downloads/jarvis')

async def test_generator():
    try:
        from app.api.v1.endpoints.document_progress import progress_generator, UploadProgress
        
        # Create mock data
        file_content = b"Test PDF content" * 1000  # Some test bytes
        filename = "test.pdf"
        progress = UploadProgress()
        upload_params = {'target_collection': 'product_documentation'}
        
        print("🔥 About to call progress_generator...")
        
        # Try to call the generator
        gen = progress_generator(file_content, filename, progress, upload_params)
        
        print("🔥 Generator created, trying to get first value...")
        
        # Try to get the first yield
        first_value = await gen.__anext__()
        print(f"🔥 First yield: {first_value}")
        
    except Exception as e:
        print(f"🚨 ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_generator())