#!/usr/bin/env python3
"""
Test script to isolate the progress generator issue
"""
import asyncio
import json

async def test_progress_generator():
    """Test function to simulate the progress generator"""
    print("🔥🔥🔥 TEST GENERATOR STARTING!")
    
    # First yield - this should appear immediately
    yield f"data: {json.dumps({'startup': 'Starting progress generator...'})}\n\n"
    print("🔥🔥🔥 FIRST YIELD COMPLETED!")
    
    # Wait a bit
    await asyncio.sleep(0.1)
    
    # Second yield
    yield f"data: {json.dumps({'step': 1, 'message': 'Test step 1'})}\n\n"
    print("🔥🔥🔥 SECOND YIELD COMPLETED!")

async def main():
    """Main test function"""
    print("Starting generator test...")
    
    async for data in test_progress_generator():
        print(f"Received: {data.strip()}")
    
    print("Generator test completed!")

if __name__ == "__main__":
    asyncio.run(main())