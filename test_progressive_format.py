#!/usr/bin/env python3
"""
Test the updated progressive response format
"""

import asyncio
import sys
import json

sys.path.append('/Users/kianwoonwong/Downloads/jarvis')

from app.services.progressive_response_service import progressive_response_service
from app.models.notebook_models import ProjectData

async def test_progressive_format():
    """Test the new progressive format"""
    print("🧪 Testing Progressive Response Format...")
    
    # Create 3 test projects
    projects = [
        ProjectData(name="Test Project 1", company="TechCorp", year="2024", description="A test project"),
        ProjectData(name="Test Project 2", company="DataSys", year="2023", description="Another test"),
        ProjectData(name="Test Project 3", company="InnoLabs", year="2022", description="Third test project")
    ]
    
    print(f"  📊 Testing with {len(projects)} projects")
    
    chunk_count = 0
    async for chunk in progressive_response_service.generate_progressive_stream(
        data=projects,
        query="show me all projects in table format",
        notebook_id="test-notebook",
        conversation_id="test-conversation"
    ):
        chunk_count += 1
        chunk_data = json.loads(chunk)
        
        print(f"\n  🔸 Chunk {chunk_count}: {chunk_data.get('type', 'unknown')} ({chunk_data.get('phase', 'none')})")
        print(f"    Content preview: {chunk_data.get('content', '')[:80]}...")
        
        if 'progress' in chunk_data:
            progress = chunk_data['progress']
            print(f"    Progress: {progress['current']}/{progress['total']} ({progress['percentage']:.1f}%)")
        
        # Show first chunk content fully for verification
        if chunk_count == 1:
            print(f"    📝 Full header content:")
            print(f"      {chunk_data.get('content', '')}")
    
    print(f"\n  ✅ Generated {chunk_count} chunks successfully")
    return True

if __name__ == "__main__":
    result = asyncio.run(test_progressive_format())
    if result:
        print("\n🎉 Progressive format test completed!")
        print("   The new format should be compatible with existing frontend")
    else:
        print("\n❌ Progressive format test failed")