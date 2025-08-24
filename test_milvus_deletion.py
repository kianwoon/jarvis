#!/usr/bin/env python3
"""
Test script to verify Milvus deletion functionality.
Run this script to test if the document deletion fix is working.
"""

import asyncio
import logging
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from sqlalchemy.orm import Session
from app.core.db import engine, SessionLocal
from app.services.document_admin_service import DocumentAdminService

# Configure logging to see debug output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_milvus_deletion():
    """Test the Milvus deletion functionality with debug logging."""
    
    # Replace with an actual document ID from your system
    test_document_id = "33997c75bf33"  # Update this with a real document ID
    
    print(f"🧪 Testing Milvus deletion for document: {test_document_id}")
    
    # Create a database session
    db: Session = SessionLocal()
    
    try:
        # Initialize the document admin service
        admin_service = DocumentAdminService()
        
        # Get document usage info first
        print("📊 Getting document usage information...")
        usage_info = await admin_service.get_document_usage_info(db, test_document_id)
        print(f"📊 Document usage info: {usage_info}")
        
        # Perform the deletion with debug logging enabled
        print("🗑️  Starting document deletion...")
        deletion_result = await admin_service.delete_document_permanently(
            db=db,
            document_id=test_document_id,
            remove_from_notebooks=True
        )
        
        print("✅ Deletion completed!")
        print(f"📊 Deletion result: {deletion_result}")
        
        if deletion_result.get('milvus_deleted'):
            print("✅ Milvus deletion reported as successful")
        else:
            print("❌ Milvus deletion failed")
            if deletion_result.get('errors'):
                print(f"🚨 Errors: {deletion_result['errors']}")
        
    except Exception as e:
        print(f"🚨 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        db.close()

if __name__ == "__main__":
    print("🚀 Starting Milvus deletion test...")
    print("⚠️  WARNING: This will actually delete the document from all systems!")
    print("⚠️  Make sure you're using a test document ID or backup your data first!")
    
    # Uncomment the next line to run the actual test
    # asyncio.run(test_milvus_deletion())
    
    print("📝 To run the test:")
    print("1. Replace 'test_document_id' with a real document ID")
    print("2. Uncomment the asyncio.run() line above")
    print("3. Run: python test_milvus_deletion.py")