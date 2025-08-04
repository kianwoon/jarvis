#!/usr/bin/env python3
"""
Quick script to run the emergency knowledge graph cleanup
Execute this to immediately reduce relationships from 486 to ≤188
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from emergency_kg_cleanup import main

if __name__ == "__main__":
    print("🚨 Starting Emergency Knowledge Graph Cleanup...")
    print("Target: Reduce relationships from 486 to ≤188 (≤4 per entity)")
    print("")
    
    try:
        asyncio.run(main())
        print("")
        print("✅ Emergency cleanup completed!")
        print("Browser performance should be significantly improved.")
    except Exception as e:
        print(f"❌ Emergency cleanup failed: {e}")
        sys.exit(1)