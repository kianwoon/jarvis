#!/usr/bin/env python3
"""
EMERGENCY PATCH: Fix knowledge graph relationship explosion
"""

import os
import sys

def apply_emergency_fixes():
    """Apply emergency fixes to stop relationship explosion"""
    
    print("🚨 APPLYING EMERGENCY KNOWLEDGE GRAPH FIXES...")
    
    # Fix 1: Disable ALL relationship creation in neo4j_service.py
    neo4j_service_path = "app/services/neo4j_service.py"
    
    with open(neo4j_service_path, 'r') as f:
        content = f.read()
    
    # Add emergency check at the beginning of create_relationship
    emergency_check = '''
    def create_relationship(self, from_id: str, to_id: str, relationship_type: str, 
                          properties: Optional[Dict[str, Any]] = None) -> bool:
        """Create a relationship between two entities"""
        
        # EMERGENCY: Disable ALL relationship creation
        logger.error("🚨 EMERGENCY: Relationship creation DISABLED to prevent explosion")
        return False
        
        # Original code follows...'''
    
    # Replace the method definition
    content = content.replace(
        "def create_relationship(self, from_id: str, to_id: str, relationship_type: str, \n                          properties: Optional[Dict[str, Any]] = None) -> bool:\n        \"\"\"Create a relationship between two entities\"\"\"",
        emergency_check
    )
    
    with open(neo4j_service_path, 'w') as f:
        f.write(content)
    
    print("✅ Disabled create_relationship in neo4j_service.py")
    
    # Fix 2: Disable batch_create_relationships_async
    kg_service_path = "app/services/knowledge_graph_service.py"
    
    with open(kg_service_path, 'r') as f:
        content = f.read()
    
    # Comment out the batch creation call
    content = content.replace(
        "relationships_stored = await neo4j_service.batch_create_relationships_async(deduplicated_batch, batch_size=100)",
        "# EMERGENCY DISABLED: relationships_stored = await neo4j_service.batch_create_relationships_async(deduplicated_batch, batch_size=100)\n                    relationships_stored = 0  # EMERGENCY: No relationships created"
    )
    
    with open(kg_service_path, 'w') as f:
        f.write(content)
    
    print("✅ Disabled batch_create_relationships_async call")
    
    # Fix 3: Create emergency config to disable all KG features
    emergency_config = """
# EMERGENCY KNOWLEDGE GRAPH SETTINGS
KNOWLEDGE_GRAPH_ENABLED=false
KNOWLEDGE_GRAPH_EXTRACTION_MODE=disabled
KNOWLEDGE_GRAPH_ENABLE_ANTI_SILO=false
KNOWLEDGE_GRAPH_ENABLE_NUCLEAR_OPTION=false
NEO4J_ENABLED=false
"""
    
    with open('.env.emergency', 'w') as f:
        f.write(emergency_config)
    
    print("✅ Created .env.emergency with all KG features disabled")
    
    print("\n🚨 EMERGENCY FIXES APPLIED!")
    print("\nNext steps:")
    print("1. Restart the application with: ./run_local.sh")
    print("2. Clean Neo4j database")
    print("3. Investigate root cause before re-enabling")

if __name__ == "__main__":
    apply_emergency_fixes()