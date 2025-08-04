#!/usr/bin/env python3
"""
Fix the broken batch relationship creation in Neo4j service
"""

def generate_proper_batch_relationship_method():
    """Generate the proper batch relationship creation method"""
    
    return '''
    def batch_create_relationships(self, relationships: List[Tuple[str, str, str, Dict[str, Any]]], batch_size: int = 100) -> int:
        """Create multiple relationships in batches for better performance"""
        try:
            if not self.is_enabled():
                return 0
            
            success_count = 0
            
            # Group relationships by type for efficient batch processing
            relationships_by_type = {}
            for from_id, to_id, rel_type, properties in relationships:
                if rel_type not in relationships_by_type:
                    relationships_by_type[rel_type] = []
                relationships_by_type[rel_type].append((from_id, to_id, properties))
            
            # Process each relationship type in batches
            for rel_type, typed_relationships in relationships_by_type.items():
                for i in range(0, len(typed_relationships), batch_size):
                    batch = typed_relationships[i:i + batch_size]
                    
                    with self.driver.session() as session:
                        # Prepare batch data
                        batch_data = []
                        for from_id, to_id, properties in batch:
                            safe_properties = self._validate_neo4j_properties(properties or {})
                            batch_data.append({
                                'from_id': from_id,
                                'to_id': to_id,
                                'properties': safe_properties
                            })
                        
                        # Use UNWIND for true batch processing
                        batch_query = f"""
                        UNWIND $batch as rel
                        MATCH (from {{id: rel.from_id}})
                        MATCH (to {{id: rel.to_id}})
                        MERGE (from)-[r:{rel_type}]->(to)
                        SET r += rel.properties,
                            r.created_at = CASE WHEN r.created_at IS NULL THEN datetime() ELSE r.created_at END,
                            r.last_updated = datetime()
                        RETURN count(r) as created_count
                        """
                        
                        try:
                            result = session.run(batch_query, batch=batch_data).single()
                            created = result['created_count'] if result else 0
                            success_count += created
                            logger.info(f"✅ Batch created {created} {rel_type} relationships")
                        except Exception as e:
                            logger.error(f"Batch creation failed for {rel_type}: {e}")
                            # Do NOT fall back to individual creation!
                            continue
            
            return success_count
            
        except Exception as e:
            logger.error(f"Failed to batch create relationships: {str(e)}")
            return 0
'''

def fix_neo4j_service():
    """Apply the fix to neo4j_service.py"""
    
    import re
    
    with open('app/services/neo4j_service.py', 'r') as f:
        content = f.read()
    
    # Find the batch_create_relationships method
    pattern = r'def batch_create_relationships\(self.*?\n(?=\s{0,4}def|\s{0,4}async def|$)'
    
    new_method = generate_proper_batch_relationship_method()
    
    # Replace the method
    content = re.sub(pattern, new_method + '\n', content, flags=re.DOTALL)
    
    with open('app/services/neo4j_service.py.fixed', 'w') as f:
        f.write(content)
    
    print("✅ Fixed batch_create_relationships method")
    print("📄 New file created: app/services/neo4j_service.py.fixed")
    print("\nTo apply the fix:")
    print("  cp app/services/neo4j_service.py.fixed app/services/neo4j_service.py")

if __name__ == "__main__":
    print("🔧 Fixing batch relationship creation...")
    fix_neo4j_service()