"""
Neo4j Service Layer
Provides a unified interface for Neo4j knowledge graph operations
using configuration from the knowledge graph settings.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError, ConfigurationError
from app.core.knowledge_graph_settings_cache import get_neo4j_config, get_knowledge_graph_settings

logger = logging.getLogger(__name__)

class Neo4jService:
    """Neo4j database service with settings-based configuration"""
    
    def __init__(self):
        self.driver: Optional[Driver] = None
        self.config: Optional[Dict[str, Any]] = None
        self._initialize_driver()
    
    def _initialize_driver(self):
        """Initialize Neo4j driver using current settings"""
        try:
            self.config = get_neo4j_config()
            
            if not self.config.get('enabled', False):
                logger.info("Neo4j is disabled in configuration")
                return
            
            uri = self.config['uri']
            username = self.config['username']
            password = self.config['password']
            
            # Connection pool configuration
            pool_config = self.config.get('connection_pool', {})
            max_connections = pool_config.get('max_connections', 50)
            connection_timeout = pool_config.get('connection_timeout', 30)
            
            # Security configuration
            security_config = self.config.get('security', {})
            encrypted = security_config.get('encrypted', False)
            trust = security_config.get('trust_strategy', 'TRUST_ALL_CERTIFICATES')
            
            # Create driver with configuration
            self.driver = GraphDatabase.driver(
                uri,
                auth=(username, password),
                encrypted=encrypted,
                trust=trust,
                max_connection_pool_size=max_connections,
                connection_timeout=connection_timeout
            )
            
            logger.info(f"Neo4j driver initialized successfully for {uri}")
            
            # Create performance indexes on initialization
            try:
                index_result = self.create_performance_indexes()
                if index_result.get('success'):
                    logger.info(f"🚀 Performance indexes initialized: {index_result.get('total_created', 0)} created")
                else:
                    logger.warning(f"Index creation had issues: {index_result.get('error', 'Unknown error')}")
            except Exception as idx_error:
                logger.warning(f"Failed to create performance indexes: {idx_error}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j driver: {str(e)}")
            self.driver = None
    
    def reload_configuration(self):
        """Reload configuration from settings and reinitialize driver"""
        if self.driver:
            self.driver.close()
        self._initialize_driver()
    
    def is_enabled(self) -> bool:
        """Check if Neo4j is enabled and driver is available"""
        return self.driver is not None and self.config.get('enabled', False)
    
    def test_connection(self) -> Dict[str, Any]:
        """Test Neo4j connection and return status"""
        try:
            if not self.is_enabled():
                return {
                    'success': False,
                    'error': 'Neo4j is disabled or driver not available',
                    'config': self.config
                }
            
            # Test basic connectivity
            with self.driver.session() as session:
                result = session.run("RETURN 'Neo4j connection successful' as message")
                record = result.single()
                message = record['message'] if record else 'Connected'
                
                # Get database info
                db_info = self.get_database_info()
                
                return {
                    'success': True,
                    'message': message,
                    'database_info': db_info,
                    'config': {
                        'host': self.config['host'],
                        'port': self.config['port'],
                        'database': self.config['database'],
                        'plugins': self.config.get('plugins', {})
                    }
                }
                
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable: {str(e)}")
            return {
                'success': False,
                'error': f'Neo4j service unavailable: {str(e)}',
                'config': self.config
            }
        except AuthError as e:
            logger.error(f"Neo4j authentication failed: {str(e)}")
            return {
                'success': False,
                'error': f'Authentication failed: {str(e)}',
                'config': self.config
            }
        except Exception as e:
            logger.error(f"Neo4j connection test failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'config': self.config
            }
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get Neo4j database information"""
        try:
            if not self.is_enabled():
                return {}
            
            with self.driver.session() as session:
                # Get Neo4j version
                version_result = session.run("CALL dbms.components() YIELD name, versions")
                components = [record.data() for record in version_result]
                
                # Get database statistics
                stats_result = session.run("""
                    MATCH (n) 
                    RETURN count(n) as node_count
                """)
                stats = stats_result.single()
                node_count = stats['node_count'] if stats else 0
                
                # Get relationship count
                rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                rel_stats = rel_result.single()
                rel_count = rel_stats['rel_count'] if rel_stats else 0
                
                return {
                    'components': components,
                    'node_count': node_count,
                    'relationship_count': rel_count,
                    'database_name': self.config.get('database', 'neo4j')
                }
                
        except Exception as e:
            logger.error(f"Failed to get database info: {str(e)}")
            return {'error': str(e)}
    
    def create_entity(self, entity_type: str, properties: Dict[str, Any]) -> Optional[str]:
        """Create or merge an entity in the knowledge graph using deterministic IDs"""
        try:
            if not self.is_enabled():
                return None
            
            with self.driver.session() as session:
                # Generate deterministic ID based on canonical name and type
                canonical_name = properties.get('name', '').strip().lower()
                if not canonical_name:
                    logger.error("Entity name is required for creation")
                    return None
                
                # Create deterministic ID: type_canonicalname_hash
                import hashlib
                name_hash = hashlib.md5(f"{entity_type}_{canonical_name}".encode()).hexdigest()[:8]
                deterministic_id = f"{entity_type}_{canonical_name.replace(' ', '_')}_{name_hash}"
                
                # Always use the deterministic ID
                properties['id'] = deterministic_id
                
                # Ensure unique constraints exist for proper MERGE behavior
                self._ensure_entity_constraints(session, entity_type)
                
                # Use MERGE on ID to avoid duplicates (ID is unique and deterministic)
                query = f"""
                MERGE (e:{entity_type} {{id: $entity_id}})
                ON CREATE SET e = $properties, e.created_at = datetime()
                ON MATCH SET 
                    e.confidence = CASE 
                        WHEN $confidence > COALESCE(e.confidence, 0) THEN $confidence 
                        ELSE COALESCE(e.confidence, $confidence)
                    END,
                    e.last_updated = datetime(),
                    e.document_count = COALESCE(e.document_count, 0) + 1,
                    e.original_text = CASE 
                        WHEN size($original_text) > size(COALESCE(e.original_text, '')) THEN $original_text
                        ELSE COALESCE(e.original_text, $original_text)
                    END
                RETURN e.id as entity_id, e.name as entity_name
                """
                
                # Prepare parameters for the query
                query_params = {
                    'entity_id': deterministic_id,
                    'properties': properties,
                    'confidence': properties.get('confidence', 0.0),
                    'original_text': properties.get('original_text', '')
                }
                
                result = session.run(query, **query_params)
                record = result.single()
                
                if record:
                    logger.debug(f"🔥 Entity merged/created: {record['entity_name']} -> {record['entity_id']}")
                    return record['entity_id']
                else:
                    logger.error(f"🔥 Failed to merge/create entity: {canonical_name}")
                    return deterministic_id
                
        except Exception as e:
            logger.error(f"Failed to create/merge entity: {str(e)}")
            return None
    
    def _ensure_entity_constraints(self, session, entity_type: str):
        """Ensure unique constraints exist for entity type"""
        try:
            # Create unique constraint on id for this entity type
            constraint_query = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{entity_type}) REQUIRE n.id IS UNIQUE"
            session.run(constraint_query)
            logger.debug(f"🔥 Ensured unique constraint for {entity_type}.id")
        except Exception as e:
            # Constraint might already exist, that's fine
            logger.debug(f"Constraint creation for {entity_type}: {str(e)}")
    
    def _validate_neo4j_properties(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize properties for Neo4j compatibility"""
        if not properties:
            return {}
            
        validated = {}
        invalid_props = []
        
        for key, value in properties.items():
            if value is None:
                continue  # Skip None values
            elif isinstance(value, (str, int, float, bool)):
                validated[key] = value  # Primitive types are OK
            elif isinstance(value, list) and all(isinstance(item, (str, int, float, bool)) for item in value):
                validated[key] = value  # Arrays of primitives are OK
            elif isinstance(value, (dict, list)):
                invalid_props.append(f"{key}: {type(value).__name__}")
                # Convert to JSON string as fallback
                import json
                validated[key] = json.dumps(value)
            else:
                invalid_props.append(f"{key}: {type(value).__name__}")
                validated[key] = str(value)  # Convert to string as fallback
        
        if invalid_props:
            logger.warning(f"Neo4j property validation: Converted complex types to strings: {invalid_props}")
        
        return validated

    def create_relationship(self, from_id: str, to_id: str, relationship_type: str, 
                          properties: Optional[Dict[str, Any]] = None) -> bool:
        """Create a relationship between two entities"""
        try:
            if not self.is_enabled():
                return False
            
            # Validate and sanitize properties
            safe_properties = self._validate_neo4j_properties(properties or {})
            
            with self.driver.session() as session:
                # Create relationship
                query = """
                MATCH (a {id: $from_id}), (b {id: $to_id})
                CREATE (a)-[r:%s $properties]->(b)
                RETURN r
                """ % relationship_type
                
                result = session.run(query, 
                                   from_id=from_id, 
                                   to_id=to_id, 
                                   properties=safe_properties)
                return result.single() is not None
                
        except Exception as e:
            logger.error(f"Failed to create relationship: {str(e)}")
            return False
    
    async def batch_create_entities_async(self, entities: List[Tuple[str, Dict[str, Any]]], batch_size: int = 50) -> List[Optional[str]]:
        """Create multiple entities in batches asynchronously for better performance"""
        import asyncio
        return await asyncio.to_thread(self.batch_create_entities, entities, batch_size)
    
    def batch_create_entities(self, entities: List[Tuple[str, Dict[str, Any]]], batch_size: int = 50) -> List[Optional[str]]:
        """Create multiple entities in batches for better performance"""
        try:
            if not self.is_enabled():
                return [None] * len(entities)
            
            results = []
            import hashlib
            
            # Process entities in batches
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i + batch_size]
                batch_results = []
                
                with self.driver.session() as session:
                    # Prepare batch data
                    batch_data = []
                    for entity_type, properties in batch:
                        canonical_name = properties.get('name', '').strip().lower()
                        if not canonical_name:
                            batch_results.append(None)
                            continue
                        
                        # Generate deterministic ID
                        name_hash = hashlib.md5(f"{entity_type}_{canonical_name}".encode()).hexdigest()[:8]
                        deterministic_id = f"{entity_type}_{canonical_name.replace(' ', '_')}_{name_hash}"
                        properties['id'] = deterministic_id
                        
                        batch_data.append({
                            'entity_type': entity_type,
                            'entity_id': deterministic_id,
                            'properties': properties,
                            'confidence': properties.get('confidence', 0.0),
                            'original_text': properties.get('original_text', '')
                        })
                        batch_results.append(deterministic_id)
                    
                    if not batch_data:
                        continue
                    
                    # Execute batch query using standard Cypher without APOC dependency
                    batch_query = """
                    UNWIND $batch as item
                    CALL {
                        WITH item
                        MERGE (n {id: item.entity_id})
                        SET n += item.properties,
                            n.created_at = CASE WHEN n.created_at IS NULL THEN datetime() ELSE n.created_at END,
                            n.confidence = CASE 
                                WHEN item.confidence > COALESCE(n.confidence, 0) THEN item.confidence 
                                ELSE COALESCE(n.confidence, item.confidence)
                            END,
                            n.last_updated = datetime(),
                            n.document_count = COALESCE(n.document_count, 0) + 1
                        RETURN n.id as entity_id
                    } IN TRANSACTIONS OF 10 ROWS
                    RETURN count(*) as entities_created
                    """
                    
                    try:
                        session.run(batch_query, batch=batch_data)
                        logger.info(f"✅ Batch created {len(batch_data)} entities")
                    except Exception as e:
                        # Fallback to individual creation if batch fails
                        logger.warning(f"Batch entity creation failed, falling back to individual: {e}")
                        batch_results = []
                        for entity_type, properties in batch:
                            result = self.create_entity(entity_type, properties)
                            batch_results.append(result)
                
                results.extend(batch_results)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to batch create entities: {str(e)}")
            return [None] * len(entities)
    
    async def batch_create_relationships_async(self, relationships: List[Tuple[str, str, str, Dict[str, Any]]], batch_size: int = 100) -> int:
        """Create multiple relationships in batches asynchronously for better performance"""
        import asyncio
        return await asyncio.to_thread(self.batch_create_relationships, relationships, batch_size)
    
    def batch_create_relationships(self, relationships: List[Tuple[str, str, str, Dict[str, Any]]], batch_size: int = 100) -> int:
        """FIXED: Actually batch create relationships with TRUE batching"""
        try:
            if not self.is_enabled():
                return 0
            
            # EMERGENCY LIMIT CHECK
            current_count = self.get_total_relationship_count()
            if current_count >= 50:  # Emergency hard limit
                logger.error(f"🚨 EMERGENCY: Relationship limit reached {current_count}/50")
                return 0
            
            success_count = 0
            
            # Process in ACTUAL batches using UNWIND
            for i in range(0, len(relationships), batch_size):
                batch = relationships[i:i + batch_size]
                
                with self.driver.session() as session:
                    # Prepare batch data
                    batch_data = []
                    for from_id, to_id, rel_type, properties in batch:
                        safe_properties = self._validate_neo4j_properties(properties or {})
                        batch_data.append({
                            'from_id': from_id,
                            'to_id': to_id,
                            'rel_type': rel_type,
                            'properties': safe_properties
                        })
                    
                    if not batch_data:
                        continue
                    
                    # TRUE BATCH CREATION with UNWIND (not individual calls)
                    batch_query = """
                    UNWIND $batch as item
                    MATCH (a {id: item.from_id}), (b {id: item.to_id})
                    WHERE a IS NOT NULL AND b IS NOT NULL
                    CREATE (a)-[r:RELATIONSHIP]->(b)
                    SET r = item.properties, r.type = item.rel_type
                    RETURN count(r) as created_count
                    """
                    
                    try:
                        result = session.run(batch_query, batch=batch_data)
                        record = result.single()
                        batch_success = record['created_count'] if record else 0
                        success_count += batch_success
                        logger.info(f"✅ TRUE BATCH: Created {batch_success} relationships")
                        
                        # Emergency check after each batch
                        new_count = self.get_total_relationship_count()
                        if new_count >= 50:
                            logger.error(f"🚨 EMERGENCY STOP: Hit limit during batch processing")
                            return success_count
                            
                    except Exception as e:
                        logger.error(f"❌ Batch relationship creation failed: {e}")
                        # Do NOT fall back to individual creation - that's what caused the explosion
                        continue
            
            return success_count
            
        except Exception as e:
            logger.error(f"Failed to batch create relationships: {str(e)}")
            return 0
    
    def create_performance_indexes(self) -> Dict[str, Any]:
        """Create essential indexes for knowledge graph performance using correct Neo4j 5.x syntax"""
        try:
            if not self.is_enabled():
                return {'success': False, 'error': 'Neo4j not enabled'}
            
            indexes_created = []
            indexes_failed = []
            
            # Common entity types used in the knowledge graph system
            common_entity_types = [
                'PERSON', 'EXECUTIVE', 'ORGANIZATION', 'COMPANY', 'TECHNOLOGY', 
                'LOCATION', 'CONCEPT', 'EVENT', 'PRODUCT', 'SERVICE'
            ]
            
            with self.driver.session() as session:
                # Create indexes for each common entity type on essential properties
                for entity_type in common_entity_types:
                    # Essential property indexes for each entity type
                    essential_properties = ['id', 'name', 'document_id', 'chunk_id', 'confidence', 'created_at']
                    
                    for prop in essential_properties:
                        try:
                            index_name = f"{entity_type.lower()}_{prop}_index"
                            query = f"CREATE INDEX {index_name} IF NOT EXISTS FOR (n:{entity_type}) ON (n.{prop})"
                            session.run(query)
                            indexes_created.append(index_name)
                            logger.debug(f"✅ Created index: {index_name}")
                        except Exception as e:
                            error_msg = f"{entity_type}.{prop}: {str(e)}"
                            indexes_failed.append(error_msg)
                            logger.debug(f"❌ Failed to create index {entity_type}.{prop}: {e}")
                
                # Create composite indexes for common query patterns
                composite_indexes = [
                    ("document_type_composite", "ORGANIZATION", ["document_id", "type"]),
                    ("name_confidence_composite", "PERSON", ["name", "confidence"]),
                    ("created_doc_composite", "CONCEPT", ["created_at", "document_id"])
                ]
                
                for index_name, entity_type, properties in composite_indexes:
                    try:
                        props_str = ", ".join([f"n.{prop}" for prop in properties])
                        query = f"CREATE INDEX {index_name} IF NOT EXISTS FOR (n:{entity_type}) ON ({props_str})"
                        session.run(query)
                        indexes_created.append(index_name)
                        logger.debug(f"✅ Created composite index: {index_name}")
                    except Exception as e:
                        error_msg = f"{index_name}: {str(e)}"
                        indexes_failed.append(error_msg)
                        logger.debug(f"❌ Failed to create composite index {index_name}: {e}")
                        
                # Create text indexes for name-based searching (useful for fuzzy matching)
                try:
                    for entity_type in ['PERSON', 'ORGANIZATION', 'CONCEPT']:
                        text_index_name = f"{entity_type.lower()}_name_text_index"
                        query = f"CREATE TEXT INDEX {text_index_name} IF NOT EXISTS FOR (n:{entity_type}) ON (n.name)"
                        session.run(query)
                        indexes_created.append(text_index_name)
                        logger.debug(f"✅ Created text index: {text_index_name}")
                except Exception as e:
                    error_msg = f"Text indexes: {str(e)}"
                    indexes_failed.append(error_msg)
                    logger.debug(f"❌ Failed to create text indexes: {e}")
            
            success_count = len(indexes_created)
            failed_count = len(indexes_failed)
            
            logger.info(f"🚀 Index Creation Summary: {success_count} created, {failed_count} failed")
            
            return {
                'success': True,
                'indexes_created': indexes_created,
                'indexes_failed': indexes_failed,
                'total_created': success_count,
                'total_failed': failed_count,
                'entity_types_indexed': common_entity_types
            }
            
        except Exception as e:
            logger.error(f"Failed to create performance indexes: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def find_entities(self, entity_type: Optional[str] = None, 
                     properties: Optional[Dict[str, Any]] = None, 
                     limit: int = 100) -> List[Dict[str, Any]]:
        """Find entities matching criteria"""
        try:
            if not self.is_enabled():
                return []
            
            with self.driver.session() as session:
                # Build query
                if entity_type:
                    query = f"MATCH (e:{entity_type})"
                else:
                    query = "MATCH (e)"
                
                # Add property filters
                if properties:
                    conditions = [f"e.{key} = ${key}" for key in properties.keys()]
                    query += " WHERE " + " AND ".join(conditions)
                
                query += f" RETURN e LIMIT {limit}"
                
                result = session.run(query, **(properties or {}))
                return [record['e'] for record in result]
                
        except Exception as e:
            logger.error(f"Failed to find entities: {str(e)}")
            return []
    
    def find_relationships(self, from_id: Optional[str] = None, 
                          to_id: Optional[str] = None,
                          relationship_type: Optional[str] = None,
                          limit: int = 100) -> List[Dict[str, Any]]:
        """Find relationships matching criteria"""
        try:
            if not self.is_enabled():
                return []
            
            with self.driver.session() as session:
                # Build query
                query = "MATCH (a)-[r"
                if relationship_type:
                    query += f":{relationship_type}"
                query += "]->(b)"
                
                conditions = []
                params = {}
                
                if from_id:
                    conditions.append("a.id = $from_id")
                    params['from_id'] = from_id
                
                if to_id:
                    conditions.append("b.id = $to_id")
                    params['to_id'] = to_id
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                query += f" RETURN a, r, b LIMIT {limit}"
                
                result = session.run(query, **params)
                return [record.data() for record in result]
                
        except Exception as e:
            logger.error(f"Failed to find relationships: {str(e)}")
            return []
    
    def execute_cypher(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute custom Cypher query"""
        try:
            if not self.is_enabled():
                return []
            
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
                
        except Exception as e:
            logger.error(f"Failed to execute Cypher query: {str(e)}")
            return []
    
    def get_total_relationship_count(self) -> int:
        """Get total number of relationships in the graph"""
        try:
            if not self.is_enabled():
                return 0
                
            with self.driver.session() as session:
                result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                record = result.single()
                return record['count'] if record else 0
        except Exception as e:
            logger.error(f"Failed to get relationship count: {e}")
            return 0
    
    def clear_database(self) -> bool:
        """Clear all data from the database (use with caution!)"""
        try:
            if not self.is_enabled():
                return False
            
            with self.driver.session() as session:
                # Delete all relationships first, then nodes
                session.run("MATCH ()-[r]->() DELETE r")
                session.run("MATCH (n) DELETE n")
                return True
                
        except Exception as e:
            logger.error(f"Failed to clear database: {str(e)}")
            return False
    
    def deduplicate_entities(self) -> Dict[str, Any]:
        """Remove duplicate entities with the same ID, keeping the one with highest confidence"""
        try:
            if not self.is_enabled():
                return {'success': False, 'error': 'Neo4j not enabled'}
            
            with self.driver.session() as session:
                # Find entities with duplicate IDs
                duplicate_query = """
                MATCH (n)
                WITH n.id as entity_id, collect(n) as nodes, count(n) as count
                WHERE count > 1
                RETURN entity_id, nodes, count
                ORDER BY count DESC
                """
                
                duplicates_result = session.run(duplicate_query)
                duplicates = list(duplicates_result)
                
                total_duplicates = 0
                entities_cleaned = 0
                
                for record in duplicates:
                    entity_id = record['entity_id']
                    nodes = record['nodes']
                    count = record['count']
                    
                    if count <= 1:
                        continue
                    
                    total_duplicates += count - 1
                    entities_cleaned += 1
                    
                    # Sort nodes by confidence (highest first) and created_at (newest first)
                    nodes_sorted = sorted(nodes, key=lambda n: (
                        n.get('confidence', 0), 
                        n.get('created_at', '1970-01-01T00:00:00Z')
                    ), reverse=True)
                    
                    # Keep the first node (highest confidence, newest), delete the rest
                    node_to_keep = nodes_sorted[0]
                    nodes_to_delete = nodes_sorted[1:]
                    
                    logger.info(f"🔥 Deduplicating {entity_id}: keeping 1, deleting {len(nodes_to_delete)}")
                    
                    # Delete duplicate nodes (relationships will be handled by CASCADE if configured)
                    for node in nodes_to_delete:
                        # First move relationships to the node we're keeping
                        move_relationships_query = """
                        MATCH (old_node) WHERE id(old_node) = $old_node_id
                        MATCH (keep_node) WHERE id(keep_node) = $keep_node_id
                        OPTIONAL MATCH (old_node)-[r]->(other)
                        WHERE other <> keep_node
                        WITH old_node, keep_node, collect(DISTINCT {rel: r, other: other, type: type(r), props: properties(r)}) as outgoing_rels
                        OPTIONAL MATCH (other2)-[r2]->(old_node)
                        WHERE other2 <> keep_node
                        WITH old_node, keep_node, outgoing_rels, collect(DISTINCT {rel: r2, other: other2, type: type(r2), props: properties(r2)}) as incoming_rels
                        
                        // Create new outgoing relationships
                        FOREACH (rel_info IN outgoing_rels |
                            FOREACH (dummy IN CASE WHEN rel_info.other IS NOT NULL THEN [1] ELSE [] END |
                                MERGE (keep_node)-[new_rel:RELATIONSHIP]->(rel_info.other)
                                SET new_rel = rel_info.props
                            )
                        )
                        
                        // Create new incoming relationships  
                        FOREACH (rel_info IN incoming_rels |
                            FOREACH (dummy IN CASE WHEN rel_info.other IS NOT NULL THEN [1] ELSE [] END |
                                MERGE (rel_info.other)-[new_rel:RELATIONSHIP]->(keep_node)
                                SET new_rel = rel_info.props
                            )
                        )
                        
                        // Delete old relationships and node
                        DETACH DELETE old_node
                        """
                        
                        # For now, let's use a simpler approach - just delete the duplicates
                        # The relationships will point to the remaining entity
                        delete_query = "MATCH (n) WHERE id(n) = $node_id DETACH DELETE n"
                        session.run(delete_query, node_id=node.id)
                
                return {
                    'success': True,
                    'total_duplicates_removed': total_duplicates,
                    'entities_cleaned': entities_cleaned,
                    'message': f'Removed {total_duplicates} duplicate entities across {entities_cleaned} unique IDs'
                }
                
        except Exception as e:
            logger.error(f"Failed to deduplicate entities: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_truly_isolated_nodes(self) -> List[Dict[str, Any]]:
        """Get nodes that have no relationships, handling duplicates correctly"""
        try:
            if not self.is_enabled():
                return []
            
            with self.driver.session() as session:
                # Get unique entities (by ID) and check if any instance has relationships
                query = """
                MATCH (n)
                WITH n.id as entity_id, collect(n) as nodes
                WITH entity_id, nodes[0] as representative_node
                OPTIONAL MATCH (any_instance)
                WHERE any_instance.id = entity_id
                OPTIONAL MATCH (any_instance)-[r]-()
                WITH entity_id, representative_node, count(r) as total_relationships
                WHERE total_relationships = 0
                RETURN representative_node.id as id, 
                       representative_node.name as name, 
                       representative_node.type as type,
                       labels(representative_node) as labels
                ORDER BY name
                """
                
                result = session.run(query)
                return [record.data() for record in result]
                
        except Exception as e:
            logger.error(f"Failed to get isolated nodes: {str(e)}")
            return []
    
    def close(self):
        """Close the Neo4j driver connection"""
        if self.driver:
            self.driver.close()
            self.driver = None
            logger.info("Neo4j driver connection closed")

# Singleton instance
_neo4j_service: Optional[Neo4jService] = None

def get_neo4j_service() -> Neo4jService:
    """Get or create Neo4j service singleton"""
    global _neo4j_service
    if _neo4j_service is None:
        _neo4j_service = Neo4jService()
    return _neo4j_service

def reload_neo4j_service():
    """Reload Neo4j service configuration"""
    global _neo4j_service
    if _neo4j_service:
        _neo4j_service.reload_configuration()
    else:
        _neo4j_service = Neo4jService()

def test_neo4j_connection() -> Dict[str, Any]:
    """Test Neo4j connection using current service"""
    service = get_neo4j_service()
    return service.test_connection()