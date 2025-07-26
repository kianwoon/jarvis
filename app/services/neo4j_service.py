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
        """Create an entity in the knowledge graph"""
        try:
            if not self.is_enabled():
                return None
            
            with self.driver.session() as session:
                # Create entity with properties
                query = f"""
                CREATE (e:{entity_type} $properties)
                RETURN e.id as entity_id
                """
                
                # Ensure entity has an ID
                if 'id' not in properties:
                    import uuid
                    properties['id'] = str(uuid.uuid4())
                
                result = session.run(query, properties=properties)
                record = result.single()
                return record['entity_id'] if record else properties['id']
                
        except Exception as e:
            logger.error(f"Failed to create entity: {str(e)}")
            return None
    
    def create_relationship(self, from_id: str, to_id: str, relationship_type: str, 
                          properties: Optional[Dict[str, Any]] = None) -> bool:
        """Create a relationship between two entities"""
        try:
            if not self.is_enabled():
                return False
            
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
                                   properties=properties or {})
                return result.single() is not None
                
        except Exception as e:
            logger.error(f"Failed to create relationship: {str(e)}")
            return False
    
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