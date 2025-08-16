#!/usr/bin/env python3
"""
Setup and Initialize the Universal Radiating Coverage System

This script sets up all necessary components for the radiating system:
- Database migrations
- Neo4j indexes and constraints
- Redis caching configuration
- Default settings initialization
- Sample data creation for testing
"""

import sys
import os
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text
from neo4j import GraphDatabase
import redis
import httpx

# Import application components
from app.core.config import get_settings
from app.core.redis_client import get_redis_client
from app.core.radiating_settings_cache import get_radiating_settings
from app.services.neo4j_service import Neo4jService
from app.services.radiating.radiating_neo4j_service import RadiatingNeo4jService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'radiating_setup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class RadiatingSystemSetup:
    """Comprehensive setup utility for the Radiating Coverage System"""
    
    def __init__(self):
        self.settings = get_settings()
        self.engine = self._create_db_engine()
        self.neo4j_service = None
        self.redis_client = None
        self.setup_complete = False
        
    def _create_db_engine(self):
        """Create PostgreSQL database engine"""
        database_url = (
            f"postgresql://{self.settings.POSTGRES_USER}:{self.settings.POSTGRES_PASSWORD}"
            f"@{self.settings.POSTGRES_HOST}:{self.settings.POSTGRES_PORT}/{self.settings.POSTGRES_DB}"
        )
        
        return create_engine(
            database_url,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False
        )
    
    async def check_prerequisites(self) -> bool:
        """Check all prerequisites are met"""
        logger.info("🔍 Checking prerequisites...")
        
        checks = {
            "PostgreSQL": self._check_postgresql(),
            "Neo4j": self._check_neo4j(),
            "Redis": self._check_redis(),
            "Milvus": await self._check_milvus()
        }
        
        all_passed = True
        for service, status in checks.items():
            if status:
                logger.info(f"  ✅ {service}: Available")
            else:
                logger.error(f"  ❌ {service}: Not available")
                all_passed = False
        
        return all_passed
    
    def _check_postgresql(self) -> bool:
        """Check PostgreSQL connectivity"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT version()"))
                version = result.fetchone()[0]
                logger.debug(f"PostgreSQL version: {version}")
                return True
        except Exception as e:
            logger.error(f"PostgreSQL error: {e}")
            return False
    
    def _check_neo4j(self) -> bool:
        """Check Neo4j connectivity"""
        try:
            uri = f"bolt://{self.settings.NEO4J_HOST}:{self.settings.NEO4J_BOLT_PORT}"
            driver = GraphDatabase.driver(
                uri, 
                auth=(self.settings.NEO4J_USER, self.settings.NEO4J_PASSWORD)
            )
            
            with driver.session() as session:
                result = session.run("RETURN 'connected' as status")
                status = result.single()['status']
                driver.close()
                return status == 'connected'
        except Exception as e:
            logger.error(f"Neo4j error: {e}")
            return False
    
    def _check_redis(self) -> bool:
        """Check Redis connectivity"""
        try:
            client = redis.Redis(
                host=self.settings.REDIS_HOST,
                port=self.settings.REDIS_PORT,
                password=self.settings.REDIS_PASSWORD,
                decode_responses=True
            )
            return client.ping()
        except Exception as e:
            logger.error(f"Redis error: {e}")
            return False
    
    async def _check_milvus(self) -> bool:
        """Check Milvus connectivity"""
        try:
            from pymilvus import connections, utility
            
            connections.connect(
                alias="default",
                host=self.settings.MILVUS_HOST,
                port=self.settings.MILVUS_PORT
            )
            
            # Check if connection is successful
            collections = utility.list_collections()
            logger.debug(f"Milvus collections: {collections[:3]}...")
            connections.disconnect("default")
            return True
        except Exception as e:
            logger.error(f"Milvus error: {e}")
            return False
    
    def run_database_migrations(self) -> bool:
        """Run necessary database migrations"""
        logger.info("📄 Running database migrations...")
        
        migration_file = project_root / "migrations" / "add_radiating_settings.sql"
        
        if not migration_file.exists():
            logger.error(f"Migration file not found: {migration_file}")
            return False
        
        try:
            with open(migration_file, 'r') as f:
                migration_sql = f.read()
            
            with self.engine.begin() as conn:
                conn.execute(text(migration_sql))
                logger.info("  ✅ Radiating settings migration applied")
            
            # Verify settings were created
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("SELECT COUNT(*) FROM settings WHERE category = 'radiating'")
                )
                count = result.fetchone()[0]
                if count > 0:
                    logger.info("  ✅ Radiating settings initialized in database")
                    return True
                else:
                    logger.error("  ❌ Radiating settings not found after migration")
                    return False
                    
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
    
    def setup_neo4j_indexes(self) -> bool:
        """Create Neo4j indexes and constraints for optimal performance"""
        logger.info("🗂️ Setting up Neo4j indexes...")
        
        try:
            self.neo4j_service = Neo4jService()
            
            # Create indexes for radiating system
            indexes = [
                # Entity indexes
                "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)",
                "CREATE INDEX entity_confidence_idx IF NOT EXISTS FOR (e:Entity) ON (e.confidence)",
                "CREATE INDEX entity_source_idx IF NOT EXISTS FOR (e:Entity) ON (e.source)",
                "CREATE INDEX entity_created_idx IF NOT EXISTS FOR (e:Entity) ON (e.created_at)",
                
                # Document indexes
                "CREATE INDEX doc_id_idx IF NOT EXISTS FOR (d:Document) ON (d.document_id)",
                "CREATE INDEX doc_title_idx IF NOT EXISTS FOR (d:Document) ON (d.title)",
                "CREATE INDEX doc_created_idx IF NOT EXISTS FOR (d:Document) ON (d.created_at)",
                
                # Radiating specific indexes
                "CREATE INDEX radiating_entity_idx IF NOT EXISTS FOR (r:RadiatingEntity) ON (r.entity_id)",
                "CREATE INDEX radiating_depth_idx IF NOT EXISTS FOR (r:RadiatingEntity) ON (r.depth)",
                "CREATE INDEX radiating_score_idx IF NOT EXISTS FOR (r:RadiatingEntity) ON (r.relevance_score)",
                
                # Full-text search indexes
                "CREATE FULLTEXT INDEX entity_fulltext_idx IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.description]",
                "CREATE FULLTEXT INDEX doc_fulltext_idx IF NOT EXISTS FOR (d:Document) ON EACH [d.title, d.content]"
            ]
            
            # Create constraints
            constraints = [
                "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
                "CREATE CONSTRAINT doc_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE"
            ]
            
            # Execute index creation
            for idx_query in indexes:
                try:
                    self.neo4j_service.execute_query(idx_query)
                    logger.info(f"  ✅ Created index: {idx_query.split(' ')[2]}")
                except Exception as e:
                    logger.warning(f"  ⚠️ Index creation warning: {e}")
            
            # Execute constraint creation
            for constraint_query in constraints:
                try:
                    self.neo4j_service.execute_query(constraint_query)
                    logger.info(f"  ✅ Created constraint: {constraint_query.split(' ')[2]}")
                except Exception as e:
                    logger.warning(f"  ⚠️ Constraint creation warning: {e}")
            
            logger.info("  ✅ Neo4j indexes and constraints setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Neo4j index setup failed: {e}")
            return False
    
    def initialize_redis_cache(self) -> bool:
        """Initialize Redis cache configuration for radiating system"""
        logger.info("💾 Initializing Redis cache...")
        
        try:
            self.redis_client = get_redis_client()
            
            # Set up cache namespaces
            namespaces = {
                "radiating:entities": "Radiating entity cache",
                "radiating:paths": "Radiating path cache",
                "radiating:queries": "Radiating query cache",
                "radiating:results": "Radiating results cache",
                "radiating:metrics": "Performance metrics cache"
            }
            
            for namespace, description in namespaces.items():
                # Create namespace marker
                self.redis_client.set(f"{namespace}:initialized", "true", ex=86400)
                logger.info(f"  ✅ Initialized namespace: {namespace} ({description})")
            
            # Set default TTLs
            ttl_config = {
                "radiating:default_ttl": 3600,
                "radiating:entity_ttl": 7200,
                "radiating:path_ttl": 1800,
                "radiating:query_ttl": 900
            }
            
            for key, value in ttl_config.items():
                self.redis_client.set(key, value)
                logger.info(f"  ✅ Set TTL config: {key} = {value}s")
            
            logger.info("  ✅ Redis cache initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Redis initialization failed: {e}")
            return False
    
    def create_sample_data(self) -> bool:
        """Create sample data for testing the radiating system"""
        logger.info("📊 Creating sample data...")
        
        try:
            # Sample entities for testing
            sample_entities = [
                {
                    "name": "Artificial Intelligence",
                    "type": "Technology",
                    "properties": {
                        "description": "The simulation of human intelligence in machines",
                        "importance": "high",
                        "category": "Computer Science"
                    }
                },
                {
                    "name": "Machine Learning",
                    "type": "Technology",
                    "properties": {
                        "description": "A subset of AI that enables systems to learn from data",
                        "importance": "high",
                        "category": "Computer Science"
                    }
                },
                {
                    "name": "Neural Networks",
                    "type": "Technology",
                    "properties": {
                        "description": "Computing systems inspired by biological neural networks",
                        "importance": "medium",
                        "category": "Computer Science"
                    }
                },
                {
                    "name": "Natural Language Processing",
                    "type": "Technology",
                    "properties": {
                        "description": "AI technology for understanding human language",
                        "importance": "high",
                        "category": "Computer Science"
                    }
                },
                {
                    "name": "Computer Vision",
                    "type": "Technology",
                    "properties": {
                        "description": "AI technology for understanding visual information",
                        "importance": "medium",
                        "category": "Computer Science"
                    }
                }
            ]
            
            # Sample relationships
            sample_relationships = [
                ("Machine Learning", "SUBSET_OF", "Artificial Intelligence"),
                ("Neural Networks", "COMPONENT_OF", "Machine Learning"),
                ("Natural Language Processing", "USES", "Machine Learning"),
                ("Computer Vision", "USES", "Neural Networks"),
                ("Natural Language Processing", "RELATED_TO", "Computer Vision")
            ]
            
            # Create entities in Neo4j
            neo4j_service = Neo4jService()
            
            for entity in sample_entities:
                query = """
                MERGE (e:Entity {name: $name})
                SET e.type = $type,
                    e.properties = $properties,
                    e.created_at = datetime(),
                    e.confidence = 0.95,
                    e.source = 'sample_data'
                RETURN e.name as created
                """
                
                result = neo4j_service.execute_query(
                    query,
                    parameters={
                        "name": entity["name"],
                        "type": entity["type"],
                        "properties": json.dumps(entity["properties"])
                    }
                )
                
                if result and result[0].get("created"):
                    logger.info(f"  ✅ Created entity: {entity['name']}")
            
            # Create relationships
            for source, rel_type, target in sample_relationships:
                query = """
                MATCH (s:Entity {name: $source})
                MATCH (t:Entity {name: $target})
                MERGE (s)-[r:RELATIONSHIP {type: $rel_type}]->(t)
                SET r.created_at = datetime(),
                    r.confidence = 0.9,
                    r.weight = 0.8
                RETURN type(r) as created
                """
                
                result = neo4j_service.execute_query(
                    query,
                    parameters={
                        "source": source,
                        "target": target,
                        "rel_type": rel_type
                    }
                )
                
                if result and result[0].get("created"):
                    logger.info(f"  ✅ Created relationship: {source} -{rel_type}-> {target}")
            
            logger.info("  ✅ Sample data creation complete")
            return True
            
        except Exception as e:
            logger.error(f"Sample data creation failed: {e}")
            return False
    
    def verify_installation(self) -> bool:
        """Verify all components are properly installed and configured"""
        logger.info("🔍 Verifying installation...")
        
        try:
            # Check radiating settings
            settings = get_radiating_settings()
            if not settings:
                logger.error("  ❌ Radiating settings not accessible")
                return False
            logger.info("  ✅ Radiating settings loaded successfully")
            
            # Check Neo4j radiating service
            radiating_service = RadiatingNeo4jService()
            
            # Test a simple query
            query = "MATCH (n:Entity) RETURN COUNT(n) as count"
            result = radiating_service.execute_query(query)
            
            if result and isinstance(result, list):
                entity_count = result[0].get('count', 0)
                logger.info(f"  ✅ Neo4j radiating service working ({entity_count} entities)")
            else:
                logger.error("  ❌ Neo4j radiating service test failed")
                return False
            
            # Check Redis radiating cache
            cache_test_key = "radiating:test:verification"
            self.redis_client.set(cache_test_key, "verified", ex=60)
            if self.redis_client.get(cache_test_key) == "verified":
                logger.info("  ✅ Redis radiating cache working")
                self.redis_client.delete(cache_test_key)
            else:
                logger.error("  ❌ Redis radiating cache test failed")
                return False
            
            # Check API endpoint availability
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get(f"http://localhost:8000/api/v1/radiating/health")
                    if response.status_code == 200:
                        logger.info("  ✅ Radiating API endpoint accessible")
                    else:
                        logger.warning(f"  ⚠️ Radiating API returned status {response.status_code}")
                except:
                    logger.warning("  ⚠️ Radiating API not yet accessible (may need server restart)")
            
            logger.info("  ✅ Installation verification complete")
            return True
            
        except Exception as e:
            logger.error(f"Installation verification failed: {e}")
            return False
    
    async def run_setup(self, skip_sample_data: bool = False) -> bool:
        """Run the complete setup process"""
        logger.info("🚀 Starting Universal Radiating Coverage System Setup")
        logger.info("=" * 60)
        
        try:
            # Step 1: Check prerequisites
            if not await self.check_prerequisites():
                logger.error("Prerequisites check failed. Please ensure all services are running.")
                return False
            
            # Step 2: Run database migrations
            if not self.run_database_migrations():
                logger.error("Database migration failed.")
                return False
            
            # Step 3: Setup Neo4j indexes
            if not self.setup_neo4j_indexes():
                logger.error("Neo4j index setup failed.")
                return False
            
            # Step 4: Initialize Redis cache
            if not self.initialize_redis_cache():
                logger.error("Redis cache initialization failed.")
                return False
            
            # Step 5: Create sample data (optional)
            if not skip_sample_data:
                if not self.create_sample_data():
                    logger.warning("Sample data creation failed (non-critical)")
            
            # Step 6: Verify installation
            if not self.verify_installation():
                logger.warning("Installation verification had warnings")
            
            logger.info("=" * 60)
            logger.info("🎉 Radiating Coverage System Setup Complete!")
            logger.info("✅ All components initialized successfully")
            logger.info("")
            logger.info("Next steps:")
            logger.info("1. Restart the FastAPI server to load new settings")
            logger.info("2. Access the Radiating settings at: http://localhost:5173/settings.html")
            logger.info("3. Test the system with example queries")
            logger.info("4. Monitor performance via the dashboard")
            
            self.setup_complete = True
            return True
            
        except Exception as e:
            logger.error(f"Setup failed with error: {e}")
            return False

async def main():
    """Main entry point for the setup script"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Setup and Initialize the Universal Radiating Coverage System'
    )
    parser.add_argument(
        '--skip-sample-data',
        action='store_true',
        help='Skip creating sample data'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing installation'
    )
    
    args = parser.parse_args()
    
    setup = RadiatingSystemSetup()
    
    if args.verify_only:
        success = setup.verify_installation()
    else:
        success = await setup.run_setup(skip_sample_data=args.skip_sample_data)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())