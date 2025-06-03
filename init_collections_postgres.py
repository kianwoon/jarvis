"""
Initialize collections system in PostgreSQL - create tables and default collection
Run this inside the Docker container where PostgreSQL is accessible
"""

import json
from sqlalchemy import text, create_engine
from sqlalchemy.orm import sessionmaker
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PostgreSQL connection - use environment variables from Docker
DATABASE_URL = (
    f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}:{os.getenv('POSTGRES_PASSWORD', 'postgres')}"
    f"@{os.getenv('POSTGRES_HOST', 'postgres')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'llmops')}"
)

def init_collections_postgres():
    """Initialize collection system in PostgreSQL"""
    
    # Create engine and session
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Step 1: Create tables using raw SQL for PostgreSQL
    try:
        with engine.connect() as conn:
            # Create collection_registry table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS collection_registry (
                    id SERIAL PRIMARY KEY,
                    collection_name VARCHAR(100) UNIQUE NOT NULL,
                    collection_type VARCHAR(50) NOT NULL,
                    description TEXT,
                    metadata_schema JSONB,
                    search_config JSONB,
                    access_config JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Create collection_statistics table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS collection_statistics (
                    id SERIAL PRIMARY KEY,
                    collection_name VARCHAR(100) UNIQUE NOT NULL REFERENCES collection_registry(collection_name),
                    document_count INTEGER DEFAULT 0,
                    total_chunks INTEGER DEFAULT 0,
                    storage_size_mb FLOAT DEFAULT 0.0,
                    avg_search_latency_ms FLOAT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Create user_collection_access table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS user_collection_access (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(100) NOT NULL,
                    collection_name VARCHAR(100) NOT NULL REFERENCES collection_registry(collection_name),
                    permission_level VARCHAR(20) DEFAULT 'read',
                    granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            conn.commit()
            logger.info("✅ Collection tables created successfully in PostgreSQL")
    except Exception as e:
        logger.error(f"❌ Error creating collection tables: {e}")
        return False
    
    # Step 2: Create default collection
    db = SessionLocal()
    try:
        # Check if default collection exists
        result = db.execute(
            text("SELECT id FROM collection_registry WHERE collection_name = :name"),
            {"name": 'default_knowledge'}
        ).fetchone()
        
        if result:
            logger.info("ℹ️  Default collection 'default_knowledge' already exists")
            return True
        
        # Create default collection
        collection_data = {
            'collection_name': 'default_knowledge',
            'collection_type': 'general',
            'description': 'Default collection for general documents and knowledge base',
            'metadata_schema': {
                'chunk_size': 1500,
                'chunk_overlap': 200,
                'fields': []
            },
            'search_config': {
                'strategy': 'balanced',
                'similarity_threshold': 0.7,
                'max_results': 10,
                'enable_bm25': True,
                'bm25_weight': 0.3
            },
            'access_config': {
                'restricted': False,
                'allowed_users': []
            }
        }
        
        # Insert collection
        db.execute(text("""
            INSERT INTO collection_registry 
            (collection_name, collection_type, description, metadata_schema, 
             search_config, access_config, created_at, updated_at)
            VALUES (:collection_name, :collection_type, :description, :metadata_schema::jsonb, 
                    :search_config::jsonb, :access_config::jsonb, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """), {
            'collection_name': collection_data['collection_name'],
            'collection_type': collection_data['collection_type'],
            'description': collection_data['description'],
            'metadata_schema': json.dumps(collection_data['metadata_schema']),
            'search_config': json.dumps(collection_data['search_config']),
            'access_config': json.dumps(collection_data['access_config'])
        })
        
        # Initialize statistics
        db.execute(text("""
            INSERT INTO collection_statistics 
            (collection_name, document_count, total_chunks, storage_size_mb, last_updated)
            VALUES (:collection_name, 0, 0, 0.0, CURRENT_TIMESTAMP)
        """), {'collection_name': collection_data['collection_name']})
        
        db.commit()
        logger.info("✅ Default collection 'default_knowledge' created successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating default collection: {e}")
        db.rollback()
        return False
    finally:
        db.close()

if __name__ == "__main__":
    print("🚀 Initializing collections system in PostgreSQL...")
    print(f"📍 Connecting to: {DATABASE_URL}")
    if init_collections_postgres():
        print("✅ Collections system initialized successfully in PostgreSQL!")
    else:
        print("❌ Failed to initialize collections system")