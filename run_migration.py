#!/usr/bin/env python3
"""
Knowledge Graph Enhancement Migration Runner

This script safely applies database migrations for the knowledge graph enhancements.
It includes rollback capabilities and comprehensive error handling.
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Add the app directory to the path so we can import our config
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'migration_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class MigrationRunner:
    """Safe database migration runner with rollback capabilities"""
    
    def __init__(self):
        self.settings = get_settings()
        self.engine = self._create_engine()
        self.migration_file = Path(__file__).parent / "migrations" / "add_knowledge_graph_enhancements.sql"
        
    def _create_engine(self):
        """Create database engine with proper configuration"""
        database_url = (
            f"postgresql://{self.settings.POSTGRES_USER}:{self.settings.POSTGRES_PASSWORD}"
            f"@{self.settings.POSTGRES_HOST}:{self.settings.POSTGRES_PORT}/{self.settings.POSTGRES_DB}"
        )
        
        return create_engine(
            database_url,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=True  # Enable SQL logging for migration
        )
    
    def check_connection(self):
        """Verify database connection"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT version()"))
                version = result.fetchone()[0]
                logger.info(f"✅ Database connection successful. PostgreSQL version: {version}")
                return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def backup_existing_tables(self):
        """Create backup of existing tables that might be affected"""
        backup_queries = [
            "CREATE TABLE IF NOT EXISTS settings_backup AS SELECT * FROM settings WHERE category = 'llm';"
        ]
        
        try:
            with self.engine.begin() as conn:
                for query in backup_queries:
                    logger.info(f"Creating backup: {query}")
                    conn.execute(text(query))
            logger.info("✅ Backup completed successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            return False
    
    def check_table_exists(self, table_name):
        """Check if a table already exists"""
        query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = :table_name
        );
        """
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), {"table_name": table_name})
                return result.fetchone()[0]
        except Exception as e:
            logger.error(f"Error checking table existence: {e}")
            return False
    
    def run_migration(self, force=False):
        """Execute the migration with safety checks"""
        logger.info("🚀 Starting Knowledge Graph Enhancement Migration")
        
        # Pre-flight checks
        if not self.check_connection():
            logger.error("❌ Migration aborted: Database connection failed")
            return False
        
        # Check if migration was already applied
        tables_to_check = [
            'knowledge_graph_documents',
            'extraction_quality_metrics', 
            'graph_schema_evolution',
            'document_cross_references'
        ]
        
        existing_tables = []
        for table in tables_to_check:
            if self.check_table_exists(table):
                existing_tables.append(table)
        
        if existing_tables and not force:
            logger.warning(f"⚠️ Tables already exist: {existing_tables}")
            logger.warning("Migration may have been applied already. Use --force to override.")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                logger.info("Migration aborted by user")
                return False
        
        # Create backup
        if not self.backup_existing_tables():
            logger.error("❌ Migration aborted: Backup failed")
            return False
        
        # Read migration file
        if not self.migration_file.exists():
            logger.error(f"❌ Migration file not found: {self.migration_file}")
            return False
        
        try:
            with open(self.migration_file, 'r') as f:
                migration_sql = f.read()
            
            logger.info(f"📄 Loaded migration from: {self.migration_file}")
            
            # Execute migration
            with self.engine.begin() as conn:
                logger.info("🔄 Executing migration...")
                conn.execute(text(migration_sql))
                logger.info("✅ Migration executed successfully")
            
            # Verify results
            return self._verify_migration()
            
        except SQLAlchemyError as e:
            logger.error(f"❌ Migration failed with database error: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Migration failed with unexpected error: {e}")
            return False
    
    def _verify_migration(self):
        """Verify that migration was successful"""
        verification_queries = [
            ("knowledge_graph_documents", "SELECT COUNT(*) FROM knowledge_graph_documents"),
            ("extraction_quality_metrics", "SELECT COUNT(*) FROM extraction_quality_metrics"),
            ("graph_schema_evolution", "SELECT COUNT(*) FROM graph_schema_evolution"),
            ("document_cross_references", "SELECT COUNT(*) FROM document_cross_references"),
        ]
        
        try:
            with self.engine.connect() as conn:
                logger.info("🔍 Verifying migration results...")
                
                for table_name, query in verification_queries:
                    result = conn.execute(text(query))
                    count = result.fetchone()[0]
                    logger.info(f"✅ Table '{table_name}': {count} rows")
                
                # Check that initial schema version was inserted
                schema_check = conn.execute(text("SELECT schema_version FROM graph_schema_evolution WHERE schema_version = '1.0.0'"))
                if schema_check.fetchone():
                    logger.info("✅ Initial schema version (1.0.0) inserted successfully")
                else:
                    logger.warning("⚠️ Initial schema version not found")
                
                logger.info("✅ Migration verification completed successfully")
                return True
                
        except Exception as e:
            logger.error(f"❌ Migration verification failed: {e}")
            return False
    
    def rollback_migration(self):
        """Rollback the migration (drop created tables)"""
        rollback_queries = [
            "DROP TABLE IF EXISTS document_cross_references CASCADE;",
            "DROP TABLE IF EXISTS extraction_quality_metrics CASCADE;", 
            "DROP TABLE IF EXISTS knowledge_graph_documents CASCADE;",
            "DROP TABLE IF EXISTS graph_schema_evolution CASCADE;",
            "DROP FUNCTION IF EXISTS update_updated_at_column() CASCADE;"
        ]
        
        try:
            with self.engine.begin() as conn:
                logger.info("🔄 Rolling back migration...")
                for query in rollback_queries:
                    logger.info(f"Executing: {query}")
                    conn.execute(text(query))
                
                logger.info("✅ Migration rollback completed successfully")
                return True
                
        except Exception as e:
            logger.error(f"❌ Migration rollback failed: {e}")
            return False

def main():
    """Main migration runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Knowledge Graph Enhancement Migration Runner')
    parser.add_argument('--rollback', action='store_true', help='Rollback the migration')
    parser.add_argument('--force', action='store_true', help='Force migration even if tables exist')
    parser.add_argument('--verify-only', action='store_true', help='Only verify migration status')
    
    args = parser.parse_args()
    
    runner = MigrationRunner()
    
    if args.verify_only:
        runner._verify_migration()
        return
    
    if args.rollback:
        logger.info("⚠️ ROLLBACK MODE: This will DROP all knowledge graph enhancement tables!")
        response = input("Are you sure you want to rollback? (y/N): ")
        if response.lower() == 'y':
            success = runner.rollback_migration()
        else:
            logger.info("Rollback aborted by user")
            return
    else:
        success = runner.run_migration(force=args.force)
    
    if success:
        logger.info("🎉 Migration completed successfully!")
        sys.exit(0)
    else:
        logger.error("💥 Migration failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()