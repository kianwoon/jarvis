#!/usr/bin/env python3
"""Add is_active column to langgraph_agents table"""

from sqlalchemy import create_engine, text
from app.core.config import get_settings
import os

settings = get_settings()

# Try PostgreSQL first, fall back to SQLite
try:
    DATABASE_URL = (
        f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )
    engine = create_engine(DATABASE_URL)
    is_postgres = True
    print("Connected to PostgreSQL")
except Exception as e:
    print(f"PostgreSQL connection failed: {e}")
    DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sqlite.db")
    DATABASE_URL = f"sqlite:///{DB_PATH}"
    engine = create_engine(DATABASE_URL)
    is_postgres = False
    print("Connected to SQLite")

def add_is_active_column():
    """Add is_active column if it doesn't exist"""
    with engine.connect() as conn:
        # Check if column exists
        if is_postgres:
            check_query = text("""
                SELECT EXISTS (
                    SELECT 1 
                    FROM information_schema.columns 
                    WHERE table_name = 'langgraph_agents' 
                    AND column_name = 'is_active'
                );
            """)
        else:
            # SQLite approach
            check_query = text("PRAGMA table_info(langgraph_agents)")
        
        result = conn.execute(check_query)
        
        if is_postgres:
            column_exists = result.scalar()
        else:
            # For SQLite, check if 'is_active' is in the column list
            columns = [row[1] for row in result]
            column_exists = 'is_active' in columns
        
        if not column_exists:
            print("Adding is_active column...")
            if is_postgres:
                alter_query = text("""
                    ALTER TABLE langgraph_agents 
                    ADD COLUMN is_active BOOLEAN DEFAULT TRUE;
                """)
            else:
                # SQLite doesn't support ALTER TABLE ADD COLUMN with DEFAULT in older versions
                # So we'll add it without default and then update
                alter_query = text("""
                    ALTER TABLE langgraph_agents 
                    ADD COLUMN is_active BOOLEAN;
                """)
            
            conn.execute(alter_query)
            conn.commit()
            
            # Set default value for existing rows
            update_query = text("""
                UPDATE langgraph_agents 
                SET is_active = TRUE 
                WHERE is_active IS NULL;
            """)
            conn.execute(update_query)
            conn.commit()
            
            print("Successfully added is_active column")
        else:
            print("is_active column already exists")

if __name__ == "__main__":
    add_is_active_column()