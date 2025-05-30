"""
Migration script to add config column to langgraph_agents table
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.db import engine, SessionLocal, LangGraphAgent
from sqlalchemy import text
import json

def migrate_agent_config():
    """Add config column and set default values"""
    
    # Read SQL file
    with open('add_agent_config_columns.sql', 'r') as f:
        sql_commands = f.read().split(';')
    
    # Execute each command
    with engine.connect() as conn:
        for command in sql_commands:
            command = command.strip()
            if command:
                try:
                    conn.execute(text(command))
                    conn.commit()
                    print(f"Executed: {command[:50]}...")
                except Exception as e:
                    print(f"Error executing command: {e}")
                    # Continue with other commands
    
    print("Migration completed!")
    
    # Verify the changes
    db = SessionLocal()
    try:
        agents = db.query(LangGraphAgent).all()
        print("\nAgent configurations:")
        for agent in agents:
            config = agent.config if hasattr(agent, 'config') else {}
            print(f"- {agent.role}: {json.dumps(config)}")
    finally:
        db.close()

if __name__ == "__main__":
    migrate_agent_config()