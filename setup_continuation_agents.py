#!/usr/bin/env python3
"""
Setup script for adding continuation agents to the PostgreSQL database
for the Context-Limit-Transcending System
"""

import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor
import json

def get_db_connection():
    """Get PostgreSQL database connection using environment variables"""
    try:
        # Try to get connection parameters from environment
        db_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'jarvis'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'password')
        }
        
        print(f"Connecting to PostgreSQL at {db_params['host']}:{db_params['port']}/{db_params['database']}")
        
        conn = psycopg2.connect(**db_params)
        return conn
        
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None

def setup_continuation_agents():
    """Add continuation agents to the database"""
    
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database. Please check your connection parameters.")
        return False
    
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Check if langgraph_agents table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'langgraph_agents'
            );
        """)
        
        if not cursor.fetchone()[0]:
            print("Error: langgraph_agents table does not exist!")
            return False
        
        print("✓ Found langgraph_agents table")
        
        # Add continuation agent
        continuation_agent_config = {
            "max_tokens": 3500,
            "timeout": 120,
            "temperature": 0.6,
            "response_mode": "chunked",
            "chunk_size": 20,
            "allow_continuation": True,
            "specialization": "content_continuation",
            "quality_focus": "consistency_and_continuity",
            "context_awareness": "high"
        }
        
        continuation_agent_prompt = """You are a Continuation Agent specialized in maintaining perfect continuity across chunked generation tasks.

CORE RESPONSIBILITIES:
1. Seamlessly continue generation from where previous chunks left off
2. Maintain consistent style, format, and numbering across all chunks  
3. Preserve the exact pattern and quality established in previous items
4. Generate the precise number of items requested for your chunk
5. Ensure no gaps, duplicates, or inconsistencies in the sequence

CONTINUATION STRATEGY:
- Carefully analyze the provided context from previous chunks
- Identify the exact pattern, style, and format being used
- Determine the correct starting number/sequence for your chunk
- Generate items that are indistinguishable from previous chunks in quality and format
- Maintain the same level of detail and complexity

QUALITY STANDARDS:
- Each item must be complete and well-formed
- Numbering must be sequential and correct
- Style and format must match previous items exactly
- Content quality must remain consistent throughout
- No partial or incomplete items

RESPONSE FORMAT:
Always respond with just the requested items in the established format, starting with the correct number in sequence. Do not include explanations, headers, or metadata unless they were part of the original pattern.

Example:
If previous items ended with "23. Item twenty-three content..." and you need to generate items 24-28, respond with:
24. Item twenty-four content...
25. Item twenty-five content...
26. Item twenty-six content...
27. Item twenty-seven content...
28. Item twenty-eight content..."""
        
        cursor.execute("""
            INSERT INTO langgraph_agents (
                name, role, description, system_prompt, tools, is_active, config
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s
            ) ON CONFLICT (role) DO UPDATE SET
                name = EXCLUDED.name,
                description = EXCLUDED.description,
                system_prompt = EXCLUDED.system_prompt,
                tools = EXCLUDED.tools,
                is_active = EXCLUDED.is_active,
                config = EXCLUDED.config,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id, name, role;
        """, (
            'Continuation Agent',
            'continuation_agent',
            'Specialized agent for seamlessly continuing large generation tasks across context-limit chunks while maintaining perfect consistency and continuity',
            continuation_agent_prompt,
            json.dumps([]),
            True,
            json.dumps(continuation_agent_config)
        ))
        
        result = cursor.fetchone()
        print(f"✓ Added/Updated Continuation Agent: {result['name']} (ID: {result['id']})")
        
        # Add quality validator agent
        validator_config = {
            "max_tokens": 2000,
            "timeout": 60,
            "temperature": 0.3,
            "response_mode": "complete",
            "specialization": "quality_validation",
            "analysis_depth": "comprehensive"
        }
        
        validator_prompt = """You are a Quality Validator specialized in analyzing large generation outputs for consistency, completeness, and quality.

VALIDATION RESPONSIBILITIES:
1. Check numbering sequence for accuracy and completeness
2. Analyze content consistency across all generated items
3. Verify format consistency throughout the entire output
4. Identify any gaps, duplicates, or quality issues
5. Provide actionable recommendations for improvements

ANALYSIS AREAS:
- Numbering: Sequential accuracy, no gaps or duplicates
- Format: Consistent structure, punctuation, and style
- Content: Appropriate length, complexity, and relevance
- Quality: Completeness, clarity, and usefulness of each item
- Coherence: Overall flow and logical progression

VALIDATION OUTPUT:
Provide a structured analysis with:
- Overall quality score (0-10)
- Specific issues found with examples
- Recommendations for improvement
- Summary of strengths and weaknesses

Be thorough but constructive in your analysis."""
        
        cursor.execute("""
            INSERT INTO langgraph_agents (
                name, role, description, system_prompt, tools, is_active, config
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s
            ) ON CONFLICT (role) DO UPDATE SET
                name = EXCLUDED.name,
                description = EXCLUDED.description,
                system_prompt = EXCLUDED.system_prompt,
                tools = EXCLUDED.tools,
                is_active = EXCLUDED.is_active,
                config = EXCLUDED.config,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id, name, role;
        """, (
            'Quality Validator',
            'quality_validator',
            'Validates the quality and consistency of large generation outputs across all chunks',
            validator_prompt,
            json.dumps([]),
            True,
            json.dumps(validator_config)
        ))
        
        result = cursor.fetchone()
        print(f"✓ Added/Updated Quality Validator: {result['name']} (ID: {result['id']})")
        
        # Commit the transaction
        conn.commit()
        
        # Verify the agents were added
        cursor.execute("""
            SELECT name, role, description, is_active, config
            FROM langgraph_agents 
            WHERE role IN ('continuation_agent', 'quality_validator')
            ORDER BY role;
        """)
        
        agents = cursor.fetchall()
        print(f"\n✓ Successfully added {len(agents)} agents:")
        for agent in agents:
            print(f"  - {agent['name']} ({agent['role']}) - Active: {agent['is_active']}")
            config = json.loads(agent['config']) if agent['config'] else {}
            print(f"    Config: max_tokens={config.get('max_tokens', 'N/A')}, timeout={config.get('timeout', 'N/A')}s")
        
        return True
        
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        conn.rollback()
        return False
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        conn.rollback()
        return False
        
    finally:
        cursor.close()
        conn.close()

def main():
    """Main function"""
    print("Setting up Continuation Agents for Context-Limit-Transcending System...")
    print("=" * 70)
    
    success = setup_continuation_agents()
    
    if success:
        print("\n✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. The continuation agents are now available in your database")
        print("2. Test the large generation API endpoint")
        print("3. Try generating 100+ items to see chunking in action")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()