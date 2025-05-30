#!/usr/bin/env python
"""
Script to check agent prompts for [Unknown] references
"""
import os
import sys

# Add the parent directory to the path so we can import app
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

from app.core.db import SessionLocal, LangGraphAgent

def check_agent_prompts():
    db = SessionLocal()
    try:
        print("Checking agent prompts for [Unknown] references...\n")
        
        # Get all agents
        agents = db.query(LangGraphAgent).filter(LangGraphAgent.is_active == True).all()
        
        for agent in agents:
            if "[Unknown]" in agent.system_prompt or "[Unknown]" in agent.description:
                print(f"Found [Unknown] in agent: {agent.name}")
                print(f"  Role: {agent.role}")
                
                if "[Unknown]" in agent.system_prompt:
                    # Show context around [Unknown]
                    prompt_lines = agent.system_prompt.split('\n')
                    for i, line in enumerate(prompt_lines):
                        if "[Unknown]" in line:
                            print(f"  Line {i+1}: {line.strip()}")
                            # Show surrounding lines for context
                            if i > 0:
                                print(f"  Line {i}: {prompt_lines[i-1].strip()}")
                            if i < len(prompt_lines) - 1:
                                print(f"  Line {i+2}: {prompt_lines[i+1].strip()}")
                print()
        
        # Also check for specific problematic agent
        tech_architect = db.query(LangGraphAgent).filter(
            LangGraphAgent.name == "technical_architect"
        ).first()
        
        if tech_architect:
            print("\nTechnical Architect System Prompt:")
            print("-" * 50)
            print(tech_architect.system_prompt[:500] + "..." if len(tech_architect.system_prompt) > 500 else tech_architect.system_prompt)
            
    finally:
        db.close()

if __name__ == "__main__":
    check_agent_prompts()