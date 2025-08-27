#!/usr/bin/env python3

"""
Automated Claude .claude.json Cleanup Tool
Silent operation for cron/automation - no prompts
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

def cleanup_claude_json():
    """Clean the .claude.json file silently"""
    home = Path.home()
    claude_json = home / ".claude.json"
    
    # Check if file exists
    if not claude_json.exists():
        print(f"SKIP: .claude.json not found at {claude_json}")
        return 0
    
    # Get initial size
    initial_size = claude_json.stat().st_size
    
    # Read and parse JSON
    try:
        with open(claude_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: JSON decode failed - {str(e)}")
        return 1
    except Exception as e:
        print(f"ERROR: Failed to read file - {str(e)}")
        return 1
    
    # Count what we'll remove
    total_history = 0
    cleaned_projects = 0
    
    if 'projects' in data:
        for project_path, project_data in data['projects'].items():
            if isinstance(project_data, dict) and 'history' in project_data:
                total_history += len(project_data['history'])
                del project_data['history']
                cleaned_projects += 1
    
    # Skip if nothing to clean
    if total_history == 0:
        print("OK: No history to clean")
        return 0
    
    # Create backup only if significant cleanup
    if total_history > 0:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_json = home / f".claude.json.backup.{timestamp}"
        shutil.copy2(claude_json, backup_json)
    
    # Write cleaned file
    try:
        with open(claude_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"ERROR: Failed to write cleaned file - {str(e)}")
        return 1
    
    # Calculate space saved
    new_size = claude_json.stat().st_size
    saved_bytes = initial_size - new_size
    saved_mb = saved_bytes / (1024 * 1024)
    
    # Log results
    print(f"CLEANED: {cleaned_projects} projects, {total_history} entries, {saved_mb:.2f}MB saved")
    
    return 0

if __name__ == "__main__":
    exit_code = cleanup_claude_json()
    sys.exit(exit_code)
