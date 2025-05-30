"""
Script to fix Redis connection issues in cache modules
"""

import os
import re

cache_files = [
    "app/core/embedding_settings_cache.py",
    "app/core/iceberg_settings_cache.py", 
    "app/core/mcp_tools_cache.py",
    "app/core/vector_db_settings_cache.py"
]

def fix_redis_connection(filepath):
    """Fix Redis connection to use lazy initialization"""
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Skip if already fixed
    if "_get_redis_client" in content:
        print(f"✓ {filepath} already fixed")
        return
    
    # Pattern to find Redis connection initialization
    pattern = r'(r = redis\.Redis\(.*?\))'
    
    # Replace with lazy initialization
    replacement = '''# Don't create connection at import time
r = None

def _get_redis_client():
    """Get Redis client with lazy initialization"""
    global r
    if r is None:
        try:
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            r.ping()
        except Exception as e:
            print(f"Redis connection failed: {e}")
            r = None
    return r'''
    
    # Replace the Redis initialization
    new_content = re.sub(pattern, replacement, content)
    
    # Update all r.get, r.set, r.hset etc calls to use _get_redis_client()
    def replace_redis_call(match):
        method = match.group(1)
        args = match.group(2)
        return f'''redis_client = _get_redis_client()
    if redis_client:
        try:
            return redis_client.{method}({args})
        except Exception as e:
            print(f"Redis error: {{e}}")
            return None'''
    
    # This is a simplified fix - in practice we'd need more sophisticated replacements
    print(f"! {filepath} needs manual fixing - Redis calls need to be updated")
    
    with open(filepath + '.backup', 'w') as f:
        f.write(content)
    print(f"  Created backup: {filepath}.backup")

if __name__ == "__main__":
    for filepath in cache_files:
        if os.path.exists(filepath):
            fix_redis_connection(filepath)
        else:
            print(f"✗ {filepath} not found")