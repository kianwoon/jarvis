#!/usr/bin/env python3
"""
Fix LLM prompt caching issue by:
1. Verifying current database prompt
2. Clearing all caches
3. Reloading settings
4. Optionally restarting the backend service
"""

import json
import subprocess
import sys
import time
from typing import Dict, Any

def run_command(cmd: str) -> str:
    """Run a shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        print(f"Error running command: {e}")
        return ""

def check_database_prompt():
    """Check the current prompt in database"""
    print("\n🔍 Checking database prompt...")
    cmd = """PGPASSWORD=postgres psql -h localhost -p 5432 -U postgres -d llm_platform -t -c "SELECT length(settings->'main_llm'->>'system_prompt') as prompt_length FROM settings WHERE category = 'llm';" """
    result = run_command(cmd).strip()
    print(f"   Database prompt length: {result} characters")
    
    # Get first 200 chars of prompt
    cmd = """PGPASSWORD=postgres psql -h localhost -p 5432 -U postgres -d llm_platform -t -c "SELECT left(settings->'main_llm'->>'system_prompt', 200) FROM settings WHERE category = 'llm';" """
    preview = run_command(cmd).strip()
    print(f"   Preview: {preview[:100]}...")
    
    return int(result) if result.isdigit() else 0

def check_redis_cache():
    """Check current Redis cache"""
    print("\n🔍 Checking Redis cache...")
    cmd = "redis-cli -h localhost -p 6379 get llm_settings_cache"
    result = run_command(cmd)
    
    if result:
        try:
            settings = json.loads(result)
            prompt = settings.get('main_llm', {}).get('system_prompt', '')
            print(f"   Redis cache prompt length: {len(prompt)} characters")
            print(f"   Preview: {prompt[:100]}...")
            return len(prompt)
        except:
            print("   ❌ Failed to parse Redis cache")
            return 0
    else:
        print("   ⚠️ No Redis cache found")
        return 0

def clear_redis_cache():
    """Clear Redis LLM settings cache"""
    print("\n🧹 Clearing Redis cache...")
    cmd = "redis-cli -h localhost -p 6379 del llm_settings_cache"
    result = run_command(cmd).strip()
    if result == "1":
        print("   ✅ Redis cache cleared successfully")
        return True
    else:
        print("   ⚠️ No cache to clear or failed")
        return False

def reload_llm_cache():
    """Force reload LLM cache via API"""
    print("\n🔄 Reloading LLM cache via API...")
    cmd = "curl -s -X POST http://localhost:8000/api/v1/settings/llm/cache/reload"
    result = run_command(cmd)
    
    try:
        response = json.loads(result)
        if response.get('success'):
            print("   ✅ Cache reloaded successfully")
            return True
        else:
            print(f"   ❌ Failed: {response}")
            return False
    except:
        print(f"   ❌ Failed to reload cache: {result}")
        return False

def restart_backend():
    """Restart the backend Docker container"""
    print("\n🔄 Restarting backend service...")
    
    # Get container ID
    cmd = "docker ps --filter name=jarvis-app-1 --format '{{.ID}}'"
    container_id = run_command(cmd).strip()
    
    if not container_id:
        print("   ❌ Backend container not found")
        return False
    
    print(f"   Found container: {container_id}")
    
    # Restart container
    cmd = f"docker restart {container_id}"
    result = run_command(cmd).strip()
    
    if result == container_id:
        print("   ✅ Backend restarted successfully")
        print("   ⏳ Waiting for service to be ready...")
        time.sleep(10)
        
        # Check if service is responding
        cmd = "curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/api/v1/health"
        for i in range(10):
            status = run_command(cmd).strip()
            if status == "200":
                print("   ✅ Backend is ready")
                return True
            time.sleep(2)
        
        print("   ⚠️ Backend restarted but not responding yet")
        return False
    else:
        print("   ❌ Failed to restart backend")
        return False

def verify_fix():
    """Verify that the fix worked"""
    print("\n✅ Verifying fix...")
    
    # Check database
    db_length = check_database_prompt()
    
    # Check Redis after reload
    redis_length = check_redis_cache()
    
    if db_length > 1500 and redis_length > 1500:
        print("\n🎉 SUCCESS: New prompt is active in both database and cache!")
        print(f"   Database prompt: {db_length} characters")
        print(f"   Redis cache: {redis_length} characters")
        return True
    else:
        print("\n⚠️ WARNING: Prompt lengths don't match expected values")
        print(f"   Database: {db_length} (expected > 1500)")
        print(f"   Redis: {redis_length} (expected > 1500)")
        return False

def main():
    """Main execution flow"""
    print("=" * 60)
    print("🛠️  LLM Prompt Cache Fix Tool")
    print("=" * 60)
    
    # Step 1: Check current state
    print("\n📊 Current State:")
    db_length = check_database_prompt()
    redis_length = check_redis_cache()
    
    if db_length == redis_length and db_length > 1500:
        print("\n✅ Prompts are already synchronized and updated!")
        return 0
    
    # Step 2: Clear cache
    clear_redis_cache()
    
    # Step 3: Reload cache
    if not reload_llm_cache():
        print("\n⚠️ Cache reload failed, trying backend restart...")
        restart_backend()
        time.sleep(5)
        reload_llm_cache()
    
    # Step 4: Verify
    time.sleep(2)
    if verify_fix():
        print("\n🎉 Fix completed successfully!")
        print("\n📝 Next steps:")
        print("   1. Test the chat to confirm new prompt is active")
        print("   2. If still not working, restart backend: docker restart jarvis-app-1")
        return 0
    else:
        print("\n⚠️ Fix may not be complete. Consider restarting backend:")
        print("   docker restart jarvis-app-1")
        return 1

if __name__ == "__main__":
    sys.exit(main())