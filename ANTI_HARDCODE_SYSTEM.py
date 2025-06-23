"""
PERMANENT ANTI-HARDCODE ENFORCEMENT SYSTEM
==========================================

This module prevents hardcoding by enforcing dynamic configuration patterns.
Any code that violates the no-hardcode rule will fail immediately.
"""

import ast
import re
import sys
from typing import List, Dict, Any
from pathlib import Path

class HardcodeDetector:
    """Detects and prevents hardcoded values in code"""
    
    # FORBIDDEN PATTERNS - These will cause immediate failure
    FORBIDDEN_PATTERNS = [
        # Numeric literals in business logic
        r'\.get\(["\'][^"\']+["\']\s*,\s*\d+\.?\d*\)',  # dict.get('key', 123)
        r'=\s*\d+\.?\d*\s*#.*(?:threshold|limit|max|min|count)',  # var = 0.5 # threshold
        r'if\s+\w+\s*[<>=]+\s*\d+\.?\d*',  # if score > 0.5
        r'range\(\d+\)',  # range(10)
        r'[:]\d+[:]?',  # slice operations like [:500]
        
        # String literals for keywords/config
        r'\[["\'][^"\']*["\']\s*,\s*["\'][^"\']*["\'].*\]',  # ['keyword1', 'keyword2']
        r'=\s*["\'][^"\']*["\'].*#.*(?:keyword|config|setting)',
        
        # Time/size limits
        r'sleep\(\d+\)',  # sleep(5)
        r'timeout.*=.*\d+',  # timeout=30
        r'max.*=.*\d+',  # max_items=10
    ]
    
    # ALLOWED PATTERNS - These are exceptions
    ALLOWED_PATTERNS = [
        r'get_llm_settings\(\)',
        r'\.get\(["\'][^"\']+["\']\s*,\s*{}\)',  # .get('key', {})
        r'\.get\(["\'][^"\']+["\']\s*,\s*\[\]\)',  # .get('key', [])
        r'\.get\(["\'][^"\']+["\']\s*,\s*["\'][^"\']*["\'].*\)',  # .get('key', 'default')
    ]

def check_file_for_hardcoding(file_path: str) -> List[str]:
    """Check a single file for hardcoded values"""
    violations = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Skip comments and imports
            if line.strip().startswith('#') or line.strip().startswith('import') or line.strip().startswith('from'):
                continue
                
            # Check for forbidden patterns
            for pattern in HardcodeDetector.FORBIDDEN_PATTERNS:
                if re.search(pattern, line):
                    # Check if it's an allowed exception
                    is_allowed = any(re.search(allowed, line) for allowed in HardcodeDetector.ALLOWED_PATTERNS)
                    if not is_allowed:
                        violations.append(f"Line {line_num}: {line.strip()}")
                        
    except Exception as e:
        violations.append(f"Error reading file: {e}")
        
    return violations

def enforce_no_hardcode_rule(directories: List[str]) -> bool:
    """Enforce no-hardcode rule across directories"""
    all_violations = {}
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            continue
            
        python_files = list(path.rglob("*.py"))
        
        for file_path in python_files:
            violations = check_file_for_hardcoding(str(file_path))
            if violations:
                all_violations[str(file_path)] = violations
    
    if all_violations:
        print("🚨 HARDCODE VIOLATIONS DETECTED! 🚨")
        print("=" * 50)
        for file_path, violations in all_violations.items():
            print(f"\n📁 FILE: {file_path}")
            for violation in violations:
                print(f"  ❌ {violation}")
        
        print("\n" + "=" * 50)
        print("❌ DEPLOYMENT BLOCKED - FIX HARDCODING FIRST!")
        return False
    
    print("✅ NO HARDCODE VIOLATIONS FOUND")
    return True

class ConfigEnforcer:
    """Enforces proper configuration usage patterns"""
    
    @staticmethod
    def require_config_pattern():
        """Template for proper configuration usage"""
        return """
# REQUIRED PATTERN - Use this template:

# 1. Get settings
llm_settings = get_llm_settings()
config_section = llm_settings.get('section_name', {})

# 2. Get configurable values with defaults
parameter = config_section.get('parameter_name', default_value)

# 3. Use the parameter (NO direct hardcoding)
if some_value > parameter:  # ✅ CORRECT
    # do something

# FORBIDDEN PATTERNS:
# if some_value > 0.5:  # ❌ HARDCODED
# max_items = 10        # ❌ HARDCODED  
# keywords = ['a', 'b'] # ❌ HARDCODED
"""

def create_pre_commit_hook():
    """Create pre-commit hook to prevent hardcoded commits"""
    hook_content = '''#!/bin/bash
# Pre-commit hook to prevent hardcoded values

echo "🔍 Checking for hardcoded values..."

python /path/to/ANTI_HARDCODE_SYSTEM.py

if [ $? -ne 0 ]; then
    echo "❌ Commit blocked due to hardcoded values!"
    echo "💡 Fix violations before committing"
    exit 1
fi

echo "✅ No hardcoding detected - commit allowed"
'''

    hook_path = Path(".git/hooks/pre-commit")
    hook_path.write_text(hook_content)
    hook_path.chmod(0o755)
    print("✅ Pre-commit hook installed")

def main():
    """Main enforcement function"""
    directories_to_check = [
        "app/langchain",
        "app/api/v1/endpoints", 
        "app/core",
        "llm-ui/src"
    ]
    
    success = enforce_no_hardcode_rule(directories_to_check)
    
    if not success:
        print("\n🛠️  FIX GUIDE:")
        print(ConfigEnforcer.require_config_pattern())
        sys.exit(1)
    
    print("🎉 All files pass hardcode check!")

if __name__ == "__main__":
    main()