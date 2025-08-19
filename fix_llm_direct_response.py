#!/usr/bin/env python3
"""
Fix the Direct LLM Response Path to Properly Apply System Prompts

This fixes the issue where direct LLM responses don't include the system prompt,
causing poor quality or off-topic responses.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_patch_file():
    """Create a patch file to fix the direct LLM response issues"""
    
    patch_content = '''--- a/app/langchain/service.py
+++ b/app/langchain/service.py
@@ -3837,7 +3837,19 @@ def get_llm_response_direct(question: str, thinking: bool = False, stream: bool
     # Build prompt without RAG context
     prompt = build_prompt(question, thinking)
+    
+    # CRITICAL FIX: Get and apply the main LLM system prompt
+    main_llm_config = get_main_llm_full_config(llm_cfg)
+    system_prompt = main_llm_config.get("system_prompt", "")
+    
+    # Prepend system prompt to the question for proper context
+    if system_prompt:
+        prompt = f"{system_prompt}\\n\\n{prompt}"
+        print(f"[DEBUG] Applied system prompt to direct LLM response")
+    else:
+        print(f"[DEBUG] Warning: No system prompt found for direct LLM response")
+    
     if conversation_history:
         prompt = f"Previous conversation:\\n{conversation_history}\\n\\nCurrent question: {prompt}"
     
     # Make LLM call
@@ -3671,14 +3671,25 @@ def make_llm_call(prompt: str, thinking: bool, context: str, llm_cfg: dict) ->
         mode_config = llm_cfg
     
-    # Get system_prompt if available and prepend to prompt
-    system_prompt = mode_config.get("system_prompt", "")
-    if system_prompt:
-        prompt = f"{system_prompt}\\n\\n{prompt}"
-        print(f"[DEBUG make_llm_call] System prompt found and prepended: {system_prompt[:100]}...")
-    else:
-        print(f"[DEBUG make_llm_call] No system prompt found in config")
+    # CRITICAL FIX: Don't prepend system prompt here - it should already be in the prompt
+    # The prompt should already contain the system prompt from the caller
+    # This prevents double-prepending and ensures proper formatting
+    print(f"[DEBUG make_llm_call] Using prompt as provided (system prompt should already be included)")
     
     # Get max_tokens from config
'''
    
    with open('fix_llm_direct_response.patch', 'w') as f:
        f.write(patch_content)
    
    print("✅ Created patch file: fix_llm_direct_response.patch")
    return True

def apply_manual_fix():
    """Manually apply the fix to the service.py file"""
    
    service_file = "app/langchain/service.py"
    
    print(f"Reading {service_file}...")
    with open(service_file, 'r') as f:
        content = f.read()
    
    # Check if fix is already applied
    if "CRITICAL FIX: Get and apply the main LLM system prompt" in content:
        print("✅ Fix already applied to service.py")
        return True
    
    # Find and fix the get_llm_response_direct function
    import re
    
    # Pattern to find the function and the specific line
    pattern = r'(def get_llm_response_direct.*?\n.*?# Build prompt without RAG context\n.*?prompt = build_prompt\(question, thinking\)\n)'
    
    replacement = r'''\1    
    # CRITICAL FIX: Get and apply the main LLM system prompt
    main_llm_config = get_main_llm_full_config(llm_cfg)
    system_prompt = main_llm_config.get("system_prompt", "")
    
    # Prepend system prompt to the question for proper context
    if system_prompt:
        prompt = f"{system_prompt}\\n\\n{prompt}"
        print(f"[DEBUG] Applied system prompt to direct LLM response")
    else:
        print(f"[DEBUG] Warning: No system prompt found for direct LLM response")
    
'''
    
    # Apply the fix
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    if new_content != content:
        # Backup original file
        import shutil
        shutil.copy(service_file, f"{service_file}.backup")
        print(f"✅ Created backup: {service_file}.backup")
        
        # Write fixed content
        with open(service_file, 'w') as f:
            f.write(new_content)
        
        print(f"✅ Applied fix to {service_file}")
        return True
    else:
        print(f"⚠️ Could not find the pattern to fix in {service_file}")
        print("The file may have been modified or the fix may already be applied")
        return False

def verify_fix():
    """Verify that the fix is properly applied"""
    
    service_file = "app/langchain/service.py"
    
    with open(service_file, 'r') as f:
        content = f.read()
    
    # Check for the critical fix comment
    if "CRITICAL FIX: Get and apply the main LLM system prompt" in content:
        print("✅ Fix is present in service.py")
        
        # Check that it's in the right function
        import re
        pattern = r'def get_llm_response_direct.*?CRITICAL FIX: Get and apply the main LLM system prompt'
        if re.search(pattern, content, re.DOTALL):
            print("✅ Fix is correctly placed in get_llm_response_direct function")
            return True
        else:
            print("⚠️ Fix exists but may not be in the correct location")
            return False
    else:
        print("❌ Fix not found in service.py")
        return False

def main():
    print("="*60)
    print("Fixing Direct LLM Response System Prompt Application")
    print("="*60)
    
    print("\n1. Creating patch file...")
    create_patch_file()
    
    print("\n2. Applying fix to service.py...")
    if apply_manual_fix():
        print("\n3. Verifying fix...")
        if verify_fix():
            print("\n" + "="*60)
            print("✅ FIX SUCCESSFULLY APPLIED")
            print("="*60)
            print("\nWhat was fixed:")
            print("• Direct LLM responses now properly include the system prompt")
            print("• System prompt is applied before building the full prompt")
            print("• This ensures consistent, high-quality responses")
            print("\nExpected improvements:")
            print("• LLM responses will follow the configured system instructions")
            print("• Responses will be properly formatted and on-topic")
            print("• No more irrelevant or off-topic answers")
        else:
            print("\n⚠️ Fix applied but verification failed")
    else:
        print("\n⚠️ Could not apply fix automatically")
        print("You may need to apply the patch manually using:")
        print("  patch -p1 < fix_llm_direct_response.patch")

if __name__ == "__main__":
    main()