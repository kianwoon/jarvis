#!/usr/bin/env python3
"""
Test script to verify radiating system fixes:
1. Model server URL configuration (no hardcoded localhost)
2. Neo4j query property access
3. Docker environment detection
"""

import asyncio
import os
import sys

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_radiating_config():
    """Test that radiating config properly gets model server URL"""
    print("\n=== Testing Radiating Model Configuration ===")
    
    try:
        from app.core.radiating_settings_cache import get_model_config
        
        config = get_model_config()
        print(f"✓ Model config loaded successfully")
        print(f"  - Model: {config.get('model')}")
        print(f"  - Model Server: {config.get('model_server')}")
        print(f"  - Temperature: {config.get('temperature')}")
        print(f"  - Max Tokens: {config.get('max_tokens')}")
        
        # Check that model_server is not hardcoded localhost
        model_server = config.get('model_server', '')
        if 'localhost:11434' in model_server or '127.0.0.1:11434' in model_server:
            print(f"✗ WARNING: Model server still has hardcoded localhost: {model_server}")
            return False
        else:
            print(f"✓ Model server properly configured (no hardcoded localhost)")
            return True
            
    except Exception as e:
        print(f"✗ Failed to test radiating config: {e}")
        return False

async def test_docker_detection():
    """Test Docker environment detection in OllamaLLM"""
    print("\n=== Testing Docker Environment Detection ===")
    
    try:
        from app.llm.ollama import OllamaLLM, LLMConfig
        
        # Check if we're in Docker
        is_docker = os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER")
        print(f"  - Docker environment detected: {is_docker}")
        
        # Test URL conversion
        test_url = "http://localhost:11434"
        config = LLMConfig(
            model_name="test",
            temperature=0.7,
            top_p=0.9,
            max_tokens=100
        )
        
        # Create OllamaLLM instance
        llm = OllamaLLM(config, base_url=test_url)
        
        if is_docker:
            if "host.docker.internal" in llm.base_url:
                print(f"✓ URL properly converted for Docker: {llm.base_url}")
                return True
            else:
                print(f"✗ URL not converted for Docker: {llm.base_url}")
                return False
        else:
            if "localhost" in llm.base_url:
                print(f"✓ URL kept as localhost for non-Docker: {llm.base_url}")
                return True
            else:
                print(f"✗ Unexpected URL change: {llm.base_url}")
                return False
                
    except Exception as e:
        print(f"✗ Failed to test Docker detection: {e}")
        return False

async def test_neo4j_queries():
    """Test that Neo4j queries use proper property access"""
    print("\n=== Testing Neo4j Query Properties ===")
    
    try:
        # Read the file to check the queries
        neo4j_file = "/Users/kianwoonwong/Downloads/jarvis/app/services/radiating/storage/radiating_neo4j_service.py"
        with open(neo4j_file, 'r') as f:
            content = f.read()
        
        # Check for problematic property access
        issues = []
        
        if "r.properties as properties" in content:
            issues.append("Still using 'r.properties' instead of 'properties(r)'")
        
        if "r.strength as strength" in content:
            issues.append("Still using 'r.strength' instead of proper property access")
            
        if "r.bidirectional as bidirectional" in content:
            issues.append("Still using 'r.bidirectional' instead of proper property access")
            
        if "neighbor.properties as properties" in content:
            issues.append("Still using 'neighbor.properties' instead of 'properties(neighbor)'")
        
        if issues:
            print("✗ Found issues in Neo4j queries:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("✓ Neo4j queries use proper property access functions")
            return True
            
    except Exception as e:
        print(f"✗ Failed to test Neo4j queries: {e}")
        return False

async def test_jarvis_llm_integration():
    """Test that JarvisLLM properly integrates with radiating system"""
    print("\n=== Testing JarvisLLM Integration ===")
    
    try:
        from app.llm.ollama import JarvisLLM
        from app.core.radiating_settings_cache import get_model_config
        
        # Get radiating model config
        model_config = get_model_config()
        
        # Try to create JarvisLLM with radiating config
        llm = JarvisLLM(
            model=model_config.get('model', 'llama3.1:8b'),
            mode=model_config.get('llm_mode', 'non-thinking'),
            max_tokens=model_config.get('max_tokens', 4096),
            temperature=model_config.get('temperature', 0.7),
            model_server=model_config['model_server']
        )
        
        print(f"✓ JarvisLLM created successfully with radiating config")
        print(f"  - Base URL: {llm.base_url}")
        
        # Check Docker handling
        is_docker = os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER")
        if is_docker and "localhost" in model_config['model_server']:
            if "host.docker.internal" in llm.base_url:
                print(f"✓ Docker URL conversion working in JarvisLLM")
            else:
                print(f"✗ Docker URL conversion not working in JarvisLLM")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test JarvisLLM integration: {e}")
        return False

async def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Radiating System Fixes")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(await test_radiating_config())
    results.append(await test_docker_detection())
    results.append(await test_neo4j_queries())
    results.append(await test_jarvis_llm_integration())
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! The radiating system fixes are working correctly.")
        return 0
    else:
        print(f"✗ {total - passed} test(s) failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)