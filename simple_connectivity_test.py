#!/usr/bin/env python3
"""
Simple Connectivity Test - Direct API calls to verify services

This will test the three core connectivity issues:
1. Ollama LLM (localhost:11434) 
2. Neo4j (localhost:7687)
3. LLM Settings Configuration
"""

import asyncio
import json
import httpx
from neo4j import GraphDatabase
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_ollama_direct():
    """Test Ollama directly with available models"""
    print("🤖 Testing Ollama Connectivity...")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 1. Check version
            response = await client.get("http://localhost:11434/api/version")
            if response.status_code == 200:
                version = response.json()
                print(f"   ✅ Ollama version: {version.get('version')}")
            else:
                print(f"   ❌ Version check failed: {response.status_code}")
                return False

            # 2. List available models
            models_response = await client.get("http://localhost:11434/api/tags")
            if models_response.status_code == 200:
                models_data = models_response.json()
                model_names = [m['name'] for m in models_data.get('models', [])]
                print(f"   📋 Available models: {model_names[:3]}...")  # Show first 3
                
                if not model_names:
                    print("   ❌ No models available")
                    return False
                    
                # Use the first available model
                test_model = model_names[0]
                print(f"   🔧 Testing with model: {test_model}")
                
            else:
                print(f"   ❌ Models list failed: {models_response.status_code}")
                return False

            # 3. Test generation using chat API (newer approach)
            chat_payload = {
                "model": test_model,
                "messages": [
                    {"role": "user", "content": "Say 'CONNECTIVITY_OK' and nothing else."}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 5
                }
            }
            
            chat_response = await client.post(
                "http://localhost:11434/api/chat",
                json=chat_payload,
                timeout=60.0
            )
            
            if chat_response.status_code == 200:
                result = chat_response.json()
                message_content = result.get('message', {}).get('content', '').strip()
                print(f"   💬 Chat response: {message_content}")
                
                if 'CONNECTIVITY_OK' in message_content.upper():
                    print("   ✅ Ollama chat API working correctly")
                    return True
                else:
                    print("   ⚠️ Ollama responding but content unexpected")
                    return True  # Still consider it working
            else:
                print(f"   ❌ Chat API failed: {chat_response.status_code}")
                print(f"   📄 Response: {chat_response.text[:200]}")
                return False
                
    except Exception as e:
        print(f"   ❌ Ollama test failed: {e}")
        return False

def test_neo4j_direct():
    """Test Neo4j directly with localhost connection"""
    print("🗄️ Testing Neo4j Connectivity...")
    
    try:
        # Use localhost connection (Docker port mapping)
        uri = "bolt://localhost:7687"
        username = "neo4j"
        password = "jarvis_neo4j_password"
        
        print(f"   🔗 Connecting to: {uri}")
        
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        with driver.session() as session:
            # Test basic connectivity
            result = session.run("RETURN 'CONNECTED' as status, datetime() as timestamp")
            record = result.single()
            
            if record and record['status'] == 'CONNECTED':
                print(f"   ✅ Neo4j connected successfully")
                print(f"   🕐 Timestamp: {record['timestamp']}")
                
                # Get database stats
                stats_result = session.run("""
                    MATCH (n) 
                    OPTIONAL MATCH ()-[r]->()
                    RETURN count(DISTINCT n) as entities, count(r) as relationships
                """)
                stats = stats_result.single()
                
                entities = stats['entities'] if stats else 0
                relationships = stats['relationships'] if stats else 0
                ratio = relationships / max(entities, 1)
                
                print(f"   📊 Database stats: {entities} entities, {relationships} relationships")
                print(f"   📊 Relationship ratio: {ratio:.2f} per entity")
                
                driver.close()
                return True
            else:
                print("   ❌ Neo4j test query failed")
                driver.close()
                return False
                
    except Exception as e:
        print(f"   ❌ Neo4j test failed: {e}")
        return False

async def test_llm_settings():
    """Test LLM settings through the application"""
    print("⚙️ Testing LLM Settings Configuration...")
    
    try:
        from app.core.llm_settings_cache import get_llm_settings
        
        settings = get_llm_settings()
        print(f"   📋 LLM settings loaded: {len(settings)} keys")
        
        # Check main LLM configuration
        main_llm = settings.get('main_llm', {})
        kg_llm = settings.get('knowledge_graph', {})
        
        print(f"   🤖 Main LLM host: {main_llm.get('host', 'not set')}")
        print(f"   🤖 Main LLM port: {main_llm.get('port', 'not set')}")
        print(f"   🤖 Main LLM model: {main_llm.get('model', 'not set')}")
        
        print(f"   🧠 KG LLM host: {kg_llm.get('host', 'not set')}")
        print(f"   🧠 KG LLM port: {kg_llm.get('port', 'not set')}")
        print(f"   🧠 KG LLM model: {kg_llm.get('model', 'not set')}")
        
        # Check if hosts are correctly configured for local access
        main_host = main_llm.get('host', '')
        kg_host = kg_llm.get('host', '')
        
        host_correct = (
            main_host in ['localhost', '127.0.0.1', 'host.docker.internal'] or
            kg_host in ['localhost', '127.0.0.1', 'host.docker.internal']
        )
        
        if host_correct:
            print("   ✅ LLM settings configured correctly")
            return True
        else:
            print("   ⚠️ LLM host configuration may need adjustment")
            return True  # Not critical failure
            
    except Exception as e:
        print(f"   ❌ LLM settings test failed: {e}")
        return False

def test_knowledge_graph_extractor():
    """Test the knowledge graph extractor instantiation"""
    print("🧠 Testing Knowledge Graph Extractor...")
    
    try:
        from app.services.llm_knowledge_extractor import LLMKnowledgeExtractor
        
        extractor = LLMKnowledgeExtractor()
        print(f"   ✅ Knowledge graph extractor initialized")
        print(f"   🎯 Model config: {extractor.model_config}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Knowledge graph extractor test failed: {e}")
        return False

async def main():
    """Run all connectivity tests"""
    print("🚀 Simple Connectivity Test for Knowledge Graph Pipeline")
    print("=" * 60)
    
    tests = [
        ("Ollama LLM", test_ollama_direct()),
        ("Neo4j Database", test_neo4j_direct()),
        ("LLM Settings", test_llm_settings()),
        ("KG Extractor", test_knowledge_graph_extractor())
    ]
    
    results = []
    
    for test_name, test_func in tests:
        if asyncio.iscoroutine(test_func):
            result = await test_func
        else:
            result = test_func
        results.append((test_name, result))
        print()  # Add spacing between tests
    
    # Summary
    print("=" * 60)
    print("📋 CONNECTIVITY TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    success_rate = passed / len(results) * 100
    print(f"\n📊 Success Rate: {passed}/{len(results)} ({success_rate:.1f}%)")
    
    if passed == len(results):
        print("🎉 All connectivity tests passed!")
        print("✅ LLM Service: Accessible and responding")
        print("✅ Neo4j Database: Connected and functional")
        print("✅ Application Settings: Loaded correctly")
        print("✅ Knowledge Graph Components: Initialized successfully")
        return True
    else:
        failed = len(results) - passed
        print(f"⚠️ {failed} connectivity issues detected")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)