#!/usr/bin/env python3
"""
Direct test of MCP endpoint to bypass import issues
"""
import requests
import json

def test_mcp_endpoint():
    """Test the MCP endpoint directly"""
    try:
        print("🔍 Testing MCP rag_knowledge_search endpoint directly")
        print("=" * 60)
        
        # Test the MCP endpoint directly
        url = "http://localhost:8000/mcp/call"
        
        payload = {
            "method": "rag_knowledge_search",
            "params": {
                "query": "search internal knowledge base. relationship between beyondsoft and tencent in details"
            }
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        print(f"Making request to: {url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        print(f"\nResponse status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            
            # Analyze the response
            if 'result' in result:
                success = result['result'].get('success', False)
                query = result['result'].get('query', '')
                response_text = result['result'].get('response', '')
                sources = result['result'].get('sources', [])
                
                print(f"\n📊 Analysis:")
                print(f"  Success: {success}")
                print(f"  Query: {query}")
                print(f"  Response length: {len(response_text)} chars")
                print(f"  Sources count: {len(sources)}")
                
                if response_text:
                    print(f"\n📄 Response preview (first 500 chars):")
                    print("-" * 50)
                    print(response_text[:500])
                    print("-" * 50)
                    
                    # Check for keywords
                    response_lower = response_text.lower()
                    tencent_count = response_lower.count('tencent')
                    beyondsoft_count = response_lower.count('beyondsoft')
                    partnership_count = response_lower.count('partnership')
                    
                    print(f"\n🔍 Keyword Analysis:")
                    print(f"  Tencent mentions: {tencent_count}")
                    print(f"  Beyondsoft mentions: {beyondsoft_count}")
                    print(f"  Partnership mentions: {partnership_count}")
                
                if sources:
                    print(f"\n📚 Sources:")
                    for i, source in enumerate(sources[:3]):
                        print(f"  {i+1}. {source}")
        else:
            print(f"Request failed: {response.text}")
            
    except requests.exceptions.ConnectoinError:
        print("❌ Cannot connect to localhost:8000 - is the server running?")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_database_direct():
    """Test database connection directly"""
    try:
        print(f"\n🗄️  Testing Database Direct Access")
        print("=" * 60)
        
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))
        
        from app.core.db import SessionLocal
        from sqlalchemy import text
        
        db = SessionLocal()
        try:
            # First, check table structure
            query_desc = text("SELECT column_name FROM information_schema.columns WHERE table_name = 'collection_registry'")
            columns = db.execute(query_desc).fetchall()
            print(f"📋 Collection Registry Columns: {[col.column_name for col in columns]}")
            
            # Check collection registry
            query = text("SELECT * FROM collection_registry LIMIT 10")
            results = db.execute(query).fetchall()
            
            print(f"\n📊 Collection Registry ({len(results)} collections):")
            for row in results:
                # Print available attributes dynamically
                row_dict = dict(row._mapping) if hasattr(row, '_mapping') else {}
                print(f"  - Row: {row_dict}")
                
            # Check for partnership collection - try different column names
            possible_name_columns = ['name', 'collection_name', 'title']
            for col_name in possible_name_columns:
                try:
                    query2 = text(f"SELECT * FROM collection_registry WHERE {col_name} = 'partnership'")
                    partnership = db.execute(query2).fetchone()
                    if partnership:
                        print(f"\n✅ Partnership collection found in column '{col_name}':")
                        partnership_dict = dict(partnership._mapping) if hasattr(partnership, '_mapping') else {}
                        print(f"  Data: {partnership_dict}")
                        break
                except Exception as e:
                    print(f"  Column '{col_name}' not found: {e}")
            else:
                print(f"\n❌ Partnership collection not found in any column")
                
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Database error: {e}")

if __name__ == "__main__":
    test_mcp_endpoint()
    test_database_direct()
    print(f"\n✨ Direct testing completed")