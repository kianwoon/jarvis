#!/usr/bin/env python3
"""
Example: How to use the radiating routing in the langchain endpoint

This example shows how to make API calls to use the radiating coverage system
through the langchain endpoint.
"""

import requests
import json
import sys
from typing import Optional, Dict, Any

def call_rag_with_radiating(
    question: str,
    max_depth: int = 3,
    strategy: str = "hybrid",
    filters: Optional[Dict[str, Any]] = None,
    api_url: str = "http://localhost:8000"
) -> None:
    """
    Call the RAG endpoint with radiating coverage enabled
    
    Args:
        question: The query to process
        max_depth: Maximum traversal depth for radiating exploration
        strategy: Radiating strategy ('hybrid', 'breadth_first', 'depth_first')
        filters: Optional filters for entity/relationship types
        api_url: Base URL of the API server
    """
    
    # Prepare the request payload
    payload = {
        "question": question,
        "use_radiating": True,  # Enable radiating mode
        "radiating_config": {
            "max_depth": max_depth,
            "strategy": strategy,
            "filters": filters or {},
            "include_coverage_data": True
        },
        "conversation_id": "example_session_001"
    }
    
    print(f"🌟 Sending radiating query: {question}")
    print(f"   Configuration: depth={max_depth}, strategy={strategy}")
    print("-" * 60)
    
    try:
        # Make the API call
        response = requests.post(
            f"{api_url}/api/v1/langchain/rag",
            json=payload,
            stream=True,  # Enable streaming
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            print(f"❌ Error: API returned status {response.status_code}")
            print(f"   Response: {response.text}")
            return
        
        # Process the streaming response
        entities_found = 0
        relationships_found = 0
        final_answer = ""
        
        for line in response.iter_lines():
            if line:
                try:
                    # Parse the JSON response
                    data = json.loads(line.decode('utf-8'))
                    
                    # Handle different event types
                    if data.get("type") == "chat_start":
                        print(f"✅ Radiating mode activated")
                        config = data.get("config", {})
                        print(f"   Config: {config}")
                        
                    elif data.get("type") == "status":
                        print(f"📊 {data.get('message', '')}")
                        
                    elif data.get("type") == "entity_discovered":
                        entities_found += 1
                        entity = data.get("entity", {})
                        print(f"   🔍 Entity: {entity.get('name', 'Unknown')} (type: {entity.get('type', 'Unknown')})")
                        
                    elif data.get("type") == "relationship_found":
                        relationships_found += 1
                        rel = data.get("relationship", {})
                        print(f"   🔗 Relationship: {rel}")
                        
                    elif data.get("type") == "traversal_progress":
                        current = data.get("current_depth", 0)
                        explored = data.get("entities_explored", 0)
                        total = data.get("total_entities", 0)
                        print(f"   📈 Progress: Depth {current}, Explored {explored}/{total} entities")
                        
                    elif data.get("token"):
                        # Accumulate text tokens
                        final_answer += data.get("token", "")
                        sys.stdout.write(data.get("token", ""))
                        sys.stdout.flush()
                        
                    elif data.get("answer"):
                        # Final complete answer
                        if not final_answer:  # Only if we haven't been streaming tokens
                            final_answer = data.get("answer", "")
                            print(f"\n\n📝 Answer: {final_answer}")
                        
                        # Show statistics
                        print(f"\n\n📊 Statistics:")
                        print(f"   - Entities found: {data.get('entities_found', entities_found)}")
                        print(f"   - Relationships found: {data.get('relationships_found', relationships_found)}")
                        
                        # Show coverage data if available
                        if data.get("coverage"):
                            coverage = data.get("coverage", {})
                            print(f"\n📍 Coverage Data:")
                            print(f"   - Total nodes: {coverage.get('total_nodes', 0)}")
                            print(f"   - Nodes explored: {coverage.get('nodes_explored', 0)}")
                            print(f"   - Coverage ratio: {coverage.get('coverage_ratio', 0):.2%}")
                        
                except json.JSONDecodeError:
                    # Skip malformed JSON lines
                    continue
        
        print("\n" + "=" * 60)
        print("✅ Radiating query completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API server")
        print("   Make sure the server is running with: ./run_local.sh")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main function with example queries"""
    
    print("=" * 60)
    print("Radiating Coverage Routing Examples")
    print("=" * 60)
    
    # Example 1: Simple radiating query
    print("\n📌 Example 1: Basic Radiating Query")
    call_rag_with_radiating(
        question="What are the connections between artificial intelligence and healthcare?",
        max_depth=2,
        strategy="hybrid"
    )
    
    print("\n" + "=" * 60)
    
    # Example 2: Filtered radiating query
    print("\n📌 Example 2: Filtered Radiating Query")
    call_rag_with_radiating(
        question="Explore the technology landscape around machine learning",
        max_depth=3,
        strategy="breadth_first",
        filters={
            "entity_types": ["Technology", "Company", "Research"],
            "relationship_types": ["uses", "develops", "competes_with"]
        }
    )
    
    print("\n" + "=" * 60)
    
    # Example 3: Comparison with normal mode
    print("\n📌 Example 3: Comparison - Normal vs Radiating")
    
    print("\n🔹 Normal Mode (without radiating):")
    normal_payload = {
        "question": "Tell me about quantum computing",
        "use_radiating": False
    }
    
    print("   This would go through query classification and normal RAG/LLM routing")
    
    print("\n🔹 Radiating Mode:")
    print("   Same query with radiating would explore the knowledge graph")
    print("   discovering related entities, relationships, and hidden connections")
    
    print("\n" + "=" * 60)
    print("📚 Usage Notes:")
    print("1. use_radiating=True bypasses query classification")
    print("2. Similar to @agent detection, it has its own dedicated handler")
    print("3. Returns streaming responses with entity/relationship discoveries")
    print("4. Maintains backward compatibility - existing code continues to work")
    print("5. Can be combined with collections and conversation history")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python use_radiating_routing.py")
        print("\nThis script demonstrates how to use the radiating routing feature")
        print("in the langchain endpoint. Make sure the API server is running first.")
    else:
        main()