#!/usr/bin/env python3
"""
Investigate why vector similarity search returns irrelevant documents
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def test_vector_similarity():
    """Test vector embeddings and similarity"""
    try:
        print("🔍 Investigating Vector Similarity Issue")
        print("=" * 60)
        
        # Test with simplified embedding approach
        import requests
        import numpy as np
        from typing import List
        
        def get_embedding(text: str, endpoint: str = "http://localhost:8012/embeddings") -> List[float]:
            """Get embedding for text"""
            try:
                response = requests.post(
                    endpoint,
                    json={
                        "input": text,
                        "model": "bge-large-en-v1.5"
                    },
                    timeout=10
                )
                if response.status_code == 200:
                    result = response.json()
                    return result["data"][0]["embedding"]
                else:
                    print(f"Embedding request failed: {response.status_code}")
                    return None
            except Exception as e:
                print(f"Embedding error: {e}")
                return None
        
        def cosine_similarity(a: List[float], b: List[float]) -> float:
            """Calculate cosine similarity between two vectors"""
            a = np.array(a)
            b = np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        # Test texts
        query = "partnership between beyondsoft and tencent"
        
        test_texts = {
            "Tencent Partnership": "Beyondsoft's Partnership with Tencent started in 2012 with cloud infrastructure support",
            "n8n AI Agent": "n8n ai agent messages workflow automation platform for connecting different services",
            "Alibaba Partnership": "Beyondsoft has strategic partnership with Alibaba Cloud since 2015",
            "Random Content": "Weather patterns and climate change environmental topics sustainability"
        }
        
        print(f"Query: '{query}'")
        print()
        
        # Get query embedding
        query_embedding = get_embedding(query)
        if not query_embedding:
            print("❌ Failed to get query embedding")
            return
        
        print(f"Query embedding vector length: {len(query_embedding)}")
        print()
        
        # Test similarity with each document
        similarities = []
        
        for doc_name, content in test_texts.items():
            print(f"📄 Testing: {doc_name}")
            print(f"Content: {content}")
            
            # Get document embedding
            doc_embedding = get_embedding(content)
            if doc_embedding:
                # Calculate similarity
                similarity = cosine_similarity(query_embedding, doc_embedding)
                similarities.append((doc_name, similarity))
                
                print(f"Vector Similarity: {similarity:.6f}")
                print(f"Distance (1-similarity): {1-similarity:.6f}")
            else:
                print("❌ Failed to get document embedding")
            
            print("-" * 40)
        
        # Show rankings
        print("\n🏆 Vector Similarity Rankings:")
        similarities.sort(key=lambda x: x[1], reverse=True)
        for i, (doc_name, similarity) in enumerate(similarities):
            distance = 1 - similarity
            print(f"{i+1}. {doc_name}: similarity={similarity:.6f}, distance={distance:.6f}")
        
        # Check if high similarity explains the issue
        print(f"\n💡 Analysis:")
        if len(similarities) >= 2:
            top_sim = similarities[0][1]
            second_sim = similarities[1][1]
            print(f"Top similarity: {top_sim:.6f}")
            print(f"Second similarity: {second_sim:.6f}")
            print(f"Difference: {top_sim - second_sim:.6f}")
            
            if abs(top_sim - second_sim) < 0.1:
                print("⚠️  Small difference between top similarities - vector embeddings may be too similar")
            else:
                print("✅ Clear distinction between similarities")
                
    except Exception as e:
        print(f"❌ Investigation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vector_similarity()