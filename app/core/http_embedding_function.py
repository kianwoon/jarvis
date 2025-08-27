"""
HTTP Embedding Function for external embedding services
"""
import requests
import json
from typing import List
from fastapi import HTTPException


class HTTPEmbeddingFunction:
    """Custom embedding function for HTTP endpoint compatibility."""
    
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        # Use the provided endpoint instead of hardcoded URL
        if not self.endpoint:
            raise ValueError("Embedding endpoint must be provided - no hardcoding allowed")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            # Use lowercase for consistent embeddings
            normalized_text = text.lower().strip()
            payload = {"texts": [normalized_text]}
            
            print(f"[DEBUG] HTTPEmbeddingFunction: Sending embedding request for text length: {len(normalized_text)}")
            
            try:
                # Add timeout to prevent hanging - USE CONFIGURED ENDPOINT
                resp = requests.post(
                    self.endpoint, 
                    json=payload,
                    timeout=30  # 30 second timeout
                )
                print(f"[DEBUG] HTTPEmbeddingFunction: Received response with status {resp.status_code}")
                
                if resp.status_code == 422:
                    print("Response content:", resp.content)
                resp.raise_for_status()
                
                # Parse JSON response with error handling for KeyError 'embeddings'
                try:
                    response_json = resp.json()
                    print(f"[DEBUG] HTTPEmbeddingFunction: Response JSON keys: {list(response_json.keys())}")
                    
                    if "embeddings" not in response_json:
                        print(f"[ERROR] HTTPEmbeddingFunction: KeyError - 'embeddings' key not found in response")
                        print(f"[ERROR] HTTPEmbeddingFunction: Full response: {response_json}")
                        raise KeyError("'embeddings' key not found in embedding service response")
                    
                    if not isinstance(response_json["embeddings"], list) or len(response_json["embeddings"]) == 0:
                        print(f"[ERROR] HTTPEmbeddingFunction: Embeddings list is empty or invalid")
                        print(f"[ERROR] HTTPEmbeddingFunction: Embeddings value: {response_json['embeddings']}")
                        raise ValueError("Embeddings list is empty or invalid")
                    
                    embedding_data = response_json["embeddings"][0]
                    embeddings.append(embedding_data)
                    print(f"[DEBUG] HTTPEmbeddingFunction: Successfully got embedding with dimension {len(embedding_data)}")
                    
                except KeyError as ke:
                    print(f"[ERROR] HTTPEmbeddingFunction: KeyError accessing response - {str(ke)}")
                    print(f"[ERROR] HTTPEmbeddingFunction: Response content: {resp.content}")
                    raise HTTPException(status_code=500, detail=f"Invalid embedding service response format: {str(ke)}")
                except json.JSONDecodeError as je:
                    print(f"[ERROR] HTTPEmbeddingFunction: JSON decode error - {str(je)}")
                    print(f"[ERROR] HTTPEmbeddingFunction: Raw response: {resp.content}")
                    raise HTTPException(status_code=500, detail=f"Invalid JSON response from embedding service: {str(je)}")
                
            except requests.exceptions.Timeout as e:
                print(f"[ERROR] HTTPEmbeddingFunction: Timeout after 30 seconds for text: {normalized_text[:50]}...")
                raise HTTPException(status_code=504, detail=f"Embedding service timeout: {str(e)}")
            except requests.exceptions.ConnectionError as e:
                print(f"[ERROR] HTTPEmbeddingFunction: Connection error to qwen-embedder service: {str(e)}")
                raise HTTPException(status_code=503, detail=f"Embedding service unavailable: {str(e)}")
            except Exception as e:
                print(f"[ERROR] HTTPEmbeddingFunction: Unexpected error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")
                
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        # Also normalize queries to lowercase
        return self.embed_documents([text])[0]