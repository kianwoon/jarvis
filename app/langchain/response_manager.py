"""
Response management utilities for handling long agent outputs
"""
from typing import Dict, List, Optional, AsyncGenerator
import hashlib
import json

class ResponseManager:
    """Manages response chunking and continuation for long outputs"""
    
    def __init__(self, chunk_size: int = 3000):
        self.chunk_size = chunk_size
        self.response_cache: Dict[str, Dict] = {}
    
    def should_chunk_response(self, response: str, agent_config: Dict) -> bool:
        """Determine if response needs chunking based on agent config"""
        response_mode = agent_config.get("response_mode", "complete")
        
        if response_mode == "chunked":
            return True
        
        # Auto-detect if response is too long
        max_tokens = agent_config.get("max_tokens", 4000)
        # Rough estimate: 1 token ≈ 4 characters
        estimated_tokens = len(response) / 4
        
        return estimated_tokens > max_tokens * 0.9  # Chunk if > 90% of max
    
    def chunk_response(self, response: str, chunk_size: Optional[int] = None) -> List[str]:
        """Split response into chunks at natural boundaries"""
        chunk_size = chunk_size or self.chunk_size
        
        if len(response) <= chunk_size:
            return [response]
        
        chunks = []
        current_chunk = ""
        
        # Try to split at natural boundaries
        paragraphs = response.split('\n\n')
        
        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 > chunk_size and current_chunk:
                # Save current chunk
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n"
                current_chunk += para
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def create_continuation_prompt(self, original_query: str, previous_response: str, continuation_request: str = "continue") -> str:
        """Create a prompt for continuing a truncated response"""
        return f"""Previous query: {original_query}

My previous response was cut off. Here's what I said so far:
{previous_response[-500:]}  # Last 500 chars for context

Please continue from where I left off. {continuation_request}"""
    
    def store_response_state(self, agent_name: str, query_id: str, response: str, metadata: Dict):
        """Store response state for potential continuation"""
        self.response_cache[f"{agent_name}:{query_id}"] = {
            "response": response,
            "metadata": metadata,
            "timestamp": json.dumps(metadata.get("timestamp", ""))
        }
    
    def get_response_state(self, agent_name: str, query_id: str) -> Optional[Dict]:
        """Retrieve stored response state"""
        return self.response_cache.get(f"{agent_name}:{query_id}")
    
    def generate_query_id(self, query: str) -> str:
        """Generate a unique ID for a query"""
        return hashlib.md5(query.encode()).hexdigest()[:8]
    
    async def stream_chunked_response(
        self, 
        response: str, 
        agent_name: str,
        chunk_size: Optional[int] = None
    ) -> AsyncGenerator[Dict, None]:
        """Stream response in chunks with proper formatting"""
        chunks = self.chunk_response(response, chunk_size)
        
        for i, chunk in enumerate(chunks):
            is_final = i == len(chunks) - 1
            
            yield {
                "type": "agent_response_chunk",
                "agent": agent_name,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "content": chunk,
                "is_final": is_final
            }

# Global instance
response_manager = ResponseManager()