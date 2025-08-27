"""
Test the new SIMPLE RAG implementation
"""
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from app.api.v1.endpoints.langchain import rag_endpoint, RAGRequest

async def test_simple_rag():
    """Test the brain-dead simple RAG implementation"""
    print("🧪 Testing Simple RAG Implementation...")
    
    # Mock the MCP tool call to return fake documents
    mock_documents = [
        {
            "content": "The sky is blue because of Rayleigh scattering.",
            "source": "physics_doc.pdf",
            "score": 0.95
        },
        {
            "content": "Blue light has shorter wavelengths than red light.",
            "source": "optics_guide.txt", 
            "score": 0.87
        }
    ]
    
    mock_rag_result = {
        "result": {
            "documents": mock_documents
        }
    }
    
    # Mock LLM response
    class MockLLMResponse:
        def __init__(self, text):
            self.text = text
    
    mock_llm_stream = [
        MockLLMResponse("The sky appears blue "),
        MockLLMResponse("due to a phenomenon called "),
        MockLLMResponse("Rayleigh scattering.")
    ]
    
    # Test the new simple RAG endpoint
    request = RAGRequest(
        question="Why is the sky blue?",
        conversation_id="test-123"
    )
    
    with patch('app.langchain.service.call_mcp_tool', return_value=mock_rag_result), \
         patch('app.core.simple_conversation_manager.conversation_manager.add_message', new_callable=AsyncMock), \
         patch('app.llm.ollama.OllamaLLM') as mock_llm_class:
        
        # Setup mock LLM
        mock_llm = Mock()
        mock_llm.generate_stream = AsyncMock(return_value=iter(mock_llm_stream))
        mock_llm_class.return_value = mock_llm
        
        # Call the endpoint
        response = rag_endpoint(request)
        
        # Collect all streamed chunks
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode('utf-8'))
        
        print(f"✅ Got {len(chunks)} response chunks")
        
        # Verify we got the expected response structure
        found_status = False
        found_tokens = False
        found_final_answer = False
        
        for chunk in chunks:
            try:
                data = json.loads(chunk.strip())
                if data.get("type") == "status":
                    found_status = True
                    print(f"📊 Status: {data.get('message')}")
                elif "token" in data:
                    found_tokens = True
                    print(f"🔤 Token: {data.get('token')}")
                elif "answer" in data:
                    found_final_answer = True
                    print(f"✨ Final answer: {data.get('answer')}")
                    print(f"📄 Documents: {len(data.get('documents', []))}")
            except json.JSONDecodeError:
                continue
        
        # Verify all expected parts were found
        assert found_status, "❌ No status messages found"
        assert found_tokens, "❌ No token streaming found"  
        assert found_final_answer, "❌ No final answer found"
        
        print("✅ Simple RAG test passed!")
        print("✅ All response components present: status, tokens, final answer")
        return True

if __name__ == "__main__":
    asyncio.run(test_simple_rag())