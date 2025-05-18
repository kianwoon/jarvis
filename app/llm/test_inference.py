import asyncio
from app.llm import QwenInference, LLMConfig

async def test_basic_inference():
    """Test basic text generation with Qwen."""
    print("Initializing Qwen model...")
    config = LLMConfig(
        model_name="Qwen/Qwen-7B-Chat",
        temperature=0.7,
        max_tokens=100
    )
    
    inference = QwenInference(config)
    
    # Test prompt
    prompt = "Write a short poem about artificial intelligence."
    
    print(f"\nGenerating response for prompt: {prompt}")
    try:
        response = await inference.generate(prompt)
        print("\nGenerated response:")
        print("-" * 50)
        print(response.text)
        print("-" * 50)
        print("\nMetadata:", response.metadata)
    except Exception as e:
        print(f"Error during generation: {str(e)}")

async def test_streaming_inference():
    """Test streaming text generation with Qwen."""
    print("\nTesting streaming generation...")
    config = LLMConfig(
        model_name="Qwen/Qwen-7B-Chat",
        temperature=0.7,
        max_tokens=100
    )
    
    inference = QwenInference(config)
    
    # Test prompt
    prompt = "Explain quantum computing in simple terms."
    
    print(f"\nStreaming response for prompt: {prompt}")
    try:
        print("\nStreaming response:")
        print("-" * 50)
        async for chunk in inference.generate_stream(prompt):
            print(chunk.text, end="", flush=True)
        print("\n" + "-" * 50)
    except Exception as e:
        print(f"Error during streaming: {str(e)}")

async def main():
    """Run all tests."""
    print("Starting Qwen inference tests...")
    
    # Test basic inference
    await test_basic_inference()
    
    # Test streaming inference
    await test_streaming_inference()

if __name__ == "__main__":
    asyncio.run(main()) 