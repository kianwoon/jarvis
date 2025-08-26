"""
Startup initialization for the application
"""

import asyncio
from typing import Optional


async def initialize_models():
    """Initialize ML models at startup to avoid delays during requests"""
    
    from app.core.reranker_config import RerankerConfig
    import os
    
    # Set environment variables for local-only mode to prevent HuggingFace cloud calls
    if RerankerConfig.force_local_only():
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        print("[Startup] Configured reranker for local-only mode")
    
    # Initialize Qwen3-Reranker-4B if configured
    if RerankerConfig.should_preload():
        try:
            print("[Startup] Pre-initializing Qwen3-Reranker-4B...")
            from app.rag.qwen_reranker import get_qwen_reranker
            
            # Run in executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            device = RerankerConfig.get_device()
            await loop.run_in_executor(None, lambda: get_qwen_reranker(device=device))
            
            print("[Startup] Qwen3-Reranker-4B initialized successfully")
        except Exception as e:
            print(f"[Startup] Failed to initialize Qwen3-Reranker-4B: {e}")
            print("[Startup] RAG will fall back to LLM-based reranking")
    else:
        print("[Startup] Qwen3-Reranker-4B preloading is disabled")
        if RerankerConfig.is_enabled():
            print("[Startup] Reranker will be loaded on first use")
        else:
            print("[Startup] Reranker is disabled in this environment")


async def initialize_default_collection():
    """Initialize the default knowledge collection at startup"""
    try:
        print("[Startup] Initializing default collection...")
        from app.core.collection_initializer import initialize_default_collection_async
        
        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: initialize_default_collection_async())
        
        result_data = await result if asyncio.iscoroutine(result) else result
        
        if result_data["success"]:
            if result_data["already_exists"]:
                print("[Startup] Default collection 'default_knowledge' already exists")
            else:
                print(f"[Startup] {result_data['message']}")
                print(f"[Startup] - Database created: {result_data['database_created']}")
                print(f"[Startup] - Milvus created: {result_data['milvus_created']}")
                print(f"[Startup] - Milvus available: {result_data['milvus_available']}")
        else:
            print(f"[Startup] Failed to initialize default collection: {result_data.get('error', 'Unknown error')}")
            print("[Startup] Collection initialization will be available via API endpoint")
            
    except Exception as e:
        print(f"[Startup] Error during collection initialization: {e}")
        print("[Startup] Collection initialization will be available via API endpoint")


async def register_internal_mcp_tools():
    """Register internal MCP tools at startup"""
    try:
        print("[Startup] Registering internal MCP tools...")
        from app.mcp_services.rag_tool_registration import register_rag_mcp_tool
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, register_rag_mcp_tool)
        
        print("[Startup] Internal MCP tools registered successfully")
    except Exception as e:
        print(f"[Startup] Failed to register internal MCP tools: {e}")

async def startup_tasks():
    """Run all startup tasks"""
    await initialize_models()
    await initialize_default_collection()
    await register_internal_mcp_tools()
    print("[Startup] All startup tasks completed")


def run_startup_sync():
    """Run startup tasks synchronously (for non-async contexts)"""
    try:
        asyncio.run(startup_tasks())
    except RuntimeError:
        # If already in an event loop, create a task
        asyncio.create_task(startup_tasks())