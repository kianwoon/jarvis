"""
Startup initialization for the application
"""

import asyncio
from typing import Optional


async def initialize_models():
    """Initialize ML models at startup to avoid delays during requests"""
    
    from app.core.reranker_config import RerankerConfig
    
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


async def startup_tasks():
    """Run all startup tasks"""
    await initialize_models()
    print("[Startup] All startup tasks completed")


def run_startup_sync():
    """Run startup tasks synchronously (for non-async contexts)"""
    try:
        asyncio.run(startup_tasks())
    except RuntimeError:
        # If already in an event loop, create a task
        asyncio.create_task(startup_tasks())