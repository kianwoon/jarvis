#!/usr/bin/env python3
"""
Verify Docker app can load Qwen3-Reranker-0.6B after rebuild
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def verify_docker_reranker():
    """Verify reranker works in Docker environment"""
    try:
        print("🐳 Verifying Docker Qwen3-Reranker-0.6B Loading")
        print("=" * 60)
        
        # Check environment
        print("📋 Environment Check:")
        print(f"   Python version: {sys.version}")
        
        # Check transformers version
        try:
            import transformers
            print(f"   Transformers version: {transformers.__version__}")
            
            from packaging import version
            if version.parse(transformers.__version__) >= version.parse("4.51.0"):
                print("   ✅ Transformers version supports Qwen3")
            else:
                print("   ❌ Transformers version may not support Qwen3")
        except ImportError as e:
            print(f"   ❌ Transformers import failed: {e}")
            return
        
        # Check if we're in Docker
        is_docker = os.path.exists('/root') or os.environ.get('DOCKER_ENVIRONMENT')
        print(f"   Docker environment: {is_docker}")
        
        # Check model files
        print("\n📁 Model Files Check:")
        model_paths = [
            "/root/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/6e9e69830b95c52b5fd889b7690dda3329508de3",
            "/root/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B"
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                print(f"   ✅ {path}")
                try:
                    files = os.listdir(path)
                    print(f"      Files: {files[:5]}...")  # Show first 5 files
                except:
                    pass
            else:
                print(f"   ❌ {path}")
        
        print("\n🔄 Testing Reranker Loading...")
        
        # Clear any cached instance
        from app.rag import qwen_reranker
        qwen_reranker._reranker_instance = None
        qwen_reranker._init_lock = False
        
        from app.rag.qwen_reranker import get_qwen_reranker
        
        print("   Attempting to load reranker...")
        reranker = get_qwen_reranker()
        
        if reranker:
            print("   ✅ Reranker loaded successfully!")
            print(f"      Device: {reranker.device}")
            print(f"      Using CausalLM: {reranker.use_causal_lm}")
            
            # Test basic functionality
            print("\n🔍 Testing Reranking Functionality...")
            
            class MockDoc:
                def __init__(self, content):
                    self.page_content = content
            
            test_docs = [
                (MockDoc("Beyondsoft partnership with Tencent Cloud infrastructure"), 0.8),
                (MockDoc("Weather forecast for tomorrow"), 0.6),
                (MockDoc("Alibaba Cloud services and solutions"), 0.7),
                (MockDoc("Tencent Beyondsoft collaboration details"), 0.75)
            ]
            
            query = "beyondsoft tencent partnership"
            
            try:
                results = reranker.rerank(
                    query=query,
                    documents=test_docs,
                    top_k=2
                )
                
                print(f"   ✅ Reranking completed! Got {len(results)} results")
                print("   📊 Top Results:")
                
                for i, result in enumerate(results):
                    content_preview = result.document.page_content[:50] + "..."
                    print(f"      {i+1}. Score: {result.score:.4f} - {content_preview}")
                
                # Check if relevant docs scored higher
                if len(results) >= 2:
                    top_content = results[0].document.page_content.lower()
                    if any(word in top_content for word in ['beyondsoft', 'tencent', 'partnership']):
                        print("   ✅ Reranking prioritized relevant content correctly")
                    else:
                        print("   ⚠️  Reranking results may need tuning")
                
                print("\n🎉 SUCCESS: Docker Qwen3-Reranker-0.6B is working!")
                print("   The reranker is now ready for production use")
                
            except Exception as rerank_error:
                print(f"   ❌ Reranking functionality failed: {rerank_error}")
                return False
                
        else:
            print("   ❌ Failed to load reranker")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_docker_reranker()
    if success:
        print("\n✅ VERIFICATION PASSED: Reranker is working in Docker")
    else:
        print("\n❌ VERIFICATION FAILED: Issues detected")
        sys.exit(1)