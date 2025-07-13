#!/usr/bin/env python3
"""
Trace RAG execution step by step to find where retrieval fails
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))
sys.path.insert(0, os.path.dirname(__file__))

from app.langchain.service import rag_answer
from app.core.llm_settings_cache import get_llm_settings, get_main_llm_full_config

def test_rag_answer_directly():
    """Test the rag_answer function directly with tracing"""
    try:
        print("🔍 Testing rag_answer function directly")
        print("=" * 60)
        
        query = "search internal knowledge base. relationship between beyondsoft and tencent in details"
        
        # Get LLM settings for context
        llm_settings = get_llm_settings()
        main_llm_config = get_main_llm_full_config(llm_settings)
        
        print(f"Query: {query}")
        print(f"LLM Model: {main_llm_config.get('model', 'Unknown')}")
        print(f"Max Tokens: {main_llm_config.get('max_tokens', 'Unknown')}")
        
        # Mock context from the Milvus data we found
        mock_context = """
Core Infrastructure & Operations Team - Ensure robust, scalable, and secure operation of Tencent's core technical infrastructure—including data centers, server farms, and network systems—that support both internal operations and customer-facing applications.

Methodology:
• Utilize advanced monitoring and alerting tools (e.g., Tencent's native monitoring systems, Prometheus, Grafana) integrated with ITIL-based incident management processes.
• Managing network performance, load balancing, and automated scaling to support millions of concurrent users.
• Implementing redundancy and disaster recovery protocols to guarantee uninterrupted service delivery.

Corporate Development Group (CDG) ≈200+ Staff - Corporate Development Group (CDG) is pivotal in driving Tencent's strategic planning, investment, and digital transformation initiatives. Our dedicated IT support services team—comprising approximately 200 staff—is organized into specialized units.

Database Solutions:
• Designing, developing, and optimizing TDSQL for strong consistency, high availability, and horizontal scalability.
• Managing fully automated relational database services (MySQL, PostgreSQL, SQL Server) with backup, scaling, and performance optimizations.
• Providing high-performance NoSQL solutions (TencentDB for Redis, MongoDB, etc.) for low-latency data access and caching.

Partnership with Beyondsoft:
Through this long-term, multi-faceted partnership, Beyondsoft has become an indispensable strategic partner—delivering high-quality technical support and innovative solutions that consistently drive Tencent's technological advancement and business success.

Technical Teams:
• Builds and maintains Tencent's IT infrastructure (servers, networks, data centers)
• Develops internal engineering tools and shared technology services
• Ensures security compliance
• Foundational tech frameworks
• Data center operations
• Internal R&D projects
"""
        
        print(f"\nContext length: {len(mock_context)} characters")
        
        # Call rag_answer directly
        print(f"\n📤 Calling rag_answer...")
        
        response = rag_answer(
            question=query,
            context=mock_context,
            conversation_history="",
            thinking=False
        )
        
        print(f"\n📥 RAG Response:")
        print(f"Length: {len(response)} characters")
        print(f"Content: {response}")
        
        # Analyze response quality
        response_lower = response.lower()
        tencent_mentions = response_lower.count('tencent')
        beyondsoft_mentions = response_lower.count('beyondsoft')
        
        print(f"\n📊 Response Analysis:")
        print(f"  Tencent mentions: {tencent_mentions}")
        print(f"  Beyondsoft mentions: {beyondsoft_mentions}")
        print(f"  Response/Context ratio: {len(response)/len(mock_context):.2f}")
        
        if len(response) < 1000:
            print(f"⚠️  ISSUE: Response is too short for the available context!")
        
    except Exception as e:
        print(f"❌ Error in rag_answer test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rag_answer_directly()
    print(f"\n✨ RAG tracing completed")