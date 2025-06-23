"""
Demo script for the Standalone Agent-Based RAG Module

This script demonstrates the key features and capabilities of the RAG agent system
including intelligent routing, multi-collection search, and different integration patterns.
"""

import asyncio
import logging
import time
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import RAG components
from app.rag_agent.interfaces.rag_interface import StandaloneRAGInterface, rag_query
from app.rag_agent.interfaces.integration_adapters import (
    StandardChatAdapter, MultiAgentAdapter, AgentContext
)
from app.rag_agent.utils.types import SearchContext, RAGOptions


async def demo_basic_queries():
    """Demonstrate basic RAG queries"""
    
    print("🔍 DEMO 1: Basic RAG Queries")
    print("=" * 50)
    
    rag = StandaloneRAGInterface()
    
    # Test queries for different domains
    test_queries = [
        "What is our data retention policy?",
        "How do I configure API access?", 
        "What are the KYC requirements?",
        "Explain our risk management framework",
        "How do I troubleshoot login issues?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 30)
        
        start_time = time.time()
        response = await rag.query(query)
        duration = time.time() - start_time
        
        print(f"Response ({duration:.2f}s, confidence: {response.confidence_score:.2f}):")
        print(response.content[:200] + "..." if len(response.content) > 200 else response.content)
        print(f"Sources: {len(response.sources)} from collections: {response.collections_searched}")
        print()


async def demo_streaming_queries():
    """Demonstrate streaming RAG queries"""
    
    print("🌊 DEMO 2: Streaming RAG Queries")
    print("=" * 50)
    
    rag = StandaloneRAGInterface()
    
    query = "What are the compliance requirements for customer onboarding?"
    print(f"Streaming Query: {query}")
    print("-" * 30)
    
    async for chunk in rag.query_stream(query):
        if chunk.chunk_type == "status":
            print(f"Status: {chunk.content}")
        elif chunk.chunk_type == "content":
            print(f"Content: {chunk.content}")
            if chunk.is_final:
                print(f"Final response with {len(chunk.sources)} sources")
        elif chunk.chunk_type == "error":
            print(f"Error: {chunk.content}")
    
    print()


async def demo_context_aware_queries():
    """Demonstrate context-aware queries"""
    
    print("🎯 DEMO 3: Context-Aware RAG Queries")
    print("=" * 50)
    
    rag = StandaloneRAGInterface()
    
    # Create different contexts
    contexts = [
        ("Compliance Officer", SearchContext(
            user_id="compliance_user", 
            domain="compliance",
            required_accuracy="high",
            user_permissions=["regulatory_compliance", "audit_reports"]
        )),
        ("Developer", SearchContext(
            user_id="dev_user",
            domain="technical", 
            required_accuracy="high",
            user_permissions=["technical_docs", "api_documentation"]
        )),
        ("Customer Support", SearchContext(
            user_id="support_user",
            domain="support",
            required_accuracy="medium",
            user_permissions=["customer_support", "product_documentation"]
        ))
    ]
    
    query = "What authentication methods are available?"
    
    for role, context in contexts:
        print(f"\nQuerying as {role}:")
        print(f"Context: domain={context.domain}, permissions={len(context.user_permissions)}")
        print("-" * 30)
        
        response = await rag.query(query, context=context)
        
        print(f"Collections searched: {response.collections_searched}")
        print(f"Confidence: {response.confidence_score:.2f}")
        print(f"Response preview: {response.content[:150]}...")
        print()


async def demo_advanced_options():
    """Demonstrate advanced RAG options"""
    
    print("⚙️ DEMO 4: Advanced RAG Options")
    print("=" * 50)
    
    rag = StandaloneRAGInterface()
    
    # Test with different option configurations
    query = "How do we handle data privacy compliance?"
    
    option_configs = [
        ("High Precision", RAGOptions(
            confidence_threshold=0.8,
            max_iterations=1,
            include_execution_trace=True
        )),
        ("High Recall", RAGOptions(
            confidence_threshold=0.4,
            max_iterations=3,
            max_results_per_collection=15
        )),
        ("Fast Response", RAGOptions(
            max_iterations=1,
            execution_timeout_ms=5000,
            max_results_per_collection=5
        ))
    ]
    
    for config_name, options in option_configs:
        print(f"\nConfiguration: {config_name}")
        print(f"Options: threshold={options.confidence_threshold}, iterations={options.max_iterations}")
        print("-" * 30)
        
        start_time = time.time()
        response = await rag.query(query, options=options)
        duration = time.time() - start_time
        
        print(f"Duration: {duration:.2f}s")
        print(f"Confidence: {response.confidence_score:.2f}")
        print(f"Sources: {len(response.sources)}")
        print(f"Collections: {response.collections_searched}")
        
        if options.include_execution_trace and response.execution_trace:
            print(f"Execution trace: {response.execution_trace.total_time_ms}ms, "
                  f"{len(response.execution_trace.steps_executed)} steps")
        print()


async def demo_chat_integration():
    """Demonstrate chat system integration"""
    
    print("💬 DEMO 5: Chat System Integration")
    print("=" * 50)
    
    chat_adapter = StandardChatAdapter()
    conversation_id = "demo_conversation"
    
    # Simulate a conversation
    conversation = [
        "Hello, I need help with our privacy policy",
        "Specifically, how long do we retain customer data?",
        "What about data deletion requests?",
        "Are there any exceptions to the deletion policy?"
    ]
    
    for i, message in enumerate(conversation, 1):
        print(f"\nUser Message {i}: {message}")
        print("-" * 30)
        
        # Process message with conversation context
        response = await chat_adapter.process_chat_message(
            message=message,
            conversation_id=conversation_id,
            user_id="demo_user",
            user_context={"domain": "compliance", "permissions": ["policies"]}
        )
        
        print(f"Assistant Response (confidence: {response.confidence:.2f}):")
        print(response.message[:200] + "..." if len(response.message) > 200 else response.message)
        print(f"Sources: {len(response.sources)}")
        print(f"Processing time: {response.metadata.get('processing_time_ms', 0)}ms")
        print()


async def demo_agent_integration():
    """Demonstrate multi-agent system integration"""
    
    print("🤖 DEMO 6: Multi-Agent System Integration")
    print("=" * 50)
    
    agent_adapter = MultiAgentAdapter()
    
    # Define different agent types
    agents = [
        ("Compliance Agent", AgentContext(
            agent_id="compliance_agent",
            role="compliance_officer",
            current_task="regulatory_review",
            execution_context={"priority": "high"}
        )),
        ("Technical Agent", AgentContext(
            agent_id="tech_agent", 
            role="technical_specialist",
            current_task="api_documentation",
            execution_context={"accuracy_required": "high"}
        )),
        ("Research Agent", AgentContext(
            agent_id="research_agent",
            role="research_analyst", 
            current_task="information_gathering",
            execution_context={"scope": "comprehensive"}
        ))
    ]
    
    query = "What security measures are in place for data transmission?"
    
    for agent_name, agent_context in agents:
        print(f"\n{agent_name} Query:")
        print(f"Role: {agent_context.role}, Task: {agent_context.current_task}")
        print("-" * 30)
        
        result = await agent_adapter.knowledge_search_tool(
            query=query,
            agent_context=agent_context,
            tool_config={
                "confidence_threshold": 0.7,
                "max_iterations": 2,
                "accuracy": "high"
            }
        )
        
        print(f"Success: {result.success}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Sources: {len(result.sources)}")
        print(f"Content preview: {result.content[:150]}...")
        
        if result.recommendations:
            print(f"Recommendations: {', '.join(result.recommendations[:2])}")
        print()


async def demo_batch_processing():
    """Demonstrate batch query processing"""
    
    print("📦 DEMO 7: Batch Query Processing")
    print("=" * 50)
    
    rag = StandaloneRAGInterface()
    
    # Multiple related queries
    batch_queries = [
        "What is our password policy?",
        "How often should passwords be changed?", 
        "What are the requirements for strong passwords?",
        "How do we handle password resets?",
        "What is our policy on password sharing?"
    ]
    
    print(f"Processing {len(batch_queries)} queries in parallel...")
    print("-" * 30)
    
    start_time = time.time()
    responses = await rag.batch_query(
        queries=batch_queries,
        context=SearchContext(domain="security"),
        options=RAGOptions(confidence_threshold=0.6),
        max_concurrent=3
    )
    total_time = time.time() - start_time
    
    print(f"\nBatch processing completed in {total_time:.2f}s")
    print(f"Average per query: {total_time/len(batch_queries):.2f}s")
    print()
    
    for i, (query, response) in enumerate(zip(batch_queries, responses), 1):
        print(f"Query {i}: {query}")
        print(f"  Confidence: {response.confidence_score:.2f}")
        print(f"  Sources: {len(response.sources)}")
        print(f"  Collections: {response.collections_searched}")
        print()


async def demo_health_and_performance():
    """Demonstrate health checking and performance monitoring"""
    
    print("📊 DEMO 8: Health Check and Performance Monitoring")
    print("=" * 50)
    
    rag = StandaloneRAGInterface()
    
    # Perform health check
    print("Performing health check...")
    health_status = await rag.health_check()
    
    print(f"System Status: {health_status['status']}")
    print(f"Collections Available: {health_status.get('collections_available', 0)}")
    print()
    
    # Show performance stats
    print("Performance Statistics:")
    stats = rag.get_performance_stats()
    
    interface_stats = stats.get('interface_stats', {})
    print(f"  Total Queries: {interface_stats.get('total_queries', 0)}")
    print(f"  Average Response Time: {interface_stats.get('avg_response_time_ms', 0):.0f}ms")
    print(f"  Success Rate: {interface_stats.get('success_rate', 0):.1%}")
    
    orchestrator_stats = stats.get('orchestrator_stats', {})
    print(f"  Orchestrator Queries: {orchestrator_stats.get('total_queries', 0)}")
    print(f"  Orchestrator Success Rate: {orchestrator_stats.get('success_rate', 0):.1%}")
    print()


async def run_all_demos():
    """Run all demonstration scenarios"""
    
    print("🚀 Standalone Agent-Based RAG Module Demo")
    print("=" * 60)
    print("This demo showcases the key features of the RAG agent system:")
    print("- Intelligent LLM-based collection routing")
    print("- Multi-collection search and result fusion") 
    print("- Context-aware query processing")
    print("- Integration with chat and agent systems")
    print("- Streaming and batch processing capabilities")
    print("=" * 60)
    print()
    
    demos = [
        demo_basic_queries,
        demo_streaming_queries, 
        demo_context_aware_queries,
        demo_advanced_options,
        demo_chat_integration,
        demo_agent_integration,
        demo_batch_processing,
        demo_health_and_performance
    ]
    
    for i, demo_func in enumerate(demos, 1):
        try:
            await demo_func()
            if i < len(demos):
                print("⏸️  Press Enter to continue to next demo...")
                input()
                print()
        except Exception as e:
            logger.error(f"Demo {i} failed: {e}")
            print(f"❌ Demo {i} failed: {e}")
            print()
            continue
    
    print("✅ All demos completed successfully!")
    print("\nThe RAG agent system is ready for integration with your applications.")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_all_demos())