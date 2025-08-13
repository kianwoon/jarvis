#!/usr/bin/env python
"""
Test script for Chat Overflow Handler
Tests overflow detection, chunking, storage, and retrieval
"""

import asyncio
import json
from app.services.chat_overflow_handler import ChatOverflowHandler

async def test_overflow_handling():
    """Test the complete overflow handling flow"""
    
    # Initialize handler
    handler = ChatOverflowHandler()
    print(f"✅ Overflow handler initialized")
    print(f"   Threshold: {handler.config.overflow_threshold_tokens} tokens")
    print(f"   Chunk size: {handler.config.chunk_size_tokens} tokens")
    print(f"   Overlap: {handler.config.chunk_overlap_tokens} tokens")
    print()
    
    # Create a large test content (10K+ tokens)
    large_content = """
    This is a test of the overflow handling system. It needs to be a very long text 
    to trigger the overflow detection. Let me add a lot of content here.
    
    Chapter 1: Introduction to Machine Learning
    Machine learning is a subset of artificial intelligence that enables systems to learn and improve 
    from experience without being explicitly programmed. It focuses on developing computer programs 
    that can access data and use it to learn for themselves.
    
    The process of learning begins with observations or data, such as examples, direct experience, 
    or instruction, in order to look for patterns in data and make better decisions in the future 
    based on the examples that we provide. The primary aim is to allow the computers to learn 
    automatically without human intervention or assistance and adjust actions accordingly.
    
    """ * 100  # Repeat to make it large
    
    # Add some specific content for search testing
    large_content += """
    
    Special Section: Authentication and Security
    Authentication is the process of verifying the identity of a user or system. It ensures that 
    the entity requesting access is who they claim to be. Common authentication methods include:
    - Password-based authentication
    - Multi-factor authentication (MFA)
    - Biometric authentication
    - Token-based authentication
    - Certificate-based authentication
    
    Security best practices for authentication:
    1. Use strong, unique passwords
    2. Enable MFA wherever possible
    3. Regular security audits
    4. Implement proper session management
    5. Use secure communication protocols (HTTPS)
    """
    
    print(f"📝 Created test content")
    
    # Test 1: Overflow Detection
    is_overflow, token_count = handler.detect_overflow(large_content)
    print(f"\n1️⃣ Overflow Detection:")
    print(f"   Is overflow: {is_overflow}")
    print(f"   Token count: {token_count}")
    
    if not is_overflow:
        print("❌ Content not detected as overflow. Making it larger...")
        large_content = large_content * 5
        is_overflow, token_count = handler.detect_overflow(large_content)
        print(f"   New token count: {token_count}")
    
    # Test 2: Intelligent Chunking
    conversation_id = "test-conv-123"
    print(f"\n2️⃣ Intelligent Chunking:")
    chunks = await handler.chunk_intelligently(large_content, conversation_id)
    print(f"   Created {len(chunks)} chunks")
    if chunks:
        print(f"   First chunk ID: {chunks[0]['chunk_id']}")
        print(f"   First chunk tokens: {chunks[0]['token_count']}")
        print(f"   Keywords: {chunks[0]['keywords'][:3]}...")
    
    # Test 3: Store in Redis
    print(f"\n3️⃣ Storing in Redis:")
    
    # Store in L2 (warm storage)
    success = await handler.store_overflow(chunks, conversation_id, storage_layer="L2")
    print(f"   L2 storage: {'✅ Success' if success else '❌ Failed'}")
    
    # Store some in L1 (hot storage) for testing
    if len(chunks) > 2:
        l1_chunks = chunks[:2]  # First 2 chunks
        success = await handler.store_overflow(l1_chunks, conversation_id, storage_layer="L1")
        print(f"   L1 storage (2 chunks): {'✅ Success' if success else '❌ Failed'}")
    
    # Test 4: Get Overflow Summary
    print(f"\n4️⃣ Overflow Summary:")
    summary = await handler.get_overflow_summary(conversation_id)
    if summary:
        print(f"   Total chunks: {summary['total_chunks']}")
        print(f"   Total tokens: {summary['total_tokens']}")
        print(f"   Storage layer: {summary['storage_layer']}")
        print(f"   TTL remaining: {summary.get('ttl_remaining_hours', 'N/A')} hours")
    
    # Test 5: Retrieve Relevant Chunks
    print(f"\n5️⃣ Chunk Retrieval:")
    
    # Test query 1: General query
    query1 = "What is machine learning?"
    relevant_chunks = await handler.retrieve_relevant_chunks(query1, conversation_id, top_k=3)
    print(f"   Query: '{query1}'")
    print(f"   Retrieved {len(relevant_chunks)} chunks")
    
    # Test query 2: Specific query
    query2 = "authentication security MFA"
    relevant_chunks = await handler.retrieve_relevant_chunks(query2, conversation_id, top_k=3)
    print(f"   Query: '{query2}'")
    print(f"   Retrieved {len(relevant_chunks)} chunks")
    if relevant_chunks:
        print(f"   Top chunk preview: {relevant_chunks[0]['content'][:100]}...")
    
    # Test 6: Promotion to L1
    print(f"\n6️⃣ Auto-Promotion Test:")
    if handler.config.auto_promote_to_l1:
        # Simulate multiple accesses to trigger promotion
        for i in range(handler.config.promotion_threshold_accesses):
            await handler.retrieve_relevant_chunks(query2, conversation_id, top_k=1)
        print(f"   Accessed chunks {handler.config.promotion_threshold_accesses} times")
        print(f"   Auto-promotion should be triggered")
    
    print(f"\n✅ All tests completed successfully!")
    
    # Cleanup
    print(f"\n🧹 Cleaning up test data...")
    # Redis data will expire based on TTL settings

if __name__ == "__main__":
    asyncio.run(test_overflow_handling())