#!/usr/bin/env python3
"""
Debug script to examine what content is actually being retrieved for Tencent partnership queries
"""
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def test_tencent_partnership_retrieval():
    """Test what content is actually retrieved for Tencent partnership queries"""
    try:
        from app.langchain.service import handle_rag_query
        from app.core.llm_settings_cache import get_llm_settings
        from app.core.embedding_settings_cache import get_embedding_settings
        from app.core.vector_db_settings_cache import get_vector_db_settings
        
        print("🔍 Testing Tencent Partnership Content Retrieval")
        print("=" * 60)
        
        # Test query
        test_query = "search internal knowledge base. relationship between beyondsoft and tencent in details"
        
        # Get required configurations
        embedding_cfg = get_embedding_settings()
        vector_db_cfg = get_vector_db_settings()
        llm_cfg = get_llm_settings()
        
        print(f"Query: {test_query}")
        print(f"Collections to search: ['partnership', 'default_knowledge']")
        
        # Call handle_rag_query directly to see what content is retrieved
        print("\n📄 Calling handle_rag_query...")
        
        try:
            results = handle_rag_query(
                question=test_query,
                thinking=False,
                collections=['partnership'],  # Focus on partnership collection only
                collection_strategy="specific"
            )
            
            # Analyze results
            if results and len(results) > 1:
                context = results[1]  # Second item should be context
                sources = results[0] if isinstance(results[0], list) else []
                
                print(f"\n✅ Retrieved {len(sources)} documents")
                print(f"📝 Context length: {len(context)} characters")
                
                # Analyze content for Tencent-specific information
                context_lower = context.lower()
                tencent_mentions = context_lower.count('tencent')
                beyondsoft_mentions = context_lower.count('beyondsoft')
                partnership_mentions = context_lower.count('partnership')
                
                print(f"\n🔍 Content Analysis:")
                print(f"  Tencent mentions: {tencent_mentions}")
                print(f"  Beyondsoft mentions: {beyondsoft_mentions}")
                print(f"  Partnership mentions: {partnership_mentions}")
                
                # Show first 500 chars of context
                print(f"\n📄 Context Preview (first 500 chars):")
                print("-" * 40)
                print(context[:500])
                print("-" * 40)
                
                # Analyze individual sources
                print(f"\n📚 Document Sources Analysis:")
                for i, source in enumerate(sources[:5]):  # First 5 sources
                    if hasattr(source, 'metadata') and 'source' in source.metadata:
                        source_file = source.metadata['source']
                        content = source.page_content if hasattr(source, 'page_content') else str(source)
                        content_preview = content[:100].replace('\n', ' ')
                        
                        print(f"  {i+1}. File: {source_file}")
                        print(f"     Content: {content_preview}...")
                        print(f"     Tencent in content: {'tencent' in content.lower()}")
                        print()
                
            else:
                print("❌ No results returned from handle_rag_query")
                
        except Exception as e:
            print(f"❌ Error in handle_rag_query: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Error in test setup: {e}")
        import traceback
        traceback.print_exc()

def check_specific_tencent_documents():
    """Check if Tencent partnership documents exist in the database"""
    try:
        from app.core.db import SessionLocal
        from sqlalchemy import text
        
        print(f"\n🗄️  Database Document Check")
        print("=" * 60)
        
        db = SessionLocal()
        try:
            # Query for documents containing 'tencent' in partnership collection
            query = text("""
                SELECT 
                    collection_name,
                    source,
                    LEFT(content, 200) as content_preview,
                    LENGTH(content) as content_length
                FROM documents 
                WHERE collection_name = 'partnership' 
                AND (LOWER(content) LIKE '%tencent%' OR LOWER(source) LIKE '%tencent%')
                ORDER BY content_length DESC
                LIMIT 10
            """)
            
            results = db.execute(query).fetchall()
            
            if results:
                print(f"✅ Found {len(results)} Tencent-related documents in partnership collection:")
                for i, row in enumerate(results, 1):
                    print(f"\n{i}. Source: {row.source}")
                    print(f"   Collection: {row.collection_name}")
                    print(f"   Content length: {row.content_length} chars")
                    print(f"   Preview: {row.content_preview}...")
            else:
                print("❌ No Tencent-related documents found in partnership collection")
                
                # Check if any documents exist in partnership collection at all
                query2 = text("""
                    SELECT COUNT(*) as total_docs, 
                           array_agg(DISTINCT source) as sources
                    FROM documents 
                    WHERE collection_name = 'partnership'
                """)
                
                result2 = db.execute(query2).fetchone()
                print(f"\n📊 Partnership collection has {result2.total_docs} total documents")
                if result2.sources:
                    print(f"📂 Sources: {result2.sources}")
                
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Database check failed: {e}")

if __name__ == "__main__":
    test_tencent_partnership_retrieval()
    check_specific_tencent_documents()
    print(f"\n✨ Debug completed")