#!/usr/bin/env python3
"""
Simple test of the relevance scoring logic without dependencies
"""

def simple_relevance_score(query: str, context: str) -> float:
    """Simplified version of the relevance scoring logic"""
    if not context or not query:
        return 0.0
        
    import math
    from collections import Counter
    
    # Convert to lowercase for comparison
    query_lower = query.lower()
    context_lower = context.lower()
    
    # Stop words list
    stop_words = {
        'the', 'is', 'are', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
        'a', 'an', 'what', 'how', 'when', 'where', 'why', 'like', 'it', 'this', 'that', 'these', 
        'those', 'can', 'could', 'would', 'should', 'may', 'might', 'must', 'shall', 'will',
        'do', 'does', 'did', 'have', 'has', 'had', 'be', 'been', 'being', 'was', 'were',
        'find', 'get', 'show', 'tell', 'about', 'info', 'information', 'news', 'give', 'provide', 
        'need', 'want', 'please', 'help'
    }
    
    # Extract meaningful words
    query_words = [w for w in query_lower.split() if w not in stop_words and len(w) > 2]
    context_words_list = [w for w in context_lower.split() if len(w) > 2]
    
    if not query_words:
        query_words = query_lower.split()
    
    # Calculate term frequency (TF) for context
    context_word_freq = Counter(context_words_list)
    total_words = len(context_words_list)
    
    # Calculate TF-IDF-like score
    tfidf_score = 0.0
    matched_terms = 0
    matching_details = []
    
    for word in query_words:
        # Direct match
        if word in context_word_freq:
            tf = context_word_freq[word] / total_words if total_words > 0 else 0
            # Simulate IDF with a bonus for rare terms
            idf_bonus = 1.0 if context_word_freq[word] <= 2 else 0.5
            tfidf_score += tf * idf_bonus
            matched_terms += 1
            matching_details.append(f"Direct match: '{word}' (freq: {context_word_freq[word]}, tf: {tf:.3f})")
        # Partial match (stemming-like)
        else:
            partial_matches = [w for w in context_word_freq if word in w or w in word]
            if partial_matches:
                best_match = max(partial_matches, key=lambda w: context_word_freq[w])
                tf = context_word_freq[best_match] / total_words if total_words > 0 else 0
                tfidf_score += tf * 0.7  # Lower weight for partial matches
                matched_terms += 0.7
                matching_details.append(f"Partial match: '{word}' -> '{best_match}' (freq: {context_word_freq[best_match]}, tf: {tf:.3f})")
    
    # Normalize by query length
    base_score = matched_terms / len(query_words) if query_words else 0.0
    
    # BM25-like saturation (diminishing returns for high term frequency)
    bm25_score = tfidf_score / (tfidf_score + 0.5)
    
    # Exact phrase match bonus
    phrase_bonus = 0.0
    if len(query_words) > 1 and query_lower in context_lower:
        phrase_bonus = 0.3
    
    # Final score combining multiple factors
    final_score = base_score * 0.4 + bm25_score * 0.4 + phrase_bonus
    
    return final_score, matching_details

def test_scoring():
    """Test scoring with different content types"""
    
    query = "partnership between beyondsoft and tencent"
    
    test_docs = {
        "Tencent Partnership": """
        Beyondsoft's Partnership with Tencent started in 2012 with cloud infrastructure support.
        The collaboration focuses on cloud computing, AI development, and enterprise solutions.
        Tencent Cloud development teams work with Beyondsoft on container platforms and DevOps.
        """,
        "n8n AI Agent": """
        n8n ai agent messages workflow automation platform for connecting different services.
        The n8n platform enables users to build automated workflows with visual interface.
        AI agents can be configured to handle various tasks and messaging integrations.
        Configure AI agents with workflow automation. AI agent workflow development platform.
        """,
        "Alibaba Partnership": """
        Beyondsoft has strategic partnership with Alibaba Cloud since 2015.
        The partnership includes cloud migration services and AI development collaboration.
        Alibaba and Beyondsoft work together on enterprise digital transformation.
        """,
        "Random Content": """
        This is completely unrelated content about weather patterns and climate change.
        No mention of partnerships, companies, or technology platforms here.
        Just general information about environmental topics and sustainability.
        """
    }
    
    print("🔍 Testing Relevance Scoring")
    print("=" * 60)
    print(f"Query: '{query}'")
    print(f"Query words (after stop word removal): {[w for w in query.lower().split() if len(w) > 2]}")
    print()
    
    results = []
    
    for doc_name, content in test_docs.items():
        print(f"📄 Document: {doc_name}")
        print(f"Content preview: {content.strip()[:80]}...")
        
        score, details = simple_relevance_score(query, content)
        results.append((doc_name, score))
        
        print(f"Relevance Score: {score:.4f}")
        print("Matching details:")
        for detail in details:
            print(f"  - {detail}")
        print("-" * 40)
    
    print("\n🏆 Final Rankings:")
    results.sort(key=lambda x: x[1], reverse=True)
    for i, (doc_name, score) in enumerate(results):
        print(f"{i+1}. {doc_name}: {score:.4f}")

if __name__ == "__main__":
    test_scoring()