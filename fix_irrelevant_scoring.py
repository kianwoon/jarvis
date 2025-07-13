#!/usr/bin/env python3
"""
Fix the irrelevant document scoring issue by adding keyword relevance threshold
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def fix_irrelevant_scoring():
    """Fix the issue where irrelevant documents score high due to vector similarity"""
    try:
        print("🔧 Fixing Irrelevant Document Scoring Issue")
        print("=" * 60)
        
        # Read the current service file
        service_file = "/Users/kianwoonwong/Downloads/jarvis/app/langchain/service.py"
        
        with open(service_file, 'r') as f:
            content = f.read()
        
        # Find the section where combined_score is calculated
        if "combined_score = (similarity * vector_weight) + (enhanced_keyword_score * keyword_weight)" in content:
            print("✅ Found the hybrid scoring section")
            
            # Add keyword relevance threshold filter
            old_section = """            vector_weight = 1 - keyword_weight
            combined_score = (similarity * vector_weight) + (enhanced_keyword_score * keyword_weight)
            
            filtered_and_ranked.append((doc, score, similarity, keyword_relevance, combined_score))"""
            
            new_section = """            vector_weight = 1 - keyword_weight
            combined_score = (similarity * vector_weight) + (enhanced_keyword_score * keyword_weight)
            
            # Apply keyword relevance threshold for partnership queries to filter out irrelevant documents
            # This prevents documents with high vector similarity but zero keyword relevance from appearing
            if is_partnership_query:
                # For partnership queries, require minimum keyword relevance to prevent false positives
                from app.core.rag_settings_cache import get_search_strategy_settings
                try:
                    search_settings = get_search_strategy_settings()
                    min_keyword_relevance = search_settings.get('min_keyword_relevance_threshold', 0.1)
                except:
                    min_keyword_relevance = 0.1
                
                if keyword_relevance < min_keyword_relevance:
                    print(f"[DEBUG] Filtering out document with low keyword relevance: {keyword_relevance:.3f} < {min_keyword_relevance}")
                    continue  # Skip this document
            
            filtered_and_ranked.append((doc, score, similarity, keyword_relevance, combined_score))"""
            
            # Replace the section
            updated_content = content.replace(old_section, new_section)
            
            if updated_content != content:
                # Write the updated content
                with open(service_file, 'w') as f:
                    f.write(updated_content)
                
                print("✅ Successfully added keyword relevance threshold filter")
                print("📊 Changes made:")
                print("  - Added minimum keyword relevance threshold for partnership queries")
                print("  - Documents with high vector similarity but low keyword relevance will be filtered out")
                print("  - Threshold configurable via search_strategy_settings.min_keyword_relevance_threshold")
                print("  - Default threshold: 0.1 (10% keyword relevance required)")
                
                # Now update the settings to set the threshold
                print("\n🔧 Configuring keyword relevance threshold...")
                
                from app.core.rag_settings_cache import update_rag_setting
                
                # Set minimum keyword relevance threshold
                success = update_rag_setting('search_strategy', 'min_keyword_relevance_threshold', 0.15)
                
                if success:
                    print("✅ Set minimum keyword relevance threshold to 0.15 (15%)")
                    print("💡 This will filter out documents like 'n8n ai agent messages.pdf' that have:")
                    print("   - High vector similarity (~85.9%)")  
                    print("   - Zero keyword relevance (0%)")
                    print("   - But get included due to hybrid scoring")
                else:
                    print("❌ Failed to update keyword relevance threshold setting")
                    
            else:
                print("❌ Failed to find and replace the target section")
        else:
            print("❌ Could not find the hybrid scoring section to modify")
            
    except Exception as e:
        print(f"❌ Fix failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fix_irrelevant_scoring()