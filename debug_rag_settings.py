#!/usr/bin/env python3
"""
Debug script to examine current RAG settings configuration
"""
import json
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def check_rag_settings():
    """Check and display current RAG settings from database"""
    try:
        from app.core.db import SessionLocal, Settings as SettingsModel
        
        db = SessionLocal()
        try:
            # Get RAG settings from database
            rag_settings_row = db.query(SettingsModel).filter(SettingsModel.category == 'rag').first()
            
            if not rag_settings_row:
                print("❌ No RAG settings found in database")
                return
            
            settings = rag_settings_row.settings
            if not settings:
                print("❌ RAG settings row exists but contains no data")
                return
                
            print("🔍 Current RAG Settings Configuration:")
            print("=" * 60)
            
            # Parse and display each section
            sections_to_check = [
                'bm25_scoring',
                'reranking', 
                'document_retrieval',
                'performance',
                'collection_selection',
                'search_strategy'
            ]
            
            for section in sections_to_check:
                print(f"\n📋 {section.upper().replace('_', ' ')}:")
                print("-" * 40)
                
                if section in settings:
                    section_data = settings[section]
                    for key, value in section_data.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"  ❌ Section '{section}' not found")
            
            # Check for any other sections not in our expected list
            other_sections = set(settings.keys()) - set(sections_to_check)
            if other_sections:
                print(f"\n📋 OTHER SECTIONS:")
                print("-" * 40)
                for section in other_sections:
                    print(f"  {section}: {type(settings[section]).__name__} with {len(settings[section]) if isinstance(settings[section], dict) else 'N/A'} items")
            
            # Display full JSON for reference
            print(f"\n📄 FULL RAG SETTINGS JSON:")
            print("-" * 40)
            print(json.dumps(settings, indent=2))
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Error checking RAG settings: {e}")
        import traceback
        traceback.print_exc()

def check_bm25_integration():
    """Check if BM25 is properly integrated in the current system"""
    print(f"\n🔧 BM25 Integration Check:")
    print("=" * 60)
    
    try:
        from app.rag.bm25_processor import BM25Processor
        processor = BM25Processor()
        print(f"✅ BM25Processor loaded successfully")
        print(f"  k1 parameter: {processor.k1}")
        print(f"  b parameter: {processor.b}")
        print(f"  Stop words count: {len(processor.stop_words)}")
    except Exception as e:
        print(f"❌ BM25Processor failed to load: {e}")
    
    try:
        from app.rag.qwen_reranker import get_qwen_reranker
        from app.core.reranker_config import RerankerConfig
        
        print(f"\n🎯 Reranker Status:")
        print(f"  Enabled: {RerankerConfig.is_enabled()}")
        print(f"  Should preload: {RerankerConfig.should_preload()}")
        print(f"  Device: {RerankerConfig.get_device()}")
        print(f"  Batch size: {RerankerConfig.get_batch_size()}")
        
        if RerankerConfig.is_enabled():
            try:
                reranker = get_qwen_reranker()
                if reranker:
                    print(f"  ✅ Qwen reranker instance available")
                else:
                    print(f"  ❌ Qwen reranker failed to initialize")
            except Exception as e:
                print(f"  ❌ Qwen reranker error: {e}")
        
    except Exception as e:
        print(f"❌ Reranker check failed: {e}")

def analyze_potential_issues():
    """Analyze potential configuration issues affecting response quality"""
    print(f"\n🔍 Potential Issues Analysis:")
    print("=" * 60)
    
    try:
        from app.core.rag_settings_cache import get_rag_settings
        settings = get_rag_settings()
        
        issues = []
        recommendations = []
        
        # Check document retrieval settings
        if 'document_retrieval' in settings:
            doc_settings = settings['document_retrieval']
            
            max_docs = doc_settings.get('max_documents_per_collection', 0)
            if max_docs < 20:
                issues.append(f"Low max_documents_per_collection: {max_docs} (may limit content richness)")
                recommendations.append("Consider increasing max_documents_per_collection to 20-30")
            
            similarity_threshold = doc_settings.get('similarity_threshold', 0)
            if similarity_threshold > 1.0:
                issues.append(f"High similarity_threshold: {similarity_threshold} (may exclude relevant docs)")
                recommendations.append("Consider lowering similarity_threshold to 0.8-1.0")
        
        # Check BM25 settings
        if 'bm25_scoring' in settings:
            bm25_settings = settings['bm25_scoring']
            
            k1 = bm25_settings.get('k1', 1.2)
            b = bm25_settings.get('b', 0.75)
            
            if k1 < 1.0 or k1 > 2.0:
                issues.append(f"BM25 k1 parameter outside optimal range: {k1} (optimal: 1.2-2.0)")
                recommendations.append("Adjust BM25 k1 to 1.2-1.5 for better term frequency handling")
            
            if b < 0.5 or b > 1.0:
                issues.append(f"BM25 b parameter outside optimal range: {b} (optimal: 0.7-0.8)")
                recommendations.append("Adjust BM25 b to 0.75 for better document length normalization")
        
        # Check reranking settings
        if 'reranking' in settings:
            rerank_settings = settings['reranking']
            
            rerank_weight = rerank_settings.get('rerank_weight', 0.7)
            if rerank_weight < 0.5:
                issues.append(f"Low rerank_weight: {rerank_weight} (may not benefit from reranking)")
                recommendations.append("Increase rerank_weight to 0.6-0.8 for better relevance")
        
        # Display findings
        if issues:
            print("⚠️  POTENTIAL ISSUES:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("✅ No obvious configuration issues detected")
        
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")

if __name__ == "__main__":
    print("🚀 RAG Settings Debug Script")
    print("=" * 60)
    
    check_rag_settings()
    check_bm25_integration()  
    analyze_potential_issues()
    
    print(f"\n✨ Debug script completed")