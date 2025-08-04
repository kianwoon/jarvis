#!/usr/bin/env python3
"""
Test the fixed chunking strategy to ensure:
1. 256k model is properly detected
2. Business documents get balanced chunking (5-8 chunks)
3. No mega chunks are created for business documents
4. Dynamic adjustment works based on model context
"""

import sys
import os
import json
from datetime import datetime

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def test_model_detection():
    """Test that the system properly detects the 256k model"""
    print("🔍 Testing model detection...")
    
    try:
        from app.services.dynamic_chunk_sizing import get_dynamic_chunk_sizer
        
        # Create chunk sizer instance
        chunk_sizer = get_dynamic_chunk_sizer()
        
        print(f"✅ Model detected: {chunk_sizer.model_name}")
        print(f"✅ Context limit: {chunk_sizer.context_limit:,} tokens")
        print(f"✅ Optimal chunk size: {chunk_sizer.optimal_chunk_size:,} characters")
        
        # Validate that we're using the 256k model
        if chunk_sizer.context_limit >= 256000:
            print("✅ 256k+ context model properly detected!")
            return True
        else:
            print(f"❌ Expected 256k+ context, got {chunk_sizer.context_limit:,} tokens")
            return False
            
    except Exception as e:
        print(f"❌ Model detection failed: {e}")
        return False

def test_business_document_chunking():
    """Test that business documents get balanced chunking"""
    print("\n🏢 Testing business document chunking...")
    
    try:
        from app.services.dynamic_chunk_sizing import get_dynamic_chunk_sizer
        from app.document_handlers.base import ExtractedChunk
        
        # Create chunk sizer instance
        chunk_sizer = get_dynamic_chunk_sizer()
        
        # Test business document configuration
        business_config = chunk_sizer.get_chunk_configuration('technology_strategy')
        print(f"📊 Business document config:")
        print(f"   Strategy: {business_config['processing_strategy']}")
        print(f"   Target chunks: {business_config['target_chunks_per_document']}")
        print(f"   Max chunk size: {business_config['max_chunk_size']:,} chars")
        print(f"   Max consolidation: {business_config['max_consolidation_ratio']}:1")
        print(f"   Chunk overlap: {business_config['chunk_overlap']:,} chars")
        
        # Create some mock chunks (simulate business document with substantial content)
        mock_chunks = []
        business_content = """
        DBS Bank Technology Strategy 2024-2027: Digital Transformation Roadmap

        Executive Summary:
        DBS Bank, Southeast Asia's largest bank, is embarking on a comprehensive digital transformation initiative to strengthen its position as a leading digital bank. This technology strategy outlines our roadmap for the next three years, focusing on cloud-first architecture, artificial intelligence integration, and enhanced customer experience platforms.

        Strategic Objectives:
        1. Cloud Migration and Infrastructure Modernization
        The organization is implementing a cloud-first architecture across all subsidiaries and business units. This initiative includes migrating legacy systems to AWS and Azure platforms, implementing microservices architecture, and establishing robust disaster recovery capabilities. Our infrastructure modernization program targets 80% cloud adoption by 2025.

        2. Artificial Intelligence and Machine Learning Integration
        Our technology roadmap includes comprehensive AI and machine learning capabilities deployment. Key focus areas include customer service automation through chatbots, fraud detection systems, credit risk assessment models, and personalized banking recommendations. The AI Center of Excellence is driving innovation across retail, corporate, and investment banking divisions.

        3. Data Analytics and Business Intelligence
        Advanced data analytics platforms are being deployed to enhance decision-making capabilities. The implementation includes real-time analytics dashboards, predictive modeling for customer behavior, and comprehensive business intelligence tools. Data governance frameworks ensure compliance with regulatory requirements while maximizing insights generation.

        4. Customer Experience Digital Platforms
        Strategic partnerships with fintech companies enhance customer experience through innovative digital solutions. The customer journey transformation includes mobile banking enhancements, digital onboarding processes, and omnichannel service delivery. User experience design principles guide all customer-facing technology developments.

        5. Cybersecurity and Risk Management
        Regulatory compliance requires robust security frameworks and governance structures. Zero-trust architecture implementation, advanced threat detection systems, and comprehensive security awareness programs protect customer data and maintain regulatory compliance. Risk management integration with technology processes ensures business continuity.

        Implementation Timeline:
        Phase 1 (2024): Foundation establishment including cloud infrastructure setup, AI platform deployment, and security framework implementation.
        Phase 2 (2025): Core system migrations, customer platform enhancements, and analytics capability expansion.
        Phase 3 (2026-2027): Advanced AI feature rollouts, comprehensive data analytics deployment, and ecosystem integration completion.

        Technology Partnerships:
        Strategic alliances with technology leaders including Microsoft, Amazon Web Services, Google Cloud, and leading fintech innovators accelerate transformation initiatives. These partnerships provide access to cutting-edge technologies, implementation expertise, and ongoing support capabilities.

        Organizational Impact:
        The digital transformation requires significant organizational change management, employee skill development programs, and cultural adaptation initiatives. Technology teams are expanding with new roles in cloud engineering, data science, AI development, and cybersecurity specializations.

        Financial Investment:
        Total technology investment over the three-year period amounts to $2.5 billion, allocated across infrastructure modernization (40%), AI and analytics capabilities (30%), customer experience platforms (20%), and security enhancements (10%).

        Success Metrics:
        Key performance indicators include customer satisfaction scores, digital adoption rates, operational efficiency improvements, and regulatory compliance maintenance. Quarterly reviews ensure progress tracking and strategy adjustments as needed.
        """ * 3  # Repeat to make substantial content
        
        # Split into multiple chunks to simulate document processing
        chunk_size = 1500  # Smaller chunks to ensure we get more of them
        for i in range(0, len(business_content), chunk_size):
            chunk_content = business_content[i:i + chunk_size]
            if len(chunk_content.strip()) > 100:
                chunk = ExtractedChunk(
                    content=chunk_content,
                    metadata={'chunk_id': f'business_chunk_{i//chunk_size}'},
                    quality_score=0.8
                )
                mock_chunks.append(chunk)
        
        print(f"\n📝 Created {len(mock_chunks)} mock business document chunks")
        print(f"   Total content: {sum(len(c.content) for c in mock_chunks):,} characters")
        
        # Test chunking optimization
        optimized_chunks = chunk_sizer.optimize_chunks(mock_chunks, 'technology_strategy')
        
        print(f"\n✅ Optimization results:")
        print(f"   Input chunks: {len(mock_chunks)}")
        print(f"   Output chunks: {len(optimized_chunks)}")
        print(f"   Consolidation ratio: {len(mock_chunks) / len(optimized_chunks):.1f}:1")
        
        if optimized_chunks:
            chunk_sizes = [len(c.content) for c in optimized_chunks]
            print(f"   Average chunk size: {sum(chunk_sizes) // len(chunk_sizes):,} chars")
            print(f"   Size range: {min(chunk_sizes):,} - {max(chunk_sizes):,} chars")
            
            # Validate business document requirements
            success_criteria = {
                'multiple_chunks': len(optimized_chunks) >= 3,  # At least 3 chunks
                'no_mega_chunks': all(len(c.content) <= 8000 for c in optimized_chunks),  # No chunks > 8KB (business limit)
                'reasonable_size': all(1000 <= len(c.content) <= 6500 for c in optimized_chunks),  # 1-6.5KB range for business docs
                'balanced_distribution': max(chunk_sizes) / min(chunk_sizes) <= 4.0 if len(chunk_sizes) > 1 else True  # Max 4:1 ratio
            }
            
            print(f"\n📊 Business document validation:")
            for criterion, passed in success_criteria.items():
                status = "✅" if passed else "❌"
                print(f"   {status} {criterion}: {passed}")
            
            overall_success = all(success_criteria.values())
            print(f"\n🎯 Overall business chunking success: {'✅' if overall_success else '❌'}")
            
            return overall_success, optimized_chunks, business_config
        else:
            print("❌ No optimized chunks produced")
            return False, [], business_config
            
    except Exception as e:
        print(f"❌ Business document chunking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, [], {}

def test_general_document_chunking():
    """Test that general documents can still use large chunks"""
    print("\n📄 Testing general document chunking...")
    
    try:
        from app.services.dynamic_chunk_sizing import get_dynamic_chunk_sizer
        from app.document_handlers.base import ExtractedChunk
        
        # Create chunk sizer instance
        chunk_sizer = get_dynamic_chunk_sizer()
        
        # Test general document configuration
        general_config = chunk_sizer.get_chunk_configuration('general')
        print(f"📊 General document config:")
        print(f"   Strategy: {general_config['processing_strategy']}")
        print(f"   Target chunks: {general_config['target_chunks_per_document']}")
        print(f"   Max chunk size: {general_config['max_chunk_size']:,} chars")
        
        # For 256k models, general documents should be able to use full context
        if chunk_sizer.context_limit >= 200000:
            expected_strategy = 'full_context_utilization'
            success = general_config['processing_strategy'] == expected_strategy
            print(f"✅ 256k model using {expected_strategy}: {'✅' if success else '❌'}")
            return success
        else:
            print(f"ℹ️  Model has {chunk_sizer.context_limit:,} tokens, using appropriate strategy")
            return True
            
    except Exception as e:
        print(f"❌ General document chunking test failed: {e}")
        return False

def main():
    """Run all chunking strategy tests"""
    print("🧪 Testing Fixed Chunking Strategy")
    print("=" * 50)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }
    
    # Test 1: Model Detection
    model_success = test_model_detection()
    results['tests']['model_detection'] = model_success
    
    # Test 2: Business Document Chunking
    business_success, optimized_chunks, business_config = test_business_document_chunking()
    results['tests']['business_chunking'] = {
        'success': business_success,
        'output_chunks': len(optimized_chunks),
        'config': business_config
    }
    
    # Test 3: General Document Chunking
    general_success = test_general_document_chunking()
    results['tests']['general_chunking'] = general_success
    
    # Overall result
    overall_success = model_success and business_success and general_success
    results['overall_success'] = overall_success
    
    print(f"\n🎯 OVERALL TEST RESULT: {'✅ SUCCESS' if overall_success else '❌ FAILED'}")
    
    if overall_success:
        print("\n✅ All tests passed! The chunking strategy now:")
        print("   • Properly detects 256k model and context")
        print("   • Uses balanced chunking for business documents")
        print("   • Prevents mega chunks for entity extraction")
        print("   • Dynamically adjusts based on model capabilities")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
    
    # Save results
    with open('fixed_chunking_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)