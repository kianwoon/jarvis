#!/usr/bin/env python3
"""
Test configurable chunking strategies based on processing purpose.
This demonstrates how the same business document can be chunked differently
based on whether it's for knowledge graph extraction or general processing.
"""

import sys
import os
import json
from datetime import datetime

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def create_business_document():
    """Create a business document for testing"""
    content = """
DBS Bank Technology Strategy 2024-2027: Comprehensive Digital Transformation Framework

Executive Summary:
DBS Bank, as Southeast Asia's leading financial institution, is embarking on an ambitious digital transformation journey that will reshape our technological infrastructure and customer experience delivery. This comprehensive strategy document outlines our three-year roadmap focusing on cloud-native architecture, artificial intelligence integration, advanced analytics platforms, and next-generation customer experience solutions.

Strategic Vision and Objectives:
Our digital transformation initiative is built on four foundational pillars: Infrastructure Modernization, Data-Driven Decision Making, Customer-Centric Innovation, and Operational Excellence. Each pillar represents a critical component of our evolution from a traditional banking institution to a technology-enabled financial services leader.

Infrastructure Modernization Initiative:
The organization is implementing a comprehensive cloud-first architecture across all subsidiaries and business units throughout the Asia-Pacific region. This multi-phase initiative includes migrating legacy mainframe systems to modern cloud platforms, implementing microservices architecture, establishing robust disaster recovery capabilities, and deploying advanced cybersecurity frameworks. Our infrastructure modernization program targets 85% cloud adoption by Q4 2025, with primary partnerships established with Amazon Web Services, Microsoft Azure, and Google Cloud Platform.

Artificial Intelligence and Machine Learning Integration:
Our technology roadmap includes comprehensive AI and machine learning capabilities deployment across retail banking, corporate banking, investment banking, and wealth management divisions. Key focus areas encompass customer service automation through advanced chatbot systems, sophisticated fraud detection algorithms, enhanced credit risk assessment models, personalized banking product recommendations, and predictive analytics for customer behavior analysis. The AI Center of Excellence, established in Singapore, is driving innovation initiatives across all business verticals with dedicated teams for natural language processing, computer vision, and deep learning applications.

Data Analytics and Business Intelligence Platform:
Advanced data analytics platforms are being deployed to enhance strategic decision-making capabilities across all organizational levels. The implementation includes real-time analytics dashboards, comprehensive predictive modeling for customer behavior patterns, advanced business intelligence tools, and integrated reporting systems. Our data governance framework ensures strict compliance with regulatory requirements while maximizing insights generation and maintaining data quality standards. The centralized data lake architecture supports both structured and unstructured data processing with advanced ETL pipelines.

Customer Experience Digital Transformation:
Strategic partnerships with leading fintech companies enhance customer experience through innovative digital solutions and seamless service delivery channels. The customer journey transformation encompasses mobile banking platform enhancements, streamlined digital onboarding processes, omnichannel service delivery integration, and personalized financial advisory services. User experience design principles guide all customer-facing technology developments with continuous feedback loops and iterative improvement processes.

Cybersecurity and Risk Management Framework:
Regulatory compliance requires robust security frameworks and comprehensive governance structures to protect customer data and maintain operational integrity. Our cybersecurity strategy includes zero-trust architecture implementation, advanced threat detection systems, comprehensive security awareness programs, and incident response protocols. Risk management integration with technology processes ensures business continuity while maintaining regulatory compliance across multiple jurisdictions.

Implementation Timeline and Milestones:
Phase 1 (2024): Foundation establishment including core cloud infrastructure setup, AI platform deployment, cybersecurity framework implementation, and initial data migration processes.
Phase 2 (2025): Comprehensive core system migrations, customer platform enhancements, advanced analytics capability expansion, and regional deployment initiatives.
Phase 3 (2026-2027): Advanced AI feature rollouts, comprehensive data analytics deployment, ecosystem integration completion, and performance optimization initiatives.

Technology Partnership Ecosystem:
Strategic alliances with technology leaders including Microsoft Corporation, Amazon Web Services, Google Cloud Platform, IBM, Oracle, and leading fintech innovators accelerate transformation initiatives while providing access to cutting-edge technologies, implementation expertise, and ongoing technical support capabilities. These partnerships enable rapid deployment of emerging technologies and ensure competitive advantage in the digital banking landscape.

Organizational Change Management:
The digital transformation requires significant organizational change management initiatives, comprehensive employee skill development programs, and cultural adaptation strategies. Technology teams are expanding with new specialized roles in cloud engineering, data science, artificial intelligence development, cybersecurity specializations, and digital product management. Training programs ensure workforce readiness for emerging technologies and evolving customer expectations.

Financial Investment and Resource Allocation:
Total technology investment over the three-year period amounts to $3.2 billion, strategically allocated across infrastructure modernization (45%), AI and advanced analytics capabilities (25%), customer experience platform development (20%), and cybersecurity enhancement initiatives (10%). Return on investment projections indicate significant operational efficiency improvements and enhanced customer satisfaction metrics.

Performance Metrics and Success Criteria:
Key performance indicators include customer satisfaction scores, digital adoption rates, operational efficiency improvements, regulatory compliance maintenance, system reliability metrics, and competitive positioning assessments. Quarterly business reviews ensure continuous progress tracking, strategy adjustments as needed, and alignment with evolving market conditions and customer expectations.

Risk Assessment and Mitigation Strategies:
Comprehensive risk assessment encompasses technology risks, operational risks, regulatory compliance risks, and market competition risks. Mitigation strategies include redundant system architectures, comprehensive backup and recovery procedures, regular security audits, and continuous monitoring of emerging threats and regulatory changes.

Future Innovation Pipeline:
Beyond the three-year implementation timeline, DBS Bank is investing in emerging technologies including quantum computing research, blockchain applications, Internet of Things integration, and advanced robotics for operational automation. Innovation labs in Singapore, Hong Kong, and India are exploring next-generation financial services applications and customer experience innovations.
"""
    return content

def test_configurable_chunking():
    """Test configurable chunking strategies"""
    print("🧪 Testing Configurable Chunking Strategies")
    print("=" * 60)
    
    try:
        from app.services.dynamic_chunk_sizing import get_dynamic_chunk_sizer
        from app.document_handlers.base import ExtractedChunk
        
        # Create chunk sizer instance
        chunk_sizer = get_dynamic_chunk_sizer()
        
        print(f"✅ Model: {chunk_sizer.model_name}")
        print(f"✅ Context: {chunk_sizer.context_limit:,} tokens (~{chunk_sizer.context_limit * 4:,} chars)")
        print(f"✅ Base optimal chunk size: {chunk_sizer.optimal_chunk_size:,} chars")
        
        # Create business document
        business_content = create_business_document()
        document_size = len(business_content)
        print(f"✅ Business document: {document_size:,} characters")
        
        # Split into realistic initial chunks
        chunk_size = 2500
        mock_chunks = []
        for i in range(0, len(business_content), chunk_size):
            chunk_content = business_content[i:i + chunk_size]
            if len(chunk_content.strip()) > 100:
                chunk = ExtractedChunk(
                    content=chunk_content,
                    metadata={'chunk_id': f'business_chunk_{i//chunk_size}'},
                    quality_score=0.8
                )
                mock_chunks.append(chunk)
        
        print(f"📝 Initial chunks: {len(mock_chunks)}")
        print(f"   Average size: {sum(len(c.content) for c in mock_chunks) // len(mock_chunks):,} chars")
        
        # Test different processing purposes
        processing_purposes = [
            ('knowledge_graph', '🧠 Knowledge Graph Extraction'),
            ('rag', '📚 RAG Processing'),
            ('general_processing', '📄 General Processing')
        ]
        
        results = {}
        
        for purpose, description in processing_purposes:
            print(f"\n{description}:")
            print("-" * 40)
            
            # Get configuration
            config = chunk_sizer.get_chunk_configuration('technology_strategy', purpose)
            print(f"   Strategy: {config['processing_strategy']}")
            print(f"   Target chunks: {config.get('target_chunks_per_document', 'N/A')}")
            print(f"   Max chunk size: {config['max_chunk_size']:,} chars")
            
            # Optimize chunks
            optimized_chunks = chunk_sizer.optimize_chunks(mock_chunks, 'technology_strategy', purpose)
            
            # Analyze results
            chunk_sizes = [len(c.content) for c in optimized_chunks]
            consolidation_ratio = len(mock_chunks) / len(optimized_chunks) if optimized_chunks else 0
            
            print(f"   Result: {len(mock_chunks)} → {len(optimized_chunks)} chunks")
            print(f"   Consolidation: {consolidation_ratio:.1f}:1")
            print(f"   Size range: {min(chunk_sizes):,} - {max(chunk_sizes):,} chars")
            print(f"   Average size: {sum(chunk_sizes) // len(chunk_sizes):,} chars")
            
            # Calculate context utilization
            total_content = sum(chunk_sizes)
            max_possible_context = chunk_sizer.context_limit * 4  # Convert tokens to chars
            utilization = (total_content / max_possible_context) * 100
            print(f"   Context utilization: {utilization:.1f}% of {chunk_sizer.context_limit:,} tokens")
            
            results[purpose] = {
                'strategy': config['processing_strategy'],
                'initial_chunks': len(mock_chunks),
                'optimized_chunks': len(optimized_chunks),
                'consolidation_ratio': consolidation_ratio,
                'size_range': [min(chunk_sizes), max(chunk_sizes)],
                'average_size': sum(chunk_sizes) // len(chunk_sizes),
                'context_utilization_percent': utilization,
                'target_chunks': config.get('target_chunks_per_document', 'N/A')
            }
        
        # Summary comparison
        print(f"\n🎯 STRATEGY COMPARISON:")
        print("=" * 60)
        
        for purpose, data in results.items():
            utilization = data['context_utilization_percent']
            chunks = data['optimized_chunks']
            avg_size = data['average_size']
            
            if purpose == 'knowledge_graph':
                print(f"🧠 KG Extraction: {chunks} chunks, avg {avg_size:,} chars ({utilization:.1f}% context)")
                print(f"   → Optimized for entity extraction quality")
            elif purpose == 'rag':
                print(f"📚 RAG Processing: {chunks} chunks, avg {avg_size:,} chars ({utilization:.1f}% context)")
                print(f"   → Balanced for retrieval and context")
            else:
                print(f"📄 General Processing: {chunks} chunks, avg {avg_size:,} chars ({utilization:.1f}% context)")
                print(f"   → Maximizes model context utilization")
        
        # Key insights
        print(f"\n💡 KEY INSIGHTS:")
        print("-" * 40)
        print(f"• Your 256k model can handle {document_size:,} chars in ONE chunk")
        print(f"• Knowledge Graph extraction uses smaller chunks for better entity recognition")
        print(f"• General processing can utilize full context capacity")
        print(f"• RAG processing balances retrieval quality with context size")
        
        # Save results
        output = {
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'name': chunk_sizer.model_name,
                'context_limit': chunk_sizer.context_limit,
                'optimal_chunk_size': chunk_sizer.optimal_chunk_size
            },
            'document_info': {
                'size_chars': document_size,
                'initial_chunks': len(mock_chunks)
            },
            'processing_strategies': results
        }
        
        with open('configurable_chunking_results.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✅ Results saved to: configurable_chunking_results.json")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_configurable_chunking()
    sys.exit(0 if success else 1)