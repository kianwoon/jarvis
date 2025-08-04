#!/usr/bin/env python3

"""
Test script to verify aggressive knowledge graph relationship filtering.
This test validates that the relationship count is reduced from 1200+ to 150-300.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from services.llm_knowledge_extractor import LLMKnowledgeExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample business document that would generate many relationships
SAMPLE_BUSINESS_DOCUMENT = """
DBS Bank Strategic Digital Transformation Report 2024

Executive Summary
DBS Bank, Southeast Asia's leading financial institution, continues its digital transformation journey through innovative technology partnerships and strategic acquisitions. CEO Piyush Gupta leads the organization's expansion into new markets while maintaining strong competitive advantages in mobile banking and digital payments.

Technology Infrastructure
The bank operates on AWS cloud infrastructure, utilizing Java microservices and React frontend applications. The core banking platform integrates with multiple payment gateways including PayLah!, PAYM, and international SWIFT networks. The technology team, led by CTO Jane Smith, manages over 200 APIs serving millions of daily transactions.

Strategic Partnerships
DBS collaborates with Alibaba Cloud for China market expansion, Microsoft Azure for AI capabilities, and Google Cloud for analytics. The bank's partnership with Singapore's GovTech enables seamless integration with national digital identity systems. Investment in blockchain technology supports trade finance automation and smart contract implementations.

Financial Performance
Q3 2024 revenue reached $4.2 billion, representing 15% growth year-over-year. The digital banking segment contributed $1.8 billion, with mobile app usage increasing 40%. Cost savings of $120 million were achieved through automation and AI implementation. The bank targets $500 million in annual cost reductions by 2025.

Market Position
DBS maintains market leadership in Singapore with 45% market share in consumer banking. The bank competes directly with OCBC Bank, UOB, and regional players like HSBC and Standard Chartered. International expansion focuses on Indonesia, Thailand, and Hong Kong markets where DBS operates subsidiaries and joint ventures.

Innovation Initiatives
The bank's innovation lab develops AI-powered credit scoring, real-time fraud detection, and personalized wealth management solutions. Machine learning algorithms analyze customer transaction patterns to provide predictive insights. The digital wallet PayLah! processes over 1 million transactions daily across Southeast Asia.

Regulatory Compliance
DBS works closely with Monetary Authority of Singapore (MAS) on regulatory compliance and digital banking frameworks. The bank implements comprehensive cybersecurity measures including multi-factor authentication, encryption, and continuous monitoring systems. ESG initiatives support sustainable finance and carbon-neutral banking operations.

Future Outlook
The bank plans to expand its digital capabilities through continued investment in cloud computing, artificial intelligence, and blockchain technology. Strategic partnerships with fintech companies will accelerate innovation in payment processing, lending automation, and customer experience enhancement. DBS aims to become the leading digital bank in Asia by 2030.
"""

async def test_aggressive_filtering():
    """Test that aggressive filtering reduces relationships to target range"""
    
    print("🚨 TESTING AGGRESSIVE KNOWLEDGE GRAPH FILTERING")
    print("=" * 60)
    
    # Initialize the extractor
    extractor = LLMKnowledgeExtractor()
    
    # Extract knowledge graph
    print(f"📊 Processing document ({len(SAMPLE_BUSINESS_DOCUMENT):,} characters)")
    start_time = datetime.now()
    
    try:
        result = await extractor.extract_with_llm(
            text=SAMPLE_BUSINESS_DOCUMENT,
            context={"analysis_type": "strategic", "domain": "banking"},
            domain_hints=["banking", "fintech", "digital transformation"]
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Analyze results
        entity_count = len(result.entities)
        relationship_count = len(result.relationships)
        relationships_per_entity = relationship_count / max(entity_count, 1)
        
        print(f"⏱️  Processing time: {processing_time:.1f} seconds")
        print(f"📊 Extraction Results:")
        print(f"   Entities: {entity_count}")
        print(f"   Relationships: {relationship_count}")
        print(f"   Ratio: {relationships_per_entity:.1f} relationships per entity")
        print(f"   Confidence: {result.confidence_score:.3f}")
        
        # Validation checks
        success_checks = []
        
        # Check 1: Relationship count within target range (150-300)
        rel_count_ok = 150 <= relationship_count <= 300
        success_checks.append(rel_count_ok)
        status = "✅ PASS" if rel_count_ok else "❌ FAIL"
        print(f"   {status} Relationship count: {relationship_count} (target: 150-300)")
        
        # Check 2: Relationships per entity ratio (2-4)
        ratio_ok = 2.0 <= relationships_per_entity <= 4.0
        success_checks.append(ratio_ok)
        status = "✅ PASS" if ratio_ok else "❌ FAIL"
        print(f"   {status} Relationships per entity: {relationships_per_entity:.1f} (target: 2.0-4.0)")
        
        # Check 3: No generic relationship types
        generic_types = ['RELATED_TO', 'ASSOCIATED_WITH', 'CONNECTED_TO', 'MENTIONED_WITH']
        generic_found = [r for r in result.relationships if r.relationship_type in generic_types]
        no_generic_ok = len(generic_found) == 0
        success_checks.append(no_generic_ok)
        status = "✅ PASS" if no_generic_ok else "❌ FAIL"
        print(f"   {status} No generic relationship types: {len(generic_found)} found")
        
        # Check 4: High confidence relationships (>=0.7)
        high_conf_rels = [r for r in result.relationships if float(r.confidence or 0) >= 0.7]
        high_conf_ratio = len(high_conf_rels) / max(relationship_count, 1)
        high_conf_ok = high_conf_ratio >= 0.5  # At least 50% should be high confidence
        success_checks.append(high_conf_ok)
        status = "✅ PASS" if high_conf_ok else "❌ FAIL"
        print(f"   {status} High confidence ratio: {high_conf_ratio:.1%} (target: >=50%)")
        
        # Show relationship type distribution
        print(f"\n📊 Relationship Type Distribution:")
        rel_type_counts = {}
        for rel in result.relationships:
            rel_type = rel.relationship_type
            rel_type_counts[rel_type] = rel_type_counts.get(rel_type, 0) + 1
        
        for rel_type, count in sorted(rel_type_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {rel_type}: {count}")
        
        # Show sample high-quality relationships
        print(f"\n📊 Sample High-Quality Relationships:")
        high_quality_rels = sorted(result.relationships, key=lambda r: float(r.confidence or 0), reverse=True)[:5]
        for i, rel in enumerate(high_quality_rels, 1):
            print(f"   {i}. {rel.source_entity} --[{rel.relationship_type}]--> {rel.target_entity} (conf: {rel.confidence:.2f})")
        
        # Overall result
        all_passed = all(success_checks)
        overall_status = "✅ SUCCESS" if all_passed else "❌ FAILED"
        print(f"\n{overall_status}: Aggressive filtering test")
        
        if all_passed:
            print("🎉 Knowledge graph filtering is working correctly!")
            print("   • Relationship count reduced to target range")
            print("   • Relationships per entity within optimal ratio")
            print("   • Generic relationship types eliminated")
            print("   • High confidence relationships prioritized")
        else:
            print("⚠️  Some filtering criteria not met. Review implementation.")
            
        return all_passed
        
    except Exception as e:
        print(f"❌ ERROR during extraction: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_aggressive_filtering())
    sys.exit(0 if success else 1)