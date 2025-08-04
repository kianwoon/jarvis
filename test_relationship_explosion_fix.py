#!/usr/bin/env python3
"""
EMERGENCY TEST: Validate that relationship explosion fixes actually work
This script tests that the fixes reduce relationships instead of making them worse.
"""

import asyncio
import json
from typing import Dict, List, Any
from app.services.llm_knowledge_extractor import LLMKnowledgeExtractor

# Sample business document that was causing the explosion
SAMPLE_BUSINESS_TEXT = """
DBS Bank's Digital Transformation Journey in Singapore

In Q2 2024, DBS Bank CEO Piyush Gupta announced a comprehensive $2.5 billion digital transformation 
initiative targeting the Southeast Asian market. The bank partnered with Microsoft Azure and AWS 
to implement cloud-native architecture across their retail banking division.

Key executives involved include:
- CTO David Gledhill overseeing technology implementation
- CFO Sok Hui Chng managing the $2.5B budget allocation
- CDO Paul Cobban leading digital customer experience

The transformation includes:
1. Migration of core banking systems to Kubernetes-based microservices
2. Implementation of real-time fraud detection using TensorFlow models
3. Launch of POSB DigiBank mobile app targeting 5 million users by Q4 2024
4. Partnership with Grab Financial for digital wallet integration

Financial targets:
- 30% reduction in operational costs by FY2025
- Revenue growth of $500M from digital channels
- Customer acquisition: 2M new digital-only accounts

The bank faces competition from Sea Group's SeaBank and Ant Financial's regional expansion.
Strategic partnerships include collaborations with Singtel for 5G banking services and 
integration with PayNow instant payment system.

Regional expansion plans cover Jakarta, Mumbai, and Hong Kong markets with localized 
digital banking solutions. The initiative aligns with MAS (Monetary Authority of Singapore) 
guidelines for digital banking licenses issued in 2023.
"""

class RelationshipExplosionValidator:
    """Validates that the emergency fixes actually prevent relationship explosion"""
    
    def __init__(self):
        self.extractor = LLMKnowledgeExtractor()
        
    async def test_extraction_with_fixes(self):
        """Test extraction with the emergency fixes applied"""
        print("=" * 80)
        print("TESTING RELATIONSHIP EXPLOSION FIXES")
        print("=" * 80)
        
        try:
            # Extract with current (fixed) system
            result = await self.extractor.extract_knowledge(SAMPLE_BUSINESS_TEXT)
            
            entities_count = len(result.entities)
            relationships_count = len(result.relationships)
            ratio = relationships_count / max(entities_count, 1)
            
            print(f"\n📊 EXTRACTION RESULTS WITH FIXES:")
            print(f"   Entities: {entities_count}")
            print(f"   Relationships: {relationships_count}")
            print(f"   Ratio: {ratio:.1f} relationships per entity")
            
            # Analyze relationship types
            relationship_types = {}
            inference_methods = {}
            
            for rel in result.relationships:
                rel_type = rel.relationship_type
                relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
                
                # Check if this is a co-occurrence relationship
                if hasattr(rel, 'properties') and rel.properties:
                    method = rel.properties.get('inference_method', 'llm_extracted')
                    inference_methods[method] = inference_methods.get(method, 0) + 1
            
            print(f"\n🔍 RELATIONSHIP TYPE BREAKDOWN:")
            for rel_type, count in sorted(relationship_types.items()):
                print(f"   {rel_type}: {count}")
                
            print(f"\n🔍 INFERENCE METHOD BREAKDOWN:")
            for method, count in sorted(inference_methods.items()):
                print(f"   {method}: {count}")
            
            # Validate that fixes are working
            print(f"\n✅ VALIDATION CHECKS:")
            
            # Check 1: Total relationships should be <= 150 (nuclear cap)
            nuclear_cap_check = relationships_count <= 150
            print(f"   Nuclear Cap (≤150): {'✅ PASS' if nuclear_cap_check else '❌ FAIL'} ({relationships_count})")
            
            # Check 2: Ratio should be reasonable (≤10 per entity)
            ratio_check = ratio <= 10
            print(f"   Reasonable Ratio (≤10): {'✅ PASS' if ratio_check else '❌ FAIL'} ({ratio:.1f})")
            
            # Check 3: Co-occurrence relationships should be limited
            cooccurrence_count = inference_methods.get('cooccurrence', 0) + inference_methods.get('enhanced_cooccurrence', 0)
            cooccurrence_check = cooccurrence_count <= 75  # Max 50 + 25 from both functions
            print(f"   Co-occurrence Limit (≤75): {'✅ PASS' if cooccurrence_check else '❌ FAIL'} ({cooccurrence_count})")
            
            # Check 4: Should not have exponential explosion pattern
            explosion_check = relationships_count < 1000  # Much less than the 3764 we had
            print(f"   No Explosion (<1000): {'✅ PASS' if explosion_check else '❌ FAIL'} ({relationships_count})")
            
            # Overall assessment
            all_checks_pass = nuclear_cap_check and ratio_check and cooccurrence_check and explosion_check
            
            print(f"\n🎯 OVERALL ASSESSMENT:")
            if all_checks_pass:
                print("   ✅ ALL CHECKS PASS - Emergency fixes are working!")
            else:
                print("   ❌ SOME CHECKS FAILED - Additional fixes may be needed")
                
            # Show quality of top relationships
            print(f"\n🏆 TOP 10 HIGHEST CONFIDENCE RELATIONSHIPS:")
            sorted_rels = sorted(result.relationships, key=lambda r: r.confidence, reverse=True)
            for i, rel in enumerate(sorted_rels[:10], 1):
                method = rel.properties.get('inference_method', 'llm') if rel.properties else 'llm'
                print(f"   {i:2d}. {rel.source_entity} -[{rel.relationship_type}]-> {rel.target_entity}")
                print(f"       Confidence: {rel.confidence:.2f}, Method: {method}")
            
            return {
                'entities': entities_count,
                'relationships': relationships_count,
                'ratio': ratio,
                'all_checks_pass': all_checks_pass,
                'cooccurrence_count': cooccurrence_count
            }
            
        except Exception as e:
            print(f"❌ ERROR during extraction: {e}")
            return None
    
    def show_historical_comparison(self, current_results):
        """Show comparison with the historical explosion"""
        print(f"\n📈 HISTORICAL COMPARISON:")
        print(f"   {'Stage':<20} {'Entities':<10} {'Relationships':<15} {'Ratio':<8} {'Status'}")
        print(f"   {'-'*20} {'-'*10} {'-'*15} {'-'*8} {'-'*20}")
        print(f"   {'Original':<20} {'30':<10} {'244':<15} {'8.1':<8} {'Baseline'}")
        print(f"   {'Fix Attempt 1':<20} {'73':<10} {'1,238':<15} {'17.0':<8} {'❌ Made worse'}")
        print(f"   {'Fix Attempt 2':<20} {'122':<10} {'3,764':<15} {'31.0':<8} {'💥 Explosion'}")
        
        if current_results:
            status = "✅ Fixed" if current_results['all_checks_pass'] else "⚠️ Partial fix"
            print(f"   {'Current (Fixed)':<20} {current_results['entities']:<10} {current_results['relationships']:<15} {current_results['ratio']:<8.1f} {status}")
        
        print(f"\n💡 KEY INSIGHTS:")
        print(f"   - The explosion was caused by DUPLICATE co-occurrence generation")
        print(f"   - O(n²) entity pair combinations with 122 entities = 7,381 potential pairs") 
        print(f"   - Low co-occurrence thresholds (5-6) created thousands of relationships")
        print(f"   - Multiple generation phases accumulated relationships")
        print(f"   - Nuclear caps and disabled duplicate generation should fix the issue")

async def main():
    """Run the explosion fix validation"""
    validator = RelationshipExplosionValidator()
    
    print("🚑 EMERGENCY VALIDATION: Testing relationship explosion fixes")
    print("This test validates that the fixes prevent exponential relationship growth\n")
    
    # Test the fixed extraction
    current_results = await validator.test_extraction_with_fixes()
    
    # Show historical comparison
    validator.show_historical_comparison(current_results)
    
    print(f"\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    
    if current_results and current_results['all_checks_pass']:
        print("""
✅ SUCCESS: The emergency fixes appear to be working!

Key fixes applied:
1. 🚫 DISABLED duplicate co-occurrence generation (main culprit)
2. 🚑 Added nuclear hard cap of 150 relationships (cannot be bypassed)
3. 📊 Increased co-occurrence thresholds from 5-6 to 25-30
4. 🛡️ Added per-function caps (25 and 50 relationships max)

The relationship explosion has been contained. The system should now generate
a reasonable number of high-quality relationships instead of thousands of
low-quality co-occurrence relationships.
        """)
    else:
        print("""
⚠️ PARTIAL SUCCESS: Some improvements but additional fixes may be needed.

If relationships are still too high:
1. Further increase co-occurrence thresholds
2. Add more aggressive filtering
3. Reduce the nuclear cap from 150 to 100
4. Add entity limits to prevent O(n²) explosion
        """)

if __name__ == "__main__":
    asyncio.run(main())