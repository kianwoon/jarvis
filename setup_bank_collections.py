#!/usr/bin/env python3
"""
Setup script to create bank-specific collections
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000/api/v1"

# Bank-specific collections to create
BANK_COLLECTIONS = [
    {
        "collection_name": "regulatory_compliance",
        "collection_type": "regulatory_compliance",
        "description": "Banking regulations, compliance documents, and regulatory filings including Basel III/IV, Dodd-Frank, SOX, KYC/AML requirements",
        "metadata_schema": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "fields": [
                {"name": "regulations", "type": "string"},
                {"name": "compliance_year", "type": "string"},
                {"name": "regulatory_body", "type": "string"}
            ]
        },
        "search_config": {
            "strategy": "exact",
            "similarity_threshold": 0.8,
            "max_results": 20,
            "enable_bm25": True,
            "bm25_weight": 0.4
        },
        "access_config": {
            "restricted": True,
            "allowed_users": ["compliance_team", "legal_team", "risk_team"]
        }
    },
    {
        "collection_name": "product_documentation",
        "collection_type": "product_documentation",
        "description": "Banking products and services documentation including features, pricing, eligibility criteria, and terms",
        "metadata_schema": {
            "chunk_size": 1200,
            "chunk_overlap": 200,
            "fields": [
                {"name": "product_name", "type": "string"},
                {"name": "product_category", "type": "string"},
                {"name": "rates", "type": "string"}
            ]
        },
        "search_config": {
            "strategy": "balanced",
            "similarity_threshold": 0.7,
            "max_results": 15,
            "enable_bm25": True,
            "bm25_weight": 0.3
        },
        "access_config": {
            "restricted": False,
            "allowed_users": []
        }
    },
    {
        "collection_name": "risk_management",
        "collection_type": "risk_management",
        "description": "Risk assessments, control frameworks, mitigation strategies for credit, market, operational, and compliance risks",
        "metadata_schema": {
            "chunk_size": 1500,
            "chunk_overlap": 300,
            "fields": [
                {"name": "risk_types", "type": "string"},
                {"name": "risk_level", "type": "string"},
                {"name": "control_framework", "type": "string"}
            ]
        },
        "search_config": {
            "strategy": "comprehensive",
            "similarity_threshold": 0.65,
            "max_results": 25,
            "enable_bm25": True,
            "bm25_weight": 0.35
        },
        "access_config": {
            "restricted": True,
            "allowed_users": ["risk_team", "audit_team", "executives"]
        }
    },
    {
        "collection_name": "customer_support",
        "collection_type": "customer_support",
        "description": "Customer support documentation, FAQs, troubleshooting guides, and service scripts",
        "metadata_schema": {
            "chunk_size": 1000,
            "chunk_overlap": 150,
            "fields": [
                {"name": "issue_category", "type": "string"},
                {"name": "resolution_status", "type": "string"},
                {"name": "product_area", "type": "string"}
            ]
        },
        "search_config": {
            "strategy": "balanced",
            "similarity_threshold": 0.7,
            "max_results": 10,
            "enable_bm25": True,
            "bm25_weight": 0.4
        },
        "access_config": {
            "restricted": False,
            "allowed_users": []
        }
    },
    {
        "collection_name": "audit_reports",
        "collection_type": "audit_reports",
        "description": "Internal and external audit reports, inspection findings, and remediation tracking",
        "metadata_schema": {
            "chunk_size": 2000,
            "chunk_overlap": 400,
            "fields": [
                {"name": "audit_type", "type": "string"},
                {"name": "audit_period", "type": "string"},
                {"name": "audit_status", "type": "string"}
            ]
        },
        "search_config": {
            "strategy": "temporal",
            "similarity_threshold": 0.75,
            "max_results": 20,
            "enable_bm25": True,
            "bm25_weight": 0.3
        },
        "access_config": {
            "restricted": True,
            "allowed_users": ["audit_team", "executives", "board_members"]
        }
    },
    {
        "collection_name": "training_materials",
        "collection_type": "training_materials",
        "description": "Employee training materials, onboarding documentation, certification programs, and compliance training",
        "metadata_schema": {
            "chunk_size": 1500,
            "chunk_overlap": 250,
            "fields": [
                {"name": "training_type", "type": "string"},
                {"name": "target_audience", "type": "string"},
                {"name": "certification_required", "type": "string"}
            ]
        },
        "search_config": {
            "strategy": "comprehensive",
            "similarity_threshold": 0.65,
            "max_results": 15,
            "enable_bm25": True,
            "bm25_weight": 0.35
        },
        "access_config": {
            "restricted": False,
            "allowed_users": []
        }
    }
]

def create_collection(collection_data):
    """Create a single collection"""
    try:
        response = requests.post(
            f"{BASE_URL}/collections/",
            json=collection_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print(f"✅ Created collection: {collection_data['collection_name']}")
            return True
        elif response.status_code == 400 and "already exists" in response.text:
            print(f"⚠️  Collection already exists: {collection_data['collection_name']}")
            return True
        else:
            print(f"❌ Failed to create {collection_data['collection_name']}: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error creating {collection_data['collection_name']}: {str(e)}")
        return False

def main():
    """Main function to create all bank collections"""
    print("🏦 Setting up bank-specific collections...")
    print("=" * 50)
    
    success_count = 0
    
    for collection in BANK_COLLECTIONS:
        if create_collection(collection):
            success_count += 1
        print()
    
    print("=" * 50)
    print(f"✅ Successfully created/verified {success_count}/{len(BANK_COLLECTIONS)} collections")
    
    # Show current collections
    print("\n📊 Current collections:")
    try:
        response = requests.get(f"{BASE_URL}/collections/")
        if response.status_code == 200:
            collections = response.json()
            for col in collections:
                restricted = "🔒" if col.get("access_config", {}).get("restricted") else "🔓"
                print(f"  {restricted} {col['collection_name']} ({col['collection_type']}) - {col['description'][:60]}...")
        else:
            print("  Failed to fetch collections")
    except Exception as e:
        print(f"  Error fetching collections: {str(e)}")

if __name__ == "__main__":
    main()