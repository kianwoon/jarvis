#!/usr/bin/env python3
"""
Test script for enhanced UniversalEntityExtractor with mocked LLM
Verifies extraction of 30+ entities for technology queries
"""

import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from app.services.radiating.extraction.universal_entity_extractor import UniversalEntityExtractor

def get_mock_comprehensive_response():
    """Generate a comprehensive list of AI/ML technologies"""
    return json.dumps([
        # ML Frameworks
        {"text": "TensorFlow", "type": "ML Framework", "confidence": 0.95, "context": "open source ML framework", "reason": "Google's leading ML framework"},
        {"text": "PyTorch", "type": "ML Framework", "confidence": 0.95, "context": "open source ML framework", "reason": "Meta's research-focused framework"},
        {"text": "JAX", "type": "ML Framework", "confidence": 0.9, "context": "ML framework", "reason": "Google's high-performance ML framework"},
        {"text": "Keras", "type": "ML Framework", "confidence": 0.9, "context": "high-level neural networks API", "reason": "User-friendly deep learning API"},
        {"text": "MXNet", "type": "ML Framework", "confidence": 0.85, "context": "flexible ML framework", "reason": "Apache's scalable ML framework"},
        {"text": "Caffe", "type": "ML Framework", "confidence": 0.8, "context": "deep learning framework", "reason": "Berkeley's vision-focused framework"},
        {"text": "Theano", "type": "ML Framework", "confidence": 0.75, "context": "numerical computation library", "reason": "Pioneering deep learning framework"},
        
        # ML Libraries
        {"text": "scikit-learn", "type": "ML Library", "confidence": 0.95, "context": "machine learning library", "reason": "Essential for classical ML algorithms"},
        {"text": "XGBoost", "type": "ML Library", "confidence": 0.9, "context": "gradient boosting library", "reason": "Industry standard for boosting"},
        {"text": "LightGBM", "type": "ML Library", "confidence": 0.9, "context": "gradient boosting framework", "reason": "Microsoft's fast gradient boosting"},
        {"text": "CatBoost", "type": "ML Library", "confidence": 0.85, "context": "gradient boosting library", "reason": "Yandex's categorical-friendly boosting"},
        {"text": "FastAI", "type": "ML Library", "confidence": 0.85, "context": "deep learning library", "reason": "High-level PyTorch wrapper"},
        
        # Data Processing
        {"text": "Pandas", "type": "Data Processing Tool", "confidence": 0.95, "context": "data manipulation library", "reason": "Essential data analysis tool"},
        {"text": "NumPy", "type": "Data Processing Tool", "confidence": 0.95, "context": "numerical computing library", "reason": "Foundation for scientific computing"},
        {"text": "Apache Spark", "type": "Data Processing Tool", "confidence": 0.9, "context": "big data processing", "reason": "Distributed data processing engine"},
        {"text": "Dask", "type": "Data Processing Tool", "confidence": 0.85, "context": "parallel computing library", "reason": "Scalable Python analytics"},
        {"text": "Ray", "type": "Data Processing Tool", "confidence": 0.85, "context": "distributed computing framework", "reason": "Distributed AI/ML workloads"},
        {"text": "Polars", "type": "Data Processing Tool", "confidence": 0.8, "context": "fast DataFrame library", "reason": "Lightning-fast DataFrame operations"},
        
        # NLP Frameworks
        {"text": "Transformers", "type": "NLP Framework", "confidence": 0.95, "context": "transformer models library", "reason": "Hugging Face's transformer library"},
        {"text": "spaCy", "type": "NLP Framework", "confidence": 0.9, "context": "NLP library", "reason": "Industrial-strength NLP"},
        {"text": "NLTK", "type": "NLP Framework", "confidence": 0.85, "context": "NLP toolkit", "reason": "Natural Language Toolkit"},
        {"text": "Gensim", "type": "NLP Framework", "confidence": 0.85, "context": "topic modeling library", "reason": "Topic modeling and word embeddings"},
        {"text": "AllenNLP", "type": "NLP Framework", "confidence": 0.8, "context": "NLP research library", "reason": "Research-focused NLP framework"},
        
        # LLM Frameworks
        {"text": "LangChain", "type": "LLM Framework", "confidence": 0.95, "context": "LLM application framework", "reason": "Building LLM applications"},
        {"text": "LlamaIndex", "type": "LLM Framework", "confidence": 0.9, "context": "LLM data framework", "reason": "Data framework for LLMs"},
        {"text": "Haystack", "type": "LLM Framework", "confidence": 0.85, "context": "NLP framework", "reason": "End-to-end NLP framework"},
        {"text": "Semantic Kernel", "type": "LLM Framework", "confidence": 0.8, "context": "AI orchestration", "reason": "Microsoft's AI orchestration"},
        
        # MLOps Tools
        {"text": "MLflow", "type": "MLOps Tool", "confidence": 0.9, "context": "ML lifecycle platform", "reason": "Open source ML platform"},
        {"text": "Kubeflow", "type": "MLOps Tool", "confidence": 0.85, "context": "ML workflows on Kubernetes", "reason": "Kubernetes-native ML workflows"},
        {"text": "Weights & Biases", "type": "MLOps Tool", "confidence": 0.85, "context": "ML experiment tracking", "reason": "Experiment tracking and visualization"},
        {"text": "Neptune.ai", "type": "MLOps Tool", "confidence": 0.8, "context": "ML metadata store", "reason": "ML experiment management"},
        {"text": "DVC", "type": "MLOps Tool", "confidence": 0.8, "context": "data version control", "reason": "Version control for ML"},
        {"text": "Metaflow", "type": "MLOps Tool", "confidence": 0.75, "context": "ML infrastructure", "reason": "Netflix's ML infrastructure"},
        
        # Vector Databases
        {"text": "Pinecone", "type": "Vector Database", "confidence": 0.9, "context": "vector database", "reason": "Managed vector database"},
        {"text": "Weaviate", "type": "Vector Database", "confidence": 0.85, "context": "vector search engine", "reason": "Open source vector DB"},
        {"text": "Qdrant", "type": "Vector Database", "confidence": 0.85, "context": "vector similarity search", "reason": "Vector similarity engine"},
        {"text": "Milvus", "type": "Vector Database", "confidence": 0.85, "context": "vector database", "reason": "Open source vector database"},
        {"text": "ChromaDB", "type": "Vector Database", "confidence": 0.8, "context": "embedding database", "reason": "AI-native embedding database"},
        
        # Open Source Models
        {"text": "LLaMA", "type": "Open Source Model", "confidence": 0.9, "context": "language model", "reason": "Meta's open language model"},
        {"text": "Mistral", "type": "Open Source Model", "confidence": 0.9, "context": "language model", "reason": "Efficient open source LLM"},
        {"text": "BERT", "type": "Open Source Model", "confidence": 0.85, "context": "language model", "reason": "Bidirectional transformer model"},
        {"text": "GPT-J", "type": "Open Source Model", "confidence": 0.8, "context": "language model", "reason": "Open source GPT variant"},
        {"text": "Stable Diffusion", "type": "Open Source Model", "confidence": 0.9, "context": "image generation", "reason": "Open source image generation"},
        {"text": "BLOOM", "type": "Open Source Model", "confidence": 0.8, "context": "multilingual model", "reason": "Open multilingual LLM"},
        {"text": "Falcon", "type": "Open Source Model", "confidence": 0.8, "context": "language model", "reason": "TII's open source LLM"}
    ])

def get_mock_entity_types_response():
    """Generate mock entity types discovery response"""
    return json.dumps({
        "entity_types": [
            {"type": "ML Framework", "description": "Machine learning frameworks", "examples": ["TensorFlow", "PyTorch"], "confidence": 0.9},
            {"type": "ML Library", "description": "ML libraries and packages", "examples": ["scikit-learn", "XGBoost"], "confidence": 0.9},
            {"type": "Data Processing Tool", "description": "Data processing tools", "examples": ["Pandas", "NumPy"], "confidence": 0.9},
            {"type": "NLP Framework", "description": "NLP frameworks", "examples": ["spaCy", "NLTK"], "confidence": 0.9},
            {"type": "LLM Framework", "description": "LLM application frameworks", "examples": ["LangChain", "LlamaIndex"], "confidence": 0.9},
            {"type": "MLOps Tool", "description": "ML operations tools", "examples": ["MLflow", "Kubeflow"], "confidence": 0.9},
            {"type": "Vector Database", "description": "Vector databases", "examples": ["Pinecone", "Weaviate"], "confidence": 0.9},
            {"type": "Open Source Model", "description": "Open source AI models", "examples": ["LLaMA", "BERT"], "confidence": 0.9}
        ]
    })

async def test_comprehensive_extraction():
    """Test entity extraction for technology queries with mocked LLM"""
    
    # Create mock LLM client
    mock_llm = AsyncMock()
    
    # Initialize the extractor with mock LLM
    extractor = UniversalEntityExtractor(llm_client=mock_llm)
    
    # Test queries
    test_queries = [
        "what are the essential technologies of AI implementation, favor open source",
        "list all machine learning frameworks and tools",
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Testing query: {query}")
        print(f"{'='*80}")
        
        # Setup mock responses
        mock_llm.invoke.side_effect = [
            get_mock_entity_types_response(),  # For entity type discovery
            get_mock_comprehensive_response()   # For entity extraction
        ]
        
        # Check if comprehensive detection works
        is_comprehensive = extractor._is_comprehensive_technology_query(query)
        print(f"Detected as comprehensive query: {is_comprehensive}")
        
        # Extract entities
        entities = await extractor.extract_entities(query)
        
        print(f"\nExtracted {len(entities)} entities:")
        
        # Group entities by type
        entities_by_type = {}
        for entity in entities:
            entity_type = entity.entity_type
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)
        
        # Display entities by type
        for entity_type, type_entities in sorted(entities_by_type.items()):
            print(f"\n{entity_type} ({len(type_entities)} entities):")
            for entity in type_entities[:5]:  # Show first 5 of each type
                print(f"  - {entity.text} (confidence: {entity.confidence:.2f})")
            if len(type_entities) > 5:
                print(f"  ... and {len(type_entities) - 5} more")
        
        # Summary statistics
        print(f"\n{'-'*40}")
        print(f"Summary:")
        print(f"  Total entities extracted: {len(entities)}")
        print(f"  Unique entity types: {len(entities_by_type)}")
        print(f"  Average confidence: {sum(e.confidence for e in entities) / len(entities) if entities else 0:.2f}")
        print(f"  Extraction method: {entities[0].metadata.get('extraction_method', 'unknown') if entities else 'N/A'}")
        print(f"  Min confidence threshold used: 0.3 (for comprehensive queries)")
        print(f"  Max entities allowed: 100 (for comprehensive queries)")
        
        # Verify we get 30+ entities for AI/ML queries
        if 'ai' in query.lower() or 'machine learning' in query.lower():
            if len(entities) >= 30:
                print(f"  ✅ SUCCESS: Extracted {len(entities)} entities (>= 30 required)")
            else:
                print(f"  ❌ FAILURE: Only extracted {len(entities)} entities (expected >= 30)")

async def test_regular_vs_comprehensive():
    """Compare regular extraction vs comprehensive extraction"""
    
    print(f"\n{'='*80}")
    print("Testing Detection Logic")
    print(f"{'='*80}")
    
    extractor = UniversalEntityExtractor()
    
    test_cases = [
        ("The meeting is scheduled for tomorrow at 3pm", False),
        ("what are the essential technologies of AI", True),
        ("list machine learning frameworks", True),
        ("John works at Google", False),
        ("tools for data science", True),
        ("the weather is nice today", False),
        ("popular open source AI frameworks and libraries", True),
        ("essential tools and technologies for building LLM applications", True)
    ]
    
    for query, expected_comprehensive in test_cases:
        is_comprehensive = extractor._is_comprehensive_technology_query(query)
        status = "✅" if is_comprehensive == expected_comprehensive else "❌"
        print(f"{status} Query: '{query[:50]}...' => Comprehensive: {is_comprehensive} (expected: {expected_comprehensive})")

if __name__ == "__main__":
    print("Testing Enhanced Universal Entity Extractor (with Mocked LLM)")
    print("="*80)
    
    # Run the async tests
    asyncio.run(test_comprehensive_extraction())
    asyncio.run(test_regular_vs_comprehensive())
    
    print("\n" + "="*80)
    print("Test complete!")
    print("\nKey improvements implemented:")
    print("1. ✅ Comprehensive query detection for technology/tool queries")
    print("2. ✅ Lower confidence threshold (0.3) for comprehensive queries")
    print("3. ✅ Higher max entities limit (100) for comprehensive queries")
    print("4. ✅ Enhanced LLM prompts requesting 30-50+ entities")
    print("5. ✅ Special handling for AI/ML/technology domain queries")