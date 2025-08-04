#!/usr/bin/env python3
"""
Debug Chunk Processing Issues

This script diagnoses why chunks are being filtered out during graph processing.
"""

import asyncio
import time
from pathlib import Path

# Import document processing services
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.document_handlers.base import ExtractedChunk
from app.document_handlers.graph_processor import get_graph_document_processor
from app.services.dynamic_chunk_sizing import get_dynamic_chunk_sizer, get_optimal_chunk_config
from app.core.knowledge_graph_settings_cache import (
    get_knowledge_graph_settings, 
    detect_business_document, 
    get_business_optimized_settings
)

class ChunkProcessingDebugger:
    """Debug chunk processing issues"""
    
    def __init__(self):
        self.graph_processor = get_graph_document_processor()
        self.chunk_sizer = get_dynamic_chunk_sizer()
        
        # Initialize text splitter
        kg_settings = get_knowledge_graph_settings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=kg_settings.get('chunk_size', 1000),
            chunk_overlap=kg_settings.get('chunk_overlap', 200),
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Test document path
        self.test_document = "/Users/kianwoonwong/Downloads/jarvis/DBS Technology Strategy (Confidential).pdf"
    
    async def load_and_chunk_document(self, file_path: str) -> list:
        """Load document and create chunks using PyPDFLoader"""
        try:
            # Load document using PyPDFLoader
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Generate document ID for chunk metadata
            document_id = f"debug_{Path(file_path).stem}_{int(time.time())}"
            
            # Split into chunks
            chunks = []
            for i, doc in enumerate(documents):
                text_chunks = self.text_splitter.split_text(doc.page_content)
                
                for j, chunk_text in enumerate(text_chunks):
                    chunk_id = f"{document_id}_page{i}_chunk{j}"
                    chunk = ExtractedChunk(
                        content=chunk_text,
                        metadata={
                            'chunk_id': chunk_id,
                            'document_id': document_id,
                            'page_number': i,
                            'chunk_index': j,
                            'source': file_path,
                            'total_chunks': len(text_chunks)
                        },
                        quality_score=1.0
                    )
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            print(f"❌ Failed to load and chunk document: {e}")
            return []
    
    async def debug_chunk_filtering(self):
        """Debug why chunks are being filtered out"""
        print("🔍 DEBUGGING CHUNK PROCESSING")
        print("="*60)
        
        # Load chunks
        print("📄 Loading document...")
        chunks = await self.load_and_chunk_document(self.test_document)
        print(f"✅ Loaded {len(chunks)} chunks")
        
        if not chunks:
            print("❌ No chunks loaded, cannot debug filtering")
            return
        
        # Show chunk info
        print(f"\n📊 CHUNK STATISTICS:")
        lengths = [len(chunk.content.strip()) for chunk in chunks]
        print(f"   Count: {len(chunks)}")
        print(f"   Length range: {min(lengths)} - {max(lengths)} chars")
        print(f"   Average length: {sum(lengths) // len(lengths)} chars")
        
        # Show sample chunks
        print(f"\n📝 SAMPLE CHUNKS:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"   Chunk {i+1}: {len(chunk.content)} chars")
            print(f"      Preview: {chunk.content[:100]}...")
            print(f"      Quality score: {chunk.quality_score}")
        
        # Test business document detection
        print(f"\n🏢 BUSINESS DOCUMENT DETECTION:")
        filename = Path(self.test_document).name
        sample_content = ' '.join(chunk.content[:500] for chunk in chunks[:3])
        
        is_business_filename = detect_business_document(filename=filename)
        is_business_content = detect_business_document(content=sample_content)
        
        print(f"   Filename detection: {is_business_filename}")
        print(f"   Content detection: {is_business_content}")
        
        # Test graph processor filtering
        print(f"\n🧠 GRAPH PROCESSOR ANALYSIS:")
        print(f"   Processor model: {self.graph_processor.chunk_sizer.model_name}")
        print(f"   Context limit: {self.graph_processor.chunk_sizer.context_limit:,} tokens")
        print(f"   Min chunk length: {self.graph_processor.min_chunk_length}")
        print(f"   Max chunk length: {self.graph_processor.max_chunk_length}")
        
        # Test chunk filtering manually
        print(f"\n🔧 MANUAL CHUNK FILTERING TEST:")
        
        # Detect document type
        document_type = 'general'
        if is_business_filename or is_business_content:
            document_type = 'technology_strategy'
        
        print(f"   Document type: {document_type}")
        
        # Get chunk configuration
        config = get_optimal_chunk_config(document_type)
        print(f"   Chunk config strategy: {config['processing_strategy']}")
        print(f"   Max chunk size: {config['max_chunk_size']:,} chars")
        print(f"   Min chunk size: {config['min_chunk_size']:,} chars")
        
        # Manually filter chunks
        suitable_chunks = []
        filtered_out = []
        
        for i, chunk in enumerate(chunks):
            content_length = len(chunk.content.strip())
            
            if content_length < config['min_chunk_size']:
                filtered_out.append((i, chunk, f"Too short: {content_length} < {config['min_chunk_size']}"))
            elif content_length > config['max_chunk_size'] * 1.1:
                filtered_out.append((i, chunk, f"Too long: {content_length} > {config['max_chunk_size'] * 1.1}"))
            else:
                suitable_chunks.append((i, chunk))
        
        print(f"   Suitable chunks: {len(suitable_chunks)}")
        print(f"   Filtered out: {len(filtered_out)}")
        
        if filtered_out:
            print(f"\n❌ FILTERED OUT CHUNKS:")
            for i, chunk, reason in filtered_out[:5]:
                print(f"      Chunk {i}: {reason}")
                print(f"         Preview: {chunk.content[:50]}...")
        
        if suitable_chunks:
            print(f"\n✅ SUITABLE CHUNKS:")
            for i, chunk in suitable_chunks[:3]:
                print(f"      Chunk {i}: {len(chunk.content)} chars")
                print(f"         Preview: {chunk.content[:50]}...")
        
        # Test actual graph processor filtering
        print(f"\n🔬 ACTUAL GRAPH PROCESSOR FILTERING:")
        try:
            # Call the actual filtering method
            filtered_chunks = self.graph_processor._filter_chunks_for_graph_processing(chunks)
            print(f"   Graph processor result: {len(filtered_chunks)} suitable chunks")
            
            if len(filtered_chunks) == 0:
                print("   ❌ All chunks filtered out by graph processor!")
                
                # Check the exact filtering logic
                print(f"\n🔍 DETAILED FILTERING ANALYSIS:")
                print(f"   Graph processor min_chunk_length: {self.graph_processor.min_chunk_length}")
                print(f"   Graph processor max_chunk_length: {self.graph_processor.max_chunk_length}")
                
                for i, chunk in enumerate(chunks[:5]):
                    content_length = len(chunk.content.strip())
                    print(f"   Chunk {i}: {content_length} chars")
                    
                    if content_length < self.graph_processor.min_chunk_length:
                        print(f"      ❌ Below min: {content_length} < {self.graph_processor.min_chunk_length}")
                    elif content_length > self.graph_processor.max_chunk_length * 1.1:
                        print(f"      ❌ Above max: {content_length} > {self.graph_processor.max_chunk_length * 1.1}")
                    else:
                        print(f"      ✅ Within range")
            else:
                print(f"   ✅ {len(filtered_chunks)} chunks passed filtering")
                
        except Exception as e:
            print(f"   ❌ Error calling graph processor: {e}")
        
        # Test chunk optimization
        print(f"\n⚙️ CHUNK OPTIMIZATION TEST:")
        try:
            from app.services.dynamic_chunk_sizing import optimize_chunks_for_model
            optimized_chunks = optimize_chunks_for_model(chunks, document_type)
            print(f"   Original chunks: {len(chunks)}")
            print(f"   Optimized chunks: {len(optimized_chunks)}")
            
            if optimized_chunks:
                opt_lengths = [len(chunk.content) for chunk in optimized_chunks]
                print(f"   Optimized length range: {min(opt_lengths)} - {max(opt_lengths)} chars")
                print(f"   Optimized average: {sum(opt_lengths) // len(opt_lengths)} chars")
                
        except Exception as e:
            print(f"   ❌ Error during optimization: {e}")

async def main():
    """Main debug execution"""
    debugger = ChunkProcessingDebugger()
    await debugger.debug_chunk_filtering()

if __name__ == "__main__":
    asyncio.run(main())