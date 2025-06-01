#!/usr/bin/env python3
import sys
import asyncio
sys.path.append('.')

async def test_decomposer():
    from app.agents.task_decomposer import TaskDecomposer
    
    decomposer = TaskDecomposer()
    chunks = await decomposer.decompose_large_task(
        query='generate 50 interview questions',
        target_count=50
    )
    
    print('🔍 Decomposer Analysis:')
    print('=' * 40)
    print(f'Target count: 50')
    print(f'Number of chunks: {len(chunks)}')
    
    total_items = 0
    for i, chunk in enumerate(chunks):
        print(f'Chunk {chunk.chunk_number}: items {chunk.start_index}-{chunk.end_index} (size: {chunk.chunk_size})')
        total_items += chunk.chunk_size
    
    print(f'Total items across all chunks: {total_items}')
    
    if total_items != 50:
        print(f'❌ ISSUE: Expected 50 total, but chunks add up to {total_items}')
    else:
        print('✅ Chunking looks correct')

if __name__ == "__main__":
    try:
        asyncio.run(test_decomposer())
    except Exception as e:
        print(f'❌ Test failed: {e}')
        import traceback
        traceback.print_exc()