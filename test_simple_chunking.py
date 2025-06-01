#!/usr/bin/env python3
import math

def calculate_optimal_chunk_size(target_count: int) -> int:
    """Test the chunking logic"""
    base_chunk_size = 15  # Conservative default
    
    # Adjust based on total target count - IMPROVED LOGIC FOR BETTER UX
    if target_count <= 20:
        chunk_size = min(target_count, 10)  # Small tasks don't need chunking
    elif target_count <= 30:
        # For 21-30 items, use 2 chunks to avoid too many small chunks
        chunk_size = max(10, target_count // 2)  # Aim for roughly 2 chunks
    elif target_count <= 50:
        # For 31-50 items, aim for 2-3 chunks maximum for better UX
        if target_count <= 40:
            chunk_size = max(15, target_count // 2)  # 2 chunks for 31-40 items
        else:
            chunk_size = max(17, target_count // 3)  # 3 chunks for 41-50 items
    else:
        chunk_size = base_chunk_size
    
    return chunk_size

def test_chunking(target_count: int):
    chunk_size = calculate_optimal_chunk_size(target_count)
    total_chunks = math.ceil(target_count / chunk_size)
    
    print(f'Target count: {target_count}')
    print(f'Calculated chunk size: {chunk_size}')
    print(f'Number of chunks: {total_chunks}')
    
    # Show how items are distributed
    total_items = 0
    for i in range(total_chunks):
        start_index = i * chunk_size + 1
        end_index = min((i + 1) * chunk_size, target_count)
        actual_chunk_size = end_index - start_index + 1
        
        print(f'Chunk {i+1}: items {start_index}-{end_index} (size: {actual_chunk_size})')
        total_items += actual_chunk_size
    
    print(f'Total items across all chunks: {total_items}')
    
    if total_items != target_count:
        print(f'❌ ISSUE: Expected {target_count} total, but chunks add up to {total_items}')
    else:
        print('✅ Chunking looks correct')

if __name__ == "__main__":
    print("Testing chunking for 30 items:")
    print("=" * 40)
    test_chunking(30)
    
    print("\nTesting chunking for 50 items:")
    print("=" * 40)
    test_chunking(50)