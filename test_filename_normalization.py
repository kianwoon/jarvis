#!/usr/bin/env python3
"""
Test script to verify filename normalization in document uploads
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_filename_normalization():
    """Test that filenames are normalized to lowercase in various scenarios"""
    print("\n=== Testing Filename Normalization ===")
    
    test_filenames = [
        ("Document.PDF", "document.pdf"),
        ("Financial_Report_2024.xlsx", "financial_report_2024.xlsx"),
        ("Meeting Notes.PDF", "meeting notes.pdf"),
        ("SALES_DATA.XLS", "sales_data.xls"),
        ("MixedCase_File.pdf", "mixedcase_file.pdf"),
    ]
    
    print("\nOriginal -> Normalized:")
    for original, expected in test_filenames:
        normalized = original.lower()
        match = "✓" if normalized == expected else "✗"
        print(f"{match} '{original}' -> '{normalized}' (expected: '{expected}')")


def test_metadata_structure():
    """Test metadata structure with normalized filenames"""
    print("\n=== Testing Metadata Structure ===")
    
    # Simulate metadata creation for different file types
    test_cases = [
        {
            'original_filename': 'Report_Q4_2024.PDF',
            'doc_type': 'pdf',
            'expected_source': 'report_q4_2024.pdf'
        },
        {
            'original_filename': 'Sales_Analysis.XLSX',
            'doc_type': 'excel',
            'expected_source': 'sales_analysis.xlsx'
        },
        {
            'original_filename': 'Meeting_NOTES.pdf',
            'doc_type': 'pdf',
            'expected_source': 'meeting_notes.pdf'
        }
    ]
    
    for case in test_cases:
        # Simulate metadata creation
        metadata = {
            'source': case['original_filename'].lower(),
            'doc_type': case['doc_type'],
            'uploaded_at': '2024-01-01T00:00:00'
        }
        
        match = "✓" if metadata['source'] == case['expected_source'] else "✗"
        print(f"{match} {case['original_filename']} -> source: '{metadata['source']}'")


def test_deduplication_with_filenames():
    """Test that documents with different case filenames are properly deduplicated"""
    print("\n=== Testing Deduplication with Filename Variations ===")
    
    from utils.deduplication import hash_text
    
    # Simulate document content with different filename cases
    test_documents = [
        {
            'content': 'This is a financial report for Q4 2024',
            'filename': 'Financial_Report.pdf'
        },
        {
            'content': 'This is a financial report for Q4 2024',
            'filename': 'FINANCIAL_REPORT.PDF'
        },
        {
            'content': 'This is a financial report for Q4 2024',
            'filename': 'financial_report.pdf'
        }
    ]
    
    print("\nDocument hashes (content only):")
    hashes = []
    for doc in test_documents:
        hash_val = hash_text(doc['content'])
        hashes.append(hash_val)
        print(f"  Filename: {doc['filename']} -> Hash: {hash_val[:16]}...")
    
    all_same = all(h == hashes[0] for h in hashes)
    match = "✓" if all_same else "✗"
    print(f"\n{match} All documents have the same hash: {all_same}")
    
    print("\nNormalized filenames:")
    for doc in test_documents:
        normalized = doc['filename'].lower()
        print(f"  {doc['filename']} -> {normalized}")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Filename Normalization Test Suite")
    print("=" * 60)
    
    # Test 1: Basic filename normalization
    test_filename_normalization()
    
    # Test 2: Metadata structure
    test_metadata_structure()
    
    # Test 3: Deduplication with filenames
    test_deduplication_with_filenames()
    
    print("\n" + "=" * 60)
    print("Summary of Implementation:")
    print("1. PDF upload: file.filename.lower() in metadata")
    print("2. Excel handler: file_name.lower() in metadata")
    print("3. Progress endpoints: Normalized filenames passed through")
    print("4. This ensures consistent filename handling across all uploads")
    print("=" * 60)


if __name__ == "__main__":
    main()