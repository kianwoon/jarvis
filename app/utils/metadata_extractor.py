"""
Metadata extraction utilities for document files.
Extracts creation_date and last_modified_date from various sources.
"""

import os
import re
from datetime import datetime
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path

try:
    import pypdf
except ImportError:
    pypdf = None

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extract metadata from filenames and file system."""
    
    # Date patterns commonly found in filenames
    DATE_PATTERNS = [
        # ISO format: 2024-01-15, 20240115
        (r'(\d{4})-(\d{1,2})-(\d{1,2})', '%Y-%m-%d'),
        (r'(?:^|[^\d])(\d{8})(?:[^\d]|$)', '%Y%m%d'),  # Standalone 8-digit dates
        (r'(\d{4})(\d{2})(\d{2})', '%Y%m%d'),
        
        # Quarter patterns: Q1_2024, Q3-2024
        (r'Q([1-4])[-_](\d{4})', 'Q%q-%Y'),
        
        # Month year: Jan_2024, January-2024, 01_2024
        (r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-_](\d{4})', '%b-%Y'),
        (r'(January|February|March|April|May|June|July|August|September|October|November|December)[-_](\d{4})', '%B-%Y'),
        (r'(\d{1,2})[-_](\d{4})', '%m-%Y'),
        
        # Effective/Valid dates
        (r'Effective[-_](\d{8})', '%Y%m%d'),
        (r'Valid[-_]from[-_](\d{4})-(\d{1,2})-(\d{1,2})', '%Y-%m-%d'),
        
        # Year only: 2024 (should be last to avoid false matches)
        (r'(?:^|[^\d])(\d{4})(?:[^\d]|$)', '%Y'),
    ]
    
    # Version patterns
    VERSION_PATTERNS = [
        r'[Vv](\d+(?:\.\d+)*)',
        r'Rev(\d+)',
        r'Version[-_](\d+(?:\.\d+)*)',
    ]
    
    @staticmethod
    def extract_from_filename(filename: str) -> Dict[str, Optional[str]]:
        """
        Extract metadata from filename patterns.
        
        Args:
            filename: The filename to analyze
            
        Returns:
            Dict with creation_date and last_modified_date (ISO format)
        """
        metadata = {
            'creation_date': None,
            'last_modified_date': None
        }
        
        # Remove file extension for cleaner parsing
        name_without_ext = Path(filename).stem
        
        # Extract dates
        dates_found = []
        for pattern, date_format in MetadataExtractor.DATE_PATTERNS:
            matches = re.findall(pattern, name_without_ext, re.IGNORECASE)
            for match in matches:
                try:
                    if date_format == 'Q%q-%Y':
                        # Handle quarter patterns
                        quarter, year = match
                        month = (int(quarter) - 1) * 3 + 1
                        date_obj = datetime(int(year), month, 1)
                    elif isinstance(match, tuple):
                        # Handle multi-group patterns
                        date_str = '-'.join(match)
                        date_obj = datetime.strptime(date_str, date_format)
                    else:
                        # Handle single group patterns
                        if date_format == '%Y':
                            date_obj = datetime(int(match), 1, 1)
                        else:
                            date_obj = datetime.strptime(match, date_format)
                    
                    dates_found.append(date_obj)
                except (ValueError, IndexError) as e:
                    logger.debug(f"Failed to parse date from {match}: {e}")
                    continue
        
        # Sort dates and assign
        if dates_found:
            dates_found.sort()
            # Earliest date as creation_date
            metadata['creation_date'] = dates_found[0].isoformat()
            # Latest date as last_modified_date
            metadata['last_modified_date'] = dates_found[-1].isoformat()
        
        # Check for version information (might indicate modification)
        version_match = None
        for pattern in MetadataExtractor.VERSION_PATTERNS:
            match = re.search(pattern, name_without_ext, re.IGNORECASE)
            if match:
                version_match = match.group(1)
                break
        
        if version_match and not metadata['last_modified_date']:
            # If we have a version but no modified date, use current date
            metadata['last_modified_date'] = datetime.now().isoformat()
        
        return metadata
    
    @staticmethod
    def extract_from_file_system(file_path: str) -> Dict[str, Optional[str]]:
        """
        Extract metadata from file system.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict with creation_date and last_modified_date (ISO format)
        """
        metadata = {
            'creation_date': None,
            'last_modified_date': None
        }
        
        try:
            stat = os.stat(file_path)
            
            # Get creation time (birth time on some systems, ctime on others)
            if hasattr(stat, 'st_birthtime'):
                # macOS, some BSD
                creation_time = datetime.fromtimestamp(stat.st_birthtime)
            else:
                # Linux, Windows - use ctime as fallback
                creation_time = datetime.fromtimestamp(stat.st_ctime)
            
            # Get modification time
            modification_time = datetime.fromtimestamp(stat.st_mtime)
            
            metadata['creation_date'] = creation_time.isoformat()
            metadata['last_modified_date'] = modification_time.isoformat()
            
        except (OSError, IOError) as e:
            logger.error(f"Failed to get file system metadata for {file_path}: {e}")
        
        return metadata
    
    @staticmethod
    def extract_from_pdf_metadata(file_path: str) -> Dict[str, Optional[str]]:
        """
        Extract metadata from PDF internal properties.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dict with creation_date and last_modified_date (ISO format)
        """
        metadata = {
            'creation_date': None,
            'last_modified_date': None
        }
        
        if not pypdf or not file_path.lower().endswith('.pdf'):
            return metadata
        
        try:
            with open(file_path, 'rb') as pdf_file:
                pdf_reader = pypdf.PdfReader(pdf_file)
                if pdf_reader.metadata:
                    # Try to get creation date
                    if '/CreationDate' in pdf_reader.metadata:
                        creation_str = pdf_reader.metadata['/CreationDate']
                        # PDF dates are in format: D:YYYYMMDDHHmmSS
                        if creation_str.startswith('D:'):
                            creation_str = creation_str[2:]
                        try:
                            # Parse first 14 characters as datetime
                            date_obj = datetime.strptime(creation_str[:14], '%Y%m%d%H%M%S')
                            metadata['creation_date'] = date_obj.isoformat()
                        except:
                            pass
                    
                    # Try to get modification date
                    if '/ModDate' in pdf_reader.metadata:
                        mod_str = pdf_reader.metadata['/ModDate']
                        if mod_str.startswith('D:'):
                            mod_str = mod_str[2:]
                        try:
                            date_obj = datetime.strptime(mod_str[:14], '%Y%m%d%H%M%S')
                            metadata['last_modified_date'] = date_obj.isoformat()
                        except:
                            pass
        except Exception as e:
            logger.debug(f"Failed to extract PDF metadata from {file_path}: {e}")
        
        return metadata
    
    @staticmethod
    def extract_metadata(file_path: str, filename: Optional[str] = None) -> Dict[str, str]:
        """
        Extract metadata using a hierarchical approach:
        1. Try to extract from filename
        2. Try to extract from PDF internal metadata (if PDF)
        3. Fall back to file system metadata
        4. Use current date as last resort
        
        Args:
            file_path: Path to the file
            filename: Optional filename to use (if different from file_path)
            
        Returns:
            Dict with creation_date and last_modified_date (ISO format)
        """
        if filename is None:
            filename = os.path.basename(file_path)
        
        # First try: Extract from filename
        filename_metadata = MetadataExtractor.extract_from_filename(filename)
        
        # Second try: Extract from PDF metadata (if applicable)
        pdf_metadata = MetadataExtractor.extract_from_pdf_metadata(file_path)
        
        # Third try: Extract from file system
        filesystem_metadata = MetadataExtractor.extract_from_file_system(file_path)
        
        # Combine results with priority to filename extraction, then PDF metadata, then filesystem
        final_metadata = {
            'creation_date': filename_metadata['creation_date'] or 
                           pdf_metadata['creation_date'] or
                           filesystem_metadata['creation_date'] or 
                           datetime.now().isoformat(),
            'last_modified_date': filename_metadata['last_modified_date'] or 
                                pdf_metadata['last_modified_date'] or
                                filesystem_metadata['last_modified_date'] or 
                                datetime.now().isoformat()
        }
        
        # Log extraction source for debugging
        logger.info(f"Metadata extraction for {filename}:")
        logger.info(f"  Filename extraction: creation={filename_metadata['creation_date']}, modified={filename_metadata['last_modified_date']}")
        logger.info(f"  PDF metadata extraction: creation={pdf_metadata['creation_date']}, modified={pdf_metadata['last_modified_date']}")
        logger.info(f"  Filesystem extraction: creation={filesystem_metadata['creation_date']}, modified={filesystem_metadata['last_modified_date']}")
        
        if filename_metadata['creation_date']:
            logger.info(f"  → Using creation_date from filename: {final_metadata['creation_date']}")
        elif pdf_metadata['creation_date']:
            logger.info(f"  → Using creation_date from PDF metadata: {final_metadata['creation_date']}")
        elif filesystem_metadata['creation_date']:
            logger.info(f"  → Using creation_date from filesystem: {final_metadata['creation_date']}")
        else:
            logger.info(f"  → Using creation_date as current time: {final_metadata['creation_date']}")
            
        if filename_metadata['last_modified_date']:
            logger.info(f"  → Using last_modified_date from filename: {final_metadata['last_modified_date']}")
        elif pdf_metadata['last_modified_date']:
            logger.info(f"  → Using last_modified_date from PDF metadata: {final_metadata['last_modified_date']}")
        elif filesystem_metadata['last_modified_date']:
            logger.info(f"  → Using last_modified_date from filesystem: {final_metadata['last_modified_date']}")
        else:
            logger.info(f"  → Using last_modified_date as current time: {final_metadata['last_modified_date']}")
        
        return final_metadata


def test_metadata_extraction():
    """Test the metadata extraction with various filename patterns."""
    test_filenames = [
        "Q3_2024_Sales_Report.xlsx",
        "Policy_V1.2_Effective_20250101.pdf",
        "Annual_Report_2023.docx",
        "Meeting_Notes_2024-03-15.doc",
        "Budget_Jan_2024_Final.xlsx",
        "Compliance_Document_20240815_v3.pdf",
        "Project_Plan_Q2-2024.pptx",
        "Employee_Handbook_Rev5.pdf",
        "Financial_Statement_2023-12-31.xlsx",
        "Training_Material_Version_2.5.pptx"
    ]
    
    extractor = MetadataExtractor()
    for filename in test_filenames:
        metadata = extractor.extract_from_filename(filename)
        print(f"\nFilename: {filename}")
        print(f"  Creation Date: {metadata['creation_date']}")
        print(f"  Last Modified: {metadata['last_modified_date']}")


if __name__ == "__main__":
    test_metadata_extraction()