"""
Excel document handler for XLS and XLSX files
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import os
import re
from .base import DocumentHandler, ExtractedChunk, ExtractionPreview

class ExcelHandler(DocumentHandler):
    """Handler for Excel files (XLS/XLSX)"""
    
    SUPPORTED_EXTENSIONS = ['.xls', '.xlsx']
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.xls', '.xlsx']
        
        # Excel-specific settings
        self.MAX_ROWS_PER_CHUNK = 20  # Group rows into chunks
        self.MAX_EMPTY_ROWS = 5  # Skip if more than 5 empty rows
        self.MIN_COLUMNS = 2  # Minimum columns to consider valid table
        
    def validate(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """Validate Excel file"""
        # Common validation first
        is_valid, error = super().validate(file_path)
        if not is_valid:
            return is_valid, error
            
        # Check file extension
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.supported_extensions:
            return False, f"Unsupported file type: {ext}"
            
        # Try to read file
        try:
            # Just try to read sheet names to validate
            if ext == '.xlsx':
                xl_file = pd.ExcelFile(file_path, engine='openpyxl')
            else:  # .xls
                xl_file = pd.ExcelFile(file_path, engine='xlrd')
            sheet_names = xl_file.sheet_names
            xl_file.close()
            
            if not sheet_names:
                return False, "Excel file has no sheets"
                
        except Exception as e:
            return False, f"Failed to read Excel file: {str(e)}"
            
        return True, None
        
    def extract(self, file_path: str, options: Dict[str, Any] = None) -> List[ExtractedChunk]:
        """Extract content from Excel file"""
        options = options or {}
        chunks = []
        
        # Determine engine based on file extension
        ext = os.path.splitext(file_path)[1].lower()
        engine = 'openpyxl' if ext == '.xlsx' else 'xlrd'
        
        # Read Excel file
        xl_file = pd.ExcelFile(file_path, engine=engine)
        
        # Get file metadata
        file_id = self._generate_file_id(file_path)
        file_name = os.path.basename(file_path)
        
        # Process each sheet (or selected sheets)
        selected_sheets = options.get('sheets', xl_file.sheet_names)
        exclude_sheets = options.get('exclude_sheets', [])
        
        for sheet_idx, sheet_name in enumerate(xl_file.sheet_names):
            if sheet_name not in selected_sheets or sheet_name in exclude_sheets:
                continue
                
            print(f"Processing sheet: {sheet_name}")
            
            try:
                # Read sheet
                df = xl_file.parse(sheet_name)
                
                # Skip empty sheets
                if df.empty or len(df.columns) < self.MIN_COLUMNS:
                    print(f"  Skipping empty/invalid sheet: {sheet_name}")
                    continue
                    
                # Extract chunks from sheet
                sheet_chunks = self._extract_from_dataframe(
                    df, sheet_name, sheet_idx, file_id, file_name
                )
                chunks.extend(sheet_chunks)
                
            except Exception as e:
                print(f"  Error processing sheet {sheet_name}: {str(e)}")
                continue
                
        xl_file.close()
        
        # Filter quality chunks
        filtered_chunks = self.filter_quality_chunks(chunks)
        print(f"Extracted {len(filtered_chunks)} quality chunks from {len(chunks)} total")
        
        return filtered_chunks
        
    def get_preview(self, file_path: str, max_chunks: int = 5) -> ExtractionPreview:
        """Get extraction preview"""
        # Extract with limited processing
        chunks = self.extract(file_path, options={'preview_mode': True})
        
        # Get file metadata
        ext = os.path.splitext(file_path)[1].lower()
        engine = 'openpyxl' if ext == '.xlsx' else 'xlrd'
        xl_file = pd.ExcelFile(file_path, engine=engine)
        
        file_metadata = {
            'filename': os.path.basename(file_path),
            'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
            'sheets': xl_file.sheet_names,
            'sheet_count': len(xl_file.sheet_names)
        }
        
        xl_file.close()
        
        # Warnings
        warnings = []
        if len(chunks) > 100:
            warnings.append(f"Large file: will create {len(chunks)} chunks")
        if any(chunk.quality_score < 0.5 for chunk in chunks[:max_chunks]):
            warnings.append("Some chunks have low quality scores")
            
        return ExtractionPreview(
            total_chunks=len(chunks),
            sample_chunks=chunks[:max_chunks],
            file_metadata=file_metadata,
            warnings=warnings
        )
        
    def _extract_from_dataframe(self, df: pd.DataFrame, sheet_name: str, 
                               sheet_idx: int, file_id: str, file_name: str) -> List[ExtractedChunk]:
        """Extract chunks from a pandas DataFrame"""
        chunks = []
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Detect table structure
        header_row = self._detect_header_row(df)
        if header_row > 0:
            # Re-read with proper header
            df = df.iloc[header_row:].reset_index(drop=True)
            df.columns = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)
            
        # Get column headers for context
        headers = list(df.columns)
        
        # Group rows into chunks
        total_rows = len(df)
        chunk_idx = 0
        
        for start_row in range(0, total_rows, self.MAX_ROWS_PER_CHUNK):
            end_row = min(start_row + self.MAX_ROWS_PER_CHUNK, total_rows)
            chunk_df = df.iloc[start_row:end_row]
            
            # Skip chunks with too many empty rows
            non_empty_rows = chunk_df.dropna(how='all').shape[0]
            if non_empty_rows < 3:  # Less than 3 non-empty rows
                continue
                
            # Convert to text representation
            content = self._dataframe_to_text(chunk_df, headers)
            
            if not content.strip():
                continue
                
            # Create metadata
            metadata = {
                'source': file_name.lower(),  # Normalize filename to lowercase
                'doc_type': 'excel',
                'sheet_name': sheet_name,
                'sheet_index': sheet_idx,
                'row_range': f"{start_row + 1}-{end_row}",  # 1-indexed for users
                'column_headers': headers,
                'has_headers': header_row >= 0,
                'uploaded_at': datetime.now().isoformat(),
                'file_id': file_id,
                'chunk_type': 'table_rows'
            }
            
            # Add context about the table
            table_context = self._infer_table_context(headers, chunk_df)
            if table_context:
                metadata['table_context'] = table_context
                
            chunk_id = self.generate_chunk_id(
                file_id, chunk_idx, 
                sheet=sheet_idx, 
                rows=f"{start_row}_{end_row}"
            )
            metadata['chunk_id'] = chunk_id
            
            chunk = ExtractedChunk(
                content=content,
                metadata=metadata
            )
            chunks.append(chunk)
            chunk_idx += 1
            
        return chunks
        
    def _detect_header_row(self, df: pd.DataFrame) -> int:
        """Detect header row in DataFrame"""
        # Simple heuristic: look for row with mostly strings
        # and different from numeric data below
        for i in range(min(5, len(df))):
            row = df.iloc[i]
            
            # Check if row has mostly non-numeric values
            non_numeric = sum(1 for val in row if not pd.isna(val) and 
                            not isinstance(val, (int, float, np.number)))
            
            if non_numeric > len(row) * 0.6:  # 60% non-numeric
                # Check if next rows are more numeric
                if i + 1 < len(df):
                    next_row = df.iloc[i + 1]
                    next_numeric = sum(1 for val in next_row if 
                                     isinstance(val, (int, float, np.number)))
                    if next_numeric > len(next_row) * 0.5:
                        return i
                        
        return -1  # No header detected
        
    def _dataframe_to_text(self, df: pd.DataFrame, headers: List[str]) -> str:
        """Convert DataFrame to readable text format"""
        lines = []
        
        # Add headers as context
        lines.append("Table data:")
        lines.append("Columns: " + " | ".join(str(h) for h in headers))
        lines.append("-" * 50)
        
        # Convert each row
        for idx, row in df.iterrows():
            row_parts = []
            for col, value in row.items():
                if pd.notna(value):
                    # Format based on type
                    if isinstance(value, (int, float)):
                        formatted = f"{col}: {value:g}"  # Remove trailing zeros
                    else:
                        formatted = f"{col}: {str(value).strip()}"
                    row_parts.append(formatted)
                    
            if row_parts:  # Only add non-empty rows
                lines.append(" | ".join(row_parts))
                
        return "\n".join(lines)
        
    def _infer_table_context(self, headers: List[str], df: pd.DataFrame) -> Optional[str]:
        """Try to infer what the table is about"""
        context_clues = []
        
        # Check headers for common patterns
        header_text = " ".join(str(h).lower() for h in headers)
        
        # Financial data
        if any(term in header_text for term in ['revenue', 'profit', 'expense', 'cost', 'income']):
            context_clues.append("Financial data")
            
        # Time series
        if any(term in header_text for term in ['date', 'month', 'year', 'quarter', 'time']):
            context_clues.append("Time series data")
            
        # Product/inventory
        if any(term in header_text for term in ['product', 'item', 'sku', 'inventory', 'stock']):
            context_clues.append("Product/Inventory data")
            
        # Customer/sales
        if any(term in header_text for term in ['customer', 'client', 'sales', 'order']):
            context_clues.append("Customer/Sales data")
            
        # Check for numeric columns (might be metrics)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > len(df.columns) * 0.5:
            context_clues.append("Numerical/Metrics data")
            
        return ", ".join(context_clues) if context_clues else None
        
    def _generate_file_id(self, file_path: str) -> str:
        """Generate unique file ID"""
        import hashlib
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:12]
        return f"excel_{file_hash}"
    
    def extract_preview(self, file_path: str, max_rows: int = 10) -> Dict[str, Any]:
        """Extract preview information from Excel file"""
        try:
            # Read Excel file
            excel_file = pd.ExcelFile(file_path)
            
            preview_data = {
                'sheets': [],
                'metadata': {
                    'total_sheets': len(excel_file.sheet_names),
                    'file_size_kb': round(os.path.getsize(file_path) / 1024, 2)
                }
            }
            
            # Preview each sheet
            for sheet_name in excel_file.sheet_names[:3]:  # Preview first 3 sheets
                df = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=max_rows)
                
                # Get sheet preview
                sheet_preview = {
                    'name': sheet_name,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'column_names': list(df.columns),
                    'data_preview': []
                }
                
                # Add first few rows as preview
                for idx, row in df.head(5).iterrows():
                    sheet_preview['data_preview'].append(row.to_dict())
                
                # Detect data types
                data_types = {}
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    if dtype.startswith('float') or dtype.startswith('int'):
                        data_types[col] = 'numeric'
                    elif dtype == 'object':
                        # Check if it's likely a date
                        sample = df[col].dropna().head(1)
                        if not sample.empty:
                            try:
                                pd.to_datetime(sample.iloc[0])
                                data_types[col] = 'date'
                            except:
                                data_types[col] = 'text'
                        else:
                            data_types[col] = 'text'
                    else:
                        data_types[col] = dtype
                
                sheet_preview['column_types'] = data_types
                preview_data['sheets'].append(sheet_preview)
            
            return preview_data
            
        except Exception as e:
            return {
                'error': f"Failed to generate preview: {str(e)}",
                'metadata': {}
            }