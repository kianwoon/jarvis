"""
Document preview endpoint for all supported file types
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Dict, Any
import os
import tempfile
import shutil

# Import document handlers
from app.document_handlers.excel_handler import ExcelHandler
from app.document_handlers.word_handler import WordHandler
from app.document_handlers.powerpoint_handler import PowerPointHandler

router = APIRouter()

# Document handler registry
PREVIEW_HANDLERS = {
    '.xlsx': ExcelHandler(),
    '.xls': ExcelHandler(),
    '.docx': WordHandler(),
    '.doc': WordHandler(),
    '.pptx': PowerPointHandler(),
    '.ppt': PowerPointHandler(),
}


@router.post("/preview")
async def preview_document(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Generate a preview of the uploaded document
    Returns structured preview data based on file type
    """
    # Validate file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in PREVIEW_HANDLERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported types: {', '.join(PREVIEW_HANDLERS.keys())}"
        )
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Save uploaded file
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Get appropriate handler
        handler = PREVIEW_HANDLERS[file_ext]
        
        # Extract preview
        preview_data = handler.extract_preview(temp_path)
        
        # Add common metadata
        preview_data['file_info'] = {
            'filename': file.filename.lower(),
            'file_type': file_ext[1:],  # Remove the dot
            'file_size': len(content),
            'file_size_mb': round(len(content) / (1024 * 1024), 2)
        }
        
        return {
            'status': 'success',
            'preview': preview_data
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate preview: {str(e)}"
        )
        
    finally:
        # Cleanup
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


@router.get("/preview/test")
async def test_preview_endpoint():
    """Test endpoint to verify preview service is running"""
    return {
        'status': 'ok',
        'supported_types': list(PREVIEW_HANDLERS.keys()),
        'handlers': {
            ext: type(handler).__name__ 
            for ext, handler in PREVIEW_HANDLERS.items()
        }
    }