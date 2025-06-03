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
# Preview handler registry - lazy initialization to avoid import errors
def get_preview_handlers():
    """Get preview handlers with lazy initialization"""
    handlers = {}
    
    # Initialize handlers that are available
    try:
        handlers['.xlsx'] = ExcelHandler()
        handlers['.xls'] = ExcelHandler()
    except ImportError:
        logger.warning("Excel handler not available")
    
    try:
        handlers['.docx'] = WordHandler()
        handlers['.doc'] = WordHandler()
    except ImportError:
        logger.warning("Word handler not available")
    
    try:
        from app.document_handlers.powerpoint_handler import PowerPointHandler
        handlers['.pptx'] = PowerPointHandler()
        handlers['.ppt'] = PowerPointHandler()
    except ImportError:
        logger.warning("PowerPoint handler not available")
    
    return handlers

# Cache the handlers after first initialization
_preview_handlers = None

def get_preview_handler_for_file(file_extension):
    """Get the appropriate preview handler for a file extension"""
    global _preview_handlers
    if _preview_handlers is None:
        _preview_handlers = get_preview_handlers()
    return _preview_handlers.get(file_extension)

PREVIEW_HANDLERS = None  # Will be lazy-loaded


@router.post("/preview")
async def preview_document(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Generate a preview of the uploaded document
    Returns structured preview data based on file type
    """
    # Validate file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    available_handlers = get_preview_handlers()
    if file_ext not in available_handlers:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported types: {', '.join(available_handlers.keys())}"
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
        handler = get_preview_handler_for_file(file_ext)
        
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
        'supported_types': list(get_preview_handlers().keys()),
        'handlers': {
            ext: type(handler).__name__ 
            for ext, handler in get_preview_handlers().items()
        }
    }