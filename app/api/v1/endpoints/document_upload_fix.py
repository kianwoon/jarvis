"""
Alternative upload endpoint that handles file reading issues
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import shutil
import tempfile
import os
from pathlib import Path

router = APIRouter()

@router.post("/upload_pdf_alt")
async def upload_pdf_alternative(file: UploadFile = File(...)):
    """Alternative upload that saves file first to avoid I/O issues"""
    
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Create a temporary file
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"upload_{file.filename}")
    
    try:
        # Save uploaded file to disk first
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get file size
        file_size = os.path.getsize(temp_path)
        
        # Now we can use the regular upload_pdf_progress logic
        # by creating a mock file object from the saved file
        from app.api.v1.endpoints.document import progress_generator, UploadProgress
        
        # Create a mock UploadFile from the saved file
        class MockUploadFile:
            def __init__(self, filename, path):
                self.filename = filename
                self._path = path
                self._content = None
                
            async def read(self):
                if self._content is None:
                    with open(self._path, 'rb') as f:
                        self._content = f.read()
                return self._content
                
            async def seek(self, pos):
                pass  # No-op for our use case
        
        mock_file = MockUploadFile(file.filename, temp_path)
        progress = UploadProgress()
        
        # Clean up the temp file after streaming
        async def cleanup_generator():
            try:
                async for chunk in progress_generator(mock_file, progress):
                    yield chunk
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
        
        return StreamingResponse(
            cleanup_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")