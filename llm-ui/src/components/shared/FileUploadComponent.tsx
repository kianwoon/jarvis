import React, { useState, useRef } from 'react';
import {
  Box,
  Button,
  IconButton,
  LinearProgress,
  Typography,
  Alert,
  Chip,
  Paper,
  Fade
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  AttachFile as AttachIcon,
  CheckCircle as SuccessIcon,
  Error as ErrorIcon,
  Description as DocIcon
} from '@mui/icons-material';

interface FileUploadProps {
  onUploadStart?: (file: File) => void;
  onUploadProgress?: (progress: number) => void;
  onUploadSuccess?: (result: any) => void;
  onUploadError?: (error: string) => void;
  disabled?: boolean;
  collection?: string;
  autoClassify?: boolean;
}

interface UploadState {
  file: File | null;
  progress: number;
  status: 'idle' | 'uploading' | 'success' | 'error';
  message: string;
  result?: any;
}

const ALLOWED_TYPES = {
  'application/pdf': '.pdf',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
  'application/vnd.ms-excel': '.xls',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
  'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
  'application/vnd.ms-powerpoint': '.ppt'
};

const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB

const FileUploadComponent: React.FC<FileUploadProps> = ({
  onUploadStart,
  onUploadProgress,
  onUploadSuccess,
  onUploadError,
  disabled = false,
  collection,
  autoClassify = true
}) => {
  const [uploadState, setUploadState] = useState<UploadState>({
    file: null,
    progress: 0,
    status: 'idle',
    message: ''
  });
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateFile = (file: File): string | null => {
    if (!Object.keys(ALLOWED_TYPES).includes(file.type)) {
      return `Unsupported file type. Allowed: ${Object.values(ALLOWED_TYPES).join(', ')}`;
    }
    if (file.size > MAX_FILE_SIZE) {
      return `File too large. Maximum size: ${MAX_FILE_SIZE / 1024 / 1024}MB`;
    }
    return null;
  };

  const uploadFile = async (file: File) => {
    const validationError = validateFile(file);
    if (validationError) {
      setUploadState({
        file: null,
        progress: 0,
        status: 'error',
        message: validationError
      });
      onUploadError?.(validationError);
      return;
    }

    setUploadState({
      file,
      progress: 0,
      status: 'uploading',
      message: 'Uploading and processing...'
    });

    onUploadStart?.(file);

    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const params = new URLSearchParams();
      if (collection) params.append('collection_name', collection);
      params.append('auto_classify', autoClassify.toString());

      const response = await fetch(`/api/v1/document/upload-multi?${params}`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Upload failed: ${response.statusText}`);
      }

      // Simulate progress updates during processing
      const progressInterval = setInterval(() => {
        setUploadState(prev => {
          const newProgress = Math.min(prev.progress + 10, 90);
          onUploadProgress?.(newProgress);
          return { ...prev, progress: newProgress };
        });
      }, 200);

      const result = await response.json();
      clearInterval(progressInterval);

      setUploadState({
        file,
        progress: 100,
        status: 'success',
        message: `Successfully processed ${result.unique_chunks || 0} chunks`,
        result
      });

      onUploadProgress?.(100);
      onUploadSuccess?.(result);

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Upload failed';
      setUploadState({
        file,
        progress: 0,
        status: 'error',
        message: errorMessage
      });
      onUploadError?.(errorMessage);
    }
  };

  const handleFileSelect = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    uploadFile(files[0]);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    handleFileSelect(e.dataTransfer.files);
  };

  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };

  const clearState = () => {
    setUploadState({
      file: null,
      progress: 0,
      status: 'idle',
      message: ''
    });
  };

  const getStatusColor = () => {
    switch (uploadState.status) {
      case 'success': return 'success';
      case 'error': return 'error';
      case 'uploading': return 'primary';
      default: return 'inherit';
    }
  };

  const getStatusIcon = () => {
    switch (uploadState.status) {
      case 'success': return <SuccessIcon />;
      case 'error': return <ErrorIcon />;
      case 'uploading': return <UploadIcon />;
      default: return <AttachIcon />;
    }
  };

  return (
    <Box>
      <input
        ref={fileInputRef}
        type="file"
        accept={Object.keys(ALLOWED_TYPES).join(',')}
        onChange={(e) => handleFileSelect(e.target.files)}
        style={{ display: 'none' }}
      />

      {/* Simple Upload Icon Button */}
      <IconButton
        onClick={handleButtonClick}
        disabled={disabled || uploadState.status === 'uploading'}
        color={getStatusColor()}
        title="Upload document (PDF, Excel, Word, PowerPoint)"
        size="small"
      >
        {getStatusIcon()}
      </IconButton>

      {/* Compact Status Display */}
      {uploadState.status === 'uploading' && (
        <Box sx={{ position: 'absolute', top: '100%', left: 0, zIndex: 1000, mt: 1 }}>
          <Paper sx={{ p: 2, minWidth: 280, maxWidth: 320 }} elevation={3}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
              <DocIcon fontSize="small" />
              <Typography variant="body2" noWrap sx={{ flex: 1 }}>
                {uploadState.file?.name}
              </Typography>
            </Box>
            <LinearProgress 
              variant="determinate" 
              value={uploadState.progress}
              sx={{ height: 4, borderRadius: 2, mb: 1 }}
            />
            <Typography variant="caption" color="text.secondary">
              {uploadState.progress}% - {uploadState.message}
            </Typography>
          </Paper>
        </Box>
      )}
    </Box>
  );
};

export default FileUploadComponent;