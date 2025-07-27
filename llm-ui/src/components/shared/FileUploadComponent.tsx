import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Button,
  IconButton,
  LinearProgress,
  Typography,
  Alert,
  Chip,
  Fade,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Backdrop,
  Modal
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  AttachFile as AttachIcon,
  CheckCircle as SuccessIcon,
  Error as ErrorIcon,
  Description as DocIcon,
  CheckCircle as CompletedIcon,
  RadioButtonUnchecked as PendingIcon,
  AccessTime as ProcessingIcon
} from '@mui/icons-material';

interface FileUploadProps {
  onUploadStart?: (file: File) => void;
  onUploadProgress?: (progress: number) => void;
  onUploadSuccess?: (result: any) => void;
  onUploadError?: (error: string) => void;
  disabled?: boolean;
  collection?: string;
  autoClassify?: boolean;
  useProgressTracking?: boolean;
}

interface UploadState {
  file: File | null;
  progress: number;
  status: 'idle' | 'selecting' | 'uploading' | 'success' | 'error';
  message: string;
  result?: any;
  currentStep?: number;
  totalSteps?: number;
  stepName?: string;
  details?: any;
}

interface Collection {
  id: number;
  collection_name: string;
  collection_type: string;
  description: string;
  access_config: {
    restricted: boolean;
    allowed_users?: string[];
  };
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

// Define the dynamic process steps - backend may use 6, 7, or 8 steps
const getProcessSteps = (totalSteps: number) => {
  if (totalSteps === 8) {
    // PDF processing with 8 steps (current backend)
    return [
      { id: 1, name: 'Saving uploaded file', icon: '📁', description: 'Saving file to temporary location' },
      { id: 2, name: 'Loading PDF content', icon: '📄', description: 'Reading PDF content and structure' },
      { id: 3, name: 'Splitting into chunks', icon: '✂️', description: 'Breaking content into manageable pieces' },
      { id: 4, name: 'Preparing embeddings', icon: '🔧', description: 'Setting up embedding generation' },
      { id: 5, name: 'Determining collection', icon: '🎯', description: 'Selecting target collection' },
      { id: 6, name: 'Checking for duplicates', icon: '🔄', description: 'Identifying duplicate content' },
      { id: 7, name: 'Generating embeddings', icon: '🔮', description: 'Creating AI embeddings for content' },
      { id: 8, name: 'Inserting into database', icon: '💾', description: 'Storing processed content' }
    ];
  } else if (totalSteps === 7) {
    // PDF processing with 7 steps
    return [
      { id: 1, name: 'Saving uploaded file', icon: '📁', description: 'Saving file to temporary location' },
      { id: 2, name: 'Loading PDF content', icon: '📄', description: 'Reading PDF content and structure' },
      { id: 3, name: 'Splitting into chunks', icon: '✂️', description: 'Breaking content into manageable pieces' },
      { id: 4, name: 'Preparing embeddings', icon: '🔧', description: 'Setting up embedding generation' },
      { id: 5, name: 'Checking for duplicates', icon: '🔄', description: 'Identifying duplicate content' },
      { id: 6, name: 'Generating embeddings', icon: '🔮', description: 'Creating AI embeddings for content' },
      { id: 7, name: 'Inserting into database', icon: '💾', description: 'Storing processed content' }
    ];
  } else {
    // Multi-file processing with 6 steps
    return [
      { id: 1, name: 'Validating file', icon: '🔍', description: 'Checking file format and size' },
      { id: 2, name: 'Extracting content', icon: '📄', description: 'Reading document content and structure' },
      { id: 3, name: 'Generating embeddings', icon: '🔮', description: 'Creating AI embeddings for content' },
      { id: 4, name: 'Checking for duplicates', icon: '🔄', description: 'Identifying duplicate content' },
      { id: 5, name: 'Processing embeddings', icon: '⚡', description: 'Batch processing embeddings' },
      { id: 6, name: 'Inserting into database', icon: '💾', description: 'Storing processed content' }
    ];
  }
};

const FileUploadComponent: React.FC<FileUploadProps> = ({
  onUploadStart,
  onUploadProgress,
  onUploadSuccess,
  onUploadError,
  disabled = false,
  collection,
  autoClassify = true,
  useProgressTracking = true
}) => {
  const [uploadState, setUploadState] = useState<UploadState>({
    file: null,
    progress: 0,
    status: 'idle',
    message: ''
  });
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // Collection selection state
  const [collections, setCollections] = useState<Collection[]>([]);
  const [selectedCollection, setSelectedCollection] = useState<string>('');
  const [classificationResult, setClassificationResult] = useState<any>(null);
  const [showCollectionDialog, setShowCollectionDialog] = useState(false);
  const [pendingFile, setPendingFile] = useState<File | null>(null);

  // Load collections on component mount
  useEffect(() => {
    const loadCollections = async () => {
      try {
        console.log('🔄 Loading collections...');
        const response = await fetch('/api/v1/collections/');
        if (response.ok) {
          const data = await response.json();
          console.log('✅ Collections loaded:', data.length, 'collections');
          setCollections(data);
        } else {
          console.log('❌ Failed to load collections:', response.status, response.statusText);
          const errorText = await response.text();
          console.log('❌ Response body:', errorText);
        }
      } catch (error) {
        console.error('❌ Error loading collections:', error);
      }
    };
    loadCollections();
  }, []);

  // Get suggested collection for file using backend classification
  const getSuggestedCollection = async (file: File): Promise<string> => {
    try {
      //console.log('🤖 Getting AI-powered collection suggestion from backend...');
      
      // Create a small sample of the file for classification
      const formData = new FormData();
      formData.append('file', file);
      formData.append('auto_classify', 'true');
      
      // Use the backend's document classifier endpoint
      const response = await fetch('/api/v1/document/classify', {
        method: 'POST',
        body: formData
      });
      
      if (response.ok) {
        const result = await response.json();
        //console.log('✅ Backend classification result:', result);
        
        // Backend returns 'collection' field with suggested collection name
        if (result.collection) {
          //console.log('🎯 Backend suggested collection:', result.collection);
          //console.log('📊 Confidence:', result.confidence);
          //console.log('💡 Reason:', result.reason);
          
          // Store full classification result for UI display
          setClassificationResult(result);
          
          return result.collection;
        }
        
        // Fall back to collection type mapping if no direct collection
        if (result.collection_type) {
          const typeToCollection: { [key: string]: string } = {
            'regulatory_compliance': 'regulatory_compliance',
            'product_documentation': 'product_documentation', 
            'risk_management': 'risk_management',
            'customer_support': 'customer_support',
            'audit_reports': 'audit_reports',
            'training_materials': 'training_materials',
            'technical_docs': 'technical_docs',
            'policies_procedures': 'default_knowledge',
            'meeting_notes': 'default_knowledge',
            'contracts_legal': 'default_knowledge',
            'general': 'default_knowledge'
          };
          return typeToCollection[result.collection_type] || 'default_knowledge';
        }
      } else {
        //console.log('⚠️ Backend classification failed, using filename fallback');
      }
      
      // Fallback to simple filename analysis if backend fails
      const filename = file.name.toLowerCase();
      if (filename.includes('audit') || filename.includes('review')) {
        return 'audit_reports';
      } else if (filename.includes('risk') || filename.includes('assessment')) {
        return 'risk_management';
      } else if (filename.includes('product') || filename.includes('service')) {
        return 'product_documentation';
      } else if (filename.includes('compliance') || filename.includes('regulatory')) {
        return 'regulatory_compliance';
      } else if (filename.includes('training') || filename.includes('onboarding')) {
        return 'training_materials';
      } else if (filename.includes('support') || filename.includes('faq')) {
        return 'customer_support';
      } else if (filename.includes('api') || filename.includes('technical')) {
        return 'technical_docs';
      } else if (filename.includes('partner') || filename.includes('collaboration')) {
        return 'partnership';
      }
      
      return 'default_knowledge';
    } catch (error) {
      //console.error('❌ Error getting collection suggestion:', error);
      return 'default_knowledge';
    }
  };

  const validateFile = (file: File): string | null => {
    if (!Object.keys(ALLOWED_TYPES).includes(file.type)) {
      return `Unsupported file type. Allowed: ${Object.values(ALLOWED_TYPES).join(', ')}`;
    }
    if (file.size > MAX_FILE_SIZE) {
      return `File too large. Maximum size: ${MAX_FILE_SIZE / 1024 / 1024}MB`;
    }
    return null;
  };

  const uploadFile = async (file: File, targetCollection?: string) => {
    //console.log('🚀 uploadFile called with:', file.name, 'collection:', targetCollection);
    setUploadState({
      file,
      progress: 0,
      status: 'uploading',
      message: 'Initializing upload...',
      currentStep: 0,
      totalSteps: 7  // Will be updated when backend sends actual total_steps
    });

    onUploadStart?.(file);

    if (useProgressTracking) {
      //console.log('📊 Using progress tracking upload');
      await uploadWithProgress(file, targetCollection);
    } else {
      //console.log('📤 Using simple upload');
      await uploadSimple(file, targetCollection);
    }
  };

  const uploadSimple = async (file: File, targetCollection?: string) => {
    try {
      const formData = new FormData();
      formData.append('file', file);
      const finalCollection = targetCollection || collection;
      
      // Route to appropriate endpoint based on file type
      const fileExt = file.name.toLowerCase().split('.').pop();
      
      let endpoint: string;
      if (fileExt === 'pdf') {
        endpoint = '/api/v1/documents/upload_pdf';
        if (finalCollection) formData.append('collection_name', finalCollection);
        formData.append('auto_classify', autoClassify.toString());
      } else {
        const params = new URLSearchParams();
        if (finalCollection) params.append('collection_name', finalCollection);
        params.append('auto_classify', autoClassify.toString());
        endpoint = `/api/v1/documents/upload-multi?${params}`;
      }

      const response = await fetch(endpoint, {
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

      // Add file information to result for success display
      const enhancedResult = {
        ...result,
        filename: file.name,
        file_type: file.name.toLowerCase().split('.').pop() || 'unknown',
        collection: finalCollection || 'default_knowledge'
      };

      setUploadState({
        file,
        progress: 100,
        status: 'success',
        message: `Successfully processed ${result.unique_chunks || 0} chunks`,
        result: enhancedResult
      });

      onUploadProgress?.(100);
      onUploadSuccess?.(enhancedResult);
      
      // Keep modal open for 3 seconds to show success
      setTimeout(() => {
        setUploadState({
          file: null,
          progress: 0,
          status: 'idle',
          message: '',
          currentStep: 0,
          totalSteps: 0,
          stepName: '',
          details: undefined
        });
      }, 3000);

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

  const uploadWithProgress = async (file: File, targetCollection?: string) => {
    try {
      const formData = new FormData();
      formData.append('file', file);
      const finalCollection = targetCollection || collection;
      if (finalCollection) formData.append('collection_name', finalCollection);
      formData.append('auto_classify', autoClassify.toString());

      // Route to appropriate endpoint based on file type
      const fileExt = file.name.toLowerCase().split('.').pop();
      const endpoint = fileExt === 'pdf' 
        ? '/api/v1/documents/upload_pdf_simple' 
        : '/api/v1/documents/upload-multi-progress';

      console.log('📡 Making request to:', endpoint);
      console.log('📦 FormData collection:', finalCollection);
      console.log('📄 File type:', fileExt);
      
      // Add timeout to prevent hanging
      const controller = new AbortController();
      const timeoutId = setTimeout(() => {
        console.log('⏰ Upload timeout reached, aborting request');
        controller.abort();
      }, 10 * 60 * 1000); // 10 minutes timeout
      
      // Check if we're using the simple endpoint (non-streaming)
      const isSimpleEndpoint = endpoint.includes('upload_pdf_simple');
      
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);

      console.log('📡 Response status:', response.status);
      console.log('📡 Response headers:', Array.from(response.headers.entries()));

      if (!response.ok) {
        const errorData = await response.json();
        console.log('❌ Error response:', errorData);
        throw new Error(errorData.detail || `Upload failed: ${response.statusText}`);
      }

      if (isSimpleEndpoint) {
        // Handle simple JSON response
        const result = await response.json();
        console.log('📡 Simple upload result:', result);
        
        // Simulate progress updates for UI
        setUploadState(prev => ({
          ...prev,
          currentStep: 1,
          totalSteps: 7,
          stepName: 'Processing file...',
          progress: 25
        }));
        
        // Small delay to show progress
        await new Promise(resolve => setTimeout(resolve, 500));
        
        setUploadState(prev => ({
          ...prev,
          currentStep: 7,
          totalSteps: 7,
          stepName: 'Upload complete!',
          progress: 100,
          details: { status: 'success', collection: result.collection },
          status: 'success',
          message: `Upload completed successfully!`
        }));

        // Create enhanced result for success callback
        const enhancedResult = {
          status: 'success',
          filename: file.name,
          file_type: file.name.toLowerCase().split('.').pop() || 'unknown',
          collection: result.collection
        };
        
        onUploadSuccess?.(enhancedResult);
        onUploadProgress?.(100);
        
        // Keep modal open for 3 seconds to show success
        setTimeout(() => {
          setUploadState(prev => ({
            ...prev,
            status: 'idle',
            file: null,
            progress: 0,
            currentStep: 0,
            totalSteps: 0,
            stepName: '',
            details: undefined
          }));
        }, 3000);
        
        return; // End simple endpoint handling
      }

      // Handle SSE streaming for other endpoints
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      if (!reader) {
        throw new Error('No response body reader available');
      }

      console.log('📡 Starting to read SSE stream...');
      let chunkCount = 0;
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          console.log('📡 SSE stream finished, total chunks:', chunkCount);
          break;
        }

        chunkCount++;
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.substring(6));
              
              if (data.error) {
                throw new Error(data.error);
              }

              setUploadState(prev => ({
                ...prev,
                currentStep: data.current_step,
                totalSteps: data.total_steps,
                stepName: data.step_name,
                progress: data.progress_percent,
                details: data.details,
                message: data.step_name + (data.details ? ` (${Object.keys(data.details).length} details)` : '')
              }));

              onUploadProgress?.(data.progress_percent);

              if (data.progress_percent === 100) {
                const result = data.details;
                const enhancedResult = {
                  ...result,
                  filename: file.name,
                  file_type: file.name.toLowerCase().split('.').pop() || 'unknown',
                  collection: finalCollection || 'default_knowledge'
                };
                setUploadState(prev => ({
                  ...prev,
                  status: 'success',
                  message: `Successfully processed ${result?.unique_chunks || 0} chunks`,
                  result: enhancedResult
                }));
                onUploadSuccess?.(enhancedResult);
                
                setTimeout(() => {
                  setUploadState(prev => ({
                    ...prev,
                    status: 'idle',
                    file: null,
                    progress: 0,
                    currentStep: 0,
                    totalSteps: 0,
                    stepName: '',
                    details: undefined
                  }));
                }, 3000);
                return;
              }
            } catch (parseError) {
              console.warn('Failed to parse progress data:', parseError);
            }
          }
        }
      }

    } catch (error) {
      let errorMessage = 'Upload failed';
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          errorMessage = 'Upload timed out after 10 minutes. Please try again or contact support if the file is very large.';
        } else {
          errorMessage = error.message;
        }
      }
      console.error('❌ Upload error:', error);
      setUploadState({
        file,
        progress: 0,
        status: 'error',
        message: errorMessage
      });
      onUploadError?.(errorMessage);
    }
  };

  const handleFileSelect = async (files: FileList | null) => {
    //console.log('🔍 handleFileSelect called with files:', files);
    if (!files || files.length === 0) {
      //console.log('❌ No files selected');
      return;
    }
    
    const file = files[0];
    //console.log('📁 Selected file:', file.name, file.type, file.size);
    
    const validationError = validateFile(file);
    if (validationError) {
      //console.log('❌ Validation error:', validationError);
      setUploadState({
        file: null,
        progress: 0,
        status: 'error',
        message: validationError
      });
      onUploadError?.(validationError);
      return;
    }

    //console.log('✅ File validation passed');

    // If collection is pre-selected, upload directly
    if (collection) {
      //console.log('🎯 Using pre-selected collection:', collection);
      uploadFile(file, collection);
      return;
    }

    // Show collection selection dialog
    //console.log('🔄 Getting suggested collection...');
    setPendingFile(file);
    const suggested = await getSuggestedCollection(file);
    //console.log('💡 Suggested collection:', suggested);
    setSelectedCollection(suggested);
    setShowCollectionDialog(true);
    //console.log('📋 Collection dialog should now be open');
  };

  const handleCollectionConfirm = () => {
    //console.log('✅ Collection confirmed:', selectedCollection);
    if (pendingFile && selectedCollection) {
      //console.log('🚀 Starting upload with collection:', selectedCollection);
      setShowCollectionDialog(false);
      uploadFile(pendingFile, selectedCollection);
      setPendingFile(null);
    } else {
      //console.log('❌ Missing pendingFile or selectedCollection:', { pendingFile, selectedCollection });
    }
  };

  const handleCollectionCancel = () => {
    setShowCollectionDialog(false);
    setPendingFile(null);
    setSelectedCollection('');
  };


  const handleButtonClick = () => {
    //console.log('🖱️ Upload button clicked');
    //console.log('📂 File input ref:', fileInputRef.current);
    
    // Clear the file input value to allow re-selecting the same file
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    
    fileInputRef.current?.click();
    //console.log('📤 File input click triggered');
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

  // Helper function to get step status
  const getStepStatus = (stepNumber: number) => {
    const currentStep = uploadState.currentStep || 0;
    // Steps before current step are completed
    if (stepNumber < currentStep) return 'completed';
    // Current step is processing
    if (stepNumber === currentStep) return 'processing';
    // Steps after current step are pending
    return 'pending';
  };

  // Helper function to get step icon
  const getStepIcon = (stepNumber: number) => {
    const status = getStepStatus(stepNumber);
    switch (status) {
      case 'completed':
        return <CompletedIcon color="success" />;
      case 'processing':
        return <ProcessingIcon color="primary" />;
      default:
        return <PendingIcon color="disabled" />;
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

      {/* Collection Selection Dialog */}
      <Dialog open={showCollectionDialog} onClose={handleCollectionCancel} maxWidth="md" fullWidth>
        <DialogTitle>
          Select Collection for Upload
        </DialogTitle>
        <DialogContent>
          <Box sx={{ mb: 3 }}>
            <Typography variant="body2" color="text.secondary">
              File: <strong>{pendingFile?.name}</strong>
            </Typography>
            
            {/* AI Classification Results */}
            {classificationResult && (
              <Box sx={{ mt: 2, p: 2, bgcolor: 'action.hover', borderRadius: 2 }}>
                <Typography variant="subtitle2" sx={{ color: 'primary.main', fontWeight: 'bold', mb: 1 }}>
                  🤖 AI Classification Results
                </Typography>
                
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                  <Typography variant="body2" color="text.primary">
                    Suggested: <strong>{classificationResult.collection}</strong>
                  </Typography>
                  
                  {/* Confidence Score */}
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="body2" color="text.secondary">
                      Confidence:
                    </Typography>
                    <LinearProgress 
                      variant="determinate" 
                      value={classificationResult.confidence * 100}
                      sx={{ 
                        width: 100, 
                        height: 6, 
                        borderRadius: 3,
                        backgroundColor: 'grey.200',
                        '& .MuiLinearProgress-bar': {
                          backgroundColor: 
                            classificationResult.confidence >= 0.7 ? 'success.main' :
                            classificationResult.confidence >= 0.4 ? 'warning.main' : 'error.main'
                        }
                      }}
                    />
                    <Typography variant="body2" color="text.secondary">
                      {Math.round(classificationResult.confidence * 100)}%
                    </Typography>
                  </Box>
                </Box>
                
                {/* Reasoning */}
                <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic' }}>
                  💡 {classificationResult.reason}
                </Typography>
                
                {/* Low Confidence Warning */}
                {classificationResult.confidence < 0.5 && (
                  <Alert severity="warning" sx={{ mt: 2 }}>
                    <Typography variant="body2">
                      Low confidence detection. Please verify the suggested collection or choose manually.
                    </Typography>
                  </Alert>
                )}
              </Box>
            )}
          </Box>
          
          <FormControl fullWidth sx={{ mt: 2 }}>
            <InputLabel>Collection</InputLabel>
            <Select
              value={selectedCollection}
              onChange={(e) => setSelectedCollection(e.target.value)}
              label="Collection"
            >
              {collections.length === 0 ? (
                <MenuItem disabled>
                  <Typography variant="body2" color="text.secondary">
                    No collections available
                  </Typography>
                </MenuItem>
              ) : (
                collections.map((coll) => (
                  <MenuItem key={coll.id} value={coll.collection_name}>
                    <Box>
                      <Typography variant="body2" fontWeight="bold">
                        {coll.collection_name}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {coll.description}
                      </Typography>
                      {coll.access_config.restricted && (
                        <Chip 
                          label="Restricted" 
                          size="small" 
                          color="warning" 
                          sx={{ ml: 1 }}
                        />
                      )}
                    </Box>
                  </MenuItem>
                ))
              )}
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCollectionCancel}>Cancel</Button>
          <Button 
            onClick={handleCollectionConfirm} 
            variant="contained" 
            disabled={!selectedCollection}
          >
            Upload to {selectedCollection}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Prominent Progress Modal */}
      <Modal
        open={uploadState.status === 'uploading' || uploadState.status === 'success'}
        closeAfterTransition={false}
        disableEscapeKeyDown
        slots={{ backdrop: Backdrop }}
        slotProps={{
          backdrop: {
            timeout: 500,
            sx: { backgroundColor: 'rgba(0, 0, 0, 0.8)' }
          },
        }}
      >
        <Fade in={uploadState.status === 'uploading' || uploadState.status === 'success'}>
          <Box sx={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            width: 800,
            maxWidth: '95vw',
            maxHeight: '90vh',
            bgcolor: 'background.paper',
            border: '3px solid',
            borderColor: 'primary.main',
            boxShadow: 24,
            p: 4,
            borderRadius: 3,
            overflow: 'auto'
          }}>
            <Typography 
              variant="h5" 
              component="h2" 
              sx={{ 
                mb: 3, 
                display: 'flex', 
                alignItems: 'center', 
                gap: 1,
                color: 'primary.main',
                fontWeight: 'bold'
              }}
            >
              <DocIcon sx={{ fontSize: 32 }} />
              Uploading: {uploadState.file?.name}
            </Typography>
            
            {/* Overall Progress Bar */}
            <Box sx={{ mb: 4 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" sx={{ color: 'primary.main', fontWeight: 'bold' }}>
                  {uploadState.stepName || 'Processing...'}
                </Typography>
                <Typography variant="h6" sx={{ color: 'text.primary', fontWeight: 'bold' }}>
                  {uploadState.progress}%
                </Typography>
              </Box>
              <LinearProgress 
                variant="determinate" 
                value={uploadState.progress}
                sx={{ 
                  height: 12, 
                  borderRadius: 6,
                  backgroundColor: 'action.hover',
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: 'primary.main',
                  }
                }}
              />
            </Box>

            {/* Step Progress */}
            {uploadState.currentStep && uploadState.currentStep > 0 && uploadState.totalSteps && (
              <Box sx={{ mb: 4 }}>
                <Typography variant="body1" sx={{ mb: 2, color: 'text.primary', fontWeight: 'bold' }}>
                  Step {uploadState.currentStep} of {uploadState.totalSteps}
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={(uploadState.currentStep / uploadState.totalSteps) * 100}
                  sx={{ 
                    height: 8, 
                    borderRadius: 4,
                    backgroundColor: 'action.hover',
                    '& .MuiLinearProgress-bar': {
                      backgroundColor: 'success.main',
                    }
                  }}
                />
              </Box>
            )}

            {/* Detailed Process Steps List - only show during upload */}
            {uploadState.status === 'uploading' && uploadState.totalSteps && uploadState.totalSteps > 0 && (
              <Box sx={{ mb: 4 }}>
                <Typography variant="h6" sx={{ mb: 2, color: 'text.primary', fontWeight: 'bold' }}>
                  📋 Process Steps
                </Typography>
                <List dense>
                  {getProcessSteps(uploadState.totalSteps).map((step) => {
                    const status = getStepStatus(step.id);
                    const textColor = status === 'completed' ? 'success.main' : 
                                     status === 'processing' ? 'primary.main' : 'text.secondary';
                    return (
                      <ListItem key={step.id} sx={{ py: 0.5 }}>
                        <ListItemIcon sx={{ minWidth: 36 }}>
                          {getStepIcon(step.id)}
                        </ListItemIcon>
                        <ListItemText
                          primary={
                            <Typography variant="body2" sx={{ color: textColor, fontWeight: status === 'processing' ? 'bold' : 'normal' }}>
                              {step.icon} {step.name}
                            </Typography>
                          }
                          secondary={
                            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                              {step.description}
                            </Typography>
                          }
                        />
                      </ListItem>
                    );
                  })}
                </List>
              </Box>
            )}

            {/* Embeddings Progress Bar */}
            {uploadState.details?.embedding_progress && uploadState.details?.total_embeddings && (
              <Box sx={{ mb: 4 }}>
                <Typography variant="h6" sx={{ mb: 2, color: 'text.primary', fontWeight: 'bold' }}>
                  🔮 Embeddings Progress
                </Typography>
                <Box sx={{ mb: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                    <Typography variant="body2" sx={{ color: 'primary.main', fontWeight: 'bold' }}>
                      Processing embeddings batch
                    </Typography>
                    <Typography variant="body2" sx={{ color: 'text.primary' }}>
                      {uploadState.details.embedding_progress}/{uploadState.details.total_embeddings}
                    </Typography>
                  </Box>
                  <LinearProgress 
                    variant="determinate" 
                    value={(uploadState.details.embedding_progress / uploadState.details.total_embeddings) * 100}
                    sx={{ 
                      height: 10, 
                      borderRadius: 5,
                      backgroundColor: 'action.hover',
                      '& .MuiLinearProgress-bar': {
                        backgroundColor: 'info.main',
                      }
                    }}
                  />
                  <Typography variant="caption" sx={{ color: 'text.secondary', mt: 1, display: 'block' }}>
                    {Math.round((uploadState.details.embedding_progress / uploadState.details.total_embeddings) * 100)}% complete
                  </Typography>
                </Box>
              </Box>
            )}

            {/* Detailed Progress Info */}
            {uploadState.details && (
              <Box sx={{ 
                mt: 2, 
                p: 3, 
                bgcolor: 'action.hover', 
                borderRadius: 2,
                border: '1px solid',
                borderColor: 'divider'
              }}>
                {uploadState.details.chunks_to_process && (
                  <Typography variant="body1" sx={{ color: 'text.primary', mb: 1 }} display="block">
                    📄 Chunks to process: <Typography component="span" sx={{ color: 'primary.main', fontWeight: 'bold' }}>{uploadState.details.chunks_to_process}</Typography>
                  </Typography>
                )}
                {uploadState.details.embedding_progress && uploadState.details.total_embeddings && (
                  <Typography variant="body1" sx={{ color: 'text.primary', mb: 1 }} display="block">
                    🔮 Embeddings: <Typography component="span" sx={{ color: 'primary.main', fontWeight: 'bold' }}>{uploadState.details.embedding_progress}/{uploadState.details.total_embeddings}</Typography>
                  </Typography>
                )}
                {uploadState.details.unique_chunks && (
                  <Typography variant="body1" sx={{ color: 'text.primary', mb: 1 }} display="block">
                    ✨ Unique chunks: <Typography component="span" sx={{ color: 'primary.main', fontWeight: 'bold' }}>{uploadState.details.unique_chunks}</Typography>
                    {uploadState.details.duplicates && <Typography component="span" sx={{ color: 'warning.main' }}> ({uploadState.details.duplicates} duplicates filtered)</Typography>}
                  </Typography>
                )}
                {uploadState.details.status === 'success' && (
                  <Typography variant="body1" sx={{ color: 'success.main', mt: 2, fontWeight: 'bold' }} display="block">
                    ✅ <strong>{uploadState.details.unique_chunks_inserted} chunks inserted successfully!</strong>
                  </Typography>
                )}
              </Box>
            )}

            {/* Keep modal open longer on success */}
            {uploadState.status === 'success' && (
              <Box sx={{ mt: 3, textAlign: 'center' }}>
                <Typography variant="h6" sx={{ color: 'success.main', fontWeight: 'bold' }}>
                  🎉 Upload Complete!
                </Typography>
              </Box>
            )}
          </Box>
        </Fade>
      </Modal>
    </Box>
  );
};

export default FileUploadComponent;