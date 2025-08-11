import React, { useState, useCallback, useEffect } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  CardActions,
  Grid,
  Typography,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Alert,
  LinearProgress,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Tooltip,
  SelectChangeEvent,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Divider,
  Skeleton,
  CircularProgress
} from '@mui/material';
import {
  CloudUpload,
  Delete,
  Edit,
  Visibility,
  Description,
  FilePresent,
  CheckCircle,
  Error as ErrorIcon,
  ExpandMore as ExpandMoreIcon,
  Preview as PreviewIcon,
  AccessTime as TimeIcon
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import { MessageContent } from '../shared/MessageContent';

interface ReferenceDocument {
  document_id: string;  // Primary identifier from API
  name: string;
  document_type: string;
  category?: string;
  original_filename?: string;  // Made optional as API might not always return it
  file_size_bytes: number;
  extraction_model?: string;  // Made optional as API might not always return it
  extraction_confidence?: number;
  extracted_markdown?: string;
  recommended_extraction_modes?: string[];
  created_at: string;
  updated_at: string;
  is_active: boolean;
  processing_status?: 'uploading' | 'extracting' | 'completed' | 'failed';
}

interface IDCReferenceManagerProps {
  references: ReferenceDocument[];
  onRefresh: () => void;
  onUploadComplete?: (document: ReferenceDocument) => void;
}

interface ReferenceDocumentContent {
  document_id: string;
  name: string;
  document_type: string;
  category?: string;
  content: string;
  content_preview: string;
  content_stats: {
    total_characters: number;
    total_words: number;
    total_lines: number;
    heading_count: number;
    code_block_count: number;
    list_item_count: number;
    estimated_reading_time_minutes: number;
  };
}

// Component for inline preview content
const InlinePreviewContent: React.FC<{ referenceId: string }> = ({ referenceId }) => {
  const [content, setContent] = useState<ReferenceDocumentContent | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchContent = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const response = await fetch(`/api/v1/idc/references/${referenceId}/content`);
        if (!response.ok) {
          throw new Error(`Failed to fetch content: ${response.statusText}`);
        }
        
        const data = await response.json();
        if (data.status === 'success') {
          const contentData: ReferenceDocumentContent = {
            document_id: data.document_id,
            name: data.name,
            document_type: data.document_type,
            category: data.category,
            content: data.content,
            content_preview: data.content_preview,
            content_stats: data.content_stats
          };
          setContent(contentData);
        } else {
          throw new Error(data.message || 'Failed to load reference content');
        }
      } catch (error) {
        console.error('Failed to fetch reference content:', error);
        setError(error instanceof Error ? error.message : 'Failed to load reference document content');
      } finally {
        setLoading(false);
      }
    };

    if (referenceId) {
      fetchContent();
    }
  }, [referenceId]);

  if (loading) {
    return (
      <Box sx={{ py: 2 }}>
        <Skeleton variant="text" width="80%" />
        <Skeleton variant="text" width="60%" />
        <Skeleton variant="text" width="90%" />
        <Skeleton variant="rectangular" height={200} sx={{ mt: 2 }} />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error">
        {error}
      </Alert>
    );
  }

  if (!content) {
    return (
      <Alert severity="info">
        No content available for this document
      </Alert>
    );
  }

  return (
    <Box>
      {/* Content Stats */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Description color="primary" />
          {content.name}
        </Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
          <Chip 
            label={`${content.content_stats.total_words.toLocaleString()} words`}
            size="small" 
            variant="outlined"
          />
          <Chip 
            label={`${content.content_stats.total_lines.toLocaleString()} lines`}
            size="small" 
            variant="outlined"
          />
          <Chip 
            label={`${content.content_stats.heading_count} headings`}
            size="small" 
            variant="outlined"
          />
          {content.content_stats.code_block_count > 0 && (
            <Chip 
              label={`${content.content_stats.code_block_count} code blocks`}
              size="small" 
              variant="outlined"
            />
          )}
          {content.content_stats.list_item_count > 0 && (
            <Chip 
              label={`${content.content_stats.list_item_count} list items`}
              size="small" 
              variant="outlined"
            />
          )}
          <Chip 
            icon={<TimeIcon fontSize="small" />}
            label={`${content.content_stats.estimated_reading_time_minutes} min read`}
            size="small" 
            color="info"
          />
        </Box>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          Type: {content.document_type}
          {content.category && ` • Category: ${content.category}`}
        </Typography>
        <Divider sx={{ mt: 2 }} />
      </Box>

      {/* Rendered Markdown Content */}
      <Paper 
        elevation={0} 
        sx={{ 
          p: 3, 
          backgroundColor: 'background.default',
          border: '1px solid',
          borderColor: 'divider',
          borderRadius: 2,
          maxHeight: '500px',
          overflow: 'auto'
        }}
      >
        <MessageContent content={content.content} />
      </Paper>
    </Box>
  );
};

const IDCReferenceManager: React.FC<IDCReferenceManagerProps> = ({
  references,
  onRefresh,
  onUploadComplete
}) => {
  const [availableModels, setAvailableModels] = useState<Array<{name: string, id: string, size: string, modified: string, context_length: string}>>([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [previewDialogOpen, setPreviewDialogOpen] = useState(false);
  const [selectedReference, setSelectedReference] = useState<ReferenceDocument | null>(null);
  const [loadingPreviewDialog, setLoadingPreviewDialog] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [extractionProgress, setExtractionProgress] = useState(0);
  const [updating, setUpdating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Inline preview state
  const [selectedForPreview, setSelectedForPreview] = useState<string>('');
  const [showInlinePreview, setShowInlinePreview] = useState(false);
  const [loadingPreview, setLoadingPreview] = useState(false);
  
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    document_type: 'guidelines',
    category: '',
    extraction_model: 'qwen3:30b-a3b-q4_K_M',
    file: null as File | null
  });

  const [editFormData, setEditFormData] = useState({
    name: '',
    document_type: 'guidelines',
    category: '',
    extraction_model: 'qwen3:30b-a3b-q4_K_M',
    content: '',
    reExtract: false
  });

  useEffect(() => {
    fetchAvailableModels();
  }, []);

  const fetchAvailableModels = async () => {
    setLoadingModels(true);
    try {
      const response = await fetch('/api/v1/ollama/models');
      if (response.ok) {
        const data = await response.json();
        
        let modelsList: any[] = [];
        if (data.success && data.models) {
          // Ollama is available - use actual models
          modelsList = data.models;
        } else if (data.fallback_models && data.fallback_models.length > 0) {
          // Ollama not available - use fallback models from API
          modelsList = data.fallback_models;
        } else {
          // Last resort fallback
          modelsList = [
            { name: 'llama3.1:8b', id: 'fallback-01', size: '4.7 GB', modified: 'N/A', context_length: '128,000' },
            { name: 'deepseek-r1:8b', id: 'fallback-04', size: '4.9 GB', modified: 'N/A', context_length: '65,536' },
            { name: 'qwen2.5:32b', id: 'fallback-03', size: '19 GB', modified: 'N/A', context_length: '32,768' }
          ];
        }
        
        setAvailableModels(modelsList);
        
        // Set default model if available
        const modelNames = modelsList.map((m: any) => m.name);
        if (modelNames.length > 0 && !modelNames.includes(formData.extraction_model)) {
          setFormData(prev => ({
            ...prev,
            extraction_model: modelNames[0] || 'llama3.1:8b'
          }));
        }
      } else {
        // HTTP error - try to parse response for fallback models
        try {
          const data = await response.json();
          const fallbackModels = data.fallback_models && data.fallback_models.length > 0 
            ? data.fallback_models 
            : [
                { name: 'llama3.1:8b', id: 'fallback-01', size: '4.7 GB', modified: 'N/A', context_length: '128,000' },
                { name: 'deepseek-r1:8b', id: 'fallback-04', size: '4.9 GB', modified: 'N/A', context_length: '65,536' }
              ];
          setAvailableModels(fallbackModels);
        } catch {
          // Last resort fallback
          const fallbackModels = [
            { name: 'llama3.1:8b', id: 'fallback-01', size: '4.7 GB', modified: 'N/A', context_length: '128,000' },
            { name: 'deepseek-r1:8b', id: 'fallback-04', size: '4.9 GB', modified: 'N/A', context_length: '65,536' }
          ];
          setAvailableModels(fallbackModels);
        }
      }
    } catch (error) {
      console.error('Failed to fetch models:', error);
      // Use fallback models as last resort
      const fallbackModels = [
        { name: 'llama3.1:8b', id: 'fallback-01', size: '4.7 GB', modified: 'N/A', context_length: '128,000' },
        { name: 'deepseek-r1:8b', id: 'fallback-04', size: '4.9 GB', modified: 'N/A', context_length: '65,536' },
        { name: 'qwen2.5:32b', id: 'fallback-03', size: '19 GB', modified: 'N/A', context_length: '32,768' }
      ];
      setAvailableModels(fallbackModels);
    } finally {
      setLoadingModels(false);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        const file = acceptedFiles[0];
        setFormData({ 
          ...formData, 
          file,
          name: formData.name || file.name.replace(/\.[^/.]+$/, '')
        });
      }
    },
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/msword': ['.doc'],
      'text/plain': ['.txt'],
      'text/markdown': ['.md']
    },
    maxFiles: 1
  });

  const handleUpload = async () => {
    if (!formData.file || !formData.name) {
      setError('Please provide a name and select a file');
      return;
    }

    setUploading(true);
    setUploadProgress(0);
    setExtractionProgress(0);
    setError(null);

    try {
      const data = new FormData();
      data.append('file', formData.file);
      data.append('name', formData.name);
      data.append('document_type', formData.document_type);
      data.append('category', formData.category);
      data.append('extraction_model', formData.extraction_model);

      // Simulate upload progress
      const uploadInterval = setInterval(() => {
        setUploadProgress(prev => Math.min(prev + 10, 90));
      }, 200);

      const response = await fetch('/api/v1/idc/reference/upload', {
        method: 'POST',
        body: data
      });

      clearInterval(uploadInterval);
      setUploadProgress(100);

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      // Simulate extraction progress
      setExtractionProgress(10);
      const extractionInterval = setInterval(() => {
        setExtractionProgress(prev => Math.min(prev + 15, 95));
      }, 500);

      const result = await response.json();
      
      clearInterval(extractionInterval);
      setExtractionProgress(100);

      if (onUploadComplete) {
        onUploadComplete(result);
      }

      // Reset form
      setFormData({
        name: '',
        description: '',
        document_type: 'guidelines',
        category: '',
        extraction_model: availableModels.length > 0 ? availableModels[0].name : 'qwen3:30b-a3b-q4_K_M',
        file: null
      });
      
      setUploadDialogOpen(false);
      onRefresh();
    } catch (error) {
      console.error('Upload failed:', error);
      setError(error instanceof Error ? error.message : 'Upload failed');
    } finally {
      setUploading(false);
      setUploadProgress(0);
      setExtractionProgress(0);
    }
  };

  const handleDelete = async (id: string) => {
    if (!window.confirm('Are you sure you want to delete this reference document?')) {
      return;
    }

    try {
      const response = await fetch(`/api/v1/idc/reference/${id}`, {
        method: 'DELETE'
      });

      if (!response.ok) {
        throw new Error('Delete failed');
      }

      onRefresh();
    } catch (error) {
      console.error('Delete failed:', error);
      setError('Failed to delete reference document');
    }
  };

  const handleEdit = async (reference: ReferenceDocument) => {
    setUpdating(true);
    setError(null);
    
    try {
      // Fetch the full content for editing
      const response = await fetch(`/api/v1/idc/references/${reference.document_id}/content`);
      if (response.ok) {
        const data = await response.json();
        if (data.status === 'success') {
          // Populate edit form with current reference data including content
          setEditFormData({
            name: reference.name,
            document_type: reference.document_type,
            category: reference.category || '',
            extraction_model: reference.extraction_model || (availableModels.length > 0 ? availableModels[0].name : 'qwen3:30b-a3b-q4_K_M'),
            content: data.content || '',
            reExtract: false
          });
          setSelectedReference(reference);
          setEditDialogOpen(true);
        } else {
          throw new Error(data.message || 'Failed to load reference content');
        }
      } else {
        // Fallback - use existing data without content editing
        setEditFormData({
          name: reference.name,
          document_type: reference.document_type,
          category: reference.category || '',
          extraction_model: reference.extraction_model || (availableModels.length > 0 ? availableModels[0].name : 'qwen3:30b-a3b-q4_K_M'),
          content: '',
          reExtract: false
        });
        setSelectedReference(reference);
        setEditDialogOpen(true);
        setError('Could not load document content for editing. You can still update metadata.');
      }
    } catch (error) {
      console.error('Failed to fetch reference content for editing:', error);
      // Fallback - use existing data without content editing
      setEditFormData({
        name: reference.name,
        document_type: reference.document_type,
        category: reference.category || '',
        extraction_model: reference.extraction_model || (availableModels.length > 0 ? availableModels[0].name : 'qwen3:30b-a3b-q4_K_M'),
        content: '',
        reExtract: false
      });
      setSelectedReference(reference);
      setEditDialogOpen(true);
      setError('Could not load document content for editing. You can still update metadata.');
    } finally {
      setUpdating(false);
    }
  };

  const handleUpdate = async () => {
    if (!selectedReference || !editFormData.name) {
      setError('Please provide a name');
      return;
    }

    setUpdating(true);
    setError(null);

    try {
      const updateData = {
        name: editFormData.name,
        document_type: editFormData.document_type,
        category: editFormData.category,
        extraction_model: editFormData.extraction_model,
        content: editFormData.content,
        re_extract: editFormData.reExtract
      };

      const response = await fetch(`/api/v1/idc/reference/${selectedReference.document_id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(updateData)
      });

      if (!response.ok) {
        throw new Error(`Update failed: ${response.statusText}`);
      }

      const result = await response.json();

      // Reset form and close dialog
      setEditFormData({
        name: '',
        document_type: 'guidelines',
        category: '',
        extraction_model: availableModels.length > 0 ? availableModels[0].name : 'qwen3:30b-a3b-q4_K_M',
        content: '',
        reExtract: false
      });
      
      setEditDialogOpen(false);
      setSelectedReference(null);
      onRefresh();
    } catch (error) {
      console.error('Update failed:', error);
      setError(error instanceof Error ? error.message : 'Update failed');
    } finally {
      setUpdating(false);
    }
  };

  const handlePreview = async (reference: ReferenceDocument) => {
    setLoadingPreviewDialog(true);
    try {
      // First try to fetch the full content
      // Use document_id which is the actual field from API
      const response = await fetch(`/api/v1/idc/references/${reference.document_id}/content`);
      if (response.ok) {
        const data = await response.json();
        if (data.status === 'success') {
          // Create an enhanced reference with the full content
          const enhancedReference = {
            ...reference,
            extracted_markdown: data.content
          };
          setSelectedReference(enhancedReference);
          setPreviewDialogOpen(true);
          return;
        }
      }
      
      // Fallback to existing markdown if API fails
      setSelectedReference(reference);
      setPreviewDialogOpen(true);
    } catch (error) {
      console.error('Error fetching preview content:', error);
      // Still show the preview dialog with existing data
      setSelectedReference(reference);
      setPreviewDialogOpen(true);
    } finally {
      setLoadingPreviewDialog(false);
    }
  };

  const handleInlinePreview = (reference: ReferenceDocument) => {
    if (selectedForPreview === reference.document_id && showInlinePreview) {
      // If same document is already selected and showing, toggle off
      setShowInlinePreview(false);
      setSelectedForPreview('');
    } else {
      // Select new document and show preview
      setSelectedForPreview(reference.document_id);
      setShowInlinePreview(true);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  const getDocumentTypeColor = (type: string) => {
    switch (type) {
      case 'guidelines': return 'primary';
      case 'qa_sheet': return 'secondary';
      case 'template': return 'success';
      case 'rules': return 'warning';
      default: return 'default';
    }
  };

  return (
    <Box>
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h5">Reference Documents</Typography>
        <Button
          variant="contained"
          startIcon={<CloudUpload />}
          onClick={() => setUploadDialogOpen(true)}
          sx={{ textTransform: 'none' }}
        >
          Upload Reference
        </Button>
      </Box>

      {error && (
        <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {references.map((ref) => (
          <Grid item xs={12} sm={6} md={4} key={ref.document_id}>
            <Card elevation={2} sx={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: '300px' }}>
              <CardContent sx={{ flex: 1, pb: 1 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="h6" gutterBottom noWrap>
                    {ref.name}
                  </Typography>
                  {ref.processing_status === 'completed' && (
                    <CheckCircle color="success" fontSize="small" />
                  )}
                  {ref.processing_status === 'failed' && (
                    <ErrorIcon color="error" fontSize="small" />
                  )}
                </Box>
                
                <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
                  <Chip
                    label={ref.document_type.replace('_', ' ')}
                    size="small"
                    color={getDocumentTypeColor(ref.document_type) as any}
                  />
                  {ref.category && (
                    <Chip label={ref.category} size="small" variant="outlined" />
                  )}
                  <Chip 
                    label={ref.extraction_model} 
                    size="small" 
                    variant="outlined"
                    color="info"
                  />
                </Box>

                <Typography variant="body2" color="text.secondary" gutterBottom>
                  <FilePresent fontSize="small" sx={{ verticalAlign: 'middle', mr: 0.5 }} />
                  {ref.original_filename}
                </Typography>
                
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Size: {formatFileSize(ref.file_size_bytes)}
                </Typography>
                
                {ref.extraction_confidence && (
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Confidence: {(ref.extraction_confidence * 100).toFixed(1)}%
                  </Typography>
                )}
                
                {ref.recommended_extraction_modes && ref.recommended_extraction_modes.length > 0 && (
                  <Box sx={{ mt: 1 }}>
                    <Typography variant="caption" color="text.secondary">
                      Recommended modes:
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 0.5, mt: 0.5, flexWrap: 'wrap' }}>
                      {ref.recommended_extraction_modes.map(mode => (
                        <Chip key={mode} label={mode} size="small" variant="outlined" />
                      ))}
                    </Box>
                  </Box>
                )}
                
                <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                  Created: {new Date(ref.created_at).toLocaleDateString()}
                </Typography>
              </CardContent>
              
              <CardActions 
                sx={{ 
                  justifyContent: 'flex-start', 
                  px: 2, 
                  py: 1.5,
                  borderTop: '1px solid',
                  borderColor: 'divider',
                  backgroundColor: (theme) => theme.palette.mode === 'dark' ? theme.palette.grey[900] : theme.palette.grey[50],
                  gap: 0.5,
                  position: 'relative',
                  zIndex: 1
                }}
              >
                <Tooltip title="View extracted content">
                  <IconButton 
                    size="small" 
                    color="primary"
                    disabled={loadingPreviewDialog}
                    onClick={async (e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      console.log('Preview button clicked for ref:', ref.document_id, ref.name);
                      await handlePreview(ref);
                    }}
                  >
                    {loadingPreviewDialog ? <CircularProgress size={16} /> : <Visibility />}
                  </IconButton>
                </Tooltip>
                <Tooltip title="Preview inline">
                  <IconButton 
                    size="small" 
                    color={selectedForPreview === ref.document_id ? "secondary" : "primary"}
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      console.log('Inline preview button clicked for ref:', ref.document_id, ref.name);
                      handleInlinePreview(ref);
                    }}
                  >
                    <PreviewIcon />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Edit reference">
                  <IconButton 
                    size="small" 
                    color="primary"
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      console.log('Edit button clicked for ref:', ref.document_id, ref.name);
                      handleEdit(ref);
                    }}
                  >
                    <Edit />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Delete reference">
                  <IconButton 
                    size="small" 
                    color="error"
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      console.log('Delete button clicked for ref:', ref.document_id, ref.name);
                      handleDelete(ref.document_id);
                    }}
                  >
                    <Delete />
                  </IconButton>
                </Tooltip>
              </CardActions>
            </Card>
          </Grid>
        ))}
        
        {references.length === 0 && (
          <Grid item xs={12}>
            <Paper sx={{ p: 4, textAlign: 'center' }}>
              <Description sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
              <Typography variant="h6" color="text.secondary">
                No reference documents uploaded yet
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Upload your first reference document to get started
              </Typography>
            </Paper>
          </Grid>
        )}
      </Grid>

      {/* Inline Preview Panel */}
      {selectedForPreview && showInlinePreview && (
        <Paper sx={{ p: 3, mb: 3, mt: 3 }}>
          <Accordion 
            expanded={showInlinePreview} 
            onChange={() => setShowInlinePreview(!showInlinePreview)}
            sx={{ backgroundColor: 'transparent', boxShadow: 'none' }}
          >
            <AccordionSummary
              expandIcon={<ExpandMoreIcon />}
              sx={{ px: 0, minHeight: 'auto' }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <PreviewIcon color="primary" />
                <Typography variant="h6">
                  Document Preview: {references.find(r => r.document_id === selectedForPreview)?.name}
                </Typography>
                {loadingPreview && <CircularProgress size={16} />}
              </Box>
            </AccordionSummary>
            
            <AccordionDetails sx={{ px: 0 }}>
              <InlinePreviewContent referenceId={selectedForPreview} />
            </AccordionDetails>
          </Accordion>
        </Paper>
      )}

      {/* Upload Dialog */}
      <Dialog
        open={uploadDialogOpen}
        onClose={() => !uploading && setUploadDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Upload Reference Document</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            label="Name"
            value={formData.name}
            onChange={(e) => setFormData({ ...formData, name: e.target.value })}
            margin="normal"
            required
            disabled={uploading}
          />
          
          <TextField
            fullWidth
            label="Category (Optional)"
            value={formData.category}
            onChange={(e) => setFormData({ ...formData, category: e.target.value })}
            margin="normal"
            disabled={uploading}
            placeholder="e.g., Legal, Technical, Educational"
          />
          
          <FormControl fullWidth margin="normal">
            <InputLabel>Document Type</InputLabel>
            <Select
              value={formData.document_type}
              onChange={(e: SelectChangeEvent) => setFormData({ ...formData, document_type: e.target.value })}
              label="Document Type"
              disabled={uploading}
            >
              <MenuItem value="guidelines">Guidelines</MenuItem>
              <MenuItem value="qa_sheet">Q&A Sheet</MenuItem>
              <MenuItem value="template">Template</MenuItem>
              <MenuItem value="rules">Rules</MenuItem>
            </Select>
          </FormControl>
          
          <FormControl fullWidth margin="normal">
            <InputLabel>Extraction Model</InputLabel>
            <Select
              value={formData.extraction_model}
              onChange={(e: SelectChangeEvent) => setFormData({ ...formData, extraction_model: e.target.value })}
              label="Extraction Model"
              disabled={uploading || loadingModels}
            >
              {loadingModels ? (
                <MenuItem disabled>Loading models...</MenuItem>
              ) : (
                availableModels.map(model => (
                  <MenuItem key={model.name} value={model.name}>{model.name}</MenuItem>
                ))
              )}
            </Select>
          </FormControl>

          <Box
            {...getRootProps()}
            sx={{
              mt: 2,
              p: 3,
              border: '2px dashed',
              borderColor: isDragActive ? 'primary.main' : 'grey.400',
              borderRadius: 1,
              textAlign: 'center',
              cursor: uploading ? 'not-allowed' : 'pointer',
              backgroundColor: isDragActive ? 'action.hover' : 'background.paper',
              opacity: uploading ? 0.6 : 1
            }}
          >
            <input {...getInputProps()} disabled={uploading} />
            <CloudUpload sx={{ fontSize: 48, color: 'text.secondary', mb: 1 }} />
            {formData.file ? (
              <Typography>{formData.file.name}</Typography>
            ) : (
              <Typography color="text.secondary">
                {isDragActive ? 'Drop the file here' : 'Drag & drop or click to select'}
              </Typography>
            )}
            <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
              Supported: PDF, DOCX, DOC, TXT, MD
            </Typography>
          </Box>

          {uploading && (
            <Box sx={{ mt: 3 }}>
              <Typography variant="body2" gutterBottom>
                Uploading document...
              </Typography>
              <LinearProgress variant="determinate" value={uploadProgress} sx={{ mb: 2 }} />
              
              {uploadProgress === 100 && (
                <>
                  <Typography variant="body2" gutterBottom>
                    Extracting content to markdown...
                  </Typography>
                  <LinearProgress variant="determinate" value={extractionProgress} />
                </>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setUploadDialogOpen(false)} disabled={uploading}>
            Cancel
          </Button>
          <Button
            onClick={handleUpload}
            variant="contained"
            disabled={!formData.file || !formData.name || uploading}
          >
            {uploading ? 'Processing...' : 'Upload & Extract'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Edit Dialog */}
      <Dialog
        open={editDialogOpen}
        onClose={() => !updating && setEditDialogOpen(false)}
        maxWidth="lg"
        fullWidth
        PaperProps={{
          sx: { height: '90vh' }
        }}
      >
        <DialogTitle>Edit Reference Document</DialogTitle>
        <DialogContent sx={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
          <Box sx={{ flexGrow: 1, overflow: 'auto', pt: 1 }}>
            <TextField
            fullWidth
            label="Name"
            value={editFormData.name}
            onChange={(e) => setEditFormData({ ...editFormData, name: e.target.value })}
            margin="normal"
            required
            disabled={updating}
          />
          
          <TextField
            fullWidth
            label="Category (Optional)"
            value={editFormData.category}
            onChange={(e) => setEditFormData({ ...editFormData, category: e.target.value })}
            margin="normal"
            disabled={updating}
            placeholder="e.g., Legal, Technical, Educational"
          />
          
          <FormControl fullWidth margin="normal">
            <InputLabel>Document Type</InputLabel>
            <Select
              value={editFormData.document_type}
              onChange={(e: SelectChangeEvent) => setEditFormData({ ...editFormData, document_type: e.target.value })}
              label="Document Type"
              disabled={updating}
            >
              <MenuItem value="guidelines">Guidelines</MenuItem>
              <MenuItem value="qa_sheet">Q&A Sheet</MenuItem>
              <MenuItem value="template">Template</MenuItem>
              <MenuItem value="rules">Rules</MenuItem>
            </Select>
          </FormControl>
          
          <FormControl fullWidth margin="normal">
            <InputLabel>Extraction Model</InputLabel>
            <Select
              value={editFormData.extraction_model}
              onChange={(e: SelectChangeEvent) => setEditFormData({ ...editFormData, extraction_model: e.target.value })}
              label="Extraction Model"
              disabled={updating || loadingModels}
            >
              {loadingModels ? (
                <MenuItem disabled>Loading models...</MenuItem>
              ) : (
                availableModels.map(model => (
                  <MenuItem key={model.name} value={model.name}>{model.name}</MenuItem>
                ))
              )}
            </Select>
          </FormControl>

          <TextField
            fullWidth
            label="Markdown Content"
            value={editFormData.content}
            onChange={(e) => setEditFormData({ ...editFormData, content: e.target.value })}
            margin="normal"
            multiline
            rows={12}
            disabled={updating}
            placeholder="Edit the markdown content of the document..."
            helperText="You can edit the extracted markdown content directly. Changes will be saved when you update."
          />

          <Box sx={{ mt: 2, mb: 1 }}>
            <FormControl>
              <label>
                <input
                  type="checkbox"
                  checked={editFormData.reExtract}
                  onChange={(e) => setEditFormData({ ...editFormData, reExtract: e.target.checked })}
                  disabled={updating}
                  style={{ marginRight: '8px' }}
                />
                Re-extract content with new model (will overwrite content above)
              </label>
            </FormControl>
          </Box>
          
          {editFormData.reExtract && (
            <Alert severity="info" sx={{ mt: 1 }}>
              Re-extraction will process the document with the selected model and may take some time.
            </Alert>
          )}

          {updating && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" gutterBottom>
                {editFormData.reExtract ? 'Updating document and re-extracting content...' : 'Updating document...'}
              </Typography>
              <LinearProgress />
            </Box>
          )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialogOpen(false)} disabled={updating}>
            Cancel
          </Button>
          <Button
            onClick={handleUpdate}
            variant="contained"
            disabled={!editFormData.name || updating}
          >
            {updating ? 'Updating...' : 'Update'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Preview Dialog */}
      <Dialog
        open={previewDialogOpen}
        onClose={() => setPreviewDialogOpen(false)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>
          {selectedReference?.name} - Extracted Content Preview
        </DialogTitle>
        <DialogContent>
          {selectedReference?.extracted_markdown ? (
            <Box>
              {/* Document stats */}
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Document Statistics
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                  <Chip 
                    label={`${selectedReference?.extracted_markdown?.length?.toLocaleString() || 0} characters`}
                    size="small" 
                    variant="outlined"
                  />
                  <Chip 
                    label={`${selectedReference?.extracted_markdown?.split(' ')?.length?.toLocaleString() || 0} words`}
                    size="small" 
                    variant="outlined"
                  />
                  <Chip 
                    label={`${selectedReference?.extracted_markdown?.split('\n')?.length?.toLocaleString() || 0} lines`}
                    size="small" 
                    variant="outlined"
                  />
                  <Chip 
                    label={`${Math.ceil((selectedReference?.extracted_markdown?.split(' ')?.length || 0) / 200)} min read`}
                    size="small" 
                    color="info"
                    icon={<TimeIcon fontSize="small" />}
                  />
                </Box>
                <Divider sx={{ mb: 2 }} />
              </Box>
              
              {/* Rendered Markdown Content */}
              <Paper 
                elevation={0} 
                sx={{ 
                  p: 3, 
                  backgroundColor: 'background.default',
                  border: '1px solid',
                  borderColor: 'divider',
                  borderRadius: 2,
                  maxHeight: '70vh',
                  overflow: 'auto'
                }}
              >
                <MessageContent content={selectedReference?.extracted_markdown || ''} />
              </Paper>
            </Box>
          ) : (
            <Box>
              <Alert severity="warning" sx={{ mb: 2 }}>
                No extracted content available for this document. This could mean:
              </Alert>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                • The document is still processing
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                • The extraction failed
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                • The document type is not supported for content extraction
              </Typography>
              
              <Paper 
                elevation={0} 
                sx={{ 
                  p: 3, 
                  backgroundColor: 'background.default',
                  border: '1px solid',
                  borderColor: 'divider',
                  borderRadius: 2
                }}
              >
                <Typography variant="h6" gutterBottom>
                  Document Information
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>Name:</strong> {selectedReference?.name || 'Unknown'}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>Type:</strong> {selectedReference?.document_type || 'Unknown'}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>Original File:</strong> {selectedReference?.original_filename || 'Unknown'}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>Size:</strong> {selectedReference?.file_size_bytes ? formatFileSize(selectedReference?.file_size_bytes) : 'Unknown'}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>Status:</strong> {selectedReference?.processing_status || 'Unknown'}
                </Typography>
                {selectedReference?.extraction_model && (
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Extraction Model:</strong> {selectedReference?.extraction_model}
                  </Typography>
                )}
                {selectedReference?.extraction_confidence && (
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Extraction Confidence:</strong> {(selectedReference?.extraction_confidence * 100).toFixed(1)}%
                  </Typography>
                )}
              </Paper>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setPreviewDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default IDCReferenceManager;