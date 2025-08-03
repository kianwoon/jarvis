import React, { useState, useCallback, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  Typography,
  Button,
  Alert,
  LinearProgress,
  Chip,
  Grid,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Switch,
  Divider,
  Avatar,
  CircularProgress,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Badge,
  Tooltip,
  Fab,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  Description as DocumentIcon,
  Analytics as AnalyticsIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  Visibility as VisibilityIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  Storage as StorageIcon,
  AccountTree as GraphIcon,
  Timeline as TimelineIcon,
  Speed as SpeedIcon,
  ExpandMore as ExpandMoreIcon,
  Cancel as CancelIcon,
  PlayArrow as ProcessIcon,
  Pause as PauseIcon,
  Settings as SettingsIcon,
  Download as DownloadIcon,
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';

interface ProcessingProgress {
  document_id: string;
  filename: string;
  total_steps: number;
  current_step: number;
  step_name: string;
  chunks_processed: number;
  total_chunks: number;
  entities_extracted: number;
  relationships_extracted: number;
  processing_time_ms: number;
  status: string;
  error_message?: string;
  warnings?: string[];
}

interface UploadedDocument {
  id: string;
  filename: string;
  size: number;
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled';
  progress?: ProcessingProgress;
  result?: any;
  uploadTime: Date;
}

const UnifiedDocumentUpload: React.FC = () => {
  const [documents, setDocuments] = useState<UploadedDocument[]>([]);
  const [processingMode, setProcessingMode] = useState<'unified' | 'milvus-only' | 'neo4j-only'>('unified');
  const [collectionName, setCollectionName] = useState('default');
  const [enableLLMEnhancement, setEnableLLMEnhancement] = useState(true);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  const [batchProcessing, setBatchProcessing] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);
  const [activeStep, setActiveStep] = useState(0);

  const steps = [
    'Upload Documents',
    'Configure Processing',
    'Process & Extract',
    'Review Results'
  ];

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newDocuments = acceptedFiles.map((file) => ({
      id: Date.now() + Math.random().toString(36),
      filename: file.name,
      size: file.size,
      status: 'pending' as const,
      uploadTime: new Date()
    }));
    
    setDocuments(prev => [...prev, ...newDocuments]);
    
    if (!batchProcessing) {
      // Auto-process single files
      newDocuments.forEach(doc => processDocument(file, doc));
    }
  }, [batchProcessing]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingfile.document': ['.docx'],
      'text/plain': ['.txt']
    },
    multiple: true
  });

  const processDocument = async (file: File, document: UploadedDocument) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('processing_mode', processingMode);
    formData.append('collection_name', collectionName);
    formData.append('enable_llm_enhancement', enableLLMEnhancement.toString());

    // Update document status
    setDocuments(prev => prev.map(doc => 
      doc.id === document.id 
        ? { ...doc, status: 'processing' }
        : doc
    ));

    try {
      // Start SSE connection for real-time updates
      const eventSource = new EventSource(`/api/v1/documents/process-unified/${document.id}/progress`);
      eventSourceRef.current = eventSource;

      eventSource.onmessage = (event) => {
        const progress: ProcessingProgress = JSON.parse(event.data);
        
        setDocuments(prev => prev.map(doc =>
          doc.id === document.id
            ? { 
                ...doc, 
                progress,
                status: progress.status === 'completed' ? 'completed' : 
                       progress.status === 'failed' ? 'failed' : 'processing'
              }
            : doc
        ));

        if (progress.status === 'completed' || progress.status === 'failed') {
          eventSource.close();
        }
      };

      eventSource.onerror = () => {
        console.error('SSE connection error');
        eventSource.close();
      };

      // Start processing
      const response = await fetch('/api/v1/documents/process-unified', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Processing failed: ${response.statusText}`);
      }

      const result = await response.json();
      
      setDocuments(prev => prev.map(doc =>
        doc.id === document.id
          ? { ...doc, result, status: result.success ? 'completed' : 'failed' }
          : doc
      ));

    } catch (error) {
      console.error('Processing error:', error);
      setDocuments(prev => prev.map(doc =>
        doc.id === document.id
          ? { 
              ...doc, 
              status: 'failed',
              progress: {
                ...document.progress!,
                status: 'failed',
                error_message: error instanceof Error ? error.message : 'Unknown error'
              }
            }
          : doc
      ));
    }
  };

  const processBatch = async () => {
    const pendingDocs = documents.filter(doc => doc.status === 'pending');
    
    for (const doc of pendingDocs) {
      // Process each document with delay to avoid overwhelming the system
      await new Promise(resolve => setTimeout(resolve, 1000));
      // Note: In real implementation, you'd need to access the original File objects
      // This would require storing them in state or using a different approach
    }
  };

  const cancelProcessing = (documentId: string) => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }
    
    setDocuments(prev => prev.map(doc =>
      doc.id === documentId
        ? { ...doc, status: 'cancelled' }
        : doc
    ));
  };

  const removeDocument = (documentId: string) => {
    setDocuments(prev => prev.filter(doc => doc.id !== documentId));
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success';
      case 'failed': return 'error';
      case 'processing': return 'primary';
      case 'cancelled': return 'warning';
      default: return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircleIcon />;
      case 'failed': return <ErrorIcon />;
      case 'processing': return <CircularProgress size={20} />;
      case 'cancelled': return <CancelIcon />;
      default: return <InfoIcon />;
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDuration = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}min`;
  };

  return (
    <Box sx={{ maxWidth: 1200, mx: 'auto', p: 3 }}>
      {/* Header */}
      <Card sx={{ mb: 3 }}>
        <CardHeader
          title="Unified Document Processing"
          subheader="Process documents simultaneously into Milvus (vector search) and Neo4j (knowledge graph)"
          avatar={<Avatar><UploadIcon /></Avatar>}
          action={
            <Chip 
              label={processingMode === 'unified' ? 'Unified Mode' : 
                     processingMode === 'milvus-only' ? 'Vector Only' : 'Graph Only'}
              color="primary"
            />
          }
        />
      </Card>

      {/* Processing Configuration */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Grid container spacing={3}>
            {/* Basic Configuration */}
            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>Processing Configuration</Typography>
              
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Processing Mode</InputLabel>
                <Select
                  value={processingMode}
                  onChange={(e) => setProcessingMode(e.target.value as any)}
                >
                  <MenuItem value="unified">
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Badge badgeContent="Recommended" color="primary">
                        <StorageIcon sx={{ mr: 1 }} />
                      </Badge>
                      <Box sx={{ ml: 2 }}>
                        <Typography>Unified (Milvus + Neo4j)</Typography>
                        <Typography variant="caption" color="text.secondary">
                          Full processing with vector search and knowledge graph
                        </Typography>
                      </Box>
                    </Box>
                  </MenuItem>
                  <MenuItem value="milvus-only">
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <StorageIcon sx={{ mr: 1 }} />
                      <Box sx={{ ml: 1 }}>
                        <Typography>Milvus Only</Typography>
                        <Typography variant="caption" color="text.secondary">
                          Vector embeddings for semantic search
                        </Typography>
                      </Box>
                    </Box>
                  </MenuItem>
                  <MenuItem value="neo4j-only">
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <GraphIcon sx={{ mr: 1 }} />
                      <Box sx={{ ml: 1 }}>
                        <Typography>Neo4j Only</Typography>
                        <Typography variant="caption" color="text.secondary">
                          Knowledge graph extraction only
                        </Typography>
                      </Box>
                    </Box>
                  </MenuItem>
                </Select>
              </FormControl>

              <FormControlLabel
                control={
                  <Switch
                    checked={enableLLMEnhancement}
                    onChange={(e) => setEnableLLMEnhancement(e.target.checked)}
                  />
                }
                label="Enable LLM Enhancement"
              />
            </Grid>

            {/* Advanced Options */}
            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>Options</Typography>
              
              <FormControlLabel
                control={
                  <Switch
                    checked={batchProcessing}
                    onChange={(e) => setBatchProcessing(e.target.checked)}
                  />
                }
                label="Batch Processing Mode"
              />
              
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Process multiple files together with optimized resource usage
              </Typography>

              <Button
                variant="outlined"
                startIcon={<SettingsIcon />}
                onClick={() => setShowAdvancedSettings(true)}
                size="small"
              >
                Advanced Settings
              </Button>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* File Upload Area */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box
            {...getRootProps()}
            sx={{
              border: '2px dashed',
              borderColor: isDragActive ? 'primary.main' : 'grey.300',
              borderRadius: 2,
              p: 4,
              textAlign: 'center',
              cursor: 'pointer',
              backgroundColor: isDragActive ? 'action.hover' : 'background.paper',
              transition: 'all 0.3s ease'
            }}
          >
            <input {...getInputProps()} />
            <UploadIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              {isDragActive ? 'Drop files here...' : 'Drop files here or click to browse'}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Supports PDF, DOC, DOCX, and TXT files
            </Typography>
            
            {batchProcessing && documents.filter(d => d.status === 'pending').length > 0 && (
              <Button
                variant="contained"
                startIcon={<ProcessIcon />}
                onClick={processBatch}
                sx={{ mt: 2 }}
              >
                Process Batch ({documents.filter(d => d.status === 'pending').length} files)
              </Button>
            )}
          </Box>
        </CardContent>
      </Card>

      {/* Document List */}
      {documents.length > 0 && (
        <Card>
          <CardHeader
            title={`Documents (${documents.length})`}
            action={
              <Box>
                <Chip 
                  label={`${documents.filter(d => d.status === 'completed').length} completed`}
                  color="success"
                  size="small"
                  sx={{ mr: 1 }}
                />
                <Chip 
                  label={`${documents.filter(d => d.status === 'processing').length} processing`}
                  color="primary"
                  size="small"
                />
              </Box>
            }
          />
          <CardContent>
            <List>
              {documents.map((doc) => (
                <Paper key={doc.id} sx={{ mb: 2, overflow: 'hidden' }}>
                  <ListItem>
                    <Avatar sx={{ mr: 2 }}>
                      <DocumentIcon />
                    </Avatar>
                    <ListItemText
                      primary={
                        <Box display="flex" alignItems="center">
                          <Typography variant="subtitle1">{doc.filename}</Typography>
                          <Chip
                            icon={getStatusIcon(doc.status)}
                            label={doc.status.charAt(0).toUpperCase() + doc.status.slice(1)}
                            color={getStatusColor(doc.status) as any}
                            size="small"
                            sx={{ ml: 2 }}
                          />
                        </Box>
                      }
                      secondary={
                        <Box>
                          <Typography variant="body2" color="text.secondary">
                            {formatFileSize(doc.size)} • Uploaded {doc.uploadTime.toLocaleTimeString()}
                          </Typography>
                          {doc.progress && (
                            <Box sx={{ mt: 1 }}>
                              <Typography variant="body2" gutterBottom>
                                Step {doc.progress.current_step}/{doc.progress.total_steps}: {doc.progress.step_name}
                              </Typography>
                              <LinearProgress
                                variant="determinate"
                                value={(doc.progress.current_step / doc.progress.total_steps) * 100}
                                sx={{ mb: 1 }}
                              />
                              <Grid container spacing={2}>
                                <Grid item xs={3}>
                                  <Typography variant="caption">
                                    Chunks: {doc.progress.chunks_processed}/{doc.progress.total_chunks}
                                  </Typography>
                                </Grid>
                                <Grid item xs={3}>
                                  <Typography variant="caption">
                                    Entities: {doc.progress.entities_extracted}
                                  </Typography>
                                </Grid>
                                <Grid item xs={3}>
                                  <Typography variant="caption">
                                    Relations: {doc.progress.relationships_extracted}
                                  </Typography>
                                </Grid>
                                <Grid item xs={3}>
                                  <Typography variant="caption">
                                    {doc.progress.processing_time_ms > 0 && 
                                      `Time: ${formatDuration(doc.progress.processing_time_ms)}`}
                                  </Typography>
                                </Grid>
                              </Grid>
                            </Box>
                          )}
                        </Box>
                      }
                    />
                    <ListItemSecondaryAction>
                      <Box>
                        {doc.status === 'processing' && (
                          <Tooltip title="Cancel Processing">
                            <IconButton onClick={() => cancelProcessing(doc.id)}>
                              <CancelIcon />
                            </IconButton>
                          </Tooltip>
                        )}
                        {doc.status === 'completed' && (
                          <Tooltip title="View Results">
                            <IconButton>
                              <VisibilityIcon />
                            </IconButton>
                          </Tooltip>
                        )}
                        <Tooltip title="Remove">
                          <IconButton onClick={() => removeDocument(doc.id)}>
                            <DeleteIcon />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </ListItemSecondaryAction>
                  </ListItem>

                  {/* Detailed Results */}
                  {doc.result && (
                    <Accordion>
                      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                        <Typography variant="subtitle2">Processing Results</Typography>
                      </AccordionSummary>
                      <AccordionDetails>
                        <Grid container spacing={2}>
                          <Grid item xs={12} sm={6} md={3}>
                            <Paper sx={{ p: 2, textAlign: 'center' }}>
                              <Typography variant="h6" color="primary">
                                {doc.result.total_chunks}
                              </Typography>
                              <Typography variant="body2">Chunks</Typography>
                            </Paper>
                          </Grid>
                          <Grid item xs={12} sm={6} md={3}>
                            <Paper sx={{ p: 2, textAlign: 'center' }}>
                              <Typography variant="h6" color="secondary">
                                {doc.result.entities_extracted}
                              </Typography>
                              <Typography variant="body2">Entities</Typography>
                            </Paper>
                          </Grid>
                          <Grid item xs={12} sm={6} md={3}>
                            <Paper sx={{ p: 2, textAlign: 'center' }}>
                              <Typography variant="h6" color="success.main">
                                {doc.result.relationships_extracted}
                              </Typography>
                              <Typography variant="body2">Relationships</Typography>
                            </Paper>
                          </Grid>
                          <Grid item xs={12} sm={6} md={3}>
                            <Paper sx={{ p: 2, textAlign: 'center' }}>
                              <Typography variant="h6" color="warning.main">
                                {(doc.result.extraction_confidence * 100).toFixed(1)}%
                              </Typography>
                              <Typography variant="body2">Confidence</Typography>
                            </Paper>
                          </Grid>
                        </Grid>

                        {processingMode === 'unified' && (
                          <Box sx={{ mt: 2 }}>
                            <Typography variant="subtitle2" gutterBottom>Storage Locations</Typography>
                            <Grid container spacing={1}>
                              <Grid item xs={6}>
                                <Chip 
                                  icon={<StorageIcon />}
                                  label={`Milvus: ${doc.result.milvus_collection || 'default'}`}
                                  size="small"
                                />
                              </Grid>
                              <Grid item xs={6}>
                                <Chip 
                                  icon={<GraphIcon />}
                                  label={`Neo4j: ${doc.result.neo4j_graph_id}`}
                                  size="small"
                                />
                              </Grid>
                            </Grid>
                          </Box>
                        )}
                      </AccordionDetails>
                    </Accordion>
                  )}

                  {/* Error Display */}
                  {doc.progress?.error_message && (
                    <Alert severity="error" sx={{ m: 2 }}>
                      {doc.progress.error_message}
                    </Alert>
                  )}

                  {/* Warnings */}
                  {doc.progress?.warnings && doc.progress.warnings.length > 0 && (
                    <Alert severity="warning" sx={{ m: 2 }}>
                      <Typography variant="body2">Warnings:</Typography>
                      <ul>
                        {doc.progress.warnings.map((warning, idx) => (
                          <li key={idx}>{warning}</li>
                        ))}
                      </ul>
                    </Alert>
                  )}
                </Paper>
              ))}
            </List>
          </CardContent>
        </Card>
      )}

      {/* Processing Statistics */}
      {documents.length > 0 && (
        <Box sx={{ position: 'fixed', bottom: 16, right: 16 }}>
          <Fab color="primary" variant="extended">
            <AnalyticsIcon sx={{ mr: 1 }} />
            {documents.filter(d => d.status === 'completed').length}/{documents.length} Complete
          </Fab>
        </Box>
      )}

      {/* Advanced Settings Dialog */}
      <Dialog 
        open={showAdvancedSettings} 
        onClose={() => setShowAdvancedSettings(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Advanced Processing Settings</DialogTitle>
        <DialogContent>
          <Grid container spacing={3}>
            <Grid item xs={12} sm={6}>
              <Typography variant="subtitle2" gutterBottom>Vector Database Settings</Typography>
              <TextField
                fullWidth
                label="Collection Name"
                value={collectionName}
                onChange={(e) => setCollectionName(e.target.value)}
                sx={{ mb: 2 }}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <Typography variant="subtitle2" gutterBottom>Knowledge Graph Settings</Typography>
              <FormControlLabel
                control={<Switch defaultChecked />}
                label="Enable Anti-Silo Relationships"
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowAdvancedSettings(false)}>Cancel</Button>
          <Button variant="contained" onClick={() => setShowAdvancedSettings(false)}>
            Apply Settings
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default UnifiedDocumentUpload;