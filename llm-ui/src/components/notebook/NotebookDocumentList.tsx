import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  IconButton,
  Button,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  InputAdornment,
  Alert,
  Chip,
  Skeleton,
  Checkbox,
  FormControlLabel,
  Divider,
  Card,
  CardContent,
  Menu,
  MenuItem,
  CircularProgress,
  LinearProgress,
  Tooltip,
  Switch,
  ToggleButton,
  ToggleButtonGroup,
  DialogContentText
} from '@mui/material';
import {
  Add as AddIcon,
  Upload as UploadIcon,
  Delete as DeleteIcon,
  Description as DocumentIcon,
  Search as SearchIcon,
  Close as CloseIcon,
  InsertDriveFile as FileIcon,
  PictureAsPdf as PdfIcon,
  Article as TextIcon,
  Image as ImageIcon,
  MoreVert as MoreVertIcon,
  CloudUpload as CloudUploadIcon,
  AccessTime as TimeIcon,
  Storage as SizeIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Psychology as EmbeddingIcon,
  DataObject as ProcessingIcon,
  SaveAlt as StoringIcon,
  Security as SecurityIcon,
  AdminPanelSettings as AdminPanelSettingsIcon,
  Warning as WarningIcon
} from '@mui/icons-material';
import { 
  notebookAPI, 
  NotebookWithDocuments, 
  Document, 
  NotebookDocument,
  getErrorMessage, 
  formatFileSize,
  formatRelativeTime
} from './NotebookAPI';
import DocumentAdmin from './DocumentAdmin';

interface NotebookDocumentListProps {
  notebook: NotebookWithDocuments;
  onDocumentChange: () => void;
}

const NotebookDocumentList: React.FC<NotebookDocumentListProps> = ({ 
  notebook, 
  onDocumentChange 
}) => {
  const [availableDocuments, setAvailableDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [searchTerm, setSearchTerm] = useState('');
  
  // Add documents dialog
  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [selectedDocuments, setSelectedDocuments] = useState<Set<string>>(new Set());
  const [addingDocuments, setAddingDocuments] = useState(false);
  
  // Upload dialog
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [uploadFiles, setUploadFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<{ [key: string]: {
    stage: 'uploading' | 'processing' | 'embedding' | 'storing' | 'complete' | 'error';
    progress: number;
    bytesUploaded?: number;
    totalBytes?: number;
    error?: string;
  } }>({});

  // Context menu
  const [contextMenu, setContextMenu] = useState<{
    mouseX: number;
    mouseY: number;
    document: NotebookDocument;
  } | null>(null);

  // Admin mode
  const [adminMode, setAdminMode] = useState(false);
  const [adminConfirmDialogOpen, setAdminConfirmDialogOpen] = useState(false);
  const [preselectedDocumentId, setPreselectedDocumentId] = useState<string | null>(null);

  // Load available documents when add dialog opens
  useEffect(() => {
    if (addDialogOpen) {
      loadAvailableDocuments();
    }
  }, [addDialogOpen]);

  const loadAvailableDocuments = async () => {
    try {
      setLoading(true);
      setError('');
      const docs = await notebookAPI.getAvailableDocuments();
      // Filter out documents that are already in the notebook
      const notebookDocIds = new Set(notebook.documents.map(nd => nd.document_id));
      const available = docs.filter(doc => !notebookDocIds.has(doc.document_id));
      setAvailableDocuments(available);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const handleAddDocuments = async () => {
    if (selectedDocuments.size === 0) return;

    try {
      setAddingDocuments(true);
      setError('');
      
      const promises = Array.from(selectedDocuments).map(docId =>
        notebookAPI.addDocumentToNotebook(notebook.id, docId)
      );
      
      await Promise.all(promises);
      
      setSelectedDocuments(new Set());
      setAddDialogOpen(false);
      onDocumentChange();
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setAddingDocuments(false);
    }
  };

  const handleRemoveDocument = async (documentId: string) => {
    try {
      setError('');
      await notebookAPI.removeDocumentFromNotebook(notebook.id, documentId);
      onDocumentChange();
      setContextMenu(null);
    } catch (err) {
      setError(getErrorMessage(err));
    }
  };

  const handleAdminModeToggle = () => {
    if (!adminMode) {
      setAdminConfirmDialogOpen(true);
    } else {
      setAdminMode(false);
      setPreselectedDocumentId(null);
    }
  };

  const handleConfirmAdminMode = () => {
    setAdminMode(true);
    setAdminConfirmDialogOpen(false);
  };

  const handleDeletePermanently = (documentId: string) => {
    setPreselectedDocumentId(documentId);
    setAdminMode(true);
    setContextMenu(null);
  };

  const handleAdminClose = () => {
    setAdminMode(false);
    setPreselectedDocumentId(null);
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    setUploadFiles(files);
    setUploadDialogOpen(true);
  };

  const handleUploadFiles = async () => {
    if (uploadFiles.length === 0) return;

    try {
      setUploading(true);
      setError('');
      setUploadProgress({});

      // Process files sequentially to avoid overwhelming the backend
      for (const file of uploadFiles) {
        try {
          await uploadFileWithProgress(file);
        } catch (err) {
          setUploadProgress(prev => ({ 
            ...prev, 
            [file.name]: { 
              stage: 'error', 
              progress: 0, 
              error: getErrorMessage(err) 
            } 
          }));
          // Continue with other files instead of stopping
        }
      }
      
      // Note: Completion handling is done within simulateProcessingStages
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setUploading(false);
    }
  };

  const uploadFileWithProgress = async (file: File): Promise<void> => {
    const fileName = file.name;
    
    // Stage 1: File Upload
    setUploadProgress(prev => ({ 
      ...prev, 
      [fileName]: { 
        stage: 'uploading', 
        progress: 0, 
        bytesUploaded: 0, 
        totalBytes: file.size 
      } 
    }));

    const formData = new FormData();
    formData.append('file', file);

    try {
      const xhr = new XMLHttpRequest();
      
      return new Promise<void>((resolve, reject) => {
        // Track upload progress
        xhr.upload.onprogress = (event) => {
          if (event.lengthComputable) {
            const uploadPercent = Math.round((event.loaded / event.total) * 100);
            setUploadProgress(prev => ({ 
              ...prev, 
              [fileName]: { 
                ...prev[fileName], 
                progress: Math.min(uploadPercent, 95), // Cap at 95% until processing starts
                bytesUploaded: event.loaded
              } 
            }));
          }
        };

        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            // Stage 2: Start processing stages
            simulateProcessingStages(fileName);
            resolve();
          } else {
            reject(new Error(`Upload failed: ${xhr.statusText}`));
          }
        };

        xhr.onerror = () => reject(new Error('Upload failed'));
        
        xhr.open('POST', `/api/v1/notebooks/${notebook.id}/upload`);
        xhr.send(formData);
      });
    } catch (error) {
      throw error;
    }
  };

  const simulateProcessingStages = async (fileName: string) => {
    const stages = [
      { stage: 'processing', duration: 2000, label: 'Processing PDF & Chunking' },
      { stage: 'embedding', duration: 30000, label: 'Generating Embeddings' }, // Longest stage
      { stage: 'storing', duration: 3000, label: 'Storing in Vector Database' },
    ] as const;

    for (const { stage, duration } of stages) {
      setUploadProgress(prev => ({ 
        ...prev, 
        [fileName]: { 
          ...prev[fileName], 
          stage, 
          progress: 0
        } 
      }));

      // Simulate progress for this stage
      const startTime = Date.now();
      const updateInterval = setInterval(() => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min((elapsed / duration) * 100, 100);
        
        setUploadProgress(prev => ({ 
          ...prev, 
          [fileName]: { 
            ...prev[fileName], 
            progress: Math.round(progress)
          } 
        }));

        if (progress >= 100) {
          clearInterval(updateInterval);
        }
      }, 100);

      // Wait for stage to complete
      await new Promise(resolve => setTimeout(resolve, duration));
      clearInterval(updateInterval);
    }

    // Final completion
    setUploadProgress(prev => ({ 
      ...prev, 
      [fileName]: { 
        ...prev[fileName], 
        stage: 'complete', 
        progress: 100
      } 
    }));

    // Check if all files are complete and trigger cleanup
    setTimeout(() => {
      setUploadProgress(currentProgress => {
        const completedCount = Object.values(currentProgress).filter(p => p.stage === 'complete').length;
        const totalFiles = uploadFiles.length;
        
        if (completedCount === totalFiles) {
          // All files completed, clean up after a brief display
          setTimeout(() => {
            setUploadFiles([]);
            setUploadDialogOpen(false);
            setUploadProgress({});
            onDocumentChange();
          }, 2000);
        }
        
        return currentProgress;
      });
    }, 100);
  };

  const getFileIcon = (contentType: string) => {
    if (!contentType) return <FileIcon />;
    if (contentType.includes('pdf')) return <PdfIcon color="error" />;
    if (contentType.includes('image')) return <ImageIcon color="primary" />;
    if (contentType.includes('text') || contentType.includes('document')) return <TextIcon color="info" />;
    return <FileIcon />;
  };

  const handleContextMenu = (event: React.MouseEvent, document: NotebookDocument) => {
    event.preventDefault();
    setContextMenu({
      mouseX: event.clientX + 2,
      mouseY: event.clientY - 6,
      document
    });
  };

  const filteredDocuments = (notebook.documents || []).filter(doc =>
    doc?.document_name?.toLowerCase().includes(searchTerm.toLowerCase()) || false
  );

  const filteredAvailableDocuments = (availableDocuments || []).filter(doc =>
    doc?.filename?.toLowerCase().includes(searchTerm.toLowerCase()) || false
  );

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Controls */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
        <TextField
          placeholder="Search documents..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon />
              </InputAdornment>
            ),
          }}
          sx={{ flex: 1, maxWidth: 400 }}
        />
        
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
          {/* Admin Mode Toggle */}
          <ToggleButton
            value="admin"
            selected={adminMode}
            onChange={handleAdminModeToggle}
            color={adminMode ? 'error' : 'standard'}
            sx={{ 
              minWidth: '120px',
              border: adminMode ? '1px solid' : '1px solid',
              borderColor: adminMode ? 'error.main' : 'divider'
            }}
          >
            {adminMode ? <AdminPanelSettingsIcon /> : <SecurityIcon />}
            <Typography variant="caption" sx={{ ml: 0.5 }}>
              {adminMode ? 'Admin ON' : 'Admin Mode'}
            </Typography>
          </ToggleButton>
          
          {!adminMode && (
            <>
              <input
                accept="*/*"
                style={{ display: 'none' }}
                id="upload-button-file"
                multiple
                type="file"
                onChange={handleFileUpload}
              />
              <label htmlFor="upload-button-file">
                <Button
                  variant="outlined"
                  component="span"
                  startIcon={<UploadIcon />}
                >
                  Upload
                </Button>
              </label>
              
              <Button
                variant="contained"
                startIcon={<AddIcon />}
                onClick={() => setAddDialogOpen(true)}
              >
                Add Documents
              </Button>
            </>
          )}
        </Box>
      </Box>

      {/* Admin Mode Warning */}
      {adminMode && (
        <Alert 
          severity="warning" 
          sx={{ mb: 2 }}
          icon={<WarningIcon />}
        >
          <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
            Admin Mode Active - Permanent Deletion Enabled
          </Typography>
          <Typography variant="body2">
            Documents can be permanently deleted from all systems. Use with caution!
          </Typography>
        </Alert>
      )}

      {/* Error Display */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>
          {error}
        </Alert>
      )}

      {/* Documents List or Admin Panel */}
      {adminMode ? (
        <DocumentAdmin 
          documents={notebook.documents}
          onDocumentChange={onDocumentChange}
          onClose={handleAdminClose}
        />
      ) : (
        <Paper sx={{ flex: 1, overflow: 'auto' }}>
        {notebook.documents.length === 0 ? (
          <Box sx={{ p: 6, textAlign: 'center' }}>
            <DocumentIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" color="text.secondary" gutterBottom>
              No documents in this notebook
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              Add existing documents or upload new ones to get started.
            </Typography>
            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
              <Button
                variant="outlined"
                startIcon={<AddIcon />}
                onClick={() => setAddDialogOpen(true)}
              >
                Add Documents
              </Button>
              <input
                accept="*/*"
                style={{ display: 'none' }}
                id="empty-upload-button"
                multiple
                type="file"
                onChange={handleFileUpload}
              />
              <label htmlFor="empty-upload-button">
                <Button
                  variant="contained"
                  component="span"
                  startIcon={<UploadIcon />}
                >
                  Upload Files
                </Button>
              </label>
            </Box>
          </Box>
        ) : (
          <List>
            {filteredDocuments.map((notebookDoc) => (
              <React.Fragment key={notebookDoc.document_id}>
                <ListItem
                  sx={{ 
                    '&:hover': { backgroundColor: 'action.hover' },
                    cursor: 'pointer'
                  }}
                  onContextMenu={(e) => handleContextMenu(e, notebookDoc)}
                >
                  <ListItemIcon>
                    {getFileIcon(notebookDoc.document_type)}
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Typography variant="subtitle1" noWrap>
                        {notebookDoc.document_name}
                      </Typography>
                    }
                    secondary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
                        <Chip
                          icon={<SizeIcon />}
                          label={formatFileSize(notebookDoc.metadata?.file_size_bytes || 0)}
                          size="small"
                          variant="outlined"
                        />
                        <Chip
                          icon={<TimeIcon />}
                          label={formatRelativeTime(notebookDoc.added_at)}
                          size="small"
                          variant="outlined"
                        />
                      </Box>
                    }
                  />
                  <ListItemSecondaryAction>
                    <IconButton 
                      edge="end"
                      onClick={(e) => handleContextMenu(e, notebookDoc)}
                    >
                      <MoreVertIcon />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
                <Divider variant="inset" component="li" />
              </React.Fragment>
            ))}
            
            {filteredDocuments.length === 0 && searchTerm && (
              <Box sx={{ p: 3, textAlign: 'center' }}>
                <Typography color="text.secondary">
                  No documents match your search
                </Typography>
              </Box>
            )}
          </List>
        )}
        </Paper>
      )}

      {/* Add Documents Dialog */}
      <Dialog
        open={addDialogOpen}
        onClose={() => !addingDocuments && setAddDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            Add Documents to Notebook
            <IconButton 
              onClick={() => setAddDialogOpen(false)}
              disabled={addingDocuments}
            >
              <CloseIcon />
            </IconButton>
          </Box>
        </DialogTitle>
        <DialogContent>
          {loading ? (
            <Box>
              {[...Array(5)].map((_, index) => (
                <Skeleton key={index} height={60} sx={{ mb: 1 }} />
              ))}
            </Box>
          ) : (
            <>
              {availableDocuments.length === 0 ? (
                <Alert severity="info">
                  No additional documents available. All documents are already in this notebook or you haven't uploaded any documents yet.
                </Alert>
              ) : (
                <List sx={{ maxHeight: 400, overflow: 'auto' }}>
                  {filteredAvailableDocuments.map((document) => (
                    <ListItem key={document.document_id} sx={{ pl: 0 }}>
                      <FormControlLabel
                        control={
                          <Checkbox
                            checked={selectedDocuments.has(document.document_id)}
                            onChange={(e) => {
                              const newSelected = new Set(selectedDocuments);
                              if (e.target.checked) {
                                newSelected.add(document.document_id);
                              } else {
                                newSelected.delete(document.document_id);
                              }
                              setSelectedDocuments(newSelected);
                            }}
                            disabled={addingDocuments}
                          />
                        }
                        label={
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            {getFileIcon(document.file_type)}
                            <Box>
                              <Typography variant="body1">
                                {document.filename}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                {formatFileSize(document.file_size_bytes)} • {formatRelativeTime(document.created_at)}
                              </Typography>
                            </Box>
                          </Box>
                        }
                        sx={{ margin: 0, width: '100%' }}
                      />
                    </ListItem>
                  ))}
                </List>
              )}
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button 
            onClick={() => setAddDialogOpen(false)}
            disabled={addingDocuments}
          >
            Cancel
          </Button>
          <Button
            onClick={handleAddDocuments}
            variant="contained"
            disabled={selectedDocuments.size === 0 || addingDocuments}
            startIcon={addingDocuments ? <CircularProgress size={16} /> : <AddIcon />}
          >
            Add {selectedDocuments.size > 0 ? `${selectedDocuments.size} ` : ''}Document{selectedDocuments.size !== 1 ? 's' : ''}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Enhanced Upload Progress Dialog */}
      <Dialog
        open={uploadDialogOpen}
        onClose={() => !uploading && setUploadDialogOpen(false)}
        maxWidth="md"
        fullWidth
        disableEscapeKeyDown={uploading}
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            Upload Documents
            <IconButton 
              onClick={() => setUploadDialogOpen(false)}
              disabled={uploading}
            >
              <CloseIcon />
            </IconButton>
          </Box>
        </DialogTitle>
        <DialogContent>
          {uploadFiles.length === 0 ? (
            <Alert severity="info">No files selected</Alert>
          ) : (
            <Box>
              {uploadFiles.map((file) => {
                const fileProgress = uploadProgress[file.name];
                const getStageInfo = () => {
                  if (!fileProgress) return { icon: <CloudUploadIcon />, label: 'Ready to upload', color: 'text.secondary' };
                  
                  switch (fileProgress.stage) {
                    case 'uploading':
                      return { 
                        icon: <CloudUploadIcon color="primary" />, 
                        label: `Uploading... ${fileProgress.bytesUploaded ? `(${formatFileSize(fileProgress.bytesUploaded)}/${formatFileSize(fileProgress.totalBytes || 0)})` : ''}`,
                        color: 'primary.main'
                      };
                    case 'processing':
                      return { 
                        icon: <ProcessingIcon color="info" />, 
                        label: 'Processing PDF & Chunking...',
                        color: 'info.main'
                      };
                    case 'embedding':
                      return { 
                        icon: <EmbeddingIcon color="warning" />, 
                        label: 'Generating Embeddings (this may take a few minutes)...',
                        color: 'warning.main'
                      };
                    case 'storing':
                      return { 
                        icon: <StoringIcon color="info" />, 
                        label: 'Storing in Vector Database...',
                        color: 'info.main'
                      };
                    case 'complete':
                      return { 
                        icon: <CheckCircleIcon color="success" />, 
                        label: 'Upload Complete!',
                        color: 'success.main'
                      };
                    case 'error':
                      return { 
                        icon: <ErrorIcon color="error" />, 
                        label: `Error: ${fileProgress.error}`,
                        color: 'error.main'
                      };
                    default:
                      return { icon: <CloudUploadIcon />, label: 'Ready', color: 'text.secondary' };
                  }
                };

                const stageInfo = getStageInfo();
                
                return (
                  <Card key={file.name} sx={{ mb: 2, opacity: fileProgress?.stage === 'complete' ? 0.8 : 1 }}>
                    <CardContent sx={{ pb: '16px !important' }}>
                      {/* File Info Header */}
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                        {getFileIcon(file.type)}
                        <Box sx={{ ml: 1, flex: 1, minWidth: 0 }}>
                          <Typography variant="body1" noWrap>
                            {file.name}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {formatFileSize(file.size)}
                          </Typography>
                        </Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          {stageInfo.icon}
                        </Box>
                      </Box>

                      {/* Progress Section */}
                      {fileProgress && (
                        <Box>
                          {/* Stage Label */}
                          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                            <Typography 
                              variant="body2" 
                              sx={{ color: stageInfo.color, fontWeight: 500 }}
                            >
                              {stageInfo.label}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {fileProgress.progress}%
                            </Typography>
                          </Box>
                          
                          {/* Progress Bar */}
                          <LinearProgress 
                            variant="determinate" 
                            value={fileProgress.progress}
                            color={
                              fileProgress.stage === 'error' ? 'error' :
                              fileProgress.stage === 'complete' ? 'success' :
                              fileProgress.stage === 'embedding' ? 'warning' : 'primary'
                            }
                            sx={{ 
                              height: 6, 
                              borderRadius: 3,
                              backgroundColor: 'rgba(0,0,0,0.1)'
                            }}
                          />
                          
                          {/* Time Estimate */}
                          {fileProgress.stage === 'embedding' && fileProgress.progress < 100 && (
                            <Typography 
                              variant="caption" 
                              color="text.secondary" 
                              sx={{ mt: 1, display: 'block', fontStyle: 'italic' }}
                            >
                              Embedding generation typically takes 2-3 minutes for document processing
                            </Typography>
                          )}
                          
                          {fileProgress.stage === 'complete' && (
                            <Typography 
                              variant="caption" 
                              color="success.main" 
                              sx={{ mt: 1, display: 'block', fontWeight: 500 }}
                            >
                              Document successfully processed and added to notebook
                            </Typography>
                          )}
                        </Box>
                      )}
                    </CardContent>
                  </Card>
                );
              })}
              
              {/* Overall Progress Summary */}
              {uploading && (
                <Alert 
                  severity="info" 
                  sx={{ mt: 2 }}
                  icon={<CircularProgress size={20} />}
                >
                  <Typography variant="body2">
                    Processing {uploadFiles.length} file{uploadFiles.length !== 1 ? 's' : ''}... 
                    Please keep this dialog open until all files are complete.
                  </Typography>
                </Alert>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button 
            onClick={() => setUploadDialogOpen(false)}
            disabled={uploading}
          >
            Cancel
          </Button>
          <Button
            onClick={handleUploadFiles}
            variant="contained"
            disabled={uploadFiles.length === 0 || uploading}
            startIcon={uploading ? <CircularProgress size={16} /> : <CloudUploadIcon />}
          >
            {uploading ? 'Processing...' : `Upload ${uploadFiles.length > 0 ? `${uploadFiles.length} ` : ''}File${uploadFiles.length !== 1 ? 's' : ''}`}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Context Menu */}
      <Menu
        open={contextMenu !== null}
        onClose={() => setContextMenu(null)}
        anchorReference="anchorPosition"
        anchorPosition={
          contextMenu !== null
            ? { top: contextMenu.mouseY, left: contextMenu.mouseX }
            : undefined
        }
      >
        <MenuItem 
          onClick={() => {
            if (contextMenu) {
              handleRemoveDocument(contextMenu.document.document_id);
            }
          }}
          sx={{ color: 'error.main' }}
        >
          <DeleteIcon sx={{ mr: 1 }} />
          Remove from Notebook
        </MenuItem>
        {adminMode && (
          <MenuItem 
            onClick={() => {
              if (contextMenu) {
                handleDeletePermanently(contextMenu.document.document_id);
              }
            }}
            sx={{ color: 'error.dark', borderTop: '1px solid', borderColor: 'divider' }}
          >
            <WarningIcon sx={{ mr: 1 }} />
            Delete Permanently
          </MenuItem>
        )}
      </Menu>

      {/* Admin Mode Confirmation Dialog */}
      <Dialog
        open={adminConfirmDialogOpen}
        onClose={() => setAdminConfirmDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1, color: 'warning.main' }}>
          <WarningIcon />
          Enable Admin Mode?
        </DialogTitle>
        <DialogContent>
          <DialogContentText>
            Admin mode allows permanent deletion of documents from all systems including:
          </DialogContentText>
          <Box sx={{ mt: 2, pl: 2 }}>
            <Typography variant="body2" component="li">Vector embeddings in Milvus database</Typography>
            <Typography variant="body2" component="li">Document records in PostgreSQL</Typography>
            <Typography variant="body2" component="li">Knowledge graph relationships in Neo4j</Typography>
            <Typography variant="body2" component="li">All notebook associations</Typography>
          </Box>
          <Alert severity="error" sx={{ mt: 2 }}>
            <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
              This action cannot be undone!
            </Typography>
          </Alert>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAdminConfirmDialogOpen(false)}>
            Cancel
          </Button>
          <Button
            onClick={handleConfirmAdminMode}
            variant="contained"
            color="error"
            startIcon={<AdminPanelSettingsIcon />}
          >
            Enable Admin Mode
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default NotebookDocumentList;