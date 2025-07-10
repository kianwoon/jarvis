import React, { useState, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  Collapse,
  IconButton,
  Alert,
  Button,
  Divider,
  Tooltip,
  Chip,
  CircularProgress
} from '@mui/material';
import {
  ExpandMore as ExpandIcon,
  ExpandLess as CollapseIcon,
  CloudUpload as UploadIcon,
  Clear as ClearIcon,
  Refresh as RefreshIcon,
  Description as DocumentIcon
} from '@mui/icons-material';

import { useDocumentState } from '../../hooks/useDocumentState';
import { useRAGMode } from '../../hooks/useRAGMode';
import DocumentPriorityToggle from './DocumentPriorityToggle';
import DocumentStatusCard from './DocumentStatusCard';
import FileUploadComponent from '../shared/FileUploadComponent';

interface TempDocumentPanelProps {
  conversationId: string;
  disabled?: boolean;
  variant?: 'compact' | 'full';
  defaultExpanded?: boolean;
  onDocumentChange?: (activeCount: number, totalCount: number) => void;
}

const TempDocumentPanel: React.FC<TempDocumentPanelProps> = ({
  conversationId,
  disabled = false,
  variant = 'full',
  defaultExpanded = false,
  onDocumentChange
}) => {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);

  // Use custom hooks for state management
  const {
    documents,
    loading,
    error,
    uploadDocument,
    deleteDocument,
    toggleDocumentInclusion,
    refreshDocuments,
    clearDocuments,
    getActiveDocuments,
    getTotalDocuments
  } = useDocumentState(conversationId);

  const {
    ragMode,
    togglePriorityMode,
    setStrategy,
    setFallbackToPersistent,
    isInMemoryRAGActive,
    getRAGDescription,
    updateActiveDocumentCount
  } = useRAGMode(conversationId);

  // Update parent component when document count changes
  React.useEffect(() => {
    const activeCount = getActiveDocuments().length;
    const totalCount = getTotalDocuments();
    updateActiveDocumentCount(activeCount);
    onDocumentChange?.(activeCount, totalCount);
  }, [documents, getActiveDocuments, getTotalDocuments, updateActiveDocumentCount, onDocumentChange]);

  // Auto-expand when no documents are present to make feature discoverable
  // Disabled to prevent taking up too much space
  // React.useEffect(() => {
  //   if (variant === 'compact' && getTotalDocuments() === 0 && !expanded) {
  //     setExpanded(true);
  //   }
  // }, [variant, getTotalDocuments, expanded]);

  const handleFileUpload = useCallback(async (file: File) => {
    try {
      const result = await uploadDocument(file, {
        ttlHours: 2,
        autoInclude: true,
        enableInMemoryRag: true
      });

      if (result) {
        // Auto-expand panel when first document is uploaded
        if (documents.length === 0) {
          setExpanded(true);
        }
      }
    } catch (err) {
      console.error('Upload failed:', err);
    }
  }, [uploadDocument, documents.length]);

  const handleClearAll = useCallback(async () => {
    if (window.confirm('Are you sure you want to remove all documents? This action cannot be undone.')) {
      await clearDocuments();
    }
  }, [clearDocuments]);

  const activeDocuments = getActiveDocuments();
  const totalDocuments = getTotalDocuments();
  const hasDocuments = totalDocuments > 0;

  if (variant === 'compact') {
    return (
      <Box sx={{ mb: 2 }}>
        <Paper sx={{ p: 2, border: '1px solid', borderColor: 'divider' }}>
          {/* Compact Header */}
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <DocumentIcon color="primary" />
              <Typography variant="body1" sx={{ fontWeight: 'bold', color: 'primary.main' }}>
                Temporary Documents
              </Typography>
              {hasDocuments && (
                <Chip 
                  label={`${activeDocuments.length}/${totalDocuments}`}
                  size="small"
                  color={activeDocuments.length > 0 ? 'primary' : 'default'}
                  variant="outlined"
                />
              )}
              {!hasDocuments && (
                <Chip 
                  label="Upload docs for priority search"
                  size="small"
                  color="info"
                  variant="outlined"
                />
              )}
            </Box>
            
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <input
                type="file"
                accept=".pdf,.docx,.xlsx,.pptx,.txt"
                style={{ display: 'none' }}
                id="temp-upload-input"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) {
                    handleFileUpload(file);
                    e.target.value = ''; // Reset input
                  }
                }}
                disabled={disabled}
              />
              <label htmlFor="temp-upload-input">
                <IconButton component="span" disabled={disabled} size="small">
                  <UploadIcon />
                </IconButton>
              </label>
              
              <IconButton 
                onClick={() => setExpanded(!expanded)} 
                size="small"
                disabled={disabled}
              >
                {expanded ? <CollapseIcon /> : <ExpandIcon />}
              </IconButton>
            </Box>
          </Box>

          <DocumentPriorityToggle
            enabled={ragMode.priorityMode}
            onChange={togglePriorityMode}
            documentCount={activeDocuments.length}
            strategy={ragMode.strategy}
            onStrategyChange={setStrategy}
            fallbackToPersistent={ragMode.fallbackToPersistent}
            onFallbackChange={setFallbackToPersistent}
            disabled={disabled}
            variant="compact"
          />

          <Collapse in={expanded}>
            <Box sx={{ mt: 2 }}>
              {loading && (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                  <CircularProgress size={16} />
                  <Typography variant="body2" color="text.secondary">
                    Loading documents...
                  </Typography>
                </Box>
              )}

              {error && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  {error}
                </Alert>
              )}

              {!hasDocuments && !loading && (
                <Alert severity="info" sx={{ textAlign: 'center' }}>
                  <Typography variant="body2" sx={{ fontWeight: 'bold', mb: 1 }}>
                    No documents uploaded yet
                  </Typography>
                  <Typography variant="body2">
                    Upload documents above to enable fast in-memory search that prioritizes your content over the knowledge base.
                  </Typography>
                </Alert>
              )}

              {documents.map((doc) => (
                <DocumentStatusCard
                  key={doc.tempDocId}
                  tempDocId={doc.tempDocId}
                  filename={doc.filename}
                  status={doc.status}
                  isIncluded={doc.isIncluded}
                  metadata={doc.metadata}
                  uploadProgress={doc.uploadProgress}
                  error={doc.error}
                  onToggleInclusion={toggleDocumentInclusion}
                  onDelete={deleteDocument}
                  variant="compact"
                  disabled={disabled}
                />
              ))}
            </Box>
          </Collapse>
        </Paper>
      </Box>
    );
  }

  return (
    <Box sx={{ mb: 2 }}>
      <Paper sx={{ border: '1px solid', borderColor: 'divider' }}>
        {/* Header */}
        <Box sx={{ 
          p: 2, 
          borderBottom: '1px solid', 
          borderColor: 'divider',
          bgcolor: 'background.default'
        }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <DocumentIcon color="primary" />
              <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                Temporary Documents
              </Typography>
              {hasDocuments && (
                <Chip 
                  label={`${activeDocuments.length} active / ${totalDocuments} total`}
                  color={activeDocuments.length > 0 ? 'primary' : 'default'}
                  variant="outlined"
                />
              )}
            </Box>
            
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Tooltip title="Refresh documents">
                <IconButton 
                  onClick={refreshDocuments} 
                  disabled={disabled || loading}
                  size="small"
                >
                  <RefreshIcon />
                </IconButton>
              </Tooltip>
              
              {hasDocuments && (
                <Tooltip title="Clear all documents">
                  <IconButton 
                    onClick={handleClearAll} 
                    disabled={disabled || loading}
                    size="small"
                    color="error"
                  >
                    <ClearIcon />
                  </IconButton>
                </Tooltip>
              )}
              
              <IconButton 
                onClick={() => setExpanded(!expanded)} 
                disabled={disabled}
                size="small"
              >
                {expanded ? <CollapseIcon /> : <ExpandIcon />}
              </IconButton>
            </Box>
          </Box>

          {/* RAG Mode Toggle */}
          <DocumentPriorityToggle
            enabled={ragMode.priorityMode}
            onChange={togglePriorityMode}
            documentCount={activeDocuments.length}
            strategy={ragMode.strategy}
            onStrategyChange={setStrategy}
            fallbackToPersistent={ragMode.fallbackToPersistent}
            onFallbackChange={setFallbackToPersistent}
            disabled={disabled}
            description={getRAGDescription()}
            variant="detailed"
          />
        </Box>

        {/* Content */}
        <Collapse in={expanded}>
          <Box sx={{ p: 2 }}>
            {/* Upload Section */}
            <Box sx={{ mb: 3 }}>
              <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 'bold' }}>
                Upload Documents
              </Typography>
              
              <Box sx={{ 
                border: '2px dashed', 
                borderColor: 'divider',
                borderRadius: 2,
                p: 3,
                textAlign: 'center',
                bgcolor: 'action.hover'
              }}>
                <UploadIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 1 }} />
                <Typography variant="body1" sx={{ mb: 2 }}>
                  Drop files here or click to upload
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  Supported formats: PDF, DOCX, XLSX, PPTX, TXT (max 50MB)
                </Typography>
                
                <input
                  type="file"
                  accept=".pdf,.docx,.xlsx,.pptx,.txt"
                  style={{ display: 'none' }}
                  id="temp-upload-input-full"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) {
                      handleFileUpload(file);
                      e.target.value = ''; // Reset input
                    }
                  }}
                  disabled={disabled}
                />
                <label htmlFor="temp-upload-input-full">
                  <Button component="span" variant="contained" disabled={disabled}>
                    Choose Files
                  </Button>
                </label>
              </Box>
            </Box>

            <Divider sx={{ my: 2 }} />

            {/* Documents List */}
            <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 'bold' }}>
              Uploaded Documents
            </Typography>

            {loading && (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, p: 2 }}>
                <CircularProgress size={20} />
                <Typography variant="body2" color="text.secondary">
                  Loading documents...
                </Typography>
              </Box>
            )}

            {error && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {error}
                <Button 
                  size="small" 
                  onClick={refreshDocuments}
                  sx={{ ml: 1 }}
                >
                  Retry
                </Button>
              </Alert>
            )}

            {!hasDocuments && !loading && (
              <Alert severity="info" sx={{ textAlign: 'center' }}>
                <Typography variant="body1" sx={{ mb: 1 }}>
                  No documents uploaded yet
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Upload documents to enable fast in-memory search that prioritizes your content
                </Typography>
              </Alert>
            )}

            {/* Document Cards */}
            {documents.map((doc) => (
              <DocumentStatusCard
                key={doc.tempDocId}
                tempDocId={doc.tempDocId}
                filename={doc.filename}
                status={doc.status}
                isIncluded={doc.isIncluded}
                metadata={doc.metadata}
                uploadProgress={doc.uploadProgress}
                error={doc.error}
                onToggleInclusion={toggleDocumentInclusion}
                onDelete={deleteDocument}
                variant="detailed"
                disabled={disabled}
              />
            ))}

            {/* Summary */}
            {hasDocuments && (
              <Box sx={{ 
                mt: 3, 
                p: 2, 
                bgcolor: 'action.hover', 
                borderRadius: 1 
              }}>
                <Typography variant="body2" color="text.secondary">
                  <strong>Active Mode:</strong> {getRAGDescription()}
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  <strong>Performance:</strong> In-memory RAG provides instant search results from your uploaded documents
                </Typography>
              </Box>
            )}
          </Box>
        </Collapse>
      </Paper>
    </Box>
  );
};

export default TempDocumentPanel;