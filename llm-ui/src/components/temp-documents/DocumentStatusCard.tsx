import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardActions,
  Box,
  Typography,
  IconButton,
  Switch,
  Chip,
  LinearProgress,
  Tooltip,
  Menu,
  MenuItem,
  Divider,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from '@mui/material';
import {
  Description as FileIcon,
  Delete as DeleteIcon,
  MoreVert as MoreIcon,
  Visibility as PreviewIcon,
  Info as InfoIcon,
  CheckCircle as ReadyIcon,
  Error as ErrorIcon,
  CloudUpload as UploadIcon,
  Memory as MemoryIcon,
  Storage as StorageIcon,
  AccountTree as GraphIcon
} from '@mui/icons-material';

interface DocumentMetadata {
  upload_timestamp?: string;
  file_size?: number;
  chunk_count?: number;
  in_memory_rag_enabled?: boolean;
  knowledge_graph_enabled?: boolean;
  knowledge_graph_stats?: {
    success?: boolean;
    total_entities?: number;
    total_relationships?: number;
    processing_time_ms?: number;
    errors?: string[];
  };
  [key: string]: any;
}

interface DocumentStatusCardProps {
  tempDocId: string;
  filename: string;
  status: 'uploading' | 'processing' | 'ready' | 'error';
  isIncluded: boolean;
  metadata: DocumentMetadata;
  uploadProgress?: number;
  error?: string;
  onToggleInclusion: (tempDocId: string, included: boolean) => void;
  onDelete: (tempDocId: string) => void;
  onPreview?: (tempDocId: string) => void;
  variant?: 'compact' | 'detailed';
  disabled?: boolean;
}

const DocumentStatusCard: React.FC<DocumentStatusCardProps> = ({
  tempDocId,
  filename,
  status,
  isIncluded,
  metadata,
  uploadProgress = 0,
  error,
  onToggleInclusion,
  onDelete,
  onPreview,
  variant = 'detailed',
  disabled = false
}) => {
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);
  const [showDetails, setShowDetails] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);

  const handleMenuClick = (event: React.MouseEvent<HTMLElement>) => {
    event.preventDefault();
    event.stopPropagation();
    console.log('Menu click - before:', menuAnchor);
    setMenuAnchor(event.currentTarget);
    console.log('Menu click - after setting');
    // Force a direct action as a test
    if (window.confirm('Delete this document?')) {
      onDelete(tempDocId);
    }
  };

  const handleMenuClose = () => {
    setMenuAnchor(null);
  };

  const handleToggleInclusion = () => {
    onToggleInclusion(tempDocId, !isIncluded);
  };

  const handleDelete = () => {
    setConfirmDelete(true);
    handleMenuClose();
  };

  const confirmDeleteAction = () => {
    onDelete(tempDocId);
    setConfirmDelete(false);
  };

  const handlePreview = () => {
    onPreview?.(tempDocId);
    handleMenuClose();
  };

  const getStatusIcon = () => {
    switch (status) {
      case 'uploading':
        return <UploadIcon color="primary" />;
      case 'processing':
        return <UploadIcon color="primary" />;
      case 'ready':
        return <ReadyIcon color="success" />;
      case 'error':
        return <ErrorIcon color="error" />;
      default:
        return <FileIcon />;
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'uploading':
      case 'processing':
        return 'primary';
      case 'ready':
        return 'success';
      case 'error':
        return 'error';
      default:
        return 'primary';
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'uploading':
        return 'Uploading';
      case 'processing':
        return 'Processing';
      case 'ready':
        return 'Ready';
      case 'error':
        return 'Error';
      default:
        return 'Unknown';
    }
  };

  const formatFileSize = (bytes?: number): string => {
    if (!bytes) return 'Unknown size';
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${Math.round(bytes / Math.pow(1024, i) * 100) / 100} ${sizes[i]}`;
  };

  const formatTimestamp = (timestamp?: string): string => {
    if (!timestamp) return 'Unknown';
    try {
      return new Date(timestamp).toLocaleString();
    } catch {
      return 'Invalid date';
    }
  };

  const isActionable = status === 'ready' && !disabled;

  if (variant === 'compact') {
    return (
      <Box sx={{ 
        display: 'flex', 
        alignItems: 'center', 
        gap: 2, 
        p: 1,
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: 1,
        bgcolor: isIncluded ? 'action.selected' : 'background.paper'
      }}>
        {/* Status Icon */}
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          {getStatusIcon()}
        </Box>

        {/* File Info */}
        <Box sx={{ flex: 1, minWidth: 0 }}>
          <Typography variant="body2" noWrap title={filename}>
            {filename}
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Chip 
              label={getStatusText()} 
              size="small" 
              color={getStatusColor()}
              variant="outlined"
            />
            {metadata.in_memory_rag_enabled && (
              <Chip 
                icon={<MemoryIcon />}
                label="In-Memory" 
                size="small" 
                color="info"
                variant="outlined"
              />
            )}
            {metadata.knowledge_graph_enabled && (
              <Chip 
                icon={<GraphIcon />}
                label={metadata.knowledge_graph_stats?.success 
                  ? `Graph: ${metadata.knowledge_graph_stats.total_entities || 0}E ${metadata.knowledge_graph_stats.total_relationships || 0}R`
                  : "Graph"
                } 
                size="small" 
                color={metadata.knowledge_graph_stats?.success ? "success" : "warning"}
                variant="outlined"
              />
            )}
          </Box>
        </Box>

        {/* Upload Progress */}
        {(status === 'uploading' || status === 'processing') && (
          <Box sx={{ width: 100 }}>
            <LinearProgress 
              variant="determinate" 
              value={uploadProgress} 
              color={getStatusColor()}
            />
            <Typography variant="caption" color="text.secondary">
              {uploadProgress}%
            </Typography>
          </Box>
        )}

        {/* Inclusion Toggle */}
        {isActionable && (
          <Switch
            checked={isIncluded}
            onChange={handleToggleInclusion}
            size="small"
          />
        )}

        {/* Actions */}
        <IconButton 
          size="small" 
          onClick={handleMenuClick}
          disabled={disabled}
        >
          <MoreIcon />
        </IconButton>
      </Box>
    );
  }

  return (
    <>
      <Card sx={{ 
        mb: 2, 
        border: isIncluded ? '2px solid' : '1px solid',
        borderColor: isIncluded ? 'primary.main' : 'divider',
        bgcolor: isIncluded ? 'action.selected' : 'background.paper'
      }}>
        <CardContent sx={{ pb: 1 }}>
          {/* Header */}
          <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2, mb: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
              {getStatusIcon()}
            </Box>
            
            <Box sx={{ flex: 1, minWidth: 0 }}>
              <Typography variant="h6" noWrap title={filename}>
                {filename}
              </Typography>
              
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
                <Chip 
                  label={getStatusText()} 
                  size="small" 
                  color={getStatusColor()}
                />
                
                {metadata.in_memory_rag_enabled && (
                  <Tooltip title="Using in-memory RAG for fast searching">
                    <Chip 
                      icon={<MemoryIcon />}
                      label="In-Memory RAG" 
                      size="small" 
                      color="info"
                      variant="outlined"
                    />
                  </Tooltip>
                )}
                
                {metadata.knowledge_graph_enabled && (
                  <Tooltip title={`Knowledge graph: ${metadata.knowledge_graph_stats?.total_entities || 0} entities, ${metadata.knowledge_graph_stats?.total_relationships || 0} relationships`}>
                    <Chip 
                      icon={<GraphIcon />}
                      label={metadata.knowledge_graph_stats?.success 
                        ? `Knowledge Graph (${metadata.knowledge_graph_stats.total_entities || 0}E, ${metadata.knowledge_graph_stats.total_relationships || 0}R)`
                        : "Knowledge Graph"
                      } 
                      size="small" 
                      color={metadata.knowledge_graph_stats?.success ? "success" : "warning"}
                      variant="outlined"
                    />
                  </Tooltip>
                )}
                
                {metadata.chunk_count && (
                  <Chip 
                    label={`${metadata.chunk_count} chunks`}
                    size="small" 
                    variant="outlined"
                  />
                )}
              </Box>
            </Box>

            <IconButton onClick={handleMenuClick} disabled={disabled}>
              <MoreIcon />
            </IconButton>
          </Box>

          {/* Upload Progress */}
          {(status === 'uploading' || status === 'processing') && (
            <Box sx={{ mb: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2" color="text.secondary">
                  {status === 'uploading' ? 'Uploading...' : 'Processing...'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {uploadProgress}%
                </Typography>
              </Box>
              <LinearProgress 
                variant="determinate" 
                value={uploadProgress} 
                color={getStatusColor()}
              />
            </Box>
          )}

          {/* Error Message */}
          {status === 'error' && error && (
            <Box sx={{ 
              p: 1, 
              bgcolor: 'error.main', 
              color: 'error.contrastText',
              borderRadius: 1,
              mb: 2
            }}>
              <Typography variant="body2">
                Error: {error}
              </Typography>
            </Box>
          )}

          {/* Metadata */}
          <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 1 }}>
            {metadata.file_size && (
              <Box>
                <Typography variant="caption" color="text.secondary">
                  Size
                </Typography>
                <Typography variant="body2">
                  {formatFileSize(metadata.file_size)}
                </Typography>
              </Box>
            )}
            
            {metadata.upload_timestamp && (
              <Box>
                <Typography variant="caption" color="text.secondary">
                  Uploaded
                </Typography>
                <Typography variant="body2">
                  {formatTimestamp(metadata.upload_timestamp)}
                </Typography>
              </Box>
            )}
          </Box>
        </CardContent>

        <CardActions sx={{ justifyContent: 'space-between', px: 2, py: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography variant="body2" color="text.secondary">
              Include in chat:
            </Typography>
            <Switch
              checked={isIncluded}
              onChange={handleToggleInclusion}
              disabled={!isActionable}
              size="small"
            />
          </Box>

          <Button
            size="small"
            startIcon={<InfoIcon />}
            onClick={() => setShowDetails(true)}
          >
            Details
          </Button>
        </CardActions>
      </Card>

      {/* Action Menu */}
      <Menu
        anchorEl={menuAnchor}
        open={Boolean(menuAnchor)}
        onClose={handleMenuClose}
        disablePortal={false}
        container={document.body}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'right',
        }}
        transformOrigin={{
          vertical: 'top',
          horizontal: 'right',
        }}
      >
        <MenuItem onClick={() => { setShowDetails(true); handleMenuClose(); }}>
          <InfoIcon sx={{ mr: 1 }} fontSize="small" />
          View Details
        </MenuItem>
        {onPreview && (
          <MenuItem onClick={handlePreview}>
            <PreviewIcon sx={{ mr: 1 }} fontSize="small" />
            Preview
          </MenuItem>
        )}
        <Divider />
        <MenuItem 
          onClick={handleDelete}
          sx={{ color: 'error.main' }}
        >
          <DeleteIcon sx={{ mr: 1 }} fontSize="small" />
          Delete
        </MenuItem>
      </Menu>

      {/* Details Dialog */}
      <Dialog open={showDetails} onClose={() => setShowDetails(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Document Details</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'grid', gap: 2 }}>
            <Box>
              <Typography variant="subtitle2" color="text.secondary">
                Filename
              </Typography>
              <Typography variant="body1">
                {filename}
              </Typography>
            </Box>
            
            <Box>
              <Typography variant="subtitle2" color="text.secondary">
                Document ID
              </Typography>
              <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                {tempDocId}
              </Typography>
            </Box>
            
            <Box>
              <Typography variant="subtitle2" color="text.secondary">
                Status
              </Typography>
              <Chip 
                label={getStatusText()} 
                color={getStatusColor()}
                size="small"
              />
            </Box>
            
            {Object.entries(metadata).map(([key, value]) => (
              <Box key={key}>
                <Typography variant="subtitle2" color="text.secondary">
                  {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </Typography>
                <Typography variant="body2">
                  {typeof value === 'boolean' ? (value ? 'Yes' : 'No') : String(value)}
                </Typography>
              </Box>
            ))}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowDetails(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation */}
      <Dialog open={confirmDelete} onClose={() => setConfirmDelete(false)}>
        <DialogTitle>Confirm Delete</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete "{filename}"? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmDelete(false)}>Cancel</Button>
          <Button onClick={confirmDeleteAction} color="error" variant="contained">
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default DocumentStatusCard;