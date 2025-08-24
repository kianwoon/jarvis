import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  Checkbox,
  Alert,
  CircularProgress,
  LinearProgress,
  Divider,
  Chip,
  IconButton,
  Collapse,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
  FormControlLabel
} from '@mui/material';
import {
  Delete as DeleteIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Description as DocumentIcon,
  Storage as DatabaseIcon,
  Psychology as EmbeddingIcon,
  AccountTree as GraphIcon,
  Speed as CacheIcon,
  Book as NotebookIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Close as CloseIcon
} from '@mui/icons-material';
import { notebookAPI, Document, getErrorMessage, formatFileSize, formatRelativeTime } from './NotebookAPI';

interface DocumentUsageInfo {
  document_id: string;
  filename: string;
  file_type?: string;
  file_size_bytes?: number;
  milvus_collection?: string;
  notebook_count: number;
  notebooks_using: Array<{
    id: string;
    name: string;
    added_at?: string;
  }>;
  cross_references: number;
  deletion_impact: {
    will_remove_from_notebooks: number;
    will_delete_vectors: boolean;
    will_delete_cross_references: boolean;
  };
}

interface DocumentDeletionSummary {
  document_id: string;
  started_at: string;
  completed_at?: string;
  success: boolean;
  milvus_deleted: boolean;
  database_deleted: boolean;
  notebooks_removed: number;
  neo4j_deleted: boolean;
  cache_cleared: boolean;
  errors: string[];
}

interface DocumentDeleteResponse {
  success: boolean;
  message: string;
  total_requested?: number;
  successful_deletions?: number;
  failed_deletions?: number;
  deletion_details: DocumentDeletionSummary[];
  overall_errors: string[];
}

interface DocumentAdminProps {
  documents: Document[];
  onDocumentChange: () => void;
  onClose: () => void;
}

const DocumentAdmin: React.FC<DocumentAdminProps> = ({ documents, onDocumentChange, onClose }) => {
  const [selectedDocuments, setSelectedDocuments] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [documentUsageInfo, setDocumentUsageInfo] = useState<DocumentUsageInfo[]>([]);
  const [deletionResults, setDeletionResults] = useState<DocumentDeleteResponse | null>(null);
  const [showResults, setShowResults] = useState(false);
  const [expandedDetails, setExpandedDetails] = useState<Set<string>>(new Set());

  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      setSelectedDocuments(new Set(documents.map(doc => doc.document_id)));
    } else {
      setSelectedDocuments(new Set());
    }
  };

  const handleSelectDocument = (documentId: string, checked: boolean) => {
    const newSelected = new Set(selectedDocuments);
    if (checked) {
      newSelected.add(documentId);
    } else {
      newSelected.delete(documentId);
    }
    setSelectedDocuments(newSelected);
  };

  const handleDeleteClick = async () => {
    if (selectedDocuments.size === 0) return;

    setLoading(true);
    setError('');
    setDocumentUsageInfo([]);

    try {
      // Get usage info for all selected documents
      const usageInfoPromises = Array.from(selectedDocuments).map(async (docId) => {
        try {
          return await notebookAPI.getDocumentUsageInfo(docId);
        } catch (err) {
          return {
            document_id: docId,
            filename: 'Unknown',
            notebook_count: 0,
            notebooks_using: [],
            cross_references: 0,
            deletion_impact: {
              will_remove_from_notebooks: 0,
              will_delete_vectors: false,
              will_delete_cross_references: false
            }
          };
        }
      });

      const usageData = await Promise.all(usageInfoPromises);
      setDocumentUsageInfo(usageData);
      setDeleteDialogOpen(true);

    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const handleConfirmDelete = async () => {
    setLoading(true);
    setError('');

    try {
      const result = await notebookAPI.deleteDocumentsPermanently({
        document_ids: Array.from(selectedDocuments),
        remove_from_notebooks: true,
        confirm_permanent_deletion: true
      });

      setDeletionResults(result);
      setShowResults(true);
      setDeleteDialogOpen(false);
      
      // If deletion was successful, refresh the document list
      if (result.success || (result.successful_deletions && result.successful_deletions > 0)) {
        onDocumentChange();
        setSelectedDocuments(new Set());
      }

    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const toggleDetailExpansion = (documentId: string) => {
    const newExpanded = new Set(expandedDetails);
    if (newExpanded.has(documentId)) {
      newExpanded.delete(documentId);
    } else {
      newExpanded.add(documentId);
    }
    setExpandedDetails(newExpanded);
  };

  const getFileIcon = (fileType?: string) => {
    if (!fileType) return <DocumentIcon />;
    if (fileType.includes('pdf')) return <DocumentIcon color="error" />;
    return <DocumentIcon color="primary" />;
  };

  const getDeletionIcon = (step: keyof DocumentDeletionSummary, summary: DocumentDeletionSummary) => {
    const stepValue = summary[step];
    if (typeof stepValue === 'boolean') {
      return stepValue ? <CheckCircleIcon color="success" /> : <ErrorIcon color="error" />;
    }
    return <InfoIcon color="info" />;
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2, p: 2, bgcolor: 'warning.light', borderRadius: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <WarningIcon color="warning" />
          <Typography variant="h6" color="warning.dark">
            Document Admin - Permanent Deletion
          </Typography>
        </Box>
        <IconButton onClick={onClose}>
          <CloseIcon />
        </IconButton>
      </Box>

      <Alert severity="error" sx={{ mb: 2 }}>
        <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
          WARNING: This will permanently delete documents from ALL systems
        </Typography>
        <Typography variant="body2">
          • Removes vectors from Milvus database<br/>
          • Deletes from PostgreSQL database<br/>
          • Removes from all notebooks<br/>
          • Clears knowledge graph data<br/>
          • Cannot be undone!
        </Typography>
      </Alert>

      {/* Controls */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
        <FormControlLabel
          control={
            <Checkbox
              checked={selectedDocuments.size === documents.length && documents.length > 0}
              indeterminate={selectedDocuments.size > 0 && selectedDocuments.size < documents.length}
              onChange={(e) => handleSelectAll(e.target.checked)}
            />
          }
          label="Select All"
        />
        <Typography variant="body2" color="text.secondary">
          {selectedDocuments.size} of {documents.length} selected
        </Typography>
        <Button
          variant="contained"
          color="error"
          startIcon={<DeleteIcon />}
          onClick={handleDeleteClick}
          disabled={selectedDocuments.size === 0 || loading}
        >
          Delete {selectedDocuments.size > 0 ? `${selectedDocuments.size} ` : ''}Document{selectedDocuments.size !== 1 ? 's' : ''} Permanently
        </Button>
      </Box>

      {/* Error Display */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>
          {error}
        </Alert>
      )}

      {/* Document List */}
      <Paper sx={{ flex: 1, overflow: 'auto' }}>
        <List>
          {documents.map((doc) => (
            <React.Fragment key={doc.document_id}>
              <ListItem>
                <ListItemIcon>
                  <Checkbox
                    checked={selectedDocuments.has(doc.document_id)}
                    onChange={(e) => handleSelectDocument(doc.document_id, e.target.checked)}
                  />
                </ListItemIcon>
                <ListItemIcon>
                  {getFileIcon(doc.file_type)}
                </ListItemIcon>
                <ListItemText
                  primary={
                    <Typography variant="subtitle1">
                      {doc.filename}
                    </Typography>
                  }
                  secondary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
                      <Chip
                        label={formatFileSize(doc.file_size_bytes || 0)}
                        size="small"
                        variant="outlined"
                      />
                      <Chip
                        label={formatRelativeTime(doc.created_at)}
                        size="small"
                        variant="outlined"
                      />
                      <Chip
                        label={`ID: ${doc.document_id.slice(0, 8)}...`}
                        size="small"
                        variant="outlined"
                      />
                    </Box>
                  }
                />
              </ListItem>
              <Divider variant="inset" component="li" />
            </React.Fragment>
          ))}
        </List>
      </Paper>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={() => !loading && setDeleteDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle sx={{ bgcolor: 'error.light', color: 'error.contrastText' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <WarningIcon />
            Confirm Permanent Deletion
          </Box>
        </DialogTitle>
        <DialogContent>
          <Alert severity="error" sx={{ mb: 3 }}>
            <Typography variant="body1" sx={{ fontWeight: 'bold', mb: 1 }}>
              This action cannot be undone!
            </Typography>
            <Typography variant="body2">
              You are about to permanently delete {selectedDocuments.size} document{selectedDocuments.size !== 1 ? 's' : ''} 
              from all systems. This will affect the data shown below.
            </Typography>
          </Alert>

          {documentUsageInfo.map((info) => (
            <Card key={info.document_id} sx={{ mb: 2 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  {info.filename}
                </Typography>
                
                <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 2, mb: 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <NotebookIcon color="primary" />
                    <Typography variant="body2">
                      {info.notebook_count} notebook{info.notebook_count !== 1 ? 's' : ''}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <EmbeddingIcon color="secondary" />
                    <Typography variant="body2">
                      {info.milvus_collection ? `Collection: ${info.milvus_collection}` : 'No vectors'}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <GraphIcon color="info" />
                    <Typography variant="body2">
                      {info.cross_references} cross-reference{info.cross_references !== 1 ? 's' : ''}
                    </Typography>
                  </Box>
                </Box>

                {info.notebooks_using.length > 0 && (
                  <Box>
                    <Typography variant="subtitle2" gutterBottom>
                      Will be removed from notebooks:
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {info.notebooks_using.slice(0, 3).map((notebook) => (
                        <Chip
                          key={notebook.id}
                          label={notebook.name}
                          size="small"
                          color="warning"
                        />
                      ))}
                      {info.notebooks_using.length > 3 && (
                        <Chip
                          label={`+${info.notebooks_using.length - 3} more`}
                          size="small"
                          variant="outlined"
                        />
                      )}
                    </Box>
                  </Box>
                )}
              </CardContent>
            </Card>
          ))}
        </DialogContent>
        <DialogActions>
          <Button 
            onClick={() => setDeleteDialogOpen(false)}
            disabled={loading}
          >
            Cancel
          </Button>
          <Button
            onClick={handleConfirmDelete}
            variant="contained"
            color="error"
            disabled={loading}
            startIcon={loading ? <CircularProgress size={16} /> : <DeleteIcon />}
          >
            {loading ? 'Deleting...' : 'Confirm Permanent Deletion'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Results Dialog */}
      <Dialog
        open={showResults}
        onClose={() => setShowResults(false)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {deletionResults?.success ? <CheckCircleIcon color="success" /> : <WarningIcon color="warning" />}
            Deletion Results
          </Box>
        </DialogTitle>
        <DialogContent>
          {deletionResults && (
            <Box>
              <Alert 
                severity={deletionResults.success ? 'success' : 'warning'} 
                sx={{ mb: 3 }}
              >
                <Typography variant="body1">
                  {deletionResults.message}
                </Typography>
                {!deletionResults.success && deletionResults.overall_errors.length > 0 && (
                  <Box sx={{ mt: 1 }}>
                    <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                      Overall Errors:
                    </Typography>
                    {deletionResults.overall_errors.map((error, index) => (
                      <Typography key={index} variant="body2" color="error">
                        • {error}
                      </Typography>
                    ))}
                  </Box>
                )}
              </Alert>

              {deletionResults.deletion_details.map((detail) => (
                <Card key={detail.document_id} sx={{ mb: 2 }}>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                      <Typography variant="h6">
                        Document {detail.document_id.slice(0, 8)}...
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        {detail.success ? <CheckCircleIcon color="success" /> : <ErrorIcon color="error" />}
                        <IconButton
                          size="small"
                          onClick={() => toggleDetailExpansion(detail.document_id)}
                        >
                          {expandedDetails.has(detail.document_id) ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                        </IconButton>
                      </Box>
                    </Box>

                    <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                      <Tooltip title={detail.milvus_deleted ? 'Deleted from Milvus' : 'Failed to delete from Milvus'}>
                        <Chip
                          icon={getDeletionIcon('milvus_deleted', detail)}
                          label="Vectors"
                          size="small"
                          color={detail.milvus_deleted ? 'success' : 'error'}
                        />
                      </Tooltip>
                      <Tooltip title={detail.database_deleted ? 'Deleted from database' : 'Failed to delete from database'}>
                        <Chip
                          icon={getDeletionIcon('database_deleted', detail)}
                          label="Database"
                          size="small"
                          color={detail.database_deleted ? 'success' : 'error'}
                        />
                      </Tooltip>
                      <Tooltip title={`Removed from ${detail.notebooks_removed} notebooks`}>
                        <Chip
                          icon={<NotebookIcon />}
                          label={`${detail.notebooks_removed} Notebooks`}
                          size="small"
                          color="info"
                        />
                      </Tooltip>
                      <Tooltip title={detail.neo4j_deleted ? 'Deleted from knowledge graph' : 'Failed to delete from knowledge graph'}>
                        <Chip
                          icon={getDeletionIcon('neo4j_deleted', detail)}
                          label="Graph"
                          size="small"
                          color={detail.neo4j_deleted ? 'success' : 'error'}
                        />
                      </Tooltip>
                      <Tooltip title={detail.cache_cleared ? 'Cache cleared' : 'Failed to clear cache'}>
                        <Chip
                          icon={getDeletionIcon('cache_cleared', detail)}
                          label="Cache"
                          size="small"
                          color={detail.cache_cleared ? 'success' : 'error'}
                        />
                      </Tooltip>
                    </Box>

                    <Collapse in={expandedDetails.has(detail.document_id)}>
                      {detail.errors.length > 0 && (
                        <Box sx={{ mt: 2 }}>
                          <Typography variant="subtitle2" color="error" gutterBottom>
                            Errors:
                          </Typography>
                          {detail.errors.map((error, index) => (
                            <Typography key={index} variant="body2" color="error">
                              • {error}
                            </Typography>
                          ))}
                        </Box>
                      )}
                      <Box sx={{ mt: 2 }}>
                        <Typography variant="caption" color="text.secondary">
                          Started: {new Date(detail.started_at).toLocaleString()}<br/>
                          {detail.completed_at && `Completed: ${new Date(detail.completed_at).toLocaleString()}`}
                        </Typography>
                      </Box>
                    </Collapse>
                  </CardContent>
                </Card>
              ))}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowResults(false)}>
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default DocumentAdmin;