import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  TableSortLabel,
  TextField,
  InputAdornment,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Alert,
  CircularProgress,
  LinearProgress,
  Collapse,
  Card,
  CardContent,
  IconButton,
  Tooltip,
  Checkbox,
  FormControlLabel,
  Grid,
  Divider,
  Badge,
  Skeleton
} from '@mui/material';
import {
  Delete as DeleteIcon,
  Search as SearchIcon,
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
  Close as CloseIcon,
  Refresh as RefreshIcon,
  FilterList as FilterIcon,
  Clear as ClearIcon,
  GetApp as ExportIcon,
  Visibility as ViewIcon,
  Settings as AdminIcon,
  ExitToApp as ExitIcon
} from '@mui/icons-material';
import { 
  notebookAPI, 
  SystemDocument, 
  PaginationInfo, 
  SystemDocumentStats,
  DocumentUsageInfo,
  DocumentDeleteResponse,
  DocumentDeletionSummary,
  getErrorMessage, 
  formatFileSize, 
  formatRelativeTime 
} from '../notebook/NotebookAPI';


const DocumentAdminPage: React.FC = () => {
  // State management
  const [documents, setDocuments] = useState<SystemDocument[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>('');
  const [selectedDocuments, setSelectedDocuments] = useState<Set<string>>(new Set());
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [documentUsageInfo, setDocumentUsageInfo] = useState<DocumentUsageInfo[]>([]);
  const [deletionResults, setDeletionResults] = useState<DocumentDeleteResponse | null>(null);
  const [showResults, setShowResults] = useState(false);
  const [expandedDetails, setExpandedDetails] = useState<Set<string>>(new Set());

  // Filter and pagination state
  const [search, setSearch] = useState('');
  const [fileTypeFilter, setFileTypeFilter] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [collectionFilter, setCollectionFilter] = useState('');
  const [showOrphanedOnly, setShowOrphanedOnly] = useState(false);
  const [pagination, setPagination] = useState<PaginationInfo>({
    page: 1,
    page_size: 25,
    total_count: 0,
    total_pages: 0,
    has_next: false,
    has_prev: false
  });
  const [sortBy, setSortBy] = useState('created_at');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  
  // Statistics state
  const [stats, setStats] = useState<SystemDocumentStats>({
    total_documents: 0,
    unique_file_types: 0,
    unique_collections: 0,
    total_size_bytes: 0,
    completed_documents: 0,
    failed_documents: 0,
    orphaned_documents: 0
  });

  // Load documents with current filters
  const loadDocuments = useCallback(async () => {
    try {
      setLoading(true);
      setError('');
      
      const result = await notebookAPI.getAllSystemDocuments({
        page: pagination.page,
        page_size: pagination.page_size,
        search: search || undefined,
        file_type: fileTypeFilter || undefined,
        status: statusFilter || undefined,
        collection: collectionFilter || undefined,
        sort_by: sortBy,
        sort_order: sortOrder
      });
      
      let filteredDocuments = result.documents;
      
      // Apply orphaned filter on client side if needed
      if (showOrphanedOnly) {
        filteredDocuments = result.documents.filter(doc => doc.is_orphaned);
      }
      
      setDocuments(filteredDocuments);
      setPagination(result.pagination);
      setStats(result.summary_stats);
      
    } catch (err) {
      setError(getErrorMessage(err));
      console.error('Failed to load system documents:', err);
    } finally {
      setLoading(false);
    }
  }, [pagination.page, pagination.page_size, search, fileTypeFilter, statusFilter, collectionFilter, showOrphanedOnly, sortBy, sortOrder]);

  // Load documents on component mount and when filters change
  useEffect(() => {
    loadDocuments();
  }, [loadDocuments]);

  // Handle search with debouncing
  useEffect(() => {
    const timer = setTimeout(() => {
      if (pagination.page !== 1) {
        setPagination(prev => ({ ...prev, page: 1 }));
      } else {
        loadDocuments();
      }
    }, 500);
    
    return () => clearTimeout(timer);
  }, [search]);

  // Reset to first page when filters change
  useEffect(() => {
    if (pagination.page !== 1) {
      setPagination(prev => ({ ...prev, page: 1 }));
    }
  }, [fileTypeFilter, statusFilter, collectionFilter, showOrphanedOnly, sortBy, sortOrder]);

  // Document selection handlers
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

  // Sorting handler
  const handleSort = (field: string) => {
    if (sortBy === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setSortOrder('desc');
    }
  };

  // Pagination handler
  const handleChangePage = (event: unknown, newPage: number) => {
    setPagination(prev => ({ ...prev, page: newPage + 1 }));
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newPageSize = parseInt(event.target.value, 10);
    setPagination(prev => ({ 
      ...prev, 
      page: 1,
      page_size: newPageSize 
    }));
  };

  // Clear all filters
  const handleClearFilters = () => {
    setSearch('');
    setFileTypeFilter('');
    setStatusFilter('');
    setCollectionFilter('');
    setShowOrphanedOnly(false);
    setSortBy('created_at');
    setSortOrder('desc');
    setPagination(prev => ({ ...prev, page: 1 }));
  };

  // Delete dialog handlers
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
      setDocumentUsageInfo(usageData as DocumentUsageInfo[]);
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
        await loadDocuments();
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

  const handleExitAdminMode = () => {
    // Navigate back to notebook page
    window.location.href = '/notebook.html';
  };

  const getFileIcon = (fileType?: string) => {
    if (!fileType) return <DocumentIcon />;
    if (fileType.includes('pdf')) return <DocumentIcon color="error" />;
    return <DocumentIcon color="primary" />;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success';
      case 'processing': return 'warning';
      case 'failed': return 'error';
      default: return 'default';
    }
  };

  const getDeletionIcon = (step: keyof DocumentDeletionSummary, summary: DocumentDeletionSummary) => {
    const stepValue = summary[step];
    if (typeof stepValue === 'boolean') {
      return stepValue ? <CheckCircleIcon color="success" /> : <ErrorIcon color="error" />;
    }
    return <InfoIcon color="info" />;
  };

  // Get unique filter values
  const uniqueFileTypes = [...new Set(documents.map(doc => doc.file_type).filter(Boolean))];
  const uniqueStatuses = [...new Set(documents.map(doc => doc.processing_status).filter(Boolean))];
  const uniqueCollections = [...new Set(documents.map(doc => doc.milvus_collection).filter(Boolean))];

  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column', p: 3, bgcolor: 'background.default' }}>
      {/* Header */}
      <Paper elevation={2} sx={{ mb: 3, p: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <AdminIcon color="primary" fontSize="large" />
            <Box>
              <Typography variant="h4" component="h1" fontWeight="bold">
                System Document Administration
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Manage all documents across the entire system
              </Typography>
            </Box>
          </Box>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button
              variant="outlined"
              startIcon={<RefreshIcon />}
              onClick={loadDocuments}
              disabled={loading}
            >
              Refresh
            </Button>
            <Button
              variant="outlined"
              startIcon={<ExportIcon />}
              onClick={() => {/* TODO: Export functionality */}}
            >
              Export
            </Button>
            <Button
              variant="contained"
              color="primary"
              startIcon={<ExitIcon />}
              onClick={handleExitAdminMode}
            >
              Exit Admin Mode
            </Button>
          </Box>
        </Box>

        {/* Statistics Cards */}
        <Grid container spacing={2} sx={{ mb: 2 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card variant="outlined">
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <Typography variant="h5" color="primary" fontWeight="bold">
                  {loading ? <Skeleton width={60} /> : stats.total_documents.toLocaleString()}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Total Documents
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card variant="outlined">
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <Typography variant="h5" color="success.main" fontWeight="bold">
                  {loading ? <Skeleton width={60} /> : stats.completed_documents.toLocaleString()}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Completed
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card variant="outlined">
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <Typography variant="h5" color="warning.main" fontWeight="bold">
                  {loading ? <Skeleton width={60} /> : stats.orphaned_documents.toLocaleString()}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Orphaned
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card variant="outlined">
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <Typography variant="h5" color="info.main" fontWeight="bold">
                  {loading ? <Skeleton width={60} /> : formatFileSize(stats.total_size_bytes)}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Total Size
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Filters */}
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={3}>
            <TextField
              fullWidth
              size="small"
              placeholder="Search documents..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon />
                  </InputAdornment>
                )
              }}
            />
          </Grid>
          <Grid item xs={6} md={2}>
            <FormControl fullWidth size="small">
              <InputLabel>File Type</InputLabel>
              <Select
                value={fileTypeFilter}
                label="File Type"
                onChange={(e) => setFileTypeFilter(e.target.value)}
              >
                <MenuItem value="">All Types</MenuItem>
                {uniqueFileTypes.map(type => (
                  <MenuItem key={type} value={type}>{type.toUpperCase()}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={6} md={2}>
            <FormControl fullWidth size="small">
              <InputLabel>Status</InputLabel>
              <Select
                value={statusFilter}
                label="Status"
                onChange={(e) => setStatusFilter(e.target.value)}
              >
                <MenuItem value="">All Status</MenuItem>
                {uniqueStatuses.map(status => (
                  <MenuItem key={status} value={status}>{status}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={6} md={2}>
            <FormControl fullWidth size="small">
              <InputLabel>Collection</InputLabel>
              <Select
                value={collectionFilter}
                label="Collection"
                onChange={(e) => setCollectionFilter(e.target.value)}
              >
                <MenuItem value="">All Collections</MenuItem>
                {uniqueCollections.map(collection => (
                  <MenuItem key={collection} value={collection}>{collection}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={6} md={2}>
            <FormControlLabel
              control={
                <Checkbox
                  checked={showOrphanedOnly}
                  onChange={(e) => setShowOrphanedOnly(e.target.checked)}
                />
              }
              label="Orphaned Only"
            />
          </Grid>
          <Grid item xs={12} md={1}>
            <Button
              variant="outlined"
              startIcon={<ClearIcon />}
              onClick={handleClearFilters}
              fullWidth
            >
              Clear
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {/* Error Display */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>
          {error}
        </Alert>
      )}

      {/* Action Bar */}
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

      {/* Document Table */}
      <Paper sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        {loading && <LinearProgress />}
        
        <TableContainer sx={{ flex: 1 }}>
          <Table stickyHeader>
            <TableHead>
              <TableRow>
                <TableCell padding="checkbox">
                  <Checkbox
                    checked={selectedDocuments.size === documents.length && documents.length > 0}
                    indeterminate={selectedDocuments.size > 0 && selectedDocuments.size < documents.length}
                    onChange={(e) => handleSelectAll(e.target.checked)}
                  />
                </TableCell>
                <TableCell>
                  <TableSortLabel
                    active={sortBy === 'filename'}
                    direction={sortBy === 'filename' ? sortOrder : 'asc'}
                    onClick={() => handleSort('filename')}
                  >
                    Document
                  </TableSortLabel>
                </TableCell>
                <TableCell>
                  <TableSortLabel
                    active={sortBy === 'file_size_bytes'}
                    direction={sortBy === 'file_size_bytes' ? sortOrder : 'asc'}
                    onClick={() => handleSort('file_size_bytes')}
                  >
                    Size
                  </TableSortLabel>
                </TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Collection</TableCell>
                <TableCell>Notebooks</TableCell>
                <TableCell>
                  <TableSortLabel
                    active={sortBy === 'created_at'}
                    direction={sortBy === 'created_at' ? sortOrder : 'asc'}
                    onClick={() => handleSort('created_at')}
                  >
                    Created
                  </TableSortLabel>
                </TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {loading && documents.length === 0 ? (
                [...Array(5)].map((_, index) => (
                  <TableRow key={index}>
                    <TableCell><Skeleton /></TableCell>
                    <TableCell><Skeleton /></TableCell>
                    <TableCell><Skeleton /></TableCell>
                    <TableCell><Skeleton /></TableCell>
                    <TableCell><Skeleton /></TableCell>
                    <TableCell><Skeleton /></TableCell>
                    <TableCell><Skeleton /></TableCell>
                    <TableCell><Skeleton /></TableCell>
                  </TableRow>
                ))
              ) : documents.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={8} align="center">
                    <Typography variant="body1" color="text.secondary" sx={{ py: 4 }}>
                      No documents found matching your criteria
                    </Typography>
                  </TableCell>
                </TableRow>
              ) : (
                documents.map((doc) => (
                  <TableRow key={doc.document_id} hover>
                    <TableCell padding="checkbox">
                      <Checkbox
                        checked={selectedDocuments.has(doc.document_id)}
                        onChange={(e) => handleSelectDocument(doc.document_id, e.target.checked)}
                      />
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        {getFileIcon(doc.file_type)}
                        <Box>
                          <Typography variant="subtitle2" sx={{ fontWeight: 'medium' }}>
                            {doc.filename}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            ID: {doc.document_id.slice(0, 8)}...
                          </Typography>
                        </Box>
                        {doc.is_orphaned && (
                          <Chip
                            label="Orphaned"
                            size="small"
                            color="warning"
                            variant="outlined"
                          />
                        )}
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {formatFileSize(doc.file_size_bytes)}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {doc.chunks_processed}/{doc.total_chunks} chunks
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={doc.processing_status}
                        size="small"
                        color={getStatusColor(doc.processing_status) as any}
                        variant="outlined"
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" sx={{ maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {doc.milvus_collection || 'None'}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Badge badgeContent={doc.notebook_count} color="primary">
                        <NotebookIcon />
                      </Badge>
                      {doc.notebooks_using.length > 0 && (
                        <Box sx={{ mt: 0.5 }}>
                          {doc.notebooks_using.slice(0, 2).map(nb => (
                            <Chip
                              key={nb.id}
                              label={nb.name}
                              size="small"
                              variant="outlined"
                              sx={{ mr: 0.5, mb: 0.5 }}
                            />
                          ))}
                          {doc.notebooks_using.length > 2 && (
                            <Typography variant="caption" color="text.secondary">
                              +{doc.notebooks_using.length - 2} more
                            </Typography>
                          )}
                        </Box>
                      )}
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {formatRelativeTime(doc.created_at)}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Tooltip title="View Details">
                        <IconButton
                          size="small"
                          onClick={() => toggleDetailExpansion(doc.document_id)}
                        >
                          {expandedDetails.has(doc.document_id) ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </TableContainer>

        {/* Pagination */}
        <TablePagination
          component="div"
          count={pagination.total_count}
          page={pagination.page - 1}
          onPageChange={handleChangePage}
          rowsPerPage={pagination.page_size}
          onRowsPerPageChange={handleChangeRowsPerPage}
          rowsPerPageOptions={[10, 25, 50, 100]}
        />
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

export default DocumentAdminPage;