import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Card,
  CardContent,
  Grid,
  Chip,
  LinearProgress,
  Button,
  IconButton,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Alert,
  CircularProgress,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from '@mui/material';
import {
  ExpandMore,
  CheckCircle,
  Error as ErrorIcon,
  Warning,
  Download,
  Visibility,
  Assessment,
  Timeline,
  Speed,
  Memory,
  Score,
  CompareArrows,
  Refresh,
  FilterList
} from '@mui/icons-material';

interface ValidationSession {
  session_id: string;
  reference_document_id: string;
  reference_name?: string;
  input_filename: string;
  extraction_mode: string;
  validation_model: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  overall_score?: number;
  confidence_score?: number;
  completeness_score?: number;
  total_units_extracted?: number;
  units_processed?: number;
  units_failed?: number;
  average_context_usage?: number;
  max_context_usage_recorded?: number;
  processing_time_ms?: number;
  created_at: string;
  completed_at?: string;
}

interface UnitValidationResult {
  unit_index: number;
  unit_type: string;
  unit_content: string;
  validation_score: number;
  confidence_score: number;
  validation_feedback: string;
  context_tokens_used: number;
  context_usage_percentage: number;
  processing_time_ms: number;
  requires_human_review?: boolean;
  matched_reference_sections?: string[];
}

interface ValidationResults {
  session: ValidationSession;
  unit_results: UnitValidationResult[];
  summary: {
    total_units: number;
    average_score: number;
    high_scores: number;
    medium_scores: number;
    low_scores: number;
    coverage_percentage: number;
    uncovered_sections?: string[];
    recommendations?: string[];
  };
}

interface IDCResultsViewerProps {
  sessions: ValidationSession[];
  onRefresh?: () => void;
  onSessionSelect?: (sessionId: string) => void;
}

const IDCResultsViewer: React.FC<IDCResultsViewerProps> = ({
  sessions,
  onRefresh,
  onSessionSelect
}) => {
  const [selectedSession, setSelectedSession] = useState<string | null>(null);
  const [results, setResults] = useState<ValidationResults | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [expandedUnits, setExpandedUnits] = useState<Set<number>>(new Set());
  const [detailsDialogOpen, setDetailsDialogOpen] = useState(false);
  const [selectedUnit, setSelectedUnit] = useState<UnitValidationResult | null>(null);

  useEffect(() => {
    if (selectedSession) {
      loadSessionResults(selectedSession);
    }
  }, [selectedSession]);

  const loadSessionResults = async (sessionId: string) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`/api/v1/idc/validate/${sessionId}/results/detailed`);
      
      if (!response.ok) {
        throw new Error('Failed to load results');
      }
      
      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error('Failed to load results:', error);
      setError('Failed to load validation results');
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success';
      case 'processing': return 'info';
      case 'failed': return 'error';
      default: return 'default';
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'success';
    if (score >= 0.6) return 'warning';
    return 'error';
  };

  const formatDuration = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}min`;
  };

  const handleExportResults = async () => {
    if (!results) return;
    
    try {
      const response = await fetch(`/api/v1/idc/validate/${selectedSession}/export`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json'
        }
      });
      
      if (!response.ok) {
        throw new Error('Export failed');
      }
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `validation_results_${selectedSession}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Export failed:', error);
      setError('Failed to export results');
    }
  };

  const handleUnitClick = (unit: UnitValidationResult) => {
    setSelectedUnit(unit);
    setDetailsDialogOpen(true);
  };

  const filteredSessions = sessions.filter(session => {
    if (filterStatus === 'all') return true;
    return session.status === filterStatus;
  });

  return (
    <Box>
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h5">Validation Results</Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={<FilterList />}
            onClick={() => setFilterStatus(filterStatus === 'all' ? 'completed' : 'all')}
          >
            {filterStatus === 'all' ? 'All' : 'Completed Only'}
          </Button>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={onRefresh}
          >
            Refresh
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Sessions List */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, maxHeight: 600, overflow: 'auto' }}>
            <Typography variant="h6" gutterBottom>
              Validation Sessions
            </Typography>
            
            {filteredSessions.length === 0 ? (
              <Typography color="text.secondary" sx={{ mt: 2 }}>
                No validation sessions found
              </Typography>
            ) : (
              <List>
                {filteredSessions.map(session => (
                  <ListItem
                    key={session.session_id}
                    button
                    selected={selectedSession === session.session_id}
                    onClick={() => setSelectedSession(session.session_id)}
                    sx={{ mb: 1, borderRadius: 1 }}
                  >
                    <ListItemIcon>
                      {session.status === 'completed' && <CheckCircle color="success" />}
                      {session.status === 'processing' && <CircularProgress size={20} />}
                      {session.status === 'failed' && <ErrorIcon color="error" />}
                      {session.status === 'pending' && <Speed color="action" />}
                    </ListItemIcon>
                    <ListItemText
                      primary={session.input_filename}
                      secondary={
                        <Box>
                          <Typography variant="caption" display="block">
                            Mode: {session.extraction_mode}
                          </Typography>
                          <Typography variant="caption" display="block">
                            {new Date(session.created_at).toLocaleString()}
                          </Typography>
                          {session.overall_score !== undefined && (
                            <Chip
                              label={`Score: ${(session.overall_score * 100).toFixed(0)}%`}
                              size="small"
                              color={getScoreColor(session.overall_score) as any}
                              sx={{ mt: 0.5 }}
                            />
                          )}
                        </Box>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            )}
          </Paper>
        </Grid>

        {/* Results Details */}
        <Grid item xs={12} md={8}>
          {loading ? (
            <Paper sx={{ p: 4, textAlign: 'center' }}>
              <CircularProgress />
              <Typography sx={{ mt: 2 }}>Loading results...</Typography>
            </Paper>
          ) : selectedSession && results ? (
            <Box>
              {/* Summary Card */}
              <Card sx={{ mb: 3 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Validation Summary
                  </Typography>
                  
                  <Grid container spacing={2}>
                    <Grid item xs={6} sm={3}>
                      <Box sx={{ textAlign: 'center' }}>
                        <Score color="primary" sx={{ fontSize: 40 }} />
                        <Typography variant="h4">
                          {(results.session.overall_score! * 100).toFixed(0)}%
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Overall Score
                        </Typography>
                      </Box>
                    </Grid>
                    
                    <Grid item xs={6} sm={3}>
                      <Box sx={{ textAlign: 'center' }}>
                        <Assessment color="info" sx={{ fontSize: 40 }} />
                        <Typography variant="h4">
                          {results.session.units_processed}/{results.session.total_units_extracted}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Units Processed
                        </Typography>
                      </Box>
                    </Grid>
                    
                    <Grid item xs={6} sm={3}>
                      <Box sx={{ textAlign: 'center' }}>
                        <Memory color="warning" sx={{ fontSize: 40 }} />
                        <Typography variant="h4">
                          {(results.session.average_context_usage! * 100).toFixed(0)}%
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Avg Context Usage
                        </Typography>
                      </Box>
                    </Grid>
                    
                    <Grid item xs={6} sm={3}>
                      <Box sx={{ textAlign: 'center' }}>
                        <Speed color="success" sx={{ fontSize: 40 }} />
                        <Typography variant="h4">
                          {formatDuration(results.session.processing_time_ms!)}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Processing Time
                        </Typography>
                      </Box>
                    </Grid>
                  </Grid>
                  
                  <Divider sx={{ my: 2 }} />
                  
                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={4}>
                      <Typography variant="body2" color="text.secondary">
                        Extraction Mode
                      </Typography>
                      <Typography variant="body1">
                        {results.session.extraction_mode}
                      </Typography>
                    </Grid>
                    
                    <Grid item xs={12} sm={4}>
                      <Typography variant="body2" color="text.secondary">
                        Validation Model
                      </Typography>
                      <Typography variant="body1">
                        {results.session.validation_model}
                      </Typography>
                    </Grid>
                    
                    <Grid item xs={12} sm={4}>
                      <Typography variant="body2" color="text.secondary">
                        Reference Document
                      </Typography>
                      <Typography variant="body1">
                        {results.session.reference_name || 'N/A'}
                      </Typography>
                    </Grid>
                  </Grid>
                  
                  <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
                    <Button
                      variant="contained"
                      startIcon={<Download />}
                      onClick={handleExportResults}
                      size="small"
                    >
                      Export Results
                    </Button>
                  </Box>
                </CardContent>
              </Card>

              {/* Score Distribution */}
              <Card sx={{ mb: 3 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Score Distribution
                  </Typography>
                  
                  <Grid container spacing={2}>
                    <Grid item xs={4}>
                      <Paper sx={{ p: 2, bgcolor: 'success.light', color: 'success.contrastText' }}>
                        <Typography variant="h4">{results.summary.high_scores}</Typography>
                        <Typography variant="body2">High Scores (80%+)</Typography>
                      </Paper>
                    </Grid>
                    
                    <Grid item xs={4}>
                      <Paper sx={{ p: 2, bgcolor: 'warning.light', color: 'warning.contrastText' }}>
                        <Typography variant="h4">{results.summary.medium_scores}</Typography>
                        <Typography variant="body2">Medium Scores (60-80%)</Typography>
                      </Paper>
                    </Grid>
                    
                    <Grid item xs={4}>
                      <Paper sx={{ p: 2, bgcolor: 'error.light', color: 'error.contrastText' }}>
                        <Typography variant="h4">{results.summary.low_scores}</Typography>
                        <Typography variant="body2">Low Scores (&lt;60%)</Typography>
                      </Paper>
                    </Grid>
                  </Grid>
                  
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="body2" gutterBottom>
                      Coverage: {results.summary.coverage_percentage}%
                    </Typography>
                    <LinearProgress 
                      variant="determinate" 
                      value={results.summary.coverage_percentage}
                      sx={{ height: 8, borderRadius: 4 }}
                    />
                  </Box>
                </CardContent>
              </Card>

              {/* Unit Results */}
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Unit-by-Unit Results
                  </Typography>
                  
                  <TableContainer>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Unit #</TableCell>
                          <TableCell>Type</TableCell>
                          <TableCell>Score</TableCell>
                          <TableCell>Confidence</TableCell>
                          <TableCell>Context</TableCell>
                          <TableCell>Time</TableCell>
                          <TableCell>Actions</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {results.unit_results.map((unit) => (
                          <TableRow key={unit.unit_index}>
                            <TableCell>{unit.unit_index}</TableCell>
                            <TableCell>
                              <Chip label={unit.unit_type} size="small" />
                            </TableCell>
                            <TableCell>
                              <Chip
                                label={`${(unit.validation_score * 100).toFixed(0)}%`}
                                size="small"
                                color={getScoreColor(unit.validation_score) as any}
                              />
                            </TableCell>
                            <TableCell>
                              {(unit.confidence_score * 100).toFixed(0)}%
                            </TableCell>
                            <TableCell>
                              <Tooltip title={`${unit.context_tokens_used} tokens`}>
                                <Chip
                                  label={`${unit.context_usage_percentage.toFixed(0)}%`}
                                  size="small"
                                  color={unit.context_usage_percentage < 40 ? 'success' : 'warning'}
                                />
                              </Tooltip>
                            </TableCell>
                            <TableCell>{formatDuration(unit.processing_time_ms)}</TableCell>
                            <TableCell>
                              <IconButton
                                size="small"
                                onClick={() => handleUnitClick(unit)}
                              >
                                <Visibility />
                              </IconButton>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              </Card>

              {/* Recommendations */}
              {results.summary.recommendations && results.summary.recommendations.length > 0 && (
                <Card sx={{ mt: 3 }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Recommendations
                    </Typography>
                    <List>
                      {results.summary.recommendations.map((rec, index) => (
                        <ListItem key={index}>
                          <ListItemIcon>
                            <Warning color="warning" />
                          </ListItemIcon>
                          <ListItemText primary={rec} />
                        </ListItem>
                      ))}
                    </List>
                  </CardContent>
                </Card>
              )}
            </Box>
          ) : (
            <Paper sx={{ p: 4, textAlign: 'center' }}>
              <Assessment sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
              <Typography variant="h6" color="text.secondary">
                Select a validation session to view results
              </Typography>
            </Paper>
          )}
        </Grid>
      </Grid>

      {/* Unit Details Dialog */}
      <Dialog
        open={detailsDialogOpen}
        onClose={() => setDetailsDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Unit {selectedUnit?.unit_index} Details
        </DialogTitle>
        <DialogContent>
          {selectedUnit && (
            <Box>
              <Grid container spacing={2} sx={{ mb: 2 }}>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    Validation Score
                  </Typography>
                  <Typography variant="h6">
                    {(selectedUnit.validation_score * 100).toFixed(1)}%
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    Confidence Score
                  </Typography>
                  <Typography variant="h6">
                    {(selectedUnit.confidence_score * 100).toFixed(1)}%
                  </Typography>
                </Grid>
              </Grid>
              
              <Divider sx={{ my: 2 }} />
              
              <Typography variant="subtitle2" gutterBottom>
                Unit Content
              </Typography>
              <Paper sx={{ p: 2, bgcolor: 'grey.50', mb: 2 }}>
                <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                  {selectedUnit.unit_content}
                </Typography>
              </Paper>
              
              <Typography variant="subtitle2" gutterBottom>
                Validation Feedback
              </Typography>
              <Paper sx={{ p: 2, bgcolor: 'info.light', mb: 2 }}>
                <Typography variant="body2">
                  {selectedUnit.validation_feedback}
                </Typography>
              </Paper>
              
              {selectedUnit.matched_reference_sections && (
                <>
                  <Typography variant="subtitle2" gutterBottom>
                    Matched Reference Sections
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    {selectedUnit.matched_reference_sections.map((section, idx) => (
                      <Chip key={idx} label={section} size="small" />
                    ))}
                  </Box>
                </>
              )}
              
              <Box sx={{ mt: 2 }}>
                <Typography variant="caption" color="text.secondary">
                  Processing Time: {formatDuration(selectedUnit.processing_time_ms)} | 
                  Context Usage: {selectedUnit.context_usage_percentage.toFixed(1)}% ({selectedUnit.context_tokens_used} tokens)
                </Typography>
              </Box>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDetailsDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default IDCResultsViewer;