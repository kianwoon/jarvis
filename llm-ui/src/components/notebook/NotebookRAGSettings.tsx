import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardHeader,
  CardContent,
  TextField,
  Button,
  Typography,
  Switch,
  FormControl,
  FormControlLabel,
  Slider,
  Select,
  MenuItem,
  InputLabel,
  Grid,
  Divider,
  CircularProgress,
  Alert,
  Tooltip,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Psychology as RAGIcon,
  Search as SearchIcon,
  Speed as PerformanceIcon,
  Tune as TuneIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';

// RAG Settings interface
interface RAGSettings {
  reranking: {
    batch_size: number;
    num_to_rerank: number;
    rerank_weight: number;
    rerank_threshold: number;
    enable_qwen_reranker: boolean;
    enable_advanced_reranking: boolean;
  };
  performance: {
    enable_caching: boolean;
    cache_ttl_hours: number;
    connection_timeout_s: number;
    execution_timeout_ms: number;
    vector_search_nprobe: number;
    max_concurrent_searches: number;
  };
  bm25_scoring: {
    b: number;
    k1: number;
    bm25_weight: number;
    enable_bm25: boolean;
    corpus_batch_size: number;
  };
  agent_settings: {
    min_relevance_score: number;
    confidence_threshold: number;
    default_query_strategy: string;
    complex_query_threshold: number;
    collection_size_threshold: number;
    max_results_per_collection: number;
    enable_collection_auto_detection: boolean;
  };
  search_strategy: {
    keyword_weight: number;
    search_strategy: string;
    semantic_weight: number;
    similarity_threshold: number;
    enable_hybrid_search: boolean;
    fallback_to_keyword: boolean;
  };
}

// Notebook Retrieval Settings interface (from notebook_llm category)
interface NotebookRetrievalSettings {
  max_retrieval_chunks: number;
  retrieval_multiplier: number;
  collection_multiplier: number;
  neighbor_chunk_radius: number;
  include_neighboring_chunks: boolean;
  document_completeness_threshold: number;
  enable_document_aware_retrieval: boolean;
}

interface NotebookRAGSettingsProps {
  notebookId?: string;
  onSettingsChange?: (settings: RAGSettings) => void;
}

const NotebookRAGSettings: React.FC<NotebookRAGSettingsProps> = ({ 
  notebookId, 
  onSettingsChange 
}) => {
  const { enqueueSnackbar } = useSnackbar();
  
  // State management
  const [settings, setSettings] = useState<RAGSettings | null>(null);
  const [retrievalSettings, setRetrievalSettings] = useState<NotebookRetrievalSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load settings on mount
  useEffect(() => {
    loadSettings();
  }, []);

  // Load RAG settings
  const loadSettings = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Load both RAG settings and notebook retrieval settings
      const [ragResponse, notebookResponse] = await Promise.all([
        fetch(`/api/v1/settings/rag`),
        fetch(`/api/v1/settings/notebook_llm`)
      ]);
      
      // Handle RAG settings
      if (ragResponse.ok) {
        const ragData = await ragResponse.json();
        if (ragData.settings && Object.keys(ragData.settings).length > 0) {
          setSettings(ragData.settings);
          onSettingsChange?.(ragData.settings);
        } else {
          const defaultSettings = getDefaultSettings();
          setSettings(defaultSettings);
        }
      } else {
        const defaultSettings = getDefaultSettings();
        setSettings(defaultSettings);
      }

      // Handle notebook retrieval settings
      if (notebookResponse.ok) {
        const notebookData = await notebookResponse.json();
        if (notebookData.notebook_llm) {
          const retrievalData = {
            max_retrieval_chunks: notebookData.notebook_llm.max_retrieval_chunks || 200,
            retrieval_multiplier: notebookData.notebook_llm.retrieval_multiplier || 3,
            collection_multiplier: notebookData.notebook_llm.collection_multiplier || 4,
            neighbor_chunk_radius: notebookData.notebook_llm.neighbor_chunk_radius || 2,
            include_neighboring_chunks: notebookData.notebook_llm.include_neighboring_chunks ?? true,
            document_completeness_threshold: notebookData.notebook_llm.document_completeness_threshold || 0.8,
            enable_document_aware_retrieval: notebookData.notebook_llm.enable_document_aware_retrieval ?? true
          };
          setRetrievalSettings(retrievalData);
        }
      } else {
        const defaultRetrievalSettings = getDefaultRetrievalSettings();
        setRetrievalSettings(defaultRetrievalSettings);
      }
    } catch (error) {
      console.error('Failed to load RAG settings:', error);
      setError(error instanceof Error ? error.message : 'Failed to load settings');
      enqueueSnackbar('Failed to load RAG settings', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  // Save settings
  const saveSettings = async () => {
    if (!settings || !retrievalSettings) {
      enqueueSnackbar('No settings to save', { variant: 'warning' });
      return;
    }

    setSaving(true);
    setError(null);
    
    try {
      // Save both RAG settings and notebook retrieval settings
      const [ragResponse, notebookResponse] = await Promise.all([
        fetch('/api/v1/settings/rag', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            settings: settings,
            persist_to_db: true,
            reload_cache: true
          })
        }),
        // Need to get current notebook_llm settings and merge retrieval settings
        fetch('/api/v1/settings/notebook_llm').then(async (response) => {
          if (response.ok) {
            const currentData = await response.json();
            const updatedNotebookSettings = {
              ...currentData.notebook_llm,
              ...retrievalSettings
            };
            
            return fetch('/api/v1/settings/notebook_llm', {
              method: 'PUT',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                notebook_llm: updatedNotebookSettings,
                persist_to_db: true,
                reload_cache: true
              })
            });
          }
          throw new Error('Failed to load current notebook settings');
        })
      ]);

      if (ragResponse.ok && notebookResponse.ok) {
        enqueueSnackbar('RAG settings saved successfully', { variant: 'success' });
        onSettingsChange?.(settings);
      } else {
        throw new Error('Failed to save one or more settings');
      }
    } catch (error) {
      console.error('Failed to save RAG settings:', error);
      setError(error instanceof Error ? error.message : 'Failed to save settings');
      enqueueSnackbar('Failed to save RAG settings', { variant: 'error' });
    } finally {
      setSaving(false);
    }
  };

  // Reset to defaults
  const resetToDefaults = () => {
    const defaultSettings = getDefaultSettings();
    const defaultRetrievalSettings = getDefaultRetrievalSettings();
    setSettings(defaultSettings);
    setRetrievalSettings(defaultRetrievalSettings);
    enqueueSnackbar('Settings reset to defaults', { variant: 'info' });
  };

  // Handle field changes
  const handleFieldChange = (section: keyof RAGSettings, field: string, value: any) => {
    if (!settings) return;
    
    setSettings(prev => prev ? ({
      ...prev,
      [section]: {
        ...prev[section],
        [field]: value
      }
    }) : null);
  };

  // Handle retrieval field changes
  const handleRetrievalFieldChange = (field: keyof NotebookRetrievalSettings, value: any) => {
    if (!retrievalSettings) return;
    
    setRetrievalSettings(prev => prev ? ({
      ...prev,
      [field]: value
    }) : null);
  };

  // Get default settings
  const getDefaultSettings = (): RAGSettings => ({
    reranking: {
      batch_size: 10,
      num_to_rerank: 20,
      rerank_weight: 0.7,
      rerank_threshold: 0.7,
      enable_qwen_reranker: true,
      enable_advanced_reranking: true
    },
    performance: {
      enable_caching: false,
      cache_ttl_hours: 2,
      connection_timeout_s: 300,
      execution_timeout_ms: 30000,
      vector_search_nprobe: 10,
      max_concurrent_searches: 5
    },
    bm25_scoring: {
      b: 0.75,
      k1: 1.2,
      bm25_weight: 0.3,
      enable_bm25: true,
      corpus_batch_size: 2000
    },
    agent_settings: {
      min_relevance_score: 0.4,
      confidence_threshold: 0.6,
      default_query_strategy: "auto",
      complex_query_threshold: 0.15,
      collection_size_threshold: 0,
      max_results_per_collection: 10,
      enable_collection_auto_detection: true
    },
    search_strategy: {
      keyword_weight: 0.3,
      search_strategy: "auto",
      semantic_weight: 0.7,
      similarity_threshold: 0.7,
      enable_hybrid_search: true,
      fallback_to_keyword: true
    }
  });

  // Get default retrieval settings
  const getDefaultRetrievalSettings = (): NotebookRetrievalSettings => ({
    max_retrieval_chunks: 200,
    retrieval_multiplier: 3,
    collection_multiplier: 4,
    neighbor_chunk_radius: 2,
    include_neighboring_chunks: true,
    document_completeness_threshold: 0.8,
    enable_document_aware_retrieval: true
  });

  if (loading || !settings || !retrievalSettings) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" p={4}>
        <CircularProgress />
        <Typography ml={2}>Loading RAG settings...</Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Card>
        <CardHeader
          title="RAG Configuration"
          subheader="Configure Retrieval Augmented Generation settings"
          action={
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button
                size="small"
                onClick={resetToDefaults}
                startIcon={<RefreshIcon />}
              >
                Reset to Defaults
              </Button>
              <Button
                variant="contained"
                onClick={saveSettings}
                disabled={saving}
                startIcon={saving ? <CircularProgress size={16} /> : null}
              >
                {saving ? 'Saving...' : 'Save Settings'}
              </Button>
            </Box>
          }
        />
        <CardContent>
          {error && (
            <Alert severity="error" sx={{ mb: 3 }}>
              {error}
            </Alert>
          )}

          <Grid container spacing={3}>
            {/* Retrieval Settings */}
            <Grid item xs={12}>
              <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <RAGIcon />
                    <Typography variant="h6">Chunk Retrieval Settings</Typography>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="Max Retrieval Chunks"
                        type="number"
                        value={retrievalSettings.max_retrieval_chunks}
                        onChange={(e) => handleRetrievalFieldChange('max_retrieval_chunks', parseInt(e.target.value))}
                        inputProps={{ min: 50, max: 500 }}
                        fullWidth
                        helperText="Maximum chunks to retrieve from vector database"
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="Retrieval Multiplier"
                        type="number"
                        value={retrievalSettings.retrieval_multiplier}
                        onChange={(e) => handleRetrievalFieldChange('retrieval_multiplier', parseInt(e.target.value))}
                        inputProps={{ min: 1, max: 10 }}
                        fullWidth
                        helperText="Controls chunk retrieval multiplier. Note: max_sources in models is set to 15 (was 5)"
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="Collection Multiplier"
                        type="number"
                        value={retrievalSettings.collection_multiplier}
                        onChange={(e) => handleRetrievalFieldChange('collection_multiplier', parseInt(e.target.value))}
                        inputProps={{ min: 1, max: 10 }}
                        fullWidth
                        helperText="Additional multiplier for collection coverage"
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="Neighbor Chunk Radius"
                        type="number"
                        value={retrievalSettings.neighbor_chunk_radius}
                        onChange={(e) => handleRetrievalFieldChange('neighbor_chunk_radius', parseInt(e.target.value))}
                        inputProps={{ min: 0, max: 5 }}
                        fullWidth
                        helperText="How many neighboring chunks to include"
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={retrievalSettings.include_neighboring_chunks}
                            onChange={(e) => handleRetrievalFieldChange('include_neighboring_chunks', e.target.checked)}
                          />
                        }
                        label="Include Neighboring Chunks"
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={retrievalSettings.enable_document_aware_retrieval}
                            onChange={(e) => handleRetrievalFieldChange('enable_document_aware_retrieval', e.target.checked)}
                          />
                        }
                        label="Enable Document-Aware Retrieval"
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <Typography gutterBottom>
                        Document Completeness Threshold: {retrievalSettings.document_completeness_threshold}
                      </Typography>
                      <Slider
                        value={retrievalSettings.document_completeness_threshold}
                        onChange={(_, value) => handleRetrievalFieldChange('document_completeness_threshold', value)}
                        min={0}
                        max={1}
                        step={0.1}
                        marks
                        valueLabelDisplay="auto"
                      />
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>
            </Grid>

            {/* Search Strategy Settings */}
            <Grid item xs={12}>
              <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <SearchIcon />
                    <Typography variant="h6">Search Strategy</Typography>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                      <FormControl fullWidth>
                        <InputLabel>Search Strategy</InputLabel>
                        <Select
                          value={settings.search_strategy.search_strategy}
                          label="Search Strategy"
                          onChange={(e) => handleFieldChange('search_strategy', 'search_strategy', e.target.value)}
                        >
                          <MenuItem value="auto">Auto</MenuItem>
                          <MenuItem value="vector">Vector Only</MenuItem>
                          <MenuItem value="hybrid">Hybrid</MenuItem>
                          <MenuItem value="keyword">Keyword Only</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={settings.search_strategy.enable_hybrid_search}
                            onChange={(e) => handleFieldChange('search_strategy', 'enable_hybrid_search', e.target.checked)}
                          />
                        }
                        label="Enable Hybrid Search"
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <Typography gutterBottom>
                        Keyword Weight: {settings.search_strategy.keyword_weight}
                      </Typography>
                      <Slider
                        value={settings.search_strategy.keyword_weight}
                        onChange={(_, value) => handleFieldChange('search_strategy', 'keyword_weight', value)}
                        min={0}
                        max={1}
                        step={0.1}
                        marks
                        valueLabelDisplay="auto"
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <Typography gutterBottom>
                        Similarity Threshold: {settings.search_strategy.similarity_threshold}
                      </Typography>
                      <Slider
                        value={settings.search_strategy.similarity_threshold}
                        onChange={(_, value) => handleFieldChange('search_strategy', 'similarity_threshold', value)}
                        min={0}
                        max={1}
                        step={0.1}
                        marks
                        valueLabelDisplay="auto"
                      />
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>
            </Grid>

            {/* Reranking Settings */}
            <Grid item xs={12}>
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <TuneIcon />
                    <Typography variant="h6">Reranking</Typography>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={settings.reranking.enable_qwen_reranker}
                            onChange={(e) => handleFieldChange('reranking', 'enable_qwen_reranker', e.target.checked)}
                          />
                        }
                        label="Enable Qwen Reranker"
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={settings.reranking.enable_advanced_reranking}
                            onChange={(e) => handleFieldChange('reranking', 'enable_advanced_reranking', e.target.checked)}
                          />
                        }
                        label="Enable Advanced Reranking"
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="Number to Rerank"
                        type="number"
                        value={settings.reranking.num_to_rerank}
                        onChange={(e) => handleFieldChange('reranking', 'num_to_rerank', parseInt(e.target.value))}
                        inputProps={{ min: 10, max: 100 }}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="Batch Size"
                        type="number"
                        value={settings.reranking.batch_size}
                        onChange={(e) => handleFieldChange('reranking', 'batch_size', parseInt(e.target.value))}
                        inputProps={{ min: 1, max: 50 }}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <Typography gutterBottom>
                        Rerank Weight: {settings.reranking.rerank_weight}
                      </Typography>
                      <Slider
                        value={settings.reranking.rerank_weight}
                        onChange={(_, value) => handleFieldChange('reranking', 'rerank_weight', value)}
                        min={0}
                        max={1}
                        step={0.1}
                        marks
                        valueLabelDisplay="auto"
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <Typography gutterBottom>
                        Rerank Threshold: {settings.reranking.rerank_threshold}
                      </Typography>
                      <Slider
                        value={settings.reranking.rerank_threshold}
                        onChange={(_, value) => handleFieldChange('reranking', 'rerank_threshold', value)}
                        min={0}
                        max={1}
                        step={0.1}
                        marks
                        valueLabelDisplay="auto"
                      />
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>
            </Grid>

            {/* Agent Settings */}
            <Grid item xs={12}>
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <RAGIcon />
                    <Typography variant="h6">Agent Settings</Typography>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="Max Results Per Collection"
                        type="number"
                        value={settings.agent_settings.max_results_per_collection}
                        onChange={(e) => handleFieldChange('agent_settings', 'max_results_per_collection', parseInt(e.target.value))}
                        inputProps={{ min: 1, max: 50 }}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={settings.agent_settings.enable_collection_auto_detection}
                            onChange={(e) => handleFieldChange('agent_settings', 'enable_collection_auto_detection', e.target.checked)}
                          />
                        }
                        label="Enable Collection Auto Detection"
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <Typography gutterBottom>
                        Min Relevance Score: {settings.agent_settings.min_relevance_score}
                      </Typography>
                      <Slider
                        value={settings.agent_settings.min_relevance_score}
                        onChange={(_, value) => handleFieldChange('agent_settings', 'min_relevance_score', value)}
                        min={0}
                        max={1}
                        step={0.1}
                        marks
                        valueLabelDisplay="auto"
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <Typography gutterBottom>
                        Confidence Threshold: {settings.agent_settings.confidence_threshold}
                      </Typography>
                      <Slider
                        value={settings.agent_settings.confidence_threshold}
                        onChange={(_, value) => handleFieldChange('agent_settings', 'confidence_threshold', value)}
                        min={0}
                        max={1}
                        step={0.1}
                        marks
                        valueLabelDisplay="auto"
                      />
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>
            </Grid>

            {/* Performance Settings */}
            <Grid item xs={12}>
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <PerformanceIcon />
                    <Typography variant="h6">Performance</Typography>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={settings.performance.enable_caching}
                            onChange={(e) => handleFieldChange('performance', 'enable_caching', e.target.checked)}
                          />
                        }
                        label="Enable Caching"
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="Cache TTL (hours)"
                        type="number"
                        value={settings.performance.cache_ttl_hours}
                        onChange={(e) => handleFieldChange('performance', 'cache_ttl_hours', parseInt(e.target.value))}
                        inputProps={{ min: 1, max: 24 }}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="Vector Search nprobe"
                        type="number"
                        value={settings.performance.vector_search_nprobe}
                        onChange={(e) => handleFieldChange('performance', 'vector_search_nprobe', parseInt(e.target.value))}
                        inputProps={{ min: 1, max: 100 }}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="Max Concurrent Searches"
                        type="number"
                        value={settings.performance.max_concurrent_searches}
                        onChange={(e) => handleFieldChange('performance', 'max_concurrent_searches', parseInt(e.target.value))}
                        inputProps={{ min: 1, max: 20 }}
                        fullWidth
                      />
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>
            </Grid>

            {/* BM25 Scoring Settings */}
            <Grid item xs={12}>
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <TuneIcon />
                    <Typography variant="h6">BM25 Scoring</Typography>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={settings.bm25_scoring.enable_bm25}
                            onChange={(e) => handleFieldChange('bm25_scoring', 'enable_bm25', e.target.checked)}
                          />
                        }
                        label="Enable BM25"
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="Corpus Batch Size"
                        type="number"
                        value={settings.bm25_scoring.corpus_batch_size}
                        onChange={(e) => handleFieldChange('bm25_scoring', 'corpus_batch_size', parseInt(e.target.value))}
                        inputProps={{ min: 100, max: 5000 }}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <Typography gutterBottom>
                        BM25 Weight: {settings.bm25_scoring.bm25_weight}
                      </Typography>
                      <Slider
                        value={settings.bm25_scoring.bm25_weight}
                        onChange={(_, value) => handleFieldChange('bm25_scoring', 'bm25_weight', value)}
                        min={0}
                        max={1}
                        step={0.1}
                        marks
                        valueLabelDisplay="auto"
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <Typography gutterBottom>
                        K1 Parameter: {settings.bm25_scoring.k1}
                      </Typography>
                      <Slider
                        value={settings.bm25_scoring.k1}
                        onChange={(_, value) => handleFieldChange('bm25_scoring', 'k1', value)}
                        min={0.5}
                        max={2.0}
                        step={0.1}
                        marks
                        valueLabelDisplay="auto"
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <Typography gutterBottom>
                        B Parameter: {settings.bm25_scoring.b}
                      </Typography>
                      <Slider
                        value={settings.bm25_scoring.b}
                        onChange={(_, value) => handleFieldChange('bm25_scoring', 'b', value)}
                        min={0}
                        max={1}
                        step={0.1}
                        marks
                        valueLabelDisplay="auto"
                      />
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Box>
  );
};

export default NotebookRAGSettings;