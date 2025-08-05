import React, { useState } from 'react';
import {
  Card,
  CardHeader,
  CardContent,
  Typography,
  Box,
  Switch,
  Slider,
  Tabs,
  Tab,
  Alert,
  Button,
  Chip,
  Grid,
  TextField,
  FormControlLabel,
  RadioGroup,
  Radio,
  FormControl,
  Divider,
  CircularProgress,
  Badge,
  Select,
  MenuItem,
  InputLabel,
  Paper,
  Tooltip,
  IconButton,
  InputAdornment
} from '@mui/material';
import {
  Settings as SettingsIcon,
  Psychology as BrainIcon,
  CheckCircle as CheckCircleIcon,
  Cancel as XCircleIcon,
  TrendingUp as TrendingUpIcon,
  Warning as AlertTriangleIcon,
  Refresh as RefreshIcon,
  Cached as CacheIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
  Tune as TuneIcon,
  AccountTree as NetworkIcon,
  Psychology as EntityIcon,
  Link as RelationshipIcon,
  Timeline as TemporalIcon,
  Place as GeographicIcon,
  Speed as PerformanceIcon,
  ExpandMore as ExpandMoreIcon,
  Search as SearchIcon,
  FilterList as FilterIcon,
  FlashOn as QuickSetupIcon,
  Memory as ModelIcon,
  Storage as DatabaseIcon,
  Schema as SchemaIcon,
  Code as PromptsIcon,
  BugReport as TestIcon,
  Hub as AntiSiloIcon
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import PromptManagement from './PromptManagement';
import KnowledgeGraphAntiSiloSettings from './KnowledgeGraphAntiSiloSettings';

interface DiscoveredEntity {
  type: string;
  description: string;
  examples: string[];
  confidence: number;
  frequency: number;
  status: 'pending' | 'accepted' | 'rejected';
  first_seen: string;
  last_seen: string;
}

interface DiscoveredRelationship {
  type: string;
  description: string;
  inverse?: string;
  examples: string[];
  confidence: number;
  frequency: number;
  status: 'pending' | 'accepted' | 'rejected';
  first_seen: string;
  last_seen: string;
}

interface SchemaStats {
  total_entities_discovered: number;
  total_relationships_discovered: number;
  entities_accepted: number;
  entities_pending: number;
  relationships_accepted: number;
  relationships_pending: number;
  last_discovery: string | null;
  last_updated: string;
}

interface KnowledgeGraphSettingsProps {
  data: any;
  onChange: (field: string, value: any) => void;
  onShowSuccess?: (message?: string) => void;
}

// Comprehensive Extraction Settings Interfaces
interface EntityProcessingSettings {
  enable_entity_consolidation: boolean;
  entity_consolidation_threshold: number;
  enable_entity_deduplication: boolean;
  enable_flexible_entity_matching: boolean;
  enable_alias_detection: boolean;
  alias_detection_threshold: number;
  enable_abbreviation_matching: boolean;
  abbreviation_matching_threshold: number;
  enable_hub_entities: boolean;
  hub_entity_threshold: number;
  enable_fuzzy_matching: boolean;
  fuzzy_matching_threshold: number;
  entity_merge_strategy: string;
  min_entity_confidence: number;
  confidence_merge_strategy: string;
}

interface RelationshipProcessingSettings {
  enable_relationship_deduplication: boolean;
  enable_relationship_propagation: boolean;
  relationship_propagation_depth: number;
  enable_synthetic_relationships: boolean;
  synthetic_relationship_confidence: number;
  enable_cross_reference_analysis: boolean;
  cross_reference_threshold: number;
  enable_multi_chunk_relationships: boolean;
  enable_semantic_relationship_inference: boolean;
  min_relationship_confidence: number;
  relationship_quality_threshold: number;
  prioritize_pattern_relationships: boolean;
  enable_relationship_priority_deduplication: boolean;
  enable_relationship_recommendation: boolean;
  relationship_recommendation_threshold: number;
  bridge_relationship_confidence: number;
}

interface AdvancedLinkingSettings {
  enable_temporal_linking: boolean;
  temporal_linking_window: number;
  enable_temporal_coherence: boolean;
  temporal_coherence_threshold: number;
  enable_geographic_linking: boolean;
  geographic_linking_threshold: number;
  enable_contextual_linking: boolean;
  contextual_linking_threshold: number;
  enable_semantic_clustering: boolean;
  clustering_similarity_threshold: number;
  enable_hierarchical_linking: boolean;
  hierarchical_linking_depth: number;
  enable_synonym_detection: boolean;
  synonym_detection_threshold: number;
  enable_semantic_similarity_networks: boolean;
  semantic_network_threshold: number;
}

interface NetworkAnalysisSettings {
  enable_connectivity_analysis: boolean;
  connectivity_analysis_threshold: number;
  enable_isolation_detection: boolean;
  isolation_detection_threshold: number;
  enable_semantic_bridge_entities: boolean;
  semantic_bridge_threshold: number;
  enable_cross_document_linking: boolean;
  enable_multi_document_analysis: boolean;
  multi_document_analysis_threshold: number;
  enable_document_bridge_relationships: boolean;
  enable_anti_silo: boolean;
  anti_silo_similarity_threshold: number;
  anti_silo_type_boost: number;
}

interface QualityPerformanceSettings {
  enable_llm_enhancement: boolean;
  llm_confidence_threshold: number;
  enable_graph_enrichment: boolean;
  graph_enrichment_depth: number;
  max_proximity_distance: number;
  enable_type_based_clustering: boolean;
  enable_cooccurrence_analysis: boolean;
  extraction_prompt: string;
}

interface ExtractionSettings extends 
  EntityProcessingSettings, 
  RelationshipProcessingSettings, 
  AdvancedLinkingSettings, 
  NetworkAnalysisSettings, 
  QualityPerformanceSettings {}

// Knowledge Graph Model Selector Component
const KnowledgeGraphModelSelector: React.FC<{
  value: string;
  data: any;
  onChange: (field: string, value: any) => void;
  onShowSuccess?: (message?: string) => void;
}> = ({ value, data, onChange, onShowSuccess }) => {
  const [models, setModels] = React.useState<Array<{name: string, id: string, size: string, modified: string, context_length: string}>>([]);
  const [loading, setLoading] = React.useState(false);

  React.useEffect(() => {
    fetchAvailableModels();
  }, []);

  const fetchAvailableModels = async () => {
    setLoading(true);
    try {
      // Call API to get available models
      const response = await fetch('/api/v1/ollama/models');
      if (response.ok) {
        const data = await response.json();
        setModels(data.models || []);
      } else {
        // Fallback to hardcoded models if API fails
        setModels([
          { name: 'qwen3:0.6b', id: '7df6b6e09427', size: '522 MB', modified: '2 minutes ago', context_length: 'Unknown' },
          { name: 'deepseek-r1:8b', id: '6995872bfe4c', size: '5.2 GB', modified: '5 weeks ago', context_length: 'Unknown' },
          { name: 'qwen3:30b-a3b', id: '2ee832bc15b5', size: '18 GB', modified: '7 weeks ago', context_length: 'Unknown' }
        ]);
      }
    } catch (error) {
      console.error('Failed to fetch models:', error);
      // Use hardcoded models as fallback
      setModels([
        { name: 'qwen3:0.6b', id: '7df6b6e09427', size: '522 MB', modified: '2 minutes ago', context_length: 'Unknown' },
        { name: 'deepseek-r1:8b', id: '6995872bfe4c', size: '5.2 GB', modified: '5 weeks ago', context_length: 'Unknown' },
        { name: 'qwen3:30b-a3b', id: '2ee832bc15b5', size: '18 GB', modified: '7 weeks ago', context_length: 'Unknown' }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const selectedModel = models.find(m => m.name === value);

  return (
    <Card variant="outlined" sx={{ mb: 2 }}>
      <CardHeader 
        title="LLM Model Configuration"
        subheader="Select and configure your language model for knowledge graph extraction"
        action={
          <Tooltip title="Update available models and reload Knowledge Graph LLM cache">
            <Button 
              size="small" 
              onClick={async () => {
                try {
                  // CRITICAL FIX: First fetch current complete settings to avoid data loss
                  const currentResponse = await fetch('/api/v1/settings/knowledge_graph');
                  if (!currentResponse.ok) {
                    throw new Error('Failed to fetch current settings');
                  }
                  const currentData = await currentResponse.json();
                  const currentSettings = currentData.settings || {};
                  
                  console.log('Current settings keys:', Object.keys(currentSettings));
                  console.log('Critical fields preserved:', ['prompts', 'extraction', 'learning', 'discovered_schemas'].filter(key => currentSettings[key]));
                  
                  // Only update the specific fields that might have changed, preserve all others
                  const updatedSettings = {
                    ...currentSettings, // Preserve ALL existing fields
                    // Only override specific fields that are displayed/modified in the UI
                    model_config: {
                      ...currentSettings.model_config, // Preserve existing model_config fields
                      model: data?.model_config?.model || currentSettings.model_config?.model || "qwen3:30b-a3b-instruct-2507-q4_K_M",
                      temperature: data?.model_config?.temperature !== undefined ? data?.model_config?.temperature : (currentSettings.model_config?.temperature !== undefined ? currentSettings.model_config?.temperature : 0.1),
                      repeat_penalty: data?.model_config?.repeat_penalty !== undefined ? data?.model_config?.repeat_penalty : (currentSettings.model_config?.repeat_penalty !== undefined ? currentSettings.model_config?.repeat_penalty : 1.1),
                      system_prompt: data?.model_config?.system_prompt || currentSettings.model_config?.system_prompt || "You are an expert knowledge graph extraction system. Extract entities and relationships from text accurately and comprehensively.",
                      max_tokens: data?.model_config?.max_tokens || currentSettings.model_config?.max_tokens || 4096,
                      context_length: data?.model_config?.context_length || currentSettings.model_config?.context_length || 40960,
                      model_server: data?.model_config?.model_server || currentSettings.model_config?.model_server || "http://localhost:11434"
                    },
                    neo4j: data?.neo4j || currentSettings.neo4j || {
                      enabled: true,
                      host: "localhost",
                      port: 7687,
                      http_port: 7474,
                      database: "neo4j",
                      username: "neo4j",
                      password: "jarvis_neo4j_password",
                      uri: "bolt://localhost:7687"
                    },
                    anti_silo: data?.anti_silo || currentSettings.anti_silo || {
                      enabled: true,
                      similarity_threshold: 0.5,
                      cross_document_linking: true,
                      max_relationships_per_entity: 100
                    }
                  };
                  
                  console.log('Sending updated settings keys:', Object.keys(updatedSettings));
                  
                  // Save the merged settings (backend will handle deep merge as additional safety)
                  const saveResponse = await fetch('/api/v1/settings/knowledge_graph', {
                    method: 'PUT',
                    headers: {
                      'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                      settings: updatedSettings,
                      persist_to_db: true,
                      reload_cache: true
                    }),
                  });
                  
                  if (!saveResponse.ok) {
                    throw new Error('Failed to save settings');
                  }
                  
                  // Then fetch available models
                  await fetchAvailableModels();
                  
                  // Finally reload knowledge graph cache
                  const cacheResponse = await fetch('/api/v1/settings/knowledge-graph/cache/reload', { method: 'POST' });
                  if (cacheResponse.ok) {
                    const result = await cacheResponse.json();
                    console.log('Knowledge Graph LLM cache reloaded:', result);
                    if (onShowSuccess) {
                      onShowSuccess('Settings saved, models updated, and cache reloaded successfully!');
                    }
                  } else {
                    console.error('Failed to reload knowledge graph cache');
                    if (onShowSuccess) {
                      onShowSuccess('Settings saved and models updated, but cache reload failed');
                    }
                  }
                } catch (error) {
                  console.error('Error in update process:', error);
                  if (onShowSuccess) {
                    onShowSuccess('Failed to save settings or update models');
                  }
                }
              }}
              startIcon={loading ? <CircularProgress size={16} /> : <RefreshIcon />}
              disabled={loading}
              variant="contained"
            >
              {loading ? 'Updating...' : 'Update Models & Cache'}
            </Button>
          </Tooltip>
        }
      />
      <CardContent>
        {/* Model Selector */}
        <FormControl fullWidth sx={{ mb: 3 }}>
          <InputLabel id="kg-model-label">Select Model</InputLabel>
          <Select
            labelId="kg-model-label"
            value={value}
            label="Select Model"
            onChange={(e) => {
              const newModelName = e.target.value;
              const newSelectedModel = models.find(m => m.name === newModelName);
              
              // Update the model field in model_config
              const currentModelConfig = data?.model_config || {};
              const updatedModelConfig = { ...currentModelConfig, model: newModelName };
              onChange('model_config', updatedModelConfig);
              
              // Auto-update context_length and max_tokens for knowledge graph
              if (newSelectedModel && newSelectedModel.context_length !== 'Unknown') {
                const contextLength = parseInt(newSelectedModel.context_length.replace(/,/g, ''));
                if (!isNaN(contextLength)) {
                  const updatedWithContext = { ...updatedModelConfig, context_length: contextLength };
                  const suggestedMaxTokens = Math.floor(contextLength * 0.75);
                  const updatedWithTokens = { ...updatedWithContext, max_tokens: suggestedMaxTokens };
                  onChange('model_config', updatedWithTokens);
                }
              }
            }}
            disabled={loading}
            endAdornment={loading && <CircularProgress size={20} />}
          >
            {models.map((model) => (
              <MenuItem key={model.id} value={model.name}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography>{model.name}</Typography>
                    {value === model.name && (
                      <Chip 
                        label="Active" 
                        size="small" 
                        color="success" 
                        icon={<CheckCircleIcon />}
                      />
                    )}
                  </Box>
                  <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                    <Typography variant="caption" color="text.secondary">
                      {model.size}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Context: {model.context_length}
                    </Typography>
                  </Box>
                </Box>
              </MenuItem>
            ))}
            <Divider />
            <MenuItem value="custom">
              <Typography color="primary">Enter custom model name...</Typography>
            </MenuItem>
          </Select>
        </FormControl>

        {/* Current Model Information Panel */}
        {selectedModel && (
          <Paper sx={{ p: 3, backgroundColor: 'action.hover', border: '1px solid', borderColor: 'primary.light' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
              <CheckCircleIcon color="success" />
              <Typography variant="h6" color="primary">
                Current Model: {selectedModel.name}
              </Typography>
            </Box>
            
            <Grid container spacing={3}>
              <Grid item xs={12} md={3}>
                <Box sx={{ textAlign: 'center', p: 2, backgroundColor: 'background.paper', borderRadius: 1 }}>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                    Model Size
                  </Typography>
                  <Typography variant="h6" color="primary">
                    {selectedModel.size}
                  </Typography>
                </Box>
              </Grid>
              
              <Grid item xs={12} md={3}>
                <Box sx={{ textAlign: 'center', p: 2, backgroundColor: 'background.paper', borderRadius: 1 }}>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                    Context Length
                  </Typography>
                  <Typography variant="h6" color="primary">
                    {selectedModel.context_length}
                  </Typography>
                </Box>
              </Grid>
              
              <Grid item xs={12} md={3}>
                <Box sx={{ textAlign: 'center', p: 2, backgroundColor: 'background.paper', borderRadius: 1 }}>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                    Last Modified
                  </Typography>
                  <Typography variant="h6" color="primary">
                    {selectedModel.modified}
                  </Typography>
                </Box>
              </Grid>
              
              <Grid item xs={12} md={3}>
                <Box sx={{ textAlign: 'center', p: 2, backgroundColor: 'background.paper', borderRadius: 1 }}>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                    Model ID
                  </Typography>
                  <Typography variant="h6" color="primary" sx={{ fontSize: '0.9rem' }}>
                    {selectedModel.id.substring(0, 8)}...
                  </Typography>
                </Box>
              </Grid>
            </Grid>
          </Paper>
        )}
      </CardContent>
    </Card>
  );
};

const KnowledgeGraphSettings: React.FC<KnowledgeGraphSettingsProps> = ({ 
  data, 
  onChange, 
  onShowSuccess 
}) => {
  // USE EXACT DATABASE VALUES - no fallbacks, no defaults
  const actualModelConfig = data?.model_config || {};
  const { enqueueSnackbar } = useSnackbar();
  const [testText, setTestText] = useState('');
  const [discoveryResults, setDiscoveryResults] = useState<any>(null);
  const [tabValue, setTabValue] = useState(0);
  const [testingConnection, setTestingConnection] = useState(false);
  const [connectionTestResult, setConnectionTestResult] = useState<any>(null);
  const [showPassword, setShowPassword] = useState(false);


  const discoveredEntities = data?.discovered_entities || [];
  const discoveredRelationships = data?.discovered_relationships || [];
  const stats = data?.schema_stats || null;
  const [loading, setLoading] = useState(false);



  const handleApprove = async (type: string, category: 'entity' | 'relationship') => {
    try {
      // Create updated entity/relationship with approved status
      let updatedDiscoveredEntities = [...discoveredEntities];
      let updatedDiscoveredRelationships = [...discoveredRelationships];
      
      if (category === 'entity') {
        updatedDiscoveredEntities = discoveredEntities.map(entity => 
          entity.type === type ? { ...entity, status: 'accepted' as const } : entity
        );
      } else {
        updatedDiscoveredRelationships = discoveredRelationships.map(relationship => 
          relationship.type === type ? { ...relationship, status: 'accepted' as const } : relationship
        );
      }
      
      // Emit the change to parent component
      onChange('discovered_entities', updatedDiscoveredEntities);
      onChange('discovered_relationships', updatedDiscoveredRelationships);
      
      enqueueSnackbar(`${type} has been approved for use`, { 
        variant: "success" 
      });
    } catch (error) {
      enqueueSnackbar(`Failed to approve ${type}`, { 
        variant: "error" 
      });
    }
  };

  const handleReject = async (type: string, category: 'entity' | 'relationship') => {
    try {
      // Create updated entity/relationship with rejected status
      let updatedDiscoveredEntities = [...discoveredEntities];
      let updatedDiscoveredRelationships = [...discoveredRelationships];
      
      if (category === 'entity') {
        updatedDiscoveredEntities = discoveredEntities.map(entity => 
          entity.type === type ? { ...entity, status: 'rejected' as const } : entity
        );
      } else {
        updatedDiscoveredRelationships = discoveredRelationships.map(relationship => 
          relationship.type === type ? { ...relationship, status: 'rejected' as const } : relationship
        );
      }
      
      // Emit the change to parent component
      onChange('discovered_entities', updatedDiscoveredEntities);
      onChange('discovered_relationships', updatedDiscoveredRelationships);
      
      enqueueSnackbar(`${type} has been rejected`, { 
        variant: "success" 
      });
    } catch (error) {
      enqueueSnackbar(`Failed to reject ${type}`, { 
        variant: "error" 
      });
    }
  };

  const runDiscovery = async () => {
    if (!testText.trim()) return;
    
    setLoading(true);
    try {
      // Use unified settings endpoint for schema discovery
      const response = await fetch('/api/v1/settings/llm', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          // Send the discovery request as part of the knowledge_graph settings
          knowledge_graph: {
            ...data,
            discovery_request: {
              text: testText,
              action: 'discover_schema'
            }
          }
        })
      });

      if (!response.ok) {
        throw new Error(`Discovery failed: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      
      // Update the unified settings with discovery results
      if (result.knowledge_graph) {
        // Update discovered entities and relationships
        if (result.knowledge_graph.discovered_entities) {
          onChange('discovered_entities', result.knowledge_graph.discovered_entities);
        }
        
        if (result.knowledge_graph.discovered_relationships) {
          onChange('discovered_relationships', result.knowledge_graph.discovered_relationships);
        }
        
        if (result.knowledge_graph.schema_stats) {
          onChange('schema_stats', result.knowledge_graph.schema_stats);
        }
        
        const entityCount = result.knowledge_graph.discovered_entities?.length || 0;
        const relationshipCount = result.knowledge_graph.discovered_relationships?.length || 0;
        
        enqueueSnackbar(`Discovery complete: Found ${entityCount} entities and ${relationshipCount} relationships`, { 
          variant: "success" 
        });
      }
    } catch (error) {
      console.error('Schema discovery error:', error);
      enqueueSnackbar("Discovery failed: Failed to run schema discovery", { 
        variant: "error" 
      });
    } finally {
      setLoading(false);
    }
  };


  const pendingEntities = discoveredEntities.filter((e: any) => e.status === 'pending');
  const acceptedEntities = discoveredEntities.filter((e: any) => e.status === 'accepted');

  const handleTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const testNeo4jConnection = async () => {
    setTestingConnection(true);
    setConnectionTestResult(null);
    
    try {
      const response = await fetch('/api/v1/settings/knowledge-graph/test-connection', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data?.knowledge_graph || {})
      });
      
      const result = await response.json();
      setConnectionTestResult(result);
      
      if (onShowSuccess) {
        onShowSuccess(result.success ? 'Neo4j connection successful!' : `Connection failed: ${result.error}`);
      }
    } catch (error) {
      const errorResult = { success: false, error: 'Connection test failed' };
      setConnectionTestResult(errorResult);
      if (onShowSuccess) {
        onShowSuccess('Connection test failed');
      }
    } finally {
      setTestingConnection(false);
    }
  };

  return (
    <Box sx={{ p: 2 }}>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          Knowledge Graph Schema Discovery
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Configure dynamic schema discovery and manage discovered entity/relationship types
        </Typography>
      </Box>

      <Tabs value={tabValue} onChange={handleTabChange} sx={{ mb: 3 }}>
        <Tab label="Quick Setup" icon={<QuickSetupIcon />} />
        <Tab label="Model Config" icon={<ModelIcon />} />
        <Tab label="Neo4j Database" icon={<DatabaseIcon />} />
        <Tab label="Schema & Discovery" icon={<SchemaIcon />} />
        <Tab label="Prompts" icon={<PromptsIcon />} />
        <Tab label="Test & Validate" icon={<TestIcon />} />
        <Tab label="Anti-Silo" icon={<AntiSiloIcon />} />
        <Tab label="Extraction Controls" icon={<TuneIcon />} />
      </Tabs>

      {tabValue === 0 && (
        <Box>
          {/* Quick Setup Overview */}
          <Alert severity="info" sx={{ mb: 3 }}>
            <Typography variant="body2">
              <strong>Quick Setup:</strong> Configure the essential settings to get your knowledge graph extraction working. 
              For advanced configuration, use the other tabs.
            </Typography>
          </Alert>

          {/* Essential Configuration */}
          <Grid container spacing={3}>
            {/* Mode Selection */}
            <Grid item xs={12} md={6}>
              <Card variant="outlined">
                <CardHeader 
                  title="Extraction Mode"
                  subheader="Choose how the AI extracts knowledge"
                />
                <CardContent>
                  <FormControl component="fieldset">
                    <RadioGroup 
                      value={data?.mode || 'thinking'} 
                      onChange={(e) => onChange('mode', e.target.value)}
                    >
                      <FormControlLabel 
                        value="thinking" 
                        control={<Radio />} 
                        label={
                          <Box>
                            <Typography variant="body2" fontWeight={600}>Thinking Mode (Recommended)</Typography>
                            <Typography variant="caption" color="text.secondary">
                              More accurate with step-by-step reasoning
                            </Typography>
                          </Box>
                        } 
                      />
                      <FormControlLabel 
                        value="non-thinking" 
                        control={<Radio />} 
                        label={
                          <Box>
                            <Typography variant="body2" fontWeight={600}>Direct Mode</Typography>
                            <Typography variant="caption" color="text.secondary">
                              Faster extraction without reasoning steps
                            </Typography>
                          </Box>
                        } 
                      />
                    </RadioGroup>
                  </FormControl>
                </CardContent>
              </Card>
            </Grid>

            {/* Schema Mode */}
            <Grid item xs={12} md={6}>
              <Card variant="outlined">
                <CardHeader 
                  title="Schema Strategy"
                  subheader="How to define entity and relationship types"
                />
                <CardContent>
                  <FormControl component="fieldset">
                    <RadioGroup 
                      value={data?.schema_mode || 'dynamic'} 
                      onChange={(e) => onChange('schema_mode', e.target.value)}
                    >
                      <FormControlLabel 
                        value="hybrid" 
                        control={<Radio />} 
                        label={
                          <Box>
                            <Typography variant="body2" fontWeight={600}>Smart Hybrid (Recommended)</Typography>
                            <Typography variant="caption" color="text.secondary">
                              AI discovers new types + uses predefined ones
                            </Typography>
                          </Box>
                        } 
                      />
                      <FormControlLabel 
                        value="dynamic" 
                        control={<Radio />} 
                        label={
                          <Box>
                            <Typography variant="body2" fontWeight={600}>Pure LLM Discovery</Typography>
                            <Typography variant="caption" color="text.secondary">
                              AI discovers all entity and relationship types from your content
                            </Typography>
                          </Box>
                        } 
                      />
                    </RadioGroup>
                  </FormControl>
                </CardContent>
              </Card>
            </Grid>

            {/* Discovery Settings */}
            <Grid item xs={12} md={6}>
              <Card variant="outlined">
                <CardHeader 
                  title="Discovery Controls"
                  subheader="Fine-tune what gets discovered"
                />
                <CardContent>
                  <Box sx={{ mb: 2 }}>
                    <FormControlLabel 
                      control={
                        <Switch
                          checked={data?.entity_discovery?.enabled || false}
                          onChange={(e) => {
                            const updatedEntityDiscovery = { 
                              ...data?.entity_discovery, 
                              enabled: e.target.checked 
                            };
                            onChange('entity_discovery', updatedEntityDiscovery);
                          }}
                        />
                      }
                      label="Discover New Entity Types"
                    />
                    <Typography variant="caption" color="text.secondary" display="block">
                      Let AI find new types of entities (people, places, etc.)
                    </Typography>
                  </Box>
                  <Box sx={{ mb: 2 }}>
                    <FormControlLabel 
                      control={
                        <Switch
                          checked={data?.relationship_discovery?.enabled || false}
                          onChange={(e) => {
                            const updatedRelationshipDiscovery = { 
                              ...data?.relationship_discovery, 
                              enabled: e.target.checked 
                            };
                            onChange('relationship_discovery', updatedRelationshipDiscovery);
                          }}
                        />
                      }
                      label="Discover New Relationship Types"
                    />
                    <Typography variant="caption" color="text.secondary" display="block">
                      Let AI find new types of relationships between entities
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Model Selection */}
            <Grid item xs={12} md={6}>
              <Card variant="outlined">
                <CardHeader 
                  title="AI Model"
                  subheader="Choose the model for knowledge extraction"
                />
                <CardContent>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    Select a powerful model for accurate entity and relationship extraction
                  </Typography>
                  <TextField
                    fullWidth
                    label="Model Name"
                    value={actualModelConfig.model || 'NO DATA'}
                    onChange={(e) => {
                      const updatedModelConfig = { ...actualModelConfig, model: e.target.value };
                      onChange('model_config', updatedModelConfig);
                    }}
                    variant="outlined"
                    placeholder="e.g., gpt-4, claude-3-sonnet"
                    helperText="Use a capable model like GPT-4 or Claude for best results"
                  />
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Getting Started Guide */}
          <Card sx={{ mt: 3, backgroundColor: 'action.hover' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <TrendingUpIcon color="primary" />
                Getting Started Guide
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={4}>
                  <Typography variant="subtitle2" gutterBottom>1. Choose Your Settings</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Configure mode, schema strategy, and model above
                  </Typography>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Typography variant="subtitle2" gutterBottom>2. Test Your Setup</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Use "Test & Validate" tab to try extraction on sample text
                  </Typography>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Typography variant="subtitle2" gutterBottom>3. Review Results</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Check "Schema & Discovery" tab to approve discovered types
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Box>
      )}

      {tabValue === 1 && (
        <Box>
          {/* Use the enhanced ModelSelector component */}
          <KnowledgeGraphModelSelector
            value={actualModelConfig.model || ''}
            data={data}
            onChange={onChange}
            onShowSuccess={onShowSuccess}
          />
          
          {/* Additional Configuration Fields */}
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card variant="outlined">
                <CardHeader 
                  title="Fine-tuning Parameters" 
                  subheader="Control model behavior and output quality"
                />
                <CardContent>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <TextField
                      label="Model Server URL"
                      value={actualModelConfig.model_server || ''}
                      onChange={(e) => {
                        const updatedModelConfig = { ...actualModelConfig, model_server: e.target.value };
                        onChange('model_config', updatedModelConfig);
                      }}
                      fullWidth
                      variant="outlined"
                      placeholder="e.g., http://localhost:8000, https://api.openai.com"
                      helperText="API endpoint for the model"
                    />
                    <TextField
                      label="Context Length"
                      type="number"
                      value={actualModelConfig.context_length || ''}
                      onChange={(e) => {
                        const updatedModelConfig = { ...actualModelConfig, context_length: parseInt(e.target.value) || undefined };
                        onChange('model_config', updatedModelConfig);
                      }}
                      fullWidth
                      variant="outlined"
                      placeholder="e.g., 8192, 32768, 128000"
                      helperText="Maximum context window size in tokens (auto-updated when model selected)"
                    />
                    <TextField
                      label="Max Output Tokens"
                      type="number"
                      value={actualModelConfig.max_tokens || ''}
                      onChange={(e) => {
                        const updatedModelConfig = { ...actualModelConfig, max_tokens: parseInt(e.target.value) || undefined };
                        onChange('model_config', updatedModelConfig);
                      }}
                      fullWidth
                      variant="outlined"
                      placeholder="e.g., 4096, 8192"
                      helperText="Maximum tokens the model can generate (auto-updated to 75% of context length)"
                    />
                    <TextField
                      label="Temperature"
                      type="number"
                      value={actualModelConfig.temperature || ''}
                      onChange={(e) => {
                        const updatedModelConfig = { ...actualModelConfig, temperature: parseFloat(e.target.value) || undefined };
                        onChange('model_config', updatedModelConfig);
                      }}
                      fullWidth
                      variant="outlined"
                      placeholder="e.g., 0.3"
                      inputProps={{ step: 0.1, min: 0, max: 2 }}
                      helperText="Lower = more focused, Higher = more creative (0.0-2.0)"
                    />
                    <TextField
                      label="Repeat Penalty"
                      type="number"
                      value={actualModelConfig.repeat_penalty || ''}
                      onChange={(e) => {
                        const updatedModelConfig = { ...actualModelConfig, repeat_penalty: parseFloat(e.target.value) || undefined };
                        onChange('model_config', updatedModelConfig);
                      }}
                      fullWidth
                      variant="outlined"
                      placeholder="e.g., 1.1"
                      inputProps={{ step: 0.05, min: 1.0, max: 2.0 }}
                      helperText="Penalty for repetition (1.0 = none, >1.0 = penalty)"
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card variant="outlined">
                <CardHeader 
                  title="🔄 Two-Step Knowledge Extraction" 
                  subheader="Advanced two-phase process: Document Analysis → Entity Extraction"
                />
                <CardContent>
                  <Alert severity="info" sx={{ mb: 2 }}>
                    <Typography variant="body2">
                      <strong>Step 1:</strong> LLM analyzes document naturally → <strong>Step 2:</strong> LLM extracts structured entities/relationships from analysis
                    </Typography>
                  </Alert>

                  <Box sx={{ mb: 3 }}>
                    <Typography variant="h6" sx={{ mb: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
                      📊 Step 1: Document Analysis
                    </Typography>
                    <TextField
                      multiline
                      rows={8}
                      value={actualModelConfig.analysis_prompt || ''}
                      onChange={(e) => {
                        const updatedModelConfig = { ...actualModelConfig, analysis_prompt: e.target.value };
                        onChange('model_config', updatedModelConfig);
                      }}
                      fullWidth
                      variant="outlined"
                      placeholder="Analyze this business document comprehensively..."
                      helperText="Prompt for natural document analysis (should encourage detailed breakdown of entities and relationships)"
                    />
                  </Box>
                  
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="h6" sx={{ mb: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
                      🎯 Step 2: Entity Extraction
                    </Typography>
                    <TextField
                      multiline
                      rows={8}
                      value={actualModelConfig.extraction_prompt || ''}
                      onChange={(e) => {
                        const updatedModelConfig = { ...actualModelConfig, extraction_prompt: e.target.value };
                        onChange('model_config', updatedModelConfig);
                      }}
                      fullWidth
                      variant="outlined"
                      placeholder="Extract entities and relationships from this analysis and return ONLY valid JSON..."
                      helperText="Prompt for structured extraction from analysis (should demand specific JSON format and comprehensive extraction)"
                    />
                  </Box>

                  <Divider sx={{ my: 2 }} />
                  
                  {/* Backward compatibility field */}
                  <Box sx={{ opacity: 0.6 }}>
                    <Typography variant="subtitle2" sx={{ mb: 1, color: 'text.secondary' }}>
                      Legacy Single-Step Prompt (Deprecated)
                    </Typography>
                    <TextField
                      multiline
                      rows={3}
                      value={actualModelConfig.system_prompt || ''}
                      onChange={(e) => {
                        const updatedModelConfig = { ...actualModelConfig, system_prompt: e.target.value };
                        onChange('model_config', updatedModelConfig);
                      }}
                      fullWidth
                      variant="outlined"
                      placeholder="Legacy single-step prompt (use two-step prompts above instead)..."
                      helperText="For backward compatibility only - use the two-step prompts above for better results"
                      size="small"
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Neo4j Connection Test */}
          <Card variant="outlined" sx={{ mt: 3 }}>
            <CardHeader 
              title="Neo4j Connection Test"
              subheader="Test the connection to your Neo4j knowledge graph database"
            />
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                <Button
                  variant="contained"
                  onClick={testNeo4jConnection}
                  disabled={testingConnection}
                  startIcon={testingConnection ? <CircularProgress size={16} /> : <CheckCircleIcon />}
                >
                  {testingConnection ? 'Testing...' : 'Test Connection'}
                </Button>
              </Box>
              
              {connectionTestResult && (
                <Alert 
                  severity={connectionTestResult.success ? 'success' : 'error'} 
                  sx={{ mt: 2 }}
                >
                  <Typography variant="body2">
                    {connectionTestResult.success 
                      ? connectionTestResult.message || 'Connection successful!'
                      : connectionTestResult.error || 'Connection failed'
                    }
                  </Typography>
                  {connectionTestResult.success && connectionTestResult.database_info && (
                    <Box sx={{ mt: 1 }}>
                      <Typography variant="caption" component="div">
                        Database: {connectionTestResult.database_info.database_name || 'neo4j'}
                      </Typography>
                      <Typography variant="caption" component="div">
                        Nodes: {connectionTestResult.database_info.node_count || 0}
                      </Typography>
                      <Typography variant="caption" component="div">
                        Relationships: {connectionTestResult.database_info.relationship_count || 0}
                      </Typography>
                    </Box>
                  )}
                </Alert>
              )}
            </CardContent>
          </Card>

          {/* Performance Tips */}
          <Card sx={{ mt: 3, backgroundColor: 'action.hover' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <TrendingUpIcon color="primary" />
                Performance Tips
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={4}>
                  <Typography variant="subtitle2" gutterBottom>Model Selection</Typography>
                  <Typography variant="body2" color="text.secondary">
                    GPT-4 and Claude-3.5-Sonnet provide the best extraction accuracy
                  </Typography>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Typography variant="subtitle2" gutterBottom>Temperature Settings</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Use 0.1-0.3 for structured extraction, 0.5-0.7 for creative discovery
                  </Typography>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Typography variant="subtitle2" gutterBottom>Context Window</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Larger context windows allow processing of longer documents
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Box>
      )}

      {tabValue === 2 && (
        <Box>
          <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>
            Neo4j Database Configuration
          </Typography>
          
          <Grid container spacing={3}>
            {/* Connection Settings */}
            <Grid item xs={12} md={6}>
              <Card variant="outlined">
                <CardHeader 
                  title="Connection Settings"
                  subheader="Configure Neo4j database connection"
                />
                <CardContent>
                  <Grid container spacing={2}>
                    <Grid item xs={12}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={data?.neo4j?.enabled ?? true}
                            onChange={(e) => {
                              const updatedNeo4j = { ...data.neo4j, enabled: e.target.checked };
                              onChange('neo4j', updatedNeo4j);
                            }}
                          />
                        }
                        label="Enable Neo4j Database"
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <TextField
                        fullWidth
                        label="Host"
                        value={data?.neo4j?.host || 'neo4j'}
                        onChange={(e) => {
                          const updatedNeo4j = { ...data.neo4j, host: e.target.value };
                          onChange('neo4j', updatedNeo4j);
                        }}
                        size="small"
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <TextField
                        fullWidth
                        label="Bolt Port"
                        type="number"
                        value={data?.neo4j?.port || 7687}
                        onChange={(e) => {
                          const updatedNeo4j = { ...data.neo4j, port: parseInt(e.target.value) };
                          onChange('neo4j', updatedNeo4j);
                        }}
                        size="small"
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <TextField
                        fullWidth
                        label="HTTP Port"
                        type="number"
                        value={data?.neo4j?.http_port || 7474}
                        onChange={(e) => {
                          const updatedNeo4j = { ...data.neo4j, http_port: parseInt(e.target.value) };
                          onChange('neo4j', updatedNeo4j);
                        }}
                        size="small"
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <TextField
                        fullWidth
                        label="Database Name"
                        value={data?.neo4j?.database || 'neo4j'}
                        onChange={(e) => {
                          const updatedNeo4j = { ...data.neo4j, database: e.target.value };
                          onChange('neo4j', updatedNeo4j);
                        }}
                        size="small"
                      />
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>

            {/* Authentication */}
            <Grid item xs={12} md={6}>
              <Card variant="outlined">
                <CardHeader 
                  title="Authentication"
                  subheader="Neo4j database credentials"
                />
                <CardContent>
                  <Grid container spacing={2}>
                    <Grid item xs={12}>
                      <TextField
                        fullWidth
                        label="Username"
                        value={data?.neo4j?.username || 'neo4j'}
                        onChange={(e) => {
                          const updatedNeo4j = { ...data.neo4j, username: e.target.value };
                          onChange('neo4j', updatedNeo4j);
                        }}
                        size="small"
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <TextField
                        fullWidth
                        label="Password"
                        type={showPassword ? 'text' : 'password'}
                        value={data?.neo4j?.password || 'jarvis_neo4j_password'}
                        onChange={(e) => {
                          const updatedNeo4j = { ...data.neo4j, password: e.target.value };
                          onChange('neo4j', updatedNeo4j);
                        }}
                        size="small"
                        InputProps={{
                          endAdornment: (
                            <InputAdornment position="end">
                              <IconButton
                                onClick={() => setShowPassword(!showPassword)}
                                edge="end"
                                size="small"
                              >
                                {showPassword ? <VisibilityOffIcon /> : <VisibilityIcon />}
                              </IconButton>
                            </InputAdornment>
                          ),
                        }}
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <TextField
                        fullWidth
                        label="Connection URI"
                        value={data?.neo4j?.uri || 'bolt://neo4j:7687'}
                        onChange={(e) => {
                          const updatedNeo4j = { ...data.neo4j, uri: e.target.value };
                          onChange('neo4j', updatedNeo4j);
                        }}
                        size="small"
                        helperText="Full connection URI (auto-generated from host:port)"
                      />
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>

            {/* Connection Test */}
            <Grid item xs={12}>
              <Card variant="outlined">
                <CardHeader 
                  title="Connection Test"
                  subheader="Test the connection to your Neo4j database"
                />
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                    <Button
                      variant="contained"
                      onClick={testNeo4jConnection}
                      disabled={testingConnection}
                      startIcon={testingConnection ? <CircularProgress size={16} /> : <CheckCircleIcon />}
                    >
                      {testingConnection ? 'Testing...' : 'Test Neo4j Connection'}
                    </Button>
                  </Box>
                  
                  {connectionTestResult && (
                    <Alert 
                      severity={connectionTestResult.success ? "success" : "error"}
                      sx={{ mt: 2 }}
                    >
                      {connectionTestResult.success ? 'Connection successful!' : `Connection failed: ${connectionTestResult.error}`}
                      {connectionTestResult.success && connectionTestResult.database_info && (
                        <Box sx={{ mt: 1 }}>
                          <Typography variant="caption" component="div">
                            Database: {connectionTestResult.database_info.database_name || 'neo4j'}
                          </Typography>
                          <Typography variant="caption" component="div">
                            Nodes: {connectionTestResult.database_info.node_count || 0}
                          </Typography>
                          <Typography variant="caption" component="div">
                            Relationships: {connectionTestResult.database_info.relationship_count || 0}
                          </Typography>
                        </Box>
                      )}
                    </Alert>
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Box>
      )}

      {tabValue === 3 && (
        <Box>
          <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>
            Schema Strategy &amp; Discovery Management
          </Typography>

          <Alert severity="info" sx={{ mb: 3 }}>
            <Typography variant="body2">
              Configure how entity and relationship types are defined, and review AI-discovered schema elements.
            </Typography>
          </Alert>

          {/* Schema Strategy */}
          <Card variant="outlined" sx={{ mb: 3 }}>
            <CardHeader 
              title="Schema Strategy" 
              subheader="Choose how entity and relationship types are defined"
            />
            <CardContent>
              <RadioGroup 
                value={data?.schema_mode || 'dynamic'} 
                onChange={(e) => onChange('schema_mode', e.target.value)}
              >
                <FormControlLabel 
                  value="hybrid" 
                  control={<Radio />} 
                  label={
                    <Box>
                      <Typography variant="body2" fontWeight={600}>Smart Hybrid (Recommended)</Typography>
                      <Typography variant="caption" color="text.secondary">
                        Combines predefined types with AI-discovered types for comprehensive coverage
                      </Typography>
                    </Box>
                  } 
                />
                <FormControlLabel 
                  value="dynamic" 
                  control={<Radio />} 
                  label={
                    <Box>
                      <Typography variant="body2" fontWeight={600}>AI Discovery Only</Typography>
                      <Typography variant="caption" color="text.secondary">
                        Relies entirely on AI to discover and define entity/relationship types from content
                      </Typography>
                    </Box>
                  } 
                />
              </RadioGroup>
            </CardContent>
          </Card>

          {/* Discovery Configuration */}
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card variant="outlined">
                <CardHeader 
                  title="Entity Discovery" 
                  subheader="Configure AI discovery of entity types"
                />
                <CardContent>
                  <Box sx={{ mb: 2 }}>
                    <FormControlLabel 
                      control={
                        <Switch
                          checked={data?.entity_discovery?.enabled || false}
                          onChange={(e) => {
                            const updatedEntityDiscovery = { 
                              ...data?.entity_discovery, 
                              enabled: e.target.checked 
                            };
                            onChange('entity_discovery', updatedEntityDiscovery);
                          }}
                        />
                      }
                      label="Enable Entity Discovery"
                    />
                    <Typography variant="caption" color="text.secondary" display="block">
                      Let AI discover new entity types (Person, Organization, etc.)
                    </Typography>
                  </Box>
                  <Typography gutterBottom sx={{ fontWeight: 600 }}>
                    Confidence Threshold: {(data?.entity_discovery?.confidence_threshold || 0.75).toFixed(2)}
                  </Typography>
                  <Slider
                    value={data?.entity_discovery?.confidence_threshold || 0.75}
                    onChange={(_, value) => {
                      const updatedEntityDiscovery = { 
                        ...data?.entity_discovery, 
                        confidence_threshold: value as number 
                      };
                      onChange('entity_discovery', updatedEntityDiscovery);
                    }}
                    min={0.1}
                    max={1}
                    step={0.05}
                    valueLabelDisplay="auto"
                    sx={{ mb: 1 }}
                  />
                  <Typography variant="body2" color="text.secondary">
                    Higher values = stricter filtering (fewer but more reliable discoveries)
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card variant="outlined">
                <CardHeader 
                  title="Relationship Discovery" 
                  subheader="Configure AI discovery of relationship types"
                />
                <CardContent>
                  <Box sx={{ mb: 2 }}>
                    <FormControlLabel 
                      control={
                        <Switch
                          checked={data?.relationship_discovery?.enabled || false}
                          onChange={(e) => {
                            const updatedRelationshipDiscovery = { 
                              ...data?.relationship_discovery, 
                              enabled: e.target.checked 
                            };
                            onChange('relationship_discovery', updatedRelationshipDiscovery);
                          }}
                        />
                      }
                      label="Enable Relationship Discovery"
                    />
                    <Typography variant="caption" color="text.secondary" display="block">
                      Let AI discover new relationship types (works_for, located_in, etc.)
                    </Typography>
                  </Box>
                  <Typography gutterBottom sx={{ fontWeight: 600 }}>
                    Confidence Threshold: {(data?.relationship_discovery?.confidence_threshold || 0.7).toFixed(2)}
                  </Typography>
                  <Slider
                    value={data?.relationship_discovery?.confidence_threshold || 0.7}
                    onChange={(_, value) => {
                      const updatedRelationshipDiscovery = { 
                        ...data?.relationship_discovery, 
                        confidence_threshold: value as number 
                      };
                      onChange('relationship_discovery', updatedRelationshipDiscovery);
                    }}
                    min={0.1}
                    max={1}
                    step={0.05}
                    valueLabelDisplay="auto"
                    sx={{ mb: 1 }}
                  />
                  <Typography variant="body2" color="text.secondary">
                    Higher values = stricter filtering (fewer but more reliable discoveries)
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>


          {/* Discovery Statistics */}
          {stats && (
            <Card variant="outlined" sx={{ mt: 3 }}>
              <CardHeader 
                title="Discovery Statistics" 
                subheader="Summary of AI-discovered schema elements"
              />
              <CardContent>
                <Grid container spacing={3}>
                  <Grid item xs={6} md={3}>
                    <Box textAlign="center" sx={{ p: 2, borderRadius: 1, backgroundColor: 'action.hover' }}>
                      <Typography variant="h3" sx={{ fontWeight: 700, color: 'primary.main' }}>
                        {stats.total_entities_discovered}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Entities Discovered
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Box textAlign="center" sx={{ p: 2, borderRadius: 1, backgroundColor: 'action.hover' }}>
                      <Typography variant="h3" sx={{ fontWeight: 700, color: 'success.main' }}>
                        {stats.entities_accepted}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Entities Accepted
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Box textAlign="center" sx={{ p: 2, borderRadius: 1, backgroundColor: 'action.hover' }}>
                      <Typography variant="h3" sx={{ fontWeight: 700, color: 'primary.main' }}>
                        {stats.total_relationships_discovered}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Relationships Discovered
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Box textAlign="center" sx={{ p: 2, borderRadius: 1, backgroundColor: 'action.hover' }}>
                      <Typography variant="h3" sx={{ fontWeight: 700, color: 'success.main' }}>
                        {stats.relationships_accepted}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Relationships Accepted
                      </Typography>
                    </Box>
                  </Grid>
                </Grid>
                {stats.last_discovery && (
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 2, textAlign: 'center' }}>
                    Last discovery: {new Date(stats.last_discovery).toLocaleString()}
                  </Typography>
                )}
              </CardContent>
            </Card>
          )}

          {/* Review Discovered Types */}
          {(pendingEntities.length > 0 || acceptedEntities.length > 0) && (
            <Card variant="outlined" sx={{ mt: 3 }}>
              <CardHeader 
                title="Review Discovered Types" 
                subheader="Approve or reject AI-discovered entity types"
              />
              <CardContent>
                {pendingEntities.length > 0 && (
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      Pending Review
                      <Badge badgeContent={pendingEntities.length} color="warning">
                        <AlertTriangleIcon />
                      </Badge>
                    </Typography>
                    {pendingEntities.map((entity: any) => (
                      <Card key={entity.type} variant="outlined" sx={{ mb: 2, borderLeft: '4px solid orange' }}>
                        <CardContent>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                            <Box flex={1}>
                              <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
                                <Typography variant="h6">{entity.type}</Typography>
                                <Chip label={`${(entity.confidence * 100).toFixed(0)}%`} size="small" color="warning" />
                                <Chip label={`${entity.frequency} occurrences`} size="small" variant="outlined" />
                              </Box>
                              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                                {entity.description}
                              </Typography>
                              <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                                {entity.examples.slice(0, 3).map((example: any, idx: number) => (
                                  <Chip key={idx} label={example} size="small" variant="outlined" />
                                ))}
                                {entity.examples.length > 3 && (
                                  <Chip label={`+${entity.examples.length - 3} more`} size="small" variant="outlined" />
                                )}
                              </Box>
                            </Box>
                            <Box sx={{ display: 'flex', gap: 1, ml: 2 }}>
                              <Button
                                size="small"
                                variant="contained"
                                color="success"
                                onClick={() => handleApprove(entity.type, 'entity')}
                                startIcon={<CheckCircleIcon />}
                              >
                                Approve
                              </Button>
                              <Button
                                size="small"
                                variant="outlined"
                                color="error"
                                onClick={() => handleReject(entity.type, 'entity')}
                                startIcon={<XCircleIcon />}
                              >
                                Reject
                              </Button>
                            </Box>
                          </Box>
                        </CardContent>
                      </Card>
                    ))}
                  </Box>
                )}

                {acceptedEntities.length > 0 && (
                  <Box>
                    <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      Accepted Types
                      <Badge badgeContent={acceptedEntities.length} color="success">
                        <CheckCircleIcon />
                      </Badge>
                    </Typography>
                    <Grid container spacing={1}>
                      {acceptedEntities.map((entity: any) => (
                        <Grid item xs={12} sm={6} md={4} key={entity.type}>
                          <Card variant="outlined" sx={{ borderLeft: '4px solid green' }}>
                            <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                              <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                                {entity.type}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                {entity.description}
                              </Typography>
                            </CardContent>
                          </Card>
                        </Grid>
                      ))}
                    </Grid>
                  </Box>
                )}
              </CardContent>
            </Card>
          )}
        </Box>
      )}

      {tabValue === 4 && (
        <Box>
          <PromptManagement 
            data={data?.prompts || []} 
            onChange={(field, value) => onChange('prompts', value)} 
            onShowSuccess={onShowSuccess}
          />
        </Box>
      )}

      {tabValue === 5 && (
        <Box>
          <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>
            Test &amp; Validate Configuration
          </Typography>

          <Alert severity="info" sx={{ mb: 3 }}>
            <Typography variant="body2">
              Test your knowledge graph extraction settings by running them on sample text. 
              This helps validate your configuration before processing real documents.
            </Typography>
          </Alert>

          {/* Test Input */}
          <Card variant="outlined" sx={{ mb: 3 }}>
            <CardHeader 
              title="Sample Text Input" 
              subheader="Enter or paste text to test knowledge extraction"
            />
            <CardContent>
              <TextField
                fullWidth
                multiline
                rows={8}
                value={testText}
                onChange={(e) => setTestText(e.target.value)}
                placeholder="Example: John Smith works as CEO at Microsoft Corporation in Seattle. The company was founded in 1975 and has partnerships with Amazon and Google. Microsoft's headquarters are located in Redmond, Washington."
                variant="outlined"
                sx={{ 
                  '& .MuiInputBase-input': {
                    fontFamily: 'monospace',
                    fontSize: '14px'
                  }
                }}
                helperText="Enter descriptive text with entities (people, organizations, locations) and relationships"
              />
              
              <Box sx={{ mt: 2, display: 'flex', gap: 2, alignItems: 'center' }}>
                <Button 
                  variant="contained"
                  onClick={runDiscovery}
                  disabled={!testText.trim() || loading}
                  startIcon={loading ? <CircularProgress size={20} /> : <BrainIcon />}
                  size="large"
                >
                  {loading ? 'Analyzing...' : 'Test Extraction'}
                </Button>
                
                <Button 
                  variant="outlined"
                  onClick={() => setTestText("John Smith works as CEO at Microsoft Corporation in Seattle. The company was founded in 1975 and has partnerships with Amazon and Google. Microsoft's headquarters are located in Redmond, Washington.")}
                  disabled={loading}
                >
                  Use Sample Text
                </Button>
                
                <Button 
                  variant="text"
                  onClick={() => setTestText('')}
                  disabled={loading}
                >
                  Clear
                </Button>
              </Box>
            </CardContent>
          </Card>

          {/* Results */}
          {discoveryResults && (
            <Grid container spacing={3}>
              {/* Summary */}
              <Grid item xs={12}>
                <Alert severity="success" sx={{ mb: 2 }}>
                  <Typography variant="body1" sx={{ fontWeight: 600 }}>
                    Extraction Complete! 
                  </Typography>
                  <Typography variant="body2">
                    Found {discoveryResults.discovered_entities?.length || 0} entities and {discoveryResults.discovered_relationships?.length || 0} relationships
                  </Typography>
                </Alert>
              </Grid>

              {/* Entities */}
              {discoveryResults.discovered_entities?.length > 0 && (
                <Grid item xs={12} md={6}>
                  <Card>
                    <CardHeader 
                      title="Extracted Entities" 
                      subheader={`${discoveryResults.discovered_entities.length} entities discovered`}
                    />
                    <CardContent>
                      <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
                        {discoveryResults.discovered_entities.map((entity: any, idx: number) => (
                          <Card key={idx} variant="outlined" sx={{ mb: 2 }}>
                            <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                                  {entity.name || entity.value}
                                </Typography>
                                <Chip label={entity.type} size="small" color="primary" />
                              </Box>
                              {entity.description && (
                                <Typography variant="body2" color="text.secondary">
                                  {entity.description}
                                </Typography>
                              )}
                              {entity.confidence && (
                                <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                                  Confidence: {(entity.confidence * 100).toFixed(0)}%
                                </Typography>
                              )}
                            </CardContent>
                          </Card>
                        ))}
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              )}

              {/* Relationships */}
              {discoveryResults.discovered_relationships?.length > 0 && (
                <Grid item xs={12} md={6}>
                  <Card>
                    <CardHeader 
                      title="Extracted Relationships" 
                      subheader={`${discoveryResults.discovered_relationships.length} relationships discovered`}
                    />
                    <CardContent>
                      <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
                        {discoveryResults.discovered_relationships.map((rel: any, idx: number) => (
                          <Card key={idx} variant="outlined" sx={{ mb: 2 }}>
                            <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                                <Typography variant="body1" sx={{ fontWeight: 600 }}>
                                  {rel.source} 
                                </Typography>
                                <Chip label={rel.type} size="small" color="secondary" />
                                <Typography variant="body1" sx={{ fontWeight: 600 }}>
                                  {rel.target}
                                </Typography>
                              </Box>
                              {rel.description && (
                                <Typography variant="body2" color="text.secondary">
                                  {rel.description}
                                </Typography>
                              )}
                              {rel.confidence && (
                                <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                                  Confidence: {(rel.confidence * 100).toFixed(0)}%
                                </Typography>
                              )}
                            </CardContent>
                          </Card>
                        ))}
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              )}

              {/* No Results */}
              {(!discoveryResults.discovered_entities || discoveryResults.discovered_entities.length === 0) && 
               (!discoveryResults.discovered_relationships || discoveryResults.discovered_relationships.length === 0) && (
                <Grid item xs={12}>
                  <Alert severity="warning">
                    <Typography variant="body1" sx={{ fontWeight: 600 }}>
                      No entities or relationships detected
                    </Typography>
                    <Typography variant="body2">
                      Try adjusting your settings or using more descriptive text with clear entities and relationships.
                    </Typography>
                  </Alert>
                </Grid>
              )}
            </Grid>
          )}

          {/* Configuration Status */}
          <Card sx={{ mt: 3, backgroundColor: 'action.hover' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <SettingsIcon color="primary" />
                Current Configuration
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={3}>
                  <Typography variant="subtitle2" gutterBottom>Extraction Mode</Typography>
                  <Chip 
                    label={data?.knowledge_graph?.mode === 'thinking' ? 'Thinking Mode' : 'Direct Mode'} 
                    size="small" 
                    color={data?.knowledge_graph?.mode === 'thinking' ? 'primary' : 'default'}
                  />
                </Grid>
                <Grid item xs={12} md={3}>
                  <Typography variant="subtitle2" gutterBottom>Schema Strategy</Typography>
                  <Chip 
                    label={
                      data?.schema_mode === 'hybrid' ? 'Smart Hybrid' :
                      data?.schema_mode === 'dynamic' ? 'AI Discovery' : 'Predefined Only'
                    } 
                    size="small" 
                    color="secondary"
                  />
                </Grid>
                <Grid item xs={12} md={3}>
                  <Typography variant="subtitle2" gutterBottom>Entity Discovery</Typography>
                  <Chip 
                    label={data?.entity_discovery?.enabled ? 'Enabled' : 'Disabled'} 
                    size="small" 
                    color={data?.entity_discovery?.enabled ? 'success' : 'default'}
                  />
                </Grid>
                <Grid item xs={12} md={3}>
                  <Typography variant="subtitle2" gutterBottom>Model</Typography>
                  <Typography variant="body2" color="text.secondary">
                    {actualModelConfig.model || 'Not configured'}
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Box>
      )}
      
      {tabValue === 6 && (
        <Box>
          <KnowledgeGraphAntiSiloSettings 
            data={data}
            onChange={onChange}
            onShowSuccess={onShowSuccess}
          />
        </Box>
      )}

      {tabValue === 7 && (
        <Box>
          <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>
            Extraction Controls
          </Typography>
          
          <Alert severity="info" sx={{ mb: 3 }}>
            <Typography variant="body2">
              <strong>Advanced Extraction Settings:</strong> Fine-tune how entities and relationships are extracted, 
              processed, and linked. These settings control the sophisticated knowledge graph processing pipeline.
            </Typography>
          </Alert>

          {/* Search/Filter Box */}
          <Box sx={{ mb: 3 }}>
            <TextField
              fullWidth
              placeholder="Search extraction settings..."
              variant="outlined"
              size="small"
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon />
                  </InputAdornment>
                ),
              }}
              sx={{ maxWidth: 400 }}
            />
          </Box>

          <Grid container spacing={3}>
            {/* Entity Processing Controls */}
            <Grid item xs={12}>
              <Card variant="outlined">
                <CardHeader 
                  title={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <EntityIcon color="primary" />
                      Entity Processing Controls
                    </Box>
                  }
                  subheader="Configure how entities are identified, consolidated, and processed"
                />
                <CardContent>
                  <Grid container spacing={3}>
                    {/* Entity Consolidation */}
                    <Grid item xs={12} md={6}>
                      <Box>
                        <FormControlLabel 
                          control={
                            <Switch
                              checked={data?.extraction?.enable_entity_consolidation ?? true}
                              onChange={(e) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  enable_entity_consolidation: e.target.checked 
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                            />
                          }
                          label="Enable Entity Consolidation"
                        />
                        <Typography variant="caption" color="text.secondary" display="block">
                          Merge similar entities across document chunks
                        </Typography>
                        
                        {data?.extraction?.enable_entity_consolidation && (
                          <Box sx={{ mt: 2 }}>
                            <Typography gutterBottom sx={{ fontWeight: 600 }}>
                              Consolidation Threshold: {(data?.extraction?.entity_consolidation_threshold ?? 0.85).toFixed(2)}
                            </Typography>
                            <Slider
                              value={data?.extraction?.entity_consolidation_threshold ?? 0.85}
                              onChange={(_, value) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  entity_consolidation_threshold: value as number 
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                              min={0.1}
                              max={1}
                              step={0.05}
                              valueLabelDisplay="auto"
                              sx={{ mb: 1 }}
                            />
                          </Box>
                        )}
                      </Box>
                    </Grid>

                    {/* Entity Deduplication */}
                    <Grid item xs={12} md={6}>
                      <Box>
                        <FormControlLabel 
                          control={
                            <Switch
                              checked={data?.extraction?.enable_entity_deduplication ?? true}
                              onChange={(e) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  enable_entity_deduplication: e.target.checked 
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                            />
                          }
                          label="Enable Entity Deduplication"
                        />
                        <Typography variant="caption" color="text.secondary" display="block">
                          Remove duplicate entities within the same document
                        </Typography>
                      </Box>
                    </Grid>

                    {/* Alias Detection */}
                    <Grid item xs={12} md={6}>
                      <Box>
                        <FormControlLabel 
                          control={
                            <Switch
                              checked={data?.extraction?.enable_alias_detection ?? false}
                              onChange={(e) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  enable_alias_detection: e.target.checked 
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                            />
                          }
                          label="Enable Alias Detection"
                        />
                        <Typography variant="caption" color="text.secondary" display="block">
                          Detect alternative names for the same entity
                        </Typography>

                        {data?.extraction?.enable_alias_detection && (
                          <Box sx={{ mt: 2 }}>
                            <Typography gutterBottom sx={{ fontWeight: 600 }}>
                              Alias Detection Threshold: {(data?.extraction?.alias_detection_threshold ?? 0.9).toFixed(2)}
                            </Typography>
                            <Slider
                              value={data?.extraction?.alias_detection_threshold ?? 0.9}
                              onChange={(_, value) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  alias_detection_threshold: value as number 
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                              min={0.1}
                              max={1}
                              step={0.05}
                              valueLabelDisplay="auto"
                              sx={{ mb: 1 }}
                            />
                          </Box>
                        )}
                      </Box>
                    </Grid>

                    {/* Fuzzy Matching */}
                    <Grid item xs={12} md={6}>
                      <Box>
                        <FormControlLabel 
                          control={
                            <Switch
                              checked={data?.extraction?.enable_fuzzy_matching ?? false}
                              onChange={(e) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  enable_fuzzy_matching: e.target.checked 
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                            />
                          }
                          label="Enable Fuzzy Matching"
                        />
                        <Typography variant="caption" color="text.secondary" display="block">
                          Match entities with slight spelling variations
                        </Typography>

                        {data?.extraction?.enable_fuzzy_matching && (
                          <Box sx={{ mt: 2 }}>
                            <Typography gutterBottom sx={{ fontWeight: 600 }}>
                              Fuzzy Matching Threshold: {(data?.extraction?.fuzzy_matching_threshold ?? 0.85).toFixed(2)}
                            </Typography>
                            <Slider
                              value={data?.extraction?.fuzzy_matching_threshold ?? 0.85}
                              onChange={(_, value) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  fuzzy_matching_threshold: value as number 
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                              min={0.1}
                              max={1}
                              step={0.05}
                              valueLabelDisplay="auto"
                              sx={{ mb: 1 }}
                            />
                          </Box>
                        )}
                      </Box>
                    </Grid>

                    {/* Hub Entities */}
                    <Grid item xs={12} md={6}>
                      <Box>
                        <FormControlLabel 
                          control={
                            <Switch
                              checked={data?.extraction?.enable_hub_entities ?? false}
                              onChange={(e) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  enable_hub_entities: e.target.checked 
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                            />
                          }
                          label="Enable Hub Entity Detection"
                        />
                        <Typography variant="caption" color="text.secondary" display="block">
                          Identify central entities with many connections
                        </Typography>

                        {data?.extraction?.enable_hub_entities && (
                          <Box sx={{ mt: 2 }}>
                            <TextField
                              label="Hub Entity Threshold"
                              type="number"
                              value={data?.extraction?.hub_entity_threshold ?? 3}
                              onChange={(e) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  hub_entity_threshold: parseInt(e.target.value) || 3
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                              size="small"
                              helperText="Minimum connections to be considered a hub"
                            />
                          </Box>
                        )}
                      </Box>
                    </Grid>

                    {/* Entity Confidence */}
                    <Grid item xs={12} md={6}>
                      <Box>
                        <Typography gutterBottom sx={{ fontWeight: 600 }}>
                          Minimum Entity Confidence: {(data?.extraction?.min_entity_confidence ?? 0.5).toFixed(2)}
                        </Typography>
                        <Slider
                          value={data?.extraction?.min_entity_confidence ?? 0.5}
                          onChange={(_, value) => {
                            const updatedExtraction = { 
                              ...data?.extraction, 
                              min_entity_confidence: value as number 
                            };
                            onChange('extraction', updatedExtraction);
                          }}
                          min={0.1}
                          max={1}
                          step={0.05}
                          valueLabelDisplay="auto"
                          sx={{ mb: 1 }}
                        />
                        <Typography variant="caption" color="text.secondary">
                          Filter out low-confidence entity extractions
                        </Typography>
                      </Box>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>

            {/* Relationship Processing Controls */}
            <Grid item xs={12}>
              <Card variant="outlined">
                <CardHeader 
                  title={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <RelationshipIcon color="secondary" />
                      Relationship Processing Controls
                    </Box>
                  }
                  subheader="Configure how relationships are identified, validated, and enhanced"
                />
                <CardContent>
                  <Grid container spacing={3}>
                    {/* Relationship Deduplication */}
                    <Grid item xs={12} md={6}>
                      <Box>
                        <FormControlLabel 
                          control={
                            <Switch
                              checked={data?.extraction?.enable_relationship_deduplication ?? true}
                              onChange={(e) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  enable_relationship_deduplication: e.target.checked 
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                            />
                          }
                          label="Enable Relationship Deduplication"
                        />
                        <Typography variant="caption" color="text.secondary" display="block">
                          Remove duplicate relationships between same entities
                        </Typography>
                      </Box>
                    </Grid>

                    {/* Multi-chunk Relationships */}
                    <Grid item xs={12} md={6}>
                      <Box>
                        <FormControlLabel 
                          control={
                            <Switch
                              checked={data?.extraction?.enable_multi_chunk_relationships ?? false}
                              onChange={(e) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  enable_multi_chunk_relationships: e.target.checked 
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                            />
                          }
                          label="Enable Multi-Chunk Relationships"
                        />
                        <Typography variant="caption" color="text.secondary" display="block">
                          Detect relationships spanning across document chunks
                        </Typography>
                      </Box>
                    </Grid>

                    {/* Synthetic Relationships */}
                    <Grid item xs={12} md={6}>
                      <Box>
                        <FormControlLabel 
                          control={
                            <Switch
                              checked={data?.extraction?.enable_synthetic_relationships ?? false}
                              onChange={(e) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  enable_synthetic_relationships: e.target.checked 
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                            />
                          }
                          label="Enable Synthetic Relationships"
                        />
                        <Typography variant="caption" color="text.secondary" display="block">
                          Infer implicit relationships based on patterns
                        </Typography>

                        {data?.extraction?.enable_synthetic_relationships && (
                          <Box sx={{ mt: 2 }}>
                            <Typography gutterBottom sx={{ fontWeight: 600 }}>
                              Synthetic Confidence: {(data?.extraction?.synthetic_relationship_confidence ?? 0.5).toFixed(2)}
                            </Typography>
                            <Slider
                              value={data?.extraction?.synthetic_relationship_confidence ?? 0.5}
                              onChange={(_, value) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  synthetic_relationship_confidence: value as number 
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                              min={0.1}
                              max={1}
                              step={0.05}
                              valueLabelDisplay="auto"
                              sx={{ mb: 1 }}
                            />
                          </Box>
                        )}
                      </Box>
                    </Grid>

                    {/* Relationship Confidence */}
                    <Grid item xs={12} md={6}>
                      <Box>
                        <Typography gutterBottom sx={{ fontWeight: 600 }}>
                          Minimum Relationship Confidence: {(data?.extraction?.min_relationship_confidence ?? 0.6).toFixed(2)}
                        </Typography>
                        <Slider
                          value={data?.extraction?.min_relationship_confidence ?? 0.6}
                          onChange={(_, value) => {
                            const updatedExtraction = { 
                              ...data?.extraction, 
                              min_relationship_confidence: value as number 
                            };
                            onChange('extraction', updatedExtraction);
                          }}
                          min={0.1}
                          max={1}
                          step={0.05}
                          valueLabelDisplay="auto"
                          sx={{ mb: 1 }}
                        />
                        <Typography variant="caption" color="text.secondary">
                          Filter out low-confidence relationship extractions
                        </Typography>
                      </Box>
                    </Grid>

                    {/* Relationship Quality */}
                    <Grid item xs={12} md={6}>
                      <Box>
                        <Typography gutterBottom sx={{ fontWeight: 600 }}>
                          Relationship Quality Threshold: {(data?.extraction?.relationship_quality_threshold ?? 0.7).toFixed(2)}
                        </Typography>
                        <Slider
                          value={data?.extraction?.relationship_quality_threshold ?? 0.7}
                          onChange={(_, value) => {
                            const updatedExtraction = { 
                              ...data?.extraction, 
                              relationship_quality_threshold: value as number 
                            };
                            onChange('extraction', updatedExtraction);
                          }}
                          min={0.1}
                          max={1}
                          step={0.05}
                          valueLabelDisplay="auto"
                          sx={{ mb: 1 }}
                        />
                        <Typography variant="caption" color="text.secondary">
                          Overall quality score threshold for relationships
                        </Typography>
                      </Box>
                    </Grid>

                    {/* Pattern Relationships */}
                    <Grid item xs={12} md={6}>
                      <Box>
                        <FormControlLabel 
                          control={
                            <Switch
                              checked={data?.extraction?.prioritize_pattern_relationships ?? true}
                              onChange={(e) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  prioritize_pattern_relationships: e.target.checked 
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                            />
                          }
                          label="Prioritize Pattern Relationships"
                        />
                        <Typography variant="caption" color="text.secondary" display="block">
                          Give higher weight to relationships matching known patterns
                        </Typography>
                      </Box>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>

            {/* Advanced Linking Features */}
            <Grid item xs={12}>
              <Card variant="outlined">
                <CardHeader 
                  title={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <TemporalIcon color="info" />
                      Advanced Linking Features
                    </Box>
                  }
                  subheader="Enable sophisticated linking capabilities across time, space, and semantics"
                />
                <CardContent>
                  <Grid container spacing={3}>
                    {/* Temporal Linking */}
                    <Grid item xs={12} md={6}>
                      <Box>
                        <FormControlLabel 
                          control={
                            <Switch
                              checked={data?.extraction?.enable_temporal_linking ?? false}
                              onChange={(e) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  enable_temporal_linking: e.target.checked 
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                            />
                          }
                          label="Enable Temporal Linking"
                        />
                        <Typography variant="caption" color="text.secondary" display="block">
                          Link entities based on temporal proximity
                        </Typography>

                        {data?.extraction?.enable_temporal_linking && (
                          <Box sx={{ mt: 2 }}>
                            <TextField
                              label="Temporal Window (days)"
                              type="number"
                              value={data?.extraction?.temporal_linking_window ?? 7}
                              onChange={(e) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  temporal_linking_window: parseInt(e.target.value) || 7
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                              size="small"
                              helperText="Days within which events are considered related"
                            />
                          </Box>
                        )}
                      </Box>
                    </Grid>

                    {/* Geographic Linking */}
                    <Grid item xs={12} md={6}>
                      <Box>
                        <FormControlLabel 
                          control={
                            <Switch
                              checked={data?.extraction?.enable_geographic_linking ?? false}
                              onChange={(e) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  enable_geographic_linking: e.target.checked 
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                            />
                          }
                          label="Enable Geographic Linking"
                        />
                        <Typography variant="caption" color="text.secondary" display="block">
                          Link entities based on geographic proximity
                        </Typography>

                        {data?.extraction?.enable_geographic_linking && (
                          <Box sx={{ mt: 2 }}>
                            <Typography gutterBottom sx={{ fontWeight: 600 }}>
                              Geographic Threshold: {(data?.extraction?.geographic_linking_threshold ?? 0.7).toFixed(2)}
                            </Typography>
                            <Slider
                              value={data?.extraction?.geographic_linking_threshold ?? 0.7}
                              onChange={(_, value) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  geographic_linking_threshold: value as number 
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                              min={0.1}
                              max={1}
                              step={0.05}
                              valueLabelDisplay="auto"
                              sx={{ mb: 1 }}
                            />
                          </Box>
                        )}
                      </Box>
                    </Grid>

                    {/* Semantic Clustering */}
                    <Grid item xs={12} md={6}>
                      <Box>
                        <FormControlLabel 
                          control={
                            <Switch
                              checked={data?.extraction?.enable_semantic_clustering ?? false}
                              onChange={(e) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  enable_semantic_clustering: e.target.checked 
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                            />
                          }
                          label="Enable Semantic Clustering"
                        />
                        <Typography variant="caption" color="text.secondary" display="block">
                          Group semantically similar entities and relationships
                        </Typography>

                        {data?.extraction?.enable_semantic_clustering && (
                          <Box sx={{ mt: 2 }}>
                            <Typography gutterBottom sx={{ fontWeight: 600 }}>
                              Clustering Similarity: {(data?.extraction?.clustering_similarity_threshold ?? 0.8).toFixed(2)}
                            </Typography>
                            <Slider
                              value={data?.extraction?.clustering_similarity_threshold ?? 0.8}
                              onChange={(_, value) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  clustering_similarity_threshold: value as number 
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                              min={0.1}
                              max={1}
                              step={0.05}
                              valueLabelDisplay="auto"
                              sx={{ mb: 1 }}
                            />
                          </Box>
                        )}
                      </Box>
                    </Grid>

                    {/* Contextual Linking */}
                    <Grid item xs={12} md={6}>
                      <Box>
                        <FormControlLabel 
                          control={
                            <Switch
                              checked={data?.extraction?.enable_contextual_linking ?? false}
                              onChange={(e) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  enable_contextual_linking: e.target.checked 
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                            />
                          }
                          label="Enable Contextual Linking"
                        />
                        <Typography variant="caption" color="text.secondary" display="block">
                          Link entities appearing in similar contexts
                        </Typography>

                        {data?.extraction?.enable_contextual_linking && (
                          <Box sx={{ mt: 2 }}>
                            <Typography gutterBottom sx={{ fontWeight: 600 }}>
                              Contextual Threshold: {(data?.extraction?.contextual_linking_threshold ?? 0.7).toFixed(2)}
                            </Typography>
                            <Slider
                              value={data?.extraction?.contextual_linking_threshold ?? 0.7}
                              onChange={(_, value) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  contextual_linking_threshold: value as number 
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                              min={0.1}
                              max={1}
                              step={0.05}
                              valueLabelDisplay="auto"
                              sx={{ mb: 1 }}
                            />
                          </Box>
                        )}
                      </Box>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>

            {/* Network Analysis & Anti-Silo */}
            <Grid item xs={12}>
              <Card variant="outlined">
                <CardHeader 
                  title={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <NetworkIcon color="warning" />
                      Network Analysis & Anti-Silo
                    </Box>
                  }
                  subheader="Advanced network analysis and cross-document connectivity features"
                />
                <CardContent>
                  <Grid container spacing={3}>
                    {/* Connectivity Analysis */}
                    <Grid item xs={12} md={6}>
                      <Box>
                        <FormControlLabel 
                          control={
                            <Switch
                              checked={data?.extraction?.enable_connectivity_analysis ?? false}
                              onChange={(e) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  enable_connectivity_analysis: e.target.checked 
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                            />
                          }
                          label="Enable Connectivity Analysis"
                        />
                        <Typography variant="caption" color="text.secondary" display="block">
                          Analyze entity connectivity patterns and network structure
                        </Typography>

                        {data?.extraction?.enable_connectivity_analysis && (
                          <Box sx={{ mt: 2 }}>
                            <Typography gutterBottom sx={{ fontWeight: 600 }}>
                              Connectivity Threshold: {(data?.extraction?.connectivity_analysis_threshold ?? 0.4).toFixed(2)}
                            </Typography>
                            <Slider
                              value={data?.extraction?.connectivity_analysis_threshold ?? 0.4}
                              onChange={(_, value) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  connectivity_analysis_threshold: value as number 
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                              min={0.1}
                              max={1}
                              step={0.05}
                              valueLabelDisplay="auto"
                              sx={{ mb: 1 }}
                            />
                          </Box>
                        )}
                      </Box>
                    </Grid>

                    {/* Isolation Detection */}
                    <Grid item xs={12} md={6}>
                      <Box>
                        <FormControlLabel 
                          control={
                            <Switch
                              checked={data?.extraction?.enable_isolation_detection ?? false}
                              onChange={(e) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  enable_isolation_detection: e.target.checked 
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                            />
                          }
                          label="Enable Isolation Detection"
                        />
                        <Typography variant="caption" color="text.secondary" display="block">
                          Identify and connect isolated entities in the graph
                        </Typography>
                      </Box>
                    </Grid>

                    {/* Cross-Document Linking */}
                    <Grid item xs={12} md={6}>
                      <Box>
                        <FormControlLabel 
                          control={
                            <Switch
                              checked={data?.extraction?.enable_cross_document_linking ?? false}
                              onChange={(e) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  enable_cross_document_linking: e.target.checked 
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                            />
                          }
                          label="Enable Cross-Document Linking"
                        />
                        <Typography variant="caption" color="text.secondary" display="block">
                          Link related entities across different documents
                        </Typography>
                      </Box>
                    </Grid>

                    {/* Anti-Silo Mode */}
                    <Grid item xs={12} md={6}>
                      <Box>
                        <FormControlLabel 
                          control={
                            <Switch
                              checked={data?.extraction?.enable_anti_silo ?? true}
                              onChange={(e) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  enable_anti_silo: e.target.checked 
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                            />
                          }
                          label="Enable Anti-Silo Mode"
                        />
                        <Typography variant="caption" color="text.secondary" display="block">
                          Prevent knowledge silos by linking related concepts
                        </Typography>

                        {data?.extraction?.enable_anti_silo && (
                          <Box sx={{ mt: 2 }}>
                            <Typography gutterBottom sx={{ fontWeight: 600 }}>
                              Anti-Silo Similarity: {(data?.extraction?.anti_silo_similarity_threshold ?? 0.75).toFixed(2)}
                            </Typography>
                            <Slider
                              value={data?.extraction?.anti_silo_similarity_threshold ?? 0.75}
                              onChange={(_, value) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  anti_silo_similarity_threshold: value as number 
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                              min={0.1}
                              max={1}
                              step={0.05}
                              valueLabelDisplay="auto"
                              sx={{ mb: 1 }}
                            />
                          </Box>
                        )}
                      </Box>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>

            {/* Quality & Performance Controls */}
            <Grid item xs={12}>
              <Card variant="outlined">
                <CardHeader 
                  title={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <PerformanceIcon color="success" />
                      Quality & Performance Controls
                    </Box>
                  }
                  subheader="Fine-tune extraction quality and system performance"
                />
                <CardContent>
                  <Grid container spacing={3}>
                    {/* LLM Enhancement */}
                    <Grid item xs={12} md={6}>
                      <Box>
                        <FormControlLabel 
                          control={
                            <Switch
                              checked={data?.extraction?.enable_llm_enhancement ?? false}
                              onChange={(e) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  enable_llm_enhancement: e.target.checked 
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                            />
                          }
                          label="Enable LLM Enhancement"
                        />
                        <Typography variant="caption" color="text.secondary" display="block">
                          Use LLM to enhance and validate extracted relationships
                        </Typography>

                        {data?.extraction?.enable_llm_enhancement && (
                          <Box sx={{ mt: 2 }}>
                            <Typography gutterBottom sx={{ fontWeight: 600 }}>
                              LLM Confidence Threshold: {(data?.extraction?.llm_confidence_threshold ?? 0.8).toFixed(2)}
                            </Typography>
                            <Slider
                              value={data?.extraction?.llm_confidence_threshold ?? 0.8}
                              onChange={(_, value) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  llm_confidence_threshold: value as number 
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                              min={0.1}
                              max={1}
                              step={0.05}
                              valueLabelDisplay="auto"
                              sx={{ mb: 1 }}
                            />
                          </Box>
                        )}
                      </Box>
                    </Grid>

                    {/* Graph Enrichment */}
                    <Grid item xs={12} md={6}>
                      <Box>
                        <FormControlLabel 
                          control={
                            <Switch
                              checked={data?.extraction?.enable_graph_enrichment ?? false}
                              onChange={(e) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  enable_graph_enrichment: e.target.checked 
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                            />
                          }
                          label="Enable Graph Enrichment"
                        />
                        <Typography variant="caption" color="text.secondary" display="block">
                          Enrich the graph with additional inferred connections
                        </Typography>

                        {data?.extraction?.enable_graph_enrichment && (
                          <Box sx={{ mt: 2 }}>
                            <TextField
                              label="Enrichment Depth"
                              type="number"
                              value={data?.extraction?.graph_enrichment_depth ?? 3}
                              onChange={(e) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  graph_enrichment_depth: parseInt(e.target.value) || 3
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                              size="small"
                              helperText="How many hops to traverse for enrichment"
                            />
                          </Box>
                        )}
                      </Box>
                    </Grid>

                    {/* Co-occurrence Analysis */}
                    <Grid item xs={12} md={6}>
                      <Box>
                        <FormControlLabel 
                          control={
                            <Switch
                              checked={data?.extraction?.enable_cooccurrence_analysis ?? false}
                              onChange={(e) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  enable_cooccurrence_analysis: e.target.checked 
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                            />
                          }
                          label="Enable Co-occurrence Analysis"
                        />
                        <Typography variant="caption" color="text.secondary" display="block">
                          Analyze which entities frequently appear together
                        </Typography>
                      </Box>
                    </Grid>

                    {/* Type-based Clustering */}
                    <Grid item xs={12} md={6}>
                      <Box>
                        <FormControlLabel 
                          control={
                            <Switch
                              checked={data?.extraction?.enable_type_based_clustering ?? false}
                              onChange={(e) => {
                                const updatedExtraction = { 
                                  ...data?.extraction, 
                                  enable_type_based_clustering: e.target.checked 
                                };
                                onChange('extraction', updatedExtraction);
                              }}
                            />
                          }
                          label="Enable Type-based Clustering"
                        />
                        <Typography variant="caption" color="text.secondary" display="block">
                          Group entities by type for better organization
                        </Typography>
                      </Box>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Configuration Summary */}
          <Card sx={{ mt: 3, backgroundColor: 'action.hover' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <SettingsIcon color="primary" />
                Extraction Configuration Summary
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={3}>
                  <Typography variant="subtitle2" gutterBottom>Entity Processing</Typography>
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    <Chip 
                      label={data?.extraction?.enable_entity_consolidation ? 'Consolidation ✓' : 'Consolidation ✗'} 
                      size="small" 
                      color={data?.extraction?.enable_entity_consolidation ? 'success' : 'default'}
                    />
                    <Chip 
                      label={data?.extraction?.enable_entity_deduplication ? 'Deduplication ✓' : 'Deduplication ✗'} 
                      size="small" 
                      color={data?.extraction?.enable_entity_deduplication ? 'success' : 'default'}
                    />
                  </Box>
                </Grid>
                <Grid item xs={12} md={3}>
                  <Typography variant="subtitle2" gutterBottom>Advanced Linking</Typography>
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    <Chip 
                      label={data?.extraction?.enable_temporal_linking ? 'Temporal ✓' : 'Temporal ✗'} 
                      size="small" 
                      color={data?.extraction?.enable_temporal_linking ? 'info' : 'default'}
                    />
                    <Chip 
                      label={data?.extraction?.enable_geographic_linking ? 'Geographic ✓' : 'Geographic ✗'} 
                      size="small" 
                      color={data?.extraction?.enable_geographic_linking ? 'info' : 'default'}
                    />
                  </Box>
                </Grid>
                <Grid item xs={12} md={3}>
                  <Typography variant="subtitle2" gutterBottom>Network Analysis</Typography>
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    <Chip 
                      label={data?.extraction?.enable_anti_silo ? 'Anti-Silo ✓' : 'Anti-Silo ✗'} 
                      size="small" 
                      color={data?.extraction?.enable_anti_silo ? 'warning' : 'default'}
                    />
                    <Chip 
                      label={data?.extraction?.enable_connectivity_analysis ? 'Connectivity ✓' : 'Connectivity ✗'} 
                      size="small" 
                      color={data?.extraction?.enable_connectivity_analysis ? 'warning' : 'default'}
                    />
                  </Box>
                </Grid>
                <Grid item xs={12} md={3}>
                  <Typography variant="subtitle2" gutterBottom>Quality Controls</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Entity: {(data?.extraction?.min_entity_confidence ?? 0.5).toFixed(2)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Relationship: {(data?.extraction?.min_relationship_confidence ?? 0.6).toFixed(2)}
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Box>
      )}
    </Box>
  );
};

export default KnowledgeGraphSettings;
