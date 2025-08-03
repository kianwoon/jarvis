import React, { useState, useEffect } from 'react';
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
  Paper,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Badge,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  MenuItem,
  Select,
  InputLabel
} from '@mui/material';
import {
  Settings as SettingsIcon,
  Psychology as BrainIcon,
  CheckCircle as CheckCircleIcon,
  Cancel as XCircleIcon,
  Refresh as RefreshCwIcon,
  TrendingUp as TrendingUpIcon,
  Warning as AlertTriangleIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
  Code as CodeIcon,
  ExpandMore as ExpandMoreIcon,
  Link as LinkIcon,
  Hub as HubIcon,
  NetworkCheck as NetworkCheckIcon,
  MergeType as MergeTypeIcon,
  PsychologyAlt as PsychologyAltIcon,
  AutoGraph as AutoGraphIcon,
  Timeline as TimelineIcon,
  Map as MapIcon,
  Cloud as CloudIcon,
  Security as SecurityIcon,
  Speed as SpeedIcon,
  Storage as StorageIcon,
  Analytics as AnalyticsIcon,
  Insights as InsightsIcon,
  TrendingDown as TrendingDownIcon,
  WarningAmber as WarningAmberIcon,
  Info as InfoIcon
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';

interface AntiSiloConfig {
  enable_anti_silo: boolean;
  anti_silo_similarity_threshold: number;
  anti_silo_type_boost: number;
  enable_cooccurrence_analysis: boolean;
  enable_type_based_clustering: boolean;
  enable_hub_entities: boolean;
  hub_entity_threshold: number;
  enable_semantic_clustering: boolean;
  clustering_similarity_threshold: number;
  enable_document_bridge_relationships: boolean;
  bridge_relationship_confidence: number;
  enable_temporal_linking: boolean;
  temporal_linking_window: number;
  enable_contextual_linking: boolean;
  contextual_linking_threshold: number;
  enable_fuzzy_matching: boolean;
  fuzzy_matching_threshold: number;
  enable_alias_detection: boolean;
  alias_detection_threshold: number;
  enable_abbreviation_matching: boolean;
  abbreviation_matching_threshold: number;
  enable_synonym_detection: boolean;
  synonym_detection_threshold: number;
  enable_hierarchical_linking: boolean;
  hierarchical_linking_depth: number;
  enable_geographic_linking: boolean;
  geographic_linking_threshold: number;
  enable_temporal_coherence: boolean;
  temporal_coherence_threshold: number;
  enable_semantic_bridge_entities: boolean;
  semantic_bridge_threshold: number;
  enable_cross_reference_analysis: boolean;
  cross_reference_threshold: number;
  enable_relationship_propagation: boolean;
  relationship_propagation_depth: number;
  enable_entity_consolidation: boolean;
  entity_consolidation_threshold: number;
  enable_synthetic_relationships: boolean;
  synthetic_relationship_confidence: number;
  enable_graph_enrichment: boolean;
  graph_enrichment_depth: number;
  enable_connectivity_analysis: boolean;
  connectivity_analysis_threshold: number;
  enable_isolation_detection: boolean;
  isolation_detection_threshold: number;
  enable_nuclear_option: boolean;
  nuclear_similarity_threshold: number;
}

interface KnowledgeGraphAntiSiloSettingsProps {
  data: any;
  onChange: (field: string, value: any) => void;
  onShowSuccess?: (message?: string) => void;
}

const KnowledgeGraphAntiSiloSettings: React.FC<KnowledgeGraphAntiSiloSettingsProps> = ({ 
  data, 
  onChange, 
  onShowSuccess 
}) => {
  const { enqueueSnackbar } = useSnackbar();
  const [tabValue, setTabValue] = useState(0);
  const [testResults, setTestResults] = useState<any>(null);
  const [isTesting, setIsTesting] = useState(false);

  const antiSiloConfig: AntiSiloConfig = {
    enable_anti_silo: data?.knowledge_graph?.extraction?.enable_anti_silo ?? true,
    anti_silo_similarity_threshold: data?.knowledge_graph?.extraction?.anti_silo_similarity_threshold ?? 0.75,
    anti_silo_type_boost: data?.knowledge_graph?.extraction?.anti_silo_type_boost ?? 1.2,
    enable_cooccurrence_analysis: data?.knowledge_graph?.extraction?.enable_cooccurrence_analysis ?? true,
    enable_type_based_clustering: data?.knowledge_graph?.extraction?.enable_type_based_clustering ?? true,
    enable_hub_entities: data?.knowledge_graph?.extraction?.enable_hub_entities ?? true,
    hub_entity_threshold: data?.knowledge_graph?.extraction?.hub_entity_threshold ?? 3,
    enable_semantic_clustering: data?.knowledge_graph?.extraction?.enable_semantic_clustering ?? true,
    clustering_similarity_threshold: data?.knowledge_graph?.extraction?.clustering_similarity_threshold ?? 0.8,
    enable_document_bridge_relationships: data?.knowledge_graph?.extraction?.enable_document_bridge_relationships ?? true,
    bridge_relationship_confidence: data?.knowledge_graph?.extraction?.bridge_relationship_confidence ?? 0.6,
    enable_temporal_linking: data?.knowledge_graph?.extraction?.enable_temporal_linking ?? true,
    temporal_linking_window: data?.knowledge_graph?.extraction?.temporal_linking_window ?? 7,
    enable_contextual_linking: data?.knowledge_graph?.extraction?.enable_contextual_linking ?? true,
    contextual_linking_threshold: data?.knowledge_graph?.extraction?.contextual_linking_threshold ?? 0.7,
    enable_fuzzy_matching: data?.knowledge_graph?.extraction?.enable_fuzzy_matching ?? true,
    fuzzy_matching_threshold: data?.knowledge_graph?.extraction?.fuzzy_matching_threshold ?? 0.85,
    enable_alias_detection: data?.knowledge_graph?.extraction?.enable_alias_detection ?? true,
    alias_detection_threshold: data?.knowledge_graph?.extraction?.alias_detection_threshold ?? 0.9,
    enable_abbreviation_matching: data?.knowledge_graph?.extraction?.enable_abbreviation_matching ?? true,
    abbreviation_matching_threshold: data?.knowledge_graph?.extraction?.abbreviation_matching_threshold ?? 0.8,
    enable_synonym_detection: data?.knowledge_graph?.extraction?.enable_synonym_detection ?? true,
    synonym_detection_threshold: data?.knowledge_graph?.extraction?.synonym_detection_threshold ?? 0.75,
    enable_hierarchical_linking: data?.knowledge_graph?.extraction?.enable_hierarchical_linking ?? true,
    hierarchical_linking_depth: data?.knowledge_graph?.extraction?.hierarchical_linking_depth ?? 2,
    enable_geographic_linking: data?.knowledge_graph?.extraction?.enable_geographic_linking ?? true,
    geographic_linking_threshold: data?.knowledge_graph?.extraction?.geographic_linking_threshold ?? 0.7,
    enable_temporal_coherence: data?.knowledge_graph?.extraction?.enable_temporal_coherence ?? true,
    temporal_coherence_threshold: data?.knowledge_graph?.extraction?.temporal_coherence_threshold ?? 0.6,
    enable_semantic_bridge_entities: data?.knowledge_graph?.extraction?.enable_semantic_bridge_entities ?? true,
    semantic_bridge_threshold: data?.knowledge_graph?.extraction?.semantic_bridge_threshold ?? 0.65,
    enable_cross_reference_analysis: data?.knowledge_graph?.extraction?.enable_cross_reference_analysis ?? true,
    cross_reference_threshold: data?.knowledge_graph?.extraction?.cross_reference_threshold ?? 0.7,
    enable_relationship_propagation: data?.knowledge_graph?.extraction?.enable_relationship_propagation ?? true,
    relationship_propagation_depth: data?.knowledge_graph?.extraction?.relationship_propagation_depth ?? 2,
    enable_entity_consolidation: data?.knowledge_graph?.extraction?.enable_entity_consolidation ?? true,
    entity_consolidation_threshold: data?.knowledge_graph?.extraction?.entity_consolidation_threshold ?? 0.85,
    enable_synthetic_relationships: data?.knowledge_graph?.extraction?.enable_synthetic_relationships ?? true,
    synthetic_relationship_confidence: data?.knowledge_graph?.extraction?.synthetic_relationship_confidence ?? 0.5,
    enable_graph_enrichment: data?.knowledge_graph?.extraction?.enable_graph_enrichment ?? true,
    graph_enrichment_depth: data?.knowledge_graph?.extraction?.graph_enrichment_depth ?? 3,
    enable_connectivity_analysis: data?.knowledge_graph?.extraction?.enable_connectivity_analysis ?? true,
    connectivity_analysis_threshold: data?.knowledge_graph?.extraction?.connectivity_analysis_threshold ?? 0.4,
    enable_isolation_detection: data?.knowledge_graph?.extraction?.enable_isolation_detection ?? true,
    isolation_detection_threshold: data?.knowledge_graph?.extraction?.isolation_detection_threshold ?? 1,
    enable_nuclear_option: data?.knowledge_graph?.extraction?.enable_nuclear_option ?? false,
    nuclear_similarity_threshold: data?.knowledge_graph?.extraction?.nuclear_similarity_threshold ?? 0.1,
    ...data?.knowledge_graph?.extraction
  };

  const handleConfigChange = (field: string, value: any) => {
    const newConfig = {
      ...data?.knowledge_graph,
      extraction: {
        ...data?.knowledge_graph?.extraction,
        [field]: value
      }
    };
    onChange('knowledge_graph', newConfig);
  };

  const handleBulkConfigChange = (config: Partial<AntiSiloConfig>) => {
    const newConfig = {
      ...data?.knowledge_graph,
      extraction: {
        ...data?.knowledge_graph?.extraction,
        ...config
      }
    };
    onChange('knowledge_graph', newConfig);
  };

  const testAntiSiloConfiguration = async () => {
    setIsTesting(true);
    try {
      const response = await fetch('/api/v1/knowledge-graph/test-anti-silo', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(antiSiloConfig)
      });

      const results = await response.json();
      setTestResults(results);
      
      if (results.success) {
        enqueueSnackbar(`Anti-silo test completed: ${results.connectivity_score}% connectivity`, { 
          variant: "success" 
        });
      } else {
        enqueueSnackbar("Anti-silo test failed", { 
          variant: "error" 
        });
      }
    } catch (error) {
      enqueueSnackbar("Failed to test anti-silo configuration", { 
        variant: "error" 
      });
    } finally {
      setIsTesting(false);
    }
  };

  const presetConfigurations = {
    aggressive: {
      enable_anti_silo: true,
      anti_silo_similarity_threshold: 0.6,
      anti_silo_type_boost: 1.5,
      enable_cooccurrence_analysis: true,
      enable_type_based_clustering: true,
      enable_hub_entities: true,
      hub_entity_threshold: 2,
      enable_semantic_clustering: true,
      clustering_similarity_threshold: 0.7,
      enable_document_bridge_relationships: true,
      bridge_relationship_confidence: 0.5,
      enable_synthetic_relationships: true,
      synthetic_relationship_confidence: 0.4,
      enable_graph_enrichment: true,
      graph_enrichment_depth: 4,
      enable_relationship_propagation: true,
      relationship_propagation_depth: 3,
      enable_nuclear_option: true,
      nuclear_similarity_threshold: 0.1
    },
    balanced: {
      enable_anti_silo: true,
      anti_silo_similarity_threshold: 0.75,
      anti_silo_type_boost: 1.2,
      enable_cooccurrence_analysis: true,
      enable_type_based_clustering: true,
      enable_hub_entities: true,
      hub_entity_threshold: 3,
      enable_semantic_clustering: true,
      clustering_similarity_threshold: 0.8,
      enable_document_bridge_relationships: true,
      bridge_relationship_confidence: 0.6,
      enable_synthetic_relationships: true,
      synthetic_relationship_confidence: 0.5,
      enable_graph_enrichment: true,
      graph_enrichment_depth: 3,
      enable_relationship_propagation: true,
      relationship_propagation_depth: 2
    },
    conservative: {
      enable_anti_silo: true,
      anti_silo_similarity_threshold: 0.85,
      anti_silo_type_boost: 1.1,
      enable_cooccurrence_analysis: true,
      enable_type_based_clustering: true,
      enable_hub_entities: true,
      hub_entity_threshold: 5,
      enable_semantic_clustering: true,
      clustering_similarity_threshold: 0.9,
      enable_document_bridge_relationships: true,
      bridge_relationship_confidence: 0.7,
      enable_synthetic_relationships: false,
      synthetic_relationship_confidence: 0.6,
      enable_graph_enrichment: true,
      graph_enrichment_depth: 2,
      enable_relationship_propagation: true,
      relationship_propagation_depth: 1
    },
    minimal: {
      enable_anti_silo: true,
      anti_silo_similarity_threshold: 0.9,
      anti_silo_type_boost: 1.0,
      enable_cooccurrence_analysis: false,
      enable_type_based_clustering: true,
      enable_hub_entities: false,
      hub_entity_threshold: 10,
      enable_semantic_clustering: false,
      clustering_similarity_threshold: 0.95,
      enable_document_bridge_relationships: true,
      bridge_relationship_confidence: 0.8,
      enable_synthetic_relationships: false,
      synthetic_relationship_confidence: 0.7,
      enable_graph_enrichment: false,
      graph_enrichment_depth: 1,
      enable_relationship_propagation: false,
      relationship_propagation_depth: 1
    }
  };

  const applyPreset = (preset: string) => {
    const config = presetConfigurations[preset as keyof typeof presetConfigurations];
    if (config) {
      handleBulkConfigChange(config);
      enqueueSnackbar(`Applied ${preset} anti-silo preset`, { 
        variant: "success" 
      });
    }
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const renderSliderControl = (label: string, field: keyof AntiSiloConfig, min: number, max: number, step: number, marks?: any[]) => (
    <Box sx={{ mb: 3 }}>
      <Typography gutterBottom>{label}</Typography>
      <Slider
        value={antiSiloConfig[field] as number}
        onChange={(_, value) => handleConfigChange(field, value)}
        min={min}
        max={max}
        step={step}
        marks={marks}
        valueLabelDisplay="auto"
        sx={{ mt: 1 }}
      />
    </Box>
  );

  const renderSwitchControl = (label: string, field: keyof AntiSiloConfig, description?: string) => (
    <Box sx={{ mb: 2 }}>
      <FormControlLabel
        control={
          <Switch
            checked={antiSiloConfig[field] as boolean}
            onChange={(e) => handleConfigChange(field, e.target.checked)}
          />
        }
        label={
          <Box>
            <Typography variant="body2">{label}</Typography>
            {description && (
              <Typography variant="caption" color="text.secondary">
                {description}
              </Typography>
            )}
          </Box>
        }
      />
    </Box>
  );

  return (
    <Box>
      <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>
        Anti-Silo Configuration
      </Typography>

      <Box sx={{ mb: 3 }}>
        <Alert severity="info" sx={{ mb: 2 }}>
          <Typography variant="body2">
            <strong>Anti-Silo Features:</strong> These settings help connect isolated entities across documents,
            creating a more cohesive knowledge graph with reduced silo nodes.
          </Typography>
        </Alert>

        <Box sx={{ display: 'flex', gap: 2, mb: 3, flexWrap: 'wrap' }}>
          <Button
            variant="outlined"
            onClick={() => applyPreset('aggressive')}
            startIcon={<TrendingUpIcon />}
          >
            Aggressive Linking
          </Button>
          <Button
            variant="outlined"
            onClick={() => applyPreset('balanced')}
            startIcon={<AutoGraphIcon />}
          >
            Balanced (Recommended)
          </Button>
          <Button
            variant="outlined"
            onClick={() => applyPreset('conservative')}
            startIcon={<TrendingDownIcon />}
          >
            Conservative
          </Button>
          <Button
            variant="outlined"
            onClick={() => applyPreset('minimal')}
            startIcon={<WarningAmberIcon />}
          >
            Minimal
          </Button>
          <Button
            variant="contained"
            onClick={testAntiSiloConfiguration}
            disabled={isTesting}
            startIcon={isTesting ? <CircularProgress size={16} /> : <NetworkCheckIcon />}
          >
            {isTesting ? 'Testing...' : 'Test Configuration'}
          </Button>
        </Box>
      </Box>

      <Tabs value={tabValue} onChange={handleTabChange} sx={{ mb: 3 }}>
        <Tab label="Core Anti-Silo" icon={<LinkIcon />} />
        <Tab label="Advanced Linking" icon={<MergeTypeIcon />} />
        <Tab label="Semantic Analysis" icon={<PsychologyAltIcon />} />
        <Tab label="Graph Optimization" icon={<AutoGraphIcon />} />
      </Tabs>

      {tabValue === 0 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardHeader title="Core Anti-Silo Features" />
              <CardContent>
                {renderSwitchControl(
                  "Enable Anti-Silo System",
                  "enable_anti_silo",
                  "Master switch for all anti-silo features"
                )}
                
                {renderSliderControl(
                  "Similarity Threshold",
                  "anti_silo_similarity_threshold",
                  0.1, 1.0, 0.05,
                  [
                    { value: 0.5, label: 'Low' },
                    { value: 0.75, label: 'Medium' },
                    { value: 0.9, label: 'High' }
                  ]
                )}
                
                {renderSliderControl(
                  "Type Boost Factor",
                  "anti_silo_type_boost",
                  1.0, 2.0, 0.1
                )}
                
                {renderSwitchControl(
                  "Enable Co-occurrence Analysis",
                  "enable_cooccurrence_analysis",
                  "Analyze entity co-occurrence patterns across documents"
                )}
                
                {renderSwitchControl(
                  "Enable Type-based Clustering",
                  "enable_type_based_clustering",
                  "Group similar entity types to reduce silos"
                )}
                
                {renderSwitchControl(
                  "Enable Hub Entities",
                  "enable_hub_entities",
                  "Create central hub nodes for better connectivity"
                )}
                
                {renderSliderControl(
                  "Hub Entity Threshold",
                  "hub_entity_threshold",
                  1, 10, 1
                )}
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card>
              <CardHeader title="Document Bridge Relationships" />
              <CardContent>
                {renderSwitchControl(
                  "Enable Document Bridge Relationships",
                  "enable_document_bridge_relationships",
                  "Create relationships between entities across documents"
                )}
                
                {renderSliderControl(
                  "Bridge Relationship Confidence",
                  "bridge_relationship_confidence",
                  0.1, 1.0, 0.05
                )}
                
                {renderSwitchControl(
                  "Enable Temporal Linking",
                  "enable_temporal_linking",
                  "Link entities based on temporal proximity"
                )}
                
                {renderSliderControl(
                  "Temporal Linking Window (days)",
                  "temporal_linking_window",
                  1, 30, 1
                )}
                
                {renderSwitchControl(
                  "Enable Contextual Linking",
                  "enable_contextual_linking",
                  "Link entities based on contextual similarity"
                )}
                
                {renderSliderControl(
                  "Contextual Linking Threshold",
                  "contextual_linking_threshold",
                  0.1, 1.0, 0.05
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {tabValue === 1 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardHeader title="Fuzzy Matching & Aliases" />
              <CardContent>
                {renderSwitchControl(
                  "Enable Fuzzy Matching",
                  "enable_fuzzy_matching",
                  "Allow approximate string matching for entity names"
                )}
                
                {renderSliderControl(
                  "Fuzzy Matching Threshold",
                  "fuzzy_matching_threshold",
                  0.5, 1.0, 0.05
                )}
                
                {renderSwitchControl(
                  "Enable Alias Detection",
                  "enable_alias_detection",
                  "Detect entity aliases and variations"
                )}
                
                {renderSliderControl(
                  "Alias Detection Threshold",
                  "alias_detection_threshold",
                  0.7, 1.0, 0.05
                )}
                
                {renderSwitchControl(
                  "Enable Abbreviation Matching",
                  "enable_abbreviation_matching",
                  "Match entity abbreviations and acronyms"
                )}
                
                {renderSliderControl(
                  "Abbreviation Matching Threshold",
                  "abbreviation_matching_threshold",
                  0.6, 1.0, 0.05
                )}
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card>
              <CardHeader title="Advanced Linking Strategies" />
              <CardContent>
                {renderSwitchControl(
                  "Enable Synonym Detection",
                  "enable_synonym_detection",
                  "Use semantic synonyms for entity matching"
                )}
                
                {renderSliderControl(
                  "Synonym Detection Threshold",
                  "synonym_detection_threshold",
                  0.5, 1.0, 0.05
                )}
                
                {renderSwitchControl(
                  "Enable Hierarchical Linking",
                  "enable_hierarchical_linking",
                  "Create hierarchical relationships between entities"
                )}
                
                {renderSliderControl(
                  "Hierarchical Linking Depth",
                  "hierarchical_linking_depth",
                  1, 5, 1
                )}
                
                {renderSwitchControl(
                  "Enable Geographic Linking",
                  "enable_geographic_linking",
                  "Link entities based on geographic proximity"
                )}
                
                {renderSliderControl(
                  "Geographic Linking Threshold",
                  "geographic_linking_threshold",
                  0.5, 1.0, 0.05
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {tabValue === 2 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardHeader title="Semantic Analysis" />
              <CardContent>
                {renderSwitchControl(
                  "Enable Semantic Clustering",
                  "enable_semantic_clustering",
                  "Use semantic similarity for entity clustering"
                )}
                
                {renderSliderControl(
                  "Clustering Similarity Threshold",
                  "clustering_similarity_threshold",
                  0.5, 1.0, 0.05
                )}
                
                {renderSwitchControl(
                  "Enable Semantic Bridge Entities",
                  "enable_semantic_bridge_entities",
                  "Create semantic bridge entities between concepts"
                )}
                
                {renderSliderControl(
                  "Semantic Bridge Threshold",
                  "semantic_bridge_threshold",
                  0.5, 1.0, 0.05
                )}
                
                {renderSwitchControl(
                  "Enable Cross-Reference Analysis",
                  "enable_cross_reference_analysis",
                  "Analyze cross-references between entities"
                )}
                
                {renderSliderControl(
                  "Cross-Reference Threshold",
                  "cross_reference_threshold",
                  0.5, 1.0, 0.05
                )}
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card>
              <CardHeader title="Relationship Management" />
              <CardContent>
                {renderSwitchControl(
                  "Enable Relationship Propagation",
                  "enable_relationship_propagation",
                  "Propagate relationships through the graph"
                )}
                
                {renderSliderControl(
                  "Relationship Propagation Depth",
                  "relationship_propagation_depth",
                  1, 5, 1
                )}
                
                {renderSwitchControl(
                  "Enable Entity Consolidation",
                  "enable_entity_consolidation",
                  "Consolidate similar entities to reduce redundancy"
                )}
                
                {renderSliderControl(
                  "Entity Consolidation Threshold",
                  "entity_consolidation_threshold",
                  0.7, 1.0, 0.05
                )}
                
                {renderSwitchControl(
                  "Enable Temporal Coherence",
                  "enable_temporal_coherence",
                  "Ensure temporal consistency in relationships"
                )}
                
                {renderSliderControl(
                  "Temporal Coherence Threshold",
                  "temporal_coherence_threshold",
                  0.3, 1.0, 0.05
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {tabValue === 3 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardHeader title="Graph Optimization" />
              <CardContent>
                {renderSwitchControl(
                  "Enable Graph Enrichment",
                  "enable_graph_enrichment",
                  "Enrich the graph with additional relationships"
                )}
                
                {renderSliderControl(
                  "Graph Enrichment Depth",
                  "graph_enrichment_depth",
                  1, 5, 1
                )}
                
                {renderSwitchControl(
                  "Enable Synthetic Relationships",
                  "enable_synthetic_relationships",
                  "Generate synthetic relationships for connectivity"
                )}
                
                {renderSliderControl(
                  "Synthetic Relationship Confidence",
                  "synthetic_relationship_confidence",
                  0.3, 1.0, 0.05
                )}
                
                {renderSwitchControl(
                  "Enable Connectivity Analysis",
                  "enable_connectivity_analysis",
                  "Analyze and improve graph connectivity"
                )}
                
                {renderSliderControl(
                  "Connectivity Analysis Threshold",
                  "connectivity_analysis_threshold",
                  0.1, 1.0, 0.05
                )}
                
                {renderSwitchControl(
                  "Enable Isolation Detection",
                  "enable_isolation_detection",
                  "Detect and report isolated nodes"
                )}
                
                {renderSliderControl(
                  "Isolation Detection Threshold",
                  "isolation_detection_threshold",
                  0.5, 5, 0.5
                )}
                
                <Divider sx={{ my: 2 }} />
                
                {renderSwitchControl(
                  "Enable Nuclear Option",
                  "enable_nuclear_option",
                  "⚠️ AGGRESSIVE: Force connect ALL isolated nodes with very low thresholds"
                )}
                
                {renderSliderControl(
                  "Nuclear Similarity Threshold",
                  "nuclear_similarity_threshold",
                  0.05, 0.5, 0.05,
                  [
                    { value: 0.05, label: 'Ultra Low' },
                    { value: 0.1, label: 'Very Low' },
                    { value: 0.2, label: 'Low' },
                    { value: 0.3, label: 'Medium' }
                  ]
                )}
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card>
              <CardHeader title="Test Results" />
              <CardContent>
                {testResults ? (
                  <Box>
                    <Typography variant="h6" gutterBottom>
                      Connectivity Score: {testResults.connectivity_score}%
                    </Typography>
                    
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="body2">
                        Isolated Nodes: {testResults.isolated_nodes}
                      </Typography>
                      <Typography variant="body2">
                        Total Nodes: {testResults.total_nodes}
                      </Typography>
                      <Typography variant="body2">
                        Average Degree: {testResults.average_degree}
                      </Typography>
                      <Typography variant="body2">
                        Connected Components: {testResults.connected_components}
                    </Typography>
                    </Box>
                    
                    <Alert severity={testResults.connectivity_score > 80 ? "success" : testResults.connectivity_score > 60 ? "warning" : "error"}>
                      {testResults.connectivity_score > 80 
                        ? "Excellent connectivity - minimal silo nodes"
                        : testResults.connectivity_score > 60
                        ? "Good connectivity - some isolated nodes detected"
                        : "Poor connectivity - significant silo node issues"
                      }
                    </Alert>
                  </Box>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    Run a test to see connectivity analysis results
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      <Box sx={{ mt: 3 }}>
        <Alert severity="info">
          <Typography variant="body2">
            <strong>Pro Tips:</strong>
          </Typography>
          <Typography variant="body2" sx={{ mt: 1 }}>
            • Start with "Balanced" preset for most use cases
            <br />
            • Use "Aggressive" for documents with many similar entities
            <br />
            • Monitor connectivity scores to optimize settings
            <br />
            • Test configuration changes before applying to production
          </Typography>
        </Alert>
      </Box>
    </Box>
  );
};

export default KnowledgeGraphAntiSiloSettings;
