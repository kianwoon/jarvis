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
  Badge
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
  Code as CodeIcon
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import PromptManagement from './PromptManagement';

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

const KnowledgeGraphSettings: React.FC<KnowledgeGraphSettingsProps> = ({ 
  data, 
  onChange, 
  onShowSuccess 
}) => {
  const { enqueueSnackbar } = useSnackbar();
  const [testText, setTestText] = useState('');
  const [discoveryResults, setDiscoveryResults] = useState<any>(null);
  const [tabValue, setTabValue] = useState(0);

  // Simplified anti-silo focused configuration
  const config = data?.knowledge_graph || {
    anti_silo_mode: 'enhanced',
    entity_discovery: {
      enabled: true,
      confidence_threshold: 0.75,
      auto_linking: true,
      relationship_boost: true
    },
    relationship_discovery: {
      enabled: true,
      confidence_threshold: 0.7,
      cross_document: true,
      semantic_linking: true
    },
    static_fallback: {
      entity_types: ['Person', 'Organization', 'Location', 'Event', 'Concept'],
      relationship_types: ['works_for', 'located_in', 'part_of', 'related_to', 'causes']
    }
  };

  const discoveredEntities = data?.discovered_entities || [];
  const discoveredRelationships = data?.discovered_relationships || [];
  const stats = data?.schema_stats || null;
  const loading = false;


  const updateConfiguration = (newConfig: any) => {
    try {
      onChange('knowledge_graph', newConfig);
      if (onShowSuccess) {
        onShowSuccess("Configuration updated: Knowledge graph settings saved successfully");
      }
    } catch (error) {
      enqueueSnackbar("Update failed: Failed to update configuration", { 
        variant: "error" 
      });
    }
  };

  const handleSchemaModeChange = (mode: string) => {
    const newConfig = {
      ...config,
      schema_mode: mode
    };
    updateConfiguration(newConfig);
  };

  const toggleDiscovery = (type: 'entity' | 'relationship', enabled: boolean) => {
    const key = type === 'entity' ? 'entity_discovery' : 'relationship_discovery';
    const newConfig = {
      ...config,
      [key]: {
        ...config[key],
        enabled
      }
    };
    updateConfiguration(newConfig);
  };

  const updateConfidenceThreshold = (type: 'entity' | 'relationship', value: number) => {
    const key = type === 'entity' ? 'entity_discovery' : 'relationship_discovery';
    const newConfig = {
      ...config,
      [key]: {
        ...config[key],
        confidence_threshold: value
      }
    };
    updateConfiguration(newConfig);
  };

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
            ...config,
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

  if (!config) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '256px' }}>
        <CircularProgress />
      </Box>
    );
  }

  const pendingEntities = discoveredEntities.filter(e => e.status === 'pending');
  const pendingRelationships = discoveredRelationships.filter(r => r.status === 'pending');
  const acceptedEntities = discoveredEntities.filter(e => e.status === 'accepted');
  const acceptedRelationships = discoveredRelationships.filter(r => r.status === 'accepted');

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
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
        <Tab label="Schema Mode" />
        <Tab label="Discovery" />
        <Tab label="Prompts" />
        <Tab label="Test Discovery" />
      </Tabs>

      {tabValue === 0 && (
        <Card>
          <CardHeader title="Schema Mode Configuration" />
          <CardContent>
            <Box sx={{ mb: 4 }}>
              <Typography variant="h6" gutterBottom>Schema Mode</Typography>
              <RadioGroup 
                value={config.schema_mode} 
                onChange={(e) => handleSchemaModeChange(e.target.value)}
              >
                <FormControlLabel value="static" control={<Radio />} label="Static Only - Use predefined entity/relationship types" />
                <FormControlLabel value="dynamic" control={<Radio />} label="Dynamic Only - Use LLM-discovered types" />
                <FormControlLabel value="hybrid" control={<Radio />} label="Hybrid - Combine static types with LLM discoveries" />
              </RadioGroup>
            </Box>

            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardHeader title="Entity Discovery" />
                  <CardContent>
                    <Box sx={{ mb: 2 }}>
                      <FormControlLabel 
                        control={
                          <Switch
                            checked={config.entity_discovery?.enabled || false}
                            onChange={(e) => toggleDiscovery('entity', e.target.checked)}
                          />
                        }
                        label="Enable Entity Discovery"
                      />
                    </Box>
                    <Typography gutterBottom>Confidence Threshold: {config.entity_discovery?.confidence_threshold || 0.75}</Typography>
                    <Slider
                      value={config.entity_discovery?.confidence_threshold || 0.75}
                      onChange={(_, value) => updateConfidenceThreshold('entity', value as number)}
                      min={0.1}
                      max={1}
                      step={0.05}
                      valueLabelDisplay="auto"
                    />
                    <Typography variant="body2" color="text.secondary">
                      Minimum confidence (0-1) for accepting discovered entity types
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={6}>
                <Card>
                  <CardHeader title="Relationship Discovery" />
                  <CardContent>
                    <Box sx={{ mb: 2 }}>
                      <FormControlLabel 
                        control={
                          <Switch
                            checked={config.relationship_discovery?.enabled || false}
                            onChange={(e) => toggleDiscovery('relationship', e.target.checked)}
                          />
                        }
                        label="Enable Relationship Discovery"
                      />
                    </Box>
                    <Typography gutterBottom>Confidence Threshold: {config.relationship_discovery?.confidence_threshold || 0.7}</Typography>
                    <Slider
                      value={config.relationship_discovery?.confidence_threshold || 0.7}
                      onChange={(_, value) => updateConfidenceThreshold('relationship', value as number)}
                      min={0.1}
                      max={1}
                      step={0.05}
                      valueLabelDisplay="auto"
                    />
                    <Typography variant="body2" color="text.secondary">
                      Minimum confidence (0-1) for accepting discovered relationship types
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            <Card sx={{ mt: 3 }}>
              <CardHeader title="Static Fallback Types" />
              <CardContent>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="h6" gutterBottom>Entity Types</Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                      {(config.static_fallback?.entity_types || ['Person', 'Organization', 'Location', 'Event', 'Concept']).map(type => (
                        <Chip key={type} label={type} variant="outlined" />
                      ))}
                    </Box>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Typography variant="h6" gutterBottom>Relationship Types</Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                      {(config.static_fallback?.relationship_types || ['works_for', 'located_in', 'part_of', 'related_to', 'causes']).map(type => (
                        <Chip key={type} label={type} variant="outlined" />
                      ))}
                    </Box>
                  </Grid>
                </Grid>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                  Static types are used as fallback when schema_mode is 'static' or 'hybrid'
                </Typography>
              </CardContent>
            </Card>

            {stats && (
              <Card sx={{ mt: 3 }}>
                <CardHeader title="Discovery Statistics" />
                <CardContent>
                  <Grid container spacing={2}>
                    <Grid item xs={6} md={3}>
                      <Box textAlign="center">
                        <Typography variant="h4">{stats.total_entities_discovered}</Typography>
                        <Typography variant="body2" color="text.secondary">Entities Discovered</Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6} md={3}>
                      <Box textAlign="center">
                        <Typography variant="h4">{stats.entities_accepted}</Typography>
                        <Typography variant="body2" color="text.secondary">Accepted</Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6} md={3}>
                      <Box textAlign="center">
                        <Typography variant="h4">{stats.total_relationships_discovered}</Typography>
                        <Typography variant="body2" color="text.secondary">Relationships Discovered</Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6} md={3}>
                      <Box textAlign="center">
                        <Typography variant="h4">{stats.relationships_accepted}</Typography>
                        <Typography variant="body2" color="text.secondary">Accepted</Typography>
                      </Box>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            )}
          </CardContent>
        </Card>
      )}

      {tabValue === 1 && (
        <Card>
          <CardHeader title="Discovered Entity Types" />
          <CardContent>
            <Box>
              {pendingEntities.length > 0 && (
                <Box sx={{ mb: 3 }}>
                  <Typography variant="h6" gutterBottom>Pending Review</Typography>
                  {pendingEntities.map((entity) => (
                    <Card key={entity.type} sx={{ mb: 2 }}>
                      <CardContent>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                          <Box flex={1}>
                            <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
                              <Typography variant="h6">{entity.type}</Typography>
                              <Chip label={entity.confidence.toFixed(2)} size="small" />
                              <Chip label={`${entity.frequency}x`} size="small" />
                            </Box>
                            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                              {entity.description}
                            </Typography>
                            <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                              {entity.examples.map((example, idx) => (
                                <Chip key={idx} label={example} size="small" variant="outlined" />
                              ))}
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
                              variant="contained"
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
                  <Typography variant="h6" gutterBottom>Accepted</Typography>
                  {acceptedEntities.map((entity) => (
                    <Card key={entity.type} sx={{ mb: 2, borderLeft: '4px solid #4caf50' }}>
                      <CardContent>
                        <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
                          <Typography variant="h6">{entity.type}</Typography>
                          <Chip label="Accepted" size="small" color="success" icon={<CheckCircleIcon />} />
                        </Box>
                        <Typography variant="body2" color="text.secondary">
                          {entity.description}
                        </Typography>
                      </CardContent>
                    </Card>
                  ))}
                </Box>
              )}
            </Box>
          </CardContent>
        </Card>
      )}

      {tabValue === 2 && (
        <PromptManagement 
          data={data?.prompts || []} 
          onChange={(field, value) => onChange('prompts', value)} 
          onShowSuccess={onShowSuccess}
        />
      )}

      {tabValue === 3 && (
        <Card>
          <CardHeader title="Test Schema Discovery" />
          <CardContent>
            <Box sx={{ mb: 3 }}>
              <Typography variant="body1" gutterBottom>
                Enter text to analyze for schema discovery
              </Typography>
              <TextField
                fullWidth
                multiline
                rows={4}
                value={testText}
                onChange={(e) => setTestText(e.target.value)}
                placeholder="Paste your text here to discover new entity and relationship types..."
                variant="outlined"
              />
            </Box>
            
            <Button 
              variant="contained"
              onClick={runDiscovery}
              disabled={!testText.trim() || loading}
              startIcon={loading ? <CircularProgress size={20} /> : <BrainIcon />}
              sx={{ mb: 3 }}
            >
              {loading ? 'Analyzing...' : 'Run Discovery'}
            </Button>

            {discoveryResults && (
              <Box>
                <Alert severity="success" sx={{ mb: 3 }}>
                  Discovery Results: {discoveryResults.discovered_entities?.length || 0} entities, 
                  {discoveryResults.discovered_relationships?.length || 0} relationships discovered
                </Alert>
                
                {discoveryResults.discovered_entities?.length > 0 && (
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="h6" gutterBottom>Discovered Entities</Typography>
                    {discoveryResults.discovered_entities.map((entity: any, idx: number) => (
                      <Paper key={idx} sx={{ p: 2, mb: 1 }}>
                        <Typography variant="h6">{entity.type}</Typography>
                        <Typography variant="body2" color="text.secondary">{entity.description}</Typography>
                      </Paper>
                    ))}
                  </Box>
                )}
              </Box>
            )}
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default KnowledgeGraphSettings;
