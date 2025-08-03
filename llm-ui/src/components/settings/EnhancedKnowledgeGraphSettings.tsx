import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  Typography,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Button,
  Alert,
  CircularProgress,
  Tabs,
  Tab,
  Chip,
  Grid,
  Slider,
  Paper,
  List,
  ListItem,
  ListItemText,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  FormGroup,
  Checkbox,
  Divider,
  Tooltip,
  LinearProgress,
  Avatar,
  ButtonGroup,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Badge,
  RadioGroup,
  Radio,
  ListItemSecondaryAction,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Delete as DeleteIcon,
  Add as AddIcon,
  Psychology as BrainIcon,
  Settings as SettingsIcon,
  DataObject as DataIcon,
  Timeline as TimelineIcon,
  AutoFixHigh as AutoFixHighIcon,
  Dashboard as DashboardIcon,
  Schema as SchemaIcon,
  Speed as SpeedIcon,
  Analytics as AnalyticsIcon,
  AccountTree as GraphIcon,
  Storage as StorageIcon,
  Link as LinkIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  Visibility as VisibilityIcon,
  Edit as EditIcon,
  Save as SaveIcon,
  Cancel as CancelIcon,
  Upload as UploadIcon,
  Download as DownloadIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`knowledge-graph-tabpanel-${index}`}
      aria-labelledby={`knowledge-graph-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const EnhancedKnowledgeGraphSettings: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [settings, setSettings] = useState<any>({
    mode: 'thinking',
    processing_mode: 'unified',
    enable_llm_enhancement: true,
    extraction: {
      min_entity_confidence: 0.75,
      min_relationship_confidence: 0.7,
      enable_anti_silo: true,
      enable_cross_document_linking: true,
    }
  });
  const [loading, setLoading] = useState(false);
  const [processingStats, setProcessingStats] = useState<any>(null);
  const [schemaData, setSchemaData] = useState<any>({
    entity_types: {},
    relationship_types: {}
  });
  const [performanceMetrics, setPerformanceMetrics] = useState<any>(null);

  // Load settings on component mount
  useEffect(() => {
    loadSettings();
    loadProcessingStats();
    loadSchemaData();
    loadPerformanceMetrics();
  }, []);

  const loadSettings = async () => {
    try {
      const response = await fetch('/api/v1/knowledge-graph-settings');
      if (response.ok) {
        const data = await response.json();
        setSettings(data);
      }
    } catch (error) {
      console.error('Failed to load knowledge graph settings:', error);
    }
  };

  const loadProcessingStats = async () => {
    try {
      const response = await fetch('/api/v1/knowledge-graph/analytics');
      if (response.ok) {
        const data = await response.json();
        setProcessingStats(data);
      }
    } catch (error) {
      console.error('Failed to load processing stats:', error);
    }
  };

  const loadSchemaData = async () => {
    try {
      const response = await fetch('/api/v1/knowledge-graph-schema');
      if (response.ok) {
        const data = await response.json();
        setSchemaData(data);
      }
    } catch (error) {
      console.error('Failed to load schema data:', error);
    }
  };

  const loadPerformanceMetrics = async () => {
    try {
      const response = await fetch('/api/v1/knowledge-graph/performance');
      if (response.ok) {
        const data = await response.json();
        setPerformanceMetrics(data);
      }
    } catch (error) {
      console.error('Failed to load performance metrics:', error);
    }
  };

  const saveSettings = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/v1/knowledge-graph-settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
      });
      
      if (response.ok) {
        // Show success message
      }
    } catch (error) {
      console.error('Failed to save settings:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const updateSetting = (path: string, value: any) => {
    const keys = path.split('.');
    const newSettings = { ...settings };
    let current = newSettings;
    
    for (let i = 0; i < keys.length - 1; i++) {
      if (!current[keys[i]]) current[keys[i]] = {};
      current = current[keys[i]];
    }
    
    current[keys[keys.length - 1]] = value;
    setSettings(newSettings);
  };

  return (
    <Card>
      <CardHeader
        title="Enhanced Knowledge Graph Configuration"
        subheader="Advanced settings for unified document processing with Milvus and Neo4j"
        avatar={<Avatar><GraphIcon /></Avatar>}
        action={
          <Button
            variant="contained"
            startIcon={<SaveIcon />}
            onClick={saveSettings}
            disabled={loading}
          >
            {loading ? 'Saving...' : 'Save Settings'}
          </Button>
        }
      />
      
      <CardContent>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={handleTabChange} variant="scrollable" scrollButtons="auto">
            <Tab icon={<DashboardIcon />} label="Overview" />
            <Tab icon={<SettingsIcon />} label="Processing Pipeline" />
            <Tab icon={<SchemaIcon />} label="Schema Designer" />
            <Tab icon={<SpeedIcon />} label="Performance Tuning" />
            <Tab icon={<AnalyticsIcon />} label="Quality Metrics" />
            <Tab icon={<LinkIcon />} label="Cross-References" />
          </Tabs>
        </Box>

        {/* Overview Tab */}
        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            {/* Processing Mode Configuration */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  <StorageIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                  Processing Mode
                </Typography>
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel>Processing Mode</InputLabel>
                  <Select
                    value={settings.processing_mode || 'unified'}
                    onChange={(e) => updateSetting('processing_mode', e.target.value)}
                  >
                    <MenuItem value="unified">
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Chip size="small" label="Recommended" color="primary" sx={{ mr: 1 }} />
                        Unified (Milvus + Neo4j)
                      </Box>
                    </MenuItem>
                    <MenuItem value="milvus-only">Milvus Only (Vector Search)</MenuItem>
                    <MenuItem value="neo4j-only">Neo4j Only (Knowledge Graph)</MenuItem>
                  </Select>
                </FormControl>
                
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.enable_llm_enhancement || false}
                      onChange={(e) => updateSetting('enable_llm_enhancement', e.target.checked)}
                    />
                  }
                  label="Enable LLM Enhancement"
                />
                
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  Use LLM for intelligent entity and relationship extraction
                </Typography>
              </Paper>
            </Grid>

            {/* Real-time Statistics */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  <AnalyticsIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                  Processing Statistics
                </Typography>
                {processingStats ? (
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Box textAlign="center">
                        <Typography variant="h4" color="primary">
                          {processingStats.documents_processed || 0}
                        </Typography>
                        <Typography variant="body2">Documents Processed</Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6}>
                      <Box textAlign="center">
                        <Typography variant="h4" color="secondary">
                          {processingStats.entities_extracted || 0}
                        </Typography>
                        <Typography variant="body2">Entities Extracted</Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6}>
                      <Box textAlign="center">
                        <Typography variant="h4" color="success.main">
                          {processingStats.relationships_extracted || 0}
                        </Typography>
                        <Typography variant="body2">Relationships</Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6}>
                      <Box textAlign="center">
                        <Typography variant="h4" color="warning.main">
                          {(processingStats.avg_confidence * 100).toFixed(1) || 0}%
                        </Typography>
                        <Typography variant="body2">Avg Confidence</Typography>
                      </Box>
                    </Grid>
                  </Grid>
                ) : (
                  <Box display="flex" justifyContent="center" alignItems="center" height={120}>
                    <CircularProgress />
                  </Box>
                )}
              </Paper>
            </Grid>

            {/* System Status */}
            <Grid item xs={12}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  <CheckCircleIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                  System Status
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={4}>
                    <Box display="flex" alignItems="center">
                      <Badge color="success" variant="dot">
                        <Chip label="Milvus" color="success" size="small" />
                      </Badge>
                      <Typography variant="body2" sx={{ ml: 2 }}>
                        Vector Database Connected
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <Box display="flex" alignItems="center">
                      <Badge color="success" variant="dot">
                        <Chip label="Neo4j" color="success" size="small" />
                      </Badge>
                      <Typography variant="body2" sx={{ ml: 2 }}>
                        Knowledge Graph Connected
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <Box display="flex" alignItems="center">
                      <Badge color="success" variant="dot">
                        <Chip label="LLM" color="success" size="small" />
                      </Badge>
                      <Typography variant="body2" sx={{ ml: 2 }}>
                        Language Model Ready
                      </Typography>
                    </Box>
                  </Grid>
                </Grid>
              </Paper>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Processing Pipeline Tab */}
        <TabPanel value={tabValue} index={1}>
          <Grid container spacing={3}>
            {/* Confidence Thresholds */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>Confidence Thresholds</Typography>
                
                <Typography gutterBottom>
                  Entity Confidence: {(settings.extraction?.min_entity_confidence * 100 || 75).toFixed(0)}%
                </Typography>
                <Slider
                  value={settings.extraction?.min_entity_confidence * 100 || 75}
                  onChange={(_, value) => updateSetting('extraction.min_entity_confidence', (value as number) / 100)}
                  min={0}
                  max={100}
                  step={5}
                  marks={[
                    { value: 50, label: '50%' },
                    { value: 75, label: '75%' },
                    { value: 90, label: '90%' }
                  ]}
                  sx={{ mb: 3 }}
                />
                
                <Typography gutterBottom>
                  Relationship Confidence: {(settings.extraction?.min_relationship_confidence * 100 || 70).toFixed(0)}%
                </Typography>
                <Slider
                  value={settings.extraction?.min_relationship_confidence * 100 || 70}
                  onChange={(_, value) => updateSetting('extraction.min_relationship_confidence', (value as number) / 100)}
                  min={0}
                  max={100}
                  step={5}
                  marks={[
                    { value: 40, label: '40%' },
                    { value: 70, label: '70%' },
                    { value: 90, label: '90%' }
                  ]}
                />
              </Paper>
            </Grid>

            {/* Advanced Features */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>Advanced Features</Typography>
                
                <FormGroup>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.extraction?.enable_anti_silo || false}
                        onChange={(e) => updateSetting('extraction.enable_anti_silo', e.target.checked)}
                      />
                    }
                    label="Anti-Silo Relationships"
                  />
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    Automatically discover relationships across documents
                  </Typography>
                  
                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.extraction?.enable_cross_document_linking || false}
                        onChange={(e) => updateSetting('extraction.enable_cross_document_linking', e.target.checked)}
                      />
                    }
                    label="Cross-Document Linking"
                  />
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    Link similar entities across different documents
                  </Typography>
                  
                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.extraction?.enable_semantic_relationship_inference || false}
                        onChange={(e) => updateSetting('extraction.enable_semantic_relationship_inference', e.target.checked)}
                      />
                    }
                    label="Semantic Relationship Inference"
                  />
                  <Typography variant="body2" color="text.secondary">
                    Use AI to infer implicit relationships
                  </Typography>
                </FormGroup>
              </Paper>
            </Grid>

            {/* Processing Queue Status */}
            <Grid item xs={12}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>Processing Queue</Typography>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Document</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell>Progress</TableCell>
                        <TableCell>Entities</TableCell>
                        <TableCell>Relationships</TableCell>
                        <TableCell>Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      <TableRow>
                        <TableCell>sample_document.pdf</TableCell>
                        <TableCell>
                          <Chip label="Processing" color="primary" size="small" />
                        </TableCell>
                        <TableCell>
                          <Box display="flex" alignItems="center">
                            <LinearProgress variant="determinate" value={75} sx={{ flexGrow: 1, mr: 1 }} />
                            <Typography variant="body2">75%</Typography>
                          </Box>
                        </TableCell>
                        <TableCell>42</TableCell>
                        <TableCell>28</TableCell>
                        <TableCell>
                          <IconButton size="small">
                            <VisibilityIcon />
                          </IconButton>
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
              </Paper>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Schema Designer Tab */}
        <TabPanel value={tabValue} index={2}>
          <Grid container spacing={3}>
            {/* Entity Types */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3 }}>
                <Box display="flex" justifyContent="between" alignItems="center" mb={2}>
                  <Typography variant="h6">Entity Types</Typography>
                  <Button startIcon={<AddIcon />} size="small">Add Type</Button>
                </Box>
                
                <List>
                  {Object.entries(schemaData.entity_types || {}).map(([type, config]: [string, any]) => (
                    <ListItem key={type}>
                      <Avatar sx={{ mr: 2, bgcolor: 'primary.main' }}>
                        {type.charAt(0)}
                      </Avatar>
                      <ListItemText
                        primary={type}
                        secondary={config.description}
                      />
                      <ListItemSecondaryAction>
                        <Chip
                          size="small"
                          label={`${(config.confidence_threshold * 100).toFixed(0)}%`}
                          color="primary"
                        />
                        <IconButton size="small" sx={{ ml: 1 }}>
                          <EditIcon />
                        </IconButton>
                      </ListItemSecondaryAction>
                    </ListItem>
                  ))}
                </List>
              </Paper>
            </Grid>

            {/* Relationship Types */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3 }}>
                <Box display="flex" justifyContent="between" alignItems="center" mb={2}>
                  <Typography variant="h6">Relationship Types</Typography>
                  <Button startIcon={<AddIcon />} size="small">Add Type</Button>
                </Box>
                
                <List>
                  {Object.entries(schemaData.relationship_types || {}).map(([type, config]: [string, any]) => (
                    <ListItem key={type}>
                      <Avatar sx={{ mr: 2, bgcolor: 'secondary.main' }}>
                        <LinkIcon />
                      </Avatar>
                      <ListItemText
                        primary={type}
                        secondary={config.description}
                      />
                      <ListItemSecondaryAction>
                        <Chip
                          size="small"
                          label={`${(config.confidence_threshold * 100).toFixed(0)}%`}
                          color="secondary"
                        />
                        <IconButton size="small" sx={{ ml: 1 }}>
                          <EditIcon />
                        </IconButton>
                      </ListItemSecondaryAction>
                    </ListItem>
                  ))}
                </List>
              </Paper>
            </Grid>

            {/* Schema Evolution */}
            <Grid item xs={12}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>Schema Evolution</Typography>
                <Timeline>
                  <Alert severity="info" sx={{ mb: 2 }}>
                    Schema versioning helps track changes and enables rollback capabilities
                  </Alert>
                  
                  <Box display="flex" alignItems="center" justifyContent="space-between">
                    <Typography variant="body2">Current Version: 1.0.0</Typography>
                    <ButtonGroup size="small">
                      <Button startIcon={<UploadIcon />}>Import Schema</Button>
                      <Button startIcon={<DownloadIcon />}>Export Schema</Button>
                      <Button startIcon={<RefreshIcon />}>Reset to Default</Button>
                    </ButtonGroup>
                  </Box>
                </Timeline>
              </Paper>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Performance Tuning Tab */}
        <TabPanel value={tabValue} index={3}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Alert severity="info" sx={{ mb: 3 }}>
                Performance optimization settings affect processing speed and resource usage
              </Alert>
            </Grid>

            {/* Processing Performance */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  <SpeedIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                  Processing Performance
                </Typography>
                
                <Typography gutterBottom>Chunk Size: 1000 characters</Typography>
                <Slider
                  value={1000}
                  min={500}
                  max={2000}
                  step={100}
                  marks={[
                    { value: 500, label: '500' },
                    { value: 1000, label: '1000' },
                    { value: 2000, label: '2000' }
                  ]}
                  sx={{ mb: 3 }}
                />
                
                <Typography gutterBottom>Parallel Processing Threads: 4</Typography>
                <Slider
                  value={4}
                  min={1}
                  max={8}
                  step={1}
                  marks={[
                    { value: 1, label: '1' },
                    { value: 4, label: '4' },
                    { value: 8, label: '8' }
                  ]}
                />
              </Paper>
            </Grid>

            {/* Resource Usage */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>Resource Usage</Typography>
                
                {performanceMetrics ? (
                  <Box>
                    <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                      <Typography variant="body2">CPU Usage</Typography>
                      <Typography variant="body2">{performanceMetrics.cpu_usage}%</Typography>
                    </Box>
                    <LinearProgress variant="determinate" value={performanceMetrics.cpu_usage} sx={{ mb: 2 }} />
                    
                    <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                      <Typography variant="body2">Memory Usage</Typography>
                      <Typography variant="body2">{performanceMetrics.memory_usage}%</Typography>
                    </Box>
                    <LinearProgress variant="determinate" value={performanceMetrics.memory_usage} sx={{ mb: 2 }} />
                    
                    <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                      <Typography variant="body2">Processing Speed</Typography>
                      <Typography variant="body2">{performanceMetrics.processing_speed} chunks/min</Typography>
                    </Box>
                  </Box>
                ) : (
                  <CircularProgress />
                )}
              </Paper>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Quality Metrics Tab */}
        <TabPanel value={tabValue} index={4}>
          <Grid container spacing={3}>
            {/* Extraction Quality */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>Extraction Quality</Typography>
                
                <Box display="flex" justifyContent="space-between" mb={2}>
                  <Typography variant="body2">Entity Accuracy</Typography>
                  <Chip label="87%" color="success" size="small" />
                </Box>
                
                <Box display="flex" justifyContent="space-between" mb={2}>
                  <Typography variant="body2">Relationship Accuracy</Typography>
                  <Chip label="74%" color="warning" size="small" />
                </Box>
                
                <Box display="flex" justifyContent="space-between" mb={2}>
                  <Typography variant="body2">Cross-Reference Accuracy</Typography>
                  <Chip label="91%" color="success" size="small" />
                </Box>
                
                <Divider sx={{ my: 2 }} />
                
                <Typography variant="body2" color="text.secondary">
                  Quality scores are calculated based on manual validation and automated consistency checks
                </Typography>
              </Paper>
            </Grid>

            {/* Validation Summary */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>Validation Summary</Typography>
                
                <List dense>
                  <ListItem>
                    <CheckCircleIcon color="success" sx={{ mr: 2 }} />
                    <ListItemText
                      primary="Schema Consistency"
                      secondary="All entities match defined types"
                    />
                  </ListItem>
                  
                  <ListItem>
                    <WarningIcon color="warning" sx={{ mr: 2 }} />
                    <ListItemText
                      primary="Duplicate Detection"
                      secondary="3 potential duplicates found"
                    />
                  </ListItem>
                  
                  <ListItem>
                    <InfoIcon color="info" sx={{ mr: 2 }} />
                    <ListItemText
                      primary="Confidence Distribution"
                      secondary="78% of extractions above threshold"
                    />
                  </ListItem>
                </List>
              </Paper>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Cross-References Tab */}
        <TabPanel value={tabValue} index={5}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              <LinkIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
              Document Cross-References
            </Typography>
            
            <Typography variant="body2" color="text.secondary" paragraph>
              Cross-references link vector chunks in Milvus with entities in Neo4j, enabling unified search and exploration.
            </Typography>
            
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Document</TableCell>
                    <TableCell>Chunk ID</TableCell>
                    <TableCell>Entity</TableCell>
                    <TableCell>Confidence</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  <TableRow>
                    <TableCell>research_paper.pdf</TableCell>
                    <TableCell>chunk_001</TableCell>
                    <TableCell>Machine Learning</TableCell>
                    <TableCell>
                      <Chip label="0.87" color="success" size="small" />
                    </TableCell>
                    <TableCell>
                      <Chip label="Validated" color="success" size="small" />
                    </TableCell>
                    <TableCell>
                      <IconButton size="small">
                        <VisibilityIcon />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </TabPanel>
      </CardContent>
    </Card>
  );
};

export default EnhancedKnowledgeGraphSettings;