import React, { useState, useEffect } from 'react';
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Paper,
  Container,
  ThemeProvider,
  createTheme,
  CssBaseline,
  Tabs,
  Tab,
  Grid,
  Card,
  Button,
  Alert,
  Snackbar
} from '@mui/material';
import {
  LightMode as LightModeIcon,
  DarkMode as DarkModeIcon,
  CompareArrows as CompareIcon,
  Assessment as ResultsIcon,
  Tune as ConfigIcon,
  LibraryBooks as ReferenceIcon,
  ViewModule as ViewModuleIcon,
  ViewList as ViewListIcon
} from '@mui/icons-material';
import IDCConfigurationPanel from './components/idc/IDCConfigurationPanel';
import IDCValidationPanel from './components/idc/IDCValidationPanel';
import IDCReferenceManager from './components/idc/IDCReferenceManager';
import IDCResultsViewer from './components/idc/IDCResultsViewer';

interface ReferenceDocument {
  id: string;
  document_id: string;
  name: string;
  document_type: string;
  category?: string;
  original_filename: string;
  file_size_bytes: number;
  extraction_model: string;
  extraction_confidence?: number;
  extracted_markdown?: string;
  recommended_extraction_modes?: string[];
  created_at: string;
  updated_at: string;
  is_active: boolean;
  processing_status?: 'uploading' | 'extracting' | 'completed' | 'failed';
}

interface ValidationSession {
  id: string;
  session_id: string;
  reference_document_id: string;
  input_filename: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  overall_score?: number;
  created_at: string;
  completed_at?: string;
  extraction_model: string;
  validation_model: string;
  extraction_mode: string;
  validation_method: string;
}

interface IDCConfiguration {
  extraction_model: string;
  validation_model: string;
  extraction_mode: string;
  validation_method: string;
  max_chunk_size: number;
  parallel_workers: number;
  max_context_usage: number;
  quality_threshold: number;
}

interface SnackbarState {
  open: boolean;
  message: string;
  severity: 'success' | 'error' | 'warning' | 'info';
}

function IDCApp() {
  // Theme management
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('jarvis-dark-mode');
    return saved ? JSON.parse(saved) : false;
  });

  // Navigation state
  const [activeTab, setActiveTab] = useState(0);
  const [useCardNavigation, setUseCardNavigation] = useState(() => {
    const saved = localStorage.getItem('idc-card-navigation');
    return saved ? JSON.parse(saved) : false;
  });

  // Reference documents state
  const [referenceDocuments, setReferenceDocuments] = useState<ReferenceDocument[]>([]);
  
  // Snackbar for notifications
  const [snackbar, setSnackbar] = useState<SnackbarState>({
    open: false,
    message: '',
    severity: 'info'
  });

  // Validation sessions state
  const [validationSessions, setValidationSessions] = useState<ValidationSession[]>([]);

  // Configuration state
  const [configuration, setConfiguration] = useState<IDCConfiguration>({
    extraction_model: 'qwen3:30b-a3b-q4_K_M',
    validation_model: 'qwen3:30b-a3b-q4_K_M', 
    extraction_mode: 'paragraph',
    validation_method: 'section_match',
    max_chunk_size: 4000,
    max_context_usage: 35,
    quality_threshold: 0.9,
    parallel_workers: 5
  });


  // Create theme
  const theme = createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      primary: {
        main: '#2196f3',
      },
      secondary: {
        main: '#ff9800',
      },
      background: {
        default: darkMode ? '#121212' : '#f5f5f5',
        paper: darkMode ? '#1e1e1e' : '#ffffff',
      },
    },
  });

  // Set theme data attribute for CSS
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  // Load initial data
  useEffect(() => {
    loadReferenceDocuments();
    loadValidationSessions();
  }, []);

  const toggleDarkMode = () => {
    const newDarkMode = !darkMode;
    setDarkMode(newDarkMode);
    localStorage.setItem('jarvis-dark-mode', JSON.stringify(newDarkMode));
  };

  const toggleNavigationStyle = (useCards: boolean) => {
    setUseCardNavigation(useCards);
    localStorage.setItem('idc-card-navigation', JSON.stringify(useCards));
  };

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    switch (newValue) {
      case 0:
        window.location.href = '/';
        break;
      case 1:
        window.location.href = '/multi-agent.html';
        break;
      case 2:
        window.location.href = '/workflow.html';
        break;
      case 3:
        window.location.href = '/settings.html';
        break;
      case 4:
        window.location.href = '/knowledge-graph.html';
        break;
      case 5:
        // Already on IDC page, do nothing
        break;
    }
  };

  const loadReferenceDocuments = async () => {
    try {
      const response = await fetch('/api/v1/idc/references');
      if (response.ok) {
        const data = await response.json();
        setReferenceDocuments(data.documents || []);
      }
    } catch (error) {
      console.error('Failed to load reference documents:', error);
    }
  };

  const loadValidationSessions = async () => {
    try {
      const response = await fetch('/api/v1/idc/validate/sessions');
      if (response.ok) {
        const data = await response.json();
        setValidationSessions(data.sessions || []);
      }
    } catch (error) {
      console.error('Failed to load validation sessions:', error);
    }
  };


  const renderReferenceManagement = () => (
    <IDCReferenceManager 
      references={referenceDocuments}
      onRefresh={loadReferenceDocuments}
      onUploadComplete={(document) => {
        // Refresh the reference documents list after upload
        loadReferenceDocuments();
        setSnackbar({
          open: true,
          message: `Reference document "${document.name}" uploaded successfully`,
          severity: 'success'
        });
      }}
    />
  );

  const renderDocumentValidation = () => (
    <IDCValidationPanel 
      references={referenceDocuments.map(doc => ({
        document_id: doc.document_id, 
        name: doc.name,
        document_type: doc.document_type
      }))}
      onValidationStart={(session) => {
        // Handle validation session start
        loadValidationSessions(); // Refresh sessions
        setActiveTab(2); // Switch to Results tab
        setSnackbar({
          open: true,
          message: `Validation started for session ${session.session_id}`,
          severity: 'success'
        });
      }}
    />
  );

  const renderResults = () => (
    <IDCResultsViewer 
      sessions={validationSessions}
      onRefresh={loadValidationSessions}
      onSessionSelect={(sessionId) => {
        console.log('Selected session:', sessionId);
        // Optional: Add any additional logic when a session is selected
      }}
    />
  );

  const renderConfiguration = () => (
    <Box>
      {/* IDCConfigurationPanel handles ALL configuration - no duplication */}
      <IDCConfigurationPanel 
        onConfigChange={(config) => {
          // Update local configuration state from the panel's config
          // Add null checks to prevent undefined errors
          setConfiguration(prev => ({
            ...prev,
            extraction_model: config?.extraction?.model || prev.extraction_model,
            validation_model: config?.validation?.model || prev.validation_model,
            max_context_usage: config?.validation?.max_context_usage ? config.validation.max_context_usage * 100 : prev.max_context_usage,
            quality_threshold: config?.validation?.confidence_threshold || prev.quality_threshold
          }));
        }}
        onShowSuccess={(message) => {
          // Show success notification
          setSnackbar({
            open: true,
            message: message || 'Configuration saved successfully',
            severity: 'success'
          });
        }}
      />
    </Box>
  );

  const renderTabContent = () => {
    switch (activeTab) {
      case 0:
        return renderReferenceManagement();
      case 1:
        return renderDocumentValidation();
      case 2:
        return renderResults();
      case 3:
        return renderConfiguration();
      default:
        return renderReferenceManagement();
    }
  };

  const renderCardNavigation = () => {
    const navItems = [
      { 
        label: 'Reference Management', 
        icon: <ReferenceIcon sx={{ fontSize: { xs: 32, sm: 36, md: 40 } }} />, 
        value: 0,
        description: 'Upload and manage reference documents',
        color: '#2196f3'
      },
      { 
        label: 'Document Validation', 
        icon: <CompareIcon sx={{ fontSize: { xs: 32, sm: 36, md: 40 } }} />, 
        value: 1,
        description: 'Validate documents against references',
        color: '#4caf50'
      },
      { 
        label: 'Results', 
        icon: <ResultsIcon sx={{ fontSize: { xs: 32, sm: 36, md: 40 } }} />, 
        value: 2,
        description: 'View validation results and analytics',
        color: '#ff9800'
      },
      { 
        label: 'Configuration', 
        icon: <ConfigIcon sx={{ fontSize: { xs: 32, sm: 36, md: 40 } }} />, 
        value: 3,
        description: 'Configure validation settings',
        color: '#9c27b0'
      }
    ];

    return (
      <Box sx={{ 
        py: { xs: 2, sm: 3, md: 4 }, 
        px: { xs: 1, sm: 2 },
        background: 'linear-gradient(135deg, rgba(33, 150, 243, 0.05) 0%, rgba(156, 39, 176, 0.05) 100%)',
        borderBottom: 1,
        borderColor: 'divider'
      }}>
        <Container maxWidth="lg">
          <Grid container spacing={{ xs: 2, sm: 2, md: 3 }} justifyContent="center">
            {navItems.map((item) => (
              <Grid item xs={6} sm={6} md={3} key={item.value}>
                <Card
                  sx={{
                    cursor: 'pointer',
                    transition: 'all 0.3s ease',
                    border: activeTab === item.value ? `2px solid ${item.color}` : '2px solid transparent',
                    boxShadow: activeTab === item.value ? `0 8px 24px ${item.color}33` : 1,
                    '&:hover': {
                      transform: { xs: 'none', sm: 'translateY(-4px)' },
                      boxShadow: `0 12px 28px ${item.color}44`,
                      borderColor: item.color
                    },
                    height: '100%',
                    minHeight: { xs: 120, sm: 140, md: 160 },
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    textAlign: 'center',
                    p: { xs: 2, sm: 2.5, md: 3 },
                    bgcolor: activeTab === item.value ? `${item.color}08` : 'background.paper'
                  }}
                  onClick={() => setActiveTab(item.value)}
                >
                  <Box sx={{ color: item.color, mb: { xs: 1, sm: 1.5, md: 2 } }}>
                    {item.icon}
                  </Box>
                  <Typography 
                    variant="h6" 
                    gutterBottom 
                    sx={{ 
                      fontSize: { xs: '0.9rem', sm: '1rem', md: '1.25rem' },
                      fontWeight: activeTab === item.value ? 700 : 600,
                      color: activeTab === item.value ? item.color : 'text.primary',
                      lineHeight: 1.2
                    }}
                  >
                    {item.label}
                  </Typography>
                  <Typography 
                    variant="body2" 
                    color="text.secondary"
                    sx={{ 
                      mt: 1,
                      display: { xs: 'none', sm: 'block' },
                      fontSize: { sm: '0.75rem', md: '0.875rem' }
                    }}
                  >
                    {item.description}
                  </Typography>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Container>
      </Box>
    );
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
        {/* Header */}
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              Jarvis AI Assistant
            </Typography>
            
            <IconButton onClick={toggleDarkMode} color="inherit">
              {darkMode ? <LightModeIcon /> : <DarkModeIcon />}
            </IconButton>
          </Toolbar>
        </AppBar>

        {/* Navigation Tabs */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={5}
            onChange={handleTabChange} 
            aria-label="jarvis modes"
            centered
            sx={{
              '& .MuiTab-root': {
                fontSize: '1rem',
                fontWeight: 600,
                textTransform: 'none',
                minWidth: 120,
                padding: '12px 24px',
                '&.Mui-selected': {
                  color: 'primary.main',
                  fontWeight: 700
                }
              }
            }}
          >
            <Tab 
              label="Standard Chat" 
              id="tab-0"
              aria-controls="tabpanel-0"
            />
            <Tab 
              label="Multi-Agent" 
              id="tab-1"
              aria-controls="tabpanel-1"
            />
            <Tab 
              label="Workflow" 
              id="tab-2"
              aria-controls="tabpanel-2"
            />
            <Tab 
              label="Settings" 
              id="tab-3"
              aria-controls="tabpanel-3"
            />
            <Tab 
              label="Knowledge Graph" 
              id="tab-4"
              aria-controls="tabpanel-4"
            />
            <Tab 
              label="IDC" 
              id="tab-5"
              aria-controls="tabpanel-5"
            />
          </Tabs>
        </Box>


        {/* IDC Module Header Section */}
        <Box sx={{ 
          bgcolor: 'background.paper',
          borderBottom: 1,
          borderColor: 'divider',
          py: 3,
          px: 2,
          background: 'linear-gradient(180deg, rgba(33, 150, 243, 0.03) 0%, rgba(255, 255, 255, 0) 100%)'
        }}>
          <Container maxWidth="lg">
            <Box sx={{ 
              display: 'flex', 
              flexDirection: { xs: 'column', sm: 'row' },
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 2
            }}>
              <Box sx={{ textAlign: { xs: 'center', sm: 'left' } }}>
                <Typography 
                  variant="h4" 
                  sx={{ 
                    fontWeight: 700,
                    color: 'primary.main',
                    mb: 0.5,
                    fontSize: { xs: '1.5rem', sm: '1.75rem', md: '2rem' }
                  }}
                >
                  Intelligent Document Compliance
                </Typography>
                <Typography 
                  variant="body1" 
                  color="text.secondary"
                  sx={{ fontSize: { xs: '0.875rem', sm: '1rem' } }}
                >
                  Validate documents against reference guidelines with AI-powered analysis
                </Typography>
              </Box>
              
              {/* Toggle for Navigation Style */}
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Button
                  variant={useCardNavigation ? "outlined" : "contained"}
                  size="small"
                  startIcon={<ViewListIcon />}
                  onClick={() => toggleNavigationStyle(false)}
                  sx={{ 
                    minWidth: { xs: 80, sm: 100 },
                    fontSize: { xs: '0.75rem', sm: '0.875rem' }
                  }}
                >
                  Tab View
                </Button>
                <Button
                  variant={useCardNavigation ? "contained" : "outlined"}
                  size="small"
                  startIcon={<ViewModuleIcon />}
                  onClick={() => toggleNavigationStyle(true)}
                  sx={{ 
                    minWidth: { xs: 80, sm: 100 },
                    fontSize: { xs: '0.75rem', sm: '0.875rem' }
                  }}
                >
                  Card View
                </Button>
              </Box>
            </Box>
          </Container>
        </Box>

        {/* IDC Navigation - Dynamic Based on View Mode */}
        {useCardNavigation ? (
          renderCardNavigation()
        ) : (
          <Box sx={{ 
            borderBottom: 1, 
            borderColor: 'divider', 
            bgcolor: 'background.paper',
            boxShadow: '0 2px 4px rgba(0,0,0,0.05)'
          }}>
            <Container maxWidth="lg">
              <Box sx={{ 
                display: 'flex', 
                justifyContent: 'center',
                alignItems: 'center',
                py: 1
              }}>
                <Tabs 
                  value={activeTab}
                  onChange={(_event, newValue) => setActiveTab(newValue)}
                  aria-label="IDC navigation"
                  centered
                  variant="scrollable"
                  scrollButtons="auto"
                  allowScrollButtonsMobile
                  sx={{
                    '& .MuiTabs-indicator': {
                      height: 3,
                      borderRadius: '3px 3px 0 0',
                      background: 'linear-gradient(90deg, #2196f3 0%, #1976d2 100%)'
                    },
                    '& .MuiTab-root': {
                      fontSize: { xs: '0.85rem', sm: '0.9rem', md: '0.95rem' },
                      fontWeight: 500,
                      textTransform: 'none',
                      minWidth: { xs: 120, sm: 140, md: 150 },
                      padding: { xs: '10px 16px', sm: '11px 18px', md: '12px 20px' },
                      margin: { xs: '0 4px', sm: '0 6px', md: '0 8px' },
                      borderRadius: '8px 8px 0 0',
                      transition: 'all 0.3s ease',
                      color: 'text.secondary',
                      '&:hover': {
                        bgcolor: 'action.hover',
                        color: 'primary.main'
                      },
                      '&.Mui-selected': {
                        color: 'primary.main',
                        fontWeight: 600,
                        bgcolor: 'action.selected'
                      },
                      '& .MuiTab-iconWrapper': {
                        marginRight: { xs: 4, sm: 5, md: 6 },
                        fontSize: { xs: '1rem', sm: '1.1rem', md: '1.2rem' }
                      }
                    },
                    '& .MuiTabs-scrollButtons': {
                      '&.Mui-disabled': {
                        opacity: 0.3
                      }
                    }
                  }}
                >
                  <Tab 
                    label="Reference Management"
                    icon={<ReferenceIcon />}
                    iconPosition="start"
                  />
                  <Tab 
                    label="Document Validation"
                    icon={<CompareIcon />}
                    iconPosition="start"
                  />
                  <Tab 
                    label="Results"
                    icon={<ResultsIcon />}
                    iconPosition="start"
                  />
                  <Tab 
                    label="Configuration"
                    icon={<ConfigIcon />}
                    iconPosition="start"
                  />
                </Tabs>
              </Box>
            </Container>
          </Box>
        )}

        {/* Main Content */}
        <Container maxWidth={false} sx={{ flex: 1, py: 3, overflow: 'auto' }}>
          {renderTabContent()}
        </Container>
        
        {/* Snackbar for notifications */}
        <Snackbar
          open={snackbar.open}
          autoHideDuration={6000}
          onClose={() => setSnackbar({ ...snackbar, open: false })}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        >
          <Alert
            onClose={() => setSnackbar({ ...snackbar, open: false })}
            severity={snackbar.severity}
            sx={{ width: '100%' }}
          >
            {snackbar.message}
          </Alert>
        </Snackbar>
      </Box>
    </ThemeProvider>
  );
}

export default IDCApp;