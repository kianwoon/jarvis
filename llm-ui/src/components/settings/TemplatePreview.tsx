import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  TextField,
  Button,
  Divider,
  Grid,
  Chip,
  IconButton,
  Tooltip,
  Alert,
  Card,
  CardContent,
  CardHeader,
  CircularProgress
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
  ContentCopy as CopyIcon,
  Download as DownloadIcon,
  Code as CodeIcon,
  Preview as PreviewIcon
} from '@mui/icons-material';

interface Template {
  name: string;
  content: string;
  description: string;
  variables: string[];
  active: boolean;
}

interface TemplatePreviewProps {
  template: Template;
  category: 'synthesis_prompts' | 'formatting_templates' | 'system_behaviors';
  onClose?: () => void;
  showVariableInputs?: boolean;
  autoPreview?: boolean;
}

const TemplatePreview: React.FC<TemplatePreviewProps> = ({
  template,
  category,
  onClose,
  showVariableInputs = true,
  autoPreview = true
}) => {
  const [variableValues, setVariableValues] = useState<Record<string, string>>({});
  const [previewContent, setPreviewContent] = useState<string>('');
  const [showRaw, setShowRaw] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [previewError, setPreviewError] = useState<string>('');

  // Sample data for different categories
  const getSampleData = (): Record<string, string> => {
    switch (category) {
      case 'synthesis_prompts':
        return {
          topic: 'Renewable Energy Technologies',
          user_query: 'What are the latest developments in solar panel efficiency?',
          document_content: 'Recent research shows that perovskite solar cells have achieved efficiency rates of over 25%. Silicon-based panels continue to improve with PERC technology reaching 22% efficiency in commercial applications. Bifacial panels are gaining market share due to their ability to capture light from both sides.',
          key_themes: 'Efficiency improvements, Cost reduction, New materials, Manufacturing scalability',
          source_count: '12',
          confidence_score: '0.89',
          audience_level: 'technical professional',
          output_length: 'comprehensive',
          focus_areas: 'efficiency metrics, cost analysis, market trends',
          timestamp: new Date().toLocaleString(),
          processing_time: '1.8 seconds'
        };

      case 'formatting_templates':
        return {
          section_title: 'Research Findings Summary',
          summary: 'Analysis of 15 peer-reviewed studies on renewable energy efficiency published between 2022-2024',
          formatted_points: '• Solar panel efficiency increased by 3.2% annually\n• Manufacturing costs decreased by 12% over two years\n• Market adoption accelerated in developing countries\n• Policy incentives drove 40% of new installations',
          source_list: '[1] Chen et al. (2024), [2] Kumar & Singh (2023), [3] Environmental Research Institute (2024)',
          citation_style: 'APA',
          reference_count: '15',
          timestamp: new Date().toLocaleString(),
          confidence_score: '0.92'
        };

      case 'system_behaviors':
        return {
          request_type: 'research_synthesis',
          reasoning_mode: 'analytical',
          confidence_threshold: '0.80',
          verification_required: 'true',
          fact_check_level: 'thorough',
          fallback_strategy: 'use_simplified_response',
          citation_style: 'academic',
          processing_time: '2.1 seconds'
        };

      default:
        return {};
    }
  };

  // Initialize with sample data
  useEffect(() => {
    const sampleData = getSampleData();
    const initialValues: Record<string, string> = {};
    
    template.variables.forEach(variable => {
      initialValues[variable] = sampleData[variable] || `[${variable}]`;
    });
    
    setVariableValues(initialValues);
  }, [template, category]);

  // Generate preview when template or variables change
  useEffect(() => {
    if (autoPreview) {
      generatePreview();
    }
  }, [template.content, variableValues, autoPreview]);

  const generatePreview = () => {
    setIsProcessing(true);
    setPreviewError('');
    
    try {
      let preview = template.content;
      
      // Replace all variables with their values
      template.variables.forEach(variable => {
        const value = variableValues[variable] || `{${variable}}`;
        const regex = new RegExp(`\\{${variable}\\}`, 'g');
        preview = preview.replace(regex, value);
      });
      
      // Check for unreplaced variables
      const unreplacedMatches = preview.match(/\{([^}]+)\}/g);
      if (unreplacedMatches) {
        const unreplacedVars = unreplacedMatches.map(match => match.slice(1, -1));
        setPreviewError(`Unresolved variables: ${unreplacedVars.join(', ')}`);
      }
      
      setPreviewContent(preview);
    } catch (error) {
      setPreviewError('Error generating preview: ' + (error instanceof Error ? error.message : 'Unknown error'));
    } finally {
      setIsProcessing(false);
    }
  };

  const handleVariableChange = (variable: string, value: string) => {
    setVariableValues(prev => ({
      ...prev,
      [variable]: value
    }));
  };

  const handleCopyPreview = async () => {
    try {
      await navigator.clipboard.writeText(previewContent);
      // Could add a toast notification here
    } catch (error) {
      console.error('Failed to copy:', error);
    }
  };

  const handleDownloadPreview = () => {
    const blob = new Blob([previewContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${template.name}_preview.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const resetToSample = () => {
    const sampleData = getSampleData();
    const resetValues: Record<string, string> = {};
    
    template.variables.forEach(variable => {
      resetValues[variable] = sampleData[variable] || `[${variable}]`;
    });
    
    setVariableValues(resetValues);
  };

  return (
    <Box sx={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box>
            <Typography variant="h6" gutterBottom>
              Template Preview: {template.name}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {template.description}
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Tooltip title="Reset to sample data">
              <IconButton onClick={resetToSample} size="small">
                <RefreshIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title={showRaw ? "Show formatted view" : "Show raw template"}>
              <IconButton onClick={() => setShowRaw(!showRaw)} size="small">
                {showRaw ? <VisibilityIcon /> : <CodeIcon />}
              </IconButton>
            </Tooltip>
            <Tooltip title="Copy preview">
              <IconButton onClick={handleCopyPreview} size="small">
                <CopyIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Download preview">
              <IconButton onClick={handleDownloadPreview} size="small">
                <DownloadIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
      </Paper>

      <Grid container spacing={2} sx={{ flex: 1, overflow: 'hidden' }}>
        {/* Variable Inputs */}
        {showVariableInputs && template.variables.length > 0 && (
          <Grid item xs={12} md={4} sx={{ height: '100%' }}>
            <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <CardHeader
                title="Variable Values"
                subtitle={`${template.variables.length} variables`}
                action={
                  <Button
                    size="small"
                    startIcon={<PreviewIcon />}
                    onClick={generatePreview}
                    disabled={isProcessing}
                  >
                    {isProcessing ? <CircularProgress size={16} /> : 'Update Preview'}
                  </Button>
                }
              />
              <CardContent sx={{ flex: 1, overflow: 'auto', pt: 0 }}>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  {template.variables.map((variable) => (
                    <TextField
                      key={variable}
                      label={variable}
                      value={variableValues[variable] || ''}
                      onChange={(e) => handleVariableChange(variable, e.target.value)}
                      multiline
                      minRows={1}
                      maxRows={4}
                      size="small"
                      helperText={`Used in template as {${variable}}`}
                    />
                  ))}
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Preview Content */}
        <Grid item xs={12} md={showVariableInputs && template.variables.length > 0 ? 8 : 12} sx={{ height: '100%' }}>
          <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardHeader
              title={showRaw ? "Raw Template" : "Generated Preview"}
              action={
                <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                  <Chip
                    size="small"
                    label={`${template.variables.length} variables`}
                    color="primary"
                    variant="outlined"
                  />
                  <Chip
                    size="small"
                    label={previewContent.split(/\s+/).length + ' words'}
                    variant="outlined"
                  />
                </Box>
              }
            />
            <CardContent sx={{ flex: 1, overflow: 'auto', pt: 0 }}>
              {previewError && (
                <Alert severity="warning" sx={{ mb: 2 }}>
                  {previewError}
                </Alert>
              )}
              
              <Paper 
                variant="outlined" 
                sx={{ 
                  p: 2, 
                  height: '100%', 
                  overflow: 'auto',
                  backgroundColor: showRaw ? 'grey.50' : 'background.paper'
                }}
              >
                <Typography
                  component="pre"
                  sx={{
                    fontFamily: showRaw ? 'monospace' : 'inherit',
                    fontSize: showRaw ? '13px' : '14px',
                    lineHeight: 1.6,
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                    margin: 0,
                    color: showRaw ? 'text.secondary' : 'text.primary'
                  }}
                >
                  {showRaw ? template.content : previewContent}
                </Typography>
              </Paper>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default TemplatePreview;