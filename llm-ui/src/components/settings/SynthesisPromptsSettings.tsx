import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  TextField,
  Alert,
  Card,
  CardContent,
  IconButton,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Paper,
  Divider,
  Stack,
  Collapse
} from '@mui/material';
import {
  Edit as EditIcon,
  Save as SaveIcon,
  Cancel as CancelIcon,
  ExpandMore as ExpandMoreIcon,
  Code as CodeIcon,
  TextFields as TextFieldsIcon,
  Numbers as NumbersIcon,
  Lock as LockIcon
} from '@mui/icons-material';

interface Template {
  id?: string;
  name: string;
  content: string;
  variables?: string[];
  active?: boolean;
  version?: string;
  metadata?: any;
}

interface SynthesisPromptsSettingsProps {
  data: any;
  onChange: (field: string, value: any) => void;
  onShowSuccess?: (message?: string) => void;
  onRefresh?: () => void;
}

const SynthesisPromptsSettings: React.FC<SynthesisPromptsSettingsProps> = ({
  data,
  onChange,
  onShowSuccess,
  onRefresh
}) => {
  const [templates, setTemplates] = useState<Template[]>([]);
  const [editingTemplates, setEditingTemplates] = useState<Record<string, Template>>({});
  const [expandedTemplates, setExpandedTemplates] = useState<Record<string, boolean>>({});
  const [error, setError] = useState<string>('');

  // Extract templates from data and convert from API format to UI format
  useEffect(() => {
    console.log('[SynthesisPrompts] useEffect triggered with data:', data);
    console.log('[SynthesisPrompts] data type:', typeof data);
    console.log('[SynthesisPrompts] data keys:', data ? Object.keys(data) : 'null');
    console.log('[SynthesisPrompts] Full data structure:', JSON.stringify(data, null, 2));
    
    if (!data) {
      console.log('[SynthesisPrompts] No data provided, clearing templates');
      setTemplates([]);
      return;
    }
    
    // Try different data formats that might be returned by the API
    let settingsData = null;
    
    if (data.settings) {
      // Format: { settings: { "tool_synthesis": {...} } }
      settingsData = data.settings;
      console.log('[SynthesisPrompts] Using data.settings format');
    } else if (data && typeof data === 'object' && !Array.isArray(data)) {
      // Format: { "tool_synthesis": {...} } - direct object
      settingsData = data;
      console.log('[SynthesisPrompts] Using direct data format');
    }
    
    if (!settingsData) {
      console.log('[SynthesisPrompts] No valid settings data found:', data);
      setError('No synthesis prompt settings found');
      setTemplates([]);
      return;
    }
    
    try {
      // Convert from API format: { "tool_synthesis": { content: "...", description: "..." } }
      // To UI format: [ { id: "tool_synthesis", name: "Tool Synthesis", content: "..." } ]
      const templatesArray = Object.entries(settingsData).map(([key, template]: [string, any]) => {
        console.log(`[SynthesisPrompts] Processing template ${key}:`, template);
        return {
          id: key,
          name: template.description || key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
          content: template.content || '',
          variables: template.variables || [],
          active: template.active !== false, // Default to true if not specified
          version: template.version || '1.0',
          metadata: template.metadata || {}
        };
      });
      
      console.log(`[SynthesisPrompts] Converted ${templatesArray.length} templates:`, templatesArray);
      setTemplates(templatesArray);
      setError(''); // Clear any previous errors
      
    } catch (error) {
      console.error('[SynthesisPrompts] Error processing templates:', error);
      setError('Failed to process template data');
      setTemplates([]);
    }
  }, [data]);


  const handleStartEdit = (template: Template) => {
    if (!template.id) return;
    setEditingTemplates(prev => ({ 
      ...prev, 
      [template.id!]: { ...template } 
    }));
  };

  const handleCancelEdit = (templateId: string) => {
    setEditingTemplates(prev => {
      const newState = { ...prev };
      delete newState[templateId];
      return newState;
    });
    
  };

  const handleToggleExpand = (templateId: string) => {
    setExpandedTemplates(prev => ({
      ...prev,
      [templateId]: !prev[templateId]
    }));
  };


  const handleSaveTemplate = (templateId: string) => {
    const editingTemplate = editingTemplates[templateId];
    if (!editingTemplate?.name.trim() || !editingTemplate?.content.trim()) {
      setError('Name and content are required');
      return;
    }
    
    try {
      const templateToSave = {
        ...editingTemplate,
        id: templateId
      };
      
      // Update existing template
      const updatedTemplates = templates.map(t => 
        t.id === templateId ? templateToSave : t
      );
      
      setTemplates(updatedTemplates);
      
      // Convert back to API format for saving
      const apiFormat = updatedTemplates.reduce((acc: Record<string, any>, template) => {
        acc[template.id!] = {
          content: template.content,
          description: template.name,
          variables: template.variables || [],
          active: template.active !== false,
          version: template.version || '1.0',
          metadata: template.metadata || {}
        };
        return acc;
      }, {} as Record<string, any>);
      
      onChange('settings', apiFormat);
      
      // Clear editing state
      setEditingTemplates(prev => {
        const newState = { ...prev };
        delete newState[templateId];
        return newState;
      });
      
      if (onShowSuccess) {
        onShowSuccess('System template updated');
      }
    } catch (error) {
      setError('Failed to save template');
    }
  };

  // Available variables for all templates
  const availableVariables = ['user_query', 'question', 'tool_context', 'context', 'rag_context', 'enhanced_question', 'response_text', 'topic', 'document_content'];
  
  const insertVariable = (templateId: string, variableName: string) => {
    const editingTemplate = editingTemplates[templateId];
    if (!editingTemplate) return;
    
    const variable = `{${variableName}}`;
    setEditingTemplates(prev => ({
      ...prev,
      [templateId]: {
        ...editingTemplate,
        content: editingTemplate.content + variable
      }
    }));
  };

  return (
    <Box sx={{ width: '100%', p: 3 }}>
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError('')}>
          {error}
        </Alert>
      )}

      {/* Header with Create Button */}
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        mb: 4,
        pb: 2,
        borderBottom: 1,
        borderColor: 'divider'
      }}>
        <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2 }}>
          <LockIcon sx={{ color: 'warning.main', mt: 0.5 }} />
          <Box>
            <Typography variant="h4" sx={{ fontWeight: 600, mb: 1 }}>
              System Synthesis Templates
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ mb: 1 }}>
              These are protected system templates critical for synthesis functionality. They can only be modified, not deleted.
            </Typography>
            <Typography variant="body2" color="warning.main" sx={{ fontWeight: 500 }}>
              Edit templates inline with full content visibility. Click any template name or content to edit directly.
            </Typography>
          </Box>
        </Box>
      </Box>

      {/* Templates List - Full Width */}
      {templates.length === 0 ? (
        <Paper sx={{ p: 6, textAlign: 'center' }}>
          <CodeIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h5" color="text.secondary" sx={{ mb: 2 }}>
            {!data ? 'Loading system templates...' : 'No system templates found'}
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
            {!data 
              ? 'Please wait while we load the system synthesis templates'
              : !data.settings 
                ? 'No synthesis prompt data available. Try refreshing the page.'
                : 'System synthesis templates will appear here when available'
            }
          </Typography>
          {onRefresh ? (
            <Button 
              variant="outlined" 
              onClick={onRefresh}
              size="large"
            >
              Refresh Data
            </Button>
          ) : null}
        </Paper>
      ) : (
        <Stack spacing={2}>
          {templates.map((template) => {
            const isEditing = editingTemplates[template.id!];
            const isExpanded = expandedTemplates[template.id!];
            const editingData = isEditing || template;
            
            return (
              <Card 
                key={template.id} 
                className={`template-card ${isEditing ? 'editing' : ''}`}
                sx={{ 
                  border: 1, 
                  borderColor: isEditing ? 'primary.main' : 'divider',
                  borderRadius: 2,
                  transition: 'all 0.2s',
                  '&:hover': { 
                    borderColor: isEditing ? 'primary.main' : 'primary.light',
                    boxShadow: 2
                  }
                }}
              >
                <CardContent sx={{ p: 0 }}>
                  {/* Template Header */}
                  <Box 
                    className="template-header"
                    sx={{ 
                      p: 3,
                      pb: 2,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      cursor: 'pointer'
                    }} 
                    onClick={() => handleToggleExpand(template.id!)}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'center', flex: 1 }}>
                      {isEditing ? (
                        <TextField
                          fullWidth
                          value={editingData.name}
                          onChange={(e) => setEditingTemplates(prev => ({
                            ...prev,
                            [template.id!]: { ...editingData, name: e.target.value }
                          }))}
                          onClick={(e) => e.stopPropagation()}
                          variant="outlined"
                          size="medium"
                          sx={{ mr: 2, maxWidth: 400 }}
                          placeholder="Template Name"
                        />
                      ) : (
                        <Typography 
                          variant="h6" 
                          className="template-name-editable"
                          sx={{ 
                            fontWeight: 600,
                            mr: 2,
                            cursor: 'pointer',
                            '&:hover': { color: 'primary.main' }
                          }} 
                          onClick={(e) => {
                            e.stopPropagation();
                            handleStartEdit(template);
                          }}
                        >
                          {template.name}
                        </Typography>
                      )}
                      
                      <Chip 
                        icon={<NumbersIcon />}
                        label={`${editingData.content.length} chars`} 
                        size="small" 
                        variant="outlined"
                        className="template-stats"
                        sx={{ mr: 1 }}
                      />
                      
                      {template.variables && template.variables.length > 0 && (
                        <Chip 
                          icon={<TextFieldsIcon />}
                          label={`${template.variables.length} vars`} 
                          size="small" 
                          variant="outlined"
                          className="template-stats"
                        />
                      )}
                    </Box>
                    
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {isEditing ? (
                        <>
                          <Button
                            size="small"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleCancelEdit(template.id!);
                            }}
                            startIcon={<CancelIcon />}
                          >
                            Cancel
                          </Button>
                          <Button
                            variant="contained"
                            size="small"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleSaveTemplate(template.id!);
                            }}
                            startIcon={<SaveIcon />}
                            disabled={!editingData.name.trim() || !editingData.content.trim()}
                          >
                            Save
                          </Button>
                        </>
                      ) : (
                        <>
                          <IconButton
                            size="small"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleStartEdit(template);
                            }}
                            sx={{ '&:hover': { color: 'primary.main' } }}
                            title="Edit template (system template cannot be deleted)"
                          >
                            <EditIcon />
                          </IconButton>
                          <LockIcon 
                            sx={{ 
                              fontSize: 20, 
                              color: 'warning.main', 
                              ml: 1 
                            }} 
                            title="Protected system template"
                          />
                        </>
                      )}
                      
                      <IconButton size="small">
                        <ExpandMoreIcon className={`expand-icon ${isExpanded ? 'expanded' : ''}`} />
                      </IconButton>
                    </Box>
                  </Box>
                  
                  {/* Template Content */}
                  <Collapse in={isExpanded}>
                    <Divider />
                    <Box sx={{ p: 3 }}>
                      {isEditing ? (
                        <>
                          <TextField
                            fullWidth
                            multiline
                            rows={8}
                            value={editingData.content}
                            onChange={(e) => setEditingTemplates(prev => ({
                              ...prev,
                              [template.id!]: { ...editingData, content: e.target.value }
                            }))}
                            placeholder="Enter your template content here. Use variables like {user_query}, {context}, {topic}..."
                            variant="outlined"
                            className="template-editor-textarea"
                            sx={{ 
                              mb: 3,
                              '& .MuiInputBase-input': {
                                fontFamily: 'SF Mono, Monaco, Consolas, "Lucida Console", monospace',
                                fontSize: '0.875rem',
                                lineHeight: 1.6
                              }
                            }}
                          />
                          
                          {/* Variable Insertion */}
                          <Box sx={{ mb: 2 }}>
                            <Typography variant="subtitle2" sx={{ mb: 1, display: 'flex', alignItems: 'center' }}>
                              <CodeIcon sx={{ mr: 1, fontSize: 18 }} />
                              Quick Insert Variables:
                            </Typography>
                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                              {availableVariables.map(variable => (
                                <Chip
                                  key={variable}
                                  label={variable}
                                  size="small"
                                  onClick={() => insertVariable(template.id!, variable)}
                                  className="variable-chip"
                                  sx={{ 
                                    cursor: 'pointer',
                                    '&:hover': { 
                                      backgroundColor: 'primary.main',
                                      color: 'primary.contrastText'
                                    }
                                  }}
                                />
                              ))}
                            </Box>
                          </Box>
                        </>
                      ) : (
                        <Paper 
                          variant="outlined"
                          className="template-content-preview"
                          sx={{ 
                            p: 2,
                            backgroundColor: 'background.default',
                            cursor: 'pointer',
                            transition: 'all 0.2s',
                            '&:hover': { 
                              backgroundColor: 'action.hover',
                              borderColor: 'primary.main'
                            }
                          }}
                          onClick={() => handleStartEdit(template)}
                        >
                          <Typography 
                            variant="body2"
                            className="template-code-font"
                            sx={{ 
                              whiteSpace: 'pre-wrap',
                              fontSize: '0.875rem',
                              lineHeight: 1.6,
                              color: 'text.primary'
                            }}
                          >
                            {template.content || 'Click to add template content...'}
                          </Typography>
                        </Paper>
                      )}
                    </Box>
                  </Collapse>
                </CardContent>
              </Card>
            );
          })}
        </Stack>
      )}
    </Box>
  );
};

export default SynthesisPromptsSettings;