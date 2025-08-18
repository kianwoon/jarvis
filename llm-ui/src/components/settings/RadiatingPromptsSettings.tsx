import React, { useState, useEffect } from 'react';
import { 
  Box, 
  TextField, 
  Button, 
  Typography, 
  Paper, 
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Alert,
  Snackbar,
  IconButton,
  Tooltip,
  Chip,
  Stack,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Save as SaveIcon,
  Refresh as RefreshIcon,
  Edit as EditIcon,
  ContentCopy as CopyIcon,
  RestartAlt as ResetIcon
} from '@mui/icons-material';

interface RadiatingPromptsSettingsProps {
  settings: any;
  onUpdate: (settings: any) => void;
}

interface PromptCategory {
  name: string;
  displayName: string;
  description: string;
  prompts: {
    [key: string]: {
      name: string;
      description: string;
      value: string;
    };
  };
}

const PROMPT_CATEGORIES: { [key: string]: { displayName: string; description: string } } = {
  entity_extraction: {
    displayName: 'Entity Extraction',
    description: 'Prompts for extracting entities from text using LLM intelligence'
  },
  relationship_discovery: {
    displayName: 'Relationship Discovery',
    description: 'Prompts for discovering relationships between entities'
  },
  query_analysis: {
    displayName: 'Query Analysis',
    description: 'Prompts for analyzing user queries and extracting intent'
  },
  expansion_strategy: {
    displayName: 'Expansion Strategy',
    description: 'Prompts for expanding queries with related concepts'
  }
};

const PROMPT_METADATA: { [category: string]: { [prompt: string]: { name: string; description: string } } } = {
  entity_extraction: {
    discovery_comprehensive: {
      name: 'Comprehensive Discovery',
      description: 'Discovers entity types for comprehensive technology queries'
    },
    discovery_regular: {
      name: 'Regular Discovery',
      description: 'Discovers entity types for standard queries'
    },
    extraction_comprehensive: {
      name: 'Comprehensive Extraction',
      description: 'Extracts entities comprehensively (30-50+ entities)'
    },
    extraction_regular: {
      name: 'Regular Extraction',
      description: 'Standard entity extraction from text'
    }
  },
  relationship_discovery: {
    llm_discovery: {
      name: 'LLM Discovery',
      description: 'Discovers relationships using LLM knowledge base'
    },
    relationship_analysis: {
      name: 'Relationship Analysis',
      description: 'Analyzes relationships between entities in context'
    },
    implicit_relationships: {
      name: 'Implicit Relationships',
      description: 'Infers implicit relationships from entity properties'
    }
  },
  query_analysis: {
    entity_extraction: {
      name: 'Entity Extraction',
      description: 'Extracts key entities from user queries'
    },
    intent_identification: {
      name: 'Intent Identification',
      description: 'Identifies the primary intent of queries'
    },
    domain_extraction: {
      name: 'Domain Extraction',
      description: 'Extracts knowledge domains from queries'
    },
    temporal_extraction: {
      name: 'Temporal Extraction',
      description: 'Extracts temporal context from queries'
    }
  },
  expansion_strategy: {
    semantic_expansion: {
      name: 'Semantic Expansion',
      description: 'Finds semantically related terms and entities'
    },
    concept_expansion: {
      name: 'Concept Expansion',
      description: 'Expands queries with related concepts'
    },
    hierarchical_expansion: {
      name: 'Hierarchical Expansion',
      description: 'Finds hierarchical relationships (parent/child/sibling)'
    }
  }
};

export const RadiatingPromptsSettings: React.FC<RadiatingPromptsSettingsProps> = ({
  settings,
  onUpdate
}) => {
  const [prompts, setPrompts] = useState<{ [key: string]: any }>({});
  const [editingPrompt, setEditingPrompt] = useState<{ category: string; prompt: string } | null>(null);
  const [editValue, setEditValue] = useState('');
  const [snackbar, setSnackbar] = useState<{ open: boolean; message: string; severity: 'success' | 'error' }>({
    open: false,
    message: '',
    severity: 'success'
  });
  const [expandedCategory, setExpandedCategory] = useState<string | false>(false);

  useEffect(() => {
    // Load prompts from settings
    if (settings?.prompts) {
      setPrompts(settings.prompts);
    }
  }, [settings]);

  const handleEditPrompt = (category: string, promptKey: string) => {
    setEditingPrompt({ category, prompt: promptKey });
    setEditValue(prompts[category]?.[promptKey] || '');
  };

  const handleSavePrompt = () => {
    if (!editingPrompt) return;

    const updatedPrompts = {
      ...prompts,
      [editingPrompt.category]: {
        ...prompts[editingPrompt.category],
        [editingPrompt.prompt]: editValue
      }
    };

    setPrompts(updatedPrompts);
    
    // Update ONLY the prompts field in settings, preserving all other fields
    // CRITICAL: Ensure model_config is ALWAYS present to prevent it from being lost
    const updatedSettings = {
      ...settings,
      prompts: updatedPrompts,
      // Explicitly preserve model_config if it exists
      model_config: settings.model_config || {
        model: 'llama3.1:8b',
        max_tokens: 4096,
        temperature: 0.7,
        context_length: 128000,
        model_server: 'http://localhost:11434',
        system_prompt: ''
      }
    };
    
    onUpdate(updatedSettings);
    setEditingPrompt(null);
    setSnackbar({
      open: true,
      message: 'Prompt updated successfully',
      severity: 'success'
    });
  };

  const handleCopyPrompt = (text: string) => {
    navigator.clipboard.writeText(text);
    setSnackbar({
      open: true,
      message: 'Prompt copied to clipboard',
      severity: 'success'
    });
  };

  const handleResetPrompt = (category: string, promptKey: string) => {
    // In a real implementation, this would fetch the default prompt from the server
    setSnackbar({
      open: true,
      message: 'Reset functionality requires server implementation',
      severity: 'error'
    });
  };

  const handleReloadAll = async () => {
    try {
      const response = await fetch('/api/v1/settings/radiating/reload-prompts', {
        method: 'POST'
      });
      
      if (response.ok) {
        const data = await response.json();
        setPrompts(data.prompts);
        setSnackbar({
          open: true,
          message: 'Prompts reloaded from database',
          severity: 'success'
        });
      }
    } catch (error) {
      setSnackbar({
        open: true,
        message: 'Failed to reload prompts',
        severity: 'error'
      });
    }
  };

  const getPromptPreview = (text: string, maxLength: number = 150) => {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  };

  const countPromptVariables = (text: string): string[] => {
    const matches = text.match(/\{([^}]+)\}/g);
    if (!matches) return [];
    return [...new Set(matches.map(m => m.slice(1, -1)))];
  };

  return (
    <Box sx={{ p: 2 }}>
      <Paper elevation={1} sx={{ p: 3, mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h5" gutterBottom>
            Radiating System Prompts
          </Typography>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={handleReloadAll}
          >
            Reload All
          </Button>
        </Box>
        
        <Alert severity="info" sx={{ mb: 2 }}>
          Configure LLM prompts used by the Radiating Coverage System. Changes take effect immediately without restart.
        </Alert>

        {Object.entries(PROMPT_CATEGORIES).map(([categoryKey, categoryInfo]) => (
          <Accordion
            key={categoryKey}
            expanded={expandedCategory === categoryKey}
            onChange={(_, isExpanded) => setExpandedCategory(isExpanded ? categoryKey : false)}
            sx={{ mb: 2 }}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                <Typography variant="h6">{categoryInfo.displayName}</Typography>
                <Chip
                  label={`${Object.keys(prompts[categoryKey] || {}).length} prompts`}
                  size="small"
                  sx={{ ml: 2 }}
                />
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                {categoryInfo.description}
              </Typography>
              
              {Object.entries(PROMPT_METADATA[categoryKey] || {}).map(([promptKey, promptInfo]) => {
                const promptValue = prompts[categoryKey]?.[promptKey] || '';
                const variables = countPromptVariables(promptValue);
                
                return (
                  <Paper key={promptKey} variant="outlined" sx={{ p: 2, mb: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                      <Box>
                        <Typography variant="subtitle1" fontWeight="bold">
                          {promptInfo.name}
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          {promptInfo.description}
                        </Typography>
                        {variables.length > 0 && (
                          <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
                            <Typography variant="caption" color="textSecondary">
                              Variables:
                            </Typography>
                            {variables.map(v => (
                              <Chip key={v} label={`{${v}}`} size="small" variant="outlined" />
                            ))}
                          </Stack>
                        )}
                      </Box>
                      <Box>
                        <Tooltip title="Edit">
                          <IconButton
                            size="small"
                            onClick={() => handleEditPrompt(categoryKey, promptKey)}
                          >
                            <EditIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Copy">
                          <IconButton
                            size="small"
                            onClick={() => handleCopyPrompt(promptValue)}
                          >
                            <CopyIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Reset to Default">
                          <IconButton
                            size="small"
                            onClick={() => handleResetPrompt(categoryKey, promptKey)}
                          >
                            <ResetIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </Box>
                    
                    <Paper sx={{ 
                      p: 2, 
                      backgroundColor: (theme) => theme.palette.mode === 'dark' ? theme.palette.grey[900] : theme.palette.grey[100],
                      color: (theme) => theme.palette.mode === 'dark' ? theme.palette.grey[100] : theme.palette.grey[900]
                    }}>
                      <Typography
                        variant="body2"
                        sx={{
                          fontFamily: 'monospace',
                          whiteSpace: 'pre-wrap',
                          wordBreak: 'break-word',
                          color: 'inherit'
                        }}
                      >
                        {getPromptPreview(promptValue, 300)}
                      </Typography>
                    </Paper>
                  </Paper>
                );
              })}
            </AccordionDetails>
          </Accordion>
        ))}
      </Paper>

      {/* Edit Dialog */}
      <Dialog
        open={editingPrompt !== null}
        onClose={() => setEditingPrompt(null)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Edit Prompt: {editingPrompt && PROMPT_METADATA[editingPrompt.category]?.[editingPrompt.prompt]?.name}
        </DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            multiline
            rows={15}
            value={editValue}
            onChange={(e) => setEditValue(e.target.value)}
            variant="outlined"
            sx={{ 
              mt: 2, 
              '& textarea': {
                fontFamily: 'monospace'
              },
              '& .MuiInputBase-root': {
                backgroundColor: (theme) => theme.palette.mode === 'dark' ? theme.palette.grey[900] : theme.palette.background.paper,
                color: (theme) => theme.palette.text.primary
              },
              '& .MuiOutlinedInput-root': {
                '& fieldset': {
                  borderColor: (theme) => theme.palette.mode === 'dark' ? theme.palette.grey[700] : theme.palette.divider
                },
                '&:hover fieldset': {
                  borderColor: (theme) => theme.palette.mode === 'dark' ? theme.palette.grey[600] : theme.palette.action.hover
                }
              }
            }}
            helperText="Use {variable_name} for template variables that will be replaced at runtime"
          />
          {editValue && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="caption" color="textSecondary">
                Character count: {editValue.length} | Variables: {countPromptVariables(editValue).join(', ') || 'None'}
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditingPrompt(null)}>Cancel</Button>
          <Button
            onClick={handleSavePrompt}
            variant="contained"
            startIcon={<SaveIcon />}
          >
            Save
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={3000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
        message={snackbar.message}
      />
    </Box>
  );
};

export default RadiatingPromptsSettings;