import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  TextField,
  InputAdornment,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Tooltip,
  Divider
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Search as SearchIcon,
  Add as AddIcon,
  Info as InfoIcon,
  Code as CodeIcon
} from '@mui/icons-material';

interface VariableDefinition {
  name: string;
  description: string;
  type: 'string' | 'number' | 'boolean' | 'array' | 'object';
  example?: string;
  required?: boolean;
  category: string;
}

interface TemplateVariablePanelProps {
  category: 'synthesis_prompts' | 'formatting_templates' | 'system_behaviors';
  onVariableInsert?: (variableName: string) => void;
  onVariableInfo?: (variable: VariableDefinition) => void;
  searchable?: boolean;
  compact?: boolean;
}

const TemplateVariablePanel: React.FC<TemplateVariablePanelProps> = ({
  category,
  onVariableInsert,
  onVariableInfo,
  searchable = true,
  compact = false
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedCategories, setExpandedCategories] = useState<string[]>(['core']);

  // Define available variables based on category
  const getVariablesByCategory = (): Record<string, VariableDefinition[]> => {
    switch (category) {
      case 'synthesis_prompts':
        return {
          'Core Content': [
            {
              name: 'topic',
              description: 'Main topic or subject being synthesized',
              type: 'string',
              example: 'Climate change impacts on agriculture',
              required: true,
              category: 'core'
            },
            {
              name: 'user_query',
              description: 'Original user question or request',
              type: 'string',
              example: 'What are the effects of climate change on crop yields?',
              required: true,
              category: 'core'
            },
            {
              name: 'document_content',
              description: 'Aggregated content from retrieved documents',
              type: 'string',
              example: 'Research indicates that rising temperatures...',
              required: true,
              category: 'core'
            },
            {
              name: 'key_themes',
              description: 'Main themes extracted from source documents',
              type: 'array',
              example: 'Temperature rise, drought patterns, adaptation strategies',
              category: 'core'
            }
          ],
          'Metadata': [
            {
              name: 'source_count',
              description: 'Number of source documents used',
              type: 'number',
              example: '15',
              category: 'metadata'
            },
            {
              name: 'confidence_score',
              description: 'Overall confidence in the synthesis',
              type: 'number',
              example: '0.87',
              category: 'metadata'
            },
            {
              name: 'timestamp',
              description: 'When the synthesis was generated',
              type: 'string',
              example: '2024-01-15 14:30:22',
              category: 'metadata'
            },
            {
              name: 'processing_time',
              description: 'Time taken to generate synthesis',
              type: 'string',
              example: '2.3 seconds',
              category: 'metadata'
            }
          ],
          'User Context': [
            {
              name: 'audience_level',
              description: 'Target audience expertise level',
              type: 'string',
              example: 'academic, general, expert',
              category: 'context'
            },
            {
              name: 'output_length',
              description: 'Desired length of output',
              type: 'string',
              example: 'brief, detailed, comprehensive',
              category: 'context'
            },
            {
              name: 'focus_areas',
              description: 'Specific areas to emphasize',
              type: 'array',
              example: 'economic impacts, environmental effects',
              category: 'context'
            }
          ]
        };

      case 'formatting_templates':
        return {
          'Structure': [
            {
              name: 'section_title',
              description: 'Title for the current section',
              type: 'string',
              example: 'Executive Summary',
              required: true,
              category: 'structure'
            },
            {
              name: 'summary',
              description: 'Brief summary of the content',
              type: 'string',
              example: 'This section covers the main findings...',
              category: 'structure'
            },
            {
              name: 'formatted_points',
              description: 'Content formatted as bullet points or list',
              type: 'string',
              example: '• Point 1\n• Point 2\n• Point 3',
              category: 'structure'
            }
          ],
          'Citations': [
            {
              name: 'source_list',
              description: 'Formatted list of sources used',
              type: 'string',
              example: '[1] Smith et al. (2023), [2] Johnson (2022)',
              category: 'citations'
            },
            {
              name: 'citation_style',
              description: 'Citation format style',
              type: 'string',
              example: 'APA, MLA, Chicago',
              category: 'citations'
            },
            {
              name: 'reference_count',
              description: 'Number of references cited',
              type: 'number',
              example: '12',
              category: 'citations'
            }
          ]
        };

      case 'system_behaviors':
        return {
          'Processing': [
            {
              name: 'request_type',
              description: 'Type of request being processed',
              type: 'string',
              example: 'research_query, summarization, analysis',
              required: true,
              category: 'processing'
            },
            {
              name: 'reasoning_mode',
              description: 'Reasoning approach to use',
              type: 'string',
              example: 'analytical, creative, systematic',
              category: 'processing'
            },
            {
              name: 'confidence_threshold',
              description: 'Minimum confidence level required',
              type: 'number',
              example: '0.75',
              category: 'processing'
            }
          ],
          'Quality Control': [
            {
              name: 'verification_required',
              description: 'Whether verification steps are needed',
              type: 'boolean',
              example: 'true, false',
              category: 'quality'
            },
            {
              name: 'fact_check_level',
              description: 'Level of fact-checking to apply',
              type: 'string',
              example: 'basic, thorough, comprehensive',
              category: 'quality'
            },
            {
              name: 'fallback_strategy',
              description: 'What to do if primary processing fails',
              type: 'string',
              example: 'use_cache, simplified_response, error_message',
              category: 'quality'
            }
          ]
        };

      default:
        return {};
    }
  };

  const variables = getVariablesByCategory();
  
  // Filter variables based on search term
  const filteredVariables = Object.entries(variables).reduce((acc, [categoryName, vars]) => {
    const filtered = vars.filter(variable => 
      variable.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      variable.description.toLowerCase().includes(searchTerm.toLowerCase())
    );
    if (filtered.length > 0) {
      acc[categoryName] = filtered;
    }
    return acc;
  }, {} as Record<string, VariableDefinition[]>);

  const handleCategoryToggle = (categoryName: string) => {
    setExpandedCategories(prev => 
      prev.includes(categoryName)
        ? prev.filter(name => name !== categoryName)
        : [...prev, categoryName]
    );
  };

  const handleVariableClick = (variable: VariableDefinition) => {
    if (onVariableInsert) {
      onVariableInsert(variable.name);
    }
  };

  const getVariableTypeColor = (type: string) => {
    switch (type) {
      case 'string': return 'primary';
      case 'number': return 'secondary';
      case 'boolean': return 'success';
      case 'array': return 'warning';
      case 'object': return 'info';
      default: return 'default';
    }
  };

  if (compact) {
    // Compact view - just show variables as clickable chips
    const allVariables = Object.values(filteredVariables).flat();
    
    return (
      <Box sx={{ width: '100%' }}>
        {searchable && (
          <TextField
            fullWidth
            size="small"
            placeholder="Search variables..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon />
                </InputAdornment>
              ),
            }}
            sx={{ mb: 2 }}
          />
        )}
        
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
          {allVariables.map((variable) => (
            <Tooltip 
              key={variable.name}
              title={`${variable.description} (${variable.type})`}
              arrow
            >
              <Chip
                label={variable.name}
                size="small"
                clickable
                onClick={() => handleVariableClick(variable)}
                color={getVariableTypeColor(variable.type) as any}
                variant={variable.required ? 'filled' : 'outlined'}
                icon={<CodeIcon />}
                sx={{ 
                  cursor: 'pointer',
                  '&:hover': {
                    backgroundColor: 'primary.light',
                    color: 'primary.contrastText'
                  }
                }}
              />
            </Tooltip>
          ))}
        </Box>
        
        {allVariables.length === 0 && searchTerm && (
          <Typography variant="body2" color="text.secondary" sx={{ mt: 2, textAlign: 'center' }}>
            No variables found matching "{searchTerm}"
          </Typography>
        )}
      </Box>
    );
  }

  // Full view with categories and detailed information
  return (
    <Box sx={{ width: '100%', maxHeight: '60vh', overflow: 'auto' }}>
      <Box sx={{ p: 2 }}>
        {searchable && (
          <TextField
            fullWidth
            size="small"
            placeholder="Search variables..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon sx={{ color: 'text.secondary' }} />
                </InputAdornment>
              ),
            }}
            sx={{ 
              '& .MuiOutlinedInput-root': {
                borderRadius: 2,
                bgcolor: theme => theme.palette.mode === 'dark'
                  ? 'rgba(255, 255, 255, 0.05)'
                  : 'rgba(0, 0, 0, 0.02)',
                '&:hover .MuiOutlinedInput-notchedOutline': {
                  borderColor: 'primary.main'
                },
                '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                  borderWidth: 2
                }
              }
            }}
          />
        )}
      </Box>

      <Box sx={{ px: 1, pb: 2 }}>
        {Object.entries(filteredVariables).map(([categoryName, categoryVariables]) => (
          <Accordion
            key={categoryName}
            expanded={expandedCategories.includes(categoryVariables[0]?.category || categoryName)}
            onChange={() => handleCategoryToggle(categoryVariables[0]?.category || categoryName)}
          >
            <AccordionSummary 
              expandIcon={<ExpandMoreIcon />}
              sx={{
                minHeight: 48,
                '&.Mui-expanded': {
                  minHeight: 48
                },
                '& .MuiAccordionSummary-content': {
                  margin: '8px 0',
                  '&.Mui-expanded': {
                    margin: '8px 0'
                  }
                }
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                <Typography 
                  variant="subtitle1" 
                  sx={{ 
                    fontWeight: 600,
                    color: 'text.primary'
                  }}
                >
                  {categoryName}
                </Typography>
                <Chip 
                  size="small" 
                  label={categoryVariables.length}
                  sx={{
                    bgcolor: 'primary.main',
                    color: 'primary.contrastText',
                    fontWeight: 600,
                    '& .MuiChip-label': {
                      px: 1
                    }
                  }}
                />
              </Box>
            </AccordionSummary>
            <AccordionDetails sx={{ pt: 0, px: 2, pb: 2 }}>
              <List dense>
                {categoryVariables.map((variable, index) => (
                  <React.Fragment key={variable.name}>
                    {index > 0 && <Divider />}
                    <ListItem
                      button
                      onClick={() => handleVariableClick(variable)}
                      sx={{ 
                        borderRadius: 2,
                        mb: 0.5,
                        transition: 'all 0.2s ease',
                        '&:hover': {
                          backgroundColor: 'action.hover',
                          transform: 'translateX(4px)',
                          boxShadow: theme => theme.shadows[1]
                        },
                        '&:active': {
                          transform: 'translateX(2px)'
                        }
                      }}
                    >
                      <ListItemText
                        primary={
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Typography 
                              variant="body2" 
                              sx={{ 
                                fontFamily: 'Roboto Mono, monospace',
                                fontWeight: 600,
                                color: 'primary.main',
                                fontSize: '0.875rem'
                              }}
                            >
                              {variable.name}
                            </Typography>
                            <Chip
                              size="small"
                              label={variable.type}
                              color={getVariableTypeColor(variable.type) as any}
                              sx={{ 
                                fontSize: '0.75rem', 
                                height: 22,
                                fontWeight: 500,
                                '& .MuiChip-label': {
                                  px: 1
                                }
                              }}
                            />
                            {variable.required && (
                              <Chip
                                size="small"
                                label="required"
                                color="error"
                                variant="outlined"
                                sx={{ fontSize: '10px', height: '20px' }}
                              />
                            )}
                          </Box>
                        }
                        secondary={
                          <Box>
                            <Typography variant="caption" color="text.secondary">
                              {variable.description}
                            </Typography>
                            {variable.example && (
                              <Typography variant="caption" sx={{ display: 'block', fontStyle: 'italic', mt: 0.5 }}>
                                Example: {variable.example}
                              </Typography>
                            )}
                          </Box>
                        }
                      />
                      <ListItemSecondaryAction>
                        <Tooltip title="Click to insert variable">
                          <IconButton edge="end" size="small">
                            <AddIcon />
                          </IconButton>
                        </Tooltip>
                        {onVariableInfo && (
                          <Tooltip title="More information">
                            <IconButton 
                              edge="end" 
                              size="small"
                              onClick={(e) => {
                                e.stopPropagation();
                                onVariableInfo(variable);
                              }}
                            >
                              <InfoIcon />
                            </IconButton>
                          </Tooltip>
                        )}
                      </ListItemSecondaryAction>
                    </ListItem>
                  </React.Fragment>
                ))}
              </List>
            </AccordionDetails>
          </Accordion>
        ))}
        
        {Object.keys(filteredVariables).length === 0 && (
          <Box 
            sx={{ 
              p: 4, 
              textAlign: 'center',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: 2
            }}
          >
            <SearchIcon 
              sx={{ 
                fontSize: '2.5rem',
                color: 'text.disabled',
                opacity: 0.5
              }} 
            />
            <Typography 
              variant="body1" 
              sx={{ 
                color: 'text.secondary',
                fontWeight: 500
              }}
            >
              {searchTerm ? `No variables found matching "${searchTerm}"` : 'No variables available'}
            </Typography>
            {searchTerm && (
              <Typography 
                variant="body2" 
                sx={{ 
                  color: 'text.disabled'
                }}
              >
                Try a different search term
              </Typography>
            )}
          </Box>
        )}
      </Box>
    </Box>
  );
};

export default TemplateVariablePanel;