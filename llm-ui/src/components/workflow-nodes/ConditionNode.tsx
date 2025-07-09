import React, { useState, useEffect } from 'react';
import { Handle, Position } from 'reactflow';
import { 
  Typography, 
  Box, 
  TextField, 
  FormControl, 
  InputLabel, 
  Select, 
  MenuItem, 
  Accordion, 
  AccordionSummary, 
  AccordionDetails,
  Card,
  CardContent,
  Badge
} from '@mui/material';
import { 
  ExpandMore as ExpandMoreIcon,
  AccountTree as ConditionIcon,
  CheckCircle as TrueIcon,
  Cancel as FalseIcon
} from '@mui/icons-material';

interface ConditionNodeProps {
  data: {
    label?: string;
    condition_type?: 'simple' | 'ai_decision' | 'custom';
    operator?: 'equals' | 'not_equals' | 'contains' | 'greater_than' | 'less_than';
    compare_value?: string;
    ai_criteria?: string;
    // Status information
    status?: 'idle' | 'running' | 'success' | 'error';
    result?: boolean;
    executionData?: any;
  };
  id: string;
  updateNodeData?: (nodeId: string, newData: any) => void;
}

const ConditionNode: React.FC<ConditionNodeProps> = ({ data, id, updateNodeData }) => {
  const [expanded, setExpanded] = useState(false);
  const [conditionType, setConditionType] = useState(data.condition_type || 'simple');
  const [operator, setOperator] = useState(data.operator || 'equals');
  const [compareValue, setCompareValue] = useState(data.compare_value || '');
  const [aiCriteria, setAiCriteria] = useState(data.ai_criteria || '');
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [titleValue, setTitleValue] = useState(data.label || '');

  // Sync with data changes
  useEffect(() => {
    if (data.condition_type !== undefined) setConditionType(data.condition_type);
    if (data.operator !== undefined) setOperator(data.operator);
    if (data.compare_value !== undefined) setCompareValue(data.compare_value);
    if (data.ai_criteria !== undefined) setAiCriteria(data.ai_criteria);
    if (data.label !== undefined) setTitleValue(data.label);
  }, [data]);

  const handleConditionTypeChange = (type: string) => {
    setConditionType(type as 'simple' | 'ai_decision' | 'custom');
    updateNodeData?.(id, { condition_type: type });
  };

  const handleOperatorChange = (op: string) => {
    setOperator(op as 'equals' | 'not_equals' | 'contains' | 'greater_than' | 'less_than');
    updateNodeData?.(id, { operator: op });
  };

  const handleCompareValueChange = (value: string) => {
    setCompareValue(value);
    updateNodeData?.(id, { compare_value: value });
  };

  const handleAiCriteriaChange = (criteria: string) => {
    setAiCriteria(criteria);
    updateNodeData?.(id, { ai_criteria: criteria });
  };

  const handleTitleChange = (newTitle: string) => {
    setTitleValue(newTitle);
    updateNodeData?.(id, { label: newTitle });
  };

  // Get status color and icon
  const getStatusInfo = () => {
    if (data.status === 'running') {
      return { color: '#ff9800', icon: <ConditionIcon />, label: 'Evaluating' };
    } else if (data.status === 'success') {
      const result = data.result;
      return { 
        color: result ? '#4caf50' : '#f44336', 
        icon: result ? <TrueIcon /> : <FalseIcon />,
        label: result ? 'True' : 'False'
      };
    } else if (data.status === 'error') {
      return { color: '#f44336', icon: <FalseIcon />, label: 'Error' };
    }
    return { color: '#ff9800', icon: <ConditionIcon />, label: 'Idle' };
  };

  const statusInfo = getStatusInfo();

  // Generate condition summary
  const getConditionSummary = () => {
    switch (conditionType) {
      case 'simple':
        return `${operator} "${compareValue || '...'}"`;
      case 'ai_decision':
        return aiCriteria ? aiCriteria.substring(0, 30) + '...' : 'AI decision';
      case 'custom':
        return 'Custom logic';
      default:
        return 'Not configured';
    }
  };

  return (
    <Badge
      badgeContent={data.result !== undefined ? (data.result ? 'T' : 'F') : ''}
      color={data.result ? 'success' : 'error'}
      overlap="circular"
      anchorOrigin={{
        vertical: 'top',
        horizontal: 'right',
      }}
      invisible={data.result === undefined}
      sx={{
        '& .MuiBadge-badge': {
          fontSize: '0.75rem',
          fontWeight: 600,
          minWidth: '20px',
          height: '20px',
          borderRadius: '10px',
        }
      }}
    >
      <Card 
        sx={{ 
          minWidth: 280,
          maxWidth: 350,
          border: `2px solid ${statusInfo.color}`,
          borderRadius: 2,
          bgcolor: 'background.paper'
        }}
      >
        <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
          {/* Header */}
          <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
            <Box display="flex" alignItems="center" gap={1}>
              <Box sx={{ color: statusInfo.color }}>
                {statusInfo.icon}
              </Box>
              <Box>
                {!isEditingTitle ? (
                  <Typography 
                    variant="subtitle2" 
                    onClick={() => setIsEditingTitle(true)}
                    sx={{ 
                      fontWeight: 600, 
                      color: statusInfo.color,
                      cursor: 'pointer',
                      '&:hover': { opacity: 0.8 }
                    }}
                  >
                    {data.label || 'Condition'}
                  </Typography>
                ) : (
                  <TextField
                    value={titleValue}
                    onChange={(e) => setTitleValue(e.target.value)}
                    onBlur={() => {
                      if (titleValue !== data.label) {
                        handleTitleChange(titleValue);
                      }
                      setIsEditingTitle(false);
                    }}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        if (titleValue !== data.label) {
                          handleTitleChange(titleValue);
                        }
                        setIsEditingTitle(false);
                      } else if (e.key === 'Escape') {
                        setTitleValue(data.label || '');
                        setIsEditingTitle(false);
                      }
                    }}
                    autoFocus
                    size="small"
                    className="nodrag"
                    sx={{ '& .MuiInputBase-input': { fontSize: '0.875rem', fontWeight: 600 } }}
                  />
                )}
                <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.65rem' }}>
                  ({id})
                </Typography>
              </Box>
            </Box>
            {data.status && (
              <Typography variant="caption" sx={{ color: statusInfo.color, fontSize: '0.7rem' }}>
                {statusInfo.label}
              </Typography>
            )}
          </Box>

          {/* Condition Summary */}
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" sx={{ color: 'text.secondary', fontSize: '0.8rem' }}>
              {getConditionSummary()}
            </Typography>
          </Box>

          {/* Configuration */}
          <Box sx={{ mb: 1 }}>
            <div className="nodrag">
              <FormControl fullWidth size="small" sx={{ mb: 1 }}>
                <InputLabel>Condition Type</InputLabel>
                <Select
                  value={conditionType}
                  onChange={(e) => handleConditionTypeChange(e.target.value)}
                  label="Condition Type"
                  sx={{ fontSize: '0.875rem' }}
                >
                  <MenuItem value="simple">Simple Comparison</MenuItem>
                  <MenuItem value="ai_decision">AI-Based Decision</MenuItem>
                  <MenuItem value="custom">Custom Logic</MenuItem>
                </Select>
              </FormControl>

              {/* Simple Condition Fields */}
              {conditionType === 'simple' && (
                <Box>
                  <FormControl fullWidth size="small" sx={{ mb: 1 }}>
                    <InputLabel>Operator</InputLabel>
                    <Select
                      value={operator}
                      onChange={(e) => handleOperatorChange(e.target.value)}
                      label="Operator"
                      sx={{ fontSize: '0.875rem' }}
                    >
                      <MenuItem value="equals">Equals</MenuItem>
                      <MenuItem value="not_equals">Not Equals</MenuItem>
                      <MenuItem value="contains">Contains</MenuItem>
                      <MenuItem value="greater_than">Greater Than</MenuItem>
                      <MenuItem value="less_than">Less Than</MenuItem>
                    </Select>
                  </FormControl>
                  <TextField
                    fullWidth
                    size="small"
                    label="Compare Value"
                    value={compareValue}
                    onChange={(e) => handleCompareValueChange(e.target.value)}
                    placeholder="Value to compare against"
                    sx={{ fontSize: '0.875rem' }}
                  />
                </Box>
              )}

              {/* AI Decision Field */}
              {conditionType === 'ai_decision' && (
                <TextField
                  fullWidth
                  size="small"
                  label="Decision Criteria"
                  multiline
                  rows={3}
                  value={aiCriteria}
                  onChange={(e) => handleAiCriteriaChange(e.target.value)}
                  placeholder="Describe what the AI should evaluate..."
                  sx={{ fontSize: '0.875rem' }}
                />
              )}

              {/* Custom Logic Placeholder */}
              {conditionType === 'custom' && (
                <Box sx={{ 
                  p: 2, 
                  bgcolor: 'rgba(255, 152, 0, 0.1)', 
                  borderRadius: 1,
                  border: '1px dashed #ff9800'
                }}>
                  <Typography variant="body2" sx={{ color: '#ff9800', fontStyle: 'italic' }}>
                    Custom logic configuration will be available in a future update
                  </Typography>
                </Box>
              )}
            </div>
          </Box>

          {/* Execution Details */}
          {data.executionData && (
            <Accordion 
              expanded={expanded} 
              onChange={() => setExpanded(!expanded)}
              sx={{ 
                boxShadow: 'none', 
                '&:before': { display: 'none' },
                bgcolor: 'transparent'
              }}
            >
              <AccordionSummary
                expandIcon={<ExpandMoreIcon />}
                sx={{ 
                  p: 0, 
                  minHeight: 32, 
                  '& .MuiAccordionSummary-content': { my: 0.5 }
                }}
                className="nodrag"
              >
                <Typography variant="caption" color="text.secondary">
                  Execution Details
                </Typography>
              </AccordionSummary>
              <AccordionDetails sx={{ p: 0, pt: 1 }}>
                <Box sx={{ 
                  p: 1, 
                  bgcolor: 'rgba(0, 0, 0, 0.03)', 
                  borderRadius: 1,
                  fontSize: '0.75rem'
                }}>
                  <Typography variant="caption" display="block">
                    Input: {JSON.stringify(data.executionData.input)}
                  </Typography>
                  <Typography variant="caption" display="block">
                    Result: {data.result ? 'True' : 'False'}
                  </Typography>
                  {data.executionData.timestamp && (
                    <Typography variant="caption" display="block" sx={{ color: 'text.secondary' }}>
                      {new Date(data.executionData.timestamp).toLocaleTimeString()}
                    </Typography>
                  )}
                </Box>
              </AccordionDetails>
            </Accordion>
          )}
        </CardContent>

        {/* Handles */}
        <Handle 
          type="target" 
          position={Position.Top} 
          style={{ background: statusInfo.color }} 
        />
        <Handle 
          type="source" 
          position={Position.Left} 
          id="false"
          style={{ background: '#f44336', left: -6 }} 
        />
        <Handle 
          type="source" 
          position={Position.Right} 
          id="true"
          style={{ background: '#4caf50', right: -6 }} 
        />
      </Card>
    </Badge>
  );
};

export default ConditionNode;