import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  Chip,
  LinearProgress,
  Alert,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import InfoIcon from '@mui/icons-material/Info';

interface WorkflowResult {
  nodeId: string;
  nodeName: string;
  status: 'success' | 'error' | 'warning' | 'info';
  message?: string;
  output?: any;
  duration?: number;
}

interface EnhancedWorkflowCompletionDialogProps {
  open: boolean;
  onViewNow?: () => void;
  onSaveOnly?: () => void;
  onViewAndSave?: () => void;
  onViewLater?: () => void;
  onDismiss?: () => void;
  workflowName?: string;
  outputFormat?: string;
  executionTime?: number;
  metadata?: any;
  // Legacy props for backward compatibility
  onClose?: () => void;
  results?: WorkflowResult[];
  overallStatus?: 'success' | 'error' | 'partial';
  totalDuration?: number;
  onDownloadResults?: () => void;
  onRetry?: () => void;
}

const getStatusIcon = (status: string) => {
  switch (status) {
    case 'success':
      return <CheckCircleIcon color="success" />;
    case 'error':
      return <ErrorIcon color="error" />;
    case 'warning':
      return <ErrorIcon color="warning" />;
    default:
      return <InfoIcon color="info" />;
  }
};

const getStatusColor = (status: string) => {
  switch (status) {
    case 'success':
      return 'success';
    case 'error':
      return 'error';
    case 'warning':
      return 'warning';
    default:
      return 'info';
  }
};

const EnhancedWorkflowCompletionDialog: React.FC<EnhancedWorkflowCompletionDialogProps> = ({
  open,
  onViewNow,
  onSaveOnly,
  onViewAndSave,
  onViewLater,
  onDismiss,
  workflowName,
  outputFormat,
  executionTime,
  metadata,
  // Legacy props for backward compatibility
  onClose,
  results = [],
  overallStatus = 'success',
  totalDuration = 0,
  onDownloadResults,
  onRetry
}) => {
  const successCount = results.filter(r => r.status === 'success').length;
  const errorCount = results.filter(r => r.status === 'error').length;
  const completionRate = results.length > 0 ? (successCount / results.length) * 100 : 0;

  // Use workflow name and metadata if available, otherwise fall back to legacy props
  const displayName = workflowName || 'Workflow';
  const displayDuration = executionTime || totalDuration;
  const displayStatus = overallStatus || 'success';

  return (
    <Dialog
      open={open}
      onClose={onDismiss || onClose}
      maxWidth="md"
      fullWidth
    >
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Typography variant="h6">
            {displayName} - Execution Complete
          </Typography>
          <Chip 
            label={displayStatus.toUpperCase()} 
            color={getStatusColor(displayStatus) as any}
            variant="outlined"
          />
        </Box>
      </DialogTitle>
      
      <DialogContent>
        {/* Workflow Summary */}
        <Box sx={{ mb: 3 }}>
          <Typography variant="body1" sx={{ mb: 2 }}>
            Your workflow has completed successfully! What would you like to do with the results?
          </Typography>
          
          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
            <Chip 
              icon={<CheckCircleIcon />}
              label="Completed"
              color="success"
              variant="outlined"
              size="small"
            />
            {outputFormat && (
              <Chip 
                label={`Format: ${outputFormat}`}
                variant="outlined"
                size="small"
              />
            )}
            {displayDuration > 0 && (
              <Chip 
                label={`${displayDuration.toFixed(2)}s`}
                variant="outlined"
                size="small"
              />
            )}
          </Box>
        </Box>

        {/* Legacy Progress Summary - show if using legacy props */}
        {results.length > 0 && (
          <Box sx={{ mb: 3 }}>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              Completion Rate: {completionRate.toFixed(1)}%
            </Typography>
            <LinearProgress 
              variant="determinate" 
              value={completionRate}
              color={displayStatus === 'success' ? 'success' : displayStatus === 'error' ? 'error' : 'warning'}
              sx={{ mb: 2 }}
            />
            
            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
              <Chip 
                icon={<CheckCircleIcon />}
                label={`${successCount} Successful`}
                color="success"
                variant="outlined"
                size="small"
              />
              {errorCount > 0 && (
                <Chip 
                  icon={<ErrorIcon />}
                  label={`${errorCount} Failed`}
                  color="error"
                  variant="outlined"
                  size="small"
                />
              )}
            </Box>
          </Box>
        )}

        <Divider sx={{ mb: 2 }} />

        {/* Overall Status Alert */}
        {overallStatus === 'error' && (
          <Alert severity="error" sx={{ mb: 2 }}>
            Workflow execution failed. Some nodes encountered errors.
          </Alert>
        )}
        {overallStatus === 'partial' && (
          <Alert severity="warning" sx={{ mb: 2 }}>
            Workflow completed with warnings. Some nodes may need attention.
          </Alert>
        )}
        {overallStatus === 'success' && (
          <Alert severity="success" sx={{ mb: 2 }}>
            Workflow executed successfully! All nodes completed without errors.
          </Alert>
        )}

        {/* Node Results */}
        <Typography variant="h6" sx={{ mb: 2 }}>
          Node Execution Results
        </Typography>
        
        <List>
          {results.map((result, index) => (
            <ListItem key={index} divider>
              <ListItemIcon>
                {getStatusIcon(result.status)}
              </ListItemIcon>
              <ListItemText
                primary={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="subtitle2">
                      {result.nodeName}
                    </Typography>
                    {result.duration && (
                      <Chip 
                        label={`${result.duration.toFixed(2)}s`}
                        size="small"
                        variant="outlined"
                      />
                    )}
                  </Box>
                }
                secondary={
                  <Box>
                    {result.message && (
                      <Typography variant="body2" color="text.secondary">
                        {result.message}
                      </Typography>
                    )}
                    {result.output && (
                      <Typography variant="caption" color="text.secondary">
                        Output: {typeof result.output === 'string' 
                          ? result.output.substring(0, 100) + (result.output.length > 100 ? '...' : '')
                          : 'Object result available'
                        }
                      </Typography>
                    )}
                  </Box>
                }
              />
            </ListItem>
          ))}
        </List>

        {results.length === 0 && (
          <Typography variant="body2" color="text.secondary" textAlign="center">
            No execution results available.
          </Typography>
        )}
      </DialogContent>
      
      <DialogActions>
        {/* New workflow completion actions */}
        {onViewNow && (
          <Button onClick={onViewNow} variant="contained" color="primary">
            View Now
          </Button>
        )}
        {onSaveOnly && (
          <Button onClick={onSaveOnly} variant="outlined">
            Save Only
          </Button>
        )}
        {onViewAndSave && (
          <Button onClick={onViewAndSave} variant="contained" color="secondary">
            View & Save
          </Button>
        )}
        {onViewLater && (
          <Button onClick={onViewLater} variant="outlined">
            View Later
          </Button>
        )}
        {onDismiss && (
          <Button onClick={onDismiss} variant="outlined">
            Dismiss
          </Button>
        )}
        
        {/* Legacy actions for backward compatibility */}
        {onDownloadResults && (
          <Button onClick={onDownloadResults} variant="outlined">
            Download Results
          </Button>
        )}
        {onRetry && errorCount > 0 && (
          <Button onClick={onRetry} variant="outlined" color="warning">
            Retry Failed
          </Button>
        )}
        {onClose && (
          <Button onClick={onClose} variant="contained">
            Close
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
};

export default EnhancedWorkflowCompletionDialog;