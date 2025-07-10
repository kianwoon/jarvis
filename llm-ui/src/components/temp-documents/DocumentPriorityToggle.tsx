import React from 'react';
import {
  Box,
  Switch,
  FormControlLabel,
  Typography,
  Chip,
  Tooltip,
  IconButton,
  Menu,
  MenuItem,
  Divider
} from '@mui/material';
import {
  Settings as SettingsIcon,
  PriorityHigh as PriorityIcon,
  Storage as DatabaseIcon,
  Merge as MergeIcon,
  Info as InfoIcon
} from '@mui/icons-material';
import { useState } from 'react';

interface DocumentPriorityToggleProps {
  enabled: boolean;
  onChange: (enabled: boolean) => void;
  documentCount: number;
  strategy?: 'temp_priority' | 'parallel_fusion' | 'persistent_only';
  onStrategyChange?: (strategy: 'temp_priority' | 'parallel_fusion' | 'persistent_only') => void;
  fallbackToPersistent?: boolean;
  onFallbackChange?: (enabled: boolean) => void;
  disabled?: boolean;
  description?: string;
  variant?: 'compact' | 'detailed';
}

const DocumentPriorityToggle: React.FC<DocumentPriorityToggleProps> = ({
  enabled,
  onChange,
  documentCount,
  strategy = 'temp_priority',
  onStrategyChange,
  fallbackToPersistent = true,
  onFallbackChange,
  disabled = false,
  description,
  variant = 'compact'
}) => {
  const [settingsAnchor, setSettingsAnchor] = useState<null | HTMLElement>(null);

  const handleSettingsClick = (event: React.MouseEvent<HTMLElement>) => {
    setSettingsAnchor(event.currentTarget);
  };

  const handleSettingsClose = () => {
    setSettingsAnchor(null);
  };

  const handleStrategySelect = (newStrategy: typeof strategy) => {
    onStrategyChange?.(newStrategy);
    handleSettingsClose();
  };

  const getStrategyIcon = (strategyType: typeof strategy) => {
    switch (strategyType) {
      case 'temp_priority':
        return <PriorityIcon fontSize="small" />;
      case 'parallel_fusion':
        return <MergeIcon fontSize="small" />;
      case 'persistent_only':
        return <DatabaseIcon fontSize="small" />;
      default:
        return <PriorityIcon fontSize="small" />;
    }
  };

  const getStrategyLabel = (strategyType: typeof strategy) => {
    switch (strategyType) {
      case 'temp_priority':
        return 'Document Priority';
      case 'parallel_fusion':
        return 'Blended Search';
      case 'persistent_only':
        return 'Knowledge Base Only';
      default:
        return 'Document Priority';
    }
  };

  const getStrategyDescription = (strategyType: typeof strategy) => {
    switch (strategyType) {
      case 'temp_priority':
        return 'Search uploaded documents first, fallback to knowledge base if needed';
      case 'parallel_fusion':
        return 'Search both uploaded documents and knowledge base, blend results by relevance';
      case 'persistent_only':
        return 'Search knowledge base only, ignore uploaded documents';
      default:
        return 'Search uploaded documents first';
    }
  };

  const getStatusColor = () => {
    if (disabled || documentCount === 0) return 'default';
    return enabled ? 'primary' : 'default';
  };

  const getStatusText = () => {
    if (documentCount === 0) {
      return 'No documents';
    }
    if (disabled) {
      return 'Disabled';
    }
    return enabled ? 'Active' : 'Inactive';
  };

  const effectiveDescription = description || (
    documentCount === 0 
      ? 'Upload documents to enable document priority mode'
      : enabled 
        ? getStrategyDescription(strategy)
        : 'Standard knowledge base search'
  );

  if (variant === 'compact') {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <FormControlLabel
          control={
            <Switch
              checked={enabled}
              onChange={(e) => onChange(e.target.checked)}
              disabled={disabled || documentCount === 0}
              size="small"
            />
          }
          label={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Typography variant="body2" sx={{ fontWeight: enabled ? 'bold' : 'normal' }}>
                Doc Priority
              </Typography>
              {documentCount > 0 && (
                <Chip 
                  label={documentCount} 
                  size="small" 
                  color={enabled ? 'primary' : 'default'}
                  variant="outlined"
                />
              )}
            </Box>
          }
          sx={{ margin: 0 }}
        />
        
        {onStrategyChange && (
          <Tooltip title="RAG Strategy Settings">
            <IconButton 
              size="small" 
              onClick={handleSettingsClick}
              disabled={disabled || documentCount === 0}
            >
              <SettingsIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        )}
      </Box>
    );
  }

  return (
    <Box>
      {/* Main Toggle Section */}
      <Box sx={{ 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'space-between',
        p: 2,
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: 2,
        bgcolor: enabled ? 'action.selected' : 'background.paper'
      }}>
        <Box sx={{ flex: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1 }}>
            <FormControlLabel
              control={
                <Switch
                  checked={enabled}
                  onChange={(e) => onChange(e.target.checked)}
                  disabled={disabled || documentCount === 0}
                />
              }
              label={
                <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                  Document Priority Mode
                </Typography>
              }
              sx={{ margin: 0 }}
            />
            
            <Chip 
              label={getStatusText()}
              color={getStatusColor()}
              size="small"
              variant={enabled ? 'filled' : 'outlined'}
            />
            
            {documentCount > 0 && (
              <Chip 
                label={`${documentCount} docs`}
                size="small" 
                color="info"
                variant="outlined"
              />
            )}
          </Box>
          
          <Typography variant="body2" color="text.secondary">
            {effectiveDescription}
          </Typography>
        </Box>

        {onStrategyChange && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Tooltip title="Current RAG Strategy">
              <Chip
                icon={getStrategyIcon(strategy)}
                label={getStrategyLabel(strategy)}
                color="primary"
                variant="outlined"
                size="small"
              />
            </Tooltip>
            
            <Tooltip title="Configure RAG Strategy">
              <IconButton 
                onClick={handleSettingsClick}
                disabled={disabled || documentCount === 0}
              >
                <SettingsIcon />
              </IconButton>
            </Tooltip>
          </Box>
        )}
      </Box>

      {/* Strategy Selection Menu */}
      <Menu
        anchorEl={settingsAnchor}
        open={Boolean(settingsAnchor)}
        onClose={handleSettingsClose}
        transformOrigin={{ horizontal: 'right', vertical: 'top' }}
        anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
      >
        <MenuItem disabled>
          <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>
            RAG Strategy
          </Typography>
        </MenuItem>
        <Divider />
        
        <MenuItem 
          selected={strategy === 'temp_priority'}
          onClick={() => handleStrategySelect('temp_priority')}
        >
          <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2 }}>
            <PriorityIcon color={strategy === 'temp_priority' ? 'primary' : 'inherit'} />
            <Box>
              <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                Document Priority
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Search documents first, fallback to knowledge base
              </Typography>
            </Box>
          </Box>
        </MenuItem>
        
        <MenuItem 
          selected={strategy === 'parallel_fusion'}
          onClick={() => handleStrategySelect('parallel_fusion')}
        >
          <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2 }}>
            <MergeIcon color={strategy === 'parallel_fusion' ? 'primary' : 'inherit'} />
            <Box>
              <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                Blended Search
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Combine results from documents and knowledge base
              </Typography>
            </Box>
          </Box>
        </MenuItem>
        
        <MenuItem 
          selected={strategy === 'persistent_only'}
          onClick={() => handleStrategySelect('persistent_only')}
        >
          <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2 }}>
            <DatabaseIcon color={strategy === 'persistent_only' ? 'primary' : 'inherit'} />
            <Box>
              <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                Knowledge Base Only
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Ignore uploaded documents, use knowledge base only
              </Typography>
            </Box>
          </Box>
        </MenuItem>
        
        {onFallbackChange && strategy === 'temp_priority' && (
          <>
            <Divider />
            <MenuItem onClick={() => onFallbackChange(!fallbackToPersistent)}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Switch
                  checked={fallbackToPersistent}
                  size="small"
                />
                <Box>
                  <Typography variant="body2">
                    Fallback to Knowledge Base
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Use knowledge base if no document results
                  </Typography>
                </Box>
              </Box>
            </MenuItem>
          </>
        )}
      </Menu>
    </Box>
  );
};

export default DocumentPriorityToggle;