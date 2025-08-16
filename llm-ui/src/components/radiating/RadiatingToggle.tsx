import React, { useState, useEffect } from 'react';
import {
  Box,
  Switch,
  Typography,
  Tooltip,
  CircularProgress,
  IconButton,
  Chip,
  Fade
} from '@mui/material';
import {
  Hub as RadiatingIcon,
  Info as InfoIcon,
  CheckCircle as EnabledIcon,
  Cancel as DisabledIcon
} from '@mui/icons-material';
import { RadiatingToggleRequest, RadiatingToggleResponse } from '../../types/radiating';

interface RadiatingToggleProps {
  conversationId?: string;
  onToggle?: (enabled: boolean) => void;
  disabled?: boolean;
  size?: 'small' | 'medium' | 'large';
  showLabel?: boolean;
  showStatus?: boolean;
}

const RadiatingToggle: React.FC<RadiatingToggleProps> = ({
  conversationId,
  onToggle,
  disabled = false,
  size = 'medium',
  showLabel = true,
  showStatus = true
}) => {
  const [enabled, setEnabled] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string>('');

  // Load initial state from localStorage or API
  useEffect(() => {
    const loadState = async () => {
      try {
        // Check localStorage first
        const savedState = localStorage.getItem(`radiating-enabled-${conversationId || 'global'}`);
        if (savedState !== null) {
          setEnabled(JSON.parse(savedState));
        }

        // Optionally fetch from API to sync
        const response = await fetch('/api/v1/radiating/status', {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          }
        });

        if (response.ok) {
          const data = await response.json();
          setEnabled(data.enabled || false);
          localStorage.setItem(
            `radiating-enabled-${conversationId || 'global'}`,
            JSON.stringify(data.enabled)
          );
        }
      } catch (err) {
        console.warn('Failed to load radiating state:', err);
      }
    };

    loadState();
  }, [conversationId]);

  const handleToggle = async () => {
    if (disabled || loading) return;

    const newState = !enabled;
    setLoading(true);
    setError(null);

    try {
      const request: RadiatingToggleRequest = {
        enabled: newState,
        conversation_id: conversationId
      };

      const response = await fetch('/api/v1/radiating/toggle', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request)
      });

      if (!response.ok) {
        throw new Error(`Failed to toggle radiating: ${response.statusText}`);
      }

      const data: RadiatingToggleResponse = await response.json();
      
      setEnabled(data.enabled);
      setStatusMessage(data.message);
      
      // Save to localStorage
      localStorage.setItem(
        `radiating-enabled-${conversationId || 'global'}`,
        JSON.stringify(data.enabled)
      );

      // Notify parent component
      if (onToggle) {
        onToggle(data.enabled);
      }

      // Clear status message after 3 seconds
      setTimeout(() => setStatusMessage(''), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      // Revert state on error
      setEnabled(!newState);
      
      // Clear error after 5 seconds
      setTimeout(() => setError(null), 5000);
    } finally {
      setLoading(false);
    }
  };

  const getSizeStyles = () => {
    switch (size) {
      case 'small':
        return { transform: 'scale(0.8)' };
      case 'large':
        return { transform: 'scale(1.2)' };
      default:
        return {};
    }
  };

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 1,
        opacity: disabled ? 0.5 : 1,
        transition: 'opacity 0.3s'
      }}
    >
      {/* Icon */}
      <RadiatingIcon 
        sx={{ 
          color: enabled ? 'primary.main' : 'text.secondary',
          fontSize: size === 'small' ? 20 : size === 'large' ? 28 : 24,
          transition: 'color 0.3s'
        }} 
      />

      {/* Label */}
      {showLabel && (
        <Typography
          variant={size === 'small' ? 'body2' : 'body1'}
          sx={{
            fontWeight: enabled ? 600 : 400,
            color: enabled ? 'primary.main' : 'text.primary',
            transition: 'all 0.3s'
          }}
        >
          Radiating Coverage
        </Typography>
      )}

      {/* Switch with Tooltip */}
      <Tooltip
        title={
          <Box>
            <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
              Universal Radiating Coverage System
            </Typography>
            <Typography variant="caption" display="block" sx={{ mb: 1 }}>
              When enabled, the system will automatically explore related entities and relationships 
              beyond your direct query, creating a comprehensive knowledge network.
            </Typography>
            <Typography variant="caption" display="block">
              • Discovers hidden connections
              <br />
              • Expands search breadth intelligently
              <br />
              • Provides contextual depth
              <br />
              • Enhances answer completeness
            </Typography>
          </Box>
        }
        placement="top"
        arrow
      >
        <Box sx={{ display: 'flex', alignItems: 'center', ...getSizeStyles() }}>
          {loading ? (
            <CircularProgress size={20} />
          ) : (
            <Switch
              checked={enabled}
              onChange={handleToggle}
              disabled={disabled || loading}
              color="primary"
              inputProps={{ 'aria-label': 'Toggle radiating coverage' }}
            />
          )}
        </Box>
      </Tooltip>

      {/* Info Icon */}
      <Tooltip
        title="Click to learn more about radiating coverage"
        placement="top"
      >
        <IconButton
          size="small"
          onClick={() => {
            // Could open a help dialog or navigate to documentation
            window.open('/docs/radiating-coverage', '_blank');
          }}
          sx={{ 
            p: 0.5,
            opacity: 0.7,
            '&:hover': { opacity: 1 }
          }}
        >
          <InfoIcon fontSize="small" />
        </IconButton>
      </Tooltip>

      {/* Status Indicator */}
      {showStatus && (
        <Fade in={enabled}>
          <Chip
            icon={enabled ? <EnabledIcon /> : <DisabledIcon />}
            label={enabled ? 'Active' : 'Inactive'}
            size="small"
            color={enabled ? 'success' : 'default'}
            variant={enabled ? 'filled' : 'outlined'}
            sx={{ ml: 1 }}
          />
        </Fade>
      )}

      {/* Status Message */}
      {statusMessage && (
        <Fade in={!!statusMessage}>
          <Typography
            variant="caption"
            sx={{
              ml: 1,
              color: 'success.main',
              fontStyle: 'italic'
            }}
          >
            {statusMessage}
          </Typography>
        </Fade>
      )}

      {/* Error Message */}
      {error && (
        <Fade in={!!error}>
          <Typography
            variant="caption"
            sx={{
              ml: 1,
              color: 'error.main',
              fontStyle: 'italic'
            }}
          >
            {error}
          </Typography>
        </Fade>
      )}
    </Box>
  );
};

export default RadiatingToggle;