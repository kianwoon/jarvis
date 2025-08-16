import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  LinearProgress,
  CircularProgress,
  Button,
  Chip,
  Grid,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Fade,
  Zoom,
  IconButton,
  Tooltip,
  Alert
} from '@mui/material';
import {
  Cancel as CancelIcon,
  Pause as PauseIcon,
  PlayArrow as ResumeIcon,
  Hub as EntityIcon,
  Link as RelationshipIcon,
  Layers as DepthIcon,
  Timer as TimeIcon,
  Speed as SpeedIcon,
  CheckCircle as CompleteIcon,
  Error as ErrorIcon,
  Info as InfoIcon
} from '@mui/icons-material';
import { 
  RadiatingProgress as RadiatingProgressType,
  RadiatingCancelRequest,
  RadiatingCancelResponse 
} from '../../types/radiating';

interface RadiatingProgressProps {
  jobId?: string;
  progress: RadiatingProgressType;
  onCancel?: () => void;
  onPause?: () => void;
  onResume?: () => void;
  compact?: boolean;
  showDetails?: boolean;
}

const RadiatingProgress: React.FC<RadiatingProgressProps> = ({
  jobId,
  progress,
  onCancel,
  onPause,
  onResume,
  compact = false,
  showDetails = true
}) => {
  const [cancelling, setCancelling] = useState(false);
  const [paused, setPaused] = useState(false);
  const [animationPhase, setAnimationPhase] = useState(0);

  // Animation for the radiating effect
  useEffect(() => {
    if (!progress.isActive) return;
    
    const interval = setInterval(() => {
      setAnimationPhase((prev) => (prev + 1) % 4);
    }, 500);
    
    return () => clearInterval(interval);
  }, [progress.isActive]);

  const handleCancel = async () => {
    if (!jobId || cancelling) return;
    
    setCancelling(true);
    
    try {
      const request: RadiatingCancelRequest = { job_id: jobId };
      
      const response = await fetch('/api/v1/radiating/cancel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
      });
      
      if (response.ok) {
        const data: RadiatingCancelResponse = await response.json();
        if (data.success && onCancel) {
          onCancel();
        }
      }
    } catch (error) {
      console.error('Failed to cancel radiating job:', error);
    } finally {
      setCancelling(false);
    }
  };

  const getStatusColor = () => {
    switch (progress.status) {
      case 'error': return 'error';
      case 'completed': return 'success';
      case 'traversing':
      case 'processing': return 'primary';
      default: return 'default';
    }
  };

  const getStatusIcon = () => {
    switch (progress.status) {
      case 'error': return <ErrorIcon />;
      case 'completed': return <CompleteIcon />;
      default: return <CircularProgress size={16} />;
    }
  };

  const formatTime = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    
    if (hours > 0) {
      return `${hours}h ${minutes % 60}m`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    } else {
      return `${seconds}s`;
    }
  };

  const calculateProgress = () => {
    if (progress.totalDepth === 0) return 0;
    return (progress.currentDepth / progress.totalDepth) * 100;
  };

  const calculateEntitiesPerSecond = () => {
    if (progress.elapsedTime === 0) return 0;
    return (progress.processedEntities / (progress.elapsedTime / 1000)).toFixed(1);
  };

  // Compact view for embedding in chat interface
  if (compact) {
    return (
      <Fade in={progress.isActive}>
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 2,
            p: 1,
            backgroundColor: 'action.hover',
            borderRadius: 1
          }}
        >
          <Box sx={{ position: 'relative' }}>
            <CircularProgress
              variant="determinate"
              value={calculateProgress()}
              size={40}
            />
            <Box
              sx={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)'
              }}
            >
              <Typography variant="caption" component="div" color="text.secondary">
                {`${Math.round(calculateProgress())}%`}
              </Typography>
            </Box>
          </Box>
          
          <Box sx={{ flex: 1 }}>
            <Typography variant="body2">
              Exploring depth {progress.currentDepth}/{progress.totalDepth}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {progress.entitiesDiscovered} entities • {progress.relationshipsFound} relationships
            </Typography>
          </Box>
          
          {onCancel && (
            <IconButton
              size="small"
              onClick={handleCancel}
              disabled={cancelling}
            >
              <CancelIcon />
            </IconButton>
          )}
        </Box>
      </Fade>
    );
  }

  // Full view
  return (
    <Zoom in={progress.isActive}>
      <Paper
        elevation={3}
        sx={{
          p: 3,
          backgroundColor: 'background.paper',
          borderRadius: 2,
          border: '1px solid',
          borderColor: 'primary.light'
        }}
      >
        {/* Header */}
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <EntityIcon color="primary" sx={{ fontSize: 28 }} />
            <Typography variant="h6">
              Radiating Coverage in Progress
            </Typography>
            <Chip
              icon={getStatusIcon()}
              label={progress.status}
              color={getStatusColor()}
              size="small"
              variant="filled"
            />
          </Box>
          
          <Box sx={{ display: 'flex', gap: 1 }}>
            {paused ? (
              <Tooltip title="Resume">
                <IconButton onClick={onResume} color="primary">
                  <ResumeIcon />
                </IconButton>
              </Tooltip>
            ) : (
              <Tooltip title="Pause">
                <IconButton onClick={onPause}>
                  <PauseIcon />
                </IconButton>
              </Tooltip>
            )}
            
            <Tooltip title="Cancel exploration">
              <IconButton
                onClick={handleCancel}
                disabled={cancelling}
                color="error"
              >
                {cancelling ? <CircularProgress size={24} /> : <CancelIcon />}
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {/* Progress Bar */}
        <Box sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="body2" color="text.secondary">
              Overall Progress
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {Math.round(calculateProgress())}%
            </Typography>
          </Box>
          <LinearProgress
            variant="determinate"
            value={calculateProgress()}
            sx={{ height: 8, borderRadius: 4 }}
          />
          
          {/* Depth indicators */}
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
            {Array.from({ length: progress.totalDepth }, (_, i) => (
              <Chip
                key={i}
                label={`L${i + 1}`}
                size="small"
                color={i < progress.currentDepth ? 'primary' : 'default'}
                variant={i < progress.currentDepth ? 'filled' : 'outlined'}
              />
            ))}
          </Box>
        </Box>

        {/* Animated Visualization */}
        <Box
          sx={{
            height: 120,
            position: 'relative',
            backgroundColor: 'action.hover',
            borderRadius: 2,
            overflow: 'hidden',
            mb: 3
          }}
        >
          {/* Central node */}
          <Box
            sx={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              width: 20,
              height: 20,
              backgroundColor: 'primary.main',
              borderRadius: '50%',
              zIndex: 10
            }}
          />
          
          {/* Radiating circles */}
          {[1, 2, 3].map((level) => (
            <Box
              key={level}
              sx={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                width: 40 + level * 30,
                height: 40 + level * 30,
                border: '2px solid',
                borderColor: 'primary.light',
                borderRadius: '50%',
                opacity: animationPhase === level ? 1 : 0.3,
                transition: 'opacity 0.5s',
                animation: progress.currentDepth >= level ? 'pulse 2s infinite' : 'none'
              }}
            />
          ))}
          
          {/* Current entity indicator */}
          {progress.currentEntity && (
            <Typography
              variant="caption"
              sx={{
                position: 'absolute',
                bottom: 8,
                left: '50%',
                transform: 'translateX(-50%)',
                backgroundColor: 'background.paper',
                px: 1,
                py: 0.5,
                borderRadius: 1
              }}
            >
              Processing: {progress.currentEntity}
            </Typography>
          )}
        </Box>

        {/* Statistics Grid */}
        <Grid container spacing={2} sx={{ mb: 2 }}>
          <Grid item xs={6} sm={3}>
            <Box sx={{ textAlign: 'center' }}>
              <EntityIcon color="primary" sx={{ mb: 0.5 }} />
              <Typography variant="h5">
                {progress.entitiesDiscovered}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Entities Found
              </Typography>
            </Box>
          </Grid>
          
          <Grid item xs={6} sm={3}>
            <Box sx={{ textAlign: 'center' }}>
              <RelationshipIcon color="primary" sx={{ mb: 0.5 }} />
              <Typography variant="h5">
                {progress.relationshipsFound}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Relationships
              </Typography>
            </Box>
          </Grid>
          
          <Grid item xs={6} sm={3}>
            <Box sx={{ textAlign: 'center' }}>
              <DepthIcon color="primary" sx={{ mb: 0.5 }} />
              <Typography variant="h5">
                {progress.currentDepth}/{progress.totalDepth}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Current Depth
              </Typography>
            </Box>
          </Grid>
          
          <Grid item xs={6} sm={3}>
            <Box sx={{ textAlign: 'center' }}>
              <SpeedIcon color="primary" sx={{ mb: 0.5 }} />
              <Typography variant="h5">
                {calculateEntitiesPerSecond()}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Entities/sec
              </Typography>
            </Box>
          </Grid>
        </Grid>

        {/* Detailed Information */}
        {showDetails && (
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              Processing Details
            </Typography>
            
            <List dense>
              <ListItem>
                <ListItemIcon>
                  <InfoIcon fontSize="small" />
                </ListItemIcon>
                <ListItemText
                  primary="Queue Size"
                  secondary={`${progress.queueSize} entities pending`}
                />
              </ListItem>
              
              <ListItem>
                <ListItemIcon>
                  <TimeIcon fontSize="small" />
                </ListItemIcon>
                <ListItemText
                  primary="Elapsed Time"
                  secondary={formatTime(progress.elapsedTime)}
                />
              </ListItem>
              
              {progress.estimatedTimeRemaining && (
                <ListItem>
                  <ListItemIcon>
                    <TimeIcon fontSize="small" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Estimated Time Remaining"
                    secondary={formatTime(progress.estimatedTimeRemaining)}
                  />
                </ListItem>
              )}
            </List>
          </Box>
        )}

        {/* Error Message */}
        {progress.error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {progress.error}
          </Alert>
        )}
      </Paper>
    </Zoom>
  );
};

// Add CSS animation
const style = document.createElement('style');
style.textContent = `
  @keyframes pulse {
    0% {
      transform: translate(-50%, -50%) scale(1);
      opacity: 0.8;
    }
    50% {
      transform: translate(-50%, -50%) scale(1.1);
      opacity: 0.4;
    }
    100% {
      transform: translate(-50%, -50%) scale(1);
      opacity: 0.8;
    }
  }
`;
document.head.appendChild(style);

export default RadiatingProgress;