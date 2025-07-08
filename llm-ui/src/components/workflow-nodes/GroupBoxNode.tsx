import React, { useState } from 'react';
import { NodeResizer } from 'reactflow';
import { Paper, Typography, Box, TextField, IconButton } from '@mui/material';
import { Edit as EditIcon, Check as CheckIcon } from '@mui/icons-material';

interface GroupBoxNodeProps {
  data: {
    label?: string;
    title?: string;
    borderColor?: string;
    backgroundColor?: string;
    opacity?: number;
  };
  id: string;
  selected?: boolean;
  updateNodeData?: (nodeId: string, newData: any) => void;
}

const GroupBoxNode: React.FC<GroupBoxNodeProps> = ({ data, id, selected, updateNodeData }) => {
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [title, setTitle] = useState(data.title || 'Group');

  // Predefined color options for right-click toggle
  const colorOptions = [
    { border: '#9c27b0', background: 'rgba(156, 39, 176, 0.2)' }, // Purple
    { border: '#2196f3', background: 'rgba(33, 150, 243, 0.2)' }, // Blue
    { border: '#4caf50', background: 'rgba(76, 175, 80, 0.2)' }, // Green
    { border: '#ff9800', background: 'rgba(255, 152, 0, 0.2)' }, // Orange
    { border: '#f44336', background: 'rgba(244, 67, 54, 0.2)' }, // Red
    { border: '#9e9e9e', background: 'rgba(158, 158, 158, 0.2)' }, // Gray
  ];

  const handleTitleSave = () => {
    setIsEditingTitle(false);
    if (updateNodeData) {
      updateNodeData(id, { ...data, title });
    }
  };

  const handleRightClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    // Find current color index
    const currentBorderColor = data.borderColor || '#9c27b0';
    const currentIndex = colorOptions.findIndex(option => option.border === currentBorderColor);
    
    // Get next color (cycle through options)
    const nextIndex = (currentIndex + 1) % colorOptions.length;
    const nextColor = colorOptions[nextIndex];
    
    if (updateNodeData) {
      updateNodeData(id, { 
        ...data, 
        borderColor: nextColor.border,
        backgroundColor: nextColor.background
      });
    }
  };

  const borderColor = data.borderColor || '#9c27b0';
  const backgroundColor = data.backgroundColor || 'rgba(156, 39, 176, 0.2)'; // Higher opacity
  const opacity = data.opacity || 1;

  return (
    <>
      <NodeResizer 
        color={borderColor}
        isVisible={selected}
        minWidth={200}
        minHeight={150}
      />
      <Paper 
        elevation={0}
        onContextMenu={handleRightClick}
        sx={{ 
          width: '100%',
          height: '100%',
          padding: 2,
          border: `2px ${selected ? 'solid' : 'dashed'} ${borderColor}`,
          borderRadius: 2,
          backgroundColor: backgroundColor,
          opacity: opacity,
          display: 'flex',
          flexDirection: 'column',
          pointerEvents: 'all',
          zIndex: -1, // Behind all other nodes
          position: 'relative',
          '&:hover': {
            borderStyle: 'solid',
            opacity: Math.min(opacity + 0.1, 1)
          }
        }}
      >
        <Box sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: 1,
          marginBottom: 1,
          pointerEvents: 'all'
        }}>
          {isEditingTitle ? (
            <>
              <TextField
                value={title}
                onChange={(e) => {
                  const newTitle = e.target.value;
                  setTitle(newTitle);
                  if (updateNodeData) {
                    updateNodeData(id, { ...data, title: newTitle });
                  }
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    handleTitleSave();
                  }
                }}
                size="small"
                autoFocus
                variant="standard"
                sx={{ 
                  flex: 1,
                  '& .MuiInput-root': {
                    fontSize: '0.875rem',
                    fontWeight: 600,
                    color: borderColor
                  }
                }}
              />
              <IconButton size="small" onClick={handleTitleSave}>
                <CheckIcon fontSize="small" />
              </IconButton>
            </>
          ) : (
            <>
              <Typography 
                variant="subtitle2" 
                sx={{ 
                  fontWeight: 600, 
                  color: borderColor,
                  flex: 1
                }}
              >
                {title}
              </Typography>
              <IconButton 
                size="small" 
                onClick={() => setIsEditingTitle(true)}
                sx={{ opacity: 0.7, '&:hover': { opacity: 1 } }}
              >
                <EditIcon fontSize="small" />
              </IconButton>
            </>
          )}
        </Box>
        {data.label && (
          <Typography 
            variant="caption" 
            sx={{ 
              color: 'text.secondary',
              fontStyle: 'italic'
            }}
          >
            {data.label}
          </Typography>
        )}
      </Paper>
    </>
  );
};

export default GroupBoxNode;
