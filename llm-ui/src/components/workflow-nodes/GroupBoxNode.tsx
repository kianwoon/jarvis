import React from 'react';
import { Handle, Position } from 'reactflow';
import { Paper, Typography, Box } from '@mui/material';

const GroupBoxNode: React.FC<{ data: any }> = ({ data }) => (
  <Paper elevation={2} sx={{ padding: 2, minWidth: 150, border: '2px solid #9c27b0', borderRadius: 2 }}>
    <Handle type="target" position={Position.Top} style={{ background: '#9c27b0' }} />
    <Box>
      <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#9c27b0' }}>Group</Typography>
      <Typography variant="body2">{data.label || 'Group Node'}</Typography>
    </Box>
    <Handle type="source" position={Position.Bottom} style={{ background: '#9c27b0' }} />
  </Paper>
);

export default GroupBoxNode;
