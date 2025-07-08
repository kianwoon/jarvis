import React from 'react';
import { Handle, Position } from 'reactflow';
import { Paper, Typography, Box } from '@mui/material';

const TransformNode: React.FC<{ data: any }> = ({ data }) => (
  <Paper elevation={2} sx={{ padding: 2, minWidth: 150, border: '2px solid #4caf50', borderRadius: 2 }}>
    <Handle type="target" position={Position.Top} style={{ background: '#4caf50' }} />
    <Box>
      <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#4caf50' }}>Transform</Typography>
      <Typography variant="body2">{data.label || 'Transform Node'}</Typography>
    </Box>
    <Handle type="source" position={Position.Bottom} style={{ background: '#4caf50' }} />
  </Paper>
);

export default TransformNode;
