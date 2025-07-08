import React from 'react';
import { Handle, Position } from 'reactflow';
import { Paper, Typography, Box } from '@mui/material';

const HttpNode: React.FC<{ data: any }> = ({ data }) => (
  <Paper elevation={2} sx={{ padding: 2, minWidth: 150, border: '2px solid #ff5722', borderRadius: 2 }}>
    <Handle type="target" position={Position.Top} style={{ background: '#ff5722' }} />
    <Box>
      <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#ff5722' }}>HTTP</Typography>
      <Typography variant="body2">{data.label || 'HTTP Node'}</Typography>
    </Box>
    <Handle type="source" position={Position.Bottom} style={{ background: '#ff5722' }} />
  </Paper>
);

export default HttpNode;
