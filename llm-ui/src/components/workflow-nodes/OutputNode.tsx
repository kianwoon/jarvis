import React from 'react';
import { Handle, Position } from 'reactflow';
import { Paper, Typography, Box } from '@mui/material';

const OutputNode: React.FC<{ data: any }> = ({ data }) => (
  <Paper elevation={2} sx={{ padding: 2, minWidth: 150, border: '2px solid #9c27b0', borderRadius: 2 }}>
    <Handle type="target" position={Position.Top} style={{ background: '#9c27b0' }} />
    <Box>
      <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#9c27b0' }}>Output</Typography>
      <Typography variant="body2">{data.label || 'Result'}</Typography>
    </Box>
  </Paper>
);

export default OutputNode;