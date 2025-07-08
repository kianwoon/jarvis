import React from 'react';
import { Handle, Position } from 'reactflow';
import { Paper, Typography, Box } from '@mui/material';

const ParallelNode: React.FC<{ data: any }> = ({ data }) => (
  <Paper elevation={2} sx={{ padding: 2, minWidth: 150, border: '2px solid #00bcd4', borderRadius: 2 }}>
    <Handle type="target" position={Position.Top} style={{ background: '#00bcd4' }} />
    <Box>
      <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#00bcd4' }}>Parallel</Typography>
      <Typography variant="body2">{data.label || 'Parallel Node'}</Typography>
    </Box>
    <Handle type="source" position={Position.Bottom} style={{ background: '#00bcd4' }} />
  </Paper>
);

export default ParallelNode;
