import React from 'react';
import { Handle, Position } from 'reactflow';
import { Paper, Typography, Box } from '@mui/material';

const StartNode: React.FC<{ data: any }> = ({ data }) => (
  <Paper elevation={2} sx={{ padding: 2, minWidth: 120, border: '2px solid #4caf50', borderRadius: 2, bgcolor: '#e8f5e8' }}>
    <Box textAlign="center">
      <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#4caf50' }}>START</Typography>
      <Typography variant="body2">{data.label || 'Begin'}</Typography>
    </Box>
    <Handle type="source" position={Position.Bottom} style={{ background: '#4caf50' }} />
  </Paper>
);

export default StartNode;