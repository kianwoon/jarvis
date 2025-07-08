import React from 'react';
import { Handle, Position } from 'reactflow';
import { Paper, Typography, Box } from '@mui/material';

const LoopNode: React.FC<{ data: any }> = ({ data }) => (
  <Paper elevation={2} sx={{ padding: 2, minWidth: 150, border: '2px solid #795548', borderRadius: 2 }}>
    <Handle type="target" position={Position.Top} style={{ background: '#795548' }} />
    <Box>
      <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#795548' }}>Loop</Typography>
      <Typography variant="body2">{data.label || 'Loop Node'}</Typography>
    </Box>
    <Handle type="source" position={Position.Bottom} style={{ background: '#795548' }} />
  </Paper>
);

export default LoopNode;
