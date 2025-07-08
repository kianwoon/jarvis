import React from 'react';
import { Handle, Position } from 'reactflow';
import { Paper, Typography, Box } from '@mui/material';

const LLMNode: React.FC<{ data: any }> = ({ data }) => (
  <Paper elevation={2} sx={{ padding: 2, minWidth: 150, border: '2px solid #673ab7', borderRadius: 2 }}>
    <Handle type="target" position={Position.Top} style={{ background: '#673ab7' }} />
    <Box>
      <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#673ab7' }}>LLM</Typography>
      <Typography variant="body2">{data.label || 'AI Processing'}</Typography>
    </Box>
    <Handle type="source" position={Position.Bottom} style={{ background: '#673ab7' }} />
  </Paper>
);

export default LLMNode;