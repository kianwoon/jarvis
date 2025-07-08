import React from 'react';
import { Handle, Position } from 'reactflow';
import { Paper, Typography, Box } from '@mui/material';

const VariableNode: React.FC<{ data: any }> = ({ data }) => (
  <Paper elevation={2} sx={{ padding: 2, minWidth: 150, border: '2px solid #607d8b', borderRadius: 2 }}>
    <Handle type="target" position={Position.Top} style={{ background: '#607d8b' }} />
    <Box>
      <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#607d8b' }}>Variable</Typography>
      <Typography variant="body2">{data.label || 'Variable Node'}</Typography>
    </Box>
    <Handle type="source" position={Position.Bottom} style={{ background: '#607d8b' }} />
  </Paper>
);

export default VariableNode;
