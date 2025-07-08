import React from 'react';
import { Handle, Position } from 'reactflow';
import { Paper, Typography, Box } from '@mui/material';

const DelayNode: React.FC<{ data: any }> = ({ data }) => (
  <Paper elevation={2} sx={{ padding: 2, minWidth: 150, border: '2px solid #9e9e9e', borderRadius: 2 }}>
    <Handle type="target" position={Position.Top} style={{ background: '#9e9e9e' }} />
    <Box>
      <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#9e9e9e' }}>Delay</Typography>
      <Typography variant="body2">{data.label || 'Delay Node'}</Typography>
    </Box>
    <Handle type="source" position={Position.Bottom} style={{ background: '#9e9e9e' }} />
  </Paper>
);

export default DelayNode;
