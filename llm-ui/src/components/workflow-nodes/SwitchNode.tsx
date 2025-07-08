import React from 'react';
import { Handle, Position } from 'reactflow';
import { Paper, Typography, Box } from '@mui/material';

const SwitchNode: React.FC<{ data: any }> = ({ data }) => (
  <Paper elevation={2} sx={{ padding: 2, minWidth: 150, border: '2px solid #ff9800', borderRadius: 2 }}>
    <Handle type="target" position={Position.Top} style={{ background: '#ff9800' }} />
    <Box>
      <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#ff9800' }}>Switch</Typography>
      <Typography variant="body2">{data.label || 'Switch Node'}</Typography>
    </Box>
    <Handle type="source" position={Position.Bottom} style={{ background: '#ff9800' }} />
  </Paper>
);

export default SwitchNode;
