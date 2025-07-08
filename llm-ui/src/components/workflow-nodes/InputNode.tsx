import React from 'react';
import { Handle, Position } from 'reactflow';
import { Paper, Typography, Box } from '@mui/material';

const InputNode: React.FC<{ data: any }> = ({ data }) => (
  <Paper elevation={2} sx={{ padding: 2, minWidth: 150, border: '2px solid #2196f3', borderRadius: 2 }}>
    <Box>
      <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#2196f3' }}>Input</Typography>
      <Typography variant="body2">{data.label || 'User Input'}</Typography>
    </Box>
    <Handle type="source" position={Position.Bottom} style={{ background: '#2196f3' }} />
  </Paper>
);

export default InputNode;