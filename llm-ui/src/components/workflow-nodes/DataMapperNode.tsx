import React from 'react';
import { Handle, Position } from 'reactflow';
import { Paper, Typography, Box } from '@mui/material';

const DataMapperNode: React.FC<{ data: any }> = ({ data }) => (
  <Paper elevation={2} sx={{ padding: 2, minWidth: 150, border: '2px solid #009688', borderRadius: 2 }}>
    <Handle type="target" position={Position.Top} style={{ background: '#009688' }} />
    <Box>
      <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#009688' }}>Mapper</Typography>
      <Typography variant="body2">{data.label || 'Mapper Node'}</Typography>
    </Box>
    <Handle type="source" position={Position.Bottom} style={{ background: '#009688' }} />
  </Paper>
);

export default DataMapperNode;
