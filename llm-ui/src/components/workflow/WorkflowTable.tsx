import React, { useState } from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Paper,
  IconButton,
  Chip,
  Box,
  Typography,
  Tooltip,
  TableSortLabel
} from '@mui/material';
import {
  Edit as EditIcon,
  Delete as DeleteIcon,
  ContentCopy as CopyIcon,
  PowerSettingsNew as PowerIcon,
  AccountTree as NodeIcon,
  Link as LinkIcon,
  Storage as CacheIcon
} from '@mui/icons-material';

interface WorkflowData {
  id: string;
  name: string;
  description: string;
  langflow_config: any;
  is_active: boolean;
  created_by: string;
  created_at: string;
  updated_at: string;
}

interface WorkflowTableProps {
  workflows: WorkflowData[];
  onEdit: (workflow: WorkflowData) => void;
  onDelete: (workflowId: string) => void;
  onDuplicate: (workflow: WorkflowData) => void;
  onToggleActive: (workflow: WorkflowData) => void;
}

type Order = 'asc' | 'desc';
type OrderBy = 'name' | 'updated_at' | 'created_at' | 'is_active';

const WorkflowTable: React.FC<WorkflowTableProps> = ({
  workflows,
  onEdit,
  onDelete,
  onDuplicate,
  onToggleActive
}) => {
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [order, setOrder] = useState<Order>('desc');
  const [orderBy, setOrderBy] = useState<OrderBy>('updated_at');

  const handleRequestSort = (property: OrderBy) => {
    const isAsc = orderBy === property && order === 'asc';
    setOrder(isAsc ? 'desc' : 'asc');
    setOrderBy(property);
  };

  const handleChangePage = (_: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const getWorkflowStats = (config: any) => {
    try {
      const nodeCount = config?.nodes?.length || 0;
      const edgeCount = config?.edges?.length || 0;
      const hasCache = config?.metadata?.has_cache_nodes || false;
      
      return { nodeCount, edgeCount, hasCache };
    } catch {
      return { nodeCount: 0, edgeCount: 0, hasCache: false };
    }
  };

  const sortedWorkflows = React.useMemo(() => {
    const comparator = (a: WorkflowData, b: WorkflowData) => {
      let aValue: any = a[orderBy];
      let bValue: any = b[orderBy];

      if (orderBy === 'name') {
        aValue = aValue.toLowerCase();
        bValue = bValue.toLowerCase();
      }

      if (order === 'desc') {
        return bValue > aValue ? 1 : -1;
      }
      return aValue > bValue ? 1 : -1;
    };

    return [...workflows].sort(comparator);
  }, [workflows, order, orderBy]);

  const paginatedWorkflows = sortedWorkflows.slice(
    page * rowsPerPage,
    page * rowsPerPage + rowsPerPage
  );

  return (
    <Paper sx={{ width: '100%', overflow: 'hidden' }}>
      <TableContainer sx={{ maxHeight: 'calc(100vh - 300px)' }}>
        <Table stickyHeader aria-label="workflows table">
          <TableHead>
            <TableRow>
              <TableCell>
                <TableSortLabel
                  active={orderBy === 'name'}
                  direction={orderBy === 'name' ? order : 'asc'}
                  onClick={() => handleRequestSort('name')}
                >
                  Name
                </TableSortLabel>
              </TableCell>
              <TableCell>Description</TableCell>
              <TableCell align="center">
                <TableSortLabel
                  active={orderBy === 'is_active'}
                  direction={orderBy === 'is_active' ? order : 'asc'}
                  onClick={() => handleRequestSort('is_active')}
                >
                  Status
                </TableSortLabel>
              </TableCell>
              <TableCell align="center">Stats</TableCell>
              <TableCell>
                <TableSortLabel
                  active={orderBy === 'updated_at'}
                  direction={orderBy === 'updated_at' ? order : 'asc'}
                  onClick={() => handleRequestSort('updated_at')}
                >
                  Last Modified
                </TableSortLabel>
              </TableCell>
              <TableCell align="right">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {paginatedWorkflows.map((workflow) => {
              const stats = getWorkflowStats(workflow.langflow_config);
              return (
                <TableRow
                  key={workflow.id}
                  hover
                  sx={{ '&:last-child td, &:last-child th': { border: 0 } }}
                >
                  <TableCell component="th" scope="row">
                    <Typography variant="body2" fontWeight="medium">
                      {workflow.name}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography 
                      variant="body2" 
                      color="text.secondary"
                      sx={{
                        maxWidth: 300,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap'
                      }}
                    >
                      {workflow.description || 'No description'}
                    </Typography>
                  </TableCell>
                  <TableCell align="center">
                    <Chip
                      label={workflow.is_active ? 'Active' : 'Inactive'}
                      color={workflow.is_active ? 'success' : 'default'}
                      size="small"
                    />
                  </TableCell>
                  <TableCell align="center">
                    <Box sx={{ display: 'flex', justifyContent: 'center', gap: 1 }}>
                      <Tooltip title={`${stats.nodeCount} nodes`}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                          <NodeIcon fontSize="small" color="action" />
                          <Typography variant="body2">{stats.nodeCount}</Typography>
                        </Box>
                      </Tooltip>
                      <Tooltip title={`${stats.edgeCount} connections`}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                          <LinkIcon fontSize="small" color="action" />
                          <Typography variant="body2">{stats.edgeCount}</Typography>
                        </Box>
                      </Tooltip>
                      {stats.hasCache && (
                        <Tooltip title="Has cache nodes">
                          <CacheIcon fontSize="small" color="info" />
                        </Tooltip>
                      )}
                    </Box>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2" color="text.secondary">
                      {formatDate(workflow.updated_at)}
                    </Typography>
                  </TableCell>
                  <TableCell align="right">
                    <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 0.5 }}>
                      <Tooltip title="Edit">
                        <IconButton
                          size="small"
                          onClick={() => onEdit(workflow)}
                        >
                          <EditIcon />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title={workflow.is_active ? 'Deactivate' : 'Activate'}>
                        <IconButton
                          size="small"
                          onClick={() => onToggleActive(workflow)}
                        >
                          <PowerIcon />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Duplicate">
                        <IconButton
                          size="small"
                          onClick={() => onDuplicate(workflow)}
                        >
                          <CopyIcon />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Delete">
                        <IconButton
                          size="small"
                          color="error"
                          onClick={() => onDelete(workflow.id)}
                        >
                          <DeleteIcon />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>
      <TablePagination
        rowsPerPageOptions={[5, 10, 25, 50]}
        component="div"
        count={workflows.length}
        rowsPerPage={rowsPerPage}
        page={page}
        onPageChange={handleChangePage}
        onRowsPerPageChange={handleChangeRowsPerPage}
      />
    </Paper>
  );
};

export default WorkflowTable;