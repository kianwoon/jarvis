import React, { useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Grid,
  Typography,
  Paper,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert,
  LinearProgress,
  Chip,
  Divider,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  IconButton,
  SelectChangeEvent,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Skeleton,
  CircularProgress
} from '@mui/material';
import {
  CloudUpload,
  PlayArrow,
  Description,
  TextFields,
  Notes,
  Quiz,
  Article,
  Speed,
  ExpandMore as ExpandMoreIcon,
  Visibility as PreviewIcon,
  AccessTime as TimeIcon
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import { MessageContent } from '../shared/MessageContent';

interface ReferenceDocument {
  document_id: string;  // Primary identifier from API - no 'id' field
  name: string;
  document_type: string;
}

interface ReferenceDocumentContent {
  document_id: string;
  name: string;
  document_type: string;
  category?: string;
  content: string;
  content_preview: string;
  content_stats: {
    total_characters: number;
    total_words: number;
    total_lines: number;
    heading_count: number;
    code_block_count: number;
    list_item_count: number;
    estimated_reading_time_minutes: number;
  };
}

interface ValidationSession {
  session_id: string;
  reference_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  extraction_mode: string;
  total_units: number;
  units_processed: number;
}

interface IDCValidationPanelProps {
  references: ReferenceDocument[];
  onValidationStart?: (session: ValidationSession) => void;
}

type ExtractionMode = 'sentence' | 'paragraph' | 'qa_pairs' | 'section';

interface ExtractionModeInfo {
  icon: React.ReactElement;
  title: string;
  description: string;
  useCase: string;
  contextUsage: string;
  pros: string[];
  cons: string[];
  estimatedUnits?: number;
}

const IDCValidationPanel: React.FC<IDCValidationPanelProps> = ({
  references,
  onValidationStart
}) => {
  const [selectedReference, setSelectedReference] = useState<string>('');
  const [extractionMode, setExtractionMode] = useState<ExtractionMode>('paragraph');
  const [inputFile, setInputFile] = useState<File | null>(null);
  const [validating, setValidating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [previewUnits, setPreviewUnits] = useState<any[]>([]);
  const [showPreview, setShowPreview] = useState(false);
  
  // Reference document content viewer
  const [referenceContent, setReferenceContent] = useState<ReferenceDocumentContent | null>(null);
  const [loadingContent, setLoadingContent] = useState(false);
  const [showReferenceContent, setShowReferenceContent] = useState(false);
  

  const extractionModes: Record<ExtractionMode, ExtractionModeInfo> = {
    sentence: {
      icon: <TextFields />,
      title: "Sentence-by-Sentence",
      description: "Maximum precision - every sentence validated individually",
      useCase: "Critical compliance, legal document review, detailed analysis",
      contextUsage: "~10% per validation",
      pros: [
        "Nothing gets missed",
        "Maximum precision",
        "Detailed feedback per sentence",
        "Best for critical compliance"
      ],
      cons: [
        "Longer processing time",
        "Higher cost",
        "May lose some context"
      ],
      estimatedUnits: 0
    },
    paragraph: {
      icon: <Notes />,
      title: "Paragraph-by-Paragraph",
      description: "Balanced approach - systematic paragraph validation",
      useCase: "General document review, resume screening, content validation",
      contextUsage: "~14% per validation",
      pros: [
        "Good balance of speed and precision",
        "Maintains paragraph context",
        "Reasonable processing time",
        "Systematic coverage"
      ],
      cons: [
        "May miss sentence-level details",
        "Less granular than sentence mode"
      ],
      estimatedUnits: 0
    },
    qa_pairs: {
      icon: <Quiz />,
      title: "Question & Answer Extraction",
      description: "Perfect for exams and assessments",
      useCase: "Exam grading, homework assessment, Q&A validation",
      contextUsage: "~11% per validation",
      pros: [
        "Perfect for structured Q&A",
        "Precise answer matching",
        "Automated grading",
        "Clear scoring per question"
      ],
      cons: [
        "Only works with Q&A format",
        "Requires structured content",
        "May not detect all Q&A patterns"
      ],
      estimatedUnits: 0
    },
    section: {
      icon: <Article />,
      title: "Section-Level Extraction",
      description: "Process document by logical sections",
      useCase: "Policy documents, technical manuals, structured reports",
      contextUsage: "~18% per validation",
      pros: [
        "Preserves document structure",
        "Good for hierarchical docs",
        "Maintains section context",
        "Efficient for large documents"
      ],
      cons: [
        "Less granular validation",
        "May miss details within sections",
        "Depends on document structure"
      ],
      estimatedUnits: 0
    }
  };



  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        setInputFile(acceptedFiles[0]);
        estimateUnits(acceptedFiles[0]);
      }
    },
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/msword': ['.doc'],
      'text/plain': ['.txt'],
      'text/markdown': ['.md']
    },
    maxFiles: 1
  });

  const estimateUnits = async (file: File) => {
    // Estimate units based on file size and extraction mode
    const fileSizeKB = file.size / 1024;
    const estimates = {
      sentence: Math.floor(fileSizeKB * 2),  // Rough estimate
      paragraph: Math.floor(fileSizeKB * 0.5),
      qa_pairs: Math.floor(fileSizeKB * 0.3),
      section: Math.floor(fileSizeKB * 0.1)
    };

    // Update estimates in extraction modes
    Object.keys(extractionModes).forEach(mode => {
      extractionModes[mode as ExtractionMode].estimatedUnits = estimates[mode as ExtractionMode];
    });
  };

  const fetchReferenceContent = async (referenceId: string) => {
    if (!referenceId) return;
    
    setLoadingContent(true);
    try {
      const response = await fetch(`/api/v1/idc/references/${referenceId}/content`);
      if (!response.ok) {
        throw new Error(`Failed to fetch content: ${response.statusText}`);
      }
      
      const data = await response.json();
      if (data.status === 'success') {
        // Extract the content data from the API response wrapper
        const contentData: ReferenceDocumentContent = {
          document_id: data.document_id,
          name: data.name,
          document_type: data.document_type,
          category: data.category,
          content: data.content,
          content_preview: data.content_preview,
          content_stats: data.content_stats
        };
        setReferenceContent(contentData);
        // Don't modify showReferenceContent here - let handleReferenceSelection control it
      } else {
        throw new Error(data.message || 'Failed to load reference content');
      }
    } catch (error) {
      console.error('Failed to fetch reference content:', error);
      setError(error instanceof Error ? error.message : 'Failed to load reference document content');
      setReferenceContent(null);
      setShowReferenceContent(false);
    } finally {
      setLoadingContent(false);
    }
  };

  const handleReferenceSelection = (referenceId: string) => {
    setSelectedReference(referenceId);
    if (referenceId) {
      // Always show the accordion when a reference is selected
      setShowReferenceContent(true);
      fetchReferenceContent(referenceId);
    } else {
      setReferenceContent(null);
      setShowReferenceContent(false);
    }
  };

  const handlePreviewExtraction = async () => {
    if (!inputFile) {
      setError('Please upload a document first');
      return;
    }

    try {
      const formData = new FormData();
      formData.append('file', inputFile);
      formData.append('extraction_mode', extractionMode);
      formData.append('preview_only', 'true');

      const response = await fetch('/api/v1/idc/preview/extraction', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Preview failed');
      }

      const data = await response.json();
      setPreviewUnits(data.units.slice(0, 5));
      setShowPreview(true);
    } catch (error) {
      console.error('Preview failed:', error);
      setError('Failed to preview extraction');
    }
  };

  const handleStartValidation = async () => {
    if (!selectedReference || !inputFile) {
      setError('Please select a reference document and upload an input file');
      return;
    }

    setValidating(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', inputFile);
      formData.append('reference_id', selectedReference);
      formData.append('extraction_mode', extractionMode);

      const response = await fetch('/api/v1/idc/validate/granular', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Validation failed to start');
      }

      const session = await response.json();
      
      if (onValidationStart) {
        onValidationStart(session);
      }

      // Reset form
      setInputFile(null);
      setShowPreview(false);
      setPreviewUnits([]);
    } catch (error) {
      console.error('Validation failed:', error);
      setError(error instanceof Error ? error.message : 'Failed to start validation');
    } finally {
      setValidating(false);
    }
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Document Validation
      </Typography>
      
      {error && (
        <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Step 1: Select Reference Document */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Step 1: Select Reference Document
        </Typography>
        <FormControl fullWidth>
          <InputLabel>Reference Document</InputLabel>
          <Select
            value={selectedReference}
            onChange={(e: SelectChangeEvent) => handleReferenceSelection(e.target.value)}
            label="Reference Document"
            disabled={validating}
          >
            {references.map(ref => (
              <MenuItem key={ref.document_id} value={ref.document_id}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Typography>{ref.name}</Typography>
                  <Chip label={ref.document_type} size="small" />
                </Box>
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Paper>

      {/* Reference Document Content Viewer */}
      {selectedReference && (
        <Paper sx={{ p: 3, mb: 3 }}>
          <Accordion 
            expanded={showReferenceContent} 
            onChange={() => setShowReferenceContent(!showReferenceContent)}
            sx={{ backgroundColor: 'transparent', boxShadow: 'none' }}
          >
            <AccordionSummary
              expandIcon={<ExpandMoreIcon />}
              sx={{ px: 0, minHeight: 'auto' }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <PreviewIcon color="primary" />
                <Typography variant="h6">
                  Reference Document Preview
                </Typography>
                {loadingContent && <CircularProgress size={16} />}
                {referenceContent && (
                  <Chip 
                    label={`${referenceContent.content_stats.estimated_reading_time_minutes} min read`}
                    size="small" 
                    color="info"
                  />
                )}
              </Box>
            </AccordionSummary>
            
            <AccordionDetails sx={{ px: 0 }}>
              {loadingContent ? (
                <Box sx={{ py: 2 }}>
                  <Skeleton variant="text" width="80%" />
                  <Skeleton variant="text" width="60%" />
                  <Skeleton variant="text" width="90%" />
                  <Skeleton variant="rectangular" height={100} sx={{ mt: 2 }} />
                </Box>
              ) : referenceContent ? (
                <Box>
                  {/* Content Stats */}
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Description color="primary" />
                      {referenceContent.name}
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                      <Chip 
                        label={`${referenceContent.content_stats.total_words.toLocaleString()} words`}
                        size="small" 
                        variant="outlined"
                      />
                      <Chip 
                        label={`${referenceContent.content_stats.total_lines.toLocaleString()} lines`}
                        size="small" 
                        variant="outlined"
                      />
                      <Chip 
                        label={`${referenceContent.content_stats.heading_count} headings`}
                        size="small" 
                        variant="outlined"
                      />
                      {referenceContent.content_stats.code_block_count > 0 && (
                        <Chip 
                          label={`${referenceContent.content_stats.code_block_count} code blocks`}
                          size="small" 
                          variant="outlined"
                        />
                      )}
                      {referenceContent.content_stats.list_item_count > 0 && (
                        <Chip 
                          label={`${referenceContent.content_stats.list_item_count} list items`}
                          size="small" 
                          variant="outlined"
                        />
                      )}
                      <Chip 
                        icon={<TimeIcon fontSize="small" />}
                        label={`${referenceContent.content_stats.estimated_reading_time_minutes} min read`}
                        size="small" 
                        color="info"
                      />
                    </Box>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Type: {referenceContent.document_type}
                      {referenceContent.category && ` • Category: ${referenceContent.category}`}
                    </Typography>
                  </Box>

                  {/* Rendered Markdown Content */}
                  <Paper 
                    elevation={0} 
                    sx={{ 
                      p: 3, 
                      backgroundColor: 'background.default',
                      border: '1px solid',
                      borderColor: 'divider',
                      borderRadius: 2,
                      maxHeight: '600px',
                      overflow: 'auto'
                    }}
                  >
                    <MessageContent content={referenceContent.content} />
                  </Paper>
                </Box>
              ) : (
                <Alert severity="info">
                  Select a reference document to preview its content
                </Alert>
              )}
            </AccordionDetails>
          </Accordion>
        </Paper>
      )}

      {/* Step 2: Choose Extraction Mode */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Step 2: Choose How to Extract Content
        </Typography>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          This determines how your document will be broken down for validation
        </Typography>
        
        <Grid container spacing={2} sx={{ mt: 1 }}>
          {Object.entries(extractionModes).map(([mode, info]) => (
            <Grid item xs={12} sm={6} key={mode}>
              <Card
                variant={extractionMode === mode ? "outlined" : "elevation"}
                sx={{
                  cursor: 'pointer',
                  border: extractionMode === mode ? '2px solid' : '1px solid',
                  borderColor: extractionMode === mode ? 'primary.main' : 'grey.300',
                  transition: 'all 0.3s',
                  '&:hover': {
                    borderColor: 'primary.main',
                    boxShadow: 2
                  }
                }}
                onClick={() => setExtractionMode(mode as ExtractionMode)}
              >
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                    {info.icon}
                    <Typography variant="subtitle1" fontWeight="bold">
                      {info.title}
                    </Typography>
                  </Box>
                  
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    {info.description}
                  </Typography>
                  
                  <Divider sx={{ my: 1 }} />
                  
                  <Typography variant="caption" display="block" gutterBottom>
                    <strong>Best for:</strong> {info.useCase}
                  </Typography>
                  
                  <Typography variant="caption" display="block" gutterBottom>
                    <strong>Context usage:</strong> {info.contextUsage}
                  </Typography>
                  
                  {inputFile && info.estimatedUnits !== undefined && (
                    <Chip
                      label={`~${info.estimatedUnits} units`}
                      size="small"
                      color="info"
                      sx={{ mt: 1 }}
                    />
                  )}
                  
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="caption" color="success.main">
                      <strong>Pros:</strong>
                    </Typography>
                    {info.pros.slice(0, 2).map(pro => (
                      <Typography key={pro} variant="caption" display="block" sx={{ pl: 1 }}>
                        ✓ {pro}
                      </Typography>
                    ))}
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Paper>

      {/* Step 3: Upload Document to Validate */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Step 3: Upload Document to Validate
        </Typography>
        
        <Box
          {...getRootProps()}
          sx={{
            mt: 2,
            p: 3,
            border: '2px dashed',
            borderColor: isDragActive ? 'primary.main' : 'grey.400',
            borderRadius: 1,
            textAlign: 'center',
            cursor: validating ? 'not-allowed' : 'pointer',
            backgroundColor: isDragActive ? 'action.hover' : 'background.paper',
            opacity: validating ? 0.6 : 1
          }}
        >
          <input {...getInputProps()} disabled={validating} />
          <CloudUpload sx={{ fontSize: 48, color: 'text.secondary', mb: 1 }} />
          {inputFile ? (
            <Box>
              <Typography variant="h6">{inputFile.name}</Typography>
              <Typography variant="body2" color="text.secondary">
                {(inputFile.size / 1024 / 1024).toFixed(2)} MB
              </Typography>
            </Box>
          ) : (
            <Typography color="text.secondary">
              {isDragActive ? 'Drop the file here' : 'Drag & drop or click to select'}
            </Typography>
          )}
        </Box>
        
        {inputFile && (
          <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
            <Button
              variant="outlined"
              onClick={handlePreviewExtraction}
              disabled={validating}
            >
              Preview Extraction
            </Button>
          </Box>
        )}
      </Paper>


      {/* Preview Panel */}
      {showPreview && previewUnits.length > 0 && (
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Extraction Preview ({extractionMode})
          </Typography>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            First 5 units that will be extracted and validated:
          </Typography>
          
          {previewUnits.map((unit, index) => (
            <Box key={index} sx={{ mb: 2, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="subtitle2">
                  Unit {index + 1} ({unit.type})
                </Typography>
                <Chip
                  label={`~${unit.estimated_context_usage}% context`}
                  size="small"
                  color={unit.estimated_context_usage < 20 ? 'success' : 'warning'}
                />
              </Box>
              <Typography variant="body2">
                {unit.content.substring(0, 200)}...
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Tokens: {unit.metadata?.token_count || 'N/A'}
              </Typography>
            </Box>
          ))}
        </Paper>
      )}

      {/* Start Validation */}
      <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2 }}>
        <Button
          variant="contained"
          size="large"
          startIcon={validating ? <Speed /> : <PlayArrow />}
          onClick={handleStartValidation}
          disabled={!selectedReference || !inputFile || validating}
          sx={{ minWidth: 200 }}
        >
          {validating ? 'Starting Validation...' : 'Start Systematic Validation'}
        </Button>
      </Box>
      
      {validating && (
        <Box sx={{ mt: 3 }}>
          <LinearProgress />
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1, textAlign: 'center' }}>
            Initializing validation session...
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default IDCValidationPanel;