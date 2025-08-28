"""
Universal Progressive Response Service
Handles streaming responses for large datasets in ANY format (table, list, summary, analysis).
"""

import json
import logging
from typing import List, Dict, Any, AsyncGenerator, Optional
from datetime import datetime

from app.models.notebook_models import ProjectData
from app.core.large_generation_utils import LargeGenerationConfigAccessor

logger = logging.getLogger(__name__)

class ProgressiveResponseService:
    """Universal service for generating progressive responses in any format"""
    
    def __init__(self):
        # Load configuration from large generation utils
        try:
            from app.core.timeout_settings_cache import get_timeout_value
            config_dict = {}  # Use empty dict as fallback
            self.config = LargeGenerationConfigAccessor(config_dict)
            self.batch_size = self.config.default_chunk_size  # 15 from config
        except Exception as e:
            logger.warning(f"[PROGRESSIVE_RESPONSE] Could not load config: {e}, using defaults")
            self.batch_size = 20  # Safe fallback
    
    def detect_response_format(self, query: str) -> str:
        """Detect requested format from user query"""
        query_lower = query.lower()
        
        # Table indicators
        if any(word in query_lower for word in ['table', 'counter', 'tabulate', 'columns', 'row']):
            return 'table'
        
        # List indicators  
        elif any(word in query_lower for word in ['list', 'enumerate', 'bullet', 'numbered', 'listing']):
            return 'list'
        
        # Summary indicators
        elif any(word in query_lower for word in ['summary', 'overview', 'brief', 'summarize', 'summarise']):
            return 'summary'
            
        # Analysis indicators
        elif any(word in query_lower for word in ['analyze', 'analysis', 'insights', 'breakdown', 'analyse']):
            return 'analysis'
        
        # Default to table for comprehensive queries with "all"
        if any(word in query_lower for word in ['all', 'complete', 'comprehensive', 'full']):
            return 'table'
        
        # Default fallback
        return 'table'
    
    async def generate_progressive_stream(
        self,
        data: List[ProjectData],
        query: str,
        notebook_id: str,
        conversation_id: str,
        batch_size: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """Generate progressive stream in detected format"""
        
        if batch_size:
            self.batch_size = batch_size
            
        # Detect format from query
        format_type = self.detect_response_format(query)
        total_count = len(data)
        
        logger.info(f"[PROGRESSIVE_RESPONSE] Starting {format_type} format streaming for {total_count} items")
        
        try:
            # Phase 1: Stream header
            yield await self._generate_header(format_type, total_count, notebook_id, conversation_id)
            
            # Phase 2: Stream data in batches
            processed_count = 0
            total_batches = (total_count + self.batch_size - 1) // self.batch_size
            
            for batch_start in range(0, total_count, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total_count)
                batch_data = data[batch_start:batch_end]
                current_batch = batch_start // self.batch_size + 1
                
                # Send status update before processing batch
                if current_batch > 1:  # Don't send status for first batch
                    percentage = (processed_count / total_count) * 100
                    status_message = f"Processing batch {current_batch}/{total_batches}... ({processed_count}/{total_count} items, {percentage:.1f}%)"
                    yield json.dumps({"type": "status", "message": status_message})
                
                batch_response = await self._generate_batch(
                    format_type, 
                    batch_data, 
                    batch_start + 1,  # Start numbering from 1
                    processed_count,
                    total_count
                )
                
                processed_count += len(batch_data)
                yield batch_response
                
                logger.debug(f"[PROGRESSIVE_RESPONSE] Streamed {format_type} batch {current_batch}: "
                           f"items {batch_start + 1}-{batch_end} ({processed_count}/{total_count})")
            
            # Phase 3: Send final status update and footer
            final_status = f"Completed! Processed all {total_count} items in {total_batches} batches."
            yield json.dumps({"type": "status", "message": final_status})
            
            yield await self._generate_footer(format_type, total_count, notebook_id, conversation_id)
            
            logger.info(f"[PROGRESSIVE_RESPONSE] Completed {format_type} streaming: {total_count} items")
            
        except Exception as e:
            logger.error(f"[PROGRESSIVE_RESPONSE] Error generating {format_type} stream: {str(e)}")
            yield json.dumps({
                "answer": f"Error generating {format_type} response: {str(e)}"
            })
    
    async def _generate_header(self, format_type: str, total_count: int, notebook_id: str, conversation_id: str) -> str:
        """Generate format-specific header"""
        
        if format_type == 'table':
            content = f"# Complete Project Table ({total_count} projects)\n\n"
            content += "| # | Project | Company | Year | Description |\n"
            content += "|---|---------|---------|------|-------------|\n"
            
        elif format_type == 'list':
            content = f"# Complete Project List ({total_count} projects)\n\n"
            
        elif format_type == 'summary':
            content = f"# Project Summary ({total_count} projects)\n\n"
            content += "## Overview\n"
            
        elif format_type == 'analysis':
            content = f"# Project Analysis ({total_count} projects)\n\n"
            content += "## Key Insights\n"
            
        else:
            # Default table format
            content = f"# Complete Results ({total_count} items)\n\n"
            content += "| # | Project | Company | Year | Description |\n"
            content += "|---|---------|---------|------|-------------|\n"
        
        # Return content in frontend-compatible format
        return json.dumps({"chunk": content})
    
    async def _generate_batch(
        self, 
        format_type: str, 
        batch_data: List[ProjectData], 
        start_number: int,
        processed_count: int,
        total_count: int
    ) -> str:
        """Generate format-specific batch content"""
        
        content = ""
        
        if format_type == 'table':
            content = self._generate_table_batch(batch_data, start_number)
            
        elif format_type == 'list':
            content = self._generate_list_batch(batch_data, start_number)
            
        elif format_type == 'summary':
            content = self._generate_summary_batch(batch_data, start_number)
            
        elif format_type == 'analysis':
            content = self._generate_analysis_batch(batch_data, start_number)
            
        else:
            # Default to table
            content = self._generate_table_batch(batch_data, start_number)
        
        # Return content in frontend-compatible format
        return json.dumps({"chunk": content})
    
    def _generate_table_batch(self, batch_data: List[ProjectData], start_number: int) -> str:
        """Generate table format batch"""
        content = ""
        
        for i, project in enumerate(batch_data):
            project_num = start_number + i
            
            # Format project data safely
            name = self._format_cell_content(project.name, 50) if project.name else "N/A"
            company = self._format_cell_content(project.company, 30) if project.company else "Not specified"
            year = str(project.year) if project.year else "Not specified"
            description = self._format_cell_content(project.description, 100) if project.description else "No description"
            
            content += f"| {project_num} | {name} | {company} | {year} | {description} |\n"
        
        return content
    
    def _generate_list_batch(self, batch_data: List[ProjectData], start_number: int) -> str:
        """Generate list format batch"""
        content = ""
        
        for i, project in enumerate(batch_data):
            project_num = start_number + i
            
            name = project.name if project.name else "Unnamed Project"
            company = f" at {project.company}" if project.company else ""
            year = f" ({project.year})" if project.year else ""
            description = f": {project.description}" if project.description else ""
            
            content += f"{project_num}. **{name}**{company}{year}{description}\n\n"
        
        return content
    
    def _generate_summary_batch(self, batch_data: List[ProjectData], start_number: int) -> str:
        """Generate summary format batch"""
        content = ""
        
        # Group by company for summary
        companies = {}
        for project in batch_data:
            if project.company:
                companies[project.company] = companies.get(project.company, 0) + 1
        
        if companies:
            content += f"**Projects {start_number}-{start_number + len(batch_data) - 1}:**\n"
            for company, count in companies.items():
                content += f"- {company}: {count} projects\n"
            content += "\n"
        
        return content
    
    def _generate_analysis_batch(self, batch_data: List[ProjectData], start_number: int) -> str:
        """Generate analysis format batch"""
        content = ""
        
        # Analyze years and companies
        years = [p.year for p in batch_data if p.year]
        companies = set(p.company for p in batch_data if p.company)
        
        if years:
            year_range = f"{min(years)}-{max(years)}" if len(set(years)) > 1 else str(years[0])
            content += f"**Batch {start_number//self.batch_size + 1} Analysis ({len(batch_data)} projects):**\n"
            content += f"- Time period: {year_range}\n"
            content += f"- Companies involved: {len(companies)}\n"
            content += f"- Key companies: {', '.join(list(companies)[:3])}\n\n"
        
        return content
    
    async def _generate_footer(self, format_type: str, total_count: int, notebook_id: str, conversation_id: str) -> str:
        """Generate format-specific footer"""
        
        if format_type == 'table':
            content = f"\n**Summary**: {total_count} projects listed above\n"
            
        elif format_type == 'list':
            content = f"\n---\n**Total**: {total_count} projects listed\n"
            
        elif format_type == 'summary':
            content = f"\n## Summary\n**Total Projects**: {total_count}\n"
            
        elif format_type == 'analysis':
            content = f"\n## Conclusion\nAnalyzed {total_count} projects across multiple companies and time periods.\n"
            
        else:
            content = f"\n**Total**: {total_count} items processed\n"
        
        content += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        # Return final content in frontend-compatible format
        return json.dumps({"answer": content})
    
    def _format_cell_content(self, content: str, max_length: int) -> str:
        """Format content for display with length limit"""
        if not content:
            return "N/A"
        
        content = content.strip()
        if len(content) <= max_length:
            return content
        
        return content[:max_length-3] + "..."
    
    async def should_use_progressive_response(
        self, 
        data: List[ProjectData], 
        query: str,
        min_items: Optional[int] = None
    ) -> bool:
        """Determine if progressive response should be used"""
        
        if min_items is None:
            try:
                min_items = self.config.min_items_for_chunking  # 20 from config
            except:
                min_items = 20  # Fallback
        
        if not data or len(data) < min_items:
            return False
        
        logger.debug(f"[PROGRESSIVE_RESPONSE] Should use progressive response: "
                    f"items={len(data)}, min_threshold={min_items}")
        
        return True

# Global instance
progressive_response_service = ProgressiveResponseService()