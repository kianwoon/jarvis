"""
Progressive Table Service
Handles streaming table generation for large datasets without overwhelming the LLM.
"""

import json
import logging
from typing import List, Dict, Any, AsyncGenerator, Optional
from datetime import datetime

from app.models.notebook_rag import ProjectData

logger = logging.getLogger(__name__)

class ProgressiveTableService:
    """Service for generating tables progressively in streaming chunks"""
    
    def __init__(self):
        self.batch_size = 20  # Configurable batch size
    
    async def generate_table_stream(
        self, 
        projects: List[ProjectData], 
        query: str,
        notebook_id: str,
        conversation_id: str,
        batch_size: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate a table in progressive streaming chunks.
        
        Args:
            projects: List of extracted projects
            query: Original query for context
            notebook_id: Notebook ID
            conversation_id: Conversation ID  
            batch_size: Number of projects per batch (default: 20)
            
        Yields:
            JSON strings representing table chunks
        """
        
        if batch_size:
            self.batch_size = batch_size
            
        total_projects = len(projects)
        logger.info(f"[PROGRESSIVE_TABLE] Starting progressive table generation for {total_projects} projects")
        
        try:
            # Phase 1: Send table initialization
            yield await self._generate_table_header(total_projects, notebook_id, conversation_id)
            
            # Phase 2: Process projects in batches
            processed_count = 0
            
            for batch_start in range(0, total_projects, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total_projects)
                batch_projects = projects[batch_start:batch_end]
                
                # Generate batch rows
                batch_data = await self._generate_batch_rows(
                    batch_projects, 
                    batch_start + 1,  # Start numbering from 1
                    processed_count,
                    total_projects
                )
                
                processed_count += len(batch_projects)
                
                yield batch_data
                
                logger.debug(f"[PROGRESSIVE_TABLE] Streamed batch {batch_start//self.batch_size + 1}: "
                           f"projects {batch_start + 1}-{batch_end} ({processed_count}/{total_projects})")
            
            # Phase 3: Send table completion
            yield await self._generate_table_footer(total_projects, notebook_id, conversation_id)
            
            logger.info(f"[PROGRESSIVE_TABLE] Completed progressive table generation: {total_projects} projects")
            
        except Exception as e:
            logger.error(f"[PROGRESSIVE_TABLE] Error generating progressive table: {str(e)}")
            # Send error chunk
            yield json.dumps({
                "type": "table_error",
                "error": f"Error generating table: {str(e)}",
                "notebook_id": notebook_id,
                "conversation_id": conversation_id
            })
    
    async def _generate_table_header(self, total_projects: int, notebook_id: str, conversation_id: str) -> str:
        """Generate table header chunk"""
        
        header_content = f"# Complete Project Table ({total_projects} projects)\n\n"
        header_content += "| # | Project | Company | Year | Description |\n"
        header_content += "|---|---------|---------|------|-------------|\n"
        
        return json.dumps({
            "type": "table_header",
            "content": header_content,
            "total_projects": total_projects,
            "batch_info": {
                "total_batches": (total_projects + self.batch_size - 1) // self.batch_size,
                "batch_size": self.batch_size
            },
            "notebook_id": notebook_id,
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat()
        })
    
    async def _generate_batch_rows(
        self, 
        batch_projects: List[ProjectData], 
        start_number: int,
        processed_count: int,
        total_projects: int
    ) -> str:
        """Generate table rows for a batch of projects"""
        
        rows_content = ""
        
        for i, project in enumerate(batch_projects):
            project_num = start_number + i
            
            # Format project data safely
            name = self._format_cell_content(project.name, 50) if project.name else "N/A"
            company = self._format_cell_content(project.company, 30) if project.company else "Not specified"
            year = str(project.year) if project.year else "Not specified"
            description = self._format_cell_content(project.description, 100) if project.description else "No description"
            
            rows_content += f"| {project_num} | {name} | {company} | {year} | {description} |\n"
        
        return json.dumps({
            "type": "table_batch",
            "content": rows_content,
            "batch_info": {
                "batch_start": start_number,
                "batch_end": start_number + len(batch_projects) - 1,
                "batch_size": len(batch_projects),
                "processed_total": processed_count + len(batch_projects),
                "total_projects": total_projects,
                "progress_percent": ((processed_count + len(batch_projects)) / total_projects * 100)
            },
            "timestamp": datetime.now().isoformat()
        })
    
    async def _generate_table_footer(self, total_projects: int, notebook_id: str, conversation_id: str) -> str:
        """Generate table footer with summary"""
        
        footer_content = f"\n**Summary**: {total_projects} projects listed above\n"
        footer_content += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        return json.dumps({
            "type": "table_footer",
            "content": footer_content,
            "summary": {
                "total_projects": total_projects,
                "completion_status": "success",
                "generated_at": datetime.now().isoformat()
            },
            "notebook_id": notebook_id,
            "conversation_id": conversation_id,
            "status": "complete"
        })
    
    def _format_cell_content(self, content: str, max_length: int) -> str:
        """Format content for table cell with length limit"""
        if not content:
            return "N/A"
        
        content = content.strip()
        if len(content) <= max_length:
            return content
        
        return content[:max_length-3] + "..."
    
    async def should_use_progressive_table(
        self, 
        projects: List[ProjectData], 
        query: str,
        min_projects: int = 20
    ) -> bool:
        """
        Determine if progressive table should be used based on project count and query.
        
        Args:
            projects: List of extracted projects
            query: User query
            min_projects: Minimum projects to trigger progressive mode
            
        Returns:
            True if progressive table should be used
        """
        
        if not projects or len(projects) < min_projects:
            return False
        
        # Check for table-related keywords
        query_lower = query.lower()
        table_keywords = ['table', 'list all', 'counter', 'count', 'format', 'tabulate']
        
        has_table_request = any(keyword in query_lower for keyword in table_keywords)
        
        logger.debug(f"[PROGRESSIVE_TABLE] Should use progressive table: "
                    f"projects={len(projects)}, has_table_request={has_table_request}")
        
        return has_table_request
    
    async def generate_summary_response(
        self,
        projects: List[ProjectData],
        notebook_id: str,
        conversation_id: str
    ) -> str:
        """Generate a summary response for non-table requests with many projects"""
        
        total_count = len(projects)
        
        # Group by company for summary
        companies = {}
        years = []
        
        for project in projects:
            if project.company:
                companies[project.company] = companies.get(project.company, 0) + 1
            if project.year:
                years.append(project.year)
        
        # Generate summary
        summary_content = f"# Project Summary\n\n"
        summary_content += f"Found **{total_count} projects** across {len(companies)} companies.\n\n"
        
        if companies:
            top_companies = sorted(companies.items(), key=lambda x: x[1], reverse=True)[:5]
            summary_content += "## Top Companies:\n"
            for company, count in top_companies:
                summary_content += f"- {company}: {count} projects\n"
            summary_content += "\n"
        
        if years:
            year_range = f"{min(years)}-{max(years)}"
            summary_content += f"## Time Range: {year_range}\n\n"
        
        summary_content += "💡 *Use 'table format' or 'list all projects' to see the complete detailed table.*\n"
        
        return json.dumps({
            "type": "summary_response",
            "content": summary_content,
            "summary": {
                "total_projects": total_count,
                "companies": len(companies),
                "year_range": year_range if years else "Not specified"
            },
            "notebook_id": notebook_id,
            "conversation_id": conversation_id
        })

# Global instance
progressive_table_service = ProgressiveTableService()