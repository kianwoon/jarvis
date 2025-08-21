"""
Meta-Task Execution Engine
Handles the actual execution of meta-task phases with LLM integration
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime
from app.core.llm_settings_cache import get_llm_settings
from app.llm.ollama import OllamaLLM
from app.llm.base import LLMConfig
import os

logger = logging.getLogger(__name__)

class MetaTaskExecutionEngine:
    """Handles execution of meta-task phases"""
    
    def __init__(self):
        self.logger = logger
        self.llm_settings = get_llm_settings()
        self.meta_task_settings = self._load_meta_task_settings()
    
    def _load_meta_task_settings(self) -> Dict[str, Any]:
        """Load meta_task settings directly from database"""
        try:
            from app.core.db import SessionLocal, Settings as SettingsModel
            
            db = SessionLocal()
            try:
                row = db.query(SettingsModel).filter(SettingsModel.category == 'meta_task').first()
                if row:
                    return row.settings
                else:
                    self.logger.warning("No meta_task settings found in database")
                    return {}
            finally:
                db.close()
        except Exception as e:
            self.logger.error(f"Failed to load meta_task settings: {e}")
            return {}
    
    async def execute_phase(
        self,
        phase_config: Dict[str, Any],
        input_data: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute a single phase"""
        try:
            phase_type = phase_config.get('type', 'generator')
            phase_name = phase_config.get('name', 'Unknown Phase')
            
            self.logger.info(f"Executing phase: {phase_name} ({phase_type})")
            
            # Refresh settings to get latest configuration
            self.llm_settings = get_llm_settings()
            self.meta_task_settings = self._load_meta_task_settings()
            
            # Build prompt based on phase type
            prompt = self._build_phase_prompt(phase_config, input_data, context)
            
            # Execute with LLM
            result = await self._execute_with_llm(prompt, phase_config)
            
            return {
                'success': True,
                'phase_name': phase_name,
                'phase_type': phase_type,
                'output': result,
                'execution_time_ms': result.get('execution_time_ms', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error executing phase {phase_config.get('name')}: {e}")
            return {
                'success': False,
                'phase_name': phase_config.get('name', 'Unknown'),
                'error_message': str(e)
            }
    
    def _build_phase_prompt(
        self,
        phase_config: Dict[str, Any],
        input_data: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> str:
        """Build prompt for phase execution"""
        phase_type = phase_config.get('type', 'generator')
        phase_name = phase_config.get('name', 'Phase')
        description = phase_config.get('description', 'Execute this phase')
        
        # Get system prompt based on phase type from nested model configurations
        system_prompt = ""
        
        # Map phase types to their corresponding model configurations
        phase_to_model_map = {
            "analyzer": "analyzer_model",
            "reviewer": "reviewer_model",
            "generator": "generator_model",
            "assembler": "assembler_model"
        }
        
        # Try to get system prompt from nested model configuration
        model_key = phase_to_model_map.get(phase_type)
        if model_key:
            model_config = self.meta_task_settings.get(model_key, {})
            if isinstance(model_config, dict):
                system_prompt = model_config.get("system_prompt", "")
        
        # Fallback to root level for backward compatibility (for analyzer only)
        if not system_prompt and phase_type == "analyzer":
            system_prompt = self.meta_task_settings.get("analyzer_system_prompt", "")
        
        # Use phase-specific system prompt if available
        if system_prompt:
            base_prompt = f"""{system_prompt}

PHASE: {phase_name}
DESCRIPTION: {description}

INPUT DATA:
{json.dumps(input_data, indent=2)}
"""
        else:
            # Fallback to default prompts based on phase type
            if phase_type == "analyzer":
                base_prompt = f"""You are an expert Meta-Task Analyzer executing a meta-task phase.

PHASE: {phase_name}
TYPE: {phase_type}
DESCRIPTION: {description}

INPUT DATA:
{json.dumps(input_data, indent=2)}
"""
            elif phase_type == "reviewer":
                base_prompt = f"""You are an expert Meta-Task Reviewer executing a review phase.

PHASE: {phase_name}
TYPE: {phase_type}
DESCRIPTION: {description}

INPUT DATA:
{json.dumps(input_data, indent=2)}
"""
            elif phase_type == "generator":
                base_prompt = f"""You are an expert Meta-Task Generator executing a generation phase.

PHASE: {phase_name}
TYPE: {phase_type}
DESCRIPTION: {description}

INPUT DATA:
{json.dumps(input_data, indent=2)}
"""
            elif phase_type == "assembler":
                base_prompt = f"""You are an expert Meta-Task Assembler executing an assembly phase.

PHASE: {phase_name}
TYPE: {phase_type}
DESCRIPTION: {description}

INPUT DATA:
{json.dumps(input_data, indent=2)}
"""
            else:
                # Use standard prompt structure for unknown phase types
                base_prompt = f"""You are an expert assistant executing a meta-task phase.

PHASE: {phase_name}
TYPE: {phase_type}
DESCRIPTION: {description}

INPUT DATA:
{json.dumps(input_data, indent=2)}
"""
        
        # Add context if available
        if context:
            base_prompt += f"""
PREVIOUS CONTEXT:
{json.dumps(context, indent=2)}
"""
        
        # Add phase-specific instructions (skip for analyzer as it already has comprehensive instructions)
        if phase_type == "analyzer":
            # System prompt already contains comprehensive instructions
            pass
        
        elif phase_type == "generator":
            base_prompt += """
INSTRUCTIONS:
- Generate comprehensive content based on the input and analysis
- Ensure content is well-structured and detailed
- Follow any format requirements specified
- Create content that flows naturally with previous phases
- Return the generated content in a structured format
"""
        
        elif phase_type == "reviewer":
            base_prompt += """
INSTRUCTIONS:
- Review the generated content for quality and completeness
- Check for consistency, clarity, and accuracy
- Identify areas for improvement
- Provide specific suggestions and corrections
- Return a review report with scores and recommendations
"""
        
        elif phase_type == "assembler":
            base_prompt += """
INSTRUCTIONS:
- Assemble all previous phase outputs into a final deliverable
- Ensure proper formatting and structure
- Create smooth transitions between sections
- Finalize the document with proper formatting
- Return the complete assembled document
"""
        
        base_prompt += """
Please execute this phase and return your results in a clear, structured format.
"""
        
        return base_prompt
    
    async def _execute_with_llm(
        self,
        prompt: str,
        phase_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute prompt with LLM"""
        try:
            start_time = datetime.utcnow()
            
            # Get model configuration
            model_config = self.llm_settings.get("main_llm", {})
            
            config = LLMConfig(
                model_name=model_config.get("model", "qwen3:30b-a3b"),
                temperature=phase_config.get("temperature", 0.7),
                top_p=model_config.get("top_p", 0.9),
                max_tokens=phase_config.get("max_tokens", 4000)
            )
            
            # Get model server URL
            ollama_url = model_config.get("model_server")
            if not ollama_url:
                ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            
            # Initialize LLM
            llm = OllamaLLM(config, base_url=ollama_url)
            
            # Generate response
            response_text = ""
            async for response_chunk in llm.generate_stream(prompt):
                response_text += response_chunk.text
            
            # Calculate execution time
            end_time = datetime.utcnow()
            execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Process response
            processed_response = self._process_response(response_text, phase_config)
            
            return {
                'raw_response': response_text,
                'processed_output': processed_response,
                'execution_time_ms': execution_time_ms,
                'model_used': config.model_name,
                'tokens_estimated': len(response_text) // 4  # Rough estimate
            }
            
        except Exception as e:
            self.logger.error(f"Error executing LLM: {e}")
            return {
                'raw_response': '',
                'processed_output': {},
                'error_message': str(e),
                'execution_time_ms': 0
            }
    
    def _process_response(
        self,
        response: str,
        phase_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process and structure the LLM response"""
        phase_type = phase_config.get('type', 'generator')
        
        # Try to extract JSON if present
        try:
            # Look for JSON blocks
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_data = json.loads(json_match.group(1))
                return {
                    'structured_output': json_data,
                    'raw_text': response,
                    'has_json': True
                }
        except:
            pass
        
        # Process based on phase type
        if phase_type == "analyzer":
            return {
                'analysis_text': response,
                'raw_text': response,
                'analysis_type': 'text_based'
            }
        
        elif phase_type == "generator":
            return {
                'generated_content': response,
                'content_length': len(response),
                'word_count': len(response.split()),
                'raw_text': response
            }
        
        elif phase_type == "reviewer":
            return {
                'review_text': response,
                'raw_text': response,
                'review_type': 'qualitative'
            }
        
        elif phase_type == "assembler":
            return {
                'final_document': response,
                'document_length': len(response),
                'word_count': len(response.split()),
                'raw_text': response
            }
        
        # Default processing
        return {
            'output_text': response,
            'raw_text': response,
            'processing_type': 'default'
        }
    
    async def execute_multi_phase_workflow(
        self,
        phases: List[Dict[str, Any]],
        initial_input: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute multiple phases in sequence"""
        workflow_context = {}
        workflow_output = {}
        
        yield {
            'type': 'workflow_started',
            'total_phases': len(phases),
            'initial_input': initial_input
        }
        
        for i, phase_config in enumerate(phases):
            phase_name = phase_config.get('name', f'Phase {i+1}')
            
            yield {
                'type': 'phase_started',
                'phase_number': i + 1,
                'phase_name': phase_name,
                'phase_type': phase_config.get('type', 'unknown')
            }
            
            try:
                # Execute phase
                phase_result = await self.execute_phase(
                    phase_config, 
                    initial_input, 
                    workflow_context
                )
                
                if phase_result['success']:
                    # Update context with phase output
                    workflow_context[f'phase_{i+1}_output'] = phase_result['output']
                    workflow_output[phase_name] = phase_result['output']
                    
                    yield {
                        'type': 'phase_completed',
                        'phase_number': i + 1,
                        'phase_name': phase_name,
                        'execution_time': phase_result.get('execution_time_ms', 0),
                        'output_preview': str(phase_result['output'])[:200] + "..."
                    }
                else:
                    yield {
                        'type': 'phase_failed',
                        'phase_number': i + 1,
                        'phase_name': phase_name,
                        'error': phase_result.get('error_message', 'Unknown error')
                    }
                    return
                    
            except Exception as e:
                yield {
                    'type': 'phase_failed',
                    'phase_number': i + 1,
                    'phase_name': phase_name,
                    'error': str(e)
                }
                return
        
        yield {
            'type': 'workflow_completed',
            'total_phases_completed': len(phases),
            'workflow_output': workflow_output
        }