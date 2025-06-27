"""
Tool Requirement Analyzer

Analyzes user questions to determine specific tool requirements
and matches them with available MCP tools for optimal agent selection.
"""

import re
import logging
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class ToolRequirement:
    """A specific tool requirement extracted from the question"""
    tool_type: str
    required_capabilities: List[str]
    priority: str  # high, medium, low
    alternatives: List[str]
    confidence: float

@dataclass
class ToolAnalysis:
    """Complete tool analysis for a question"""
    requirements: List[ToolRequirement]
    available_matches: Dict[str, List[str]]  # requirement -> matching tools
    missing_capabilities: List[str]
    tool_score: float
    analysis_confidence: float

class ToolRequirementAnalyzer:
    """Analyzes questions to determine tool requirements"""
    
    def __init__(self):
        # Enhanced tool patterns with more specific capability detection
        self.tool_patterns = {
            'search': {
                'keywords': [
                    'search', 'find', 'lookup', 'google', 'web', 'internet', 'query',
                    'research', 'investigate', 'discover', 'locate', 'browse'
                ],
                'capabilities': ['web_search', 'information_retrieval', 'online_research'],
                'alternatives': ['knowledge_base', 'document_search', 'database_query']
            },
            'document': {
                'keywords': [
                    'document', 'pdf', 'file', 'read', 'extract', 'parse', 'analyze',
                    'docx', 'txt', 'markdown', 'content', 'text', 'report'
                ],
                'capabilities': ['document_reading', 'text_extraction', 'file_parsing'],
                'alternatives': ['text_processing', 'content_analysis']
            },
            'email': {
                'keywords': [
                    'email', 'mail', 'send', 'reply', 'message', 'communication',
                    'inbox', 'compose', 'forward', 'gmail', 'outlook'
                ],
                'capabilities': ['email_sending', 'email_reading', 'communication'],
                'alternatives': ['messaging', 'notification']
            },
            'calendar': {
                'keywords': [
                    'calendar', 'schedule', 'meeting', 'appointment', 'event',
                    'booking', 'availability', 'time', 'date', 'plan'
                ],
                'capabilities': ['calendar_management', 'scheduling', 'event_creation'],
                'alternatives': ['time_planning', 'reminder']
            },
            'data': {
                'keywords': [
                    'data', 'csv', 'excel', 'spreadsheet', 'table', 'database',
                    'import', 'export', 'sql', 'json', 'api'
                ],
                'capabilities': ['data_processing', 'database_access', 'api_integration'],
                'alternatives': ['file_processing', 'data_analysis']
            },
            'image': {
                'keywords': [
                    'image', 'photo', 'picture', 'visual', 'diagram', 'chart',
                    'screenshot', 'graphic', 'png', 'jpg', 'jpeg'
                ],
                'capabilities': ['image_processing', 'visual_analysis', 'image_generation'],
                'alternatives': ['visual_creation', 'diagram_generation']
            },
            'code': {
                'keywords': [
                    'code', 'program', 'script', 'repository', 'github', 'git',
                    'commit', 'pull', 'push', 'branch', 'deploy'
                ],
                'capabilities': ['code_repository', 'version_control', 'deployment'],
                'alternatives': ['file_management', 'text_processing']
            },
            'audio': {
                'keywords': [
                    'audio', 'sound', 'music', 'voice', 'speech', 'recording',
                    'transcribe', 'mp3', 'wav', 'listen'
                ],
                'capabilities': ['audio_processing', 'speech_recognition', 'transcription'],
                'alternatives': ['text_to_speech', 'audio_analysis']
            },
            'video': {
                'keywords': [
                    'video', 'movie', 'clip', 'recording', 'stream', 'youtube',
                    'mp4', 'avi', 'watch', 'play'
                ],
                'capabilities': ['video_processing', 'media_streaming', 'video_analysis'],
                'alternatives': ['media_management', 'content_analysis']
            },
            'collaboration': {
                'keywords': [
                    'team', 'collaborate', 'share', 'workspace', 'slack',
                    'discord', 'chat', 'notify', 'alert'
                ],
                'capabilities': ['team_communication', 'workspace_integration', 'notifications'],
                'alternatives': ['messaging', 'communication']
            },
            'automation': {
                'keywords': [
                    'automate', 'workflow', 'trigger', 'webhook', 'zapier',
                    'integrate', 'connect', 'sync', 'batch'
                ],
                'capabilities': ['workflow_automation', 'system_integration', 'task_scheduling'],
                'alternatives': ['scripting', 'process_automation']
            },
            'analytics': {
                'keywords': [
                    'analytics', 'metrics', 'statistics', 'report', 'dashboard',
                    'track', 'monitor', 'measure', 'kpi', 'insights'
                ],
                'capabilities': ['data_analytics', 'reporting', 'monitoring'],
                'alternatives': ['data_analysis', 'visualization']
            }
        }
        
        # Tool priority indicators
        self.priority_indicators = {
            'high': [
                'must', 'required', 'need', 'essential', 'critical', 'important',
                'urgent', 'immediately', 'asap'
            ],
            'medium': [
                'should', 'would', 'prefer', 'better', 'helpful', 'useful'
            ],
            'low': [
                'could', 'might', 'optional', 'nice', 'maybe', 'consider'
            ]
        }
        
        # Action-specific patterns for more precise detection
        self.action_patterns = {
            'create': ['create', 'make', 'generate', 'build', 'compose', 'write'],
            'read': ['read', 'view', 'check', 'examine', 'analyze', 'review'],
            'update': ['update', 'edit', 'modify', 'change', 'revise', 'alter'],
            'delete': ['delete', 'remove', 'clear', 'erase', 'cleanup'],
            'send': ['send', 'deliver', 'transmit', 'forward', 'share'],
            'get': ['get', 'fetch', 'retrieve', 'obtain', 'download', 'extract'],
            'list': ['list', 'show', 'display', 'enumerate', 'catalog'],
            'search': ['search', 'find', 'lookup', 'query', 'filter']
        }
    
    def analyze_tool_requirements(self, question: str, 
                                available_tools: Dict[str, Dict] = None) -> ToolAnalysis:
        """Analyze the question to determine tool requirements"""
        
        question_lower = question.lower()
        words = re.findall(r'\b\w+\b', question_lower)
        
        # Extract tool requirements
        requirements = []
        
        for tool_type, pattern_data in self.tool_patterns.items():
            requirement = self._analyze_tool_type(
                question_lower, words, tool_type, pattern_data
            )
            if requirement:
                requirements.append(requirement)
        
        # Match requirements with available tools
        available_matches = {}
        missing_capabilities = []
        
        if available_tools:
            for req in requirements:
                matches = self._find_matching_tools(req, available_tools)
                available_matches[req.tool_type] = matches
                
                if not matches:
                    missing_capabilities.extend(req.required_capabilities)
        
        # Calculate overall tool score
        tool_score = self._calculate_tool_score(requirements, available_matches)
        
        # Calculate analysis confidence
        analysis_confidence = self._calculate_analysis_confidence(
            requirements, question, words
        )
        
        return ToolAnalysis(
            requirements=requirements,
            available_matches=available_matches,
            missing_capabilities=missing_capabilities,
            tool_score=tool_score,
            analysis_confidence=analysis_confidence
        )
    
    def _analyze_tool_type(self, question_lower: str, words: List[str], 
                          tool_type: str, pattern_data: Dict) -> Optional[ToolRequirement]:
        """Analyze if a specific tool type is required"""
        
        keywords = pattern_data['keywords']
        capabilities = pattern_data['capabilities']
        alternatives = pattern_data['alternatives']
        
        # Count keyword matches
        keyword_matches = 0
        matched_keywords = []
        
        for keyword in keywords:
            if keyword in question_lower:
                keyword_matches += 1
                matched_keywords.append(keyword)
        
        # Check for action-specific patterns
        action_score = 0
        detected_actions = []
        
        for action, action_words in self.action_patterns.items():
            for action_word in action_words:
                if action_word in words:
                    # Check if this action is relevant to the tool type
                    if self._is_action_relevant(action, tool_type):
                        action_score += 1
                        detected_actions.append(action)
                        break
        
        # Calculate priority
        priority = self._determine_priority(question_lower)
        
        # Calculate confidence
        confidence = 0.0
        if keyword_matches > 0:
            confidence = min(0.9, (keyword_matches / len(keywords)) + (action_score * 0.1))
        
        # Minimum threshold for requirement
        if confidence < 0.1:
            return None
        
        # Determine specific capabilities needed
        required_capabilities = capabilities.copy()
        if detected_actions:
            required_capabilities.extend([f"{action}_{tool_type}" for action in detected_actions])
        
        return ToolRequirement(
            tool_type=tool_type,
            required_capabilities=required_capabilities,
            priority=priority,
            alternatives=alternatives,
            confidence=confidence
        )
    
    def _is_action_relevant(self, action: str, tool_type: str) -> bool:
        """Check if an action is relevant for a tool type"""
        relevance_map = {
            'search': ['search', 'get', 'read'],
            'document': ['read', 'create', 'update', 'get'],
            'email': ['send', 'read', 'create', 'list'],
            'calendar': ['create', 'read', 'update', 'delete', 'list'],
            'data': ['read', 'create', 'update', 'get', 'list'],
            'image': ['create', 'read', 'update', 'get'],
            'code': ['read', 'create', 'update', 'get', 'send'],
            'audio': ['read', 'create', 'get'],
            'video': ['read', 'create', 'get'],
            'collaboration': ['send', 'create', 'read'],
            'automation': ['create', 'update', 'send'],
            'analytics': ['read', 'get', 'list', 'create']
        }
        
        return action in relevance_map.get(tool_type, [])
    
    def _determine_priority(self, question_lower: str) -> str:
        """Determine the priority level of tool requirements"""
        
        for priority, indicators in self.priority_indicators.items():
            for indicator in indicators:
                if indicator in question_lower:
                    return priority
        
        return 'medium'  # Default priority
    
    def _find_matching_tools(self, requirement: ToolRequirement, 
                           available_tools: Dict[str, Dict]) -> List[str]:
        """Find available tools that match the requirement"""
        
        matches = []
        
        for tool_name, tool_info in available_tools.items():
            tool_description = tool_info.get('description', '').lower()
            tool_endpoint = tool_info.get('endpoint', '').lower()
            
            # Check if tool matches any required capability
            for capability in requirement.required_capabilities:
                capability_lower = capability.lower()
                
                # Direct keyword match
                if (capability_lower in tool_name.lower() or
                    capability_lower in tool_description or
                    capability_lower in tool_endpoint):
                    matches.append(tool_name)
                    break
                
                # Partial match for compound capabilities
                capability_parts = capability_lower.split('_')
                if len(capability_parts) > 1:
                    if all(part in tool_description or part in tool_name.lower() 
                          for part in capability_parts):
                        matches.append(tool_name)
                        break
            
            # Check tool type matching
            tool_type_lower = requirement.tool_type.lower()
            if (tool_type_lower in tool_name.lower() or
                tool_type_lower in tool_description):
                if tool_name not in matches:
                    matches.append(tool_name)
        
        return matches
    
    def _calculate_tool_score(self, requirements: List[ToolRequirement], 
                            available_matches: Dict[str, List[str]]) -> float:
        """Calculate overall tool matching score"""
        
        if not requirements:
            return 0.5  # Neutral when no tools required
        
        total_score = 0.0
        total_weight = 0.0
        
        for req in requirements:
            # Weight by priority and confidence
            priority_weight = {'high': 1.0, 'medium': 0.7, 'low': 0.4}[req.priority]
            weight = req.confidence * priority_weight
            
            # Score based on availability
            matches = available_matches.get(req.tool_type, [])
            if matches:
                # Full score if tools are available
                score = 1.0
            else:
                # Partial score if alternatives might work
                score = 0.2
            
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_analysis_confidence(self, requirements: List[ToolRequirement],
                                     question: str, words: List[str]) -> float:
        """Calculate confidence in the tool analysis"""
        
        if not requirements:
            return 0.8  # High confidence when no tools needed
        
        # Base confidence from individual requirements
        avg_confidence = sum(req.confidence for req in requirements) / len(requirements)
        
        # Boost for clear tool indicators
        tool_indicators = ['using', 'with', 'via', 'through', 'by']
        if any(indicator in question.lower() for indicator in tool_indicators):
            avg_confidence += 0.1
        
        # Boost for specific tool mentions
        specific_tools = ['gmail', 'google', 'excel', 'github', 'slack', 'calendar']
        if any(tool in question.lower() for tool in specific_tools):
            avg_confidence += 0.15
        
        # Boost for action words
        action_words = [word for action_list in self.action_patterns.values() 
                       for word in action_list]
        action_matches = sum(1 for word in words if word in action_words)
        if action_matches > 0:
            avg_confidence += min(0.1, action_matches * 0.02)
        
        return min(1.0, avg_confidence)
    
    def get_tool_recommendations(self, analysis: ToolAnalysis, 
                               available_tools: Dict[str, Dict]) -> Dict[str, List[str]]:
        """Get tool recommendations based on analysis"""
        
        recommendations = {}
        
        for req in analysis.requirements:
            tool_type = req.tool_type
            
            # Get direct matches
            direct_matches = analysis.available_matches.get(tool_type, [])
            
            # Get alternative tools
            alternatives = []
            for alt_type in req.alternatives:
                for tool_name, tool_info in available_tools.items():
                    if (alt_type in tool_name.lower() or 
                        alt_type in tool_info.get('description', '').lower()):
                        if tool_name not in direct_matches:
                            alternatives.append(tool_name)
            
            recommendations[tool_type] = {
                'primary': direct_matches,
                'alternatives': alternatives[:3],  # Top 3 alternatives
                'priority': req.priority,
                'confidence': req.confidence
            }
        
        return recommendations