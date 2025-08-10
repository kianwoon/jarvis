"""
Temporal Context Manager for LLM Time Awareness

Provides comprehensive date/time awareness for the LLM system including:
- Automatic current time injection into system prompts
- Time-related query detection and tool suggestions
- Temporal context filtering for conversations
- Business hours and timezone management

This module follows the zero-hardcoding principle by using configuration
from environment variables and settings cache.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from app.core.datetime_fallback import get_current_datetime
from app.core.config import get_settings

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    from backports.zoneinfo import ZoneInfo  # For Python < 3.9

logger = logging.getLogger(__name__)

class TemporalContextManager:
    """
    Manages temporal awareness across the LLM system with configurable settings
    """
    
    def __init__(self):
        """Initialize temporal context manager with configuration from settings"""
        self.settings = get_settings()
        
        # Get timezone configuration (no hardcoding)
        self.default_timezone = self._get_timezone_config()
        
        # Get business hours configuration
        self.business_hours = self._get_business_hours_config()
        
        # Time-related query patterns (configurable via environment)
        self.time_keywords = self._get_time_keywords()
        
        # Context refresh settings
        self.context_refresh_intervals = self._get_context_refresh_config()
        
    def _get_timezone_config(self) -> str:
        """Get timezone configuration from environment/settings with fallback"""
        try:
            import os
            
            # Try environment variable first
            timezone = os.environ.get("SYSTEM_TIMEZONE")
            if timezone:
                # Validate timezone
                ZoneInfo(timezone)
                return timezone
                
            # Try settings
            if hasattr(self.settings, 'system_timezone') and self.settings.system_timezone:
                ZoneInfo(self.settings.system_timezone)
                return self.settings.system_timezone
                
            # Default to Singapore (as specified in requirements)
            return "Asia/Singapore"
            
        except Exception as e:
            logger.warning(f"Invalid timezone configuration, falling back to Asia/Singapore: {e}")
            return "Asia/Singapore"
    
    def _get_business_hours_config(self) -> Dict[str, Any]:
        """Get business hours configuration from environment/settings"""
        try:
            import os
            
            # Try environment variables first
            start_hour = os.environ.get("BUSINESS_HOURS_START", "9")
            end_hour = os.environ.get("BUSINESS_HOURS_END", "18")
            work_days = os.environ.get("BUSINESS_WORK_DAYS", "1,2,3,4,5").split(',')
            
            return {
                'start_hour': int(start_hour),
                'end_hour': int(end_hour),
                'work_days': [int(day) for day in work_days],  # Monday=1, Sunday=7
                'timezone': self.default_timezone
            }
            
        except Exception as e:
            logger.warning(f"Failed to load business hours config, using defaults: {e}")
            return {
                'start_hour': 9,
                'end_hour': 18,
                'work_days': [1, 2, 3, 4, 5],  # Mon-Fri
                'timezone': self.default_timezone
            }
    
    def _get_time_keywords(self) -> List[str]:
        """Get time-related keywords from configuration"""
        try:
            import os
            
            # Try environment configuration
            keywords_env = os.environ.get("TIME_KEYWORDS")
            if keywords_env:
                return [k.strip() for k in keywords_env.split(',')]
            
            # Use comprehensive default set
            return [
                'today', 'tomorrow', 'yesterday', 'now', 'current time', 
                'what time', 'what date', 'schedule', 'calendar', 'recent', 
                'latest', 'when', 'time', 'date', 'clock', 'hour', 'minute',
                'morning', 'afternoon', 'evening', 'night', 'business hours',
                'timezone', 'deadline', 'due date', 'appointment', 'meeting',
                'this week', 'next week', 'last week', 'this month', 'next month',
                'this year', 'age', 'duration', 'since when', 'until when'
            ]
            
        except Exception as e:
            logger.warning(f"Failed to load time keywords, using defaults: {e}")
            return ['today', 'tomorrow', 'yesterday', 'now', 'current time', 'what time']
    
    def _get_context_refresh_config(self) -> Dict[str, int]:
        """Get context refresh configuration"""
        try:
            import os
            
            return {
                'conversation_minutes': int(os.environ.get("CONTEXT_REFRESH_CONVERSATION", "30")),
                'system_prompt_minutes': int(os.environ.get("CONTEXT_REFRESH_SYSTEM_PROMPT", "60")),
                'tool_suggestions_minutes': int(os.environ.get("CONTEXT_REFRESH_TOOLS", "15"))
            }
            
        except Exception as e:
            logger.warning(f"Failed to load context refresh config, using defaults: {e}")
            return {
                'conversation_minutes': 30,
                'system_prompt_minutes': 60, 
                'tool_suggestions_minutes': 15
            }
    
    def get_current_time_context(self) -> Dict[str, Any]:
        """
        Get comprehensive current time context for system prompts and conversations
        """
        try:
            # Use existing datetime fallback service (follows zero-hardcoding)
            datetime_info = get_current_datetime()
            
            if 'error' in datetime_info:
                logger.error(f"Failed to get current datetime: {datetime_info['error']}")
                return {
                    'current_time': 'Unable to determine current time',
                    'error': datetime_info['error']
                }
            
            # Parse the datetime for additional context
            try:
                current_dt = datetime.fromisoformat(datetime_info['iso_format'].replace('Z', '+00:00'))
                timezone_dt = current_dt.astimezone(ZoneInfo(self.default_timezone))
            except:
                # Fallback if parsing fails
                timezone_dt = datetime.now(ZoneInfo(self.default_timezone))
            
            # Determine business context
            business_context = self._get_business_context(timezone_dt)
            
            # Build comprehensive context
            context = {
                'current_datetime': datetime_info['current_datetime'],
                'iso_format': datetime_info['iso_format'],
                'date': datetime_info['date'],
                'time': datetime_info['time'],
                'timezone': datetime_info['timezone'],
                'day_of_week': datetime_info['day_of_week'],
                'month': datetime_info['month'],
                'year': datetime_info.get('year', timezone_dt.year),
                'business_context': business_context,
                'temporal_guidance': self._get_temporal_guidance(timezone_dt, business_context)
            }
            
            logger.debug(f"Generated time context: {context['current_datetime']}")
            return context
            
        except Exception as e:
            logger.error(f"Failed to generate time context: {e}")
            return {
                'current_time': 'Unable to determine current time',
                'error': str(e)
            }
    
    def _get_business_context(self, current_time: datetime) -> Dict[str, Any]:
        """Determine business hours context"""
        try:
            weekday = current_time.isoweekday()  # Monday=1, Sunday=7
            hour = current_time.hour
            
            is_business_day = weekday in self.business_hours['work_days']
            is_business_hours = (
                is_business_day and 
                self.business_hours['start_hour'] <= hour < self.business_hours['end_hour']
            )
            
            return {
                'is_business_day': is_business_day,
                'is_business_hours': is_business_hours,
                'business_hours_start': self.business_hours['start_hour'],
                'business_hours_end': self.business_hours['end_hour'],
                'next_business_day': self._get_next_business_day(current_time)
            }
            
        except Exception as e:
            logger.warning(f"Failed to determine business context: {e}")
            return {
                'is_business_day': False,
                'is_business_hours': False,
                'error': str(e)
            }
    
    def _get_next_business_day(self, current_time: datetime) -> str:
        """Calculate next business day"""
        try:
            next_day = current_time + timedelta(days=1)
            while next_day.isoweekday() not in self.business_hours['work_days']:
                next_day += timedelta(days=1)
                # Safety check to prevent infinite loop
                if (next_day - current_time).days > 7:
                    break
            return next_day.strftime("%A, %B %d, %Y")
        except Exception as e:
            logger.warning(f"Failed to calculate next business day: {e}")
            return "Unknown"
    
    def _get_temporal_guidance(self, current_time: datetime, business_context: Dict) -> str:
        """Generate temporal guidance for LLM responses"""
        try:
            guidance_parts = []
            
            # Current time awareness
            guidance_parts.append(f"Current time is {current_time.strftime('%A, %B %d, %Y at %I:%M %p')}")
            
            # Business context
            if business_context.get('is_business_hours'):
                guidance_parts.append("Currently within business hours")
            elif business_context.get('is_business_day'):
                guidance_parts.append("Currently outside business hours but on a business day")
            else:
                next_biz_day = business_context.get('next_business_day', 'Unknown')
                guidance_parts.append(f"Currently outside business hours. Next business day: {next_biz_day}")
            
            return ". ".join(guidance_parts) + "."
            
        except Exception as e:
            logger.warning(f"Failed to generate temporal guidance: {e}")
            return "Current time context unavailable."
    
    def enhance_system_prompt_with_time_context(self, base_prompt: str) -> str:
        """
        Enhance a system prompt with automatic current time context
        """
        try:
            # Get current time context
            time_context = self.get_current_time_context()
            
            if 'error' in time_context:
                logger.warning(f"Could not enhance prompt with time context: {time_context['error']}")
                return base_prompt
            
            # Check if prompt already has time context to avoid duplication
            if self._has_existing_time_context(base_prompt):
                logger.debug("Prompt already contains time context, skipping enhancement")
                return base_prompt
            
            # Build time context section
            time_section = self._build_time_context_section(time_context)
            
            # Insert time context at appropriate location
            enhanced_prompt = self._insert_time_context(base_prompt, time_section)
            
            logger.debug("Successfully enhanced system prompt with time context")
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"Failed to enhance system prompt with time context: {e}")
            return base_prompt
    
    def _has_existing_time_context(self, prompt: str) -> bool:
        """Check if prompt already has COMPLETE time context to avoid duplication"""
        import re
        
        prompt_lower = prompt.lower()
        
        # Don't trigger on just "Now is year YYYY" - this is not complete time context
        if "now is year" in prompt_lower and not any(
            indicator in prompt_lower for indicator in ['date', 'time', 'day', 'month']
        ):
            logger.debug("Found 'Now is year' but no complete date/time - will add temporal context")
            return False
        
        # Check for actual complete time context patterns
        time_indicators = [
            'current date & time:', 'current datetime:', 
            'today\'s date is', 'current time is',
            'current date:', 'today is [a-z]+day',  # Today is Monday/Tuesday/etc
            r'\d{4}-\d{2}-\d{2}',  # Date pattern YYYY-MM-DD
            r'\d{1,2}:\d{2}\s*(am|pm|utc|sgt)',  # Time pattern with timezone
            'business hours', 'timezone:', 'time context:'
        ]
        
        # Check each pattern
        for pattern in time_indicators:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                logger.debug(f"Found existing time context pattern: {pattern}")
                return True
        
        logger.debug("No existing complete time context found - will add temporal context")
        return False
    
    def _build_time_context_section(self, time_context: Dict) -> str:
        """Build the time context section for system prompts"""
        try:
            sections = []
            
            # Current time information - make it VERY clear and prominent
            sections.append(f"**IMPORTANT - Current Date and Time Information:**")
            sections.append(f"- Today's Date: {time_context.get('date', 'Unknown')}")
            sections.append(f"- Current Time: {time_context.get('time', 'Unknown')}")
            sections.append(f"- Full DateTime: {time_context['current_datetime']}")
            sections.append(f"- Day of Week: {time_context['day_of_week']}")
            sections.append(f"- Timezone: {time_context.get('timezone', 'Singapore')}")
            
            # Business context if available
            if 'business_context' in time_context:
                business_ctx = time_context['business_context']
                if business_ctx.get('is_business_hours'):
                    sections.append(f"- Status: Within business hours ({business_ctx.get('business_hours_start', 9)}:00 - {business_ctx.get('business_hours_end', 18)}:00)")
                else:
                    sections.append(f"- Status: Outside business hours")
                    if business_ctx.get('next_business_day'):
                        sections.append(f"- Next Business Day: {business_ctx['next_business_day']}")
            
            # Temporal reasoning guidance - emphasize using actual date/time
            sections.append("")
            sections.append("**Temporal Reasoning Guidelines:**")
            sections.append("- ALWAYS use the current date and time shown above when mentioning dates or times")
            sections.append("- NEVER make up or hallucinate dates - use the actual current date provided")
            sections.append("- For time-sensitive queries, reference the exact date and time from above")
            sections.append("- When discussing events or deadlines, always relate them to today's date shown above")
            sections.append("- Be aware of business hours when suggesting actions or recommendations")
            
            # Log what we're adding
            logger.info(f"[TEMPORAL CONTEXT] Adding time context to prompt - Date: {time_context.get('date')}, Time: {time_context.get('time')}")
            
            return "\n".join(sections)
            
        except Exception as e:
            logger.warning(f"Failed to build time context section: {e}")
            return f"Current Date & Time: {time_context.get('current_datetime', 'Unknown')}"
    
    def _insert_time_context(self, base_prompt: str, time_section: str) -> str:
        """Insert time context at the appropriate location in the prompt"""
        try:
            # Look for common insertion points in order of preference
            insertion_patterns = [
                (r'(\*\*Available Tools:\*\*)', rf'{time_section}\n\n\1'),  # Before tools section
                (r'(\*\*Available Knowledge Collections:\*\*)', rf'{time_section}\n\n\1'),  # Before collections
                (r'(\*\*Guidelines:\*\*)', rf'{time_section}\n\n\1'),  # Before guidelines
                (r'(\*\*Tool Usage Format:\*\*)', rf'{time_section}\n\n\1'),  # Before tool format
                (r'(You can use these tools)', rf'{time_section}\n\n\1'),  # Before tool usage note
            ]
            
            # Try each insertion pattern
            for pattern, replacement in insertion_patterns:
                if re.search(pattern, base_prompt):
                    return re.sub(pattern, replacement, base_prompt, count=1)
            
            # If no specific insertion point found, add at the beginning
            return f"{base_prompt}\n\n{time_section}"
            
        except Exception as e:
            logger.warning(f"Failed to insert time context, appending to end: {e}")
            return f"{base_prompt}\n\n{time_section}"
    
    def detect_time_related_query(self, query: str) -> Tuple[bool, List[str], float]:
        """
        Detect if a query is time-related and should use datetime tools
        
        Returns:
            (is_time_related, matched_keywords, confidence_score)
        """
        try:
            query_lower = query.lower().strip()
            matched_keywords = []
            
            # Direct keyword matching
            for keyword in self.time_keywords:
                if keyword.lower() in query_lower:
                    matched_keywords.append(keyword)
            
            # Pattern-based detection
            time_patterns = [
                r'\bwhat\s+(?:time|date)\b',
                r'\bcurrent\s+(?:time|date|day)\b',
                r'\btoday\'?s?\s+(?:date|time|day)\b',
                r'\btomorrow\s+(?:is|will be)\b',
                r'\byesterday\s+(?:was|were)\b',
                r'\bnow\s+(?:is|it\'s)\b',
                r'\bschedule\b',
                r'\bdeadline\b',
                r'\bdue\s+(?:date|time)\b',
                r'\bappointment\b',
                r'\bmeeting\s+(?:time|date)\b',
                r'\bbusiness\s+hours\b',
                r'\btimezone\b',
                r'\b(?:this|next|last)\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
                r'\b(?:how\s+long|duration|since\s+when|until\s+when)\b'
            ]
            
            pattern_matches = 0
            for pattern in time_patterns:
                if re.search(pattern, query_lower):
                    pattern_matches += 1
            
            # Calculate confidence score
            keyword_score = len(matched_keywords) * 0.3
            pattern_score = pattern_matches * 0.4
            confidence = min(1.0, keyword_score + pattern_score)
            
            is_time_related = len(matched_keywords) > 0 or pattern_matches > 0
            
            if is_time_related:
                logger.debug(f"Time-related query detected: '{query}' (confidence: {confidence:.2f})")
            
            return is_time_related, matched_keywords, confidence
            
        except Exception as e:
            logger.error(f"Failed to detect time-related query: {e}")
            return False, [], 0.0
    
    def should_refresh_context(self, context_type: str, last_refresh: Optional[datetime] = None) -> bool:
        """
        Determine if context should be refreshed based on time elapsed
        """
        if not last_refresh:
            return True
        
        try:
            current_time = datetime.now(ZoneInfo(self.default_timezone))
            time_diff = (current_time - last_refresh).total_seconds() / 60  # Convert to minutes
            
            refresh_interval = self.context_refresh_intervals.get(context_type, 30)
            should_refresh = time_diff >= refresh_interval
            
            if should_refresh:
                logger.debug(f"Context refresh needed for {context_type}: {time_diff:.1f} minutes elapsed")
            
            return should_refresh
            
        except Exception as e:
            logger.warning(f"Failed to determine context refresh status: {e}")
            return True  # Default to refresh on error
    
    def add_temporal_context_to_message(self, message_content: str, role: str = "user") -> str:
        """
        Add temporal context to message content when appropriate
        """
        try:
            # Only add context to user messages and only if time-related
            if role != "user":
                return message_content
            
            is_time_related, _, confidence = self.detect_time_related_query(message_content)
            
            # Only add context for clearly time-related queries
            if not is_time_related or confidence < 0.5:
                return message_content
            
            # Get current time context
            time_context = self.get_current_time_context()
            if 'error' in time_context:
                return message_content
            
            # Add concise temporal context
            temporal_note = f"\n\n[Context: Current time is {time_context['current_datetime']}]"
            
            return message_content + temporal_note
            
        except Exception as e:
            logger.warning(f"Failed to add temporal context to message: {e}")
            return message_content
    
    def get_time_aware_conversation_window(self, conversation_history: List[Dict], current_query: str) -> List[Dict]:
        """
        Filter conversation history based on temporal relevance
        """
        try:
            if not conversation_history:
                return []
            
            # Check if current query is time-related
            is_time_related, _, _ = self.detect_time_related_query(current_query)
            
            if not is_time_related:
                # For non-time queries, use standard filtering
                return conversation_history[-4:]  # Recent context only
            
            # For time-related queries, be more selective about historical context
            current_time = datetime.now(ZoneInfo(self.default_timezone))
            filtered_history = []
            
            for message in conversation_history:
                try:
                    # Parse message timestamp if available
                    if 'timestamp' in message:
                        msg_time = datetime.fromisoformat(message['timestamp'])
                        time_diff = current_time - msg_time
                        
                        # Only include recent messages for time-related queries (within last hour)
                        if time_diff.total_seconds() > 3600:  # 1 hour
                            continue
                    
                    filtered_history.append(message)
                    
                except Exception as msg_e:
                    logger.debug(f"Could not parse message timestamp, including message: {msg_e}")
                    filtered_history.append(message)
            
            # Limit to most recent messages even after filtering
            return filtered_history[-2:]
            
        except Exception as e:
            logger.warning(f"Failed to apply time-aware conversation filtering: {e}")
            return conversation_history[-4:]  # Safe fallback

# Singleton instance following existing patterns in the codebase
_temporal_manager: Optional[TemporalContextManager] = None

def get_temporal_context_manager() -> TemporalContextManager:
    """Get or create the temporal context manager singleton"""
    global _temporal_manager
    if _temporal_manager is None:
        _temporal_manager = TemporalContextManager()
    return _temporal_manager

def reload_temporal_context_manager():
    """Reload the temporal context manager (useful for configuration changes)"""
    global _temporal_manager
    _temporal_manager = None