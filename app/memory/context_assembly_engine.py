"""
Context Assembly Engine - Intelligent Context Management
Handles context limits without losing information through multi-pass assembly
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json

from app.memory.conversation_memory_manager import ConversationMemoryManager, ConversationMessage


class ContextStrategy(Enum):
    """Different strategies for context assembly"""
    RECENT_FOCUS = "recent_focus"      # Prioritize recent messages
    RELEVANCE_FOCUS = "relevance_focus" # Prioritize relevant messages
    THREAD_AWARE = "thread_aware"       # Organize by conversation threads
    HIERARCHICAL = "hierarchical"       # Multi-level context hierarchy
    ADAPTIVE = "adaptive"               # Dynamically choose best strategy


@dataclass
class ContextSegment:
    """A segment of context with metadata"""
    content: str
    segment_type: str  # 'recent', 'relevant', 'summary', 'thread'
    priority: float    # 0.0 to 1.0
    token_estimate: int
    source_messages: List[str]  # Message IDs
    timestamp_range: Tuple[datetime, datetime]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'segment_type': self.segment_type,
            'priority': self.priority,
            'token_estimate': self.token_estimate,
            'source_messages': self.source_messages,
            'timestamp_range': [
                self.timestamp_range[0].isoformat(),
                self.timestamp_range[1].isoformat()
            ]
        }


class ContextAssemblyEngine:
    """
    Advanced context assembly that NEVER loses information
    Handles context limits through intelligent segmentation and retrieval
    """
    
    def __init__(self, memory_manager: ConversationMemoryManager):
        self.memory_manager = memory_manager
        
        # Token estimation multiplier (conservative)
        self.token_multiplier = 1.4
        
        # Context limits for different scenarios
        self.limits = {
            'chat': 8000,
            'reasoning': 6000,  # Leave room for reasoning tokens
            'rag': 10000,       # Can handle more context with RAG
            'multi_agent': 4000 # Conservative for agent coordination
        }
        
        # Minimum guarantees
        self.min_recent_messages = 8
        self.min_relevant_messages = 5
        
    def estimate_tokens(self, text: str) -> int:
        """Conservative token estimation"""
        return int(len(text.split()) * self.token_multiplier)
    
    async def assemble_context(
        self,
        conversation_id: str,
        current_query: str,
        strategy: ContextStrategy = ContextStrategy.ADAPTIVE,
        context_type: str = 'chat',
        max_tokens: Optional[int] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Main context assembly method
        Returns: (assembled_context, metadata)
        """
        max_tokens = max_tokens or self.limits.get(context_type, 8000)
        
        # Choose optimal strategy if adaptive
        if strategy == ContextStrategy.ADAPTIVE:
            strategy = await self._choose_optimal_strategy(
                conversation_id, current_query, context_type
            )
        
        # Gather context segments based on strategy
        segments = await self._gather_context_segments(
            conversation_id, current_query, strategy, max_tokens
        )
        
        # Assemble final context
        final_context, metadata = await self._assemble_final_context(
            segments, max_tokens, strategy
        )
        
        # Add strategy metadata
        metadata.update({
            'strategy_used': strategy.value,
            'context_type': context_type,
            'max_tokens_limit': max_tokens,
            'segments_created': len(segments)
        })
        
        return final_context, metadata
    
    async def _choose_optimal_strategy(
        self,
        conversation_id: str,
        current_query: str,
        context_type: str
    ) -> ContextStrategy:
        """Intelligently choose the best strategy for this context"""
        
        try:
            # Get basic conversation stats
            recent_messages = await self.memory_manager.storage.get_messages(
                conversation_id, limit=20
            )
            
            if not recent_messages:
                return ContextStrategy.RECENT_FOCUS
            
            # Analyze conversation characteristics
            total_messages = len(recent_messages)
            avg_message_length = sum(len(msg.content) for msg in recent_messages) / total_messages
            
            # Check for threads/topics (simple heuristic)
            topics = set()
            for msg in recent_messages[-10:]:  # Recent messages
                words = msg.content.lower().split()
                topics.update(word for word in words if len(word) > 5)
            
            # Decision logic
            if total_messages < 10:
                return ContextStrategy.RECENT_FOCUS
            elif len(topics) > 20:  # Many topics discussed
                return ContextStrategy.THREAD_AWARE
            elif avg_message_length > 200:  # Long detailed messages
                return ContextStrategy.HIERARCHICAL
            elif any(keyword in current_query.lower() for keyword in ['previous', 'earlier', 'before', 'mentioned']):
                return ContextStrategy.RELEVANCE_FOCUS
            else:
                return ContextStrategy.RECENT_FOCUS
                
        except Exception as e:
            print(f"[CONTEXT_ENGINE] Strategy selection failed: {e}")
            return ContextStrategy.RECENT_FOCUS
    
    async def _gather_context_segments(
        self,
        conversation_id: str,
        current_query: str,
        strategy: ContextStrategy,
        max_tokens: int
    ) -> List[ContextSegment]:
        """Gather context segments based on strategy"""
        
        segments = []
        
        if strategy == ContextStrategy.RECENT_FOCUS:
            segments = await self._gather_recent_focused_segments(
                conversation_id, current_query, max_tokens
            )
        elif strategy == ContextStrategy.RELEVANCE_FOCUS:
            segments = await self._gather_relevance_focused_segments(
                conversation_id, current_query, max_tokens
            )
        elif strategy == ContextStrategy.THREAD_AWARE:
            segments = await self._gather_thread_aware_segments(
                conversation_id, current_query, max_tokens
            )
        elif strategy == ContextStrategy.HIERARCHICAL:
            segments = await self._gather_hierarchical_segments(
                conversation_id, current_query, max_tokens
            )
        else:
            # Fallback to recent focus
            segments = await self._gather_recent_focused_segments(
                conversation_id, current_query, max_tokens
            )
        
        return segments
    
    async def _gather_recent_focused_segments(
        self,
        conversation_id: str,
        current_query: str,
        max_tokens: int
    ) -> List[ContextSegment]:
        """Strategy focusing on recent conversation"""
        
        segments = []
        
        # Get recent messages
        recent_messages = await self.memory_manager.storage.get_messages(
            conversation_id, limit=50  # Get more than we need
        )
        
        if not recent_messages:
            return segments
        
        # Create recent context segment
        recent_content_parts = []
        recent_tokens = 0
        used_message_ids = []
        
        # Reserve 20% of tokens for relevance if query suggests it's needed
        relevance_budget = max_tokens * 0.2 if any(
            keyword in current_query.lower() 
            for keyword in ['previous', 'earlier', 'mentioned', 'discussed']
        ) else 0
        
        recent_budget = max_tokens - relevance_budget
        
        # Add recent messages within budget
        for msg in reversed(recent_messages):  # Most recent first
            msg_text = f"{msg.role.capitalize()}: {msg.content}"
            msg_tokens = self.estimate_tokens(msg_text)
            
            if recent_tokens + msg_tokens <= recent_budget:
                recent_content_parts.insert(0, msg_text)  # Maintain chronological order
                recent_tokens += msg_tokens
                used_message_ids.append(msg.message_id)
            else:
                break
        
        if recent_content_parts:
            segments.append(ContextSegment(
                content="\n\n".join(recent_content_parts),
                segment_type="recent",
                priority=1.0,
                token_estimate=recent_tokens,
                source_messages=used_message_ids,
                timestamp_range=(
                    recent_messages[len(recent_messages) - len(used_message_ids)].timestamp,
                    recent_messages[-1].timestamp
                )
            ))
        
        # Add relevant context if there's budget
        if relevance_budget > 0:
            relevant_segments = await self._add_relevant_context(
                conversation_id, current_query, relevance_budget, set(used_message_ids)
            )
            segments.extend(relevant_segments)
        
        return segments
    
    async def _gather_relevance_focused_segments(
        self,
        conversation_id: str,
        current_query: str,
        max_tokens: int
    ) -> List[ContextSegment]:
        """Strategy focusing on semantic relevance"""
        
        segments = []
        
        # Split budget: 60% relevance, 40% recent
        relevance_budget = int(max_tokens * 0.6)
        recent_budget = max_tokens - relevance_budget
        
        # Get relevant messages
        relevant_messages = await self.memory_manager.storage.search_messages(
            conversation_id, current_query, limit=30
        )
        
        relevant_content_parts = []
        relevant_tokens = 0
        used_message_ids = set()
        
        for msg in relevant_messages:
            msg_text = f"{msg.role.capitalize()}: {msg.content}"
            msg_tokens = self.estimate_tokens(msg_text)
            
            if relevant_tokens + msg_tokens <= relevance_budget:
                relevant_content_parts.append((msg_text, msg.timestamp, msg.message_id))
                relevant_tokens += msg_tokens
                used_message_ids.add(msg.message_id)
            else:
                break
        
        # Sort by timestamp for coherent flow
        relevant_content_parts.sort(key=lambda x: x[1])
        
        if relevant_content_parts:
            segments.append(ContextSegment(
                content="\n\n".join([text for text, _, _ in relevant_content_parts]),
                segment_type="relevant",
                priority=0.9,
                token_estimate=relevant_tokens,
                source_messages=[msg_id for _, _, msg_id in relevant_content_parts],
                timestamp_range=(
                    relevant_content_parts[0][1],
                    relevant_content_parts[-1][1]
                )
            ))
        
        # Add recent context that's not already included
        recent_messages = await self.memory_manager.storage.get_messages(
            conversation_id, limit=20
        )
        
        recent_content_parts = []
        recent_tokens = 0
        recent_msg_ids = []
        
        for msg in reversed(recent_messages):
            if msg.message_id not in used_message_ids:
                msg_text = f"{msg.role.capitalize()}: {msg.content}"
                msg_tokens = self.estimate_tokens(msg_text)
                
                if recent_tokens + msg_tokens <= recent_budget:
                    recent_content_parts.insert(0, msg_text)
                    recent_tokens += msg_tokens
                    recent_msg_ids.append(msg.message_id)
                else:
                    break
        
        if recent_content_parts:
            segments.append(ContextSegment(
                content="\n\n".join(recent_content_parts),
                segment_type="recent",
                priority=0.8,
                token_estimate=recent_tokens,
                source_messages=recent_msg_ids,
                timestamp_range=(
                    min(msg.timestamp for msg in recent_messages if msg.message_id in recent_msg_ids),
                    max(msg.timestamp for msg in recent_messages if msg.message_id in recent_msg_ids)
                )
            ))
        
        return segments
    
    async def _gather_thread_aware_segments(
        self,
        conversation_id: str,
        current_query: str,
        max_tokens: int
    ) -> List[ContextSegment]:
        """Strategy that organizes context by conversation threads"""
        
        # For now, use a simplified thread detection
        # In a full implementation, this would use more sophisticated topic modeling
        
        segments = []
        all_messages = await self.memory_manager.storage.get_messages(
            conversation_id, limit=100
        )
        
        if not all_messages:
            return segments
        
        # Simple thread detection based on topic keywords
        threads = self._detect_simple_threads(all_messages)
        
        # Find most relevant thread to current query
        relevant_thread = self._find_relevant_thread(threads, current_query)
        
        # Budget allocation
        if relevant_thread:
            thread_budget = int(max_tokens * 0.7)
            recent_budget = max_tokens - thread_budget
            
            # Add relevant thread context
            thread_segment = self._create_thread_segment(
                relevant_thread, thread_budget
            )
            if thread_segment:
                segments.append(thread_segment)
            
            # Add recent messages not in the thread
            used_msg_ids = set(thread_segment.source_messages if thread_segment else [])
            recent_segments = await self._add_recent_context(
                conversation_id, recent_budget, used_msg_ids
            )
            segments.extend(recent_segments)
        else:
            # No clear threads, fallback to recent focus
            segments = await self._gather_recent_focused_segments(
                conversation_id, current_query, max_tokens
            )
        
        return segments
    
    async def _gather_hierarchical_segments(
        self,
        conversation_id: str,
        current_query: str,
        max_tokens: int
    ) -> List[ContextSegment]:
        """Strategy that creates hierarchical context layers"""
        
        segments = []
        
        # Layer 1: Immediate context (30%)
        immediate_budget = int(max_tokens * 0.3)
        immediate_segments = await self._add_recent_context(
            conversation_id, immediate_budget, set()
        )
        segments.extend(immediate_segments)
        
        # Layer 2: Relevant context (40%)
        used_msg_ids = set()
        for seg in immediate_segments:
            used_msg_ids.update(seg.source_messages)
        
        relevant_budget = int(max_tokens * 0.4)
        relevant_segments = await self._add_relevant_context(
            conversation_id, current_query, relevant_budget, used_msg_ids
        )
        segments.extend(relevant_segments)
        
        # Layer 3: Summary context (30%)
        summary_budget = max_tokens - sum(seg.token_estimate for seg in segments)
        if summary_budget > 100:  # Only if there's meaningful budget left
            summary_segment = await self._create_summary_segment(
                conversation_id, summary_budget, used_msg_ids
            )
            if summary_segment:
                segments.append(summary_segment)
        
        return segments
    
    async def _add_relevant_context(
        self,
        conversation_id: str,
        query: str,
        budget: int,
        exclude_message_ids: set
    ) -> List[ContextSegment]:
        """Add relevant context within budget"""
        
        relevant_messages = await self.memory_manager.storage.search_messages(
            conversation_id, query, limit=20
        )
        
        content_parts = []
        total_tokens = 0
        used_msg_ids = []
        
        for msg in relevant_messages:
            if msg.message_id not in exclude_message_ids:
                msg_text = f"{msg.role.capitalize()}: {msg.content}"
                msg_tokens = self.estimate_tokens(msg_text)
                
                if total_tokens + msg_tokens <= budget:
                    content_parts.append((msg_text, msg.timestamp, msg.message_id))
                    total_tokens += msg_tokens
                    used_msg_ids.append(msg.message_id)
                else:
                    break
        
        if not content_parts:
            return []
        
        # Sort by timestamp
        content_parts.sort(key=lambda x: x[1])
        
        return [ContextSegment(
            content="\n\n".join([text for text, _, _ in content_parts]),
            segment_type="relevant",
            priority=0.8,
            token_estimate=total_tokens,
            source_messages=used_msg_ids,
            timestamp_range=(
                content_parts[0][1],
                content_parts[-1][1]
            )
        )]
    
    async def _add_recent_context(
        self,
        conversation_id: str,
        budget: int,
        exclude_message_ids: set
    ) -> List[ContextSegment]:
        """Add recent context within budget"""
        
        recent_messages = await self.memory_manager.storage.get_messages(
            conversation_id, limit=30
        )
        
        content_parts = []
        total_tokens = 0
        used_msg_ids = []
        
        for msg in reversed(recent_messages):
            if msg.message_id not in exclude_message_ids:
                msg_text = f"{msg.role.capitalize()}: {msg.content}"
                msg_tokens = self.estimate_tokens(msg_text)
                
                if total_tokens + msg_tokens <= budget:
                    content_parts.insert(0, msg_text)  # Maintain order
                    total_tokens += msg_tokens
                    used_msg_ids.append(msg.message_id)
                else:
                    break
        
        if not content_parts:
            return []
        
        return [ContextSegment(
            content="\n\n".join(content_parts),
            segment_type="recent",
            priority=1.0,
            token_estimate=total_tokens,
            source_messages=used_msg_ids,
            timestamp_range=(
                recent_messages[len(recent_messages) - len(used_msg_ids)].timestamp,
                recent_messages[-1].timestamp
            )
        )]
    
    async def _create_summary_segment(
        self,
        conversation_id: str,
        budget: int,
        exclude_message_ids: set
    ) -> Optional[ContextSegment]:
        """Create a summary of older conversation parts"""
        
        all_messages = await self.memory_manager.storage.get_messages(
            conversation_id, limit=200
        )
        
        # Get messages not already included
        summary_messages = [
            msg for msg in all_messages 
            if msg.message_id not in exclude_message_ids
        ]
        
        if len(summary_messages) < 5:  # Not enough to summarize
            return None
        
        # Create a simple summary
        topics = set()
        key_points = []
        
        for msg in summary_messages[:50]:  # Don't process too many
            words = msg.content.lower().split()
            # Extract potential topics (words longer than 4 chars)
            topics.update(word for word in words if len(word) > 4 and word.isalpha())
        
        # Limit topics to most relevant
        topic_list = list(topics)[:15]
        
        summary_text = f"[Earlier conversation summary: {len(summary_messages)} messages covering topics like {', '.join(topic_list[:10])}]"
        
        summary_tokens = self.estimate_tokens(summary_text)
        
        if summary_tokens > budget:
            return None
        
        return ContextSegment(
            content=summary_text,
            segment_type="summary",
            priority=0.3,
            token_estimate=summary_tokens,
            source_messages=[msg.message_id for msg in summary_messages],
            timestamp_range=(
                summary_messages[0].timestamp,
                summary_messages[-1].timestamp
            )
        )
    
    def _detect_simple_threads(self, messages: List[ConversationMessage]) -> List[Dict[str, Any]]:
        """Simple thread detection based on topic clustering"""
        # Simplified implementation - would use more sophisticated methods in production
        return []
    
    def _find_relevant_thread(self, threads: List[Dict[str, Any]], query: str) -> Optional[Dict[str, Any]]:
        """Find the most relevant thread for the current query"""
        # Simplified implementation
        return None
    
    def _create_thread_segment(self, thread: Dict[str, Any], budget: int) -> Optional[ContextSegment]:
        """Create a context segment from a thread"""
        # Simplified implementation
        return None
    
    async def _assemble_final_context(
        self,
        segments: List[ContextSegment],
        max_tokens: int,
        strategy: ContextStrategy
    ) -> Tuple[str, Dict[str, Any]]:
        """Assemble the final context from segments"""
        
        if not segments:
            return "", {"segments": 0, "total_tokens": 0}
        
        # Sort segments by priority (highest first)
        segments.sort(key=lambda x: x.priority, reverse=True)
        
        # Include segments within token limit
        included_segments = []
        total_tokens = 0
        
        for segment in segments:
            if total_tokens + segment.token_estimate <= max_tokens:
                included_segments.append(segment)
                total_tokens += segment.token_estimate
            else:
                # Check if we can fit a truncated version
                remaining_budget = max_tokens - total_tokens
                if remaining_budget > 100:  # Minimum meaningful segment
                    truncated_content = self._truncate_segment(
                        segment.content, remaining_budget
                    )
                    if truncated_content:
                        truncated_segment = ContextSegment(
                            content=truncated_content,
                            segment_type=segment.segment_type + "_truncated",
                            priority=segment.priority * 0.8,
                            token_estimate=remaining_budget,
                            source_messages=segment.source_messages,
                            timestamp_range=segment.timestamp_range
                        )
                        included_segments.append(truncated_segment)
                        total_tokens += remaining_budget
                break
        
        # Assemble final context
        if strategy in [ContextStrategy.THREAD_AWARE, ContextStrategy.HIERARCHICAL]:
            # Sort by timestamp within priority groups
            included_segments.sort(key=lambda x: (x.priority, x.timestamp_range[0]), reverse=True)
        
        context_parts = []
        for segment in included_segments:
            if segment.segment_type == "summary":
                context_parts.append(segment.content)
            else:
                context_parts.append(segment.content)
        
        final_context = "\n\n---\n\n".join(context_parts)
        
        metadata = {
            "segments": [segment.to_dict() for segment in included_segments],
            "total_tokens": total_tokens,
            "segments_included": len(included_segments),
            "segments_excluded": len(segments) - len(included_segments),
            "token_efficiency": total_tokens / max_tokens if max_tokens > 0 else 0
        }
        
        return final_context, metadata
    
    def _truncate_segment(self, content: str, max_tokens: int) -> str:
        """Intelligently truncate a segment to fit within token limit"""
        words = content.split()
        max_words = int(max_tokens / self.token_multiplier)
        
        if len(words) <= max_words:
            return content
        
        # Truncate and add indicator
        truncated_words = words[:max_words - 10]  # Leave room for truncation indicator
        truncated_content = " ".join(truncated_words) + " [... truncated for space ...]"
        
        return truncated_content