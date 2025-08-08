"""
Conflict Prevention Integration Module
Integrates conflict prevention engine with synthesis pipeline
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from app.core.conflict_prevention_engine import conflict_prevention_engine
from app.langchain.conflict_aware_synthesis import conflict_aware_synthesizer
from app.core.simple_conversation_manager import conversation_manager

logger = logging.getLogger(__name__)

async def enhanced_synthesis_with_prevention(
    question: str,
    query_type: str,
    rag_context: str = "",
    tool_context: str = "",
    conversation_history: str = "",
    conversation_id: Optional[str] = None,
    info_sources: Optional[List[Dict[str, Any]]] = None,
    enable_prevention: bool = True
) -> Dict[str, Any]:
    """
    Enhanced synthesis with integrated conflict prevention
    
    This function wraps the existing synthesis pipeline with conflict prevention capabilities
    """
    
    try:
        # Step 1: Build information sources if not provided
        if info_sources is None:
            info_sources = _build_info_sources(
                rag_context=rag_context,
                tool_context=tool_context,
                conversation_history=conversation_history
            )
        
        # Step 2: Pre-check for conflicts if conversation_id provided
        conflict_check_result = None
        if conversation_id and enable_prevention:
            # Check if new synthesis would conflict with cache
            combined_content = f"{rag_context} {tool_context}"
            if combined_content.strip():
                conflict_check_result = await conflict_prevention_engine.check_for_conflicts(
                    new_content=combined_content,
                    conversation_id=conversation_id,
                    role="assistant",
                    metadata={'source': 'synthesis', 'query_type': query_type}
                )
                
                if conflict_check_result['has_conflicts']:
                    logger.info(f"[PREVENTION] Pre-synthesis conflict check found {len(conflict_check_result['conflicts'])} conflicts")
        
        # Step 3: Run conflict-aware synthesis
        synthesis_result = await conflict_aware_synthesizer.synthesize_with_conflict_prevention(
            question=question,
            info_sources=info_sources,
            conversation_history=conversation_history,
            enable_prevention=enable_prevention
        )
        
        # Step 4: Apply dynamic TTL to conversation cache
        if conversation_id and conflict_check_result:
            await _apply_dynamic_cache_settings(
                conversation_id=conversation_id,
                conflict_result=conflict_check_result,
                synthesis_result=synthesis_result
            )
        
        # Step 5: Generate enhanced response with conflict metadata
        enhanced_response = _build_enhanced_response(
            original_question=question,
            query_type=query_type,
            synthesis_result=synthesis_result,
            conflict_check_result=conflict_check_result
        )
        
        return enhanced_response
        
    except Exception as e:
        logger.error(f"[PREVENTION] Error in enhanced synthesis: {e}")
        # Fallback to basic response
        return {
            'response': f"I'll help with: {question}",
            'conflict_prevention_enabled': False,
            'error': str(e)
        }

def _build_info_sources(
    rag_context: str,
    tool_context: str,
    conversation_history: str
) -> List[Dict[str, Any]]:
    """Build prioritized information sources from contexts"""
    
    sources = []
    
    # Tool context has highest priority
    if tool_context:
        sources.append({
            'content': tool_context,
            'label': 'Tool/Search Results',
            'priority': 10,
            'freshness_score': 1.0,
            'source_type': 'tool'
        })
    
    # RAG context has medium priority
    if rag_context:
        sources.append({
            'content': rag_context,
            'label': 'Document Context',
            'priority': 7,
            'freshness_score': 0.8,
            'source_type': 'rag'
        })
    
    # Conversation history has lower priority
    if conversation_history:
        sources.append({
            'content': conversation_history,
            'label': 'Conversation History',
            'priority': 5,
            'freshness_score': 0.6,
            'source_type': 'conversation'
        })
    
    return sources

async def _apply_dynamic_cache_settings(
    conversation_id: str,
    conflict_result: Dict[str, Any],
    synthesis_result: Dict[str, Any]
) -> None:
    """Apply dynamic cache settings based on conflict analysis"""
    
    try:
        # Get Redis client
        redis_client = await conversation_manager._get_redis()
        if not redis_client:
            return
        
        # Calculate optimal TTL
        base_ttl = 86400  # 24 hours
        volatility = conflict_result.get('volatility_score', 0.0)
        
        # Apply volatility-based reduction
        if volatility > 0.8:
            ttl = int(base_ttl * 0.2)  # 80% reduction for high volatility
        elif volatility > 0.5:
            ttl = int(base_ttl * 0.5)  # 50% reduction for medium volatility
        else:
            ttl = base_ttl
        
        # Update conversation TTL
        key = f"conversation:{conversation_id}:messages"
        await redis_client.expire(key, ttl)
        
        logger.info(f"[PREVENTION] Applied dynamic TTL={ttl}s to conversation {conversation_id} (volatility={volatility:.2f})")
        
    except Exception as e:
        logger.error(f"[PREVENTION] Failed to apply dynamic cache settings: {e}")

def _build_enhanced_response(
    original_question: str,
    query_type: str,
    synthesis_result: Dict[str, Any],
    conflict_check_result: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Build enhanced response with conflict metadata"""
    
    response = {
        'question': original_question,
        'query_type': query_type,
        'synthesis': synthesis_result['synthesis'],
        'conflict_prevention_enabled': synthesis_result.get('prevention_applied', False)
    }
    
    # Add conflict information if available
    if synthesis_result.get('conflict_analysis', {}).get('has_conflicts'):
        response['conflicts_detected'] = True
        response['conflict_report'] = synthesis_result.get('conflict_report', '')
        response['conflicts_resolved'] = synthesis_result.get('sources_modified', False)
    
    # Add cache conflict information
    if conflict_check_result and conflict_check_result.get('has_conflicts'):
        response['cache_conflicts'] = {
            'count': len(conflict_check_result.get('conflicts', [])),
            'types': list(set(c['type'] for c in conflict_check_result.get('conflicts', []))),
            'resolution_strategy': conflict_check_result.get('resolution_strategy')
        }
    
    # Add prediction if available
    if conflict_check_result and conflict_check_result.get('prediction'):
        prediction = conflict_check_result['prediction']
        if prediction.get('likelihood', 0) > 0.5:
            response['conflict_prediction'] = {
                'likelihood': prediction['likelihood'],
                'most_common_type': prediction.get('most_common_type'),
                'recommendation': prediction.get('recommended_action')
            }
    
    return response

async def cleanup_historical_conflicts(days_to_keep: int = 7) -> Dict[str, int]:
    """
    Cleanup old conflict history across the system
    
    Args:
        days_to_keep: Number of days of history to retain
        
    Returns:
        Cleanup statistics
    """
    
    stats = {
        'conflict_history_cleaned': 0,
        'blocked_messages_cleaned': 0,
        'total_cleaned': 0
    }
    
    try:
        # Cleanup conflict prevention engine history
        entries_removed = await conflict_prevention_engine.cleanup_conflict_history(days_to_keep)
        stats['conflict_history_cleaned'] = entries_removed
        
        # Additional cleanup tasks can be added here
        
        stats['total_cleaned'] = sum(stats.values()) - stats['total_cleaned']
        
        logger.info(f"[PREVENTION] Cleanup completed: {stats}")
        
    except Exception as e:
        logger.error(f"[PREVENTION] Cleanup failed: {e}")
    
    return stats

async def get_system_conflict_statistics() -> Dict[str, Any]:
    """
    Get system-wide conflict prevention statistics
    
    Returns:
        System statistics
    """
    
    try:
        # Get Redis client
        redis_client = await conversation_manager._get_redis()
        if not redis_client:
            return {'status': 'redis_unavailable'}
        
        # Get pattern frequency statistics
        pattern_stats = {}
        for conflict_type in ['existence', 'version', 'temporal', 'statistics', 'availability']:
            pattern_key = f"conflict_prevention:patterns:{conflict_type}"
            frequency_data = await redis_client.hgetall(pattern_key)
            if frequency_data:
                pattern_stats[conflict_type] = len(frequency_data)
        
        # Get historical conflict count
        history_key = "conflict_prevention:history"
        history_count = await redis_client.llen(history_key)
        
        return {
            'status': 'active',
            'total_historical_conflicts': history_count,
            'pattern_frequencies': pattern_stats,
            'prevention_engine_active': True
        }
        
    except Exception as e:
        logger.error(f"[PREVENTION] Failed to get statistics: {e}")
        return {'status': 'error', 'error': str(e)}

# Export integration functions
__all__ = [
    'enhanced_synthesis_with_prevention',
    'cleanup_historical_conflicts',
    'get_system_conflict_statistics'
]