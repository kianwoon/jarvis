"""
Prompt templates for LLM routing and orchestration
"""

from typing import List, Dict, Any, Optional
from app.rag_agent.utils.types import SearchContext, ExecutionStrategy


class PromptTemplates:
    """Centralized prompt templates for RAG agent system"""
    
    def build_routing_prompt(
        self,
        query: str,
        available_tools: List[Dict],
        context: Optional[SearchContext] = None
    ) -> str:
        """Build prompt for LLM routing decision"""
        
        context_info = self._build_context_section(context)
        tools_info = self._build_tools_section(available_tools)
        
        return f"""You are an intelligent RAG router for a corporate knowledge management system. Your task is to analyze the user's query and determine which knowledge collections to search and how to search them effectively.

{context_info}

USER QUERY: "{query}"

AVAILABLE KNOWLEDGE COLLECTIONS:
{tools_info}

ROUTING INSTRUCTIONS:
1. Analyze the query to understand what information is needed
2. Select the most relevant collection(s) that contain the required information
3. For each selected collection, refine the query to optimize search results
4. Choose an appropriate execution strategy (see options below)

EXECUTION STRATEGIES:
- single_collection: Query is specific to one domain, search one collection
- parallel_search: Query needs information from multiple collections, search simultaneously  
- cross_reference: Query requires validation across collections (e.g., policy vs regulation)
- iterative_refinement: Complex query needing progressive refinement based on initial results

RESPONSE FORMAT:
Use the provided function calls to search the appropriate collections. You can call multiple collection search functions if needed.

IMPORTANT GUIDELINES:
- Be specific in your search queries - use domain-appropriate terminology
- Consider the user's role/department when selecting collections
- For compliance/regulatory queries, prioritize authoritative sources
- For technical queries, focus on technical documentation collections
- If uncertain between collections, search the most authoritative one first

Now analyze the query and make your routing decisions using the available functions."""

    def build_refinement_prompt(
        self,
        original_query: str,
        current_results: Dict,
        gaps_identified: List[str],
        context: Optional[SearchContext] = None
    ) -> str:
        """Build prompt for query refinement"""
        
        results_summary = self._summarize_results(current_results)
        
        return f"""You are refining a search query based on initial results to fill knowledge gaps.

ORIGINAL QUERY: "{original_query}"

CURRENT SEARCH RESULTS SUMMARY:
{results_summary}

IDENTIFIED KNOWLEDGE GAPS:
{chr(10).join(f"- {gap}" for gap in gaps_identified)}

TASK: Generate a refined search query that will help find the missing information while building on what was already found.

REFINEMENT GUIDELINES:
1. Address the specific gaps identified
2. Use alternative terminology or phrasing
3. Be more specific about missing aspects
4. Consider related concepts that might contain the information
5. Maintain context from successful searches

REFINED QUERY:"""

    def build_synthesis_prompt(
        self,
        original_query: str,
        all_results: List[Dict],
        strategy: ExecutionStrategy
    ) -> str:
        """Build prompt for synthesizing results from multiple searches"""
        
        results_context = self._build_synthesis_context(all_results)
        
        return f"""You are synthesizing information from multiple knowledge sources to answer a user's query comprehensively.

ORIGINAL QUERY: "{original_query}"

SEARCH RESULTS FROM MULTIPLE COLLECTIONS:
{results_context}

SYNTHESIS TASK:
Create a comprehensive, accurate response that:
1. Directly answers the user's query
2. Synthesizes information from all relevant sources
3. Identifies any conflicting information and explains discrepancies
4. Provides proper citations to sources
5. Highlights key insights and actionable information

RESPONSE FORMAT:
- Start with a direct answer to the query
- Provide detailed explanation with evidence
- Include citations: [Source: Collection Name - Document Title]
- Note any limitations or areas needing further investigation

QUALITY STANDARDS:
- Accuracy: Only use information directly supported by the sources
- Completeness: Address all aspects of the original query
- Clarity: Use clear, professional language appropriate for the user's role
- Citations: Always cite sources for factual claims

SYNTHESIZED RESPONSE:"""

    def build_execution_planning_prompt(
        self,
        query: str,
        available_tools: List[Dict],
        context: Optional[SearchContext] = None
    ) -> str:
        """Build prompt for execution planning"""
        
        tools_summary = self._build_tools_summary(available_tools)
        complexity_hints = self._analyze_query_complexity_hints(query)
        
        return f"""You are an execution planner for a multi-step RAG system. Analyze the query and create an optimal execution plan.

QUERY: "{query}"

AVAILABLE COLLECTIONS: {tools_summary}

QUERY COMPLEXITY ANALYSIS:
{complexity_hints}

PLANNING TASK:
Determine the optimal execution strategy and steps:

1. SINGLE STEP: Simple factual query from one domain
2. PARALLEL: Needs information from multiple domains simultaneously
3. SEQUENTIAL: Needs information that builds on previous searches
4. ITERATIVE: Complex query requiring refinement based on results

For each step, specify:
- Which collections to search
- How to refine the query for each collection
- Success criteria for the step
- Dependencies on previous steps

EXECUTION PLAN:"""

    def _build_context_section(self, context: Optional[SearchContext]) -> str:
        """Build context information section"""
        if not context:
            return "CONTEXT: General query, no specific user context provided."
        
        context_parts = [
            f"CONTEXT INFORMATION:",
            f"- Domain: {context.domain}",
            f"- User Role: {getattr(context, 'user_role', 'Not specified')}",
            f"- Urgency: {context.urgency_level}",
            f"- Required Accuracy: {context.required_accuracy}"
        ]
        
        if context.conversation_history:
            context_parts.append(f"- Previous Context: {len(context.conversation_history)} previous messages")
        
        return "\n".join(context_parts)
    
    def _build_tools_section(self, available_tools: List[Dict]) -> str:
        """Build formatted tools section"""
        if not available_tools:
            return "No collections available."
        
        tools_text = []
        for i, tool in enumerate(available_tools, 1):
            func_info = tool.get('function', {})
            name = func_info.get('name', 'unknown')
            description = func_info.get('description', 'No description')
            
            # Extract collection name from function name
            collection_name = name.replace('search_', '') if name.startswith('search_') else name
            
            tools_text.append(f"{i}. **{collection_name}**")
            tools_text.append(f"   Function: {name}")
            tools_text.append(f"   {description}")
            tools_text.append("")
        
        return "\n".join(tools_text)
    
    def _build_tools_summary(self, available_tools: List[Dict]) -> str:
        """Build brief tools summary"""
        tool_names = []
        for tool in available_tools:
            func_info = tool.get('function', {})
            name = func_info.get('name', 'unknown')
            collection_name = name.replace('search_', '') if name.startswith('search_') else name
            tool_names.append(collection_name)
        
        return ", ".join(tool_names[:10])  # Limit to first 10 for brevity
    
    def _summarize_results(self, current_results: Dict) -> str:
        """Summarize current search results"""
        if not current_results:
            return "No results found in current search."
        
        summary_parts = []
        
        # Add result count and sources
        total_results = current_results.get('total_results', 0)
        collections_searched = current_results.get('collections_searched', [])
        
        summary_parts.append(f"Found {total_results} results from {len(collections_searched)} collections")
        
        # Add key topics found
        if 'key_topics' in current_results:
            topics = ", ".join(current_results['key_topics'][:5])
            summary_parts.append(f"Key topics covered: {topics}")
        
        # Add confidence score if available
        if 'confidence_score' in current_results:
            score = current_results['confidence_score']
            summary_parts.append(f"Overall confidence: {score:.1%}")
        
        return "\n".join(summary_parts)
    
    def _build_synthesis_context(self, all_results: List[Dict]) -> str:
        """Build context for synthesis"""
        if not all_results:
            return "No results to synthesize."
        
        context_parts = []
        
        for i, result in enumerate(all_results, 1):
            collection = result.get('collection_name', f'Collection {i}')
            content_preview = result.get('content', '')[:200] + "..." if len(result.get('content', '')) > 200 else result.get('content', '')
            source = result.get('source', 'Unknown source')
            score = result.get('score', 0)
            
            context_parts.append(f"RESULT {i} (Collection: {collection}, Score: {score:.2f}):")
            context_parts.append(f"Source: {source}")
            context_parts.append(f"Content: {content_preview}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _analyze_query_complexity_hints(self, query: str) -> str:
        """Provide hints about query complexity"""
        query_lower = query.lower()
        hints = []
        
        # Multi-aspect queries
        if any(word in query_lower for word in ['and', 'also', 'additionally', 'furthermore']):
            hints.append("- Multi-aspect query: May need information from multiple sources")
        
        # Comparative queries
        if any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs', 'better', 'worse']):
            hints.append("- Comparative query: Needs information from multiple sources for comparison")
        
        # Compliance/regulatory queries
        if any(word in query_lower for word in ['policy', 'regulation', 'compliance', 'requirement', 'rule']):
            hints.append("- Regulatory query: Prioritize authoritative compliance sources")
        
        # Technical queries
        if any(word in query_lower for word in ['api', 'code', 'technical', 'implementation', 'system']):
            hints.append("- Technical query: Focus on technical documentation")
        
        # Process queries
        if any(word in query_lower for word in ['how to', 'process', 'procedure', 'steps', 'workflow']):
            hints.append("- Process query: Look for procedural documentation")
        
        return "\n".join(hints) if hints else "- Standard informational query"