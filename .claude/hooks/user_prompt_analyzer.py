#!/usr/bin/env python3
"""
User Prompt Analyzer Hook for Claude Code
==========================================
This hook analyzes every user message and intelligently selects appropriate agents
for task delegation. It ensures Claude follows the READ-ONLY mode and delegates
all execution to the appropriate agents.

This is the main entry point for agent selection based on user input.
"""

import re
import json
import sys
import os
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/kianwoonwong/Downloads/jarvis/.claude/hooks/hook_activity.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UserPromptAnalyzer:
    """Analyzes user prompts and maps them to appropriate agents"""
    
    def __init__(self):
        self.agent_mappings = self._initialize_agent_mappings()
        self.keyword_patterns = self._initialize_keyword_patterns()
        self.priority_indicators = self._initialize_priority_indicators()
        
    def _initialize_agent_mappings(self) -> Dict[str, Dict]:
        """Initialize agent capabilities and keywords"""
        return {
            "coder": {
                "name": "Code Agent",
                "file": "coder.md",
                "keywords": ["code", "implement", "function", "class", "method", "bug", "fix", 
                           "refactor", "optimize", "algorithm", "logic", "script", "program",
                           "feature", "module", "component", "syntax", "compile", "error"],
                "patterns": [r"write.*code", r"create.*function", r"implement.*feature",
                           r"fix.*bug", r"debug", r"refactor", r"optimize.*code"],
                "priority": 0.9
            },
            "senior-coder": {
                "name": "Senior Code Agent",
                "file": "senior-coder.md",
                "keywords": ["architecture", "design pattern", "scalability", "performance",
                           "system design", "microservice", "integration", "framework",
                           "best practice", "enterprise", "production", "deployment",
                           "jwt", "authentication", "security", "auth"],
                "patterns": [r"design.*system", r"architect.*solution", r"scale.*application",
                           r"production.*ready", r"enterprise.*solution", r"authentication.*system",
                           r"jwt.*token", r"auth.*system"],
                "priority": 1.0
            },
            "database-administrator": {
                "name": "Database Administrator",
                "file": "database-administrator.md",
                "keywords": ["database", "sql", "query", "table", "schema", "index", "migration",
                           "postgresql", "mysql", "mongodb", "redis", "neo4j", "milvus",
                           "transaction", "backup", "restore", "optimization", "normalization",
                           "db", "production database", "vector database", "rag"],
                "patterns": [r"database.*design", r"sql.*query", r"optimize.*query",
                           r"create.*table", r"migration", r"schema.*change", r"database.*down",
                           r"production.*database", r"vector.*database", r"database.*integration"],
                "priority": 0.9
            },
            "ui-theme-designer": {
                "name": "UI Theme Designer",
                "file": "ui-theme-designer.md",
                "keywords": ["ui", "ux", "design", "theme", "css", "style", "color", "layout",
                           "frontend", "react", "component", "responsive", "mobile", "desktop",
                           "animation", "transition", "typography", "spacing", "palette"],
                "patterns": [r"design.*ui", r"create.*theme", r"style.*component",
                           r"responsive.*design", r"user.*interface", r"improve.*ux"],
                "priority": 0.7
            },
            "codebase-error-analyzer": {
                "name": "Codebase Error Analyzer",
                "file": "codebase-error-analyzer.md",
                "keywords": ["error", "exception", "crash", "failure", "debug", "trace",
                           "stack trace", "log", "issue", "problem", "broken", "not working",
                           "500", "404", "timeout", "memory leak", "performance issue"],
                "patterns": [r".*error.*", r".*exception.*", r".*crash.*", r".*not.*working",
                           r"debug.*issue", r"analyze.*problem", r"fix.*broken"],
                "priority": 0.95
            },
            "llm-ai-architect": {
                "name": "LLM AI Architect",
                "file": "llm-ai-architect.md",
                "keywords": ["llm", "ai", "ml", "machine learning", "model", "training",
                           "inference", "prompt", "embedding", "vector", "rag", "langchain",
                           "openai", "anthropic", "huggingface", "transformer", "neural"],
                "patterns": [r".*llm.*", r".*ai.*model", r"machine.*learning",
                           r"train.*model", r"prompt.*engineering", r"rag.*system"],
                "priority": 0.85
            },
            "general-purpose": {
                "name": "General Purpose Agent",
                "file": "general-purpose.md",
                "keywords": ["general", "help", "assist", "task", "work", "do", "create"],
                "patterns": [r".*"],  # Catch-all
                "priority": 0.5
            }
        }
    
    def _initialize_keyword_patterns(self) -> Dict[str, List[str]]:
        """Initialize task type patterns"""
        return {
            "implementation": ["implement", "create", "build", "develop", "add", "write"],
            "modification": ["update", "modify", "change", "edit", "refactor", "improve"],
            "debugging": ["debug", "fix", "resolve", "troubleshoot", "investigate", "analyze"],
            "testing": ["test", "validate", "verify", "check", "ensure", "qa"],
            "documentation": ["document", "explain", "describe", "comment", "readme"],
            "review": ["review", "audit", "inspect", "examine", "assess"],
            "optimization": ["optimize", "improve", "enhance", "speed up", "performance"],
            "configuration": ["configure", "setup", "install", "deploy", "initialize"]
        }
    
    def _initialize_priority_indicators(self) -> Dict[str, str]:
        """Initialize priority level indicators"""
        return {
            "critical": ["urgent", "critical", "emergency", "asap", "immediately", "breaking"],
            "high": ["important", "priority", "soon", "quickly", "fast"],
            "normal": ["normal", "regular", "standard", "typical"],
            "low": ["when possible", "eventually", "low priority", "nice to have"]
        }
    
    def analyze_prompt(self, user_prompt: str) -> Dict:
        """
        Analyze user prompt and determine appropriate agents
        
        Args:
            user_prompt: The user's input message
            
        Returns:
            Dictionary with analysis results and agent recommendations
        """
        prompt_lower = user_prompt.lower()
        
        # Detect task type
        task_type = self._detect_task_type(prompt_lower)
        
        # Detect priority
        priority = self._detect_priority(prompt_lower)
        
        # Select agents based on content
        selected_agents = self._select_agents(prompt_lower, user_prompt)
        
        # Generate context
        context = self._generate_context(user_prompt, task_type, selected_agents)
        
        # Build recommendation
        recommendation = {
            "timestamp": datetime.now().isoformat(),
            "original_prompt": user_prompt,
            "analysis": {
                "task_type": task_type,
                "priority": priority,
                "detected_keywords": self._extract_keywords(prompt_lower),
                "complexity": self._estimate_complexity(user_prompt)
            },
            "selected_agents": selected_agents,
            "delegation_command": self._build_delegation_command(
                user_prompt, selected_agents, context, priority
            ),
            "reminder": self._generate_reminder(selected_agents, task_type),
            "context": context
        }
        
        # Log the analysis
        logger.info(f"Analyzed prompt: {user_prompt[:100]}...")
        logger.info(f"Selected agents: {[a['name'] for a in selected_agents]}")
        
        return recommendation
    
    def _detect_task_type(self, prompt_lower: str) -> str:
        """Detect the type of task from the prompt"""
        for task_type, keywords in self.keyword_patterns.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return task_type
        return "general"
    
    def _detect_priority(self, prompt_lower: str) -> str:
        """Detect priority level from the prompt"""
        for priority, indicators in self.priority_indicators.items():
            if any(indicator in prompt_lower for indicator in indicators):
                return priority
        return "normal"
    
    def _select_agents(self, prompt_lower: str, original_prompt: str) -> List[Dict]:
        """Select appropriate agents based on prompt content"""
        agent_scores = {}
        
        for agent_key, agent_info in self.agent_mappings.items():
            score = 0.0
            
            # Check keywords
            keyword_matches = sum(1 for keyword in agent_info["keywords"] 
                                 if keyword in prompt_lower)
            score += keyword_matches * 0.3
            
            # Check patterns
            pattern_matches = sum(1 for pattern in agent_info["patterns"]
                                 if re.search(pattern, prompt_lower))
            score += pattern_matches * 0.5
            
            # Apply priority multiplier
            score *= agent_info["priority"]
            
            if score > 0:
                agent_scores[agent_key] = score
        
        # Sort agents by score
        sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top agents (usually 2-3 agents for collaboration)
        selected = []
        for agent_key, score in sorted_agents[:3]:
            if score > 0.1:  # Threshold for relevance
                agent_info = self.agent_mappings[agent_key]
                selected.append({
                    "key": agent_key,
                    "name": agent_info["name"],
                    "file": agent_info["file"],
                    "score": score,
                    "reason": self._explain_selection(agent_key, prompt_lower)
                })
        
        # Ensure at least one agent is selected
        if not selected:
            selected.append({
                "key": "general-purpose",
                "name": self.agent_mappings["general-purpose"]["name"],
                "file": self.agent_mappings["general-purpose"]["file"],
                "score": 0.5,
                "reason": "Default agent for general tasks"
            })
        
        return selected
    
    def _explain_selection(self, agent_key: str, prompt_lower: str) -> str:
        """Explain why an agent was selected"""
        agent_info = self.agent_mappings[agent_key]
        matched_keywords = [k for k in agent_info["keywords"] if k in prompt_lower]
        
        if matched_keywords:
            return f"Detected keywords: {', '.join(matched_keywords[:3])}"
        else:
            return "Pattern matching and context analysis"
    
    def _extract_keywords(self, prompt_lower: str) -> List[str]:
        """Extract relevant keywords from the prompt"""
        all_keywords = []
        for agent_info in self.agent_mappings.values():
            for keyword in agent_info["keywords"]:
                if keyword in prompt_lower and keyword not in all_keywords:
                    all_keywords.append(keyword)
        return all_keywords[:10]  # Limit to top 10
    
    def _estimate_complexity(self, prompt: str) -> str:
        """Estimate task complexity based on prompt"""
        word_count = len(prompt.split())
        line_count = len(prompt.split('\n'))
        
        if word_count > 100 or line_count > 10:
            return "high"
        elif word_count > 50 or line_count > 5:
            return "medium"
        else:
            return "low"
    
    def _generate_context(self, prompt: str, task_type: str, agents: List[Dict]) -> str:
        """Generate context for the agents"""
        context_parts = [
            f"Task Type: {task_type}",
            f"Selected Agents: {', '.join([a['name'] for a in agents])}",
            f"Timestamp: {datetime.now().isoformat()}",
            "Source: Claude Code Hook System",
            "Execution Mode: Delegated (Claude is READ-ONLY)"
        ]
        
        # Add specific context based on task type
        if task_type == "debugging":
            context_parts.append("Focus: Error analysis and resolution")
        elif task_type == "implementation":
            context_parts.append("Focus: New feature development")
        elif task_type == "optimization":
            context_parts.append("Focus: Performance improvement")
        
        return "\n".join(context_parts)
    
    def _build_delegation_command(self, prompt: str, agents: List[Dict], 
                                  context: str, priority: str) -> str:
        """Build the delegation command for Jarvis agents via request_agent_work.py
        
        NOTE: This is for Jarvis agents (database), NOT Claude Code agents (.claude/agents/)
        """
        agent_names = [a['name'] for a in agents]
        
        # Escape quotes in prompt and context
        escaped_prompt = prompt.replace('"', '\\"').replace('\n', ' ')
        escaped_context = context.replace('"', '\\"').replace('\n', ' ')
        
        command = f'''python /Users/kianwoonwong/Downloads/jarvis/request_agent_work.py \\
    --task "{escaped_prompt}" \\
    --agents "{','.join(agent_names)}" \\
    --context "{escaped_context}" \\
    --priority {priority} \\
    --max-iterations 10'''
        
        return command
    
    def _generate_reminder(self, agents: List[Dict], task_type: str) -> str:
        """Generate a reminder message for Claude"""
        agent_names = [a['name'] for a in agents]
        
        reminder = f"""
🤖 AGENT DELEGATION REQUIRED
============================
Claude Code operates in READ-ONLY mode. All execution must be delegated to agents.

📋 Task Analysis:
- Task Type: {task_type}
- Recommended Agents: {', '.join(agent_names)}

🔧 Selected Agents:
"""
        
        for agent in agents:
            reminder += f"\n• {agent['name']} ({agent['file']})"
            reminder += f"\n  Reason: {agent['reason']}"
            reminder += f"\n  Relevance Score: {agent['score']:.2f}"
        
        reminder += """

⚠️ IMPORTANT REMINDERS:
1. You (Claude) can ONLY read and analyze code
2. ALL modifications must go through agents
3. Use request_agent_work.py for delegation
4. Never use Write, Edit, or Execute tools directly
5. Agents handle all execution and modifications

📌 Next Steps:
1. Analyze the request using Read/Grep/Glob tools
2. Create a plan for the agents
3. Delegate work using the command above
4. Monitor progress with BashOutput
5. Verify results through reading

Remember: You are the architect and analyzer. Agents are the builders.
"""
        
        return reminder


def create_hook_response(user_prompt: str) -> Dict:
    """
    Main hook entry point for user prompt analysis
    
    Args:
        user_prompt: The user's input message
        
    Returns:
        Hook response with agent recommendations
    """
    analyzer = UserPromptAnalyzer()
    
    try:
        # Analyze the prompt
        analysis = analyzer.analyze_prompt(user_prompt)
        
        # Create hook response
        response = {
            "status": "success",
            "hook_type": "user-prompt-submit",
            "analysis": analysis,
            "action_required": "delegate_to_agents",
            "display_reminder": True,
            "log_file": "/Users/kianwoonwong/Downloads/jarvis/.claude/hooks/hook_activity.log"
        }
        
        # Write analysis to file for persistence
        analysis_file = f"/Users/kianwoonwong/Downloads/jarvis/.claude/hooks/analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        response["analysis_file"] = analysis_file
        
        return response
        
    except Exception as e:
        logger.error(f"Hook analysis failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "fallback_agents": ["general-purpose"],
            "reminder": "Error in hook analysis. Please use general-purpose agent."
        }


if __name__ == "__main__":
    # Test mode - can be run directly for testing
    if len(sys.argv) > 1:
        test_prompt = " ".join(sys.argv[1:])
        result = create_hook_response(test_prompt)
        print(json.dumps(result, indent=2, default=str))
    else:
        print("User Prompt Analyzer Hook")
        print("========================")
        print("Usage: python user_prompt_analyzer.py <test prompt>")
        print("\nThis hook analyzes user prompts and selects appropriate agents.")