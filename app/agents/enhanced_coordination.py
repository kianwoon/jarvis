"""
Enhanced Multi-Agent Coordination System

This module provides advanced coordination patterns for multi-agent systems,
including negotiation, consensus building, and dynamic task allocation.
"""
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np

from app.core.redis_client import get_redis_client
from app.core.pipeline_config import get_pipeline_settings

logger = logging.getLogger(__name__)
settings = get_pipeline_settings()


class CoordinationPattern(Enum):
    """Advanced coordination patterns for multi-agent systems"""
    NEGOTIATION = "negotiation"
    CONSENSUS = "consensus"
    AUCTION = "auction"
    DELEGATION = "delegation"
    FEDERATION = "federation"
    BLACKBOARD = "blackboard"
    CONTRACT_NET = "contract_net"


@dataclass
class AgentCapability:
    """Defines what an agent can do"""
    agent_id: str
    capabilities: Set[str]
    cost: float = 1.0  # Relative cost of using this agent
    reliability: float = 0.95  # Historical success rate
    avg_execution_time: float = 30.0  # Average execution time in seconds
    max_load: int = 1  # Max concurrent tasks
    current_load: int = 0
    specializations: Dict[str, float] = field(default_factory=dict)  # Domain expertise scores


@dataclass
class TaskRequirement:
    """Defines what a task needs"""
    task_id: str
    required_capabilities: Set[str]
    preferred_capabilities: Set[str] = field(default_factory=set)
    priority: int = 5  # 1-10 scale
    deadline: Optional[datetime] = None
    estimated_effort: float = 1.0
    dependencies: List[str] = field(default_factory=list)


class EnhancedCoordinator:
    """Advanced coordinator for multi-agent systems"""
    
    def __init__(self):
        self.redis_client = get_redis_client()
        self.agent_registry: Dict[str, AgentCapability] = {}
        self.task_queue: List[TaskRequirement] = []
        self.execution_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.blackboard: Dict[str, Any] = {}  # Shared knowledge space
        
    def register_agent(self, agent_capability: AgentCapability):
        """Register an agent with its capabilities"""
        self.agent_registry[agent_capability.agent_id] = agent_capability
        logger.info(f"Registered agent {agent_capability.agent_id} with capabilities: {agent_capability.capabilities}")
        
    def submit_task(self, task: TaskRequirement):
        """Submit a task for coordination"""
        self.task_queue.append(task)
        self.task_queue.sort(key=lambda t: t.priority, reverse=True)
        
    async def coordinate(self, pattern: CoordinationPattern, 
                        tasks: List[TaskRequirement],
                        constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Coordinate task execution using specified pattern"""
        
        if pattern == CoordinationPattern.NEGOTIATION:
            return await self._negotiate_tasks(tasks, constraints)
        elif pattern == CoordinationPattern.CONSENSUS:
            return await self._build_consensus(tasks, constraints)
        elif pattern == CoordinationPattern.AUCTION:
            return await self._auction_tasks(tasks, constraints)
        elif pattern == CoordinationPattern.DELEGATION:
            return await self._delegate_tasks(tasks, constraints)
        elif pattern == CoordinationPattern.FEDERATION:
            return await self._federate_agents(tasks, constraints)
        elif pattern == CoordinationPattern.BLACKBOARD:
            return await self._blackboard_coordination(tasks, constraints)
        elif pattern == CoordinationPattern.CONTRACT_NET:
            return await self._contract_net_protocol(tasks, constraints)
        else:
            raise ValueError(f"Unknown coordination pattern: {pattern}")
    
    async def _negotiate_tasks(self, tasks: List[TaskRequirement], 
                              constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Agents negotiate task assignments based on capabilities and load"""
        assignments = {}
        negotiation_rounds = []
        
        for task in tasks:
            # Find capable agents
            capable_agents = self._find_capable_agents(task)
            
            if not capable_agents:
                logger.warning(f"No agents capable of handling task {task.task_id}")
                continue
                
            # Agents bid on tasks
            bids = []
            for agent_id in capable_agents:
                agent = self.agent_registry[agent_id]
                bid = self._calculate_bid(agent, task)
                bids.append((agent_id, bid))
            
            # Sort by bid score (higher is better)
            bids.sort(key=lambda x: x[1], reverse=True)
            
            # Assign to best bidder
            if bids:
                winner = bids[0][0]
                assignments[task.task_id] = winner
                self.agent_registry[winner].current_load += 1
                
                negotiation_rounds.append({
                    "task": task.task_id,
                    "bids": bids,
                    "winner": winner
                })
        
        return {
            "assignments": assignments,
            "negotiation_rounds": negotiation_rounds
        }
    
    async def _build_consensus(self, tasks: List[TaskRequirement],
                              constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Multiple agents work together to reach consensus on complex tasks"""
        consensus_results = {}
        
        for task in tasks:
            # Get all capable agents
            capable_agents = self._find_capable_agents(task)
            
            if len(capable_agents) < 2:
                # Not enough agents for consensus
                consensus_results[task.task_id] = {
                    "status": "insufficient_agents",
                    "agents": capable_agents
                }
                continue
            
            # Each agent proposes a solution
            proposals = {}
            for agent_id in capable_agents:
                agent = self.agent_registry[agent_id]
                proposal = {
                    "agent": agent_id,
                    "confidence": self._calculate_confidence(agent, task),
                    "approach": f"Approach by {agent_id}"  # In real implementation, get actual proposal
                }
                proposals[agent_id] = proposal
            
            # Voting phase
            votes = defaultdict(int)
            for voter_id in capable_agents:
                # Agents vote on proposals (excluding their own)
                for proposer_id, proposal in proposals.items():
                    if proposer_id != voter_id:
                        vote_weight = self._calculate_vote_weight(
                            self.agent_registry[voter_id],
                            proposal
                        )
                        votes[proposer_id] += vote_weight
            
            # Select consensus approach
            if votes:
                consensus_agent = max(votes, key=votes.get)
                consensus_results[task.task_id] = {
                    "status": "consensus_reached",
                    "selected_agent": consensus_agent,
                    "votes": dict(votes),
                    "proposals": proposals
                }
            
        return {"consensus_results": consensus_results}
    
    async def _auction_tasks(self, tasks: List[TaskRequirement],
                            constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Dutch auction mechanism for task allocation"""
        auction_results = {}
        
        for task in tasks:
            # Start with high requirement, gradually lower
            min_capability_match = 1.0
            auction_rounds = []
            
            while min_capability_match > 0.5:
                bidders = []
                
                for agent_id, agent in self.agent_registry.items():
                    capability_match = self._calculate_capability_match(agent, task)
                    if capability_match >= min_capability_match and agent.current_load < agent.max_load:
                        bid_price = self._calculate_bid_price(agent, task, capability_match)
                        bidders.append((agent_id, bid_price, capability_match))
                
                if bidders:
                    # Award to lowest bidder
                    bidders.sort(key=lambda x: x[1])
                    winner = bidders[0]
                    auction_results[task.task_id] = {
                        "winner": winner[0],
                        "price": winner[1],
                        "capability_match": winner[2],
                        "auction_rounds": len(auction_rounds) + 1
                    }
                    self.agent_registry[winner[0]].current_load += 1
                    break
                
                auction_rounds.append({
                    "min_capability": min_capability_match,
                    "bidders": len(bidders)
                })
                min_capability_match -= 0.1
        
        return {"auction_results": auction_results}
    
    async def _blackboard_coordination(self, tasks: List[TaskRequirement],
                                     constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Blackboard pattern where agents contribute to shared knowledge space"""
        
        # Initialize blackboard for tasks
        for task in tasks:
            self.blackboard[task.task_id] = {
                "status": "open",
                "contributions": [],
                "solution_fragments": {}
            }
        
        # Agents examine blackboard and contribute
        contributions = []
        for agent_id, agent in self.agent_registry.items():
            for task in tasks:
                if self._can_contribute(agent, task):
                    contribution = {
                        "agent": agent_id,
                        "task": task.task_id,
                        "type": "analysis",  # Could be: analysis, partial_solution, validation
                        "content": f"{agent_id} contribution to {task.task_id}",
                        "timestamp": datetime.now().isoformat()
                    }
                    self.blackboard[task.task_id]["contributions"].append(contribution)
                    contributions.append(contribution)
        
        # Synthesize solutions from contributions
        solutions = {}
        for task in tasks:
            task_contributions = self.blackboard[task.task_id]["contributions"]
            if len(task_contributions) >= 2:  # Need multiple contributions
                solutions[task.task_id] = {
                    "status": "solved",
                    "contributors": list(set(c["agent"] for c in task_contributions)),
                    "contribution_count": len(task_contributions)
                }
            else:
                solutions[task.task_id] = {
                    "status": "insufficient_contributions",
                    "contribution_count": len(task_contributions)
                }
        
        return {
            "blackboard_state": self.blackboard,
            "solutions": solutions
        }
    
    def _find_capable_agents(self, task: TaskRequirement) -> List[str]:
        """Find agents capable of handling a task"""
        capable = []
        for agent_id, agent in self.agent_registry.items():
            if task.required_capabilities.issubset(agent.capabilities):
                capable.append(agent_id)
        return capable
    
    def _calculate_bid(self, agent: AgentCapability, task: TaskRequirement) -> float:
        """Calculate bid score for an agent on a task"""
        # Consider multiple factors
        capability_score = len(task.required_capabilities.intersection(agent.capabilities)) / len(task.required_capabilities)
        load_factor = 1.0 - (agent.current_load / agent.max_load)
        reliability_factor = agent.reliability
        
        # Check specializations
        specialization_bonus = 0.0
        for cap in task.required_capabilities:
            if cap in agent.specializations:
                specialization_bonus += agent.specializations[cap]
        
        # Calculate deadline pressure
        deadline_factor = 1.0
        if task.deadline:
            time_available = (task.deadline - datetime.now()).total_seconds()
            if time_available < agent.avg_execution_time * 2:
                deadline_factor = 0.5  # Penalize if tight deadline
        
        bid_score = (
            capability_score * 0.3 +
            load_factor * 0.2 +
            reliability_factor * 0.2 +
            specialization_bonus * 0.2 +
            deadline_factor * 0.1
        )
        
        return bid_score
    
    def _calculate_capability_match(self, agent: AgentCapability, task: TaskRequirement) -> float:
        """Calculate how well agent capabilities match task requirements"""
        required_match = len(task.required_capabilities.intersection(agent.capabilities)) / len(task.required_capabilities)
        preferred_match = 0.0
        if task.preferred_capabilities:
            preferred_match = len(task.preferred_capabilities.intersection(agent.capabilities)) / len(task.preferred_capabilities)
        
        return required_match * 0.8 + preferred_match * 0.2
    
    def _calculate_bid_price(self, agent: AgentCapability, task: TaskRequirement, capability_match: float) -> float:
        """Calculate bid price for auction"""
        base_price = agent.cost * task.estimated_effort
        
        # Adjust for current load
        load_multiplier = 1.0 + (agent.current_load / agent.max_load) * 0.5
        
        # Adjust for capability match
        capability_multiplier = 2.0 - capability_match  # Better match = lower price
        
        return base_price * load_multiplier * capability_multiplier
    
    def _calculate_confidence(self, agent: AgentCapability, task: TaskRequirement) -> float:
        """Calculate agent's confidence in handling a task"""
        capability_match = self._calculate_capability_match(agent, task)
        
        # Consider specializations
        specialization_score = 0.0
        for cap in task.required_capabilities:
            if cap in agent.specializations:
                specialization_score = max(specialization_score, agent.specializations[cap])
        
        confidence = (
            capability_match * 0.5 +
            agent.reliability * 0.3 +
            specialization_score * 0.2
        )
        
        return min(confidence, 1.0)
    
    def _calculate_vote_weight(self, voter: AgentCapability, proposal: Dict[str, Any]) -> float:
        """Calculate voting weight based on voter expertise"""
        # In real implementation, would analyze proposal content
        # For now, use voter's reliability as weight
        return voter.reliability
    
    def _can_contribute(self, agent: AgentCapability, task: TaskRequirement) -> bool:
        """Check if agent can contribute to a task"""
        # Agent can contribute if it has any of the required capabilities
        return bool(agent.capabilities.intersection(task.required_capabilities))


class MultiAgentOrchestrator:
    """High-level orchestrator for complex multi-agent workflows"""
    
    def __init__(self):
        self.coordinator = EnhancedCoordinator()
        self.redis_client = get_redis_client()
        self.execution_id = None
        
    async def execute_pipeline(self, pipeline_config: Dict[str, Any], 
                              input_data: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute a pipeline with advanced coordination"""
        
        self.execution_id = f"pipeline_{datetime.now().timestamp()}"
        
        # Register agents from pipeline config
        for agent_config in pipeline_config.get("agents", []):
            capability = AgentCapability(
                agent_id=agent_config["agent_name"],
                capabilities=set(agent_config.get("capabilities", [])),
                cost=agent_config.get("cost", 1.0),
                reliability=agent_config.get("reliability", 0.95)
            )
            self.coordinator.register_agent(capability)
        
        # Create tasks from pipeline stages
        tasks = []
        for i, stage in enumerate(pipeline_config.get("stages", [])):
            task = TaskRequirement(
                task_id=f"stage_{i}",
                required_capabilities=set(stage.get("required_capabilities", [])),
                priority=stage.get("priority", 5),
                estimated_effort=stage.get("effort", 1.0)
            )
            tasks.append(task)
        
        # Determine coordination pattern
        pattern = CoordinationPattern(
            pipeline_config.get("coordination_pattern", "delegation")
        )
        
        # Coordinate execution
        coordination_result = await self.coordinator.coordinate(pattern, tasks)
        
        yield {
            "event": "coordination_complete",
            "data": coordination_result
        }
        
        # Execute based on coordination result
        if "assignments" in coordination_result:
            for task_id, agent_id in coordination_result["assignments"].items():
                yield {
                    "event": "agent_executing",
                    "data": {
                        "task": task_id,
                        "agent": agent_id
                    }
                }
                
                # In real implementation, would execute agent here
                await asyncio.sleep(0.1)  # Simulate execution
                
                yield {
                    "event": "agent_complete",
                    "data": {
                        "task": task_id,
                        "agent": agent_id,
                        "result": f"Result from {agent_id}"
                    }
                }


# Singleton instances
_orchestrator: Optional[MultiAgentOrchestrator] = None


def get_orchestrator() -> MultiAgentOrchestrator:
    """Get orchestrator singleton"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MultiAgentOrchestrator()
    return _orchestrator