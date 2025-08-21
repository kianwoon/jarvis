"""
Workflow State Management System
Provides centralized state handling for automation workflows
"""
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from copy import deepcopy

logger = logging.getLogger(__name__)

class WorkflowState:
    """Manages state throughout workflow execution"""
    
    def __init__(self, workflow_id: int, execution_id: str):
        self.workflow_id = workflow_id
        self.execution_id = execution_id
        self.state: Dict[str, Any] = {}
        self.checkpoints: Dict[str, Dict[str, Any]] = {}
        self.state_history: List[Dict[str, Any]] = []
        self.created_at = datetime.utcnow().isoformat()
        
    def merge_state(self, key: str, value: Any, checkpoint_name: str = None) -> bool:
        """Merge new data with existing state"""
        try:
            if key in self.state:
                if isinstance(self.state[key], dict) and isinstance(value, dict):
                    # Deep merge dictionaries
                    self.state[key] = {**self.state[key], **value}
                elif isinstance(self.state[key], list) and isinstance(value, list):
                    # Extend lists
                    self.state[key].extend(value)
                else:
                    # Replace value
                    self.state[key] = value
            else:
                self.state[key] = value
            
            self._record_change('merge', key, value, checkpoint_name)
            logger.debug(f"[WORKFLOW STATE] Merged {key}: {value}")
            return True
        except Exception as e:
            logger.error(f"[WORKFLOW STATE] Error merging state {key}: {e}")
            return False
    
    def set_state(self, key: str, value: Any, checkpoint_name: str = None) -> bool:
        """Set state value, overwriting existing"""
        try:
            self.state[key] = value
            self._record_change('set', key, value, checkpoint_name)
            logger.debug(f"[WORKFLOW STATE] Set {key}: {value}")

            
            return True
        except Exception as e:
            logger.error(f"[WORKFLOW STATE] Error setting state {key}: {e}")
            return False
    
    def get_state(self, key: str = None) -> Any:
        """Get state value or entire state"""
        try:
            if key is None:
                return deepcopy(self.state)
            return deepcopy(self.state.get(key))
        except Exception as e:
            logger.error(f"[WORKFLOW STATE] Error getting state {key}: {e}")
            return None
    
    def get_all_state(self) -> Dict[str, Any]:
        """Get all state data"""
        return self.get_state()
    
    def clear_state(self, keys: List[str] = None, checkpoint_name: str = None) -> bool:
        """Clear specific keys or entire state"""
        try:
            if keys is None:
                cleared_keys = list(self.state.keys())
                self.state.clear()
                self._record_change('clear_all', None, cleared_keys, checkpoint_name)
                logger.debug("[WORKFLOW STATE] Cleared entire state")
            else:
                cleared_values = {}
                for key in keys:
                    if key in self.state:
                        cleared_values[key] = self.state.pop(key)
                self._record_change('clear', keys, cleared_values, checkpoint_name)
                logger.debug(f"[WORKFLOW STATE] Cleared keys: {keys}")
            return True
        except Exception as e:
            logger.error(f"[WORKFLOW STATE] Error clearing state: {e}")
            return False
    
    def create_checkpoint(self, checkpoint_name: str) -> bool:
        """Create a named checkpoint of current state"""
        try:
            self.checkpoints[checkpoint_name] = {
                'state': deepcopy(self.state),
                'timestamp': datetime.utcnow().isoformat(),
                'workflow_id': self.workflow_id,
                'execution_id': self.execution_id
            }
            logger.info(f"[WORKFLOW STATE] Created checkpoint: {checkpoint_name}")
            return True
        except Exception as e:
            logger.error(f"[WORKFLOW STATE] Error creating checkpoint {checkpoint_name}: {e}")
            return False
    
    def restore_checkpoint(self, checkpoint_name: str) -> bool:
        """Restore state from a named checkpoint"""
        try:
            if checkpoint_name in self.checkpoints:
                self.state = deepcopy(self.checkpoints[checkpoint_name]['state'])
                self._record_change('restore', checkpoint_name, None, None)
                logger.info(f"[WORKFLOW STATE] Restored checkpoint: {checkpoint_name}")
                return True
            else:
                logger.warning(f"[WORKFLOW STATE] Checkpoint not found: {checkpoint_name}")
                return False
        except Exception as e:
            logger.error(f"[WORKFLOW STATE] Error restoring checkpoint {checkpoint_name}: {e}")
            return False
    
    def get_checkpoints(self) -> Dict[str, Dict[str, Any]]:
        """Get all available checkpoints"""
        return deepcopy(self.checkpoints)
    
    def _record_change(self, operation: str, key: Any, value: Any, checkpoint_name: str):
        """Record state change in history"""
        try:
            change_record = {
                'operation': operation,
                'key': key,
                'value': self._sanitize_for_logging(value),
                'checkpoint_name': checkpoint_name,
                'timestamp': datetime.utcnow().isoformat(),
                'state_size': len(self.state)
            }
            self.state_history.append(change_record)
            
            # Keep only last 100 changes to prevent memory issues
            if len(self.state_history) > 100:
                self.state_history = self.state_history[-100:]
        except Exception as e:
            logger.error(f"[WORKFLOW STATE] Error recording change: {e}")
    
    def _sanitize_for_logging(self, value: Any) -> Any:
        """Sanitize values for safe logging"""
        try:
            # Convert to string and limit length for logging
            str_value = str(value)
            if len(str_value) > 200:
                return str_value[:200] + "..."
            return value
        except:
            return "<non-serializable>"
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of current state for debugging"""
        return {
            'workflow_id': self.workflow_id,
            'execution_id': self.execution_id,
            'state_keys': list(self.state.keys()),
            'state_size': len(self.state),
            'checkpoints': list(self.checkpoints.keys()),
            'history_entries': len(self.state_history),
            'created_at': self.created_at,
            'last_change': self.state_history[-1] if self.state_history else None
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow state to dictionary for persistence"""
        return {
            'workflow_id': self.workflow_id,
            'execution_id': self.execution_id,
            'state': self.state,
            'checkpoints': self.checkpoints,
            'state_history': self.state_history,
            'created_at': self.created_at,
            'summary': self.get_state_summary()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowState':
        """Create WorkflowState from dictionary"""
        workflow_state = cls(data['workflow_id'], data['execution_id'])
        workflow_state.state = data.get('state', {})
        workflow_state.checkpoints = data.get('checkpoints', {})
        workflow_state.state_history = data.get('state_history', [])
        workflow_state.created_at = data.get('created_at', datetime.utcnow().isoformat())
        return workflow_state


class WorkflowStateManager:
    """Global manager for workflow states"""
    
    def __init__(self):
        self.active_states: Dict[str, WorkflowState] = {}
    
    def create_state(self, workflow_id: int, execution_id: str) -> WorkflowState:
        """Create new workflow state"""
        state_key = f"{workflow_id}:{execution_id}"
        workflow_state = WorkflowState(workflow_id, execution_id)
        self.active_states[state_key] = workflow_state
        logger.info(f"[WORKFLOW STATE MANAGER] Created state for {state_key}")
        return workflow_state
    
    def get_state(self, workflow_id: int, execution_id: str) -> Optional[WorkflowState]:
        """Get existing workflow state"""
        state_key = f"{workflow_id}:{execution_id}"
        return self.active_states.get(state_key)
    
    def remove_state(self, workflow_id: int, execution_id: str) -> bool:
        """Remove workflow state from memory"""
        state_key = f"{workflow_id}:{execution_id}"
        if state_key in self.active_states:
            del self.active_states[state_key]
            logger.info(f"[WORKFLOW STATE MANAGER] Removed state for {state_key}")
            return True
        return False
    
    def get_active_states(self) -> List[Dict[str, Any]]:
        """Get summary of all active states"""
        return [state.get_state_summary() for state in self.active_states.values()]


# Global state manager instance
workflow_state_manager = WorkflowStateManager()