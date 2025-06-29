"""
Ilanya Goals Engine - Goal Resolver

Determines when goals are complete, failed, or should be paused.
Implements resolution criteria based on progress, confidence, and time.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from .config import GoalsEngineConfig
from .models import Goal, GoalState


class GoalResolver:
    """
    Resolves goals based on completion criteria and failure conditions.
    
    Determines when goals should be marked as complete, failed, or paused
    based on progress, confidence, time constraints, and other factors.
    """
    
    def __init__(self, config: GoalsEngineConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the goal resolver.
        
        Args:
            config: Configuration containing resolution parameters
            logger: Optional logger for debugging and monitoring
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Resolution parameters
        self.resolution_confidence_threshold = config.resolution_confidence_threshold
        self.user_feedback_weight = config.user_feedback_weight
        self.time_decay_factor = config.time_decay_factor
        
        # Resolution tracking
        self.resolution_history: List[Dict[str, Any]] = []
        
        self.logger.info("Goal Resolver initialized")
    
    def check_resolution(self, goal: Goal) -> str:
        """
        Check if a goal should be resolved.
        
        Args:
            goal: Goal to check for resolution
            
        Returns:
            Resolution status: 'completed', 'failed', 'paused', or 'active'
        """
        # Check for completion
        if self._is_completed(goal):
            return 'completed'
        
        # Check for failure
        if self._has_failed(goal):
            return 'failed'
        
        # Check for pausing
        if self._should_be_paused(goal):
            return 'paused'
        
        # Goal is still active
        return 'active'
    
    def _is_completed(self, goal: Goal) -> bool:
        """
        Check if a goal is completed.
        
        Completion criteria:
        1. Progress above completion threshold
        2. Confidence above resolution threshold
        3. Goal is in active state
        """
        # Basic completion criteria
        if goal.state != GoalState.ACTIVE:
            return False
        
        if goal.progress < self.config.completion_threshold:
            return False
        
        if goal.confidence < self.resolution_confidence_threshold:
            return False
        
        # Additional completion checks
        if self._has_user_feedback_completion(goal):
            return True
        
        if self._has_self_assessment_completion(goal):
            return True
        
        # Time-based completion (if goal has been active for a long time with high progress)
        if self._has_time_based_completion(goal):
            return True
        
        return False
    
    def _has_failed(self, goal: Goal) -> bool:
        """
        Check if a goal has failed.
        
        Failure criteria:
        1. Low progress over extended time
        2. Low confidence
        3. No recent reinforcement
        4. Time constraints exceeded
        """
        # Check for low progress over time
        if (goal.progress < 0.1 and 
            (datetime.now() - goal.formation_time) > timedelta(hours=48)):
            return True
        
        # Check for low confidence over time
        if (goal.confidence < 0.2 and 
            (datetime.now() - goal.last_progress_update) > timedelta(hours=24)):
            return True
        
        # Check for time-to-live expiration
        if goal.time_to_live and (datetime.now() - goal.formation_time) > goal.time_to_live:
            return True
        
        # Check for user feedback indicating failure
        if self._has_user_feedback_failure(goal):
            return True
        
        return False
    
    def _should_be_paused(self, goal: Goal) -> bool:
        """
        Check if a goal should be paused.
        
        Pausing criteria:
        1. Temporary obstacles
        2. Resource constraints
        3. Competing priorities
        4. User request
        """
        # Check for competing goals
        if goal.competing_goals and len(goal.competing_goals) > 2:
            return True
        
        # Check for low buff strength (indicating reduced priority)
        if goal.get_total_buff_strength() < 0.1:
            return True
        
        # Check for user feedback indicating pause
        if self._has_user_feedback_pause(goal):
            return True
        
        # Check for time-based pausing (if goal has been inactive)
        if (goal.state == GoalState.ACTIVE and 
            (datetime.now() - goal.last_progress_update) > timedelta(hours=12)):
            return True
        
        return False
    
    def _has_user_feedback_completion(self, goal: Goal) -> bool:
        """Check if user feedback indicates goal completion."""
        # This would integrate with user feedback system
        # For now, return False as placeholder
        return False
    
    def _has_self_assessment_completion(self, goal: Goal) -> bool:
        """Check if agent's self-assessment indicates completion."""
        # Self-assessment based on goal properties
        if goal.progress >= 0.95 and goal.confidence >= 0.9:
            return True
        
        # Learning-based goals might have different completion criteria
        if goal.goal_type.value == 'learning':
            return goal.progress >= 0.8 and goal.confidence >= 0.85
        
        return False
    
    def _has_time_based_completion(self, goal: Goal) -> bool:
        """Check if goal should be completed based on time and progress."""
        time_since_formation = datetime.now() - goal.formation_time
        
        # If goal has been active for a long time with high progress, consider it complete
        if (time_since_formation > timedelta(hours=72) and 
            goal.progress >= 0.85 and 
            goal.confidence >= 0.7):
            return True
        
        return False
    
    def _has_user_feedback_failure(self, goal: Goal) -> bool:
        """Check if user feedback indicates goal failure."""
        # This would integrate with user feedback system
        # For now, return False as placeholder
        return False
    
    def _has_user_feedback_pause(self, goal: Goal) -> bool:
        """Check if user feedback indicates goal should be paused."""
        # This would integrate with user feedback system
        # For now, return False as placeholder
        return False
    
    def resolve_goal(self, goal: Goal, resolution_type: str, user_feedback: Optional[str] = None) -> Dict[str, Any]:
        """
        Resolve a goal with specific resolution type and optional user feedback.
        
        Args:
            goal: Goal to resolve
            resolution_type: Type of resolution ('completed', 'failed', 'paused')
            user_feedback: Optional user feedback about the resolution
            
        Returns:
            Resolution information
        """
        old_state = goal.state
        
        # Update goal state based on resolution type
        if resolution_type == 'completed':
            goal.state = GoalState.COMPLETED
            goal.progress = 1.0
            goal.confidence = 1.0
        elif resolution_type == 'failed':
            goal.state = GoalState.FAILED
        elif resolution_type == 'paused':
            goal.state = GoalState.PAUSED
        
        # Record resolution
        resolution_info = {
            'goal_id': goal.id,
            'goal_name': goal.name,
            'resolution_type': resolution_type,
            'old_state': old_state.value,
            'new_state': goal.state.value,
            'timestamp': datetime.now(),
            'user_feedback': user_feedback,
            'final_progress': goal.progress,
            'final_confidence': goal.confidence
        }
        
        self.resolution_history.append(resolution_info)
        
        # Log resolution
        self.logger.info(f"Goal resolved: {goal.name} ({resolution_type}) - "
                        f"Progress: {goal.progress:.3f}, Confidence: {goal.confidence:.3f}")
        
        return resolution_info
    
    def get_resolution_statistics(self) -> Dict[str, Any]:
        """Get statistics about goal resolutions."""
        if not self.resolution_history:
            return {
                'total_resolutions': 0,
                'completion_rate': 0.0,
                'failure_rate': 0.0,
                'pause_rate': 0.0
            }
        
        total_resolutions = len(self.resolution_history)
        completions = sum(1 for r in self.resolution_history if r['resolution_type'] == 'completed')
        failures = sum(1 for r in self.resolution_history if r['resolution_type'] == 'failed')
        pauses = sum(1 for r in self.resolution_history if r['resolution_type'] == 'paused')
        
        return {
            'total_resolutions': total_resolutions,
            'completion_rate': completions / total_resolutions,
            'failure_rate': failures / total_resolutions,
            'pause_rate': pauses / total_resolutions,
            'average_completion_progress': sum(
                r['final_progress'] for r in self.resolution_history 
                if r['resolution_type'] == 'completed'
            ) / completions if completions > 0 else 0.0,
            'average_completion_confidence': sum(
                r['final_confidence'] for r in self.resolution_history 
                if r['resolution_type'] == 'completed'
            ) / completions if completions > 0 else 0.0
        }
    
    def get_recent_resolutions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent goal resolutions."""
        return self.resolution_history[-limit:] if self.resolution_history else []
    
    def clear_resolution_history(self):
        """Clear the resolution history."""
        self.resolution_history.clear()
        self.logger.info("Goal resolution history cleared") 