"""
Ilanya Goals Engine - Goal Monitor

Monitors goal progress and updates goal states based on various criteria.
Implements progress tracking, confidence assessment, and goal state management.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import math

from .config import GoalsEngineConfig
from .models import Goal, GoalState


class GoalMonitor:
    """
    Monitors goal progress and manages goal states.
    
    Tracks progress toward goal completion, assesses confidence levels,
    and manages goal state transitions based on progress and time.
    """
    
    def __init__(self, config: GoalsEngineConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the goal monitor.
        
        Args:
            config: Configuration containing monitoring parameters
            logger: Optional logger for debugging and monitoring
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Monitoring parameters
        self.progress_check_interval = config.progress_check_interval
        self.completion_threshold = config.completion_threshold
        self.self_assessment_confidence = config.self_assessment_confidence
        
        # Progress tracking
        self.progress_history: Dict[str, List[Dict[str, Any]]] = {}
        
        self.logger.info("Goal Monitor initialized")
    
    def update_goals(self, goals: Dict[str, Goal], time_delta: timedelta) -> Dict[str, Any]:
        """
        Update all goals with progress monitoring and state management.
        
        Args:
            goals: Dictionary of active goals
            time_delta: Time passed since last update
            
        Returns:
            Dictionary containing monitoring results
        """
        monitoring_results = {
            'goals_updated': 0,
            'progress_changes': [],
            'state_transitions': [],
            'confidence_updates': []
        }
        
        current_time = datetime.now()
        
        for goal_id, goal in goals.items():
            # Update goal progress
            progress_change = self._update_goal_progress(goal, time_delta)
            if progress_change:
                monitoring_results['progress_changes'].append({
                    'goal_id': goal_id,
                    'old_progress': progress_change['old_progress'],
                    'new_progress': progress_change['new_progress'],
                    'change_reason': progress_change['reason']
                })
                monitoring_results['goals_updated'] += 1
            
            # Update goal confidence
            confidence_change = self._update_goal_confidence(goal)
            if confidence_change:
                monitoring_results['confidence_updates'].append({
                    'goal_id': goal_id,
                    'old_confidence': confidence_change['old_confidence'],
                    'new_confidence': confidence_change['new_confidence']
                })
            
            # Check for state transitions
            state_transition = self._check_state_transition(goal)
            if state_transition:
                monitoring_results['state_transitions'].append({
                    'goal_id': goal_id,
                    'old_state': state_transition['old_state'],
                    'new_state': state_transition['new_state'],
                    'reason': state_transition['reason']
                })
            
            # Update progress history
            self._update_progress_history(goal_id, goal, current_time)
        
        return monitoring_results
    
    def _update_goal_progress(self, goal: Goal, time_delta: timedelta) -> Optional[Dict[str, Any]]:
        """
        Update goal progress based on time and goal properties.
        
        Args:
            goal: Goal to update
            time_delta: Time passed since last update
            
        Returns:
            Progress change information or None if no change
        """
        old_progress = goal.progress
        
        # Calculate progress change based on goal type and properties
        progress_change = self._calculate_progress_change(goal, time_delta)
        
        # Apply progress change
        new_progress = max(0.0, min(1.0, old_progress + progress_change))
        
        if abs(new_progress - old_progress) > 0.001:  # Significant change threshold
            goal.update_progress(new_progress)
            
            return {
                'old_progress': old_progress,
                'new_progress': new_progress,
                'change_amount': progress_change,
                'reason': self._get_progress_change_reason(goal, progress_change)
            }
        
        return None
    
    def _calculate_progress_change(self, goal: Goal, time_delta: timedelta) -> float:
        """
        Calculate progress change based on goal properties and time.
        
        This implements a sophisticated progress model that considers:
        - Goal type and complexity
        - Current goal strength
        - Time invested
        - Buff effects
        """
        # Base progress rate depends on goal type
        base_rate = self._get_base_progress_rate(goal.goal_type)
        
        # Strength multiplier (stronger goals progress faster)
        strength_multiplier = goal.current_strength
        
        # Time multiplier (progress increases with time invested)
        time_multiplier = min(2.0, 1.0 + (time_delta.total_seconds() / 3600.0) * 0.1)
        
        # Buff multiplier (goals with more buffs progress faster)
        buff_multiplier = 1.0 + (goal.get_total_buff_strength() * 0.2)
        
        # Diminishing returns as progress increases
        progress_factor = 1.0 - (goal.progress * 0.5)
        
        # Calculate total progress change
        progress_change = (
            base_rate * 
            strength_multiplier * 
            time_multiplier * 
            buff_multiplier * 
            progress_factor * 
            (time_delta.total_seconds() / 3600.0)  # Convert to hours
        )
        
        return max(0.0, progress_change)
    
    def _get_base_progress_rate(self, goal_type) -> float:
        """Get base progress rate for different goal types."""
        rates = {
            'learning': 0.1,      # Learning goals progress steadily
            'social': 0.15,       # Social goals can progress quickly
            'creative': 0.08,     # Creative goals are slower
            'problem_solving': 0.12,  # Problem solving varies
            'self_improvement': 0.06,  # Self-improvement is slow
            'exploration': 0.14,   # Exploration can be fast
            'maintenance': 0.05,   # Maintenance is slow
            'emergent': 0.1       # Default rate for emergent goals
        }
        
        return rates.get(goal_type.value, 0.1)
    
    def _get_progress_change_reason(self, goal: Goal, progress_change: float) -> str:
        """Get human-readable reason for progress change."""
        if progress_change > 0.05:
            return "significant_progress"
        elif progress_change > 0.01:
            return "steady_progress"
        elif progress_change > 0:
            return "minor_progress"
        else:
            return "no_progress"
    
    def _update_goal_confidence(self, goal: Goal) -> Optional[Dict[str, Any]]:
        """
        Update goal confidence based on progress and other factors.
        
        Args:
            goal: Goal to update
            
        Returns:
            Confidence change information or None if no change
        """
        old_confidence = goal.confidence
        
        # Calculate new confidence based on multiple factors
        progress_confidence = goal.progress * 0.6  # Progress contributes 60%
        strength_confidence = goal.current_strength * 0.3  # Strength contributes 30%
        buff_confidence = min(0.1, goal.get_total_buff_strength() * 0.1)  # Buffs contribute 10%
        
        new_confidence = progress_confidence + strength_confidence + buff_confidence
        new_confidence = max(0.0, min(1.0, new_confidence))
        
        if abs(new_confidence - old_confidence) > 0.01:
            goal.confidence = new_confidence
            
            return {
                'old_confidence': old_confidence,
                'new_confidence': new_confidence
            }
        
        return None
    
    def _check_state_transition(self, goal: Goal) -> Optional[Dict[str, Any]]:
        """
        Check if goal should transition to a different state.
        
        Args:
            goal: Goal to check
            
        Returns:
            State transition information or None if no transition
        """
        old_state = goal.state
        
        # Check for completion
        if goal.progress >= self.completion_threshold and goal.confidence >= 0.8:
            if goal.state != GoalState.COMPLETED:
                goal.state = GoalState.COMPLETED
                return {
                    'old_state': old_state,
                    'new_state': GoalState.COMPLETED,
                    'reason': 'goal_completed'
                }
        
        # Check for activation
        elif goal.progress >= 0.5 and goal.state == GoalState.FORMING:
            goal.state = GoalState.ACTIVE
            return {
                'old_state': old_state,
                'new_state': GoalState.ACTIVE,
                'reason': 'goal_activated'
            }
        
        # Check for failure (low progress and confidence over time)
        elif (goal.progress < 0.1 and goal.confidence < 0.3 and 
              (datetime.now() - goal.formation_time) > timedelta(hours=24)):
            if goal.state not in [GoalState.FAILED, GoalState.PRUNED]:
                goal.state = GoalState.FAILED
                return {
                    'old_state': old_state,
                    'new_state': GoalState.FAILED,
                    'reason': 'goal_failed'
                }
        
        return None
    
    def _update_progress_history(self, goal_id: str, goal: Goal, timestamp: datetime):
        """Update progress history for the goal."""
        if goal_id not in self.progress_history:
            self.progress_history[goal_id] = []
        
        history_entry = {
            'timestamp': timestamp,
            'progress': goal.progress,
            'confidence': goal.confidence,
            'strength': goal.current_strength,
            'state': goal.state.value,
            'buff_strength': goal.get_total_buff_strength()
        }
        
        self.progress_history[goal_id].append(history_entry)
        
        # Limit history size
        if len(self.progress_history[goal_id]) > 100:
            self.progress_history[goal_id] = self.progress_history[goal_id][-50:]
    
    def get_goal_progress_summary(self, goal_id: str) -> Dict[str, Any]:
        """Get progress summary for a specific goal."""
        if goal_id not in self.progress_history:
            return {}
        
        history = self.progress_history[goal_id]
        if not history:
            return {}
        
        # Calculate progress statistics
        progresses = [entry['progress'] for entry in history]
        confidences = [entry['confidence'] for entry in history]
        strengths = [entry['strength'] for entry in history]
        
        return {
            'current_progress': progresses[-1] if progresses else 0.0,
            'average_progress': sum(progresses) / len(progresses) if progresses else 0.0,
            'progress_trend': self._calculate_trend(progresses),
            'current_confidence': confidences[-1] if confidences else 0.0,
            'average_confidence': sum(confidences) / len(confidences) if confidences else 0.0,
            'current_strength': strengths[-1] if strengths else 0.0,
            'average_strength': sum(strengths) / len(strengths) if strengths else 0.0,
            'history_length': len(history)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend in a list of values."""
        if len(values) < 2:
            return "stable"
        
        # Calculate linear trend
        recent_values = values[-10:] if len(values) >= 10 else values
        if len(recent_values) < 2:
            return "stable"
        
        # Simple trend calculation
        first_half = recent_values[:len(recent_values)//2]
        second_half = recent_values[len(recent_values)//2:]
        
        if not first_half or not second_half:
            return "stable"
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        if second_avg > first_avg + 0.05:
            return "increasing"
        elif second_avg < first_avg - 0.05:
            return "decreasing"
        else:
            return "stable"
    
    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """Get overall monitoring statistics."""
        total_goals = len(self.progress_history)
        total_history_entries = sum(len(history) for history in self.progress_history.values())
        
        return {
            'total_goals_monitored': total_goals,
            'total_history_entries': total_history_entries,
            'average_history_length': total_history_entries / total_goals if total_goals > 0 else 0,
            'progress_check_interval_seconds': self.progress_check_interval.total_seconds(),
            'completion_threshold': self.completion_threshold,
            'self_assessment_confidence': self.self_assessment_confidence
        }