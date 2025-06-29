# Ilanya Goals Engine - Goal Formation Interface

# Handles the field-like attraction dynamics for converting desires into goals.
# Implements mathematical thresholds and temporal stability requirements for
# goal formation based on desire strength and persistence.

# Author: KleaSCM
# Email: KleaSCM@gmail.com
# License: MIT
# Version: 0.1.0


from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import math
import logging

from .models import Goal, GoalState, GoalType
from .config import GoalsEngineConfig


class GoalFormationInterface:
    """
    Interface for converting desires into goals through field-like attraction dynamics.
    
    This interface monitors desire strengths and applies mathematical thresholds
    to determine when desires should become goals. It implements temporal stability
    requirements and field-like attraction mechanisms.
    """
    
    def __init__(self, config: GoalsEngineConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the goal formation interface.
        
        Args:
            config: Configuration containing mathematical thresholds and parameters
            logger: Optional logger for debugging and monitoring
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Track desire strength history for temporal stability
        self.desire_strength_history: Dict[str, List[Tuple[datetime, float]]] = {}
        
        # Track goals in formation process
        self.forming_goals: Dict[str, Goal] = {}
        
        # Field attraction parameters
        self.field_attraction_strength = config.field_attraction_strength
        self.max_strength_threshold = config.max_strength_threshold
        self.time_threshold = config.time_threshold
        self.formation_confidence_threshold = config.formation_confidence_threshold
        
        self.logger.info(f"Goal Formation Interface initialized with max_strength_threshold={self.max_strength_threshold}, "
                        f"time_threshold={self.time_threshold}, field_attraction_strength={self.field_attraction_strength}")
    
    def process_desires(self, desires: Dict[str, Dict]) -> List[Goal]:
        """
        Process desires and identify which should become goals.
        
        Args:
            desires: Dictionary of desire data with strength, confidence, and metadata
            
        Returns:
            List of newly formed goals
        """
        newly_formed_goals = []
        current_time = datetime.now()
        
        # Update desire strength history
        self._update_desire_history(desires, current_time)
        
        # Check each desire for goal formation potential
        for desire_id, desire_data in desires.items():
            desire_strength = desire_data.get('strength', 0.0)
            desire_confidence = desire_data.get('confidence', 0.0)
            
            # Check if desire meets formation criteria
            if self._meets_formation_criteria(desire_id, desire_strength, desire_confidence):
                # Create goal from desire
                goal = self._create_goal_from_desire(desire_id, desire_data, current_time)
                if goal:
                    newly_formed_goals.append(goal)
                    self.logger.info(f"Formed new goal: {goal.name} (ID: {goal.id}) from desire {desire_id}")
        
        return newly_formed_goals
    
    def _update_desire_history(self, desires: Dict[str, Dict], current_time: datetime):
        """Update the history of desire strengths for temporal stability analysis."""
        for desire_id, desire_data in desires.items():
            strength = desire_data.get('strength', 0.0)
            
            if desire_id not in self.desire_strength_history:
                self.desire_strength_history[desire_id] = []
            
            # Add current strength to history
            self.desire_strength_history[desire_id].append((current_time, strength))
            
            # Remove old entries beyond time threshold
            cutoff_time = current_time - self.time_threshold
            self.desire_strength_history[desire_id] = [
                (timestamp, strength) for timestamp, strength in self.desire_strength_history[desire_id]
                if timestamp > cutoff_time
            ]
    
    def _meets_formation_criteria(self, desire_id: str, strength: float, confidence: float) -> bool:
        """
        Check if a desire meets the criteria for goal formation.
        
        Criteria:
        1. Strength reaches maximum threshold
        2. Maintains maximum strength over time threshold
        3. Confidence above formation threshold
        4. Field attraction dynamics satisfied
        """
        # Check basic thresholds
        if strength < self.max_strength_threshold:
            return False
        
        if confidence < self.formation_confidence_threshold:
            return False
        
        # Check temporal stability
        if not self._has_temporal_stability(desire_id):
            return False
        
        # Check field attraction dynamics
        if not self._satisfies_field_attraction(desire_id, strength):
            return False
        
        return True
    
    def _has_temporal_stability(self, desire_id: str) -> bool:
        """
        Check if desire has maintained maximum strength over the time threshold.
        
        This implements the temporal stability requirement for goal formation.
        """
        if desire_id not in self.desire_strength_history:
            return False
        
        history = self.desire_strength_history[desire_id]
        if len(history) < 2:  # Need at least 2 data points
            return False
        
        # Check if strength has been at maximum for the required time
        max_strength_duration = timedelta(0)
        current_duration = timedelta(0)
        
        for i, (timestamp, strength) in enumerate(history):
            if strength >= self.max_strength_threshold:
                if i == 0:
                    current_duration = timedelta(0)
                else:
                    prev_timestamp = history[i-1][0]
                    current_duration += timestamp - prev_timestamp
            else:
                current_duration = timedelta(0)
            
            max_strength_duration = max(max_strength_duration, current_duration)
        
        return max_strength_duration >= self.time_threshold
    
    def _satisfies_field_attraction(self, desire_id: str, strength: float) -> bool:
        """
        Check if desire satisfies field-like attraction dynamics.
        
        Field attraction is based on the mathematical principle that desires
        naturally "attract" toward goal formation when certain conditions are met.
        """
        # Calculate field attraction probability
        attraction_probability = self._calculate_field_attraction_probability(desire_id, strength)
        
        # Apply stochastic threshold
        import random
        return random.random() < attraction_probability
    
    def _calculate_field_attraction_probability(self, desire_id: str, strength: float) -> float:
        """
        Calculate the probability of field attraction based on desire properties.
        
        This implements a mathematical model of field-like attraction where
        desires naturally evolve toward goal formation.
        """
        # Base attraction from strength
        strength_attraction = strength / self.max_strength_threshold
        
        # Temporal stability factor
        stability_factor = self._calculate_stability_factor(desire_id)
        
        # Field strength factor
        field_factor = self.field_attraction_strength
        
        # Combined probability with diminishing returns
        base_probability = strength_attraction * stability_factor * field_factor
        
        # Apply sigmoid function for smooth transition
        probability = 1.0 / (1.0 + math.exp(-10 * (base_probability - 0.8)))
        
        return min(1.0, probability)
    
    def _calculate_stability_factor(self, desire_id: str) -> float:
        """Calculate stability factor based on desire strength variance over time."""
        if desire_id not in self.desire_strength_history:
            return 0.0
        
        history = self.desire_strength_history[desire_id]
        if len(history) < 2:
            return 0.0
        
        # Calculate variance in strength
        strengths = [strength for _, strength in history]
        mean_strength = sum(strengths) / len(strengths)
        variance = sum((s - mean_strength) ** 2 for s in strengths) / len(strengths)
        
        # Lower variance = higher stability
        stability_factor = 1.0 / (1.0 + variance * 10)
        
        return stability_factor
    
    def _create_goal_from_desire(self, desire_id: str, desire_data: Dict, formation_time: datetime) -> Optional[Goal]:
        """
        Create a goal from a desire that meets formation criteria.
        
        Args:
            desire_id: ID of the desire
            desire_data: Desire data including strength, confidence, and metadata
            formation_time: Time when goal is being formed
            
        Returns:
            Newly created goal or None if creation fails
        """
        try:
            # Extract desire properties
            strength = desire_data.get('strength', 0.0)
            confidence = desire_data.get('confidence', 0.0)
            source_traits = desire_data.get('source_traits', [])
            name = desire_data.get('name', f"Goal from {desire_id}")
            description = desire_data.get('description', f"Goal formed from desire {desire_id}")
            
            # Determine goal type based on desire properties
            goal_type = self._determine_goal_type(desire_data)
            
            # Create goal
            goal = Goal(
                name=name,
                description=description,
                goal_type=goal_type,
                source_desires=[desire_id],
                source_traits=source_traits,
                formation_strength=strength,
                formation_time=formation_time,
                state=GoalState.FORMING,
                current_strength=strength,
                confidence=confidence,
                buff_decay_rate=self.config.buff_decay_rate
            )
            
            # Apply initial buffs to source traits and desires
            self._apply_initial_buffs(goal, desire_data)
            
            return goal
            
        except Exception as e:
            self.logger.error(f"Failed to create goal from desire {desire_id}: {str(e)}")
            return None
    
    def _determine_goal_type(self, desire_data: Dict) -> GoalType:
        """Determine the type of goal based on desire properties."""
        # This is a simple heuristic - can be enhanced with more sophisticated analysis
        name = desire_data.get('name', '').lower()
        description = desire_data.get('description', '').lower()
        
        if any(word in name or word in description for word in ['learn', 'understand', 'know', 'study']):
            return GoalType.LEARNING
        elif any(word in name or word in description for word in ['social', 'friend', 'talk', 'meet']):
            return GoalType.SOCIAL
        elif any(word in name or word in description for word in ['create', 'art', 'write', 'design']):
            return GoalType.CREATIVE
        elif any(word in name or word in description for word in ['solve', 'problem', 'fix', 'resolve']):
            return GoalType.PROBLEM_SOLVING
        elif any(word in name or word in description for word in ['improve', 'grow', 'develop', 'better']):
            return GoalType.SELF_IMPROVEMENT
        elif any(word in name or word in description for word in ['explore', 'discover', 'find', 'search']):
            return GoalType.EXPLORATION
        else:
            return GoalType.EMERGENT
    
    def _apply_initial_buffs(self, goal: Goal, desire_data: Dict):
        """Apply initial buffs to source traits and desires."""
        source_traits = desire_data.get('source_traits', [])
        source_desires = [desire_data.get('id', 'unknown')]
        
        # Apply trait buffs
        for trait_id in source_traits:
            buff_strength = self.config.trait_buff_multiplier
            goal.apply_trait_buff(trait_id, buff_strength)
        
        # Apply desire buffs
        for desire_id in source_desires:
            buff_strength = self.config.desire_buff_multiplier
            goal.apply_desire_buff(desire_id, buff_strength)
    
    def get_formation_statistics(self) -> Dict[str, Any]:
        """Get statistics about goal formation process."""
        total_desires = len(self.desire_strength_history)
        forming_goals = len(self.forming_goals)
        
        # Calculate average strength across all desires
        total_strength = 0.0
        strength_count = 0
        for history in self.desire_strength_history.values():
            for _, strength in history:
                total_strength += strength
                strength_count += 1
        
        avg_strength = total_strength / strength_count if strength_count > 0 else 0.0
        
        return {
            'total_desires_tracked': total_desires,
            'goals_in_formation': forming_goals,
            'average_desire_strength': avg_strength,
            'max_strength_threshold': self.max_strength_threshold,
            'time_threshold_seconds': self.time_threshold.total_seconds(),
            'field_attraction_strength': self.field_attraction_strength
        }
    
    def cleanup_old_history(self, max_age: timedelta):
        """Clean up old desire strength history to prevent memory bloat."""
        current_time = datetime.now()
        cutoff_time = current_time - max_age
        
        for desire_id in list(self.desire_strength_history.keys()):
            self.desire_strength_history[desire_id] = [
                (timestamp, strength) for timestamp, strength in self.desire_strength_history[desire_id]
                if timestamp > cutoff_time
            ]
            
            # Remove empty histories
            if not self.desire_strength_history[desire_id]:
                del self.desire_strength_history[desire_id]