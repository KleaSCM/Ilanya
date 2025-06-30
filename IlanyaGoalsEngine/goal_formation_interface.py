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
import numpy as np
from dataclasses import dataclass

from .models import Goal, GoalState, GoalType
from .config import GoalsEngineConfig


@dataclass
class ParetoPoint:
    """Represents a point on the Pareto frontier."""
    goal_id: str
    objectives: List[float]  # Multi-objective values
    is_dominated: bool = False
    rank: int = 0
    dominated_by: Optional[List['ParetoPoint']] = None
    domination_count: int = 0
    
    def __post_init__(self):
        """Initialize default values for complex attributes."""
        if self.dominated_by is None:
            self.dominated_by = []


class GoalFormationInterface:
    """
    Interface for converting desires into goals through field-like attraction dynamics.
    
    This interface monitors desire strengths and applies mathematical thresholds
    to determine when desires should become goals. It implements temporal stability
    requirements, field-like attraction mechanisms, and Pareto frontier analysis.
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
        
        # Pareto frontier analysis
        self.pareto_frontier: List[ParetoPoint] = []
        self.objective_weights = config.objective_weights
        
        # Adaptive thresholds
        self.adaptive_thresholds = {
            'strength': self.max_strength_threshold,
            'confidence': self.formation_confidence_threshold,
            'time': self.time_threshold
        }
        self.threshold_history: List[Tuple[datetime, Dict[str, float]]] = []
        
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
        
        # Update adaptive thresholds
        self._update_adaptive_thresholds(desires, current_time)
        
        # Perform Pareto frontier analysis
        pareto_candidates = self._analyze_pareto_frontier(desires)
        
        # Check each desire for goal formation potential
        for desire_id, desire_data in desires.items():
            desire_strength = desire_data.get('strength', 0.0)
            desire_confidence = desire_data.get('confidence', 0.0)
            
            # Check if desire meets formation criteria (including adaptive thresholds)
            if self._meets_formation_criteria(desire_id, desire_strength, desire_confidence, pareto_candidates):
                # Create goal from desire
                goal = self._create_goal_from_desire(desire_id, desire_data, current_time)
                if goal:
                    newly_formed_goals.append(goal)
                    self.logger.info(f"Formed new goal: {goal.name} (ID: {goal.id}) from desire {desire_id}")
        
        return newly_formed_goals
    
    def _analyze_pareto_frontier(self, desires: Dict[str, Dict]) -> List[str]:
        """
        Perform Pareto frontier analysis on desires.
        
        Args:
            desires: Dictionary of desire data
            
        Returns:
            List of desire IDs that are on the Pareto frontier
        """
        if len(desires) < 2:
            return list(desires.keys())
        
        # Create Pareto points for each desire
        pareto_points = []
        for desire_id, desire_data in desires.items():
            objectives = self._calculate_objectives(desire_data)
            pareto_points.append(ParetoPoint(
                goal_id=desire_id,
                objectives=objectives
            ))
        
        # Find Pareto frontier using non-dominated sorting
        pareto_frontier = self._non_dominated_sorting(pareto_points)
        
        # Return IDs of Pareto optimal desires
        return [point.goal_id for point in pareto_frontier if point.rank == 0]
    
    def _calculate_objectives(self, desire_data: Dict) -> List[float]:
        """
        Calculate multi-objective values for a desire.
        
        Objectives:
        1. Strength (maximize)
        2. Confidence (maximize) 
        3. Temporal stability (maximize)
        4. Field attraction (maximize)
        5. Resource efficiency (minimize, so we negate it)
        """
        strength = desire_data.get('strength', 0.0)
        confidence = desire_data.get('confidence', 0.0)
        
        # Calculate temporal stability
        stability = self._calculate_stability_factor(desire_data.get('id', ''))
        
        # Calculate field attraction
        field_attraction = self._calculate_field_attraction_probability(
            desire_data.get('id', ''), strength
        )
        
        # Calculate resource efficiency (inverse of complexity)
        resource_efficiency = 1.0 - (len(desire_data.get('source_traits', [])) * 0.1)
        resource_efficiency = max(0.0, min(1.0, resource_efficiency))
        
        return [strength, confidence, stability, field_attraction, -resource_efficiency]
    
    def _non_dominated_sorting(self, points: List[ParetoPoint]) -> List[ParetoPoint]:
        """
        Perform non-dominated sorting to find Pareto frontier.
        
        Args:
            points: List of Pareto points
            
        Returns:
            List of Pareto points with ranks assigned
        """
        # Initialize domination counts and dominated sets
        for point in points:
            point.dominated_by = []  # Always initialize as empty list
            point.domination_count = 0
        
        # Calculate domination relationships
        for i, point1 in enumerate(points):
            for j, point2 in enumerate(points):
                if i != j:
                    if self._dominates(point1.objectives, point2.objectives):
                        if point1.dominated_by is None:
                            point1.dominated_by = []
                        point1.dominated_by.append(point2)
                        point2.domination_count += 1
        
        # Assign ranks using non-dominated sorting
        current_rank = 0
        remaining_points = points.copy()
        
        while remaining_points:
            # Find points with domination_count = 0 (Pareto optimal)
            current_front = [p for p in remaining_points if p.domination_count == 0]
            
            # Assign rank to current front
            for point in current_front:
                point.rank = current_rank
                point.is_dominated = current_rank > 0
            
            # Remove current front from remaining points
            remaining_points = [p for p in remaining_points if p not in current_front]
            
            # Update domination counts for remaining points
            for point in current_front:
                if point.dominated_by:  # Check if not None
                    for dominated_point in point.dominated_by:
                        dominated_point.domination_count -= 1
            
            current_rank += 1
        
        return points
    
    def _dominates(self, objectives1: List[float], objectives2: List[float]) -> bool:
        """
        Check if objectives1 dominates objectives2.
        
        Args:
            objectives1: First set of objectives
            objectives2: Second set of objectives
            
        Returns:
            True if objectives1 dominates objectives2
        """
        # objectives1 dominates objectives2 if it's better in at least one objective
        # and not worse in any objective
        better_in_at_least_one = False
        
        for obj1, obj2 in zip(objectives1, objectives2):
            if obj1 > obj2:  # Better (higher is better for most objectives)
                better_in_at_least_one = True
            elif obj1 < obj2:  # Worse
                return False
        
        return better_in_at_least_one
    
    def _update_adaptive_thresholds(self, desires: Dict[str, Dict], current_time: datetime):
        """
        Update adaptive thresholds based on current system state.
        
        Args:
            desires: Current desire data
            current_time: Current timestamp
        """
        # Calculate system metrics
        total_desires = len(desires)
        avg_strength = np.mean([d.get('strength', 0.0) for d in desires.values()]) if desires else 0.0
        avg_confidence = np.mean([d.get('confidence', 0.0) for d in desires.values()]) if desires else 0.0
        
        # Calculate adaptive factors with more conservative bounds
        strength_factor = 1.0 + (avg_strength - 0.5) * 0.1  # Adjust by ±5% instead of ±10%
        confidence_factor = 1.0 + (avg_confidence - 0.5) * 0.1  # Adjust by ±5% instead of ±10%
        
        # System load factor (more desires = higher thresholds, but with limits)
        load_factor = 1.0 + (total_desires - 5) * 0.02  # Adjust by ±10% for 10 desires instead of ±25%
        load_factor = max(0.8, min(1.2, load_factor))  # Clamp load factor
        
        # Update thresholds with bounds
        new_strength_threshold = self.max_strength_threshold * strength_factor * load_factor
        new_confidence_threshold = self.formation_confidence_threshold * confidence_factor * load_factor
        new_time_threshold = self.time_threshold * (1.0 + (total_desires - 3) * 0.05)  # Reduced scaling
        
        # Apply bounds to prevent thresholds from becoming impossible
        self.adaptive_thresholds['strength'] = max(0.5, min(1.0, float(new_strength_threshold)))
        self.adaptive_thresholds['confidence'] = max(0.5, min(0.9, float(new_confidence_threshold)))
        self.adaptive_thresholds['time'] = max(timedelta(seconds=30), min(timedelta(minutes=10), new_time_threshold))
        
        # Store threshold history
        self.threshold_history.append((current_time, self.adaptive_thresholds.copy()))
        
        # Keep only recent history
        if len(self.threshold_history) > 100:
            self.threshold_history = self.threshold_history[-100:]
        
        self.logger.debug(f"Updated adaptive thresholds: strength={self.adaptive_thresholds['strength']:.3f}, "
                         f"confidence={self.adaptive_thresholds['confidence']:.3f}")
    
    def _update_desire_history(self, desires: Dict[str, Dict], current_time: datetime):
        """Update the history of desire strengths for temporal stability analysis."""
        for desire_id, desire_data in desires.items():
            strength = desire_data.get('strength', 0.0)
            
            if desire_id not in self.desire_strength_history:
                self.desire_strength_history[desire_id] = []
            
            # Add current strength to history
            self.desire_strength_history[desire_id].append((current_time, strength))
            
            # Remove old entries beyond time threshold
            cutoff_time = current_time - self.adaptive_thresholds['time']
            self.desire_strength_history[desire_id] = [
                (timestamp, strength) for timestamp, strength in self.desire_strength_history[desire_id]
                if timestamp > cutoff_time
            ]
    
    def _meets_formation_criteria(self, desire_id: str, strength: float, confidence: float, pareto_candidates: List[str]) -> bool:
        """
        Check if a desire meets the criteria for goal formation.
        
        Criteria:
        1. Strength reaches adaptive threshold
        2. Maintains strength over adaptive time threshold
        3. Confidence above adaptive threshold
        4. Field attraction dynamics satisfied
        5. Pareto optimal (if multiple candidates)
        """
        # Check adaptive thresholds
        if strength < self.adaptive_thresholds['strength']:
            return False
        
        if confidence < self.adaptive_thresholds['confidence']:
            return False
        
        # Check temporal stability
        if not self._has_temporal_stability(desire_id):
            return False
        
        # Check field attraction dynamics
        if not self._satisfies_field_attraction(desire_id, strength):
            return False
        
        # Check Pareto optimality (only if there are multiple candidates)
        if len(pareto_candidates) > 1 and desire_id not in pareto_candidates:
            return False
        
        return True
    
    def _has_temporal_stability(self, desire_id: str) -> bool:
        """
        Check if desire has maintained strength over the adaptive time threshold.
        
        This implements the temporal stability requirement for goal formation.
        """
        if desire_id not in self.desire_strength_history:
            return False
        
        history = self.desire_strength_history[desire_id]
        if len(history) < 2:  # Need at least 2 data points
            return False
        
        # Check if strength has been at threshold for the required time
        threshold_duration = timedelta(0)
        current_duration = timedelta(0)
        strength_threshold = self.adaptive_thresholds['strength']
        
        for i, (timestamp, strength) in enumerate(history):
            if strength >= strength_threshold:
                if i == 0:
                    current_duration = timedelta(0)
                else:
                    prev_timestamp = history[i-1][0]
                    current_duration += timestamp - prev_timestamp
            else:
                current_duration = timedelta(0)
            
            threshold_duration = max(threshold_duration, current_duration)
        
        # For demo purposes, be more lenient with temporal stability
        required_duration = self.adaptive_thresholds['time']
        if len(history) >= 3:  # If we have enough history, reduce requirement
            required_duration = required_duration * 0.5  # Only require half the time
        
        return threshold_duration >= required_duration
    
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
        strength_attraction = strength / self.adaptive_thresholds['strength']
        
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
            'max_strength_threshold': self.adaptive_thresholds['strength'],
            'time_threshold_seconds': self.adaptive_thresholds['time'].total_seconds(),
            'field_attraction_strength': self.field_attraction_strength,
            'pareto_frontier_size': len(self.pareto_frontier),
            'adaptive_thresholds': self.adaptive_thresholds.copy()
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