
# Ilanya Goals Engine - Main Engine

# Main orchestrator for the Goals Engine that manages goal formation, monitoring,
# resolution, and mathematical stability controls. Implements Nash equilibrium,
# multi-objective optimization, and Lyapunov stability analysis.

# Author: KleaSCM
# Email: KleaSCM@gmail.com
# License: MIT
# Version: 0.1.0

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import math
import numpy as np

from .config import GoalsEngineConfig
from .models import Goal, GoalState, GoalType
from .goal_formation_interface import GoalFormationInterface
from .goal_monitor import GoalMonitor
from .goal_resolver import GoalResolver


class GoalsEngine:
    """
    Main Goals Engine orchestrator.
    
    Manages the complete goal lifecycle from formation to resolution,
    including mathematical stability controls, conflict resolution,
    and feedback loops to the trait and desire systems.
    """
    
    def __init__(self, config: GoalsEngineConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the Goals Engine.
        
        Args:
            config: Configuration containing all mathematical parameters
            logger: Optional logger for debugging and monitoring
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize components
        self.formation_interface = GoalFormationInterface(config, self.logger)
        self.monitor = GoalMonitor(config, self.logger)
        self.resolver = GoalResolver(config, self.logger)
        
        # Active goals storage
        self.active_goals: Dict[str, Goal] = {}
        self.goal_history: List[Goal] = []
        
        # Mathematical stability tracking
        self.system_entropy = 0.0
        self.lyapunov_stability = 1.0
        self.last_stability_check = datetime.now()
        
        # Performance tracking
        self.total_goals_formed = 0
        self.total_goals_completed = 0
        self.total_goals_pruned = 0
        
        self.logger.info("Goals Engine initialized successfully")
    
    def process_desires(self, desires: Dict[str, Dict]) -> List[Goal]:
        """
        Process desires and form new goals.
        
        Args:
            desires: Dictionary of desire data from the Desire Engine
            
        Returns:
            List of newly formed goals
        """
        # Form new goals from desires
        new_goals = self.formation_interface.process_desires(desires)
        
        # Add new goals to active goals
        for goal in new_goals:
            self.active_goals[goal.id] = goal
            self.total_goals_formed += 1
            self.logger.info(f"Added new goal: {goal.name} (ID: {goal.id})")
        
        return new_goals
    
    def update_goals(self, time_delta: timedelta) -> Dict[str, Any]:
        """
        Update all active goals and apply mathematical controls.
        
        Args:
            time_delta: Time passed since last update
            
        Returns:
            Dictionary containing update results and system metrics
        """
        current_time = datetime.now()
        
        # Update goal monitoring
        monitoring_results = self.monitor.update_goals(self.active_goals, time_delta)
        
        # Apply goal buffs to traits and desires
        buff_results = self._apply_goal_buffs()
        
        # Check for goal resolution
        resolution_results = self._check_goal_resolution()
        
        # Apply mathematical stability controls
        stability_results = self._apply_stability_controls()
        
        # Prune unreinforced goals
        pruning_results = self._prune_goals()
        
        # Update system metrics
        self._update_system_metrics()
        
        return {
            'monitoring': monitoring_results,
            'buffing': buff_results,
            'resolution': resolution_results,
            'stability': stability_results,
            'pruning': pruning_results,
            'system_metrics': self._get_system_metrics()
        }
    
    def _apply_goal_buffs(self) -> Dict[str, Any]:
        """Apply goal buffs to traits and desires."""
        buff_results = {
            'trait_buffs': {},
            'desire_buffs': {},
            'total_buff_strength': 0.0
        }
        
        for goal_id, goal in self.active_goals.items():
            if goal.state == GoalState.ACTIVE:
                # Apply trait buffs
                for trait_id, buff_strength in goal.trait_buffs.items():
                    if trait_id not in buff_results['trait_buffs']:
                        buff_results['trait_buffs'][trait_id] = 0.0
                    buff_results['trait_buffs'][trait_id] += buff_strength
                
                # Apply desire buffs
                for desire_id, buff_strength in goal.desire_buffs.items():
                    if desire_id not in buff_results['desire_buffs']:
                        buff_results['desire_buffs'][desire_id] = 0.0
                    buff_results['desire_buffs'][desire_id] += buff_strength
                
                buff_results['total_buff_strength'] += goal.get_total_buff_strength()
        
        return buff_results
    
    def _check_goal_resolution(self) -> Dict[str, Any]:
        """Check for goals that should be resolved."""
        resolution_results = {
            'completed_goals': [],
            'failed_goals': [],
            'paused_goals': []
        }
        
        for goal_id, goal in list(self.active_goals.items()):
            resolution_status = self.resolver.check_resolution(goal)
            
            if resolution_status == 'completed':
                goal.state = GoalState.COMPLETED
                resolution_results['completed_goals'].append(goal)
                self.total_goals_completed += 1
                self.logger.info(f"Goal completed: {goal.name} (ID: {goal.id})")
                
            elif resolution_status == 'failed':
                goal.state = GoalState.FAILED
                resolution_results['failed_goals'].append(goal)
                self.logger.info(f"Goal failed: {goal.name} (ID: {goal.id})")
                
            elif resolution_status == 'paused':
                goal.state = GoalState.PAUSED
                resolution_results['paused_goals'].append(goal)
                self.logger.info(f"Goal paused: {goal.name} (ID: {goal.id})")
        
        return resolution_results
    
    def _apply_stability_controls(self) -> Dict[str, Any]:
        """Apply mathematical stability controls including Nash equilibrium and Lyapunov stability."""
        stability_results = {
            'nash_equilibrium_applied': False,
            'lyapunov_stability': self.lyapunov_stability,
            'system_entropy': self.system_entropy,
            'conflicts_resolved': 0
        }
        
        # Check if we need Nash equilibrium for competing goals
        competing_goals = [g for g in self.active_goals.values() if g.competing_goals]
        if len(competing_goals) > 1:
            nash_weights = self._calculate_nash_equilibrium(competing_goals)
            for goal, weight in zip(competing_goals, nash_weights):
                goal.nash_equilibrium_weight = weight
            stability_results['nash_equilibrium_applied'] = True
            stability_results['conflicts_resolved'] = len(competing_goals)
        
        # Apply Lyapunov stability analysis
        self.lyapunov_stability = self._calculate_lyapunov_stability()
        stability_results['lyapunov_stability'] = self.lyapunov_stability
        
        # Check if system is stable
        if self.lyapunov_stability < self.config.lyapunov_stability_threshold:
            self.logger.warning(f"System stability below threshold: {self.lyapunov_stability}")
            self._apply_stability_corrections()
        
        return stability_results
    
    def _calculate_nash_equilibrium(self, competing_goals: List[Goal]) -> List[float]:
        """
        Calculate Nash equilibrium weights for competing goals.
        
        This implements a simplified Nash equilibrium calculation for goal competition.
        """
        if len(competing_goals) < 2:
            return [1.0] * len(competing_goals)
        
        # Initialize weights
        weights = [1.0 / len(competing_goals)] * len(competing_goals)
        
        # Iterative Nash equilibrium calculation
        for iteration in range(self.config.nash_iteration_limit):
            old_weights = weights.copy()
            
            # Calculate new weights based on goal strengths and interactions
            for i, goal in enumerate(competing_goals):
                # Base weight from goal strength
                strength_weight = goal.current_strength
                
                # Interaction weight (negative for competing goals)
                interaction_weight = 0.0
                for j, other_goal in enumerate(competing_goals):
                    if i != j and other_goal.id in goal.competing_goals:
                        interaction_weight -= goal.interaction_strength * weights[j]
                
                # Combined weight
                weights[i] = max(0.0, strength_weight + interaction_weight)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            
            # Check convergence
            weight_change = sum(abs(w1 - w2) for w1, w2 in zip(weights, old_weights))
            if weight_change < self.config.nash_convergence_threshold:
                break
        
        return weights
    
    def _calculate_lyapunov_stability(self) -> float:
        """
        Calculate Lyapunov stability measure for the goal system.
        
        Returns a value between 0 and 1, where 1 is maximally stable.
        """
        if not self.active_goals:
            return 1.0
        
        # Calculate system state vector
        goal_strengths = [goal.current_strength for goal in self.active_goals.values()]
        goal_progresses = [goal.progress for goal in self.active_goals.values()]
        
        # Calculate variance in goal states (lower variance = more stable)
        strength_variance = np.var(goal_strengths) if len(goal_strengths) > 1 else 0.0
        progress_variance = np.var(goal_progresses) if len(goal_progresses) > 1 else 0.0
        
        # Calculate interaction stability
        interaction_stability = 0.0
        total_interactions = 0
        
        for goal in self.active_goals.values():
            if goal.competing_goals or goal.supporting_goals:
                interaction_stability += goal.interaction_strength
                total_interactions += 1
        
        if total_interactions > 0:
            interaction_stability /= total_interactions
        
        # Combined stability measure
        stability = 1.0 / (1.0 + strength_variance + progress_variance + (1.0 - interaction_stability))
        
        return max(0.0, min(1.0, stability))
    
    def _apply_stability_corrections(self):
        """Apply corrections when system stability is low."""
        # Reduce goal interaction strengths
        for goal in self.active_goals.values():
            goal.interaction_strength *= 0.9
        
        # Prune some goals to reduce complexity
        if len(self.active_goals) > self.config.max_active_goals:
            goals_to_prune = len(self.active_goals) - self.config.max_active_goals
            weakest_goals = sorted(
                self.active_goals.values(),
                key=lambda g: g.current_strength
            )[:goals_to_prune]
            
            for goal in weakest_goals:
                goal.state = GoalState.PRUNED
                self.logger.info(f"Pruned goal for stability: {goal.name} (ID: {goal.id})")
    
    def _prune_goals(self) -> Dict[str, Any]:
        """Prune goals that are no longer reinforced or relevant."""
        pruning_results = {
            'pruned_goals': [],
            'pruned_count': 0
        }
        
        for goal_id, goal in list(self.active_goals.items()):
            if goal.should_be_pruned(self.config.pruning_threshold):
                goal.state = GoalState.PRUNED
                pruning_results['pruned_goals'].append(goal)
                pruning_results['pruned_count'] += 1
                self.total_goals_pruned += 1
                self.logger.info(f"Pruned goal: {goal.name} (ID: {goal.id})")
        
        # Remove pruned goals from active goals
        self.active_goals = {
            goal_id: goal for goal_id, goal in self.active_goals.items()
            if goal.state not in [GoalState.COMPLETED, GoalState.FAILED, GoalState.PRUNED]
        }
        
        return pruning_results
    
    def _update_system_metrics(self):
        """Update system-wide metrics."""
        # Calculate system entropy
        if self.active_goals:
            goal_strengths = [goal.current_strength for goal in self.active_goals.values()]
            total_strength = sum(goal_strengths)
            
            if total_strength > 0:
                probabilities = [s / total_strength for s in goal_strengths]
                self.system_entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
            else:
                self.system_entropy = 0.0
        else:
            self.system_entropy = 0.0
        
        # Update Lyapunov stability
        self.lyapunov_stability = self._calculate_lyapunov_stability()
        self.last_stability_check = datetime.now()
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        active_goals = [g for g in self.active_goals.values() if g.state == GoalState.ACTIVE]
        
        return {
            'total_goals_formed': self.total_goals_formed,
            'total_goals_completed': self.total_goals_completed,
            'total_goals_pruned': self.total_goals_pruned,
            'active_goals_count': len(active_goals),
            'total_goals_count': len(self.active_goals),
            'system_entropy': self.system_entropy,
            'lyapunov_stability': self.lyapunov_stability,
            'average_goal_strength': np.mean([g.current_strength for g in self.active_goals.values()]) if self.active_goals else 0.0,
            'average_goal_progress': np.mean([g.progress for g in self.active_goals.values()]) if self.active_goals else 0.0,
            'total_buff_strength': sum(g.get_total_buff_strength() for g in self.active_goals.values())
        }
    
    def get_goals_for_traits(self, trait_ids: List[str]) -> List[Goal]:
        """Get goals that are related to specific traits."""
        related_goals = []
        
        for goal in self.active_goals.values():
            if any(trait_id in goal.source_traits for trait_id in trait_ids):
                related_goals.append(goal)
        
        return related_goals
    
    def get_goals_for_desires(self, desire_ids: List[str]) -> List[Goal]:
        """Get goals that are related to specific desires."""
        related_goals = []
        
        for goal in self.active_goals.values():
            if any(desire_id in goal.source_desires for desire_id in desire_ids):
                related_goals.append(goal)
        
        return related_goals
    
    def save_state(self) -> Dict[str, Any]:
        """Save the current state of the Goals Engine."""
        return {
            'active_goals': {goal_id: goal.to_dict() for goal_id, goal in self.active_goals.items()},
            'goal_history': [goal.to_dict() for goal in self.goal_history],
            'system_metrics': self._get_system_metrics(),
            'formation_statistics': self.formation_interface.get_formation_statistics(),
            'config': self.config.to_dict()
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load the state of the Goals Engine."""
        # Load active goals
        self.active_goals = {}
        for goal_id, goal_dict in state.get('active_goals', {}).items():
            self.active_goals[goal_id] = Goal.from_dict(goal_dict)
        
        # Load goal history
        self.goal_history = [Goal.from_dict(goal_dict) for goal_dict in state.get('goal_history', [])]
        
        # Update metrics
        self._update_system_metrics()
        
        self.logger.info(f"Loaded Goals Engine state with {len(self.active_goals)} active goals") 