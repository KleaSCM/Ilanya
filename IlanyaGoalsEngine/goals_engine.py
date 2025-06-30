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
        Update all active goals and apply system-wide effects.
        
        This method orchestrates the complete goal update cycle including
        progress updates, buffing, resolution checks, and stability controls.
        
        Args:
            time_delta: Time elapsed since last update
            
        Returns:
            Dictionary containing update results and system metrics
        """
        self.logger.debug(f"Updating {len(self.active_goals)} active goals")
        
        # Step 1: Update goal progress and decay buffs
        for goal in self.active_goals.values():
            goal.decay_buffs(time_delta)
            # Simulate progress increase (in real system, this would come from actual work)
            if goal.state == GoalState.ACTIVE:
                progress_increase = 0.002 * time_delta.total_seconds()  # Small progress per second
                new_progress = min(1.0, goal.progress + progress_increase)
                goal.update_progress(new_progress)
        
        # Step 2: Apply goal buffs to traits and desires
        buffing_results = self._apply_goal_buffs()
        
        # Step 3: Check for goal resolution (completion, failure, etc.)
        resolution_results = self._check_goal_resolution()
        
        # Step 4: Apply stability controls
        stability_results = self._apply_stability_controls()
        
        # Step 5: Resolve resource conflicts
        conflict_results = self._resolve_resource_conflicts(list(self.active_goals.values()))
        
        # Step 6: Prune goals that should be removed
        pruning_results = self._prune_goals()
        
        # Step 7: Update system metrics
        self._update_system_metrics()
        
        # Combine all results
        update_results = {
            'active_goals': len(self.active_goals),
            'total_goals': len(self.goal_history),
            'buffing_results': buffing_results,
            'resolution_results': resolution_results,
            'stability_results': stability_results,
            'conflict_results': conflict_results,
            'pruning_results': pruning_results,
            'system_metrics': self._get_system_metrics()
        }
        
        return update_results
    
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
        
        This implements an enhanced Nash equilibrium calculation with gradient-based
        optimization and improved convergence criteria for goal competition.
        """
        if len(competing_goals) < 2:
            return [1.0] * len(competing_goals)
        
        # Initialize weights with equal distribution
        weights = [1.0 / len(competing_goals)] * len(competing_goals)
        
        # Enhanced Nash equilibrium calculation with multiple convergence methods
        convergence_achieved = False
        iteration_count = 0
        
        # Method 1: Iterative best response with momentum
        momentum = 0.9
        velocity = [0.0] * len(competing_goals)
        
        for iteration in range(self.config.nash_iteration_limit):
            old_weights = weights.copy()
            
            # Calculate gradients for each goal
            gradients = self._calculate_nash_gradients(competing_goals, weights)
            
            # Update weights using gradient descent with momentum
            for i in range(len(competing_goals)):
                # Calculate new weight based on gradient
                gradient = gradients[i]
                velocity[i] = momentum * velocity[i] + self.config.nash_learning_rate * gradient
                weights[i] = max(0.0, weights[i] + velocity[i])
            
            # Normalize weights to ensure they sum to 1
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            
            # Check convergence using multiple criteria
            convergence_achieved = self._check_nash_convergence(weights, old_weights, iteration)
            
            if convergence_achieved:
                self.logger.debug(f"Nash equilibrium converged after {iteration + 1} iterations")
                break
            
            iteration_count = iteration
        
        # Method 2: If iterative method doesn't converge, use analytical solution
        if not convergence_achieved:
            self.logger.debug("Iterative Nash equilibrium did not converge, using analytical solution")
            weights = self._calculate_analytical_nash_equilibrium(competing_goals)
        
        # Apply stability constraints
        weights = self._apply_nash_stability_constraints(weights, competing_goals)
        
        return weights
    
    def _calculate_nash_gradients(self, competing_goals: List[Goal], weights: List[float]) -> List[float]:
        """
        Calculate gradients for Nash equilibrium optimization.
        
        Args:
            competing_goals: List of competing goals
            weights: Current weight distribution
            
        Returns:
            List of gradients for each goal
        """
        gradients = []
        
        for i, goal in enumerate(competing_goals):
            # Base gradient from goal strength
            strength_gradient = goal.current_strength
            
            # Interaction gradient (negative for competing goals)
            interaction_gradient = 0.0
            for j, other_goal in enumerate(competing_goals):
                if i != j and other_goal.id in goal.competing_goals:
                    # Calculate interaction strength based on goal properties
                    interaction_strength = goal.interaction_strength * other_goal.interaction_strength
                    interaction_gradient -= interaction_strength * weights[j]
            
            # Progress-based gradient (goals with higher progress get higher weights)
            progress_gradient = goal.progress * 0.5
            
            # Confidence-based gradient
            confidence_gradient = goal.confidence * 0.3
            
            # Combined gradient
            total_gradient = strength_gradient + interaction_gradient + progress_gradient + confidence_gradient
            
            # Apply gradient clipping to prevent instability
            total_gradient = max(-1.0, min(1.0, total_gradient))
            
            gradients.append(total_gradient)
        
        return gradients
    
    def _check_nash_convergence(self, weights: List[float], old_weights: List[float], iteration: int) -> bool:
        """
        Check if Nash equilibrium has converged using multiple criteria.
        
        Args:
            weights: Current weights
            old_weights: Previous weights
            iteration: Current iteration number
            
        Returns:
            True if convergence criteria are met
        """
        # Criterion 1: Weight change threshold
        weight_change = sum(abs(w1 - w2) for w1, w2 in zip(weights, old_weights))
        if weight_change < self.config.nash_convergence_threshold:
            return True
        
        # Criterion 2: Relative change threshold
        if iteration > 0:
            relative_change = weight_change / (sum(old_weights) + 1e-8)
            if relative_change < self.config.nash_convergence_threshold * 0.1:
                return True
        
        # Criterion 3: Maximum iterations reached
        if iteration >= self.config.nash_iteration_limit - 1:
            return True
        
        # Criterion 4: Oscillation detection (if weights are oscillating, consider converged)
        if iteration > 10:
            # Check if weights are oscillating around a stable point
            oscillation_threshold = 0.01
            if weight_change < oscillation_threshold:
                return True
        
        return False
    
    def _calculate_analytical_nash_equilibrium(self, competing_goals: List[Goal]) -> List[float]:
        """
        Calculate analytical Nash equilibrium for simple cases.
        
        This provides a fallback solution when iterative methods don't converge.
        """
        # Calculate utility matrix
        utility_matrix = []
        for goal in competing_goals:
            utilities = []
            for other_goal in competing_goals:
                if goal.id == other_goal.id:
                    # Self-utility based on goal strength and progress
                    utility = goal.current_strength * (1.0 + goal.progress)
                else:
                    # Cross-utility based on interaction
                    if other_goal.id in goal.competing_goals:
                        utility = -goal.interaction_strength * other_goal.interaction_strength
                    else:
                        utility = 0.0
                utilities.append(utility)
            utility_matrix.append(utilities)
        
        # Solve using simplified analytical approach
        # For 2x2 games, use explicit solution
        if len(competing_goals) == 2:
            return self._solve_2x2_nash_equilibrium(utility_matrix)
        
        # For larger games, use iterative elimination of dominated strategies
        return self._solve_nash_by_elimination(utility_matrix)
    
    def _solve_2x2_nash_equilibrium(self, utility_matrix: List[List[float]]) -> List[float]:
        """
        Solve 2x2 Nash equilibrium analytically.
        
        Args:
            utility_matrix: 2x2 utility matrix
            
        Returns:
            Nash equilibrium weights
        """
        # Extract utilities
        a11, a12 = utility_matrix[0][0], utility_matrix[0][1]
        a21, a22 = utility_matrix[1][0], utility_matrix[1][1]
        
        # Calculate mixed strategy Nash equilibrium
        denominator = (a11 + a22) - (a12 + a21)
        
        if abs(denominator) < 1e-8:  # Avoid division by zero
            return [0.5, 0.5]
        
        p1 = (a22 - a21) / denominator
        p1 = max(0.0, min(1.0, p1))  # Clamp to [0, 1]
        p2 = 1.0 - p1
        
        return [p1, p2]
    
    def _solve_nash_by_elimination(self, utility_matrix: List[List[float]]) -> List[float]:
        """
        Solve Nash equilibrium by iterative elimination of dominated strategies.
        
        Args:
            utility_matrix: NxN utility matrix
            
        Returns:
            Nash equilibrium weights
        """
        n = len(utility_matrix)
        
        # Initialize with equal weights
        weights = [1.0 / n] * n
        
        # Iterative elimination
        for iteration in range(10):
            # Find dominated strategies
            dominated = [False] * n
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        # Check if strategy j dominates strategy i
                        if all(utility_matrix[j][k] >= utility_matrix[i][k] for k in range(n)):
                            dominated[i] = True
                            break
            
            # Update weights (reduce weight of dominated strategies)
            total_weight = 0.0
            for i in range(n):
                if dominated[i]:
                    weights[i] *= 0.5  # Reduce weight of dominated strategies
                total_weight += weights[i]
            
            # Normalize
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
        
        return weights
    
    def _apply_nash_stability_constraints(self, weights: List[float], competing_goals: List[Goal]) -> List[float]:
        """
        Apply stability constraints to Nash equilibrium weights.
        
        Args:
            weights: Nash equilibrium weights
            competing_goals: List of competing goals
            
        Returns:
            Constrained weights
        """
        # Ensure minimum weight for each goal
        min_weight = 0.05
        constrained_weights = []
        
        for i, weight in enumerate(weights):
            # Apply minimum weight constraint
            constrained_weight = max(min_weight, weight)
            
            # Apply maximum weight constraint based on goal strength
            max_weight = min(0.8, competing_goals[i].current_strength)
            constrained_weight = min(max_weight, constrained_weight)
            
            constrained_weights.append(constrained_weight)
        
        # Renormalize
        total_weight = sum(constrained_weights)
        if total_weight > 0:
            constrained_weights = [w / total_weight for w in constrained_weights]
        
        return constrained_weights
    
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
        
        return max(0.0, min(1.0, float(stability)))
    
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
    
    def _resolve_resource_conflicts(self, goals: List[Goal]) -> Dict[str, Any]:
        """
        Resolve resource conflicts between goals by adjusting resource allocations.
        
        This implements intelligent resource conflict resolution that considers
        goal priorities, strengths, and dependencies to optimize resource allocation.
        
        Args:
            goals: List of active goals
            
        Returns:
            Dictionary containing resolution results and adjustments
        """
        if len(goals) < 2:
            return {'resolved_conflicts': [], 'adjustments': {}}
        
        resolved_conflicts = []
        adjustments = {}
        
        # Find all resource conflicts between goals
        for i, goal1 in enumerate(goals):
            for j, goal2 in enumerate(goals[i+1:], i+1):
                conflicts = goal1.get_resource_conflicts(goal2)
                
                for conflict_type in conflicts:
                    # Calculate current resource usage
                    req1 = goal1.get_total_resource_requirement(conflict_type)
                    req2 = goal2.get_total_resource_requirement(conflict_type)
                    total_required = req1 + req2
                    
                    if total_required > 1.0:
                        # Resolve conflict using multiple strategies
                        resolution = self._resolve_single_resource_conflict(
                            goal1, goal2, conflict_type, req1, req2
                        )
                        
                        if resolution:
                            resolved_conflicts.append({
                                'goal1_id': goal1.id,
                                'goal2_id': goal2.id,
                                'resource_type': conflict_type,
                                'original_usage': total_required,
                                'resolution': resolution
                            })
                            
                            # Apply adjustments
                            if goal1.id not in adjustments:
                                adjustments[goal1.id] = {}
                            if goal2.id not in adjustments:
                                adjustments[goal2.id] = {}
                            
                            adjustments[goal1.id][conflict_type] = resolution['goal1_adjustment']
                            adjustments[goal2.id][conflict_type] = resolution['goal2_adjustment']
        
        return {
            'resolved_conflicts': resolved_conflicts,
            'adjustments': adjustments
        }
    
    def _resolve_single_resource_conflict(self, goal1: Goal, goal2: Goal, 
                                        resource_type: str, req1: float, req2: float) -> Optional[Dict]:
        """
        Resolve a single resource conflict between two goals.
        
        Args:
            goal1: First goal
            goal2: Second goal
            resource_type: Type of resource in conflict
            req1: Current requirement of goal1
            req2: Current requirement of goal2
            
        Returns:
            Resolution strategy with adjustments for both goals
        """
        # Strategy 1: Priority-based allocation
        priority1 = self._calculate_goal_priority(goal1, resource_type)
        priority2 = self._calculate_goal_priority(goal2, resource_type)
        
        # Strategy 2: Strength-based allocation
        strength1 = goal1.current_strength
        strength2 = goal2.current_strength
        
        # Strategy 3: Progress-based allocation (favor goals closer to completion)
        progress1 = goal1.progress
        progress2 = goal2.progress
        
        # Strategy 4: Dependency-based allocation (favor prerequisite goals)
        dependency1 = self._calculate_dependency_priority(goal1)
        dependency2 = self._calculate_dependency_priority(goal2)
        
        # Calculate combined priority scores
        score1 = (priority1 * 0.4 + strength1 * 0.3 + progress1 * 0.2 + dependency1 * 0.1)
        score2 = (priority2 * 0.4 + strength2 * 0.3 + progress2 * 0.2 + dependency2 * 0.1)
        
        # Normalize scores
        total_score = score1 + score2
        if total_score == 0:
            # Fallback to equal allocation
            score1 = score2 = 0.5
        else:
            score1 /= total_score
            score2 /= total_score
        
        # Calculate new allocations
        available_resource = 1.0
        new_req1 = min(req1, available_resource * score1)
        new_req2 = min(req2, available_resource * score2)
        
        # Ensure we don't exceed available resources
        if new_req1 + new_req2 > available_resource:
            # Scale down proportionally
            scale_factor = available_resource / (new_req1 + new_req2)
            new_req1 *= scale_factor
            new_req2 *= scale_factor
        
        return {
            'goal1_adjustment': new_req1,
            'goal2_adjustment': new_req2,
            'resolution_method': 'priority_strength_progress_dependency',
            'goal1_score': score1,
            'goal2_score': score2
        }
    
    def _calculate_goal_priority(self, goal: Goal, resource_type: str) -> float:
        """Calculate priority for a goal based on its resource requirements."""
        for req in goal.resource_requirements:
            if req.resource_type == resource_type:
                return req.priority
        return 0.5  # Default priority
    
    def _calculate_dependency_priority(self, goal: Goal) -> float:
        """Calculate dependency priority (prerequisite goals get higher priority)."""
        if not goal.dependents:
            return 1.0  # No dependents = high priority (can be completed)
        
        # Goals with more dependents get higher priority
        dependent_count = len(goal.dependents)
        return min(1.0, 0.5 + (dependent_count * 0.1)) 