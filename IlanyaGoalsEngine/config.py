"""
Ilanya Goals Engine - Configuration

Configuration parameters for the Goals Engine including mathematical thresholds,
goal formation criteria, buffing mechanisms, and stability controls.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Any, List, Optional


@dataclass
class GoalsEngineConfig:
    """
    Configuration for the Goals Engine.
    
    Contains all mathematical parameters, thresholds, and behavioral settings
    for goal formation, monitoring, and resolution.
    """
    
    # Goal Formation Parameters
    max_strength_threshold: float = 1.0  # Maximum desire strength for goal formation
    time_threshold: timedelta = timedelta(minutes=5)  # Time to maintain max strength
    formation_confidence_threshold: float = 0.8  # Minimum confidence for goal formation
    
    # Goal Buffing Parameters
    trait_buff_multiplier: float = 1.5  # How much goals buff original traits
    desire_buff_multiplier: float = 1.3  # How much goals buff original desires
    buff_decay_rate: float = 0.1  # Rate at which buffs decay over time
    
    # Goal Monitoring Parameters
    progress_check_interval: timedelta = timedelta(minutes=1)  # How often to check progress
    completion_threshold: float = 0.9  # Threshold for considering goal complete
    self_assessment_confidence: float = 0.7  # Confidence in agent's self-assessment
    
    # Stability and Pruning Parameters
    lyapunov_stability_threshold: float = 0.1  # Maximum allowed system instability
    pruning_threshold: timedelta = timedelta(hours=24)  # Time before pruning unreinforced goals
    max_active_goals: int = 10  # Maximum number of simultaneously active goals
    
    # Nash Equilibrium Parameters
    nash_iteration_limit: int = 100  # Maximum iterations for Nash equilibrium
    nash_convergence_threshold: float = 1e-6  # Convergence threshold for Nash
    nash_learning_rate: float = 0.01  # Learning rate for Nash equilibrium updates
    
    # Field Attraction Parameters
    field_attraction_strength: float = 0.8  # Strength of field-like attraction dynamics
    
    # Pareto Frontier Parameters
    objective_weights: Optional[List[float]] = None  # Weights for multi-objective optimization
    
    # Resolution Parameters
    resolution_confidence_threshold: float = 0.8  # Confidence threshold for goal resolution
    user_feedback_weight: float = 0.3  # Weight given to user feedback in resolution
    time_decay_factor: float = 0.1  # Factor for time-based decay in resolution
    
    # Multi-Objective Optimization Parameters
    moo_weight_decay: float = 0.95  # Weight decay for multi-objective optimization
    moo_pareto_threshold: float = 0.1  # Threshold for Pareto optimality
    moo_max_iterations: int = 50  # Maximum iterations for MOO
    
    # Mathematical Stability Parameters
    system_entropy_threshold: float = 2.0  # Maximum allowed system entropy
    goal_interaction_decay: float = 0.2  # Decay rate for goal interactions
    
    # Logging and Debug Parameters
    log_level: str = "INFO"
    enable_debug_mode: bool = False
    save_goal_history: bool = True
    max_history_size: int = 1000
    
    def __post_init__(self):
        """Initialize default values for complex parameters."""
        if self.objective_weights is None:
            # Default weights for: [strength, confidence, stability, field_attraction, resource_efficiency]
            self.objective_weights = [0.3, 0.25, 0.2, 0.15, 0.1]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            'max_strength_threshold': self.max_strength_threshold,
            'time_threshold_seconds': self.time_threshold.total_seconds(),
            'formation_confidence_threshold': self.formation_confidence_threshold,
            'trait_buff_multiplier': self.trait_buff_multiplier,
            'desire_buff_multiplier': self.desire_buff_multiplier,
            'buff_decay_rate': self.buff_decay_rate,
            'progress_check_interval_seconds': self.progress_check_interval.total_seconds(),
            'completion_threshold': self.completion_threshold,
            'self_assessment_confidence': self.self_assessment_confidence,
            'lyapunov_stability_threshold': self.lyapunov_stability_threshold,
            'pruning_threshold_seconds': self.pruning_threshold.total_seconds(),
            'max_active_goals': self.max_active_goals,
            'nash_iteration_limit': self.nash_iteration_limit,
            'nash_convergence_threshold': self.nash_convergence_threshold,
            'nash_learning_rate': self.nash_learning_rate,
            'moo_weight_decay': self.moo_weight_decay,
            'moo_pareto_threshold': self.moo_pareto_threshold,
            'moo_max_iterations': self.moo_max_iterations,
            'resolution_confidence_threshold': self.resolution_confidence_threshold,
            'user_feedback_weight': self.user_feedback_weight,
            'time_decay_factor': self.time_decay_factor,
            'system_entropy_threshold': self.system_entropy_threshold,
            'goal_interaction_decay': self.goal_interaction_decay,
            'field_attraction_strength': self.field_attraction_strength,
            'log_level': self.log_level,
            'enable_debug_mode': self.enable_debug_mode,
            'save_goal_history': self.save_goal_history,
            'max_history_size': self.max_history_size
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GoalsEngineConfig':
        """Create configuration from dictionary."""
        # Convert seconds back to timedelta
        config_dict['time_threshold'] = timedelta(seconds=config_dict.get('time_threshold_seconds', 300))
        config_dict['progress_check_interval'] = timedelta(seconds=config_dict.get('progress_check_interval_seconds', 60))
        config_dict['pruning_threshold'] = timedelta(seconds=config_dict.get('pruning_threshold_seconds', 86400))
        
        # Remove the seconds versions
        config_dict.pop('time_threshold_seconds', None)
        config_dict.pop('progress_check_interval_seconds', None)
        config_dict.pop('pruning_threshold_seconds', None)
        
        return cls(**config_dict) 