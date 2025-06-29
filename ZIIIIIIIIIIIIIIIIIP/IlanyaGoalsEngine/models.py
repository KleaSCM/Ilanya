
# Ilanya Goals Engine - Models

# Data models for goals, goal states, and goal types.
# Implements the mathematical representation of goals and their properties.

# Author: KleaSCM
# Email: KleaSCM@gmail.com
# License: MIT
# Version: 0.1.0


from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Optional, Any
import uuid


class GoalState(Enum):
    """Enumeration of possible goal states."""
    FORMING = "forming"  # Desire is approaching goal formation threshold
    ACTIVE = "active"    # Goal is active and being pursued
    PAUSED = "paused"    # Goal is temporarily paused
    COMPLETED = "completed"  # Goal has been successfully completed
    FAILED = "failed"    # Goal has failed or been abandoned
    PRUNED = "pruned"    # Goal has been removed due to lack of reinforcement


class GoalType(Enum):
    """Enumeration of goal types based on their nature."""
    LEARNING = "learning"      # Goals focused on acquiring knowledge
    SOCIAL = "social"          # Goals related to social interactions
    CREATIVE = "creative"      # Goals involving creative expression
    PROBLEM_SOLVING = "problem_solving"  # Goals focused on solving problems
    SELF_IMPROVEMENT = "self_improvement"  # Goals for personal growth
    EXPLORATION = "exploration"  # Goals for discovering new things
    MAINTENANCE = "maintenance"  # Goals for maintaining current state
    EMERGENT = "emergent"      # Goals that emerged from interactions


@dataclass
class Goal:
    """
    Represents a goal in the Goals Engine.
    
    A goal is formed when a desire reaches maximum strength and maintains
    that strength over a time threshold. Goals create feedback loops that
    buff the original traits and desires that formed them.
    """
    
    # Core Properties
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    goal_type: GoalType = GoalType.EMERGENT
    
    # Formation Properties
    source_desires: List[str] = field(default_factory=list)  # IDs of desires that formed this goal
    source_traits: List[str] = field(default_factory=list)   # Traits that contributed to goal formation
    formation_strength: float = 0.0  # Strength of desire when goal was formed
    formation_time: datetime = field(default_factory=datetime.now)
    
    # State Properties
    state: GoalState = GoalState.FORMING
    current_strength: float = 0.0  # Current goal strength
    progress: float = 0.0  # Progress toward completion (0.0 to 1.0)
    confidence: float = 0.0  # Confidence in goal completion
    
    # Buffing Properties
    trait_buffs: Dict[str, float] = field(default_factory=dict)  # Trait ID -> buff strength
    desire_buffs: Dict[str, float] = field(default_factory=dict)  # Desire ID -> buff strength
    buff_decay_rate: float = 0.1  # Rate at which buffs decay
    
    # Temporal Properties
    last_reinforcement: Optional[datetime] = None
    last_progress_update: datetime = field(default_factory=datetime.now)
    estimated_completion_time: Optional[datetime] = None
    time_to_live: Optional[timedelta] = None  # Time before goal is pruned if not reinforced
    
    # Interaction Properties
    interaction_strength: float = 0.0  # Strength of interactions with other goals
    competing_goals: List[str] = field(default_factory=list)  # IDs of competing goals
    supporting_goals: List[str] = field(default_factory=list)  # IDs of supporting goals
    
    # Mathematical Properties
    lyapunov_stability: float = 1.0  # Stability measure for this goal
    entropy_contribution: float = 0.0  # Contribution to system entropy
    nash_equilibrium_weight: float = 0.0  # Weight in Nash equilibrium calculations
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Initialize computed properties after object creation."""
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.formation_time:
            self.formation_time = datetime.now()
        if not self.last_progress_update:
            self.last_progress_update = datetime.now()
        if not self.created_at:
            self.created_at = datetime.now()
        if not self.updated_at:
            self.updated_at = datetime.now()
    
    def update_progress(self, new_progress: float, confidence: Optional[float] = None):
        """Update goal progress and related properties."""
        self.progress = max(0.0, min(1.0, new_progress))
        if confidence is not None:
            self.confidence = max(0.0, min(1.0, confidence))
        self.last_progress_update = datetime.now()
        self.updated_at = datetime.now()
        
        # Update state based on progress
        if self.progress >= 0.9:
            self.state = GoalState.COMPLETED
        elif self.progress >= 0.5:
            self.state = GoalState.ACTIVE
        elif self.progress > 0.0:
            self.state = GoalState.FORMING
    
    def apply_trait_buff(self, trait_id: str, buff_strength: float):
        """Apply a buff to a trait that contributed to this goal."""
        self.trait_buffs[trait_id] = buff_strength
        self.updated_at = datetime.now()
    
    def apply_desire_buff(self, desire_id: str, buff_strength: float):
        """Apply a buff to a desire that contributed to this goal."""
        self.desire_buffs[desire_id] = buff_strength
        self.updated_at = datetime.now()
    
    def decay_buffs(self, time_delta: timedelta):
        """Decay all buffs based on time passed."""
        decay_factor = 1.0 - (self.buff_decay_rate * time_delta.total_seconds() / 3600.0)
        decay_factor = max(0.0, decay_factor)
        
        # Decay trait buffs
        for trait_id in self.trait_buffs:
            self.trait_buffs[trait_id] *= decay_factor
        
        # Decay desire buffs
        for desire_id in self.desire_buffs:
            self.desire_buffs[desire_id] *= decay_factor
        
        self.updated_at = datetime.now()
    
    def is_reinforced(self, time_threshold: timedelta) -> bool:
        """Check if goal has been reinforced within the time threshold."""
        if self.last_reinforcement is None:
            return False
        return (datetime.now() - self.last_reinforcement) < time_threshold
    
    def should_be_pruned(self, pruning_threshold: timedelta) -> bool:
        """Check if goal should be pruned due to lack of reinforcement."""
        if self.state in [GoalState.COMPLETED, GoalState.FAILED]:
            return True
        if self.time_to_live and (datetime.now() - self.created_at) > self.time_to_live:
            return True
        return not self.is_reinforced(pruning_threshold)
    
    def get_total_buff_strength(self) -> float:
        """Get the total strength of all active buffs."""
        trait_buff_total = sum(self.trait_buffs.values())
        desire_buff_total = sum(self.desire_buffs.values())
        return trait_buff_total + desire_buff_total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert goal to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'goal_type': self.goal_type.value,
            'source_desires': self.source_desires,
            'source_traits': self.source_traits,
            'formation_strength': self.formation_strength,
            'formation_time': self.formation_time.isoformat(),
            'state': self.state.value,
            'current_strength': self.current_strength,
            'progress': self.progress,
            'confidence': self.confidence,
            'trait_buffs': self.trait_buffs,
            'desire_buffs': self.desire_buffs,
            'buff_decay_rate': self.buff_decay_rate,
            'last_reinforcement': self.last_reinforcement.isoformat() if self.last_reinforcement else None,
            'last_progress_update': self.last_progress_update.isoformat(),
            'estimated_completion_time': self.estimated_completion_time.isoformat() if self.estimated_completion_time else None,
            'time_to_live_seconds': self.time_to_live.total_seconds() if self.time_to_live else None,
            'interaction_strength': self.interaction_strength,
            'competing_goals': self.competing_goals,
            'supporting_goals': self.supporting_goals,
            'lyapunov_stability': self.lyapunov_stability,
            'entropy_contribution': self.entropy_contribution,
            'nash_equilibrium_weight': self.nash_equilibrium_weight,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, goal_dict: Dict[str, Any]) -> 'Goal':
        """Create goal from dictionary."""
        # Convert string values back to enums
        goal_dict['goal_type'] = GoalType(goal_dict['goal_type'])
        goal_dict['state'] = GoalState(goal_dict['state'])
        
        # Convert ISO strings back to datetime
        goal_dict['formation_time'] = datetime.fromisoformat(goal_dict['formation_time'])
        goal_dict['last_progress_update'] = datetime.fromisoformat(goal_dict['last_progress_update'])
        goal_dict['created_at'] = datetime.fromisoformat(goal_dict['created_at'])
        goal_dict['updated_at'] = datetime.fromisoformat(goal_dict['updated_at'])
        
        if goal_dict.get('last_reinforcement'):
            goal_dict['last_reinforcement'] = datetime.fromisoformat(goal_dict['last_reinforcement'])
        
        if goal_dict.get('estimated_completion_time'):
            goal_dict['estimated_completion_time'] = datetime.fromisoformat(goal_dict['estimated_completion_time'])
        
        if goal_dict.get('time_to_live_seconds'):
            goal_dict['time_to_live'] = timedelta(seconds=goal_dict['time_to_live_seconds'])
            goal_dict.pop('time_to_live_seconds')
        
        return cls(**goal_dict) 