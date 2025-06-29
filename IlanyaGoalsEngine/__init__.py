"""
Ilanya Goals Engine

Advanced goal formation, monitoring, and resolution system for AI agents.
Implements field-like attraction dynamics, goal buffing mechanisms, and
mathematical stability controls for emergent goal-directed behavior.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

from .goals_engine import GoalsEngine
from .goal_formation_interface import GoalFormationInterface
from .goal_monitor import GoalMonitor
from .goal_resolver import GoalResolver
from .models import Goal, GoalState, GoalType
from .config import GoalsEngineConfig

__version__ = "0.1.0"
__author__ = "KleaSCM"
__email__ = "KleaSCM@gmail.com"

__all__ = [
    "GoalsEngine",
    "GoalFormationInterface", 
    "GoalMonitor",
    "GoalResolver",
    "Goal",
    "GoalState", 
    "GoalType",
    "GoalsEngineConfig"
] 