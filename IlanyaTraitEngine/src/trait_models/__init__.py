# Trait data models and structures for the Ilanya Trait Engine.

from .trait_data import TraitData, TraitVector, TraitMatrix
from .trait_types import TraitType, TraitCategory, TraitDimension
from .trait_state import TraitState, CognitiveState

__all__ = [
    'TraitData',
    'TraitVector', 
    'TraitMatrix',
    'TraitType',
    'TraitCategory',
    'TraitDimension',
    'TraitState',
    'CognitiveState'
] 