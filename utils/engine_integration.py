"""
Ilanya Engine Integration Module

Bidirectional connection system between Desire Engine and Trait Engine.
Maintains full modularity while enabling:
- Traits → Desires: Trait activations create/reinforce desires
- Desires → Traits: Desire reinforcement strengthens source traits
- Modular Design: Clean separation of concerns

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "IlanyaDesireEngine"))
sys.path.append(str(project_root / "IlanyaTraitEngine" / "src"))

from utils.logging_utils import setup_logger


@dataclass
class IntegrationConfig:
    """Configuration for engine integration."""
    
    # Integration parameters
    reinforcement_strength: float = 0.1  # How much desires reinforce traits
    trait_activation_threshold: float = 0.05  # Minimum trait change to create desire
    desire_reinforcement_threshold: float = 0.3  # Minimum desire strength to reinforce traits
    
    # Trait protection
    protected_traits: List[str] = field(default_factory=lambda: [
        "sexual_orientation", "gender_identity", "core_values"
    ])
    
    # Logging
    log_integration_events: bool = True
    log_trait_reinforcement: bool = True
    log_desire_creation: bool = True
    
    # Performance
    batch_size: int = 10
    max_desires_per_trait: int = 3


class EngineIntegration:
    """
    Bidirectional integration between Desire Engine and Trait Engine.
    
    Maintains full modularity while enabling:
    1. Traits → Desires: Trait activations create/reinforce desires
    2. Desires → Traits: Desire reinforcement strengthens source traits
    3. Modular Design: Clean separation of concerns
    """
    
    def __init__(self, config: Optional[IntegrationConfig] = None):
        """Initialize the integration system."""
        self.config = config or IntegrationConfig()
        
        # Set up logging
        self.logger = setup_logger(
            engine_type="integration",
            test_type="system",
            test_name="engine_integration",
            test_target="bidirectional_connection",
            log_level="DEBUG"
        )
        
        # Engine references (will be set by connect_engines)
        self.desire_engine = None
        self.trait_engine = None
        
        # Integration state
        self.trait_desire_mapping: Dict[str, List[str]] = {}
        self.desire_trait_mapping: Dict[str, List[str]] = {}
        self.reinforcement_history: List[Dict[str, Any]] = []
        
        # Statistics
        self.stats = {
            'desires_created': 0,
            'desires_reinforced': 0,
            'traits_reinforced': 0,
            'integration_cycles': 0
        }
        
        self.logger.info("Engine Integration initialized")
    
    def connect_engines(self, desire_engine, trait_engine):
        """
        Connect the two engines for bidirectional communication.
        
        Args:
            desire_engine: Instance of DesireEngine
            trait_engine: Instance of TraitEngine
        """
        self.desire_engine = desire_engine
        self.trait_engine = trait_engine
        
        self.logger.info("Engines connected for bidirectional integration")
        
        # Initialize mappings from existing state
        if hasattr(desire_engine, 'trait_desire_mapping'):
            self.trait_desire_mapping = desire_engine.trait_desire_mapping.copy()
        
        # Build reverse mapping
        self._build_desire_trait_mapping()
    
    def process_trait_activations(self, trait_data) -> Dict[str, Any]:
        """
        Process trait activations and create/reinforce desires.
        
        Args:
            trait_data: TraitData from trait engine
            
        Returns:
            Dictionary containing processing results
        """
        if not self.desire_engine:
            raise RuntimeError("Desire engine not connected")
        
        self.logger.info(f"Processing {len(trait_data.trait_matrix.traits)} trait activations")
        
        # Convert trait data to format expected by desire engine
        trait_states = self._convert_trait_data_to_states(trait_data)
        
        # Process through desire engine
        results = self.desire_engine.process_trait_activations(trait_states)
        
        # Update mappings
        self._update_trait_desire_mappings()
        
        # Log statistics
        self.stats['integration_cycles'] += 1
        if self.config.log_integration_events:
            self.logger.info(f"Integration cycle {self.stats['integration_cycles']} completed")
        
        return results
    
    def process_desire_reinforcement(self, desire_id: str, reinforcement_strength: float) -> Dict[str, Any]:
        """
        Process desire reinforcement and strengthen source traits.
        
        Args:
            desire_id: ID of the reinforced desire
            reinforcement_strength: Strength of the reinforcement
            
        Returns:
            Dictionary containing reinforcement results
        """
        if not self.trait_engine or not self.desire_engine:
            raise RuntimeError("Engines not connected")
        
        if reinforcement_strength < self.config.desire_reinforcement_threshold:
            return {'reinforced_traits': [], 'strength': 0.0}
        
        # Get desire and its source traits
        if desire_id not in self.desire_engine.desires:
            self.logger.warning(f"Desire {desire_id} not found")
            return {'reinforced_traits': [], 'strength': 0.0}
        
        desire = self.desire_engine.desires[desire_id]
        source_traits = desire.source_traits
        
        if not source_traits:
            self.logger.warning(f"Desire {desire_id} has no source traits")
            return {'reinforced_traits': [], 'strength': 0.0}
        
        # Calculate reinforcement for each source trait
        reinforced_traits = []
        total_reinforcement = 0.0
        
        for trait_name in source_traits:
            # Skip protected traits
            if trait_name in self.config.protected_traits:
                self.logger.debug(f"Skipping protected trait: {trait_name}")
                continue
            
            # Calculate trait reinforcement
            trait_reinforcement = self._calculate_trait_reinforcement(
                trait_name, reinforcement_strength, desire
            )
            
            if trait_reinforcement > 0:
                # Apply reinforcement to trait
                success = self._apply_trait_reinforcement(trait_name, trait_reinforcement)
                if success:
                    reinforced_traits.append(trait_name)
                    total_reinforcement += trait_reinforcement
                    
                    if self.config.log_trait_reinforcement:
                        self.logger.info(f"Reinforced trait {trait_name} by {trait_reinforcement:.3f}")
        
        # Update statistics
        self.stats['traits_reinforced'] += len(reinforced_traits)
        
        # Record reinforcement history
        self.reinforcement_history.append({
            'timestamp': datetime.now().isoformat(),
            'desire_id': desire_id,
            'reinforcement_strength': reinforcement_strength,
            'reinforced_traits': reinforced_traits,
            'total_reinforcement': total_reinforcement
        })
        
        return {
            'reinforced_traits': reinforced_traits,
            'strength': total_reinforcement,
            'desire_id': desire_id
        }
    
    def get_integration_summary(self) -> Dict[str, Any]:
        """Get comprehensive integration summary."""
        return {
            'stats': self.stats.copy(),
            'trait_desire_mapping': self.trait_desire_mapping,
            'desire_trait_mapping': self.desire_trait_mapping,
            'reinforcement_history': self.reinforcement_history[-10:],  # Last 10
            'config': {
                'reinforcement_strength': self.config.reinforcement_strength,
                'protected_traits': self.config.protected_traits,
                'trait_activation_threshold': self.config.trait_activation_threshold,
                'desire_reinforcement_threshold': self.config.desire_reinforcement_threshold
            }
        }
    
    def _convert_trait_data_to_states(self, trait_data) -> Dict[str, Any]:
        """Convert TraitData to format expected by desire engine."""
        trait_states = {}
        
        for trait_type, trait_vector in trait_data.trait_matrix.traits.items():
            trait_states[str(trait_type.value)] = {
                'current_value': float(trait_vector.value),
                'confidence': float(trait_vector.confidence),
                'change_rate': 0.0,  # Will be calculated if we have previous state
                'trait_type': str(trait_type.value)
            }
        
        return trait_states
    
    def _build_desire_trait_mapping(self):
        """Build reverse mapping from desires to traits."""
        self.desire_trait_mapping.clear()
        
        for trait_name, desire_ids in self.trait_desire_mapping.items():
            for desire_id in desire_ids:
                if desire_id not in self.desire_trait_mapping:
                    self.desire_trait_mapping[desire_id] = []
                self.desire_trait_mapping[desire_id].append(trait_name)
    
    def _update_trait_desire_mappings(self):
        """Update mappings after desire engine processing."""
        if self.desire_engine and hasattr(self.desire_engine, 'trait_desire_mapping'):
            self.trait_desire_mapping = self.desire_engine.trait_desire_mapping.copy()
            self._build_desire_trait_mapping()
    
    def _calculate_trait_reinforcement(self, trait_name: str, reinforcement_strength: float, desire) -> float:
        """Calculate how much to reinforce a trait based on desire reinforcement."""
        # Base reinforcement
        base_reinforcement = reinforcement_strength * self.config.reinforcement_strength
        
        # Scale by desire properties
        desire_strength_factor = desire.strength
        reinforcement_count_factor = min(desire.reinforcement_count / 10.0, 1.0)  # Cap at 10 reinforcements
        
        # Calculate final reinforcement
        final_reinforcement = base_reinforcement * desire_strength_factor * (1 + reinforcement_count_factor)
        
        return min(final_reinforcement, 0.2)  # Cap at 0.2 to prevent excessive changes
    
    def _apply_trait_reinforcement(self, trait_name: str, reinforcement_strength: float) -> bool:
        """Apply reinforcement to a trait in the trait engine."""
        try:
            # This would need to be implemented based on the trait engine's interface
            # For now, we'll log the reinforcement
            self.logger.debug(f"Would reinforce trait {trait_name} by {reinforcement_strength:.3f}")
            
            # TODO: Implement actual trait reinforcement
            # This could involve:
            # 1. Getting current trait data from trait engine
            # 2. Modifying trait values
            # 3. Updating trait engine state
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reinforce trait {trait_name}: {str(e)}")
            return False
    
    def save_integration_state(self, filepath: str):
        """Save integration state to file."""
        state = {
            'trait_desire_mapping': self.trait_desire_mapping,
            'desire_trait_mapping': self.desire_trait_mapping,
            'reinforcement_history': self.reinforcement_history,
            'stats': self.stats,
            'config': {
                'reinforcement_strength': self.config.reinforcement_strength,
                'protected_traits': self.config.protected_traits,
                'trait_activation_threshold': self.config.trait_activation_threshold,
                'desire_reinforcement_threshold': self.config.desire_reinforcement_threshold
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"Integration state saved to {filepath}")
    
    def load_integration_state(self, filepath: str):
        """Load integration state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.trait_desire_mapping = state.get('trait_desire_mapping', {})
        self.desire_trait_mapping = state.get('desire_trait_mapping', {})
        self.reinforcement_history = state.get('reinforcement_history', [])
        self.stats = state.get('stats', self.stats)
        
        # Update config if provided
        if 'config' in state:
            config_data = state['config']
            self.config.reinforcement_strength = config_data.get('reinforcement_strength', 0.1)
            self.config.protected_traits = config_data.get('protected_traits', [])
            self.config.trait_activation_threshold = config_data.get('trait_activation_threshold', 0.05)
            self.config.desire_reinforcement_threshold = config_data.get('desire_reinforcement_threshold', 0.3)
        
        self.logger.info(f"Integration state loaded from {filepath}")


# Convenience function for quick integration setup
def create_integrated_system(desire_engine, trait_engine, config: Optional[IntegrationConfig] = None):
    """
    Create an integrated system connecting desire and trait engines.
    
    Args:
        desire_engine: Instance of DesireEngine
        trait_engine: Instance of TraitEngine
        config: Optional integration configuration
        
    Returns:
        EngineIntegration instance
    """
    integration = EngineIntegration(config)
    integration.connect_engines(desire_engine, trait_engine)
    return integration 