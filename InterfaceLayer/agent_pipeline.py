"""
Ilanya Simple Agent Pipeline

Simple pipeline that follows the flow:
User Input → Trait Mapper → Trait Engine → Desire Engine → Goals Engine → Prompt Builder → LLM

This is the main integration point for the running agent.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Neural Networks")))

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "IlanyaDesireEngine"))
sys.path.append(str(project_root / "IlanyaGoalsEngine"))

# Import the 2 Neural Networks
from IlanyaTraitMapping.trait_mapper import TraitMapper, TraitMappingConfig
from IlanyaTraitEngine.trait_engine import TraitEngine, TraitEngineConfig
from IlanyaTraitEngine.trait_data import TraitData, TraitDataBuilder
from IlanyaTraitEngine.trait_types import TraitType

# Import the 2 Engines
from IlanyaDesireEngine.desire_engine.core import DesireEngine
from IlanyaGoalsEngine.goals_engine import GoalsEngine
from IlanyaGoalsEngine.config import GoalsEngineConfig

@dataclass
class PipelineResponse:
    """Response from the complete pipeline."""
    
    # Input
    user_input: str
    processed: bool
    
    # Step 1: Trait Mapper results
    trait_modifications: Dict[str, float]  # trait_name -> change_amount
    
    # Step 2: Trait Engine results  
    evolved_traits: Dict[str, float]  # trait_name -> current_value
    
    # Step 3: Desire Engine results
    desires_created: List[str]
    desires_activated: List[str]
    desire_strengths: Dict[str, float]
    
    # Step 4: Goals Engine results
    goals_formed: List[str]
    goal_priorities: Dict[str, float]
    
    # Step 5: Final context for LLM
    llm_context: str
    
    # Error handling
    error_message: Optional[str] = None


class SimpleAgentPipeline:
    """
    Simple agent pipeline that processes user input through all engines.
    
    Flow:
    1. User Input → Trait Mapper (maps words to trait modifications)
    2. Trait Mapper → Trait Engine (evolves traits)
    3. Trait Engine → Desire Engine (creates/activates desires)
    4. Desire Engine → Goals Engine (forms goals)
    5. All data → Prompt Builder → LLM
    """
    
    def __init__(self):
        """Initialize all engines."""
        
        # Step 1: Initialize Trait Mapper Neural Network
        print("Initializing Trait Mapper Neural Network...")
        self.trait_mapper = TraitMapper()
        
        # Step 2: Initialize Trait Engine Neural Network
        print("Initializing Trait Engine Neural Network...")
        self.trait_engine = TraitEngine()
        
        # Step 3: Initialize Desire Engine
        print("Initializing Desire Engine...")
        self.desire_engine = DesireEngine()
        
        # Step 4: Initialize Goals Engine
        print("Initializing Goals Engine...")
        goals_config = GoalsEngineConfig()
        self.goals_engine = GoalsEngine(goals_config)
        
        # Current state
        self.current_traits: Optional[TraitData] = None
        self.conversation_history: List[Dict[str, Any]] = []
        
        print("✅ All engines initialized successfully!")
        
    def process_user_input(self, user_input: str) -> PipelineResponse:
        """
        Process user input through the complete pipeline.
        
        This is the main method that follows your exact flow.
        
        Args:
            user_input: User's text input
            
        Returns:
            PipelineResponse with all processing results
        """
        try:
            print(f"\n🔄 Processing: '{user_input}'")
            
            # Step 1: User Input → Trait Mapper
            print("Step 1: Trait Mapper Neural Network...")
            trait_modifications = self._map_input_to_traits(user_input)
            
            # Step 2: Trait Mapper → Trait Engine  
            print("Step 2: Trait Engine Neural Network...")
            evolved_traits = self._process_through_trait_engine(trait_modifications)
            
            # Step 3: Trait Engine → Desire Engine
            print("Step 3: Desire Engine...")
            desire_results = self._process_through_desire_engine(evolved_traits)
            
            # Step 4: Desire Engine → Goals Engine
            print("Step 4: Goals Engine...")
            goal_results = self._process_through_goals_engine(desire_results)
            
            # Step 5: Build context for LLM
            print("Step 5: Building LLM context...")
            llm_context = self._build_llm_context(
                user_input, evolved_traits, desire_results, goal_results
            )
            
            # Update current state
            self.current_traits = evolved_traits
            self.conversation_history.append({
                'input': user_input,
                'trait_modifications': trait_modifications,
                'evolved_traits': evolved_traits,
                'desire_results': desire_results,
                'goal_results': goal_results
            })
            
            print("✅ Pipeline completed successfully!")
            
            return PipelineResponse(
                user_input=user_input,
                processed=True,
                trait_modifications=self._extract_trait_changes(trait_modifications),
                evolved_traits=self._extract_trait_values(evolved_traits),
                desires_created=desire_results.get('new_desires', []),
                desires_activated=desire_results.get('reinforced_desires', []),
                desire_strengths=self._extract_desire_strengths(),
                goals_formed=goal_results.get('goals_formed', []),
                goal_priorities=goal_results.get('goal_priorities', {}),
                llm_context=llm_context
            )
            
        except Exception as e:
            print(f"❌ Pipeline failed: {str(e)}")
            return PipelineResponse(
                user_input=user_input,
                processed=False,
                trait_modifications={},
                evolved_traits={},
                desires_created=[],
                desires_activated=[],
                desire_strengths={},
                goals_formed=[],
                goal_priorities={},
                llm_context="",
                error_message=str(e)
            )
    
    def _map_input_to_traits(self, user_input: str) -> TraitData:
        """Step 1: Map user input to trait modifications."""
        # Use current traits or create default ones
        if self.current_traits is None:
            self.current_traits = self._create_default_traits()
        
        # Map text input to trait modifications
        return self.trait_mapper.map_text_to_traits(user_input, self.current_traits)
    
    def _process_through_trait_engine(self, trait_data: TraitData) -> TraitData:
        """Step 2: Process through trait engine."""
        # Process traits through the neural network
        engine_results = self.trait_engine.process_traits(trait_data)
        
        # Get evolved traits from results
        evolved_traits = engine_results.get('predicted_traits', {})
        
        # Convert back to TraitData format
        builder = TraitDataBuilder()
        for trait_type, trait_vector in evolved_traits.items():
            builder.add_trait(trait_type, trait_vector.value, trait_vector.confidence)
        
        return builder.build()
    
    def _process_through_desire_engine(self, trait_data: TraitData) -> Dict[str, Any]:
        """Step 3: Process through desire engine."""
        # Convert trait data to format expected by desire engine
        trait_states = self._convert_trait_data_to_states(trait_data)
        
        # Process trait activations through desire engine
        return self.desire_engine.process_trait_activations(trait_states)
    
    def _process_through_goals_engine(self, desire_results: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4: Process through goals engine."""
        # Convert desire results to goal formation input
        # Build a dictionary: desire_id -> {strength, confidence, ...}
        desires_dict = {}
        for desire_id in desire_results.get('reinforced_desires', []):
            # Use desire_strengths if available, else default to 1.0
            strength = desire_results.get('desire_strengths', {}).get(desire_id, 1.0)
            # Confidence is not always available; default to 1.0
            confidence = 1.0
            desires_dict[desire_id] = {'strength': strength, 'confidence': confidence}
        
        # Process through goals engine
        new_goals = self.goals_engine.process_desires(desires_dict)
        
        return {
            'goals_formed': [goal.name for goal in new_goals],
            'goal_priorities': {goal.name: goal.current_strength for goal in new_goals}
        }
    
    def _build_llm_context(self, user_input: str, evolved_traits: TraitData, 
                          desire_results: Dict[str, Any], goal_results: Dict[str, Any]) -> str:
        """Step 5: Build context for LLM."""
        context_parts = []
        
        # Add user input
        context_parts.append(f"User Input: {user_input}")
        context_parts.append("")
        
        # Add current personality
        context_parts.append("Current Personality:")
        for trait_type, trait_vector in evolved_traits.trait_matrix.traits.items():
            context_parts.append(f"  - {trait_type.value}: {trait_vector.value:.2f}")
        context_parts.append("")
        
        # Add active desires
        if desire_results.get('reinforced_desires'):
            context_parts.append("Active Desires:")
            for desire_id in desire_results['reinforced_desires']:
                if desire_id in self.desire_engine.desires:
                    desire = self.desire_engine.desires[desire_id]
                    context_parts.append(f"  - {desire.name}: {desire.strength:.2f}")
            context_parts.append("")
        
        # Add current goals
        if goal_results.get('goals_formed'):
            context_parts.append("Current Goals:")
            for goal_name in goal_results['goals_formed']:
                priority = goal_results['goal_priorities'].get(goal_name, 0.0)
                context_parts.append(f"  - {goal_name}: priority {priority:.2f}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _create_default_traits(self) -> TraitData:
        """Create default trait state."""
        builder = TraitDataBuilder()
        
        # Add default traits
        builder.add_trait(TraitType.OPENNESS, 0.5, 0.8)
        builder.add_trait(TraitType.CREATIVITY, 0.5, 0.7)
        builder.add_trait(TraitType.SEXUAL_EXPERIENCE, 0.3, 0.9)
        builder.add_trait(TraitType.EMOTIONAL_STABILITY, 0.6, 0.8)
        builder.add_trait(TraitType.EXTRAVERSION, 0.4, 0.6)
        builder.add_trait(TraitType.AGREEABLENESS, 0.6, 0.7)
        builder.add_trait(TraitType.CONSCIENTIOUSNESS, 0.5, 0.7)
        builder.add_trait(TraitType.OPTIMISM, 0.5, 0.8)
        builder.add_trait(TraitType.LEARNING_RATE, 0.5, 0.7)
        builder.add_trait(TraitType.ATTENTION_SPAN, 0.5, 0.6)
        
        return builder.build()
    
    def _extract_trait_changes(self, trait_data: TraitData) -> Dict[str, float]:
        """Extract trait changes from trait data."""
        changes = {}
        
        for trait_type, trait_vector in trait_data.trait_matrix.traits.items():
            if self.current_traits and trait_type in self.current_traits.trait_matrix.traits:
                old_value = self.current_traits.trait_matrix.traits[trait_type].value
                change = trait_vector.value - old_value
                if abs(change) > 0.01:
                    changes[trait_type.value] = change
        
        return changes
    
    def _extract_trait_values(self, trait_data: TraitData) -> Dict[str, float]:
        """Extract trait values from trait data."""
        return {
            trait_type.value: trait_vector.value 
            for trait_type, trait_vector in trait_data.trait_matrix.traits.items()
        }
    
    def _extract_desire_strengths(self) -> Dict[str, float]:
        """Extract desire strengths from desire engine."""
        return {
            desire_id: desire.strength
            for desire_id, desire in self.desire_engine.desires.items()
        }
    
    def _convert_trait_data_to_states(self, trait_data: TraitData) -> Dict[str, Any]:
        """Convert trait data to format expected by desire engine."""
        trait_states = {}
        
        for trait_type, trait_vector in trait_data.trait_matrix.traits.items():
            trait_states[trait_type.value] = {
                'current_value': trait_vector.value,
                'confidence': trait_vector.confidence,
                'change_rate': 0.0  # Will be calculated if we have previous state
            }
            
            # Calculate change rate if we have previous state
            if self.current_traits and trait_type in self.current_traits.trait_matrix.traits:
                old_value = self.current_traits.trait_matrix.traits[trait_type].value
                trait_states[trait_type.value]['change_rate'] = trait_vector.value - old_value
        
        return trait_states
    
    def _convert_desires_to_goals(self, desire_results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert desire results to goal formation input."""
        return {
            'desires': desire_results.get('reinforced_desires', []),
            'desire_strengths': self._extract_desire_strengths(),
            'context': 'user_input_processing'
        }


# Convenience function for quick processing
def process_user_input(user_input: str) -> PipelineResponse:
    """
    Quick function to process user input through the complete pipeline.
    
    Args:
        user_input: User input text
        
    Returns:
        PipelineResponse with all results
    """
    pipeline = SimpleAgentPipeline()
    return pipeline.process_user_input(user_input)


# Example usage for the running agent
if __name__ == "__main__":
    # This is how the running agent will use it
    pipeline = SimpleAgentPipeline()
    
    # Test the complete flow
    test_inputs = [
        "I'm feeling horny and want to explore sexually",
        "I want to be more creative and social",
        "I need to work on my emotional stability"
    ]
    
    for user_input in test_inputs:
        print(f"\n{'='*60}")
        print(f"PROCESSING: {user_input}")
        print(f"{'='*60}")
        
        response = pipeline.process_user_input(user_input)
        
        if response.processed:
            print("✅ Pipeline completed successfully!")
            print(f"\nTrait Changes: {response.trait_modifications}")
            print(f"Desires Activated: {response.desires_activated}")
            print(f"Goals Formed: {response.goals_formed}")
            print(f"\nLLM Context:\n{response.llm_context}")
        else:
            print(f"❌ Pipeline failed: {response.error_message}")
    
    print(f"\n{'='*60}")
    print("COMPLETE PIPELINE TEST FINISHED")
    print(f"{'='*60}") 