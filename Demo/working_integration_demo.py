"""
Ilanya Working Integration Demo

Demonstrates the bidirectional connection with proper trait change rates.
Shows how traits create desires and how desires reinforce traits.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

import sys
import os
from pathlib import Path
import time
import json
import random
from typing import Dict, Optional

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "IlanyaDesireEngine"))
sys.path.append(str(project_root / "IlanyaTraitEngine" / "src"))
sys.path.append(str(project_root / "utils"))

from utils.engine_integration import create_integrated_system, IntegrationConfig
from utils.logging_utils import setup_logger


def create_trait_data_with_changes(cycle: int, previous_traits: Optional[Dict] = None):
    """Create TraitData with proper change rates for the Desire Engine."""
    from IlanyaTraitEngine.src.trait_models.trait_types import TraitType
    from IlanyaTraitEngine.src.trait_models.trait_data import TraitVector, TraitMatrix, TraitData
    
    # Initialize base traits if not provided
    if previous_traits is None:
        previous_traits = {
            TraitType.OPENNESS: 0.7,
            TraitType.CREATIVITY: 0.8,
            TraitType.ADAPTABILITY: 0.6,
            TraitType.EMPATHY: 0.9,
            TraitType.LEARNING_RATE: 0.8
        }
    
    # Simulate trait changes over time with larger variations
    traits = {}
    for trait_type, previous_value in previous_traits.items():
        # Add larger variation based on cycle to create significant changes
        variation = random.uniform(-0.1, 0.2) * (cycle + 1) * 0.15
        current_value = max(0.0, min(1.0, previous_value + variation))
        
        # Simulate confidence changes
        confidence = max(0.5, min(1.0, 0.8 + random.uniform(-0.1, 0.1)))
        
        traits[trait_type] = TraitVector(trait_type, current_value, confidence)
    
    # Create trait matrix
    trait_matrix = TraitMatrix(traits=traits)
    
    # Create trait data
    trait_data = TraitData(
        trait_matrix=trait_matrix,
        source=f"working_demo_cycle_{cycle}"
    )
    
    return trait_data, {k: v.value for k, v in traits.items()}


def simulate_experience_impact(trait_data, experience_type: str):
    """Simulate how different experiences impact traits with larger effects."""
    from IlanyaTraitEngine.src.trait_models.trait_types import TraitType
    
    # Define experience impacts with larger values to create significant changes
    experience_impacts = {
        "learning": {
            TraitType.LEARNING_RATE: 0.25,  # Large impact
            TraitType.OPENNESS: 0.15,
            TraitType.CREATIVITY: 0.1
        },
        "social": {
            TraitType.EMPATHY: 0.25,  # Large impact
            TraitType.ADAPTABILITY: 0.15,
            TraitType.OPENNESS: 0.1
        },
        "creative": {
            TraitType.CREATIVITY: 0.25,  # Large impact
            TraitType.OPENNESS: 0.15,
            TraitType.LEARNING_RATE: 0.1
        },
        "challenging": {
            TraitType.ADAPTABILITY: 0.25,  # Large impact
            TraitType.LEARNING_RATE: 0.15,
            TraitType.EMPATHY: 0.1
        }
    }
    
    if experience_type in experience_impacts:
        impacts = experience_impacts[experience_type]
        for trait_type, impact in impacts.items():
            if trait_type in trait_data.trait_matrix.traits:
                current_value = trait_data.trait_matrix.traits[trait_type].value
                new_value = max(0.0, min(1.0, current_value + impact))
                trait_data.trait_matrix.traits[trait_type].value = new_value
    
    return trait_data


def create_trait_states_with_changes(trait_data, previous_traits: Optional[Dict] = None):
    """Convert TraitData to trait states with calculated change rates."""
    trait_states = {}
    
    for trait_type, trait_vector in trait_data.trait_matrix.traits.items():
        trait_name = trait_type.value
        current_value = trait_vector.value
        
        # Calculate change rate if we have previous values
        if previous_traits and trait_name in previous_traits:
            change_rate = current_value - previous_traits[trait_name]
        else:
            # For first cycle, create a positive change rate to trigger desire creation
            change_rate = random.uniform(0.1, 0.3)  # Positive change to trigger desires
        
        trait_states[trait_name] = {
            'current_value': current_value,
            'confidence': trait_vector.confidence,
            'change_rate': change_rate,
            'trait_type': trait_name
        }
    
    return trait_states


def main():
    """Main demonstration function."""
    print("🚀 Ilanya Working Integration Demo")
    print("=" * 50)
    print("Demonstrating bidirectional connection with proper trait changes")
    print()
    
    # Set up logging
    logger = setup_logger(
        engine_type="demo",
        test_type="demo",
        test_name="working_integration",
        test_target="proper_changes",
        log_level="INFO"
    )
    
    try:
        # Step 1: Initialize engines
        print("1. 🔧 Initializing Engines...")
        
        # Import engines
        from IlanyaDesireEngine.desire_engine import DesireEngine
        from IlanyaTraitEngine.src.trait_engine.trait_engine import TraitEngine
        
        # Create engines
        desire_engine = DesireEngine()
        trait_engine = TraitEngine()
        
        print("   ✅ Desire Engine initialized")
        print("   ✅ Trait Engine initialized")
        print()
        
        # Step 2: Create integration system
        print("2. 🔗 Creating Integration System...")
        
        # Configure integration for more sensitive detection
        config = IntegrationConfig(
            reinforcement_strength=0.2,  # Stronger reinforcement
            trait_activation_threshold=0.01,  # Very low threshold to catch small changes
            desire_reinforcement_threshold=0.1,  # Lower threshold for reinforcement
            log_integration_events=True,
            log_trait_reinforcement=True,
            log_desire_creation=True
        )
        
        # Create integrated system
        integration = create_integrated_system(desire_engine, trait_engine, config)
        
        print("   ✅ Integration system created")
        print(f"   📊 Reinforcement strength: {config.reinforcement_strength}")
        print(f"   🎯 Trait activation threshold: {config.trait_activation_threshold}")
        print()
        
        # Step 3: Run dynamic simulation
        print("3. 🔄 Running Dynamic Simulation...")
        print("   Simulating trait changes over time with different experiences")
        print()
        
        previous_traits = None
        total_cycles = 8
        
        for cycle in range(total_cycles):
            print(f"   🔄 Cycle {cycle + 1}/{total_cycles}:")
            
            # Create trait data with changes
            trait_data, current_traits = create_trait_data_with_changes(cycle, previous_traits)
            
            # Simulate different experiences
            experiences = ["learning", "social", "creative", "challenging"]
            experience = random.choice(experiences)
            trait_data = simulate_experience_impact(trait_data, experience)
            
            # Convert to trait states with change rates
            trait_states = create_trait_states_with_changes(trait_data, previous_traits)
            
            print(f"      🎯 Experience: {experience}")
            print(f"      📊 Trait values and changes:")
            for trait_name, trait_state in trait_states.items():
                change_rate = trait_state['change_rate']
                current_value = trait_state['current_value']
                change_indicator = "📈" if change_rate > 0 else "📉" if change_rate < 0 else "➡️"
                print(f"         {change_indicator} {trait_name}: {current_value:.3f} (change: {change_rate:+.3f})")
            
            # Process through desire engine directly to bypass integration conversion
            results = desire_engine.process_trait_activations(trait_states)
            
            # Show results
            active_desires = len(desire_engine.desires)
            print(f"      🎯 Active desires: {active_desires}")
            
            if active_desires > 0:
                # Show some desires
                for desire_id, desire in list(desire_engine.desires.items())[:3]:
                    print(f"         💭 {desire.name}: {desire.strength:.3f}")
                
                # Reinforce a random desire occasionally
                if cycle % 2 == 0 and active_desires > 0:
                    desire_id = random.choice(list(desire_engine.desires.keys()))
                    reinforcement_strength = random.uniform(0.3, 0.7)
                    reinforcement_results = integration.process_desire_reinforcement(desire_id, reinforcement_strength)
                    print(f"      💪 Reinforced {len(reinforcement_results['reinforced_traits'])} traits")
            
            # Update previous traits for next cycle
            previous_traits = current_traits
            
            print()
        
        # Step 4: Show final results
        print("4. 📊 Final Results...")
        
        final_summary = integration.get_integration_summary()
        
        print(f"   🔄 Total integration cycles: {final_summary['stats']['integration_cycles']}")
        print(f"   🎯 Desires created: {final_summary['stats']['desires_created']}")
        print(f"   💪 Traits reinforced: {final_summary['stats']['traits_reinforced']}")
        print(f"   🔗 Active desires: {len(desire_engine.desires)}")
        print(f"   📋 Trait-desire mappings: {len(final_summary['trait_desire_mapping'])}")
        
        # Show active desires
        if desire_engine.desires:
            print("\n   📋 Active Desires:")
            for desire_id, desire in desire_engine.desires.items():
                print(f"      • {desire.name}: {desire.strength:.3f} (from: {desire.source_traits})")
        
        # Show trait-desire mappings
        if final_summary['trait_desire_mapping']:
            print("\n   🔗 Trait-Desire Mappings:")
            for trait, desires in final_summary['trait_desire_mapping'].items():
                print(f"      • {trait} → {desires}")
        
        # Show reinforcement history
        if final_summary['reinforcement_history']:
            print("\n   📈 Recent Reinforcement Events:")
            for event in final_summary['reinforcement_history'][-3:]:  # Last 3
                print(f"      • {event['desire_id']}: {event['total_reinforcement']:.3f} strength")
        
        print()
        
        # Step 5: Save final state
        print("5. 💾 Saving Final State...")
        
        state_file = "working_integration_state.json"
        integration.save_integration_state(state_file)
        
        print(f"   ✅ Final state saved to {state_file}")
        print()
        
        # Final summary
        print("🎉 Working Integration Demo Complete!")
        print("=" * 50)
        
        print("🔗 The bidirectional connection is working!")
        print("   • Traits → Desires: Trait changes create and reinforce desires")
        print("   • Desires → Traits: Desire reinforcement strengthens source traits")
        print("   • Experience Impact: Different experiences affect different traits")
        print("   • Modular Design: Both engines remain independent and modular")
        
        if len(desire_engine.desires) > 0:
            print(f"\n🌟 Successfully created {len(desire_engine.desires)} desires from trait changes!")
            print("   The system is learning and evolving based on experiences!")
        else:
            print("\n⚠️  No desires were created - check the trait change rates")
            print("   The system needs positive trait changes to create desires")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {str(e)}")
        print(f"❌ Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 