"""
Ilanya Dynamic Integration Demo

Demonstrates the bidirectional connection with actual trait changes over time.
Shows how traits create desires and how desires reinforce traits with realistic data.

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


def create_dynamic_trait_data(cycle: int, base_traits: Optional[Dict] = None):
    """Create trait data with simulated changes over time."""
    from IlanyaTraitEngine.src.trait_models.trait_types import TraitType
    from IlanyaTraitEngine.src.trait_models.trait_data import TraitVector, TraitMatrix, TraitData
    
    # Initialize base traits if not provided
    if base_traits is None:
        base_traits = {
            TraitType.OPENNESS: 0.7,
            TraitType.CREATIVITY: 0.8,
            TraitType.ADAPTABILITY: 0.6,
            TraitType.EMPATHY: 0.9,
            TraitType.LEARNING_RATE: 0.8
        }
    
    # Simulate trait changes over time
    traits = {}
    for trait_type, base_value in base_traits.items():
        # Add some variation based on cycle
        variation = random.uniform(-0.1, 0.15) * (cycle + 1) * 0.1
        new_value = max(0.0, min(1.0, base_value + variation))
        
        # Simulate confidence changes
        confidence = max(0.5, min(1.0, 0.8 + random.uniform(-0.1, 0.1)))
        
        traits[trait_type] = TraitVector(trait_type, new_value, confidence)
    
    # Create trait matrix
    trait_matrix = TraitMatrix(traits=traits)
    
    # Create trait data
    trait_data = TraitData(
        trait_matrix=trait_matrix,
        source=f"dynamic_demo_cycle_{cycle}"
    )
    
    return trait_data, base_traits


def simulate_experience_impact(trait_data, experience_type: str):
    """Simulate how different experiences impact traits."""
    from IlanyaTraitEngine.src.trait_models.trait_types import TraitType
    
    # Define experience impacts
    experience_impacts = {
        "learning": {
            TraitType.LEARNING_RATE: 0.1,
            TraitType.OPENNESS: 0.05,
            TraitType.CREATIVITY: 0.03
        },
        "social": {
            TraitType.EMPATHY: 0.1,
            TraitType.ADAPTABILITY: 0.05,
            TraitType.OPENNESS: 0.03
        },
        "creative": {
            TraitType.CREATIVITY: 0.1,
            TraitType.OPENNESS: 0.05,
            TraitType.LEARNING_RATE: 0.03
        },
        "challenging": {
            TraitType.ADAPTABILITY: 0.1,
            TraitType.LEARNING_RATE: 0.05,
            TraitType.EMPATHY: 0.03
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


def main():
    """Main demonstration function."""
    print("🚀 Ilanya Dynamic Integration Demo")
    print("=" * 50)
    print("Demonstrating bidirectional connection with realistic trait changes")
    print()
    
    # Set up logging
    logger = setup_logger(
        engine_type="demo",
        test_type="demo",
        test_name="dynamic_integration",
        test_target="realistic_changes",
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
        
        base_traits = None
        total_cycles = 10
        
        for cycle in range(total_cycles):
            print(f"   🔄 Cycle {cycle + 1}/{total_cycles}:")
            
            # Create trait data with changes
            trait_data, base_traits = create_dynamic_trait_data(cycle, base_traits)
            
            # Simulate different experiences
            experiences = ["learning", "social", "creative", "challenging"]
            experience = random.choice(experiences)
            trait_data = simulate_experience_impact(trait_data, experience)
            
            print(f"      🎯 Experience: {experience}")
            print(f"      📊 Trait values:")
            for trait_type, trait_vector in trait_data.trait_matrix.traits.items():
                print(f"         • {trait_type.value}: {trait_vector.value:.3f}")
            
            # Process through integration
            results = integration.process_trait_activations(trait_data)
            
            # Show results
            active_desires = len(desire_engine.desires)
            print(f"      🎯 Active desires: {active_desires}")
            
            if active_desires > 0:
                # Show some desires
                for desire_id, desire in list(desire_engine.desires.items())[:3]:
                    print(f"         💭 {desire.name}: {desire.strength:.3f}")
                
                # Reinforce a random desire occasionally
                if cycle % 3 == 0 and active_desires > 0:
                    desire_id = random.choice(list(desire_engine.desires.keys()))
                    reinforcement_strength = random.uniform(0.3, 0.7)
                    reinforcement_results = integration.process_desire_reinforcement(desire_id, reinforcement_strength)
                    print(f"      💪 Reinforced {len(reinforcement_results['reinforced_traits'])} traits")
            
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
        
        state_file = "dynamic_integration_state.json"
        integration.save_integration_state(state_file)
        
        print(f"   ✅ Final state saved to {state_file}")
        print()
        
        # Final summary
        print("🎉 Dynamic Integration Demo Complete!")
        print("=" * 50)
        
        print("🔗 The bidirectional connection is working dynamically!")
        print("   • Traits → Desires: Trait changes create and reinforce desires")
        print("   • Desires → Traits: Desire reinforcement strengthens source traits")
        print("   • Experience Impact: Different experiences affect different traits")
        print("   • Modular Design: Both engines remain independent and modular")
        
        if len(desire_engine.desires) > 0:
            print(f"\n🌟 Successfully created {len(desire_engine.desires)} desires from trait changes!")
            print("   The system is learning and evolving based on experiences!")
        else:
            print("\n⚠️  No desires were created - trait changes may have been too small")
            print("   Try adjusting the trait_activation_threshold for more sensitivity")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {str(e)}")
        print(f"❌ Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 