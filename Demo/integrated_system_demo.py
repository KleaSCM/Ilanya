"""
Ilanya Integrated System Demo

Demonstrates the bidirectional connection between Desire Engine and Trait Engine.
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

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "IlanyaDesireEngine"))
sys.path.append(str(project_root / "IlanyaTraitEngine" / "src"))
sys.path.append(str(project_root / "utils"))

from utils.engine_integration import EngineIntegration, IntegrationConfig, create_integrated_system
from utils.logging_utils import setup_logger


def create_sample_trait_data():
    """Create sample trait data for demonstration."""
    from IlanyaTraitEngine.src.trait_models.trait_types import TraitType
    from IlanyaTraitEngine.src.trait_models.trait_data import TraitVector, TraitMatrix, TraitData
    
    # Create sample traits
    traits = {
        TraitType.OPENNESS: TraitVector(TraitType.OPENNESS, 0.7, 0.9),
        TraitType.CREATIVITY: TraitVector(TraitType.CREATIVITY, 0.8, 0.8),
        TraitType.ADAPTABILITY: TraitVector(TraitType.ADAPTABILITY, 0.6, 0.7),
        TraitType.EMPATHY: TraitVector(TraitType.EMPATHY, 0.9, 0.8),
        TraitType.LEARNING_RATE: TraitVector(TraitType.LEARNING_RATE, 0.8, 0.9)
    }
    
    # Create trait matrix
    trait_matrix = TraitMatrix(traits=traits)
    
    # Create trait data
    trait_data = TraitData(
        trait_matrix=trait_matrix,
        source="integrated_demo"
    )
    
    return trait_data


def main():
    """Main demonstration function."""
    print("🚀 Ilanya Integrated System Demo")
    print("=" * 50)
    print("Demonstrating bidirectional connection between Desire and Trait Engines")
    print()
    
    # Set up logging
    logger = setup_logger(
        engine_type="demo",
        test_type="demo",
        test_name="integrated_system",
        test_target="bidirectional_integration",
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
        
        # Configure integration
        config = IntegrationConfig(
            reinforcement_strength=0.15,  # Stronger reinforcement
            trait_activation_threshold=0.03,  # Lower threshold for more desires
            desire_reinforcement_threshold=0.2,  # Lower threshold for reinforcement
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
        
        # Step 3: Process initial trait data
        print("3. 🧬 Processing Initial Trait Data...")
        
        trait_data = create_sample_trait_data()
        print(f"   📊 Created {len(trait_data.trait_matrix.traits)} traits:")
        
        for trait_type, trait_vector in trait_data.trait_matrix.traits.items():
            print(f"      • {trait_type.value}: {trait_vector.value:.2f} (confidence: {trait_vector.confidence:.2f})")
        
        print()
        
        # Step 4: Process trait activations through integration
        print("4. 🔄 Processing Trait Activations...")
        
        results = integration.process_trait_activations(trait_data)
        
        # Show results
        active_desires = len(desire_engine.desires)
        print(f"   🎯 Created {active_desires} desires from traits")
        
        if active_desires > 0:
            print("   📋 Active desires:")
            for desire_id, desire in list(desire_engine.desires.items())[:5]:  # Show first 5
                print(f"      • {desire.name}: {desire.strength:.3f} (from traits: {desire.source_traits})")
        
        print()
        
        # Step 5: Demonstrate desire reinforcement
        print("5. 💪 Demonstrating Desire Reinforcement...")
        
        if active_desires > 0:
            # Reinforce a desire
            desire_id = list(desire_engine.desires.keys())[0]
            desire = desire_engine.desires[desire_id]
            
            print(f"   🎯 Reinforcing desire: {desire.name}")
            print(f"   📊 Original strength: {desire.strength:.3f}")
            
            # Reinforce the desire
            reinforcement_strength = 0.5
            reinforcement_results = integration.process_desire_reinforcement(desire_id, reinforcement_strength)
            
            print(f"   💪 Reinforcement strength: {reinforcement_strength}")
            print(f"   🔄 Reinforced traits: {reinforcement_results['reinforced_traits']}")
            print(f"   📈 Total reinforcement: {reinforcement_results['strength']:.3f}")
            
            # Show updated desire strength
            print(f"   📊 New desire strength: {desire.strength:.3f}")
        else:
            print("   ⚠️  No desires to reinforce")
        
        print()
        
        # Step 6: Show integration summary
        print("6. 📊 Integration Summary...")
        
        summary = integration.get_integration_summary()
        
        print(f"   🔄 Integration cycles: {summary['stats']['integration_cycles']}")
        print(f"   🎯 Desires created: {summary['stats']['desires_created']}")
        print(f"   💪 Traits reinforced: {summary['stats']['traits_reinforced']}")
        print(f"   🔗 Trait-desire mappings: {len(summary['trait_desire_mapping'])}")
        
        # Show some mappings
        if summary['trait_desire_mapping']:
            print("   📋 Sample trait-desire mappings:")
            for trait, desires in list(summary['trait_desire_mapping'].items())[:3]:
                print(f"      • {trait} → {desires}")
        
        print()
        
        # Step 7: Save integration state
        print("7. 💾 Saving Integration State...")
        
        state_file = "integrated_system_state.json"
        integration.save_integration_state(state_file)
        
        print(f"   ✅ Integration state saved to {state_file}")
        print()
        
        # Step 8: Demonstrate multiple cycles
        print("8. 🔄 Running Multiple Integration Cycles...")
        
        for cycle in range(3):
            print(f"   🔄 Cycle {cycle + 1}:")
            
            # Simulate trait changes
            for trait_type, trait_vector in trait_data.trait_matrix.traits.items():
                # Small random change
                import random
                change = random.uniform(-0.1, 0.1)
                trait_vector.value = max(0.0, min(1.0, trait_vector.value + change))
            
            # Process through integration
            results = integration.process_trait_activations(trait_data)
            
            # Show results
            active_desires = len(desire_engine.desires)
            print(f"      📊 Active desires: {active_desires}")
            
            if active_desires > 0:
                # Reinforce a random desire
                desire_id = random.choice(list(desire_engine.desires.keys()))
                reinforcement_strength = random.uniform(0.2, 0.6)
                reinforcement_results = integration.process_desire_reinforcement(desire_id, reinforcement_strength)
                print(f"      💪 Reinforced {len(reinforcement_results['reinforced_traits'])} traits")
        
        print()
        
        # Final summary
        print("🎉 Integration Demo Complete!")
        print("=" * 50)
        
        final_summary = integration.get_integration_summary()
        print(f"📊 Final Statistics:")
        print(f"   • Integration cycles: {final_summary['stats']['integration_cycles']}")
        print(f"   • Desires created: {final_summary['stats']['desires_created']}")
        print(f"   • Traits reinforced: {final_summary['stats']['traits_reinforced']}")
        print(f"   • Active desires: {len(desire_engine.desires)}")
        print(f"   • Trait-desire mappings: {len(final_summary['trait_desire_mapping'])}")
        
        print("\n🔗 The bidirectional connection is working!")
        print("   • Traits → Desires: Trait activations create and reinforce desires")
        print("   • Desires → Traits: Desire reinforcement strengthens source traits")
        print("   • Modular Design: Both engines remain independent and modular")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {str(e)}")
        print(f"❌ Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 