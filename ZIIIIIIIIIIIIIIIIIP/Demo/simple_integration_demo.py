"""
Ilanya Simple Integration Demo

Demonstrates the bidirectional connection with manually created trait states
that have positive change rates to show desires being created.

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

from utils.engine_integration import create_integrated_system, IntegrationConfig
from utils.logging_utils import setup_logger


def create_positive_trait_states():
    """Create trait states with positive change rates to trigger desire creation."""
    
    # Create trait states with positive changes
    trait_states = {
        "openness": {
            'current_value': 0.8,
            'confidence': 0.9,
            'change_rate': 0.4,  # Large positive change!
            'trait_type': 'openness'
        },
        "creativity": {
            'current_value': 0.85,
            'confidence': 0.8,
            'change_rate': 0.35,  # Large positive change!
            'trait_type': 'creativity'
        },
        "learning_rate": {
            'current_value': 0.9,
            'confidence': 0.85,
            'change_rate': 0.5,  # Very large positive change!
            'trait_type': 'learning_rate'
        },
        "empathy": {
            'current_value': 0.75,
            'confidence': 0.8,
            'change_rate': 0.3,  # Just at threshold
            'trait_type': 'empathy'
        },
        "adaptability": {
            'current_value': 0.7,
            'confidence': 0.75,
            'change_rate': 0.25,  # Smaller but still significant
            'trait_type': 'adaptability'
        }
    }
    
    return trait_states


def main():
    """Main demonstration function."""
    print("🚀 Ilanya Simple Integration Demo")
    print("=" * 50)
    print("Demonstrating bidirectional connection with positive trait changes")
    print()
    
    # Set up logging
    logger = setup_logger(
        engine_type="demo",
        test_type="demo",
        test_name="simple_integration",
        test_target="positive_changes",
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
            reinforcement_strength=0.2,
            trait_activation_threshold=0.01,
            desire_reinforcement_threshold=0.1,
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
        
        # Step 3: Create trait states with positive changes
        print("3. 📊 Creating Trait States with Positive Changes...")
        
        trait_states = create_positive_trait_states()
        
        print("   📈 Trait states with positive changes:")
        for trait_name, trait_state in trait_states.items():
            change_rate = trait_state['change_rate']
            current_value = trait_state['current_value']
            print(f"      📈 {trait_name}: {current_value:.3f} (change: +{change_rate:.3f})")
        
        print()
        
        # Step 4: Process trait activations directly through desire engine
        print("4. 🔄 Processing Trait Activations...")
        
        # Process directly through desire engine to bypass integration conversion
        results = desire_engine.process_trait_activations(trait_states)
        
        print(f"   🎯 Processing results:")
        print(f"      • New desires created: {len(results.get('new_desires', []))}")
        print(f"      • Reinforced desires: {len(results.get('reinforced_desires', []))}")
        print(f"      • Active desires: {results.get('active_desires', 0)}")
        print(f"      • Total desires: {results.get('total_desires', 0)}")
        
        # Show active desires
        if desire_engine.desires:
            print("\n   📋 Active Desires:")
            for desire_id, desire in desire_engine.desires.items():
                print(f"      • {desire.name}: {desire.strength:.3f} (from: {desire.source_traits})")
        
        # Show trait-desire mappings
        if hasattr(desire_engine, 'trait_desire_mapping') and desire_engine.trait_desire_mapping:
            print("\n   🔗 Trait-Desire Mappings:")
            for trait, desires in desire_engine.trait_desire_mapping.items():
                print(f"      • {trait} → {desires}")
        
        print()
        
        # Step 5: Demonstrate desire reinforcement
        print("5. 💪 Demonstrating Desire Reinforcement...")
        
        if desire_engine.desires:
            # Reinforce the strongest desire
            strongest_desire_id = max(desire_engine.desires.keys(), 
                                    key=lambda x: desire_engine.desires[x].strength)
            strongest_desire = desire_engine.desires[strongest_desire_id]
            
            print(f"   🎯 Reinforcing strongest desire: {strongest_desire.name}")
            print(f"      • Current strength: {strongest_desire.strength:.3f}")
            print(f"      • Source traits: {strongest_desire.source_traits}")
            
            # Reinforce the desire
            reinforcement_strength = 0.5
            reinforcement_results = integration.process_desire_reinforcement(
                strongest_desire_id, reinforcement_strength
            )
            
            print(f"   💪 Reinforcement results:")
            print(f"      • Traits reinforced: {len(reinforcement_results['reinforced_traits'])}")
            print(f"      • Total reinforcement: {reinforcement_results['strength']:.3f}")
            
            if reinforcement_results['reinforced_traits']:
                print(f"      • Reinforced traits: {reinforcement_results['reinforced_traits']}")
        else:
            print("   ⚠️  No desires to reinforce")
        
        print()
        
        # Step 6: Show final results
        print("6. 📊 Final Results...")
        
        final_summary = integration.get_integration_summary()
        
        print(f"   🔄 Total integration cycles: {final_summary['stats']['integration_cycles']}")
        print(f"   🎯 Desires created: {final_summary['stats']['desires_created']}")
        print(f"   💪 Traits reinforced: {final_summary['stats']['traits_reinforced']}")
        print(f"   🔗 Active desires: {len(desire_engine.desires)}")
        print(f"   📋 Trait-desire mappings: {len(final_summary['trait_desire_mapping'])}")
        
        # Show reinforcement history
        if final_summary['reinforcement_history']:
            print("\n   📈 Reinforcement History:")
            for event in final_summary['reinforcement_history']:
                print(f"      • {event['desire_id']}: {event['total_reinforcement']:.3f} strength")
        
        print()
        
        # Step 7: Save final state
        print("7. 💾 Saving Final State...")
        
        state_file = "simple_integration_state.json"
        integration.save_integration_state(state_file)
        
        print(f"   ✅ Final state saved to {state_file}")
        print()
        
        # Final summary
        print("🎉 Simple Integration Demo Complete!")
        print("=" * 50)
        
        print("🔗 The bidirectional connection is working!")
        print("   • Traits → Desires: Positive trait changes create desires")
        print("   • Desires → Traits: Desire reinforcement strengthens source traits")
        print("   • Modular Design: Both engines remain independent and modular")
        
        if len(desire_engine.desires) > 0:
            print(f"\n🌟 Successfully created {len(desire_engine.desires)} desires from positive trait changes!")
            print("   The system is working as expected!")
        else:
            print("\n⚠️  No desires were created - check the trait change rates")
            print("   The system needs positive trait changes to create desires")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {str(e)}")
        print(f"❌ Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 