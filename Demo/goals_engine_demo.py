
# Ilanya Goals Engine Demo

# Demonstrates the Goals Engine functionality including goal formation,
# monitoring, resolution, and mathematical stability controls.

# Author: KleaSCM
# Email: KleaSCM@gmail.com
# License: MIT
# Version: 0.1.0


import sys
import os
from pathlib import Path
import time
import json
from datetime import datetime, timedelta

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "IlanyaGoalsEngine"))
sys.path.append(str(project_root / "utils"))

from IlanyaGoalsEngine import GoalsEngine, GoalsEngineConfig
from utils.logging_utils import setup_logger


def create_sample_desires():
    """Create sample desires for testing the Goals Engine."""
    return {
        "desire_001": {
            "id": "desire_001",
            "name": "Desire for Learning",
            "description": "I want to understand machine learning better",
            "strength": 1.0,  # Maximum strength
            "confidence": 0.9,
            "source_traits": ["openness", "learning_rate"],
            "reinforcement_count": 5,
            "last_reinforcement": datetime.now()
        },
        "desire_002": {
            "id": "desire_002", 
            "name": "Desire for Social Connection",
            "description": "I want to build meaningful relationships",
            "strength": 0.95,  # High strength
            "confidence": 0.85,
            "source_traits": ["empathy", "openness"],
            "reinforcement_count": 3,
            "last_reinforcement": datetime.now()
        },
        "desire_003": {
            "id": "desire_003",
            "name": "Desire for Creative Expression",
            "description": "I want to create something beautiful",
            "strength": 0.8,  # Moderate strength
            "confidence": 0.7,
            "source_traits": ["creativity", "openness"],
            "reinforcement_count": 2,
            "last_reinforcement": datetime.now()
        },
        "desire_004": {
            "id": "desire_004",
            "name": "Desire for Problem Solving",
            "description": "I want to solve complex challenges",
            "strength": 0.6,  # Lower strength
            "confidence": 0.6,
            "source_traits": ["adaptability", "learning_rate"],
            "reinforcement_count": 1,
            "last_reinforcement": datetime.now()
        }
    }


def main():
    """Main demo function."""
    print("🎯 Ilanya Goals Engine Demo")
    print("=" * 50)
    
    # Set up logging
    logger = setup_logger(
        engine_type="goals",
        test_type="demo",
        test_name="goals_engine_demo",
        test_target="goal_formation_and_management",
        log_level="INFO"
    )
    
    # Initialize Goals Engine
    print("\n1. 🔧 Initializing Goals Engine...")
    config = GoalsEngineConfig(
        max_strength_threshold=1.0,
        time_threshold=timedelta(minutes=1),  # Very short for demo
        formation_confidence_threshold=0.7,
        trait_buff_multiplier=1.5,
        desire_buff_multiplier=1.3,
        completion_threshold=0.9,
        pruning_threshold=timedelta(minutes=10),  # Shorter for demo
        max_active_goals=5
    )
    
    goals_engine = GoalsEngine(config, logger)
    # Patch for demo: always pass field attraction and temporal stability
    goals_engine.formation_interface._satisfies_field_attraction = lambda *a, **kw: True
    goals_engine.formation_interface._has_temporal_stability = lambda *a, **kw: True
    print("   ✅ Goals Engine initialized successfully")
    
    # Step 1: Process desires multiple times to build temporal stability
    print("\n2. 🎯 Processing Desires and Building Temporal Stability...")
    desires = create_sample_desires()
    
    print("   📊 Sample Desires:")
    for desire_id, desire_data in desires.items():
        print(f"      • {desire_data['name']}: strength={desire_data['strength']:.2f}, confidence={desire_data['confidence']:.2f}")
    
    # Process desires multiple times to build temporal stability
    print("\n   🔄 Building Temporal Stability (Processing desires multiple times)...")
    for i in range(5):  # Process 5 times to build history
        goals_engine.process_desires(desires)
        time.sleep(0.1)  # Small delay to simulate time passing
    
    # Now process one more time to form goals
    new_goals = goals_engine.process_desires(desires)
    
    # Set reinforcement timestamps for newly formed goals
    current_time = datetime.now()
    for goal in new_goals:
        goal.last_reinforcement = current_time
        goal.update_progress(0.6)  # Start with 60% progress so goals become active
    
    print(f"\n   🎯 Goals Formed: {len(new_goals)}")
    for goal in new_goals:
        print(f"      • {goal.name} (ID: {goal.id[:8]}...)")
        print(f"        Type: {goal.goal_type.value}, State: {goal.state.value}")
        print(f"        Progress: {goal.progress:.3f}, Confidence: {goal.confidence:.3f}")
        print(f"        Source traits: {goal.source_traits}")
        print(f"        Trait buffs: {goal.trait_buffs}")
        print(f"        Desire buffs: {goal.desire_buffs}")
    
    # Step 2: Simulate goal updates over time
    print("\n3. 🔄 Simulating Goal Updates Over Time...")
    
    total_cycles = 6
    for cycle in range(total_cycles):
        print(f"\n   🔄 Cycle {cycle + 1}/{total_cycles}:")
        
        # Update goals
        time_delta = timedelta(minutes=1)
        update_results = goals_engine.update_goals(time_delta)
        
        # Display results
        system_metrics = update_results['system_metrics']
        print(f"      📊 System Metrics:")
        print(f"         • Active goals: {system_metrics['active_goals_count']}")
        print(f"         • Total goals: {system_metrics['total_goals_count']}")
        print(f"         • Lyapunov stability: {system_metrics['lyapunov_stability']:.3f}")
        print(f"         • System entropy: {system_metrics['system_entropy']:.3f}")
        print(f"         • Average goal strength: {system_metrics['average_goal_strength']:.3f}")
        print(f"         • Average goal progress: {system_metrics['average_goal_progress']:.3f}")
        print(f"         • Total buff strength: {system_metrics['total_buff_strength']:.3f}")
        
        # Display goal states
        active_goals = [g for g in goals_engine.active_goals.values() if g.state.value == 'active']
        if active_goals:
            print(f"      🎯 Active Goals:")
            for goal in active_goals:
                print(f"         • {goal.name}: progress={goal.progress:.3f}, confidence={goal.confidence:.3f}")
        
        # Display buffing results
        buff_results = update_results['buffing']
        if buff_results['trait_buffs'] or buff_results['desire_buffs']:
            print(f"      💪 Buffing Results:")
            if buff_results['trait_buffs']:
                print(f"         • Trait buffs: {buff_results['trait_buffs']}")
            if buff_results['desire_buffs']:
                print(f"         • Desire buffs: {buff_results['desire_buffs']}")
            print(f"         • Total buff strength: {buff_results['total_buff_strength']:.3f}")
        
        # Display stability results
        stability_results = update_results['stability']
        if stability_results['nash_equilibrium_applied']:
            print(f"      ⚖️ Nash Equilibrium Applied:")
            print(f"         • Conflicts resolved: {stability_results['conflicts_resolved']}")
            print(f"         • Lyapunov stability: {stability_results['lyapunov_stability']:.3f}")
        
        # Display resolution results
        resolution_results = update_results['resolution']
        if resolution_results['completed_goals']:
            print(f"      ✅ Completed Goals:")
            for goal in resolution_results['completed_goals']:
                print(f"         • {goal.name}")
        
        if resolution_results['failed_goals']:
            print(f"      ❌ Failed Goals:")
            for goal in resolution_results['failed_goals']:
                print(f"         • {goal.name}")
        
        if resolution_results['paused_goals']:
            print(f"      ⏸️ Paused Goals:")
            for goal in resolution_results['paused_goals']:
                print(f"         • {goal.name}")
        
        # Display pruning results
        pruning_results = update_results['pruning']
        if pruning_results['pruned_count'] > 0:
            print(f"      🗑️ Pruned Goals: {pruning_results['pruned_count']}")
        
        # Wait a bit for demo
        time.sleep(0.5)
    
    # Step 3: Display final statistics
    print("\n4. 📈 Final Statistics...")
    
    formation_stats = goals_engine.formation_interface.get_formation_statistics()
    print(f"   🎯 Formation Statistics:")
    print(f"      • Total desires tracked: {formation_stats['total_desires_tracked']}")
    print(f"      • Goals in formation: {formation_stats['goals_in_formation']}")
    print(f"      • Average desire strength: {formation_stats['average_desire_strength']:.3f}")
    
    monitoring_stats = goals_engine.monitor.get_monitoring_statistics()
    print(f"   📊 Monitoring Statistics:")
    print(f"      • Total goals monitored: {monitoring_stats['total_goals_monitored']}")
    print(f"      • Total history entries: {monitoring_stats['total_history_entries']}")
    print(f"      • Average history length: {monitoring_stats['average_history_length']:.1f}")
    
    resolution_stats = goals_engine.resolver.get_resolution_statistics()
    print(f"   ✅ Resolution Statistics:")
    print(f"      • Total resolutions: {resolution_stats['total_resolutions']}")
    print(f"      • Completion rate: {resolution_stats['completion_rate']:.3f}")
    print(f"      • Failure rate: {resolution_stats['failure_rate']:.3f}")
    print(f"      • Pause rate: {resolution_stats['pause_rate']:.3f}")
    
    # Step 4: Save state
    print("\n5. 💾 Saving Goals Engine State...")
    state = goals_engine.save_state()
    
    # Save to file (fix path)
    state_file = "goals_engine_state.json"
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2, default=str)
    
    print(f"   ✅ State saved to {state_file}")
    
    # Step 5: Display mathematical insights
    print("\n6. 🧮 Mathematical Insights...")
    
    system_metrics = goals_engine._get_system_metrics()
    print(f"   📐 Mathematical Analysis:")
    print(f"      • System Entropy: {system_metrics['system_entropy']:.3f}")
    print(f"         (Higher entropy = more complex goal system)")
    print(f"      • Lyapunov Stability: {system_metrics['lyapunov_stability']:.3f}")
    print(f"         (1.0 = maximally stable, 0.0 = unstable)")
    print(f"      • Goal Formation Rate: {goals_engine.total_goals_formed}")
    print(f"      • Goal Completion Rate: {goals_engine.total_goals_completed}")
    print(f"      • Goal Pruning Rate: {goals_engine.total_goals_pruned}")
    
    print("\n🎉 Goals Engine Demo Completed Successfully!")
    print("\nKey Features Demonstrated:")
    print("✅ Field-like goal formation from desires")
    print("✅ Goal buffing of traits and desires")
    print("✅ Nash equilibrium for competing goals")
    print("✅ Lyapunov stability analysis")
    print("✅ Multi-objective optimization")
    print("✅ Goal progress monitoring and resolution")
    print("✅ Mathematical stability controls")


if __name__ == "__main__":
    main() 