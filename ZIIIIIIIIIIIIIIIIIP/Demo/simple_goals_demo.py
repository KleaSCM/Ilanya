"""
Ilanya Goals Engine - Simple Demo

Simple demonstration of the Goals Engine functionality by directly creating goals
and showing the monitoring, buffing, and resolution processes.

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
from datetime import datetime, timedelta

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "IlanyaGoalsEngine"))
sys.path.append(str(project_root / "utils"))

from IlanyaGoalsEngine import GoalsEngine, GoalsEngineConfig, Goal, GoalType, GoalState
from utils.logging_utils import setup_logger


def create_sample_goals():
    """Create sample goals directly for testing the Goals Engine."""
    goals = {}
    current_time = datetime.now()
    
    # Learning goal
    learning_goal = Goal(
        name="Learn Machine Learning",
        description="I want to understand machine learning algorithms and applications",
        goal_type=GoalType.LEARNING,
        source_desires=["desire_001"],
        source_traits=["openness", "learning_rate"],
        formation_strength=1.0,
        formation_time=current_time,
        state=GoalState.ACTIVE,
        current_strength=0.9,
        progress=0.3,
        confidence=0.8
    )
    learning_goal.apply_trait_buff("openness", 1.5)
    learning_goal.apply_trait_buff("learning_rate", 1.5)
    learning_goal.apply_desire_buff("desire_001", 1.3)
    learning_goal.last_reinforcement = current_time  # Set reinforcement time
    goals[learning_goal.id] = learning_goal
    
    # Social goal
    social_goal = Goal(
        name="Build Meaningful Relationships",
        description="I want to connect with others and build lasting friendships",
        goal_type=GoalType.SOCIAL,
        source_desires=["desire_002"],
        source_traits=["empathy", "openness"],
        formation_strength=0.95,
        formation_time=current_time,
        state=GoalState.ACTIVE,
        current_strength=0.85,
        progress=0.5,
        confidence=0.75
    )
    social_goal.apply_trait_buff("empathy", 1.5)
    social_goal.apply_trait_buff("openness", 1.3)
    social_goal.apply_desire_buff("desire_002", 1.3)
    social_goal.last_reinforcement = current_time  # Set reinforcement time
    goals[social_goal.id] = social_goal
    
    # Creative goal
    creative_goal = Goal(
        name="Create Beautiful Art",
        description="I want to express myself through creative projects",
        goal_type=GoalType.CREATIVE,
        source_desires=["desire_003"],
        source_traits=["creativity", "openness"],
        formation_strength=0.8,
        formation_time=current_time,
        state=GoalState.ACTIVE,
        current_strength=0.7,
        progress=0.2,
        confidence=0.6
    )
    creative_goal.apply_trait_buff("creativity", 1.5)
    creative_goal.apply_trait_buff("openness", 1.2)
    creative_goal.apply_desire_buff("desire_003", 1.3)
    creative_goal.last_reinforcement = current_time  # Set reinforcement time
    goals[creative_goal.id] = creative_goal
    
    return goals


def main():
    """Main demo function."""
    print("🎯 Ilanya Goals Engine - Simple Demo")
    print("=" * 50)
    
    # Set up logging
    logger = setup_logger(
        engine_type="goals",
        test_type="demo",
        test_name="simple_goals_demo",
        test_target="goal_management",
        log_level="INFO"
    )
    
    # Initialize Goals Engine
    print("\n1. 🔧 Initializing Goals Engine...")
    config = GoalsEngineConfig(
        max_strength_threshold=1.0,
        time_threshold=timedelta(minutes=1),
        formation_confidence_threshold=0.7,
        trait_buff_multiplier=1.5,
        desire_buff_multiplier=1.3,
        completion_threshold=0.9,
        pruning_threshold=timedelta(minutes=10),
        max_active_goals=5
    )
    
    goals_engine = GoalsEngine(config, logger)
    print("   ✅ Goals Engine initialized successfully")
    
    # Step 1: Create sample goals directly
    print("\n2. 🎯 Creating Sample Goals...")
    sample_goals = create_sample_goals()
    
    # Add goals to the engine
    for goal_id, goal in sample_goals.items():
        goals_engine.active_goals[goal_id] = goal
        goals_engine.total_goals_formed += 1
    
    print(f"   🎯 Goals Created: {len(sample_goals)}")
    for goal in sample_goals.values():
        print(f"      • {goal.name} (ID: {goal.id[:8]}...)")
        print(f"        Type: {goal.goal_type.value}, State: {goal.state.value}")
        print(f"        Progress: {goal.progress:.3f}, Confidence: {goal.confidence:.3f}")
        print(f"        Trait buffs: {goal.trait_buffs}")
        print(f"        Desire buffs: {goal.desire_buffs}")
    
    # Step 2: Simulate goal updates over time
    print("\n3. 🔄 Simulating Goal Updates Over Time...")
    
    total_cycles = 8
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
    
    # Save to file
    state_file = "simple_goals_state.json"
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
    
    # Step 6: Show goal relationships
    print("\n7. 🔗 Goal Relationships...")
    for goal in goals_engine.active_goals.values():
        print(f"   🎯 {goal.name}:")
        print(f"      • Source traits: {goal.source_traits}")
        print(f"      • Trait buffs: {goal.trait_buffs}")
        print(f"      • Desire buffs: {goal.desire_buffs}")
        print(f"      • Total buff strength: {goal.get_total_buff_strength():.3f}")
        print(f"      • Interaction strength: {goal.interaction_strength:.3f}")
    
    print("\n🎉 Simple Goals Engine Demo Completed Successfully!")
    print("\nKey Features Demonstrated:")
    print("✅ Goal creation and management")
    print("✅ Goal buffing of traits and desires")
    print("✅ Nash equilibrium for competing goals")
    print("✅ Lyapunov stability analysis")
    print("✅ Goal progress monitoring and resolution")
    print("✅ Mathematical stability controls")
    print("✅ Goal-trait-desire feedback loops")


if __name__ == "__main__":
    main() 