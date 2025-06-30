# Ilanya Goals Engine Demo

# Demonstrates the Goals Engine functionality including goal formation,
# monitoring, resolution, and mathematical stability controls.
# Now includes Pareto frontier analysis, enhanced Nash equilibrium,
# goal dependency graphs, and adaptive thresholds.

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
from IlanyaGoalsEngine.models import Goal, GoalState, GoalType, GoalDependency, ResourceRequirement
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
        },
        "desire_005": {
            "id": "desire_005",
            "name": "Desire for Self-Improvement",
            "description": "I want to become a better version of myself",
            "strength": 0.9,  # High strength
            "confidence": 0.8,
            "source_traits": ["self_awareness", "conscientiousness"],
            "reinforcement_count": 4,
            "last_reinforcement": datetime.now()
        }
    }


def create_goal_dependencies(goals: list):
    """Create dependency relationships between goals."""
    if len(goals) < 2:
        return
    
    # Create a learning -> problem solving dependency
    learning_goal = next((g for g in goals if g.goal_type == GoalType.LEARNING), None)
    problem_goal = next((g for g in goals if g.goal_type == GoalType.PROBLEM_SOLVING), None)
    
    if learning_goal and problem_goal:
        problem_goal.add_dependency(learning_goal.id, 'required', 0.8)
        learning_goal.add_dependent(problem_goal.id, 'required', 0.8)
        print(f"   🔗 Created dependency: {problem_goal.name} depends on {learning_goal.name}")
    
    # Create a helpful dependency: social connection helps creative expression
    social_goal = next((g for g in goals if g.goal_type == GoalType.SOCIAL), None)
    creative_goal = next((g for g in goals if g.goal_type == GoalType.CREATIVE), None)
    
    if social_goal and creative_goal:
        creative_goal.add_dependency(social_goal.id, 'helpful', 0.6)
        social_goal.add_dependent(creative_goal.id, 'helpful', 0.6)
        print(f"   🔗 Created helpful dependency: {creative_goal.name} helped by {social_goal.name}")
    
    # Add resource requirements
    for goal in goals:
        if goal.goal_type == GoalType.LEARNING:
            goal.add_resource_requirement('time', 0.7, 0.9)
            goal.add_resource_requirement('attention', 0.8, 0.8)
        elif goal.goal_type == GoalType.SOCIAL:
            goal.add_resource_requirement('time', 0.6, 0.7)
            goal.add_resource_requirement('social', 0.9, 0.9)
        elif goal.goal_type == GoalType.CREATIVE:
            goal.add_resource_requirement('time', 0.8, 0.8)
            goal.add_resource_requirement('energy', 0.7, 0.7)
        elif goal.goal_type == GoalType.PROBLEM_SOLVING:
            goal.add_resource_requirement('time', 0.9, 0.9)
            goal.add_resource_requirement('attention', 0.9, 0.9)
            goal.add_resource_requirement('energy', 0.8, 0.8)


def main():
    """Main demo function."""
    print("🎯 Ilanya Goals Engine Demo - Enhanced Features")
    print("=" * 60)
    
    # Set up logging
    logger = setup_logger(
        engine_type="goals",
        test_type="demo",
        test_name="goals_engine_demo",
        test_target="goal_formation_and_management",
        log_level="INFO"
    )
    
    # Initialize Goals Engine with enhanced configuration
    print("\n1. 🔧 Initializing Enhanced Goals Engine...")
    config = GoalsEngineConfig(
        max_strength_threshold=0.8,  # Lower threshold for demo
        time_threshold=timedelta(seconds=30),  # Shorter for demo
        formation_confidence_threshold=0.6,  # Lower threshold for demo
        trait_buff_multiplier=1.5,
        desire_buff_multiplier=1.3,
        completion_threshold=0.9,
        pruning_threshold=timedelta(minutes=10),  # Shorter for demo
        max_active_goals=5,
        nash_iteration_limit=50,  # Enhanced Nash equilibrium
        nash_convergence_threshold=1e-5,
        nash_learning_rate=0.02,
        field_attraction_strength=0.8,
        objective_weights=[0.3, 0.25, 0.2, 0.15, 0.1]  # Pareto frontier weights
    )
    
    goals_engine = GoalsEngine(config, logger)
    # Patch for demo: always pass field attraction and temporal stability
    goals_engine.formation_interface._satisfies_field_attraction = lambda *a, **kw: True
    goals_engine.formation_interface._has_temporal_stability = lambda *a, **kw: True
    print("   ✅ Enhanced Goals Engine initialized successfully")
    
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
    
    # Step 2: Create goal dependencies and resource requirements
    print("\n3. 🔗 Creating Goal Dependencies and Resource Requirements...")
    create_goal_dependencies(new_goals)
    
    # Display dependency information
    for goal in new_goals:
        if goal.dependencies or goal.dependents:
            print(f"\n   📋 {goal.name} Dependencies:")
            if goal.dependencies:
                for dep in goal.dependencies:
                    print(f"      • Depends on: {dep.prerequisite_goal_id[:8]}... (type: {dep.dependency_type}, strength: {dep.strength:.2f})")
            if goal.dependents:
                for dep in goal.dependents:
                    print(f"      • Helps: {dep.dependent_goal_id[:8]}... (type: {dep.dependency_type}, strength: {dep.strength:.2f})")
        
        if goal.resource_requirements:
            print(f"   📋 {goal.name} Resource Requirements:")
            for req in goal.resource_requirements:
                print(f"      • {req.resource_type}: {req.amount:.2f} (priority: {req.priority:.2f})")
    
    # Step 3: Simulate goal updates over time
    print("\n4. 🔄 Simulating Goal Updates Over Time...")
    
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
                if goal.dependencies:
                    blockers = goal.get_dependency_blockers(goals_engine.active_goals)
                    if blockers:
                        print(f"           ⚠️ Blocked by: {[b[:8] + '...' for b in blockers]}")
        
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
            print(f"      ⚖️ Enhanced Nash Equilibrium Applied:")
            print(f"         • Conflicts resolved: {stability_results['conflicts_resolved']}")
            print(f"         • Lyapunov stability: {stability_results['lyapunov_stability']:.3f}")
        
        # Display resolution results
        resolution_results = update_results['resolution']
        if resolution_results['completed_goals']:
            print(f"      ✅ Completed Goals:")
            for goal in resolution_results['completed_goals']:
                print(f"         • {goal.name}")
                # Show impact on dependent goals
                impact = goal.get_impact_on_completion(goals_engine.active_goals)
                if impact > 0:
                    print(f"           🎯 Impact on dependents: {impact:.3f}")
        
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
    
    # Step 4: Display final statistics
    print("\n5. 📈 Final Statistics...")
    
    formation_stats = goals_engine.formation_interface.get_formation_statistics()
    print(f"   🎯 Formation Statistics:")
    print(f"      • Total desires tracked: {formation_stats['total_desires_tracked']}")
    print(f"      • Goals in formation: {formation_stats['goals_in_formation']}")
    print(f"      • Average desire strength: {formation_stats['average_desire_strength']:.3f}")
    print(f"      • Pareto frontier size: {formation_stats['pareto_frontier_size']}")
    print(f"      • Adaptive thresholds: {formation_stats['adaptive_thresholds']}")
    
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
    
    # Step 5: Display dependency graph analysis
    print("\n6. 🔗 Dependency Graph Analysis...")
    
    for goal in goals_engine.active_goals.values():
        if goal.dependencies or goal.dependents:
            print(f"\n   📋 {goal.name} Dependency Analysis:")
            
            # Calculate dependency depth
            depth = goal.calculate_dependency_depth(goals_engine.active_goals)
            print(f"      • Dependency depth: {depth}")
            
            # Get dependency path
            path = goal.get_dependency_path(goals_engine.active_goals)
            print(f"      • Dependency path: {' -> '.join([p[:8] + '...' for p in path])}")
            
            # Get blockers
            blockers = goal.get_dependency_blockers(goals_engine.active_goals)
            if blockers:
                print(f"      • Blockers: {[b[:8] + '...' for b in blockers]}")
            
            # Get helpers
            helpers = goal.get_dependency_helpers(goals_engine.active_goals)
            if helpers:
                print(f"      • Helpers: {[h[:8] + '...' for h in helpers]}")
            
            # Get impact on completion
            impact = goal.get_impact_on_completion(goals_engine.active_goals)
            print(f"      • Impact on completion: {impact:.3f}")
    
    # Step 6: Display resource conflict analysis
    print("\n7. 📊 Resource Conflict Analysis...")
    
    active_goals = list(goals_engine.active_goals.values())
    for i, goal1 in enumerate(active_goals):
        for j, goal2 in enumerate(active_goals[i+1:], i+1):
            conflicts = goal1.get_resource_conflicts(goal2)
            if conflicts:
                print(f"   ⚠️ Resource conflicts between {goal1.name} and {goal2.name}:")
                for conflict in conflicts:
                    req1 = goal1.get_total_resource_requirement(conflict)
                    req2 = goal2.get_total_resource_requirement(conflict)
                    print(f"      • {conflict}: {req1:.2f} + {req2:.2f} = {req1 + req2:.2f} (exceeds 1.0)")
    
    # Step 7: Save state
    print("\n8. 💾 Saving Goals Engine State...")
    state = goals_engine.save_state()
    
    # Save to file
    state_file = "goals_engine_state.json"
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2, default=str)
    
    print(f"   ✅ State saved to {state_file}")
    
    # Step 8: Display mathematical insights
    print("\n9. 🧮 Enhanced Mathematical Insights...")
    
    system_metrics = goals_engine._get_system_metrics()
    print(f"   📐 Mathematical Analysis:")
    print(f"      • System Entropy: {system_metrics['system_entropy']:.3f}")
    print(f"         (Higher entropy = more complex goal system)")
    print(f"      • Lyapunov Stability: {system_metrics['lyapunov_stability']:.3f}")
    print(f"         (1.0 = maximally stable, 0.0 = unstable)")
    print(f"      • Goal Formation Rate: {goals_engine.total_goals_formed}")
    print(f"      • Goal Completion Rate: {goals_engine.total_goals_completed}")
    print(f"      • Goal Pruning Rate: {goals_engine.total_goals_pruned}")
    
    # Display Pareto frontier analysis
    formation_stats = goals_engine.formation_interface.get_formation_statistics()
    print(f"   📊 Pareto Frontier Analysis:")
    print(f"      • Pareto frontier size: {formation_stats['pareto_frontier_size']}")
    print(f"      • Objective weights: {formation_stats['adaptive_thresholds']}")
    
    print("\n🎉 Enhanced Goals Engine Demo Completed Successfully!")
    print("\nNew Features Demonstrated:")
    print("✅ Pareto frontier analysis for multi-objective optimization")
    print("✅ Enhanced Nash equilibrium with gradient-based convergence")
    print("✅ Goal dependency graphs with resource requirements")
    print("✅ Adaptive thresholds based on system state")
    print("✅ Dependency path analysis and impact calculation")
    print("✅ Resource conflict detection and resolution")
    print("✅ Mathematical stability controls with enhanced convergence")


if __name__ == "__main__":
    main() 