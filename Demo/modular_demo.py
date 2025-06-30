#!/usr/bin/env python3
"""
Ilanya Desire Engine - Modular Demo with Logging

Demonstrates the modular Desire Engine with comprehensive logging
and emergent desire creation capabilities.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

import sys
import os
import time
from datetime import datetime, timedelta

# Add the parent directory to the path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add the IlanyaDesireEngine directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'IlanyaDesireEngine'))

from IlanyaDesireEngine.desire_engine import DesireEngine, DesireEngineConfig
from IlanyaDesireEngine.desire_engine.models import Desire, DesireState
from utils.logging_utils import setup_logger, log_demo_start, log_demo_end, get_log_file_path


def create_sample_trait_states():
    """Create sample trait states for demonstration."""
    return {
        'openness_to_experience': {
            'current_value': 0.8,
            'change_rate': 0.3,
            'stability': 0.8
        },
        'conscientiousness': {
            'current_value': 0.7,
            'change_rate': 0.2,
            'stability': 0.9
        },
        'extraversion': {
            'current_value': 0.9,
            'change_rate': 0.4,
            'stability': 0.7
        },
        'agreeableness': {
            'current_value': 0.6,
            'change_rate': 0.15,
            'stability': 0.85
        },
        'neuroticism': {
            'current_value': 0.3,
            'change_rate': -0.1,
            'stability': 0.75
        },
        'creativity': {
            'current_value': 0.9,
            'change_rate': 0.5,
            'stability': 0.6
        },
        'curiosity': {
            'current_value': 0.8,
            'change_rate': 0.35,
            'stability': 0.7
        },
        'ambition': {
            'current_value': 0.8,
            'change_rate': 0.25,
            'stability': 0.8
        },
        'social_connection': {
            'current_value': 0.7,
            'change_rate': 0.3,
            'stability': 0.75
        },
        'learning_desire': {
            'current_value': 0.9,
            'change_rate': 0.4,
            'stability': 0.7
        },
        # Add overlapping traits to trigger interactions
        'innovation': {
            'current_value': 0.8,
            'change_rate': 0.3,
            'stability': 0.7
        },
        'exploration': {
            'current_value': 0.7,
            'change_rate': 0.25,
            'stability': 0.8
        },
        'problem_solving': {
            'current_value': 0.8,
            'change_rate': 0.35,
            'stability': 0.75
        }
    }


def print_desire_summary(desire_engine, iteration):
    """Print a summary of the current desire state."""
    print(f"\n{'='*60}")
    print(f"ITERATION {iteration} - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    
    summary = desire_engine.get_desire_summary()
    
    print(f"Active Desires: {len(summary['active_desires'])}")
    print(f"Pruned Desires: {len(summary['pruned_desires'])}")
    print(f"Goal Candidates: {len(summary['goal_candidates'])}")
    
    if summary['active_desires']:
        print("\nActive Desires:")
        for desire_id, desire_data in summary['active_desires'].items():
            state_emoji = {
                'active': '🟢',
                'reinforced': '🔵',
                'weakening': '🟡',
                'goal_candidate': '⭐',
                'pruned': '🔴'
            }.get(desire_data['state'], '⚪')
            
            emergent_marker = "🌟" if desire_data.get('emergent', False) else ""
            print(f"  {state_emoji} {desire_data['name']} {emergent_marker}")
            print(f"    Strength: {desire_data['strength']:.3f} | "
                  f"Reinforcements: {desire_data['reinforcement_count']} | "
                  f"Goal Potential: {desire_data['goal_potential']:.3f}")
    
    if summary['goal_candidates']:
        print(f"\nGoal Candidates: {summary['goal_candidates']}")
    
    metrics = summary['metrics']
    print(f"\nMetrics:")
    print(f"  Entropy: {metrics.get('entropy', 0):.3f}")
    print(f"  Complexity: {metrics.get('complexity', 0):.3f}")
    print(f"  Stability: {metrics.get('stability', 0):.3f}")
    print(f"  Average Strength: {metrics.get('average_strength', 0):.3f}")
    
    interaction_network = summary['interaction_network']
    print(f"\nInteraction Network:")
    print(f"  Total Interactions: {interaction_network.get('total_interactions', 0)}")
    print(f"  Synergy Count: {interaction_network.get('synergy_count', 0)}")
    print(f"  Conflict Count: {interaction_network.get('conflict_count', 0)}")
    print(f"  Emergent Desires: {interaction_network.get('emergent_desires', 0)}")


def main():
    """Main demonstration function."""
    # Set up logger for this demo
    logger = setup_logger(
        engine_type="desire",
        test_type="demo",
        test_name="modular_desire_engine",
        test_target="emergent_desires",
        log_level="INFO"
    )
    
    start_time = time.time()
    log_demo_start(logger, "modular_desire_engine", 
                  "Demonstrates the modular Desire Engine with comprehensive logging and emergent desire creation")
    
    try:
    print("🚀 Ilanya Desire Engine - Modular Demo with Logging")
    print("=" * 60)
        
        # Get log file path for configuration
        log_file = get_log_file_path(
            engine_type="desire",
            test_type="demo",
            test_name="modular_desire_engine",
            test_target="emergent_desires"
        )
    
    # Create configuration with comprehensive logging and lower thresholds for demo
    config = DesireEngineConfig(
        log_level="INFO",
            log_file=log_file,
        log_desire_creation=True,
        log_desire_reinforcement=True,
        log_desire_decay=True,
        log_desire_pruning=True,
        log_interactions=True,
        log_emergent_desires=True,
        log_goal_candidates=True,
        log_metrics=True,
        # Lower thresholds for demo to ensure desires are created
        reinforcement_threshold=0.01,
        pruning_threshold=0.05,
        interaction_threshold=0.1,
        synergy_threshold=0.3,
        emergent_threshold=0.4,
        conflict_threshold=0.2,
        # Faster processing for demo
        decay_check_interval=30,
        pruning_check_interval=60
    )
    
    # Initialize the desire engine
        logger.info("Initializing Desire Engine...")
    print("Initializing Desire Engine...")
    desire_engine = DesireEngine(config)
    
    # Create sample trait states
    trait_states = create_sample_trait_states()
    
        logger.info(f"Created {len(trait_states)} sample trait states")
    print(f"Created {len(trait_states)} sample trait states")
    print("Starting desire processing simulation...")
    
    # Process trait activations for multiple iterations
    for iteration in range(1, 6):
            logger.info(f"Processing iteration {iteration}")
        print(f"\n🔄 Processing iteration {iteration}...")
        
        # Process trait activations
        results = desire_engine.process_trait_activations(trait_states)
        
        # Print summary
        print_desire_summary(desire_engine, iteration)
        
        # Simulate some time passing
        if iteration < 5:
                logger.info("Waiting 2 seconds before next iteration")
            print(f"\n⏰ Waiting 2 seconds before next iteration...")
            time.sleep(2)
    
    # Demonstrate neural embeddings and attention
        logger.info("Computing desire embeddings and attention")
    print(f"\n🧠 Computing desire embeddings and attention...")
        
        # Get embeddings for all desires
        desire_list = list(desire_engine.desires.values())
        embeddings = desire_engine.embedding_module(desire_list)
        
        logger.info(f"Generated embeddings for {len(embeddings)} desires")
        print(f"Generated embeddings for {len(embeddings)} desires")
    print(f"Embedding shape: {embeddings.shape}")
        
        # Show some embedding statistics
        print("\nEmbedding Statistics:")
        print(f"  Mean embedding value: {embeddings.mean().item():.3f}")
        print(f"  Std embedding value: {embeddings.std().item():.3f}")
        print(f"  Min embedding value: {embeddings.min().item():.3f}")
        print(f"  Max embedding value: {embeddings.max().item():.3f}")
        
        # Show some attention patterns (simplified)
        print("\nAttention Patterns (simplified):")
        for i, desire_id in enumerate(list(desire_engine.desires.keys())[:3]):
            desire = desire_engine.desires[desire_id]
            embedding = embeddings[i]
            # Show embedding magnitude as a simple attention metric
            attention_magnitude = embedding.norm().item()
            print(f"  {desire.name}: attention magnitude = {attention_magnitude:.3f}")
        
        # Final summary
        logger.info("Demo completed successfully")
        print(f"\n✅ Demo completed successfully!")
        print(f"📊 Final Statistics:")
        print(f"  Total Desires: {len(desire_engine.desires)}")
        print(f"  Emergent Desires: {len([d for d in desire_engine.desires.values() if d.emergent])}")
        
        # Count goal candidates safely
        goal_candidates = 0
        for d in desire_engine.desires.values():
            try:
                if getattr(d, 'goal_candidate', False):
                    goal_candidates += 1
            except:
                pass
        print(f"  Goal Candidates: {goal_candidates}")
        
        duration = time.time() - start_time
        log_demo_end(logger, "modular_desire_engine", duration)
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Demo failed with error: {str(e)}")
        log_demo_end(logger, "modular_desire_engine", duration)
        raise


if __name__ == "__main__":
    main() 