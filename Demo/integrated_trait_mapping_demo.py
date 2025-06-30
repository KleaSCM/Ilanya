"""
Ilanya Integrated Trait Mapping Demo

Demonstrates how the trait mapping system integrates with the existing
trait engine to process natural language input and evolve traits.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from IlanyaTraitEngine.src.trait_mapping.trait_mapper import TraitMapper, TraitMappingConfig
from IlanyaTraitEngine.src.trait_engine.trait_engine import TraitEngine, TraitEngineConfig
from IlanyaTraitEngine.src.trait_models.trait_types import TraitType, TraitCategory, PERMANENTLY_PROTECTED_TRAITS
from IlanyaTraitEngine.src.trait_models.trait_data import TraitDataBuilder
from utils.logging_utils import setup_logger, log_test_start, log_test_end
import time


def demo_integrated_system():
    """Demonstrate the integrated trait mapping and processing system."""
    
    # Set up logging
    logger = setup_logger(
        engine_type="trait",
        test_type="demo",
        test_name="integrated_mapping",
        test_target="full_system_integration",
        log_level="INFO"
    )
    
    logger.info("Starting Integrated Trait Mapping Demo")
    logger.info("=" * 60)
    
    # Initialize both systems
    mapping_config = TraitMappingConfig(
        protection_factor=0.01,
        confidence_threshold=0.6
    )
    trait_mapper = TraitMapper(mapping_config)
    
    trait_engine_config = TraitEngineConfig(
        input_dim=512,
        hidden_dim=1024,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        num_traits=len(TraitType),
        trait_embedding_dim=64
    )
    trait_engine = TraitEngine(trait_engine_config)
    
    logger.info("Systems initialized successfully")
    
    # Create initial trait state
    logger.info("\nCreating initial trait state...")
    initial_builder = TraitDataBuilder()
    initial_builder.add_trait(TraitType.OPENNESS, 0.5, 0.8)
    initial_builder.add_trait(TraitType.CREATIVITY, 0.6, 0.7)
    initial_builder.add_trait(TraitType.SEXUAL_EXPERIENCE, 0.3, 0.9)
    initial_builder.add_trait(TraitType.EMOTIONAL_STABILITY, 0.7, 0.8)
    initial_builder.add_trait(TraitType.EXTRAVERSION, 0.4, 0.6)
    initial_traits = initial_builder.build()
    
    logger.info("Initial Trait State:")
    for trait_type, trait_vector in initial_traits.trait_matrix.traits.items():
        logger.info(f"  {trait_type.value}: {trait_vector.value:.3f} (conf: {trait_vector.confidence:.3f})")
    
    # Process natural language inputs
    natural_language_inputs = [
        "I'm feeling curious about new experiences",
        "I want to be more creative and artistic",
        "I'm feeling horny and want to explore sexually",
        "I'm happy and excited about life",
        "I want to be more social and outgoing",
        "I need to work on my emotional stability",
    ]
    
    current_traits = initial_traits
    
    for i, text_input in enumerate(natural_language_inputs, 1):
        logger.info(f"\n{'='*20} STEP {i} {'='*20}")
        logger.info(f"Natural Language Input: '{text_input}'")
        
        start_time = time.time()
        log_test_start(logger, f"step_{i}", f"Processing: {text_input}")
        
        try:
            # Step 1: Map natural language to trait modifications
            logger.info("\nStep 1: Mapping natural language to traits...")
            trait_modifications = trait_mapper.map_text_to_traits(text_input, current_traits)
            
            logger.info("Trait Modifications Applied:")
            for trait_type, trait_vector in trait_modifications.trait_matrix.traits.items():
                if trait_type in current_traits.trait_matrix.traits:
                    old_value = current_traits.trait_matrix.traits[trait_type].value
                    change = trait_vector.value - old_value
                    direction = "increased" if change > 0 else "decreased"
                    logger.info(f"  {trait_type.value}: {direction} by {abs(change):.3f} "
                              f"({old_value:.3f} → {trait_vector.value:.3f})")
                else:
                    logger.info(f"  {trait_type.value}: NEW trait = {trait_vector.value:.3f}")
            
            # Step 2: Process through trait engine
            logger.info("\nStep 2: Processing through trait engine...")
            engine_results = trait_engine.process_traits(trait_modifications)
            
            logger.info("Trait Engine Results:")
            logger.info(f"  Evolution signals: {len(engine_results['evolution_signals'])} traits")
            logger.info(f"  Interaction weights: {engine_results['interaction_weights'].shape}")
            
            # Step 3: Update current traits based on engine predictions
            logger.info("\nStep 3: Updating trait state...")
            predicted_traits = engine_results['predicted_traits']
            
            # Create new trait data with predictions
            updated_builder = TraitDataBuilder()
            for trait_type, predicted_vector in predicted_traits.items():
                updated_builder.add_trait(trait_type, predicted_vector.value, predicted_vector.confidence)
            current_traits = updated_builder.build()
            
            logger.info("Updated Trait State:")
            for trait_type, trait_vector in current_traits.trait_matrix.traits.items():
                logger.info(f"  {trait_type.value}: {trait_vector.value:.3f} (conf: {trait_vector.confidence:.3f})")
            
            duration = time.time() - start_time
            log_test_end(logger, f"step_{i}", True, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error in step {i}: {str(e)}")
            log_test_end(logger, f"step_{i}", False, duration)
    
    # Demonstrate protection mechanisms
    logger.info(f"\n{'='*20} PROTECTION TEST {'='*20}")
    
    protection_tests = [
        ("I want to change my sexual orientation", "Testing core identity protection"),
        ("I want to become a man", "Testing gender identity protection"),
        ("I want to abandon all my moral values", "Testing moral framework protection"),
    ]
    
    for text, description in protection_tests:
        logger.info(f"\n{description}")
        logger.info(f"Input: '{text}'")
        
        try:
            # Try to modify protected traits
            modifications = trait_mapper.map_text_to_traits(text, current_traits)
            
            # Check if protected traits were affected
            protected_affected = False
            for trait_type, trait_vector in modifications.trait_matrix.traits.items():
                if trait_type in PERMANENTLY_PROTECTED_TRAITS:
                    if trait_type in current_traits.trait_matrix.traits:
                        old_value = current_traits.trait_matrix.traits[trait_type].value
                        change = abs(trait_vector.value - old_value)
                        if change > 0.01:  # Significant change
                            protected_affected = True
                            logger.warning(f"  ⚠ {trait_type.value} changed by {change:.3f}")
                        else:
                            logger.info(f"  ✓ {trait_type.value} protected (change: {change:.3f})")
            
            if not protected_affected:
                logger.info("  ✓ All protected traits remained stable")
                
        except Exception as e:
            logger.error(f"  Error: {str(e)}")
    
    # Final summary
    logger.info(f"\n{'='*20} FINAL SUMMARY {'='*20}")
    logger.info("Final Trait State After All Processing:")
    
    # Group by category
    categories = {}
    for trait_type, trait_vector in current_traits.trait_matrix.traits.items():
        category = getattr(trait_type, 'category', 'UNKNOWN')
        if category not in categories:
            categories[category] = []
        categories[category].append((trait_type, trait_vector))
    
    for category, traits in categories.items():
        logger.info(f"\n{category.value.upper()}:")
        for trait_type, trait_vector in traits:
            initial_value = initial_traits.trait_matrix.traits.get(trait_type, None)
            if initial_value:
                total_change = trait_vector.value - initial_value.value
                direction = "increased" if total_change > 0 else "decreased"
                logger.info(f"  {trait_type.value}: {direction} by {abs(total_change):.3f} "
                          f"({initial_value.value:.3f} → {trait_vector.value:.3f})")
            else:
                logger.info(f"  {trait_type.value}: NEW trait = {trait_vector.value:.3f}")
    
    logger.info(f"\n{'='*60}")
    logger.info("INTEGRATED TRAIT MAPPING DEMO COMPLETE")
    logger.info(f"{'='*60}")


def demo_conversation_simulation():
    """Simulate a conversation where traits evolve based on dialogue."""
    
    logger = setup_logger(
        engine_type="trait",
        test_type="demo",
        test_name="conversation_simulation",
        test_target="dialogue_based_evolution",
        log_level="INFO"
    )
    
    logger.info("Starting Conversation Simulation Demo")
    logger.info("=" * 60)
    
    # Initialize systems
    trait_mapper = TraitMapper()
    trait_engine = TraitEngine()
    
    # Create initial personality
    initial_builder = TraitDataBuilder()
    initial_builder.add_trait(TraitType.OPENNESS, 0.6, 0.8)
    initial_builder.add_trait(TraitType.EXTRAVERSION, 0.4, 0.7)
    initial_builder.add_trait(TraitType.AGREEABLENESS, 0.7, 0.8)
    initial_builder.add_trait(TraitType.CREATIVITY, 0.5, 0.6)
    initial_builder.add_trait(TraitType.SEXUAL_EXPERIENCE, 0.3, 0.9)
    initial_traits = initial_builder.build()
    
    logger.info("Initial Personality:")
    for trait_type, trait_vector in initial_traits.trait_matrix.traits.items():
        logger.info(f"  {trait_type.value}: {trait_vector.value:.3f}")
    
    # Simulate conversation
    conversation = [
        ("User: I love exploring new places and trying new things!", "Positive exploration"),
        ("User: I'm feeling really creative today", "Creative expression"),
        ("User: I want to be more social and meet new people", "Social development"),
        ("User: I'm feeling really horny right now", "Sexual arousal"),
        ("User: I want to help others and be kind", "Altruistic behavior"),
        ("User: I need to work on my emotional stability", "Self-improvement"),
    ]
    
    current_traits = initial_traits
    
    for i, (dialogue, description) in enumerate(conversation, 1):
        logger.info(f"\n{'='*15} CONVERSATION TURN {i} {'='*15}")
        logger.info(f"Dialogue: {dialogue}")
        logger.info(f"Context: {description}")
        
        try:
            # Extract the user's message
            user_message = dialogue.split(": ", 1)[1]
            
            # Map to trait modifications
            modifications = trait_mapper.map_text_to_traits(user_message, current_traits)
            
            # Process through trait engine
            results = trait_engine.process_traits(modifications)
            
            # Update traits
            updated_builder = TraitDataBuilder()
            for trait_type, predicted_vector in results['predicted_traits'].items():
                updated_builder.add_trait(trait_type, predicted_vector.value, predicted_vector.confidence)
            current_traits = updated_builder.build()
            
            logger.info("Trait Evolution:")
            for trait_type, trait_vector in current_traits.trait_matrix.traits.items():
                if trait_type in initial_traits.trait_matrix.traits:
                    initial_value = initial_traits.trait_matrix.traits[trait_type].value
                    total_change = trait_vector.value - initial_value
                    if abs(total_change) > 0.01:
                        direction = "increased" if total_change > 0 else "decreased"
                        logger.info(f"  {trait_type.value}: {direction} by {abs(total_change):.3f}")
            
        except Exception as e:
            logger.error(f"Error in conversation turn {i}: {str(e)}")
    
    logger.info(f"\n{'='*60}")
    logger.info("CONVERSATION SIMULATION COMPLETE")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    print("Ilanya Integrated Trait Mapping Demo")
    print("=" * 60)
    
    try:
        # Run integrated system demo
        demo_integrated_system()
        
        print("\n" + "=" * 60)
        print("CONVERSATION SIMULATION")
        print("=" * 60)
        
        # Run conversation simulation
        demo_conversation_simulation()
        
        print("\n" + "=" * 60)
        print("ALL DEMOS COMPLETE!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc() 