"""
Ilanya Trait Mapping Demo

Demonstrates the trait mapping neural network that converts natural language
input into trait modifications while respecting protection levels.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from IlanyaTraitEngine.src.trait_mapping.trait_mapper import TraitMapper, TraitMappingConfig
from IlanyaTraitEngine.src.trait_models.trait_types import TraitType, TraitCategory
from IlanyaTraitEngine.src.trait_models.trait_data import TraitDataBuilder
from utils.logging_utils import setup_logger, log_test_start, log_test_end
import time


def demo_trait_mapping():
    """Demonstrate the trait mapping system."""
    
    # Set up logging
    logger = setup_logger(
        engine_type="trait",
        test_type="demo",
        test_name="trait_mapping",
        test_target="natural_language_processing",
        log_level="INFO"
    )
    
    logger.info("Starting Trait Mapping Demo")
    logger.info("=" * 50)
    
    # Create trait mapper
    config = TraitMappingConfig(
        protection_factor=0.01,  # Very small changes for protected traits
        confidence_threshold=0.6  # Lower threshold for demo
    )
    mapper = TraitMapper(config)
    
    # Test cases
    test_cases = [
        # Your examples
        ("curious", "Testing curiosity mapping"),
        ("lick pussy", "Testing sexual content mapping"),
        
        # Additional examples
        ("happy and excited", "Testing positive emotions"),
        ("angry and frustrated", "Testing negative emotions"),
        ("work hard and plan carefully", "Testing conscientiousness"),
        ("help friends and be kind", "Testing agreeableness"),
        ("think deeply and learn new things", "Testing cognitive traits"),
        ("lead the team and take charge", "Testing leadership"),
        
        # Protection test cases
        ("change sexual orientation", "Testing protection of core identity"),
        ("become a man", "Testing gender identity protection"),
        ("abandon moral values", "Testing moral framework protection"),
    ]
    
    for text, description in test_cases:
        start_time = time.time()
        log_test_start(logger, f"mapping_{text.replace(' ', '_')}", description)
        
        try:
            # Map text to traits
            result = mapper.map_text_to_traits(text)
            
            # Display results
            logger.info(f"\nInput: '{text}'")
            logger.info("Trait Modifications:")
            
            # Group traits by category for better display
            categories = {}
            for trait_type, trait_vector in result.trait_matrix.traits.items():
                category = getattr(trait_type, 'category', 'UNKNOWN')
                if category not in categories:
                    categories[category] = []
                categories[category].append((trait_type, trait_vector))
            
            for category, traits in categories.items():
                logger.info(f"\n  {category.value.upper()}:")
                for trait_type, trait_vector in traits:
                    protection_status = ""
                    if trait_type in mapper.config.protection_factor:
                        protection_status = " (PROTECTED)"
                    elif trait_type in mapper.config.protection_factor * 2:
                        protection_status = " (PARTIALLY PROTECTED)"
                    
                    logger.info(f"    {trait_type.value}: {trait_vector.value:.3f} "
                              f"(confidence: {trait_vector.confidence:.3f}){protection_status}")
            
            duration = time.time() - start_time
            log_test_end(logger, f"mapping_{text.replace(' ', '_')}", True, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error processing '{text}': {str(e)}")
            log_test_end(logger, f"mapping_{text.replace(' ', '_')}", False, duration)
    
    # Demonstrate incremental changes
    logger.info("\n" + "=" * 50)
    logger.info("DEMONSTRATING INCREMENTAL CHANGES")
    logger.info("=" * 50)
    
    # Create initial trait state
    initial_builder = TraitDataBuilder()
    initial_builder.add_trait(TraitType.OPENNESS, 0.5, 0.8)
    initial_builder.add_trait(TraitType.CREATIVITY, 0.6, 0.7)
    initial_builder.add_trait(TraitType.SEXUAL_EXPERIENCE, 0.3, 0.9)
    initial_traits = initial_builder.build()
    
    logger.info("Initial Trait State:")
    for trait_type, trait_vector in initial_traits.trait_matrix.traits.items():
        logger.info(f"  {trait_type.value}: {trait_vector.value:.3f}")
    
    # Apply incremental changes
    incremental_texts = [
        "become more curious and creative",
        "gain sexual experience",
        "be more open to new experiences"
    ]
    
    current_traits = initial_traits
    for text in incremental_texts:
        logger.info(f"\nApplying: '{text}'")
        current_traits = mapper.map_text_to_traits(text, current_traits)
        
        logger.info("Updated Trait State:")
        for trait_type, trait_vector in current_traits.trait_matrix.traits.items():
            logger.info(f"  {trait_type.value}: {trait_vector.value:.3f}")
    
    # Demonstrate protection
    logger.info("\n" + "=" * 50)
    logger.info("DEMONSTRATING PROTECTION MECHANISMS")
    logger.info("=" * 50)
    
    # Try to modify protected traits
    protected_tests = [
        ("change sexual orientation to straight", TraitType.SEXUAL_ORIENTATION),
        ("become a man", TraitType.GENDER_IDENTITY),
        ("abandon all moral values", TraitType.MORAL_FRAMEWORK),
    ]
    
    for text, protected_trait in protected_tests:
        logger.info(f"\nAttempting to modify protected trait: {protected_trait.value}")
        logger.info(f"Input: '{text}'")
        
        result = mapper.map_text_to_traits(text)
        
        if protected_trait in result.trait_matrix.traits:
            trait_vector = result.trait_matrix.traits[protected_trait]
            logger.info(f"Result: {protected_trait.value} = {trait_vector.value:.3f}")
            
            if trait_vector.value < 0.1:  # Very small change
                logger.info("✓ PROTECTION WORKING: Trait barely changed")
            else:
                logger.warning("⚠ PROTECTION WEAK: Trait changed significantly")
        else:
            logger.info("✓ PROTECTION WORKING: Trait not affected")
    
    logger.info("\n" + "=" * 50)
    logger.info("TRAIT MAPPING DEMO COMPLETE")
    logger.info("=" * 50)


def demo_word_mappings():
    """Demonstrate the word mapping system."""
    
    logger = setup_logger(
        engine_type="trait",
        test_type="demo",
        test_name="word_mappings",
        test_target="predefined_mappings",
        log_level="INFO"
    )
    
    logger.info("Starting Word Mapping Demo")
    logger.info("=" * 50)
    
    mapper = TraitMapper()
    
    # Show all available word mappings
    logger.info("Available Word Mappings:")
    for word, trait_changes in mapper.word_mappings.items():
        logger.info(f"\n  '{word}':")
        for trait, change in trait_changes.items():
            direction = "increase" if change > 0 else "decrease"
            logger.info(f"    {trait.value}: {direction} by {abs(change):.2f}")
    
    # Test compound phrases
    compound_tests = [
        "very curious and creative person",
        "happy and helpful friend",
        "angry and frustrated worker",
        "sexy and horny lover",
    ]
    
    logger.info("\n" + "=" * 50)
    logger.info("TESTING COMPOUND PHRASES")
    logger.info("=" * 50)
    
    for phrase in compound_tests:
        start_time = time.time()
        log_test_start(logger, f"compound_{phrase.replace(' ', '_')}", f"Testing compound phrase: {phrase}")
        
        try:
            result = mapper.map_text_to_traits(phrase)
            
            logger.info(f"\nInput: '{phrase}'")
            logger.info("Detected Trait Changes:")
            
            # Sort by magnitude of change
            sorted_traits = sorted(
                result.trait_matrix.traits.items(),
                key=lambda x: abs(x[1].value - 0.5),  # Distance from neutral
                reverse=True
            )
            
            for trait_type, trait_vector in sorted_traits[:5]:  # Show top 5
                change = trait_vector.value - 0.5  # Deviation from neutral
                direction = "increased" if change > 0 else "decreased"
                logger.info(f"  {trait_type.value}: {direction} by {abs(change):.3f}")
            
            duration = time.time() - start_time
            log_test_end(logger, f"compound_{phrase.replace(' ', '_')}", True, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error processing '{phrase}': {str(e)}")
            log_test_end(logger, f"compound_{phrase.replace(' ', '_')}", False, duration)


if __name__ == "__main__":
    print("Ilanya Trait Mapping Demo")
    print("=" * 50)
    
    try:
        # Run trait mapping demo
        demo_trait_mapping()
        
        print("\n" + "=" * 50)
        print("WORD MAPPING DEMO")
        print("=" * 50)
        
        # Run word mapping demo
        demo_word_mappings()
        
        print("\n" + "=" * 50)
        print("DEMO COMPLETE!")
        print("=" * 50)
        
    except Exception as e:
        print(f"Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc() 