"""
Ilanya Neural Trait Mapping Training Demo

Demonstrates how to use the massive word dataset to generate training data
and train the neural trait mapping system.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import json
import time
from typing import Dict, List

from IlanyaTraitEngine.src.trait_mapping.dataset_generator import TraitMappingDatasetGenerator
from IlanyaTraitEngine.src.trait_mapping.trait_mapper import TraitMapper, TraitMappingConfig
from IlanyaTraitEngine.src.trait_models.trait_types import TraitType
from utils.logging_utils import setup_logger, log_test_start, log_test_end


def demo_dataset_generation():
    """Demonstrate dataset generation using the massive word list."""
    
    logger = setup_logger(
        engine_type="trait",
        test_type="demo",
        test_name="dataset_generation",
        test_target="massive_word_dataset",
        log_level="INFO"
    )
    
    logger.info("Starting Neural Trait Mapping Training Demo")
    logger.info("=" * 60)
    
    # Initialize dataset generator with the massive word list
    logger.info("Loading massive word dataset...")
    generator = TraitMappingDatasetGenerator(
        words_file_path="IlanyaTraitEngine/src/utils/english-words/words.txt"
    )
    
    logger.info(f"Loaded {len(generator.words)} words from dataset")
    
    # Generate different types of training data
    logger.info("\nGenerating training datasets...")
    
    # 1. Word-based dataset
    logger.info("1. Generating word-based dataset...")
    word_dataset = generator.generate_synthetic_dataset(num_examples=3000)
    
    # 2. Phrase-based dataset
    logger.info("2. Generating phrase-based dataset...")
    phrase_dataset = generator.create_phrase_dataset(num_phrases=2000)
    
    # 3. Combined dataset
    combined_dataset = word_dataset + phrase_dataset
    
    logger.info(f"Generated {len(combined_dataset)} total training examples")
    logger.info(f"  - Word examples: {len(word_dataset)}")
    logger.info(f"  - Phrase examples: {len(phrase_dataset)}")
    
    # Save datasets
    logger.info("\nSaving datasets...")
    generator.save_dataset(word_dataset, "data/word_trait_mappings.json")
    generator.save_dataset(phrase_dataset, "data/phrase_trait_mappings.json")
    generator.save_dataset(combined_dataset, "data/combined_trait_mappings.json")
    
    # Show sample examples
    logger.info("\nSample training examples:")
    for i, example in enumerate(combined_dataset[:10]):
        logger.info(f"{i+1}. Text: '{example['text']}'")
        logger.info(f"   Traits: {example['traits']}")
        logger.info("")  # Empty line for spacing
    
    return combined_dataset


def demo_neural_training(dataset: List[Dict[str, any]]):
    """Demonstrate training the neural trait mapping system."""
    
    logger = setup_logger(
        engine_type="trait",
        test_type="demo",
        test_name="neural_training",
        test_target="trait_mapper_training",
        log_level="INFO"
    )
    
    logger.info("Starting Neural Trait Mapper Training")
    logger.info("=" * 60)
    
    # Initialize trait mapper
    config = TraitMappingConfig(
        protection_factor=0.01,
        confidence_threshold=0.6,
        learning_rate=0.001,
        batch_size=32,
        epochs=10
    )
    
    trait_mapper = TraitMapper(config)
    
    # Prepare training data
    logger.info("Preparing training data...")
    
    # Split dataset into training and validation
    split_idx = int(len(dataset) * 0.8)
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]
    
    logger.info(f"Training examples: {len(train_data)}")
    logger.info(f"Validation examples: {len(val_data)}")
    
    # Train the model
    logger.info("\nStarting training...")
    start_time = time.time()
    
    try:
        training_history = trait_mapper.train_on_dataset(
            train_data=train_data,
            val_data=val_data,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate
        )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Show training results
        logger.info("\nTraining Results:")
        logger.info(f"Final training loss: {training_history['train_loss'][-1]:.4f}")
        logger.info(f"Final validation loss: {training_history['val_loss'][-1]:.4f}")
        
        # Save the trained model
        logger.info("\nSaving trained model...")
        trait_mapper.save_model("models/trained_trait_mapper.pth")
        
        return trait_mapper, training_history
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return None, None


def demo_trained_model_evaluation(trait_mapper: TraitMapper):
    """Demonstrate evaluating the trained model."""
    
    logger = setup_logger(
        engine_type="trait",
        test_type="demo",
        test_name="model_evaluation",
        test_target="trained_model_testing",
        log_level="INFO"
    )
    
    logger.info("Evaluating Trained Neural Trait Mapper")
    logger.info("=" * 60)
    
    # Test cases
    test_inputs = [
        "I am feeling curious and excited about learning new things",
        "I want to be more creative and artistic",
        "I'm feeling horny and want to explore sexually",
        "I need to work on my emotional stability",
        "I want to be more social and outgoing",
        "I love helping others and being kind",
        "I want to be more organized and responsible",
        "I feel confident and optimistic about the future",
        "I need to focus and concentrate better",
        "I want to take risks and be more adventurous",
    ]
    
    logger.info("Testing trained model with various inputs:")
    
    for i, text_input in enumerate(test_inputs, 1):
        logger.info(f"\n{i}. Input: '{text_input}'")
        
        try:
            # Get trait predictions
            trait_predictions = trait_mapper.map_text_to_traits(text_input)
            
            logger.info("Predicted trait modifications:")
            for trait_type, trait_vector in trait_predictions.trait_matrix.traits.items():
                if abs(trait_vector.value) > 0.01:  # Only show significant changes
                    direction = "increased" if trait_vector.value > 0 else "decreased"
                    logger.info(f"  {trait_type.value}: {direction} by {abs(trait_vector.value):.3f} "
                              f"(confidence: {trait_vector.confidence:.3f})")
            
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}")


def demo_comparison_with_baseline():
    """Compare neural approach with baseline hardcoded approach."""
    
    logger = setup_logger(
        engine_type="trait",
        test_type="demo",
        test_name="comparison",
        test_target="neural_vs_baseline",
        log_level="INFO"
    )
    
    logger.info("Comparing Neural vs Baseline Trait Mapping")
    logger.info("=" * 60)
    
    # Test inputs
    test_inputs = [
        "I am feeling curious",
        "I want to be more creative",
        "I'm feeling horny",
        "I need emotional stability",
        "I want to be social",
    ]
    
    # Initialize both approaches
    neural_mapper = TraitMapper(TraitMappingConfig())
    
    logger.info("Testing both approaches:")
    
    for text_input in test_inputs:
        logger.info(f"\nInput: '{text_input}'")
        
        try:
            # Neural approach
            neural_result = neural_mapper.map_text_to_traits(text_input)
            
            logger.info("Neural predictions:")
            for trait_type, trait_vector in neural_result.trait_matrix.traits.items():
                if abs(trait_vector.value) > 0.01:
                    direction = "increased" if trait_vector.value > 0 else "decreased"
                    logger.info(f"  {trait_type.value}: {direction} by {abs(trait_vector.value):.3f}")
            
        except Exception as e:
            logger.error(f"Error: {str(e)}")


if __name__ == "__main__":
    print("Ilanya Neural Trait Mapping Training Demo")
    print("=" * 60)
    
    try:
        # Step 1: Generate dataset from massive word list
        print("\nSTEP 1: DATASET GENERATION")
        print("-" * 30)
        dataset = demo_dataset_generation()
        
        # Step 2: Train neural model
        print("\nSTEP 2: NEURAL TRAINING")
        print("-" * 30)
        trait_mapper, training_history = demo_neural_training(dataset)
        
        if trait_mapper:
            # Step 3: Evaluate trained model
            print("\nSTEP 3: MODEL EVALUATION")
            print("-" * 30)
            demo_trained_model_evaluation(trait_mapper)
            
            # Step 4: Comparison
            print("\nSTEP 4: COMPARISON")
            print("-" * 30)
            demo_comparison_with_baseline()
        
        print("\n" + "=" * 60)
        print("NEURAL TRAIT MAPPING TRAINING DEMO COMPLETE!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc() 