#!/usr/bin/env python3
"""
Simple test of the dataset generator with the massive word list.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'IlanyaTraitEngine'))

from IlanyaTraitEngine.src.trait_mapping.dataset_generator import TraitMappingDatasetGenerator

def main():
    print("Testing Dataset Generator with Massive Word List")
    print("=" * 60)
    
    # Initialize generator
    print("Loading massive word dataset...")
    generator = TraitMappingDatasetGenerator(
        words_file_path="IlanyaTraitEngine/src/utils/english-words/words.txt"
    )
    
    print(f"✓ Loaded {len(generator.words)} words from dataset")
    
    # Generate some examples
    print("\nGenerating training examples...")
    dataset = generator.generate_synthetic_dataset(num_examples=50)
    
    print(f"✓ Generated {len(dataset)} training examples")
    
    # Show some examples
    print("\nSample training examples:")
    for i, example in enumerate(dataset[:10]):
        print(f"{i+1}. Text: '{example['text']}'")
        print(f"   Traits: {example['traits']}")
        print()
    
    # Generate phrase examples
    print("Generating phrase examples...")
    phrases = generator.create_phrase_dataset(num_phrases=20)
    
    print(f"✓ Generated {len(phrases)} phrase examples")
    
    print("\nSample phrase examples:")
    for i, example in enumerate(phrases[:5]):
        print(f"{i+1}. Text: '{example['text']}'")
        print(f"   Traits: {example['traits']}")
        print()
    
    # Save a small dataset
    print("Saving sample dataset...")
    generator.save_dataset(dataset[:100], "data/sample_trait_mappings.json")
    print("✓ Dataset saved to data/sample_trait_mappings.json")
    
    print("\n" + "=" * 60)
    print("DATASET GENERATION TEST COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main() 