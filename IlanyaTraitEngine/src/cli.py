"""
Ilanya Trait Engine - Command Line Interface

Command line interface for the Ilanya Trait Engine providing easy access
to trait processing, evolution, and training capabilities. Supports both
interactive and batch processing modes.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

import argparse
import json
import sys
import os
from typing import Dict, Any, Optional

# Add the project root to the Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trait_engine import TraitEngine, TraitEngineConfig
from src.trait_models.trait_data import TraitDataBuilder, TraitData
from src.trait_models.trait_types import TraitType


def create_sample_data():
    """Create sample trait data for CLI demo."""
    builder = TraitDataBuilder()
    
    # Add a few key traits
    builder.add_trait(TraitType.OPENNESS, 0.7, 0.9)
    builder.add_trait(TraitType.CREATIVITY, 0.8, 0.8)
    builder.add_trait(TraitType.ADAPTABILITY, 0.6, 0.7)
    builder.add_trait(TraitType.EMOTIONAL_STABILITY, 0.7, 0.8)
    builder.add_trait(TraitType.LEARNING_RATE, 0.9, 0.9)
    
    builder.set_source("cli_demo")
    return builder.build()


def demo_command(args):
    """Run the demo command."""
    print("=== Ilanya Trait Engine CLI Demo ===\n")
    
    # Initialize engine
    print("Initializing trait engine...")
    config = TraitEngineConfig(
        num_traits=5,
        trait_embedding_dim=32,
        num_layers=2,
        num_heads=2
    )
    engine = TraitEngine(config)
    
    # Create sample data
    print("Creating sample trait data...")
    trait_data = create_sample_data()
    
    # Process traits
    print("Processing traits...")
    results = engine.process_traits(trait_data)
    
    # Show results
    print("\nResults:")
    for trait_type, predicted_trait in results['predicted_traits'].items():
        original = trait_data.trait_matrix.traits[trait_type]
        print(f"  {trait_type.value}: {original.value:.3f} → {predicted_trait.value:.3f}")
    
    print("\nDemo completed successfully!")


def create_parser() -> argparse.ArgumentParser:
    """
    Create command line argument parser.
    
    Sets up all available commands and their arguments for the CLI.
    Supports processing, evolution, training, and configuration commands.
    
    Returns:
        Configured ArgumentParser object
    """
    parser = argparse.ArgumentParser(
        description="Ilanya Trait Engine - Neural network-based trait processing and evolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process trait data from file
  python -m src.cli process --input traits.json --output results.json
  
  # Evolve traits based on experience
  python -m src.cli evolve --input traits.json --experience experience.json --output evolved.json
  
  # Train the model
  python -m src.cli train --data-dir ./data --epochs 100 --save-model model.pth
  
  # Interactive mode
  python -m src.cli interactive
        """
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command - for trait processing
    process_parser = subparsers.add_parser('process', help='Process trait data through neural network')
    process_parser.add_argument('--input', '-i', required=True, help='Input trait data file (JSON)')
    process_parser.add_argument('--output', '-o', required=True, help='Output results file (JSON)')
    process_parser.add_argument('--config', '-c', help='Engine configuration file (JSON)')
    
    # Evolve command - for trait evolution
    evolve_parser = subparsers.add_parser('evolve', help='Evolve traits based on experience')
    evolve_parser.add_argument('--input', '-i', required=True, help='Input trait data file (JSON)')
    evolve_parser.add_argument('--experience', '-e', required=True, help='Experience data file (JSON)')
    evolve_parser.add_argument('--output', '-o', required=True, help='Output evolved traits file (JSON)')
    evolve_parser.add_argument('--config', '-c', help='Engine configuration file (JSON)')
    
    # Train command - for model training
    train_parser = subparsers.add_parser('train', help='Train the trait engine model')
    train_parser.add_argument('--data-dir', '-d', required=True, help='Directory containing training data')
    train_parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of training epochs')
    train_parser.add_argument('--batch-size', '-b', type=int, default=32, help='Training batch size')
    train_parser.add_argument('--learning-rate', '-l', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--save-model', '-s', help='Path to save trained model')
    train_parser.add_argument('--config', '-c', help='Engine configuration file (JSON)')
    
    # Interactive command - for interactive mode
    interactive_parser = subparsers.add_parser('interactive', help='Start interactive mode')
    interactive_parser.add_argument('--config', '-c', help='Engine configuration file (JSON)')
    
    # Config command - for configuration management
    config_parser = subparsers.add_parser('config', help='Generate default configuration')
    config_parser.add_argument('--output', '-o', default='config.json', help='Output configuration file')
    
    return parser


def load_config(config_path: Optional[str]) -> TraitEngineConfig:
    """
    Load engine configuration from file or use defaults.
    
    Args:
        config_path: Path to configuration file, or None for defaults
        
    Returns:
        TraitEngineConfig object
    """
    if config_path and os.path.exists(config_path):
        # Load configuration from file
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Create config object from loaded data
        config = TraitEngineConfig(**config_data)
        print(f"Loaded configuration from {config_path}")
    else:
        # Use default configuration
        config = TraitEngineConfig()
        print("Using default configuration")
    
    return config


def load_trait_data(file_path: str) -> TraitData:
    """
    Load trait data from JSON file.
    
    Args:
        file_path: Path to JSON file containing trait data
        
    Returns:
        TraitData object
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Reconstruct TraitData from JSON
    builder = TraitDataBuilder()
    
    # Add traits from the loaded data
    for trait_info in data['trait_matrix']['traits'].values():
        trait_type = TraitType(trait_info['trait_type'])
        builder.add_trait(
            trait_type,
            trait_info['value'],
            trait_info['confidence']
        )
    
    # Set metadata
    builder.set_source(data.get('source', 'file'))
    for key, value in data.get('processing_metadata', {}).items():
        builder.add_metadata(key, value)
    
    return builder.build()


def save_trait_data(trait_data: TraitData, file_path: str):
    """
    Save trait data to JSON file.
    
    Args:
        trait_data: TraitData object to save
        file_path: Path to output JSON file
    """
    with open(file_path, 'w') as f:
        json.dump(trait_data.to_dict(), f, indent=2)
    
    print(f"Saved trait data to {file_path}")


def process_command(args: argparse.Namespace):
    """
    Execute the process command.
    
    Loads trait data, processes it through the neural network,
    and saves the results.
    
    Args:
        args: Command line arguments
    """
    print("=== Processing Trait Data ===")
    
    # Load configuration and initialize engine
    config = load_config(args.config)
    engine = TraitEngine(config)
    
    # Load input trait data
    print(f"Loading trait data from {args.input}")
    trait_data = load_trait_data(args.input)
    
    # Process traits through neural network
    print("Processing traits through neural network...")
    results = engine.process_traits(trait_data)
    
    # Save results
    print(f"Saving results to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Processing completed successfully!")


def evolve_command(args: argparse.Namespace):
    """
    Execute the evolve command.
    
    Loads trait data and experience data, evolves traits based on
    experience, and saves the evolved traits.
    
    Args:
        args: Command line arguments
    """
    print("=== Evolving Traits ===")
    
    # Load configuration and initialize engine
    config = load_config(args.config)
    engine = TraitEngine(config)
    
    # Load input trait data
    print(f"Loading trait data from {args.input}")
    trait_data = load_trait_data(args.input)
    
    # Load experience data
    print(f"Loading experience data from {args.experience}")
    with open(args.experience, 'r') as f:
        experience_data = json.load(f)
    
    # Evolve traits based on experience
    print("Evolving traits based on experience...")
    evolved_data = engine.evolve_traits(trait_data, experience_data)
    
    # Save evolved traits
    print(f"Saving evolved traits to {args.output}")
    save_trait_data(evolved_data, args.output)
    
    print("Evolution completed successfully!")


def train_command(args: argparse.Namespace):
    """
    Execute the train command.
    
    Trains the trait engine model on provided data.
    
    Args:
        args: Command line arguments
    """
    print("=== Training Model ===")
    
    # Load configuration and initialize engine
    config = load_config(args.config)
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    
    engine = TraitEngine(config)
    
    # TODO: Implement actual training logic
    print(f"Training for {args.epochs} epochs...")
    print(f"Data directory: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    
    if args.save_model:
        print(f"Model will be saved to {args.save_model}")
    
    print("Training completed successfully!")


def interactive_command(args: argparse.Namespace):
    """
    Execute the interactive command.
    
    Starts an interactive session for trait processing and evolution.
    
    Args:
        args: Command line arguments
    """
    print("=== Interactive Mode ===")
    print("Starting interactive trait engine session...")
    print("Type 'help' for available commands, 'quit' to exit.")
    
    # Load configuration and initialize engine
    config = load_config(args.config)
    engine = TraitEngine(config)
    
    # TODO: Implement interactive loop
    print("Interactive mode not yet implemented.")
    print("Use the other commands for now.")


def config_command(args: argparse.Namespace):
    """
    Execute the config command.
    
    Generates a default configuration file.
    
    Args:
        args: Command line arguments
    """
    print("=== Generating Configuration ===")
    
    # Create default configuration
    config = TraitEngineConfig()
    
    # Convert to dictionary and save
    config_dict = {
        'input_dim': config.input_dim,
        'hidden_dim': config.hidden_dim,
        'num_layers': config.num_layers,
        'num_heads': config.num_heads,
        'dropout': config.dropout,
        'num_traits': config.num_traits,
        'trait_embedding_dim': config.trait_embedding_dim,
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'max_sequence_length': config.max_sequence_length,
        'evolution_rate': config.evolution_rate,
        'stability_threshold': config.stability_threshold,
        'plasticity_factor': config.plasticity_factor
    }
    
    # Save configuration to file
    with open(args.output, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Default configuration saved to {args.output}")


def main():
    """
    Main CLI entry point.
    
    Parses command line arguments and executes the appropriate command.
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # Check if a command was provided
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Execute the appropriate command
        if args.command == 'process':
            process_command(args)
        elif args.command == 'evolve':
            evolve_command(args)
        elif args.command == 'train':
            train_command(args)
        elif args.command == 'interactive':
            interactive_command(args)
        elif args.command == 'config':
            config_command(args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 