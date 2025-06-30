#!/usr/bin/env python3
"""
Ilanya Trait Engine - Simple Demo

Simple demonstration script for the Ilanya Trait Engine.
Shows basic usage of trait processing, neural network predictions,
and trait evolution without complex imports or dependencies.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

import torch
import torch.nn as nn
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
from utils.logging_utils import setup_logger, log_demo_start, log_demo_end


# Simple trait type definitions
class TraitType(Enum):
    OPENNESS = "openness"
    CREATIVITY = "creativity"
    ADAPTABILITY = "adaptability"
    EMOTIONAL_STABILITY = "emotional_stability"
    LEARNING_RATE = "learning_rate"


@dataclass
class TraitVector:
    # """Simple trait vector representation."""
    trait_type: TraitType
    value: float
    confidence: float
    
    def __post_init__(self):
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Trait value must be between 0 and 1, got {self.value}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")


class SimpleTraitEmbedding(nn.Module):
    # """Simple trait embedding layer."""
    
    def __init__(self, num_traits: int, embedding_dim: int):
        super().__init__()
        self.trait_embeddings = nn.Embedding(num_traits, embedding_dim)
        self.value_projection = nn.Linear(2, embedding_dim)  # value + confidence
        self.combined_projection = nn.Linear(embedding_dim * 2, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, trait_values: torch.Tensor, trait_confidences: torch.Tensor, 
                trait_indices: torch.Tensor) -> torch.Tensor:
        batch_size, num_traits = trait_values.shape
        
        # Get trait type embeddings
        trait_type_embeddings = self.trait_embeddings(trait_indices)
        
        # Project values and confidences
        value_conf = torch.stack([trait_values, trait_confidences], dim=-1)
        value_embeddings = self.value_projection(value_conf)
        
        # Combine embeddings
        combined = torch.cat([trait_type_embeddings, value_embeddings], dim=-1)
        embedded = self.combined_projection(combined)
        
        # Apply layer normalization
        embedded = self.layer_norm(embedded)
        
        return embedded


class SimpleTraitTransformer(nn.Module):
    # """Simplified trait transformer for demo."""
    
    def __init__(self, num_traits: int, embedding_dim: int, num_layers: int = 2):
        super().__init__()
        self.num_traits = num_traits
        self.embedding_dim = embedding_dim
        
        # Trait embedding
        self.trait_embedding = SimpleTraitEmbedding(num_traits, embedding_dim)
        
        # Transformer layers (simplified)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=4,
                dim_feedforward=embedding_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Output projections with activation functions
        self.trait_output = nn.Linear(embedding_dim, 2)  # value + confidence
        self.evolution_output = nn.Linear(embedding_dim, 1)  # evolution signal
        
        # Activation functions to ensure proper output ranges
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def forward(self, trait_values: torch.Tensor, trait_confidences: torch.Tensor,
                trait_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Embed traits
        embedded = self.trait_embedding(trait_values, trait_confidences, trait_indices)
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            embedded = layer(embedded)
        
        # Generate outputs with proper activation functions
        trait_raw = self.trait_output(embedded)
        trait_predictions = self.sigmoid(trait_raw)  # Ensure values are between 0 and 1
        
        evolution_signals = self.tanh(self.evolution_output(embedded)).squeeze(-1)  # Between -1 and 1
        
        return {
            'trait_predictions': trait_predictions,
            'evolution_signals': evolution_signals,
            'embeddings': embedded
        }


class SimpleTraitEngine:
    # """Simplified trait engine for demonstration."""
    
    def __init__(self, num_traits: int = 5, embedding_dim: int = 32):
        self.num_traits = num_traits
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize neural network
        self.neural_network = SimpleTraitTransformer(num_traits, embedding_dim)
        self.neural_network.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.neural_network.parameters(), lr=1e-4)
        
        # Loss function
        self.loss_fn = nn.MSELoss()
    
    def process_traits(self, traits: Dict[TraitType, TraitVector]) -> Dict[str, Any]:
        # """Process traits through the neural network."""
        # Convert to tensors
        trait_values = torch.zeros(self.num_traits)
        trait_confidences = torch.zeros(self.num_traits)
        trait_indices = torch.zeros(self.num_traits, dtype=torch.long)
        
        for i, (trait_type, trait_vector) in enumerate(traits.items()):
            trait_values[i] = trait_vector.value
            trait_confidences[i] = trait_vector.confidence
            trait_indices[i] = list(TraitType).index(trait_type)
        
        # Add batch dimension
        trait_values = trait_values.unsqueeze(0).to(self.device)
        trait_confidences = trait_confidences.unsqueeze(0).to(self.device)
        trait_indices = trait_indices.unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.neural_network(trait_values, trait_confidences, trait_indices)
        
        # Process outputs
        trait_predictions = outputs['trait_predictions'].cpu().numpy()[0]
        evolution_signals = outputs['evolution_signals'].cpu().numpy()[0]
        
        # Convert back to trait vectors (values are already between 0-1 due to sigmoid)
        predicted_traits = {}
        for i, trait_type in enumerate(traits.keys()):
            predicted_traits[trait_type] = TraitVector(
                trait_type=trait_type,
                value=float(trait_predictions[i, 0]),
                confidence=float(trait_predictions[i, 1])
            )
        
        return {
            'predicted_traits': predicted_traits,
            'evolution_signals': evolution_signals
        }
    
    def evolve_traits(self, traits: Dict[TraitType, TraitVector], 
                     experience: Dict[str, float]) -> Dict[TraitType, TraitVector]:
        # """Evolve traits based on experience."""
        # Process current traits
        results = self.process_traits(traits)
        
        # Apply evolution based on experience
        evolved_traits = {}
        evolution_rate = 0.01
        
        for trait_type, trait_vector in traits.items():
            # Get evolution signal
            trait_index = list(traits.keys()).index(trait_type)
            evolution_signal = results['evolution_signals'][trait_index]
            
            # Apply experience modifications
            if 'stress_level' in experience:
                stress = experience['stress_level']
                if trait_type == TraitType.EMOTIONAL_STABILITY:
                    evolution_signal -= stress * 0.1  # Stress decreases emotional stability
                elif trait_type == TraitType.ADAPTABILITY:
                    evolution_signal += stress * 0.05  # Stress might increase adaptability
            
            if 'success_rate' in experience:
                success = experience['success_rate']
                if trait_type == TraitType.LEARNING_RATE:
                    evolution_signal += success * 0.1  # Success increases learning rate
            
            # Calculate new value
            evolution_delta = evolution_signal * evolution_rate
            new_value = float(np.clip(trait_vector.value + evolution_delta, 0.0, 1.0))
            
            evolved_traits[trait_type] = TraitVector(
                trait_type=trait_type,
                value=new_value,
                confidence=trait_vector.confidence
            )
        
        return evolved_traits


def main():
    """Main demonstration function."""
    # Set up logger for this demo
    logger = setup_logger(
        engine_type="trait",
        test_type="demo",
        test_name="simple_trait_engine",
        test_target="neural_network",
        log_level="INFO"
    )
    
    start_time = time.time()
    log_demo_start(logger, "simple_trait_engine", 
                  "Simple demonstration of trait processing, neural network predictions, and trait evolution")
    
    try:
        print("🧠 Ilanya Trait Engine - Simple Demo")
        print("=" * 50)
    
        # Initialize the trait engine
        logger.info("Initializing Simple Trait Engine...")
        print("Initializing Simple Trait Engine...")
        trait_engine = SimpleTraitEngine(num_traits=5, embedding_dim=32)
    
    # Create sample traits
        logger.info("Creating sample traits...")
        print("Creating sample traits...")
        sample_traits = {
            TraitType.OPENNESS: TraitVector(TraitType.OPENNESS, 0.8, 0.9),
            TraitType.CREATIVITY: TraitVector(TraitType.CREATIVITY, 0.7, 0.8),
        TraitType.ADAPTABILITY: TraitVector(TraitType.ADAPTABILITY, 0.6, 0.7),
            TraitType.EMOTIONAL_STABILITY: TraitVector(TraitType.EMOTIONAL_STABILITY, 0.9, 0.8),
            TraitType.LEARNING_RATE: TraitVector(TraitType.LEARNING_RATE, 0.8, 0.9)
    }
    
        logger.info(f"Created {len(sample_traits)} sample traits")
        print(f"Created {len(sample_traits)} sample traits:")
        for trait_type, trait_vector in sample_traits.items():
            print(f"  {trait_type.value}: {trait_vector.value:.2f} (confidence: {trait_vector.confidence:.2f})")
        
        # Process traits through the neural network
        logger.info("Processing traits through neural network...")
        print("\n🔄 Processing traits through neural network...")
        results = trait_engine.process_traits(sample_traits)
        
        logger.info("Neural network processing completed")
        print("Neural network processing completed!")
    
        # Display predictions
        print("\n📊 Trait Predictions:")
        predicted_traits = results['predicted_traits']
        for trait_type, trait_vector in predicted_traits.items():
            print(f"  {trait_type.value}: {trait_vector.value:.3f} (confidence: {trait_vector.confidence:.3f})")
        
        # Display evolution signals
        print("\n🔄 Evolution Signals:")
        evolution_signals = results['evolution_signals']
        for i, (trait_type, _) in enumerate(sample_traits.items()):
            signal = evolution_signals[i]
            direction = "↗️" if signal > 0 else "↘️" if signal < 0 else "➡️"
            print(f"  {trait_type.value}: {signal:+.3f} {direction}")
    
        # Demonstrate trait evolution
        logger.info("Demonstrating trait evolution...")
        print("\n🌱 Demonstrating trait evolution...")
        
        # Create experience that should influence traits
    experience = {
            "creative_activity": 0.8,
            "learning_opportunity": 0.9,
            "stressful_situation": 0.3,
            "social_interaction": 0.6
    }
    
        logger.info(f"Applying experience: {experience}")
        print(f"Applying experience: {experience}")
        
        # Evolve traits based on experience
        evolved_traits = trait_engine.evolve_traits(sample_traits, experience)
        
        logger.info("Trait evolution completed")
        print("Trait evolution completed!")
        
        # Display evolved traits
        print("\n📈 Evolved Traits:")
        for trait_type, original_trait in sample_traits.items():
            evolved_trait = evolved_traits[trait_type]
            change = evolved_trait.value - original_trait.value
            direction = "↗️" if change > 0 else "↘️" if change < 0 else "➡️"
            print(f"  {trait_type.value}: {original_trait.value:.3f} → {evolved_trait.value:.3f} "
                  f"({change:+.3f}) {direction}")
        
        # Demonstrate multiple evolution cycles
        logger.info("Running multiple evolution cycles...")
        print("\n🔄 Running multiple evolution cycles...")
        
        current_traits = evolved_traits.copy()
        for cycle in range(1, 4):
            logger.info(f"Evolution cycle {cycle}")
            print(f"\nCycle {cycle}:")
            
            # Vary experience slightly each cycle
            cycle_experience = {
                k: float(v + np.random.normal(0, 0.1)) for k, v in experience.items()
            }
            # Clamp to valid range
            cycle_experience = {k: max(0.0, min(1.0, v)) for k, v in cycle_experience.items()}
            
            print(f"  Experience: {cycle_experience}")
            
            # Evolve traits
            current_traits = trait_engine.evolve_traits(current_traits, cycle_experience)
            
            # Show some key trait changes
            print("  Key changes:")
            for trait_type in [TraitType.CREATIVITY, TraitType.LEARNING_RATE]:
                if trait_type in current_traits:
                    print(f"    {trait_type.value}: {current_traits[trait_type].value:.3f}")
        
        # Final summary
        logger.info("Demo completed successfully")
        print(f"\n✅ Demo completed successfully!")
        print(f"📊 Final Statistics:")
        print(f"  Total traits processed: {len(sample_traits)}")
        print(f"  Evolution cycles completed: 3")
        print(f"  Neural network parameters: {sum(p.numel() for p in trait_engine.neural_network.parameters())}")
        
        duration = time.time() - start_time
        log_demo_end(logger, "simple_trait_engine", duration)
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Demo failed with error: {str(e)}")
        log_demo_end(logger, "simple_trait_engine", duration)
        raise


if __name__ == "__main__":
    main() 