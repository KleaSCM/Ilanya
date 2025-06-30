"""
Ilanya Trait Engine - Trait Mapping Neural Network

Natural language to trait mapping system that converts text input into
trait modifications while respecting protection levels and applying
appropriate weights to different traits.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer, AutoModel
import re

from ..trait_models.trait_types import (
    TraitType, TraitCategory, PERMANENTLY_PROTECTED_TRAITS,
    PARTIALLY_PROTECTED_TRAITS, FULLY_EVOLVABLE_TRAITS
)
from ..trait_models.trait_data import TraitVector, TraitMatrix, TraitData, TraitDataBuilder


@dataclass
class TraitMappingConfig:
    """Configuration for the trait mapping neural network."""
    
    # Model configuration
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # Base embedding model
    embedding_dim: int = 384  # Dimension of sentence embeddings
    hidden_dim: int = 512  # Hidden layer dimension
    num_layers: int = 3  # Number of transformer layers
    num_heads: int = 8  # Number of attention heads
    dropout: float = 0.1  # Dropout rate
    
    # Trait mapping configuration
    max_sequence_length: int = 128  # Maximum input sequence length
    num_traits: int = len(TraitType)  # Total number of traits
    trait_embedding_dim: int = 64  # Trait embedding dimension
    
    # Learning configuration
    learning_rate: float = 1e-4  # Learning rate
    batch_size: int = 16  # Batch size for training
    
    # Protection configuration
    protection_factor: float = 0.01  # How much protected traits can change
    confidence_threshold: float = 0.7  # Minimum confidence for trait changes


class TraitMappingNetwork(nn.Module):
    """
    Neural network for mapping natural language to trait modifications.
    
    Takes text input and produces trait modification signals that can be
    applied to the existing trait system while respecting protection levels.
    """
    
    def __init__(self, config: TraitMappingConfig):
        super().__init__()
        self.config = config
        
        # Load pre-trained sentence transformer for text embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.text_encoder = AutoModel.from_pretrained(config.model_name)
        
        # Freeze the pre-trained encoder (we'll fine-tune only our layers)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Trait embedding layer - learnable representations for each trait
        self.trait_embeddings = nn.Embedding(config.num_traits, config.trait_embedding_dim)
        
        # Attention mechanism to align text with traits
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Trait prediction layers
        self.trait_predictor = nn.Sequential(
            nn.Linear(config.embedding_dim + config.trait_embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 3)  # [value_change, confidence, protection_level]
        )
        
        # Protection layer - ensures protected traits don't change too much
        self.protection_layer = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        
    def forward(self, text_inputs: List[str]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the trait mapping network.
        
        Args:
            text_inputs: List of text strings to process
            
        Returns:
            Dictionary containing trait modifications and metadata
        """
        batch_size = len(text_inputs)
        
        # Tokenize and encode text inputs
        encoded = self.tokenizer(
            text_inputs,
            padding=True,
            truncation=True,
            max_length=self.config.max_sequence_length,
            return_tensors="pt"
        )
        
        # Get text embeddings
        with torch.no_grad():
            text_outputs = self.text_encoder(**encoded)
            text_embeddings = text_outputs.last_hidden_state  # [batch_size, seq_len, embedding_dim]
        
        # Get trait embeddings for all traits
        trait_indices = torch.arange(self.config.num_traits).unsqueeze(0).expand(batch_size, -1)
        trait_embeddings = self.trait_embeddings(trait_indices)  # [batch_size, num_traits, trait_embedding_dim]
        
        # Cross-attention between text and traits
        text_embeddings = self.layer_norm(text_embeddings)
        attended_text, _ = self.cross_attention(
            query=text_embeddings,
            key=text_embeddings,
            value=text_embeddings
        )
        
        # Pool text embeddings (mean pooling)
        pooled_text = attended_text.mean(dim=1)  # [batch_size, embedding_dim]
        
        # Expand pooled text for each trait
        expanded_text = pooled_text.unsqueeze(1).expand(-1, self.config.num_traits, -1)
        
        # Concatenate text and trait embeddings
        combined = torch.cat([expanded_text, trait_embeddings], dim=-1)
        
        # Predict trait modifications
        trait_predictions = self.trait_predictor(combined)  # [batch_size, num_traits, 3]
        
        # Extract components
        value_changes = trait_predictions[:, :, 0]  # How much to change trait value
        confidences = torch.sigmoid(trait_predictions[:, :, 1])  # Confidence in prediction
        protection_levels = torch.sigmoid(trait_predictions[:, :, 2])  # Protection level
        
        # Apply protection constraints
        protected_mask = self._get_protection_mask()
        value_changes = value_changes * (1 - protected_mask * 0.99)  # Reduce changes for protected traits
        
        return {
            'value_changes': value_changes,
            'confidences': confidences,
            'protection_levels': protection_levels,
            'text_embeddings': pooled_text,
            'trait_embeddings': trait_embeddings
        }
    
    def _get_protection_mask(self) -> torch.Tensor:
        """Get mask for protected traits."""
        mask = torch.zeros(self.config.num_traits)
        
        # Mark permanently protected traits
        for trait in PERMANENTLY_PROTECTED_TRAITS:
            mask[list(TraitType).index(trait)] = 1.0
        
        # Mark partially protected traits with lower protection
        for trait in PARTIALLY_PROTECTED_TRAITS:
            mask[list(TraitType).index(trait)] = 0.5
        
        return mask


class TraitMapper:
    """
    High-level interface for mapping natural language to trait modifications.
    
    Provides easy-to-use methods for converting text input into trait
    modifications that can be applied to the existing trait system.
    """
    
    def __init__(self, config: Optional[TraitMappingConfig] = None):
        self.config = config or TraitMappingConfig()
        self.network = TraitMappingNetwork(self.config)
        
        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        
        # Pre-defined word-trait mappings for common cases
        self.word_mappings = self._initialize_word_mappings()
        
    def _initialize_word_mappings(self) -> Dict[str, Dict[TraitType, float]]:
        """Initialize common word-to-trait mappings."""
        return {
            # Curiosity and exploration
            "curious": {TraitType.OPENNESS: 0.3, TraitType.CREATIVITY: 0.2},
            "explore": {TraitType.OPENNESS: 0.25, TraitType.ADAPTABILITY: 0.15},
            "adventure": {TraitType.OPENNESS: 0.3, TraitType.RISK_TAKING: 0.2},
            
            # Sexual content (your example)
            "lick": {TraitType.SEXUAL_EXPERIENCE: 0.2, TraitType.SEXUAL_COMFORT_LEVEL: 0.1},
            "pussy": {TraitType.SEXUAL_EXPERIENCE: 0.15, TraitType.SEXUAL_COMFORT_LEVEL: 0.1},
            "horny": {TraitType.SEXUAL_EXPERIENCE: 0.25, TraitType.SEXUAL_COMFORT_LEVEL: 0.2},
            "sexy": {TraitType.SEXUAL_EXPERIENCE: 0.2, TraitType.SEXUAL_COMFORT_LEVEL: 0.15},
            
            # Emotional states
            "happy": {TraitType.OPTIMISM: 0.3, TraitType.EMOTIONAL_STABILITY: 0.2},
            "sad": {TraitType.OPTIMISM: -0.2, TraitType.EMOTIONAL_STABILITY: -0.1},
            "angry": {TraitType.EMOTIONAL_STABILITY: -0.2, TraitType.AGREEABLENESS: -0.1},
            "excited": {TraitType.EXTRAVERSION: 0.25, TraitType.OPTIMISM: 0.2},
            
            # Social interactions
            "talk": {TraitType.EXTRAVERSION: 0.2, TraitType.SOCIAL_SKILLS: 0.15},
            "friend": {TraitType.AGREEABLENESS: 0.2, TraitType.EMPATHY: 0.15},
            "help": {TraitType.AGREEABLENESS: 0.25, TraitType.EMPATHY: 0.2},
            
            # Cognitive activities
            "think": {TraitType.ANALYTICAL_THINKING: 0.2, TraitType.ATTENTION_SPAN: 0.15},
            "learn": {TraitType.LEARNING_RATE: 0.25, TraitType.ATTENTION_SPAN: 0.2},
            "create": {TraitType.CREATIVITY: 0.3, TraitType.OPENNESS: 0.2},
            
            # Behavioral patterns
            "work": {TraitType.CONSCIENTIOUSNESS: 0.2, TraitType.PERSISTENCE: 0.15},
            "plan": {TraitType.CONSCIENTIOUSNESS: 0.25, TraitType.ANALYTICAL_THINKING: 0.15},
            "lead": {TraitType.LEADERSHIP: 0.3, TraitType.EXTRAVERSION: 0.2},
        }
    
    def map_text_to_traits(self, text: str, current_traits: Optional[TraitData] = None) -> TraitData:
        """
        Map natural language text to trait modifications.
        
        Args:
            text: Input text to process
            current_traits: Current trait state (optional, for incremental changes)
            
        Returns:
            TraitData with modifications applied
        """
        # First check for exact word matches
        word_modifications = self._apply_word_mappings(text)
        
        # Then use neural network for more complex understanding
        nn_modifications = self._apply_neural_mapping(text)
        
        # Combine both approaches
        combined_modifications = self._combine_modifications(word_modifications, nn_modifications)
        
        # Apply modifications to current traits or create new ones
        if current_traits:
            return self._apply_incremental_changes(current_traits, combined_modifications)
        else:
            return self._create_new_trait_data(combined_modifications)
    
    def _apply_word_mappings(self, text: str) -> Dict[TraitType, float]:
        """Apply pre-defined word mappings."""
        modifications = {}
        text_lower = text.lower()
        
        for word, trait_changes in self.word_mappings.items():
            if word in text_lower:
                for trait, change in trait_changes.items():
                    if trait in modifications:
                        modifications[trait] += change
                    else:
                        modifications[trait] = change
        
        return modifications
    
    def _apply_neural_mapping(self, text: str) -> Dict[TraitType, float]:
        """Apply neural network mapping."""
        self.network.eval()
        
        with torch.no_grad():
            outputs = self.network([text])
            
            # Get value changes and confidences
            value_changes = outputs['value_changes'][0].cpu().numpy()
            confidences = outputs['confidences'][0].cpu().numpy()
            
            # Apply confidence threshold
            mask = confidences > self.config.confidence_threshold
            
            modifications = {}
            for i, trait in enumerate(TraitType):
                if mask[i]:
                    modifications[trait] = float(value_changes[i])
        
        return modifications
    
    def _combine_modifications(self, word_mods: Dict[TraitType, float], 
                             nn_mods: Dict[TraitType, float]) -> Dict[TraitType, float]:
        """Combine word-based and neural network modifications."""
        combined = word_mods.copy()
        
        for trait, change in nn_mods.items():
            if trait in combined:
                # Average the changes
                combined[trait] = (combined[trait] + change) / 2
            else:
                combined[trait] = change
        
        return combined
    
    def _apply_incremental_changes(self, current_traits: TraitData, 
                                 modifications: Dict[TraitType, float]) -> TraitData:
        """Apply modifications to existing trait data."""
        builder = TraitDataBuilder()
        
        # Copy existing traits with modifications
        for trait_type, trait_vector in current_traits.trait_matrix.traits.items():
            current_value = trait_vector.value
            modification = modifications.get(trait_type, 0.0)
            
            # Apply protection constraints
            if trait_type in PERMANENTLY_PROTECTED_TRAITS:
                modification *= self.config.protection_factor
            elif trait_type in PARTIALLY_PROTECTED_TRAITS:
                modification *= self.config.protection_factor * 2
            
            # Calculate new value
            new_value = max(0.0, min(1.0, current_value + modification))
            
            builder.add_trait(trait_type, new_value, trait_vector.confidence)
        
        # Add new traits that weren't in the original data
        for trait_type, modification in modifications.items():
            if trait_type not in current_traits.trait_matrix.traits:
                # Start with a moderate value and apply the modification
                base_value = 0.5
                new_value = max(0.0, min(1.0, base_value + modification))
                builder.add_trait(trait_type, new_value, 0.8)  # Moderate confidence
        
        return builder.build()
    
    def _create_new_trait_data(self, modifications: Dict[TraitType, float]) -> TraitData:
        """Create new trait data from modifications."""
        builder = TraitDataBuilder()
        
        for trait_type, modification in modifications.items():
            # Start with a moderate value and apply the modification
            base_value = 0.5
            new_value = max(0.0, min(1.0, base_value + modification))
            builder.add_trait(trait_type, new_value, 0.8)  # Moderate confidence
        
        return builder.build()
    
    def train_on_examples(self, examples: List[Tuple[str, TraitData]]):
        """
        Train the network on example text-trait pairs.
        
        Args:
            examples: List of (text, expected_traits) pairs
        """
        self.network.train()
        
        # This would implement training logic
        # For now, we'll use the word mappings as a simple approach
        pass


# Example usage and testing
if __name__ == "__main__":
    # Create trait mapper
    mapper = TraitMapper()
    
    # Test with your examples
    print("Testing 'curious':")
    result1 = mapper.map_text_to_traits("curious")
    print(f"Openness: {result1.trait_matrix.traits.get(TraitType.OPENNESS, 'Not found')}")
    print(f"Creativity: {result1.trait_matrix.traits.get(TraitType.CREATIVITY, 'Not found')}")
    
    print("\nTesting 'lick pussy':")
    result2 = mapper.map_text_to_traits("lick pussy")
    print(f"Sexual Experience: {result2.trait_matrix.traits.get(TraitType.SEXUAL_EXPERIENCE, 'Not found')}")
    print(f"Sexual Comfort: {result2.trait_matrix.traits.get(TraitType.SEXUAL_COMFORT_LEVEL, 'Not found')}") 