# Ilanya Trait Engine Neural Trait Mapping System

# Complete neural network system for mapping natural language input to trait modifications.
# Uses transformer-based architecture to learn text-to-trait mappings from data.

# Author: KleaSCM
# Email: KleaSCM@gmail.com
# License: MIT
# Version: 0.1.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer, AutoModel
import json
import os

from IlanyaTraitEngine.trait_types import (
    TraitType, TraitCategory, PERMANENTLY_PROTECTED_TRAITS,
    PARTIALLY_PROTECTED_TRAITS, FULLY_EVOLVABLE_TRAITS
)
from IlanyaTraitEngine.trait_data import TraitVector, TraitMatrix, TraitData, TraitDataBuilder


@dataclass
class TraitMappingConfig:
    """Configuration for the full neural trait mapping system."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    hidden_dim: int = 512
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1
    max_sequence_length: int = 128
    num_traits: int = 54                    # Number of trait types (updated to match TraitType enum)
    trait_embedding_dim: int = 64
    learning_rate: float = 1e-4
    batch_size: int = 16
    protection_factor: float = 0.01
    confidence_threshold: float = 0.7
    save_path: str = "models/trait_mapper.pt"
    dataset_path: str = "data/trait_mappings.json"


class TraitMappingNetwork(nn.Module):
    """Complete neural network for mapping natural language to trait modifications."""
    
    def __init__(self, config: TraitMappingConfig):
        super().__init__()
        self.config = config
        
        # Text encoder
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.text_encoder = AutoModel.from_pretrained(config.model_name)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Trait embeddings
        self.trait_embeddings = nn.Embedding(config.num_traits, config.trait_embedding_dim)
        
        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_layers = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # Prediction layers
        self.trait_predictor = nn.Sequential(
            nn.Linear(config.embedding_dim + config.trait_embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 3)  # [value_change, confidence, protection_level]
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        self.trait_layer_norm = nn.LayerNorm(config.trait_embedding_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
        
    def forward(self, text_inputs: List[str]) -> Dict[str, torch.Tensor]:
        """Forward pass through the complete trait mapping network."""
        batch_size = len(text_inputs)
        device = next(self.parameters()).device
        
        # Tokenize and encode
        encoded = self.tokenizer(
            text_inputs,
            padding=True,
            truncation=True,
            max_length=self.config.max_sequence_length,
            return_tensors="pt"
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Get text embeddings
        with torch.no_grad():
            text_outputs = self.text_encoder(**encoded)
            text_embeddings = text_outputs.last_hidden_state
        
        # Apply transformer layers
        text_embeddings = self.layer_norm(text_embeddings)
        text_embeddings = self.transformer_layers(text_embeddings)
        
        # Cross-attention
        attended_text, attention_weights = self.cross_attention(
            query=text_embeddings,
            key=text_embeddings,
            value=text_embeddings
        )
        
        # Pool text embeddings
        pooled_text = attended_text.mean(dim=1)
        
        # Get trait embeddings
        trait_indices = torch.arange(self.config.num_traits, device=device).unsqueeze(0).expand(batch_size, -1)
        trait_embeddings = self.trait_embeddings(trait_indices)
        trait_embeddings = self.trait_layer_norm(trait_embeddings)
        
        # Combine embeddings
        expanded_text = pooled_text.unsqueeze(1).expand(-1, self.config.num_traits, -1)
        combined = torch.cat([expanded_text, trait_embeddings], dim=-1)
        
        # Predict trait modifications
        trait_predictions = self.trait_predictor(combined)
        
        # Extract components
        value_changes = trait_predictions[:, :, 0]
        confidences = torch.sigmoid(trait_predictions[:, :, 1])
        protection_levels = torch.sigmoid(trait_predictions[:, :, 2])
        
        # Apply protection constraints
        protected_mask = self._get_protection_mask().to(device)
        value_changes = value_changes * (1 - protected_mask * 0.99)
        
        return {
            'value_changes': value_changes,
            'confidences': confidences,
            'protection_levels': protection_levels,
            'text_embeddings': pooled_text,
            'trait_embeddings': trait_embeddings,
            'attention_weights': attention_weights
        }
    
    def _get_protection_mask(self) -> torch.Tensor:
        """Get mask for protected traits."""
        mask = torch.zeros(self.config.num_traits)
        for trait in PERMANENTLY_PROTECTED_TRAITS:
            mask[list(TraitType).index(trait)] = 1.0
        for trait in PARTIALLY_PROTECTED_TRAITS:
            mask[list(TraitType).index(trait)] = 0.5
        return mask


class TraitMapper:
    """Complete neural trait mapping system."""
    
    def __init__(self, config: Optional[TraitMappingConfig] = None):
        self.config = config or TraitMappingConfig()
        self.network = TraitMappingNetwork(self.config)
        
        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        
        # Training components
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        # Loss functions
        self.value_loss_fn = nn.MSELoss()
        self.confidence_loss_fn = nn.BCELoss()
        
        # Load pre-trained model if available
        self._load_model()
        
    def _load_model(self):
        """Load pre-trained model if available."""
        if os.path.exists(self.config.save_path):
            try:
                checkpoint = torch.load(self.config.save_path, map_location=self.device)
                self.network.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded pre-trained model from {self.config.save_path}")
            except Exception as e:
                print(f"Failed to load model: {e}")
    
    def save_model(self):
        """Save the current model."""
        os.makedirs(os.path.dirname(self.config.save_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, self.config.save_path)
        print(f"Model saved to {self.config.save_path}")
    
    def map_text_to_traits(self, text: str, current_traits: Optional[TraitData] = None) -> TraitData:
        """Map natural language text to trait modifications using the neural network."""
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
        
        # Apply modifications to current traits or create new ones
        if current_traits:
            return self._apply_incremental_changes(current_traits, modifications)
        else:
            return self._create_new_trait_data(modifications)
    
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
                base_value = 0.5
                new_value = max(0.0, min(1.0, base_value + modification))
                builder.add_trait(trait_type, new_value, 0.8)
        
        return builder.build()
    
    def _create_new_trait_data(self, modifications: Dict[TraitType, float]) -> TraitData:
        """Create new trait data from modifications."""
        builder = TraitDataBuilder()
        
        for trait_type, modification in modifications.items():
            base_value = 0.5
            new_value = max(0.0, min(1.0, base_value + modification))
            builder.add_trait(trait_type, new_value, 0.8)
        
        return builder.build()
    
    def add_training_example(self, text: str, trait_modifications: Dict[TraitType, float]):
        """Add a new training example to the dataset."""
        traits_dict = {trait.value.upper(): value for trait, value in trait_modifications.items()}
        example = {"text": text, "traits": traits_dict}
        
        # Load existing dataset
        if os.path.exists(self.config.dataset_path):
            with open(self.config.dataset_path, 'r') as f:
                data = json.load(f)
        else:
            data = []
        
        data.append(example)
        
        # Save updated dataset
        os.makedirs(os.path.dirname(self.config.dataset_path), exist_ok=True)
        with open(self.config.dataset_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Added training example: '{text}' -> {trait_modifications}")


# Example usage
if __name__ == "__main__":
    mapper = TraitMapper()
    
    test_texts = [
        "I'm feeling curious about new experiences",
        "I'm feeling horny and want to explore sexually",
        "I want to work hard and be successful",
        "I'm feeling sad and depressed"
    ]
    
    for text in test_texts:
        result = mapper.map_text_to_traits(text)
        print(f"\nInput: '{text}'")
        print("Detected trait changes:")
        for trait_type, trait_vector in result.trait_matrix.traits.items():
            if abs(trait_vector.value - 0.5) > 0.1:
                change = trait_vector.value - 0.5
                direction = "increased" if change > 0 else "decreased"
                print(f"  {trait_type.value}: {direction} by {abs(change):.3f}") 