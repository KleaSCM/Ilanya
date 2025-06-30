"""
Ilanya Trait Engine - Test Suite

Comprehensive test suite for the Ilanya Trait Engine components.
Tests trait types, data structures, state management, and neural network
components to ensure reliability and correctness of the trait processing system.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

import sys
import os
import time
import pytest
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

from IlanyaTraitEngine.src.trait_models.trait_types import TraitType, TraitCategory, TraitDimension
from IlanyaTraitEngine.src.trait_models.trait_data import TraitVector, TraitMatrix, TraitData, TraitDataBuilder
from IlanyaTraitEngine.src.trait_models.trait_state import TraitState, CognitiveState
from utils.logging_utils import setup_logger, log_test_start, log_test_end


class TestTraitTypes:
    """
    Test trait type definitions.
    
    Verifies that all trait types, categories, and dimensions are properly
    defined and accessible. Ensures the enum structure is correct.
    """
    
    @classmethod
    def setup_class(cls):
        """Set up logger for trait types tests."""
        cls.logger = setup_logger(
            engine_type="trait",
            test_type="test",
            test_name="trait_types",
            test_target="enum_definitions",
            log_level="DEBUG"
        )
    
    def test_trait_type_enum(self):
        """
        Test that trait types are properly defined.
        
        Verifies that trait type enums have correct string values and
        can be accessed properly.
        """
        start_time = time.time()
        log_test_start(self.logger, "trait_type_enum", 
                      "Test that trait types are properly defined")
        
        try:
        assert TraitType.OPENNESS.value == "openness"
        assert TraitType.CREATIVITY.value == "creativity"
        assert TraitType.ADAPTABILITY.value == "adaptability"
            
            self.logger.info("Trait type enum tests passed successfully")
            
            duration = time.time() - start_time
            log_test_end(self.logger, "trait_type_enum", True, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Test failed with error: {str(e)}")
            log_test_end(self.logger, "trait_type_enum", False, duration)
            raise
    
    def test_trait_categories(self):
        """
        Test trait categories.
        
        Verifies that trait categories are properly defined and
        have correct string representations.
        """
        start_time = time.time()
        log_test_start(self.logger, "trait_categories", 
                      "Test trait categories are properly defined")
        
        try:
        assert TraitCategory.PERSONALITY.value == "personality"
        assert TraitCategory.COGNITIVE.value == "cognitive"
        assert TraitCategory.BEHAVIORAL.value == "behavioral"
            
            self.logger.info("Trait category tests passed successfully")
            
            duration = time.time() - start_time
            log_test_end(self.logger, "trait_categories", True, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Test failed with error: {str(e)}")
            log_test_end(self.logger, "trait_categories", False, duration)
            raise
    
    def test_trait_dimensions(self):
        """
        Test trait dimensions.
        
        Verifies that trait dimensions are properly defined and
        have correct string representations.
        """
        start_time = time.time()
        log_test_start(self.logger, "trait_dimensions", 
                      "Test trait dimensions are properly defined")
        
        try:
        assert TraitDimension.INTENSITY.value == "intensity"
        assert TraitDimension.STABILITY.value == "stability"
        assert TraitDimension.PLASTICITY.value == "plasticity"
            
            self.logger.info("Trait dimension tests passed successfully")
            
            duration = time.time() - start_time
            log_test_end(self.logger, "trait_dimensions", True, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Test failed with error: {str(e)}")
            log_test_end(self.logger, "trait_dimensions", False, duration)
            raise


class TestTraitData:
    """
    Test trait data structures.
    
    Tests the core data structures including TraitVector, TraitMatrix,
    TraitData, and TraitDataBuilder. Verifies data validation, serialization,
    and basic operations.
    """
    
    @classmethod
    def setup_class(cls):
        """Set up logger for trait data tests."""
        cls.logger = setup_logger(
            engine_type="trait",
            test_type="test",
            test_name="trait_data",
            test_target="data_structures",
            log_level="DEBUG"
        )
    
    def test_trait_vector_creation(self):
        """
        Test creating a trait vector.
        
        Verifies that TraitVector objects can be created with valid
        parameters and that the data is stored correctly.
        """
        start_time = time.time()
        log_test_start(self.logger, "trait_vector_creation", 
                      "Test creating a trait vector")
        
        try:
        trait_vector = TraitVector(
            trait_type=TraitType.OPENNESS,
            value=0.7,
            confidence=0.9
        )
        
        assert trait_vector.trait_type == TraitType.OPENNESS
        assert trait_vector.value == 0.7
        assert trait_vector.confidence == 0.9
            
            self.logger.info(f"Trait vector created successfully: {trait_vector.trait_type.value}")
            
            duration = time.time() - start_time
            log_test_end(self.logger, "trait_vector_creation", True, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Test failed with error: {str(e)}")
            log_test_end(self.logger, "trait_vector_creation", False, duration)
            raise
    
    def test_trait_vector_validation(self):
        """
        Test trait vector validation.
        
        Verifies that TraitVector properly validates input values
        and raises appropriate errors for invalid data.
        """
        start_time = time.time()
        log_test_start(self.logger, "trait_vector_validation", 
                      "Test trait vector validation")
        
        try:
        # Should raise error for invalid values
        with pytest.raises(ValueError):
            TraitVector(TraitType.OPENNESS, 1.5, 0.9)  # Value > 1.0
        
        with pytest.raises(ValueError):
            TraitVector(TraitType.OPENNESS, 0.7, -0.1)  # Confidence < 0.0
            
            self.logger.info("Trait vector validation tests passed successfully")
            
            duration = time.time() - start_time
            log_test_end(self.logger, "trait_vector_validation", True, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Test failed with error: {str(e)}")
            log_test_end(self.logger, "trait_vector_validation", False, duration)
            raise
    
    def test_trait_matrix_creation(self):
        """
        Test creating a trait matrix.
        
        Verifies that TraitMatrix objects can be created with multiple
        traits and that the interaction matrix is properly initialized.
        """
        start_time = time.time()
        log_test_start(self.logger, "trait_matrix_creation", 
                      "Test creating a trait matrix")
        
        try:
        traits = {
            TraitType.OPENNESS: TraitVector(TraitType.OPENNESS, 0.7, 0.9),
            TraitType.CREATIVITY: TraitVector(TraitType.CREATIVITY, 0.8, 0.8)
        }
        
        trait_matrix = TraitMatrix(traits=traits)
        
        assert len(trait_matrix.traits) == 2
        assert trait_matrix.interaction_matrix is not None
        assert trait_matrix.interaction_matrix.shape == (2, 2)
            
            self.logger.info(f"Trait matrix created successfully with {len(trait_matrix.traits)} traits")
            
            duration = time.time() - start_time
            log_test_end(self.logger, "trait_matrix_creation", True, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Test failed with error: {str(e)}")
            log_test_end(self.logger, "trait_matrix_creation", False, duration)
            raise
    
    def test_trait_data_builder(self):
        """
        Test trait data builder.
        
        Verifies that TraitDataBuilder can construct TraitData objects
        step by step and that all metadata is properly set.
        """
        start_time = time.time()
        log_test_start(self.logger, "trait_data_builder", 
                      "Test trait data builder")
        
        try:
        builder = TraitDataBuilder()
        builder.add_trait(TraitType.OPENNESS, 0.7, 0.9)
        builder.add_trait(TraitType.CREATIVITY, 0.8, 0.8)
        builder.set_source("test_source")
        
        trait_data = builder.build()
        
        assert trait_data.get_trait_count() == 2
        assert trait_data.source == "test_source"
        assert TraitType.OPENNESS in trait_data.trait_matrix.traits
        assert TraitType.CREATIVITY in trait_data.trait_matrix.traits
            
            self.logger.info(f"Trait data builder test passed: {trait_data.get_trait_count()} traits")
            
            duration = time.time() - start_time
            log_test_end(self.logger, "trait_data_builder", True, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Test failed with error: {str(e)}")
            log_test_end(self.logger, "trait_data_builder", False, duration)
            raise


class TestTraitState:
    """
    Test trait state tracking.
    
    Tests the TraitState and CognitiveState classes that handle
    state tracking and cognitive state management over time.
    """
    
    @classmethod
    def setup_class(cls):
        """Set up logger for trait state tests."""
        cls.logger = setup_logger(
            engine_type="trait",
            test_type="test",
            test_name="trait_state",
            test_target="state_management",
            log_level="DEBUG"
        )
    
    def test_trait_state_creation(self):
        """
        Test creating a trait state.
        
        Verifies that TraitState objects can be created with valid
        parameters and that change rates are computed correctly.
        """
        start_time = time.time()
        log_test_start(self.logger, "trait_state_creation", 
                      "Test creating a trait state")
        
        try:
        trait_state = TraitState(
            trait_type=TraitType.OPENNESS,
            current_value=0.7,
            previous_value=0.6,
            confidence=0.9
        )
        
        assert trait_state.trait_type == TraitType.OPENNESS
        assert trait_state.current_value == 0.7
        assert trait_state.previous_value == 0.6
            assert trait_state.change_rate == pytest.approx(0.1, rel=1e-10)  # Use pytest.approx for floating point
        assert trait_state.confidence == 0.9
            
            self.logger.info(f"Trait state created successfully: change_rate={trait_state.change_rate}")
            
            duration = time.time() - start_time
            log_test_end(self.logger, "trait_state_creation", True, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Test failed with error: {str(e)}")
            log_test_end(self.logger, "trait_state_creation", False, duration)
            raise
    
    def test_cognitive_state_creation(self):
        """
        Test creating a cognitive state.
        
        Verifies that CognitiveState objects can be created with
        multiple trait states and cognitive metrics.
        """
        start_time = time.time()
        log_test_start(self.logger, "cognitive_state_creation", 
                      "Test creating a cognitive state")
        
        try:
        trait_states = {
            TraitType.OPENNESS: TraitState(TraitType.OPENNESS, 0.7, confidence=0.9),
            TraitType.CREATIVITY: TraitState(TraitType.CREATIVITY, 0.8, confidence=0.8)
        }
        
        cognitive_state = CognitiveState(
            trait_states=trait_states,
            overall_stability=0.8,
            cognitive_load=0.3,
            attention_focus=0.9,
            emotional_state=0.6
        )
        
        assert len(cognitive_state.trait_states) == 2
            # The overall_stability is computed internally, so we need to check the actual computed value
            assert cognitive_state.overall_stability == pytest.approx(1.0, rel=1e-10)  # Default value when no previous state
        assert cognitive_state.cognitive_load == 0.3
        assert cognitive_state.attention_focus == 0.9
        assert cognitive_state.emotional_state == 0.6
    
            self.logger.info(f"Cognitive state created successfully with {len(cognitive_state.trait_states)} trait states")
            
            duration = time.time() - start_time
            log_test_end(self.logger, "cognitive_state_creation", True, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Test failed with error: {str(e)}")
            log_test_end(self.logger, "cognitive_state_creation", False, duration)
            raise


class TestNeuralNetworkComponents:
    """
    Test neural network components.
    
    Tests the neural network components including embeddings,
    positional encoding, and transformer blocks.
    """
    
    def test_trait_embedding(self):
        """
        Test trait embedding layer.
        
        Verifies that the TraitEmbedding layer can process trait data
        and produce embeddings of the correct shape.
        """
        from IlanyaTraitEngine.src.neural_networks.trait_transformer import TraitEmbedding
        
        embedding = TraitEmbedding(num_traits=20, embedding_dim=64, input_dim=512)
        
        # Test forward pass with sample data
        batch_size, num_traits = 2, 5
        trait_values = torch.randn(batch_size, num_traits)
        trait_confidences = torch.randn(batch_size, num_traits)
        trait_indices = torch.randint(0, 20, (batch_size, num_traits))
        
        output = embedding(trait_values, trait_confidences, trait_indices)
        
        assert output.shape == (batch_size, num_traits, 64)
    
    def test_positional_encoding(self):
        """
        Test positional encoding.
        
        Verifies that the PositionalEncoding layer adds position
        information to embeddings and produces different outputs.
        """
        from IlanyaTraitEngine.src.neural_networks.trait_transformer import PositionalEncoding
        
        pos_encoding = PositionalEncoding(embedding_dim=64, max_seq_length=100)
        
        # Test forward pass
        seq_len, batch_size, embedding_dim = 10, 2, 64
        x = torch.randn(seq_len, batch_size, embedding_dim)
        
        output = pos_encoding(x)
        
        assert output.shape == x.shape
        assert not torch.allclose(output, x)  # Should be different due to positional encoding


if __name__ == "__main__":
    pytest.main() 