#!/usr/bin/env python3
"""
Test suite for emergent desire creation functionality.

Tests the Desire Interaction Networks expansion, specifically:
- Emergent desire creation when synergy exceeds threshold
- Logging of all interaction events
- Proper handling of emergent desire properties

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

import sys
import os
import unittest
import time
from datetime import datetime

# Add the parent directory to the path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add the IlanyaDesireEngine directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'IlanyaDesireEngine'))

from IlanyaDesireEngine.desire_engine import DesireEngine, DesireEngineConfig
from IlanyaDesireEngine.desire_engine.models import Desire, DesireState
from IlanyaDesireEngine.desire_engine.modules import InteractionModule
from utils.logging_utils import setup_logger, log_test_start, log_test_end


class TestEmergentDesires(unittest.TestCase):
    """Test cases for emergent desire creation."""
    
    def setUp(self):
        """Set up test environment."""
        # Set up logger for this test
        self.logger = setup_logger(
            engine_type="desire",
            test_type="test",
            test_name="emergent_desires",
            test_target="interaction_networks",
            log_level="DEBUG"
        )
        
        # Get log file path for configuration
        from utils.logging_utils import get_log_file_path
        log_file = get_log_file_path(
            engine_type="desire",
            test_type="test", 
            test_name="emergent_desires",
            test_target="interaction_networks"
        )
        
        # Create configuration with logging enabled
        self.config = DesireEngineConfig(
            log_level="DEBUG",
            log_file=log_file,
            log_interactions=True,
            log_emergent_desires=True,
            # Set very low thresholds for testing
            interaction_threshold=0.05,
            synergy_threshold=0.2,
            emergent_threshold=0.3,
            conflict_threshold=0.1
        )
        
        self.desire_engine = DesireEngine(self.config)
        self.logger.info("Test environment set up successfully")
    
    def tearDown(self):
        """Clean up test environment."""
        self.logger.info("Test environment cleanup completed")
    
    def test_emergent_desire_creation(self):
        """Test that emergent desires are created when synergy exceeds threshold."""
        start_time = time.time()
        log_test_start(self.logger, "emergent_desire_creation", 
                      "Test that emergent desires are created when synergy exceeds threshold")
        
        try:
            # Create two desires with high synergy potential and common traits
            desire_1 = Desire(
                id="test_desire_1",
                name="Desire for Creativity",
                source_traits=["creativity", "innovation"],
                strength=0.8,
                base_strength=0.8,
                reinforcement_count=3,
                last_reinforcement=datetime.now()
            )
            
            desire_2 = Desire(
                id="test_desire_2", 
                name="Desire for Learning",
                source_traits=["learning_desire", "creativity"],
                strength=0.7,
                base_strength=0.7,
                reinforcement_count=2,
                last_reinforcement=datetime.now()
            )
            
            self.logger.info(f"Created test desires: {desire_1.name}, {desire_2.name}")
            
            # Add desires to engine
            self.desire_engine.desires["test_desire_1"] = desire_1
            self.desire_engine.desires["test_desire_2"] = desire_2
            
            # Process interactions
            interaction_results = self.desire_engine.interaction_module.process_interactions(
                self.desire_engine.desires
            )
            
            self.logger.info(f"Processed interactions, got {len(interaction_results)} results")
            
            # Check that interactions were processed
            self.assertIsInstance(interaction_results, list)
            
            # Check for emergent desire creation
            emergent_desires = [
                d for d in self.desire_engine.desires.values() 
                if d.emergent
            ]
            
            self.logger.info(f"Found {len(emergent_desires)} emergent desires")
            
            # Should have created at least one emergent desire
            self.assertGreater(len(emergent_desires), 0, 
                              "No emergent desires were created")
            
            # Check emergent desire properties
            emergent = emergent_desires[0]
            self.assertTrue(emergent.emergent, "Desire should be marked as emergent")
            self.assertIn("test_desire_1", emergent.id, "Emergent ID should reference parent desires")
            self.assertIn("test_desire_2", emergent.id, "Emergent ID should reference parent desires")
            self.assertIn("creativity", emergent.source_traits, "Should inherit source traits")
            self.assertIn("learning_desire", emergent.source_traits, "Should inherit source traits")
            
            # Check that emergent desire has reasonable strength
            self.assertGreater(emergent.strength, 0.0, "Emergent desire should have positive strength")
            self.assertLessEqual(emergent.strength, 1.0, "Emergent desire strength should not exceed 1.0")
            
            self.logger.info(f"Emergent desire created successfully: {emergent.name} (ID: {emergent.id})")
            self.logger.info(f"Emergent desire strength: {emergent.strength}")
            self.logger.info(f"Emergent desire source traits: {emergent.source_traits}")
            
            duration = time.time() - start_time
            log_test_end(self.logger, "emergent_desire_creation", True, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Test failed with error: {str(e)}")
            log_test_end(self.logger, "emergent_desire_creation", False, duration)
            raise
    
    def test_emergent_desire_logging(self):
        """Test that emergent desire creation is properly logged."""
        start_time = time.time()
        log_test_start(self.logger, "emergent_desire_logging", 
                      "Test that emergent desire creation is properly logged")
        
        try:
            # Create desires with high synergy and common traits
            desire_1 = Desire(
                id="log_test_1",
                name="Desire for Adventure",
                source_traits=["openness_to_experience", "exploration"],
                strength=0.9,
                base_strength=0.9,
                reinforcement_count=5,
                last_reinforcement=datetime.now()
            )
            
            desire_2 = Desire(
                id="log_test_2",
                name="Desire for Exploration", 
                source_traits=["curiosity", "exploration"],
                strength=0.8,
                base_strength=0.8,
                reinforcement_count=4,
                last_reinforcement=datetime.now()
            )
            
            self.logger.info(f"Created logging test desires: {desire_1.name}, {desire_2.name}")
            
            self.desire_engine.desires["log_test_1"] = desire_1
            self.desire_engine.desires["log_test_2"] = desire_2
            
            # Process interactions
            self.desire_engine.interaction_module.process_interactions(
                self.desire_engine.desires
            )
            
            self.logger.info("Interactions processed, checking log file content")
            
            # Check log file for emergent desire creation message
            with open(self.config.log_file, 'r') as f:
                log_content = f.read()
            
            # Should contain emergent desire creation log
            self.assertIn("Created emergent desire", log_content,
                         "Log should contain emergent desire creation message")
            
            # Should contain interaction logging
            self.assertIn("Interaction:", log_content,
                         "Log should contain interaction messages")
            
            self.logger.info("Logging verification completed successfully")
            
            duration = time.time() - start_time
            log_test_end(self.logger, "emergent_desire_logging", True, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Test failed with error: {str(e)}")
            log_test_end(self.logger, "emergent_desire_logging", False, duration)
            raise
    
    def test_emergent_desire_threshold(self):
        """Test that emergent desires are only created above threshold."""
        start_time = time.time()
        log_test_start(self.logger, "emergent_desire_threshold", 
                      "Test that emergent desires are only created above threshold")
        
        try:
            # Create desires with low synergy (below emergent threshold)
            desire_1 = Desire(
                id="low_synergy_1",
                name="Desire for Quiet",
                source_traits=["introversion"],
                strength=0.3,
                base_strength=0.3,
                reinforcement_count=1,
                last_reinforcement=datetime.now()
            )
            
            desire_2 = Desire(
                id="low_synergy_2",
                name="Desire for Solitude",
                source_traits=["independence"],
                strength=0.2,
                base_strength=0.2,
                reinforcement_count=1,
                last_reinforcement=datetime.now()
            )
            
            self.logger.info(f"Created low synergy desires: {desire_1.name}, {desire_2.name}")
            
            self.desire_engine.desires["low_synergy_1"] = desire_1
            self.desire_engine.desires["low_synergy_2"] = desire_2
            
            # Count desires before interaction
            initial_count = len(self.desire_engine.desires)
            self.logger.info(f"Initial desire count: {initial_count}")
            
            # Process interactions
            self.desire_engine.interaction_module.process_interactions(
                self.desire_engine.desires
            )
            
            # Count desires after interaction
            final_count = len(self.desire_engine.desires)
            self.logger.info(f"Final desire count: {final_count}")
            
            # Should not create emergent desire due to low synergy
            self.assertEqual(initial_count, final_count,
                            "Should not create emergent desire with low synergy")
            
            self.logger.info("Threshold test completed successfully - no emergent desires created")
            
            duration = time.time() - start_time
            log_test_end(self.logger, "emergent_desire_threshold", True, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Test failed with error: {str(e)}")
            log_test_end(self.logger, "emergent_desire_threshold", False, duration)
            raise
    
    def test_emergent_desire_persistence(self):
        """Test that emergent desires persist through state save/load."""
        # Create desires and trigger emergent creation
        desire_1 = Desire(
            id="persist_test_1",
            name="Desire for Innovation",
            source_traits=["creativity", "problem_solving"],
            strength=0.8,
            base_strength=0.8,
            reinforcement_count=3,
            last_reinforcement=datetime.now()
        )
        
        desire_2 = Desire(
            id="persist_test_2",
            name="Desire for Problem Solving",
            source_traits=["analytical_thinking", "problem_solving"],
            strength=0.7,
            base_strength=0.7,
            reinforcement_count=2,
            last_reinforcement=datetime.now()
        )
        
        self.desire_engine.desires["persist_test_1"] = desire_1
        self.desire_engine.desires["persist_test_2"] = desire_2
        
        # Process interactions to create emergent desire
        self.desire_engine.interaction_module.process_interactions(
            self.desire_engine.desires
        )
        
        # Count emergent desires before save
        emergent_before = len([d for d in self.desire_engine.desires.values() if d.emergent])
        
        # Save state
        save_path = os.path.join(os.path.dirname(self.config.log_file), "test_state.json")
        self.desire_engine.save_state(save_path)
        
        # Create new engine and load state
        new_engine = DesireEngine(self.config)
        new_engine.load_state(save_path)
        
        # Count emergent desires after load
        emergent_after = len([d for d in new_engine.desires.values() if d.emergent])
        
        # Should have same number of emergent desires
        self.assertEqual(emergent_before, emergent_after,
                        "Emergent desires should persist through save/load")
        
        # Check that emergent flag is preserved
        for desire in new_engine.desires.values():
            if desire.emergent:
                self.assertTrue(desire.emergent, "Emergent flag should be preserved")
    
    def test_interaction_results_structure(self):
        """Test that interaction results have the expected structure."""
        # Create test desires
        desire_1 = Desire(
            id="struct_test_1",
            name="Desire for Growth",
            source_traits=["personal_development"],
            strength=0.6,
            base_strength=0.6,
            reinforcement_count=2,
            last_reinforcement=datetime.now()
        )
        
        desire_2 = Desire(
            id="struct_test_2",
            name="Desire for Achievement",
            source_traits=["ambition"],
            strength=0.5,
            base_strength=0.5,
            reinforcement_count=1,
            last_reinforcement=datetime.now()
        )
        
        self.desire_engine.desires["struct_test_1"] = desire_1
        self.desire_engine.desires["struct_test_2"] = desire_2
        
        # Process interactions
        results = self.desire_engine.interaction_module.process_interactions(
            self.desire_engine.desires
        )
        
        # Check results structure
        if results:  # If any interactions occurred
            result = results[0]
            self.assertIn('desire_1', result, "Result should contain desire_1")
            self.assertIn('desire_2', result, "Result should contain desire_2")
            self.assertIn('interaction_strength', result, "Result should contain interaction_strength")
            self.assertIn('type', result, "Result should contain type")
            
            # Check type values
            self.assertIn(result['type'], ['synergy', 'conflict', 'neutral'],
                         "Interaction type should be valid")
            
            # Check strength is numeric
            self.assertIsInstance(result['interaction_strength'], (int, float),
                                "Interaction strength should be numeric")


class TestInteractionModule(unittest.TestCase):
    """Test cases for the InteractionModule specifically."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = DesireEngineConfig(
            interaction_threshold=0.05,
            synergy_threshold=0.2,
            emergent_threshold=0.3,
            conflict_threshold=0.1
        )
        self.interaction_module = InteractionModule(self.config)
    
    def test_calculate_interaction_strength(self):
        """Test interaction strength calculation."""
        desire_1 = Desire(
            id="calc_test_1",
            name="Test Desire 1",
            source_traits=["trait_1"],
            strength=0.8,
            base_strength=0.8,
            reinforcement_count=2,
            last_reinforcement=datetime.now()
        )
        
        desire_2 = Desire(
            id="calc_test_2",
            name="Test Desire 2",
            source_traits=["trait_2"],
            strength=0.6,
            base_strength=0.6,
            reinforcement_count=1,
            last_reinforcement=datetime.now()
        )
        
        strength = self.interaction_module._calculate_interaction_strength(desire_1, desire_2)
        
        # Should return a float between -1 and 1
        self.assertIsInstance(strength, float)
        self.assertGreaterEqual(strength, -1.0)
        self.assertLessEqual(strength, 1.0)
    
    def test_create_emergent_desire(self):
        """Test emergent desire creation method."""
        desire_1 = Desire(
            id="emergent_test_1",
            name="Parent Desire 1",
            source_traits=["trait_1", "trait_2"],
            strength=0.8,
            base_strength=0.8,
            reinforcement_count=3,
            last_reinforcement=datetime.now()
        )
        
        desire_2 = Desire(
            id="emergent_test_2",
            name="Parent Desire 2",
            source_traits=["trait_3"],
            strength=0.7,
            base_strength=0.7,
            reinforcement_count=2,
            last_reinforcement=datetime.now()
        )
        
        # Test with high interaction strength (should create emergent)
        emergent = self.interaction_module._create_emergent_desire(
            desire_1, desire_2, 0.9
        )
        
        self.assertIsNotNone(emergent, "Should create emergent desire with high interaction")
        if emergent is not None:  # Type guard for linter
            self.assertTrue(emergent.emergent, "Should be marked as emergent")
            self.assertIn("emergent_test_1", emergent.id, "ID should reference parent 1")
            self.assertIn("emergent_test_2", emergent.id, "ID should reference parent 2")
            
            # Check source traits are combined
            expected_traits = ["trait_1", "trait_2", "trait_3"]
            for trait in expected_traits:
                self.assertIn(trait, emergent.source_traits, f"Should inherit trait: {trait}")
        
        # Test with low interaction strength (should not create emergent)
        emergent_low = self.interaction_module._create_emergent_desire(
            desire_1, desire_2, 0.3
        )
        
        self.assertIsNone(emergent_low, "Should not create emergent desire with low interaction")


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2) 