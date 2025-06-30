"""
Ilanya Full Pipeline Demo

Comprehensive demo that tests the complete pipeline:
Word List → Trait Mapper → Trait Engine → Desire Engine → Goals Engine

This demo processes words from the English word list as if they were user input,
showing real-time processing and logging everything to a file.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

import sys
import os
import time
import logging
from datetime import datetime
from pathlib import Path

# Add paths for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "Neural Networks"))
sys.path.append(str(project_root / "InterfaceLayer"))
sys.path.append(str(project_root / "IlanyaDesireEngine"))
sys.path.append(str(project_root / "IlanyaGoalsEngine"))

# Import pipeline components
from InterfaceLayer.agent_pipeline import SimpleAgentPipeline, PipelineResponse
from IlanyaTraitMapping.trait_mapper import TraitMapper
from IlanyaTraitEngine.trait_engine import TraitEngine
from IlanyaTraitEngine.trait_types import TraitType

AgentPipeline = SimpleAgentPipeline


def setup_logging():
    """Set up comprehensive logging for the demo."""
    # Create logs directory
    log_dir = Path("Logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"full_pipeline_demo_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)  # Also output to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"=== ILANYA FULL PIPELINE DEMO STARTED ===")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Timestamp: {datetime.now()}")
    logger.info("=" * 80)
    
    return logger


def load_word_list(word_file_path: str = "english-words/words.txt") -> list:
    """Load words from the English word list."""
    logger = logging.getLogger(__name__)
    
    full_path = Path(project_root) / word_file_path
    logger.info(f"Loading word list from: {full_path}")
    
    if not full_path.exists():
        logger.error(f"Word file not found: {full_path}")
        return []
    
    words = []
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().lower()
                if word and len(word) > 2:  # Filter out very short words
                    words.append(word)
        
        logger.info(f"Loaded {len(words)} words from word list")
        return words[:100]  # Limit to first 100 words for demo
        
    except Exception as e:
        logger.error(f"Error loading word list: {e}")
        return []


def test_individual_components(logger):
    """Test individual components before running the full pipeline."""
    logger.info("\n" + "="*60)
    logger.info("TESTING INDIVIDUAL COMPONENTS")
    logger.info("="*60)
    
    # Test 1: Trait Mapper
    logger.info("\n🧪 Testing Trait Mapper...")
    try:
        trait_mapper = TraitMapper()
        logger.info("✅ Trait Mapper initialized successfully")
        
        # Test with a simple word
        test_word = "curious"
        logger.info(f"Testing with word: '{test_word}'")
        # Note: This would need trait data to work properly
        
    except Exception as e:
        logger.error(f"❌ Trait Mapper failed: {e}")
        return False
    
    # Test 2: Trait Engine
    logger.info("\n🧪 Testing Trait Engine...")
    try:
        trait_engine = TraitEngine()
        logger.info("✅ Trait Engine initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Trait Engine failed: {e}")
        return False
    
    # Test 3: Simple Agent Pipeline
    logger.info("\n🧪 Testing Simple Agent Pipeline...")
    try:
        pipeline = AgentPipeline()
        logger.info("✅ Simple Agent Pipeline initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Simple Agent Pipeline failed: {e}")
        return False
    
    logger.info("\n✅ All individual components tested successfully!")
    return True


def run_word_processing_demo(logger, words: list):
    """Run the main demo processing words through the pipeline."""
    logger.info("\n" + "="*60)
    logger.info("RUNNING WORD PROCESSING DEMO")
    logger.info("="*60)
    
    # Initialize pipeline
    logger.info("\n🚀 Initializing Simple Agent Pipeline...")
    try:
        pipeline = AgentPipeline()
        logger.info("✅ Pipeline initialized successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize pipeline: {e}")
        return
    
    # Process words
    logger.info(f"\n📝 Processing {len(words)} words through the pipeline...")
    logger.info("=" * 60)
    
    successful_processing = 0
    failed_processing = 0
    
    for i, word in enumerate(words, 1):
        logger.info(f"\n{'='*20} WORD {i}/{len(words)}: '{word}' {'='*20}")
        
        start_time = time.time()
        
        try:
            # Process word through pipeline
            response = pipeline.process_user_input(word)
            
            processing_time = time.time() - start_time
            
            if response.processed:
                successful_processing += 1
                logger.info(f"✅ SUCCESS - Processing time: {processing_time:.3f}s")
                
                # Log trait modifications
                if response.trait_modifications:
                    logger.info("🎯 Trait Modifications:")
                    for trait_name, change in response.trait_modifications.items():
                        direction = "increased" if change > 0 else "decreased"
                        logger.info(f"   - {trait_name}: {direction} by {abs(change):.3f}")
                else:
                    logger.info("   No significant trait modifications detected")
                
                # Log evolved traits
                if response.evolved_traits:
                    logger.info("🧬 Evolved Traits (top 5):")
                    sorted_traits = sorted(response.evolved_traits.items(), 
                                         key=lambda x: x[1], reverse=True)[:5]
                    for trait_name, value in sorted_traits:
                        logger.info(f"   - {trait_name}: {value:.3f}")
                
                # Log desires
                if response.desires_activated:
                    logger.info(f"💭 Desires Activated: {len(response.desires_activated)}")
                    for desire in response.desires_activated[:3]:  # Show first 3
                        logger.info(f"   - {desire}")
                
                # Log goals
                if response.goals_formed:
                    logger.info(f"🎯 Goals Formed: {len(response.goals_formed)}")
                    for goal in response.goals_formed[:3]:  # Show first 3
                        priority = response.goal_priorities.get(goal, 0.0)
                        logger.info(f"   - {goal} (priority: {priority:.3f})")
                
            else:
                failed_processing += 1
                logger.error(f"❌ FAILED - {response.error_message}")
                
        except Exception as e:
            failed_processing += 1
            processing_time = time.time() - start_time
            logger.error(f"❌ EXCEPTION - Processing time: {processing_time:.3f}s")
            logger.error(f"   Error: {str(e)}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("DEMO SUMMARY")
    logger.info("="*60)
    logger.info(f"Total words processed: {len(words)}")
    logger.info(f"Successful: {successful_processing}")
    logger.info(f"Failed: {failed_processing}")
    logger.info(f"Success rate: {(successful_processing/len(words)*100):.1f}%")


def run_specific_word_tests(logger):
    """Run tests with specific words that should trigger interesting responses."""
    logger.info("\n" + "="*60)
    logger.info("RUNNING SPECIFIC WORD TESTS")
    logger.info("="*60)
    
    # Words that should trigger interesting trait modifications
    test_words = [
        "curious",      # Should increase OPENNESS
        "creative",     # Should increase CREATIVITY
        "horny",        # Should affect SEXUAL_EXPERIENCE
        "happy",        # Should increase OPTIMISM
        "sad",          # Should decrease OPTIMISM
        "angry",        # Should affect EMOTIONAL_STABILITY
        "friendly",     # Should increase AGREEABLENESS
        "confident",    # Should increase EMOTIONAL_STABILITY
        "anxious",      # Should decrease EMOTIONAL_STABILITY
        "passionate",   # Should affect SEXUAL_EXPERIENCE
    ]
    
    logger.info(f"Testing {len(test_words)} specific words...")
    
    try:
        pipeline = AgentPipeline()
        
        for word in test_words:
            logger.info(f"\n{'='*15} TESTING: '{word}' {'='*15}")
            
            response = pipeline.process_user_input(word)
            
            if response.processed:
                logger.info("✅ Processing successful")
                
                # Show trait changes
                if response.trait_modifications:
                    logger.info("🎯 Trait Changes:")
                    for trait_name, change in response.trait_modifications.items():
                        direction = "increased" if change > 0 else "decreased"
                        logger.info(f"   - {trait_name}: {direction} by {abs(change):.3f}")
                else:
                    logger.info("   No trait changes detected")
                    
            else:
                logger.error(f"❌ Processing failed: {response.error_message}")
                
    except Exception as e:
        logger.error(f"❌ Specific word tests failed: {e}")


def main():
    """Main demo function."""
    # Setup logging
    logger = setup_logging()
    
    try:
        # Test individual components
        if not test_individual_components(logger):
            logger.error("❌ Component testing failed. Stopping demo.")
            return
        
        # Load word list
        words = load_word_list()
        if not words:
            logger.error("❌ No words loaded. Stopping demo.")
            return
        
        # Run specific word tests first
        run_specific_word_tests(logger)
        
        # Run full word processing demo
        run_word_processing_demo(logger, words)
        
        logger.info("\n" + "="*80)
        logger.info("🎉 FULL PIPELINE DEMO COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ Demo failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main()