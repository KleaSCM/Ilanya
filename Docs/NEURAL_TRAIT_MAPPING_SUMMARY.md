# Ilanya Neural Trait Mapping System

## Overview

The Ilanya Neural Trait Mapping System is a sophisticated AI component that converts natural language input into trait modifications for the Trait Engine. It uses a **massive dataset of 465,814 English words** to generate training data and train a neural network that can understand and map text to personality and cognitive traits.

## Key Components

### 1. Dataset Generator (`dataset_generator.py`)

**Purpose**: Generates synthetic training data using the massive word dataset

**Features**:
- **465,814 words** loaded from `IlanyaTraitEngine/src/utils/english-words/words.txt`
- Predefined word-trait associations for realistic mappings
- Generates both word-based and phrase-based training examples
- Creates template-based sentences for comprehensive coverage

**Word Categories**:
- **Curiosity & Exploration**: curious, explore, adventure, discover, learn, study, research
- **Sexual Content**: horny, sexy, aroused, passionate, intimate
- **Emotional States**: happy, sad, angry, excited, joyful, depressed, anxious, calm, peaceful
- **Social Interactions**: friendly, social, helpful, kind, generous, compassionate, empathetic
- **Cognitive Activities**: think, analyze, create, imagine, innovate, solve, focus, concentrate
- **Behavioral Patterns**: work, plan, organize, lead, manage, persist, determined, ambitious
- **Personality Traits**: confident, optimistic, patient, flexible, responsible, reliable, honest, brave, courageous

### 2. Neural Trait Mapper (`trait_mapper.py`)

**Purpose**: Neural network that maps text to trait modifications

**Architecture**:
- **Transformer-based** neural network
- **Text embedding** layer for processing natural language
- **Trait embedding** layer for representing trait states
- **Multi-head attention** for understanding context
- **Output layer** that predicts trait modifications

**Features**:
- Respects trait protection levels (permanently protected traits)
- Provides confidence scores for predictions
- Can be trained on custom datasets
- Supports incremental trait changes

### 3. Training System

**Training Data Generation**:
```python
# Generate 5,000+ training examples
generator = TraitMappingDatasetGenerator()
word_dataset = generator.generate_synthetic_dataset(num_examples=3000)
phrase_dataset = generator.create_phrase_dataset(num_phrases=2000)
combined_dataset = word_dataset + phrase_dataset
```

**Training Process**:
- **80/20 split** for training/validation
- **Batch processing** for efficient training
- **Loss tracking** for monitoring progress
- **Model saving/loading** for persistence

## Example Usage

### Basic Text-to-Trait Mapping

```python
from IlanyaTraitEngine.src.trait_mapping.trait_mapper import TraitMapper

# Initialize the mapper
mapper = TraitMapper()

# Map text to traits
text = "I am feeling curious and excited about learning new things"
trait_modifications = mapper.map_text_to_traits(text)

# Results show trait changes
for trait_type, trait_vector in trait_modifications.trait_matrix.traits.items():
    if abs(trait_vector.value) > 0.01:
        direction = "increased" if trait_vector.value > 0 else "decreased"
        print(f"{trait_type.value}: {direction} by {abs(trait_vector.value):.3f}")
```

### Training the Neural Network

```python
# Generate training data from massive word list
generator = TraitMappingDatasetGenerator()
dataset = generator.generate_synthetic_dataset(num_examples=5000)

# Train the model
mapper = TraitMapper()
training_history = mapper.train_on_dataset(
    train_data=dataset[:4000],
    val_data=dataset[4000:],
    epochs=10,
    batch_size=32,
    learning_rate=0.001
)

# Save the trained model
mapper.save_model("models/trained_trait_mapper.pth")
```

## Integration with Trait Engine

The neural trait mapper integrates seamlessly with the existing Trait Engine:

```python
from IlanyaTraitEngine.src.trait_engine.trait_engine import TraitEngine
from IlanyaTraitEngine.src.trait_mapping.trait_mapper import TraitMapper

# Initialize both systems
trait_mapper = TraitMapper()
trait_engine = TraitEngine()

# Process natural language input
text_input = "I want to be more creative and social"
trait_modifications = trait_mapper.map_text_to_traits(text_input)

# Process through trait engine
engine_results = trait_engine.process_traits(trait_modifications)

# Get evolved traits
evolved_traits = engine_results['predicted_traits']
```

## Dataset Statistics

### Word Dataset
- **Total words**: 465,814
- **Filtered words**: 465,814 (words > 2 characters)
- **Predefined mappings**: 50+ word-trait associations
- **Categories**: 7 major word categories

### Generated Training Data
- **Word examples**: 3,000
- **Phrase examples**: 1,600+
- **Total examples**: 4,600+
- **File sizes**:
  - `word_trait_mappings.json`: 445 KB
  - `phrase_trait_mappings.json`: 229 KB
  - `combined_trait_mappings.json`: 674 KB

## Protection Mechanisms

The system respects trait protection levels:

- **Permanently Protected Traits**: Cannot be modified (e.g., sexual orientation, gender identity)
- **Protected Traits**: Limited modification (e.g., core moral values)
- **Mutable Traits**: Full modification allowed (e.g., openness, creativity)

## Demo Scripts

1. **`test_dataset_generation.py`**: Simple test of dataset generation
2. **`neural_trait_mapping_training_demo.py`**: Full training demonstration
3. **`integrated_trait_mapping_demo.py`**: Integration with trait engine

## Files Created

### Core Components
- `IlanyaTraitEngine/src/trait_mapping/dataset_generator.py`
- `IlanyaTraitEngine/src/trait_mapping/trait_mapper.py`
- `IlanyaTraitEngine/src/trait_mapping/simple_trait_mapper.py`

### Demo Scripts
- `Demo/neural_trait_mapping_training_demo.py`
- `Demo/integrated_trait_mapping_demo.py`
- `test_dataset_generation.py`

### Generated Data
- `data/word_trait_mappings.json`
- `data/phrase_trait_mappings.json`
- `data/combined_trait_mappings.json`
- `data/sample_trait_mappings.json`

## Benefits

1. **Massive Vocabulary**: Uses 465K+ words for comprehensive coverage
2. **Neural Learning**: Learns from data rather than hardcoded rules
3. **Context Understanding**: Transformer architecture understands context
4. **Scalable**: Can be trained on larger datasets
5. **Protection**: Respects trait protection mechanisms
6. **Integration**: Seamlessly works with existing trait engine

## Future Enhancements

1. **Larger Training Datasets**: Use more sophisticated word associations
2. **Contextual Understanding**: Better understanding of sentence context
3. **Multi-language Support**: Extend to other languages
4. **Real-time Learning**: Learn from user interactions
5. **Advanced Protection**: More sophisticated protection mechanisms

## Usage Examples

### Simple Word Mapping
```python
# "curious" → OPENNESS: +0.3, CREATIVITY: +0.2
# "horny" → SEXUAL_EXPERIENCE: +0.25, SEXUAL_COMFORT_LEVEL: +0.2
# "happy" → OPTIMISM: +0.3, EMOTIONAL_STABILITY: +0.2
```

### Complex Phrase Mapping
```python
# "I want to be more creative and social" 
# → CREATIVITY: +0.15, OPENNESS: +0.1, EXTRAVERSION: +0.075, SOCIAL_SKILLS: +0.1
```

### Protection Example
```python
# "I want to change my sexual orientation"
# → No changes to SEXUAL_ORIENTATION (protected trait)
# → May affect other related traits within limits
```

This system provides a powerful foundation for natural language understanding in the Ilanya personality system, enabling sophisticated trait evolution based on user input while maintaining appropriate protection mechanisms. 