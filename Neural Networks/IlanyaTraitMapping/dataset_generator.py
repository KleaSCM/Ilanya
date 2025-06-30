"""
Ilanya Trait Engine - Dataset Generator

Generates synthetic training data for the trait mapping neural network
using the massive English word dataset and predefined trait associations.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

import json
import os
import random
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict

from IlanyaTraitEngine.trait_types import TraitType, TraitCategory


class TraitMappingDatasetGenerator:
    """
    Generates synthetic training data for trait mapping using word associations.
    
    Uses the massive English word dataset to create realistic text-to-trait
    mappings for training the neural network.
    """
    
    def __init__(self, words_file_path: str = "src/utils/english-words/words.txt"):
        self.words_file_path = words_file_path
        self.words = self._load_words()
        
        # Predefined word-trait associations for generating training data
        self.word_trait_mappings = self._create_word_trait_mappings()
        
        # Categories of words for different trait types
        self.word_categories = self._create_word_categories()
        
    def _load_words(self) -> Set[str]:
        """Load words from the massive word dataset."""
        words = set()
        
        if os.path.exists(self.words_file_path):
            with open(self.words_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip().lower()
                    if word and len(word) > 2:  # Filter out very short words
                        words.add(word)
        
        print(f"Loaded {len(words)} words from dataset")
        return words
    
    def _create_word_trait_mappings(self) -> Dict[str, Dict[TraitType, float]]:
        """Create predefined word-trait associations for generating training data."""
        return {
            # Curiosity and exploration
            "curious": {TraitType.OPENNESS: 0.3, TraitType.CREATIVITY: 0.2},
            "explore": {TraitType.OPENNESS: 0.25, TraitType.ADAPTABILITY: 0.15},
            "adventure": {TraitType.OPENNESS: 0.3, TraitType.RISK_TAKING: 0.2},
            "discover": {TraitType.OPENNESS: 0.25, TraitType.CREATIVITY: 0.15},
            "learn": {TraitType.LEARNING_RATE: 0.25, TraitType.ATTENTION_SPAN: 0.2},
            "study": {TraitType.LEARNING_RATE: 0.3, TraitType.CONSCIENTIOUSNESS: 0.2},
            "research": {TraitType.ANALYTICAL_THINKING: 0.3, TraitType.ATTENTION_SPAN: 0.25},
            
            # Sexual content
            "horny": {TraitType.SEXUAL_EXPERIENCE: 0.25, TraitType.SEXUAL_COMFORT_LEVEL: 0.2},
            "sexy": {TraitType.SEXUAL_EXPERIENCE: 0.2, TraitType.SEXUAL_COMFORT_LEVEL: 0.15},
            "aroused": {TraitType.SEXUAL_EXPERIENCE: 0.2, TraitType.SEXUAL_COMFORT_LEVEL: 0.15},
            "passionate": {TraitType.SEXUAL_EXPERIENCE: 0.15, TraitType.EMOTIONAL_STABILITY: 0.2},
            "intimate": {TraitType.SEXUAL_EXPERIENCE: 0.15, TraitType.EMPATHY: 0.2},
            
            # Emotional states
            "happy": {TraitType.OPTIMISM: 0.3, TraitType.EMOTIONAL_STABILITY: 0.2},
            "sad": {TraitType.OPTIMISM: -0.2, TraitType.EMOTIONAL_STABILITY: -0.1},
            "angry": {TraitType.EMOTIONAL_STABILITY: -0.2, TraitType.AGREEABLENESS: -0.1},
            "excited": {TraitType.EXTRAVERSION: 0.25, TraitType.OPTIMISM: 0.2},
            "joyful": {TraitType.OPTIMISM: 0.25, TraitType.EMOTIONAL_STABILITY: 0.15},
            "depressed": {TraitType.OPTIMISM: -0.3, TraitType.EMOTIONAL_STABILITY: -0.25},
            "anxious": {TraitType.EMOTIONAL_STABILITY: -0.2, TraitType.ATTENTION_SPAN: -0.1},
            "calm": {TraitType.EMOTIONAL_STABILITY: 0.3, TraitType.SELF_AWARENESS: 0.2},
            "peaceful": {TraitType.EMOTIONAL_STABILITY: 0.25, TraitType.OPTIMISM: 0.15},
            
            # Social interactions
            "friendly": {TraitType.AGREEABLENESS: 0.3, TraitType.EXTRAVERSION: 0.2},
            "social": {TraitType.EXTRAVERSION: 0.25, TraitType.SOCIAL_SKILLS: 0.2},
            "helpful": {TraitType.AGREEABLENESS: 0.25, TraitType.EMPATHY: 0.2},
            "kind": {TraitType.AGREEABLENESS: 0.25, TraitType.EMPATHY: 0.2},
            "generous": {TraitType.AGREEABLENESS: 0.3, TraitType.EMPATHY: 0.25},
            "compassionate": {TraitType.EMPATHY: 0.35, TraitType.AGREEABLENESS: 0.25},
            "empathetic": {TraitType.EMPATHY: 0.4, TraitType.SOCIAL_SKILLS: 0.2},
            
            # Cognitive activities
            "think": {TraitType.ANALYTICAL_THINKING: 0.2, TraitType.ATTENTION_SPAN: 0.15},
            "analyze": {TraitType.ANALYTICAL_THINKING: 0.25, TraitType.ATTENTION_SPAN: 0.2},
            "create": {TraitType.CREATIVITY: 0.3, TraitType.OPENNESS: 0.2},
            "imagine": {TraitType.CREATIVITY: 0.25, TraitType.OPENNESS: 0.2},
            "innovate": {TraitType.CREATIVITY: 0.3, TraitType.OPENNESS: 0.25},
            "solve": {TraitType.ANALYTICAL_THINKING: 0.2, TraitType.PERSISTENCE: 0.15},
            "focus": {TraitType.ATTENTION_SPAN: 0.3, TraitType.CONSCIENTIOUSNESS: 0.15},
            "concentrate": {TraitType.ATTENTION_SPAN: 0.25, TraitType.CONSCIENTIOUSNESS: 0.2},
            
            # Behavioral patterns
            "work": {TraitType.CONSCIENTIOUSNESS: 0.2, TraitType.PERSISTENCE: 0.15},
            "plan": {TraitType.CONSCIENTIOUSNESS: 0.25, TraitType.ANALYTICAL_THINKING: 0.15},
            "organize": {TraitType.CONSCIENTIOUSNESS: 0.25, TraitType.ANALYTICAL_THINKING: 0.15},
            "lead": {TraitType.LEADERSHIP: 0.3, TraitType.EXTRAVERSION: 0.2},
            "manage": {TraitType.LEADERSHIP: 0.25, TraitType.CONSCIENTIOUSNESS: 0.2},
            "persist": {TraitType.PERSISTENCE: 0.3, TraitType.CONSCIENTIOUSNESS: 0.2},
            "determined": {TraitType.PERSISTENCE: 0.3, TraitType.CONSCIENTIOUSNESS: 0.25},
            "ambitious": {TraitType.PERSISTENCE: 0.25, TraitType.LEADERSHIP: 0.2},
            
            # Personality traits
            "confident": {TraitType.EMOTIONAL_STABILITY: 0.25, TraitType.EXTRAVERSION: 0.2},
            "optimistic": {TraitType.OPTIMISM: 0.3, TraitType.EMOTIONAL_STABILITY: 0.2},
            "patient": {TraitType.EMOTIONAL_STABILITY: 0.2, TraitType.AGREEABLENESS: 0.15},
            "flexible": {TraitType.ADAPTABILITY: 0.3, TraitType.OPENNESS: 0.2},
            "responsible": {TraitType.CONSCIENTIOUSNESS: 0.25, TraitType.CONSISTENCY: 0.2},
            "reliable": {TraitType.CONSISTENCY: 0.3, TraitType.CONSCIENTIOUSNESS: 0.2},
            "honest": {TraitType.AGREEABLENESS: 0.25, TraitType.CONSISTENCY: 0.2},
            "brave": {TraitType.RISK_TAKING: 0.3, TraitType.EMOTIONAL_STABILITY: 0.2},
            "courageous": {TraitType.RISK_TAKING: 0.25, TraitType.EMOTIONAL_STABILITY: 0.2},
        }
    
    def _create_word_categories(self) -> Dict[str, List[str]]:
        """Create categories of words for different trait types."""
        return {
            "positive_emotions": ["happy", "joyful", "excited", "cheerful", "delighted", "thrilled", "ecstatic"],
            "negative_emotions": ["sad", "angry", "depressed", "anxious", "frustrated", "worried", "scared"],
            "social_words": ["friendly", "social", "helpful", "kind", "generous", "compassionate", "empathetic"],
            "cognitive_words": ["think", "analyze", "create", "imagine", "innovate", "solve", "focus"],
            "work_words": ["work", "plan", "organize", "lead", "manage", "persist", "determined"],
            "sexual_words": ["horny", "sexy", "aroused", "passionate", "intimate", "romantic", "desire"],
            "personality_words": ["confident", "optimistic", "patient", "flexible", "responsible", "reliable", "honest"],
        }
    
    def generate_synthetic_dataset(self, num_examples: int = 10000) -> List[Dict[str, Any]]:
        """
        Generate synthetic training data using word associations and random combinations.
        
        Args:
            num_examples: Number of training examples to generate
            
        Returns:
            List of training examples in the format expected by the neural network
        """
        dataset = []
        
        # Add predefined word mappings
        for word, trait_changes in self.word_trait_mappings.items():
            if word in self.words:  # Only include words that exist in our dataset
                traits_dict = {trait.value.upper(): value for trait, value in trait_changes.items()}
                dataset.append({
                    "text": word,
                    "traits": traits_dict
                })
        
        # Generate random word combinations
        for _ in range(num_examples - len(dataset)):
            example = self._generate_random_example()
            if example:
                dataset.append(example)
        
        print(f"Generated {len(dataset)} training examples")
        return dataset
    
    def _generate_random_example(self) -> Optional[Dict[str, Any]]:
        """Generate a random training example."""
        # Randomly select a word category
        category = random.choice(list(self.word_categories.keys()))
        words = self.word_categories[category]
        
        # Select 1-3 words from the category
        num_words = random.randint(1, 3)
        selected_words = random.sample(words, min(num_words, len(words)))
        
        # Create text input
        text = " ".join(selected_words)
        
        # Generate trait modifications based on category
        trait_modifications = self._generate_traits_for_category(category)
        
        if trait_modifications:
            traits_dict = {trait.value.upper(): value for trait, value in trait_modifications.items()}
            return {
                "text": text,
                "traits": traits_dict
            }
        
        return None
    
    def _generate_traits_for_category(self, category: str) -> Dict[TraitType, float]:
        """Generate trait modifications for a given word category."""
        base_traits = {
            "positive_emotions": {TraitType.OPTIMISM: 0.2, TraitType.EMOTIONAL_STABILITY: 0.15},
            "negative_emotions": {TraitType.OPTIMISM: -0.2, TraitType.EMOTIONAL_STABILITY: -0.15},
            "social_words": {TraitType.AGREEABLENESS: 0.2, TraitType.EXTRAVERSION: 0.15},
            "cognitive_words": {TraitType.ANALYTICAL_THINKING: 0.2, TraitType.ATTENTION_SPAN: 0.15},
            "work_words": {TraitType.CONSCIENTIOUSNESS: 0.2, TraitType.PERSISTENCE: 0.15},
            "sexual_words": {TraitType.SEXUAL_EXPERIENCE: 0.2, TraitType.SEXUAL_COMFORT_LEVEL: 0.15},
            "personality_words": {TraitType.EMOTIONAL_STABILITY: 0.15, TraitType.CONSISTENCY: 0.15},
        }
        
        if category in base_traits:
            traits = base_traits[category].copy()
            
            # Add some random variation
            for trait in traits:
                variation = random.uniform(-0.1, 0.1)
                traits[trait] = max(-0.5, min(0.5, traits[trait] + variation))
            
            return traits
        
        return {}
    
    def save_dataset(self, dataset: List[Dict[str, Any]], output_path: str = "data/trait_mappings.json"):
        """Save the generated dataset to a JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"Dataset saved to {output_path}")
    
    def create_phrase_dataset(self, num_phrases: int = 5000) -> List[Dict[str, Any]]:
        """Create a dataset of phrases and sentences."""
        phrases = []
        
        # Template phrases
        templates = [
            "I am feeling {emotion}",
            "I want to be more {trait}",
            "I need to {action}",
            "I love to {activity}",
            "I feel {emotion} about {topic}",
            "I want to {action} and {action2}",
            "I am {trait} and {trait2}",
            "I feel {emotion} when I {activity}",
        ]
        
        emotions = ["happy", "sad", "excited", "angry", "anxious", "confident", "worried", "joyful"]
        traits = ["creative", "social", "organized", "helpful", "confident", "patient", "flexible", "responsible"]
        actions = ["work hard", "learn new things", "help others", "be creative", "stay calm", "take risks", "plan carefully"]
        activities = ["explore", "create", "socialize", "study", "exercise", "meditate", "travel"]
        topics = ["life", "work", "relationships", "learning", "growth", "success", "challenges"]
        
        for _ in range(num_phrases):
            template = random.choice(templates)
            
            # Fill template with random words
            phrase = template.format(
                emotion=random.choice(emotions),
                trait=random.choice(traits),
                action=random.choice(actions),
                activity=random.choice(activities),
                topic=random.choice(topics),
                action2=random.choice(actions),
                trait2=random.choice(traits)
            )
            
            # Generate trait modifications based on the words in the phrase
            trait_modifications = self._analyze_phrase_traits(phrase)
            
            if trait_modifications:
                traits_dict = {trait.value.upper(): value for trait, value in trait_modifications.items()}
                phrases.append({
                    "text": phrase,
                    "traits": traits_dict
                })
        
        print(f"Generated {len(phrases)} phrase examples")
        return phrases
    
    def _analyze_phrase_traits(self, phrase: str) -> Dict[TraitType, float]:
        """Analyze a phrase and determine trait modifications."""
        phrase_lower = phrase.lower()
        modifications = {}
        
        # Check for word matches
        for word, trait_changes in self.word_trait_mappings.items():
            if word in phrase_lower:
                for trait, change in trait_changes.items():
                    if trait in modifications:
                        modifications[trait] += change * 0.5  # Reduce impact for phrases
                    else:
                        modifications[trait] = change * 0.5
        
        return modifications


# Example usage
if __name__ == "__main__":
    # Create dataset generator
    generator = TraitMappingDatasetGenerator()
    
    # Generate word-based dataset
    print("Generating word-based dataset...")
    word_dataset = generator.generate_synthetic_dataset(num_examples=5000)
    
    # Generate phrase-based dataset
    print("Generating phrase-based dataset...")
    phrase_dataset = generator.create_phrase_dataset(num_phrases=3000)
    
    # Combine datasets
    combined_dataset = word_dataset + phrase_dataset
    
    # Save the combined dataset
    generator.save_dataset(combined_dataset, "data/combined_trait_mappings.json")
    
    print(f"Total training examples: {len(combined_dataset)}")
    
    # Show some examples
    print("\nSample training examples:")
    for i, example in enumerate(combined_dataset[:5]):
        print(f"{i+1}. Text: '{example['text']}'")
        print(f"   Traits: {example['traits']}")
        print() 