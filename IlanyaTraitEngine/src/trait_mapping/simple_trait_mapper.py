"""
Ilanya Trait Engine - Simple Trait Mapper

Simplified trait mapping system that converts natural language input into
trait modifications using word-based mappings and protection rules.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import re

from ..trait_models.trait_types import (
    TraitType, TraitCategory, PERMANENTLY_PROTECTED_TRAITS,
    PARTIALLY_PROTECTED_TRAITS, FULLY_EVOLVABLE_TRAITS
)
from ..trait_models.trait_data import TraitVector, TraitMatrix, TraitData, TraitDataBuilder


@dataclass
class SimpleTraitMappingConfig:
    """Configuration for the simple trait mapping system."""
    
    # Protection configuration
    protection_factor: float = 0.01  # How much protected traits can change
    confidence_threshold: float = 0.7  # Minimum confidence for trait changes
    
    # Word matching configuration
    case_sensitive: bool = False  # Whether word matching is case sensitive
    partial_matching: bool = True  # Whether to match partial words


class SimpleTraitMapper:
    """
    Simple trait mapping system using word-based rules.
    
    Maps natural language input to trait modifications using predefined
    word-trait associations and protection rules.
    """
    
    def __init__(self, config: Optional[SimpleTraitMappingConfig] = None):
        self.config = config or SimpleTraitMappingConfig()
        self.word_mappings = self._initialize_word_mappings()
        self.phrase_mappings = self._initialize_phrase_mappings()
        
    def _initialize_word_mappings(self) -> Dict[str, Dict[TraitType, float]]:
        """Initialize word-to-trait mappings."""
        return {
            # Curiosity and exploration
            "curious": {TraitType.OPENNESS: 0.3, TraitType.CREATIVITY: 0.2},
            "explore": {TraitType.OPENNESS: 0.25, TraitType.ADAPTABILITY: 0.15},
            "adventure": {TraitType.OPENNESS: 0.3, TraitType.RISK_TAKING: 0.2},
            "discover": {TraitType.OPENNESS: 0.25, TraitType.CREATIVITY: 0.15},
            "learn": {TraitType.LEARNING_RATE: 0.25, TraitType.ATTENTION_SPAN: 0.2},
            
            # Sexual content (your examples)
            "lick": {TraitType.SEXUAL_EXPERIENCE: 0.2, TraitType.SEXUAL_COMFORT_LEVEL: 0.1},
            "pussy": {TraitType.SEXUAL_EXPERIENCE: 0.15, TraitType.SEXUAL_COMFORT_LEVEL: 0.1},
            "horny": {TraitType.SEXUAL_EXPERIENCE: 0.25, TraitType.SEXUAL_COMFORT_LEVEL: 0.2},
            "sexy": {TraitType.SEXUAL_EXPERIENCE: 0.2, TraitType.SEXUAL_COMFORT_LEVEL: 0.15},
            "aroused": {TraitType.SEXUAL_EXPERIENCE: 0.2, TraitType.SEXUAL_COMFORT_LEVEL: 0.15},
            "sexual": {TraitType.SEXUAL_EXPERIENCE: 0.15, TraitType.SEXUAL_COMFORT_LEVEL: 0.1},
            
            # Emotional states
            "happy": {TraitType.OPTIMISM: 0.3, TraitType.EMOTIONAL_STABILITY: 0.2},
            "sad": {TraitType.OPTIMISM: -0.2, TraitType.EMOTIONAL_STABILITY: -0.1},
            "angry": {TraitType.EMOTIONAL_STABILITY: -0.2, TraitType.AGREEABLENESS: -0.1},
            "excited": {TraitType.EXTRAVERSION: 0.25, TraitType.OPTIMISM: 0.2},
            "frustrated": {TraitType.EMOTIONAL_STABILITY: -0.15, TraitType.AGREEABLENESS: -0.1},
            "joyful": {TraitType.OPTIMISM: 0.25, TraitType.EMOTIONAL_STABILITY: 0.15},
            "depressed": {TraitType.OPTIMISM: -0.3, TraitType.EMOTIONAL_STABILITY: -0.25},
            
            # Social interactions
            "talk": {TraitType.EXTRAVERSION: 0.2, TraitType.SOCIAL_SKILLS: 0.15},
            "friend": {TraitType.AGREEABLENESS: 0.2, TraitType.EMPATHY: 0.15},
            "help": {TraitType.AGREEABLENESS: 0.25, TraitType.EMPATHY: 0.2},
            "social": {TraitType.EXTRAVERSION: 0.25, TraitType.SOCIAL_SKILLS: 0.2},
            "meet": {TraitType.EXTRAVERSION: 0.2, TraitType.SOCIAL_SKILLS: 0.15},
            "connect": {TraitType.EMPATHY: 0.2, TraitType.SOCIAL_SKILLS: 0.15},
            
            # Cognitive activities
            "think": {TraitType.ANALYTICAL_THINKING: 0.2, TraitType.ATTENTION_SPAN: 0.15},
            "create": {TraitType.CREATIVITY: 0.3, TraitType.OPENNESS: 0.2},
            "analyze": {TraitType.ANALYTICAL_THINKING: 0.25, TraitType.ATTENTION_SPAN: 0.2},
            "solve": {TraitType.ANALYTICAL_THINKING: 0.2, TraitType.PERSISTENCE: 0.15},
            "imagine": {TraitType.CREATIVITY: 0.25, TraitType.OPENNESS: 0.2},
            "focus": {TraitType.ATTENTION_SPAN: 0.3, TraitType.CONSCIENTIOUSNESS: 0.15},
            
            # Behavioral patterns
            "work": {TraitType.CONSCIENTIOUSNESS: 0.2, TraitType.PERSISTENCE: 0.15},
            "plan": {TraitType.CONSCIENTIOUSNESS: 0.25, TraitType.ANALYTICAL_THINKING: 0.15},
            "lead": {TraitType.LEADERSHIP: 0.3, TraitType.EXTRAVERSION: 0.2},
            "organize": {TraitType.CONSCIENTIOUSNESS: 0.25, TraitType.ANALYTICAL_THINKING: 0.15},
            "persist": {TraitType.PERSISTENCE: 0.3, TraitType.CONSCIENTIOUSNESS: 0.2},
            "risk": {TraitType.RISK_TAKING: 0.25, TraitType.ADAPTABILITY: 0.15},
            
            # Personality traits
            "kind": {TraitType.AGREEABLENESS: 0.25, TraitType.EMPATHY: 0.2},
            "patient": {TraitType.EMOTIONAL_STABILITY: 0.2, TraitType.AGREEABLENESS: 0.15},
            "confident": {TraitType.EMOTIONAL_STABILITY: 0.25, TraitType.EXTRAVERSION: 0.2},
            "flexible": {TraitType.ADAPTABILITY: 0.3, TraitType.OPENNESS: 0.2},
            "responsible": {TraitType.CONSCIENTIOUSNESS: 0.25, TraitType.CONSISTENCY: 0.2},
            "optimistic": {TraitType.OPTIMISM: 0.3, TraitType.EMOTIONAL_STABILITY: 0.2},
        }
    
    def _initialize_phrase_mappings(self) -> Dict[str, Dict[TraitType, float]]:
        """Initialize phrase-to-trait mappings for more complex patterns."""
        return {
            "work hard": {TraitType.CONSCIENTIOUSNESS: 0.3, TraitType.PERSISTENCE: 0.25},
            "be kind": {TraitType.AGREEABLENESS: 0.3, TraitType.EMPATHY: 0.25},
            "take charge": {TraitType.LEADERSHIP: 0.35, TraitType.EXTRAVERSION: 0.25},
            "think deeply": {TraitType.ANALYTICAL_THINKING: 0.3, TraitType.ATTENTION_SPAN: 0.25},
            "be creative": {TraitType.CREATIVITY: 0.35, TraitType.OPENNESS: 0.25},
            "stay calm": {TraitType.EMOTIONAL_STABILITY: 0.3, TraitType.SELF_AWARENESS: 0.2},
            "help others": {TraitType.AGREEABLENESS: 0.3, TraitType.EMPATHY: 0.25},
            "learn new": {TraitType.LEARNING_RATE: 0.3, TraitType.OPENNESS: 0.25},
            "be social": {TraitType.EXTRAVERSION: 0.3, TraitType.SOCIAL_SKILLS: 0.25},
            "take risks": {TraitType.RISK_TAKING: 0.35, TraitType.ADAPTABILITY: 0.2},
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
        # Apply word and phrase mappings
        word_modifications = self._apply_word_mappings(text)
        phrase_modifications = self._apply_phrase_mappings(text)
        
        # Combine modifications
        combined_modifications = self._combine_modifications(word_modifications, phrase_modifications)
        
        # Apply modifications to current traits or create new ones
        if current_traits:
            return self._apply_incremental_changes(current_traits, combined_modifications)
        else:
            return self._create_new_trait_data(combined_modifications)
    
    def _apply_word_mappings(self, text: str) -> Dict[TraitType, float]:
        """Apply word-based mappings."""
        modifications = {}
        text_lower = text.lower() if not self.config.case_sensitive else text
        
        for word, trait_changes in self.word_mappings.items():
            if self.config.partial_matching:
                # Check if word appears anywhere in text
                if word in text_lower:
                    for trait, change in trait_changes.items():
                        if trait in modifications:
                            modifications[trait] += change
                        else:
                            modifications[trait] = change
            else:
                # Check for whole word matches
                pattern = r'\b' + re.escape(word) + r'\b'
                if re.search(pattern, text_lower):
                    for trait, change in trait_changes.items():
                        if trait in modifications:
                            modifications[trait] += change
                        else:
                            modifications[trait] = change
        
        return modifications
    
    def _apply_phrase_mappings(self, text: str) -> Dict[TraitType, float]:
        """Apply phrase-based mappings."""
        modifications = {}
        text_lower = text.lower() if not self.config.case_sensitive else text
        
        for phrase, trait_changes in self.phrase_mappings.items():
            if phrase in text_lower:
                for trait, change in trait_changes.items():
                    if trait in modifications:
                        modifications[trait] += change
                    else:
                        modifications[trait] = change
        
        return modifications
    
    def _combine_modifications(self, word_mods: Dict[TraitType, float], 
                             phrase_mods: Dict[TraitType, float]) -> Dict[TraitType, float]:
        """Combine word and phrase modifications."""
        combined = word_mods.copy()
        
        for trait, change in phrase_mods.items():
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
    
    def get_mapping_info(self) -> Dict[str, Any]:
        """Get information about available mappings."""
        return {
            'word_count': len(self.word_mappings),
            'phrase_count': len(self.phrase_mappings),
            'words': list(self.word_mappings.keys()),
            'phrases': list(self.phrase_mappings.keys()),
            'config': {
                'protection_factor': self.config.protection_factor,
                'confidence_threshold': self.config.confidence_threshold,
                'case_sensitive': self.config.case_sensitive,
                'partial_matching': self.config.partial_matching
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Create simple trait mapper
    mapper = SimpleTraitMapper()
    
    # Test with your examples
    print("Testing 'curious':")
    result1 = mapper.map_text_to_traits("curious")
    print(f"Openness: {result1.trait_matrix.traits.get(TraitType.OPENNESS, 'Not found')}")
    print(f"Creativity: {result1.trait_matrix.traits.get(TraitType.CREATIVITY, 'Not found')}")
    
    print("\nTesting 'lick pussy':")
    result2 = mapper.map_text_to_traits("lick pussy")
    print(f"Sexual Experience: {result2.trait_matrix.traits.get(TraitType.SEXUAL_EXPERIENCE, 'Not found')}")
    print(f"Sexual Comfort: {result2.trait_matrix.traits.get(TraitType.SEXUAL_COMFORT_LEVEL, 'Not found')}")
    
    print("\nMapping info:")
    info = mapper.get_mapping_info()
    print(f"Words: {info['word_count']}, Phrases: {info['phrase_count']}") 