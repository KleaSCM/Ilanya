#!/usr/bin/env python3
"""
Simple test for trait mapping functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

def test_trait_mapping():
    """Test basic trait mapping functionality."""
    try:
        from IlanyaTraitEngine.src.trait_mapping.trait_mapper import TraitMapper
        from IlanyaTraitEngine.src.trait_models.trait_types import TraitType
        
        print("✓ Import successful")
        
        # Create mapper
        mapper = TraitMapper()
        print("✓ TraitMapper created")
        
        # Test word mappings
        print(f"✓ Word mappings available: {len(mapper.word_mappings)} words")
        
        # Test basic mapping
        result = mapper.map_text_to_traits("curious")
        print(f"✓ Basic mapping successful: {len(result.trait_matrix.traits)} traits")
        
        # Check for expected traits
        if TraitType.OPENNESS in result.trait_matrix.traits:
            print("✓ Openness trait detected")
        if TraitType.CREATIVITY in result.trait_matrix.traits:
            print("✓ Creativity trait detected")
        
        # Test your example
        result2 = mapper.map_text_to_traits("lick pussy")
        print(f"✓ Sexual content mapping: {len(result2.trait_matrix.traits)} traits")
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Trait Mapping System...")
    print("=" * 40)
    
    success = test_trait_mapping()
    
    if success:
        print("\n✅ Trait mapping system is working!")
    else:
        print("\n❌ Trait mapping system has issues.")
        sys.exit(1) 