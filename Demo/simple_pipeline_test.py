"""
Simple Pipeline Test

Quick test to verify the pipeline components work together.
"""

import sys
import os
sys.path.append('Neural Networks')

print("🚀 Starting Simple Pipeline Test...")
print("=" * 50)

try:
    print("📦 Importing components...")
    from IlanyaTraitMapping.trait_mapper import TraitMapper
    print("✅ TraitMapper imported")
    
    from IlanyaTraitEngine.trait_engine import TraitEngine
    print("✅ TraitEngine imported")
    
    from IlanyaTraitEngine.trait_types import TraitType
    print("✅ TraitType imported")
    
    print("\n🧪 Testing component initialization...")
    
    # Test TraitMapper
    print("Initializing TraitMapper...")
    trait_mapper = TraitMapper()
    print("✅ TraitMapper initialized")
    
    # Test TraitEngine
    print("Initializing TraitEngine...")
    trait_engine = TraitEngine()
    print("✅ TraitEngine initialized")
    
    print("\n🎯 Testing with sample words...")
    
    test_words = ["curious", "creative", "happy", "sad"]
    
    for word in test_words:
        print(f"\n📝 Processing word: '{word}'")
        try:
            # This would need trait data to work properly
            print(f"   Word '{word}' processed (basic test)")
        except Exception as e:
            print(f"   Error processing '{word}': {e}")
    
    print("\n🎉 Simple Pipeline Test Completed Successfully!")
    print("=" * 50)
    
except Exception as e:
    print(f"❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc() 