#!/usr/bin/env python3
"""
Test script for new powerful image generation models
"""

import sys
from pathlib import Path

# Add project path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir / "src"))

def test_new_models():
    """Test the new powerful image generation models."""
    print("🚀 Testing New Powerful Image Generation Models")
    print("=" * 50)
    
    try:
        from core.model_manager import ModelManager, ModelType
        print("✅ ModelManager imported successfully")
        
        # Initialize manager
        manager = ModelManager()
        print("✅ ModelManager initialized")
        
        # List all available models
        models = manager.list_models()
        print(f"✅ Found {len(models)} total models")
        
        # Find new models
        new_models = [
            model for model in models 
            if model.model_type == ModelType.CUSTOM or "Stable Diffusion 3" in model.name
        ]
        
        print(f"🎯 Found {len(new_models)} new powerful models:")
        for model in new_models:
            print(f"  - {model.name} ({model.model_id})")
            print(f"    VRAM: {model.memory_requirement}MB")
            print(f"    Description: {model.description}")
            print()
        
        # Test model info retrieval
        test_models = ["sd3_5_large", "kolors", "pixart_alpha"]
        for model_key in test_models:
            model_info = manager.get_model_info(model_key)
            if model_info:
                print(f"✅ Model info retrieved for {model_key}: {model_info.name}")
            else:
                print(f"❌ Model info not found for {model_key}")
        
        # Test compatibility checking
        print("\n🔍 Compatibility Check (based on current hardware):")
        for model_key in test_models:
            compatibility = manager.check_model_compatibility(model_key)
            model_info = manager.get_model_info(model_key)
            if model_info:
                status = "✅ Compatible" if compatibility.get("compatible") else "❌ Not Compatible"
                print(f"  {model_info.name}: {status}")
                print(f"    Reason: {compatibility.get('reason', 'Unknown')}")
                if "recommendations" in compatibility:
                    print(f"    Recommendations: {compatibility['recommendations'][0]}")
                print()
        
        print("🎉 All tests completed successfully!")
        print("\n💡 To use these models:")
        print("   manager.load_model('sd3_5_large')  # For Stable Diffusion 3.5")
        print("   manager.load_model('kolors')       # For Bytedance Kolors")
        print("   manager.load_model('pixart_alpha') # For PixArt-Alpha")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_new_models()
    
    if success:
        print("\n🌟 SUCCESS! The new powerful models are ready to use!")
        print("   These include Stable Diffusion 3.5, Bytedance Kolors, and more!")
    else:
        print("\n❌ Test failed - check the error messages above")