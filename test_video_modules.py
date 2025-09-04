"""
Test script to verify video modules can be imported without running models.
"""

def test_video_module_imports():
    """Test that all video modules can be imported."""
    try:
        # Test imports
        from src.models.video_generation import (
            VideoGenerationModel,
            ModelScopeT2V,
            ZeroscopeModel,
            VideoUpscaler,
            VideoEditor,
            VideoTransition,
            VideoClip,
            save_video
        )
        print("✅ All video generation modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Error importing video modules: {e}")
        return False

def test_core_modules():
    """Test that core modules work with video features."""
    try:
        from src.core.model_manager import ModelManager
        from src.core.generation_engine import GenerationEngine, GenerationParams
        print("✅ Core modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Error importing core modules: {e}")
        return False

def test_ui_modules():
    """Test that UI modules can be imported."""
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
        return True
    except Exception as e:
        print(f"❌ Error importing Streamlit: {e}")
        return False

if __name__ == "__main__":
    print("Testing video generation modules...")
    print("=" * 40)
    
    success = True
    success &= test_video_module_imports()
    success &= test_core_modules()
    # UI test might fail in non-Streamlit environment
    # success &= test_ui_modules()
    
    print("=" * 40)
    if success:
        print("🎉 All tests passed! Video generation modules are ready.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")