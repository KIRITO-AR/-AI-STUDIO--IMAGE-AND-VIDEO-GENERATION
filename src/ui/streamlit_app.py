"""
Streamlit prototype frontend for AI Generation Studio.
Provides a quick and easy web interface for image and video generation.
"""

# Try to import streamlit, but handle gracefully if not available
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Streamlit not available. Please install streamlit: pip install streamlit")
    import sys
    sys.exit(1)

import logging
import sys
import os
from pathlib import Path
import time
import zipfile
from typing import List, Dict

# Add src to path for imports
current_file_path = Path(__file__).resolve()
src_path = current_file_path.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from core import get_generation_engine, get_model_manager, GenerationParams
    from utils import get_device_info, clear_gpu_cache
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please make sure you're running from the project root directory")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="AI Generation Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .generation-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: #f9f9f9;
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'generation_engine' not in st.session_state:
        st.session_state.generation_engine = get_generation_engine()
    
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = get_model_manager()
    
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []
    
    if 'current_generation' not in st.session_state:
        st.session_state.current_generation = None

def display_header():
    """Display the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🎨 AI Generation Studio</h1>
        <p>Create stunning images and videos with AI</p>
    </div>
    """, unsafe_allow_html=True)

def display_system_info():
    """Display system information in sidebar."""
    st.sidebar.header("🔧 System Info")
    
    # Get device info
    device, gpu_detector = get_device_info()
    
    st.sidebar.write(f"**Device:** {device}")
    
    if gpu_detector.gpus:
        best_gpu = gpu_detector.get_best_gpu()
        if best_gpu:
            st.sidebar.write(f"**GPU:** {best_gpu.name}")
            st.sidebar.write(f"**VRAM:** {best_gpu.memory_total}MB")
            
            # Memory bar
            memory_used_percent = (best_gpu.memory_used / best_gpu.memory_total) * 100
            st.sidebar.progress(memory_used_percent / 100)
            st.sidebar.caption(f"Memory usage: {memory_used_percent:.1f}%")
        else:
            st.sidebar.warning("GPU detected but not accessible")
    else:
        st.sidebar.warning("No GPU detected - using CPU")
    
    # Model info
    current_model = st.session_state.model_manager.get_current_model()
    if current_model:
        st.sidebar.write(f"**Model:** {current_model.name}")
        st.sidebar.success("✓ Model loaded")
    else:
        st.sidebar.warning("No model loaded")

def model_selection_sidebar():
    """Model selection and management in sidebar."""
    st.sidebar.header("🤖 Model Selection")
    
    # Get available models and group by type
    available_models = st.session_state.model_manager.list_models()
    
    # Separate models by type
    image_models = [model for model in available_models if not model.supports_video]
    video_models = [model for model in available_models if model.supports_video]
    
    # Model type selection
    model_type = st.sidebar.radio(
        "Model Type",
        ["Image Models", "Video Models"],
        key="model_type_selection"
    )
    
    # Select models based on type
    if model_type == "Image Models":
        models_to_show = image_models
        st.sidebar.info("🖼️ Image generation models")
    else:
        models_to_show = video_models
        st.sidebar.info("🎬 Video generation models")
    
    # Create model name to model mapping
    model_names = {model.name: model for model in models_to_show}
    
    # Current model
    current_model = st.session_state.model_manager.get_current_model()
    current_name = current_model.name if current_model else None
    
    # Model selection
    if models_to_show:
        selected_model_name = st.sidebar.selectbox(
            "Choose Model",
            options=list(model_names.keys()),
            index=list(model_names.keys()).index(current_name) if current_name in model_names else 0
        )
        
        selected_model = model_names[selected_model_name]
        
        # Model info
        with st.sidebar.expander(f"ℹ️ {selected_model.name} Info"):
            st.write(f"**Type:** {selected_model.model_type.value}")
            st.write(f"**Memory Required:** {selected_model.memory_requirement}MB")
            st.write(f"**Video Support:** {'✓' if selected_model.supports_video else '✗'}")
            st.write(f"**Description:** {selected_model.description}")
        
        # Load model button
        if st.sidebar.button("🔄 Load Model"):
            if current_model and current_model.name == selected_model.name:
                st.sidebar.success("Model already loaded!")
            else:
                with st.spinner(f"Loading {selected_model.name}..."):
                    # Find model key
                    model_key = None
                    for key, model in st.session_state.model_manager.available_models.items():
                        if model.name == selected_model.name:
                            model_key = key
                            break
                    
                    if model_key and st.session_state.model_manager.load_model(model_key):
                        st.sidebar.success(f"✓ {selected_model.name} loaded!")
                        st.rerun()
                    else:
                        st.sidebar.error("Failed to load model")
    else:
        st.sidebar.warning(f"No {model_type.lower()} available")

def generation_interface():
    """Main generation interface."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("✨ Generate")
        
        # Check if model is loaded
        if not st.session_state.model_manager.is_model_loaded():
            st.warning("Please load a model first using the sidebar.")
            return
        
        current_model = st.session_state.model_manager.get_current_model()
        
        # Show model-specific interface
        if current_model.supports_video:
            # Video generation interface
            st.subheader("🎬 Video Generation")
            st.caption(f"Using: {current_model.name}")
            
            # Show model-specific tips
            if "zeroscope" in current_model.model_id.lower():
                st.info("💡 Zeroscope tip: For best results, use detailed prompts describing actions and scenes")
            elif "modelscope" in current_model.model_id.lower():
                st.info("💡 ModelScope tip: Works well with cinematic and storytelling prompts")
            else:
                st.info("💡 AnimateDiff tip: Great for animated scenes and character movements")
            
            generate_videos_tab()
        else:
            # Image generation interface
            st.subheader("🖼️ Image Generation")
            st.caption(f"Using: {current_model.name}")
            generate_images_tab()
    
    with col2:
        display_generation_settings()

def generate_images_tab():
    """Image generation interface."""
    
    # Prompt input
    prompt = st.text_area(
        "Prompt",
        placeholder="Describe what you want to generate...",
        height=100,
        key="image_prompt"
    )
    
    negative_prompt = st.text_area(
        "Negative Prompt (Optional)",
        placeholder="What you don't want in the image...",
        height=60,
        key="image_negative_prompt"
    )
    
    # Generation button
    if st.button("🎨 Generate Images", key="generate_images"):
        if not prompt.strip():
            st.error("Please enter a prompt")
            return
        
        generate_images(prompt, negative_prompt)

def generate_videos_tab():
    """Video generation interface."""
    
    # Prompt input
    prompt = st.text_area(
        "Prompt",
        placeholder="Describe the video you want to generate...",
        height=100,
        key="video_prompt"
    )
    
    negative_prompt = st.text_area(
        "Negative Prompt (Optional)",
        placeholder="What you don't want in the video...",
        height=60,
        key="video_negative_prompt"
    )
    
    # Video-specific settings
    col1, col2, col3 = st.columns(3)
    with col1:
        num_frames = st.slider("Number of Frames", 8, 64, 16, key="video_frames")
    with col2:
        fps = st.slider("FPS", 4, 30, 8, key="video_fps")
    with col3:
        video_quality = st.select_slider(
            "Quality", 
            options=["Low", "Medium", "High"], 
            value="Medium",
            key="video_quality"
        )
    
    # Advanced video settings
    with st.expander("🎬 Advanced Video Settings"):
        col1, col2 = st.columns(2)
        with col1:
            video_duration = st.slider("Duration (seconds)", 1, 8, 2, key="video_duration")
            # Update frames based on duration and FPS
            calculated_frames = video_duration * fps
            st.info(f"Calculated frames: {calculated_frames}")
        with col2:
            interpolation = st.checkbox("Frame Interpolation", value=False, key="video_interpolation")
            loop_video = st.checkbox("Loop Video", value=True, key="video_loop")
        
        # Video upscaling option
        upscale_video = st.checkbox("Upscale Video (High VRAM)", value=False, key="video_upscale")
        if upscale_video:
            st.warning("⚠️ Video upscaling requires significant VRAM (12GB+ recommended)")
    
    # Generation button
    if st.button("🎬 Generate Video", key="generate_video", use_container_width=True):
        if not prompt.strip():
            st.error("Please enter a prompt")
            return
        
        generate_video(prompt, negative_prompt, num_frames, fps, upscale_video)

def batch_generation_tab():
    """Batch generation interface."""
    current_model = st.session_state.model_manager.get_current_model()
    
    if current_model and current_model.supports_video:
        st.subheader("Batch Video Generation")
        st.caption(f"Using: {current_model.name}")
    else:
        st.subheader("Batch Image Generation")
    
    prompts_text = st.text_area(
        "Prompts (one per line)",
        placeholder="Enter multiple prompts, one per line...",
        height=150,
        key="batch_prompts"
    )
    
    negative_prompt = st.text_input(
        "Global Negative Prompt (Optional)",
        placeholder="Applied to all generations...",
        key="batch_negative_prompt"
    )
    
    if st.button("🔄 Generate Batch", key="generate_batch"):
        if not prompts_text.strip():
            st.error("Please enter at least one prompt")
            return
        
        prompts = [p.strip() for p in prompts_text.split('\n') if p.strip()]
        generate_batch(prompts, negative_prompt)

def display_generation_settings():
    """Display generation settings panel."""
    st.header("⚙️ Settings")
    
    # Basic settings
    with st.expander("📐 Size & Quality", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            width = st.selectbox("Width", [512, 768, 1024], index=0, key="gen_width")
        with col2:
            height = st.selectbox("Height", [512, 768, 1024], index=0, key="gen_height")
        
        steps = st.slider("Inference Steps", 10, 50, 20, key="gen_steps")
        guidance = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5, key="gen_guidance")
        num_images = st.slider("Number of Images", 1, 4, 1, key="gen_num_images")
    
    # Advanced settings
    with st.expander("🔬 Advanced"):
        seed = st.number_input("Seed (-1 for random)", -1, 2**31-1, -1, key="gen_seed")
        scheduler = st.selectbox(
            "Scheduler",
            ["ddim", "ddpm", "euler_a", "euler", "dpm"],
            index=0,
            key="gen_scheduler"
        )
    
    # Performance actions
    with st.expander("🚀 Performance"):
        if st.button("🗑️ Clear GPU Cache"):
            clear_gpu_cache()
            st.success("GPU cache cleared!")
        
        if st.button("📊 Show GPU Stats"):
            display_gpu_stats()

def display_gpu_stats():
    """Display detailed GPU statistics."""
    device, gpu_detector = get_device_info()
    
    if gpu_detector.gpus:
        for i, gpu in enumerate(gpu_detector.gpus):
            st.write(f"**GPU {i}: {gpu.name}**")
            st.write(f"Memory: {gpu.memory_used}MB / {gpu.memory_total}MB")
            st.write(f"Utilization: {gpu.utilization}%")
            if gpu.temperature:
                st.write(f"Temperature: {gpu.temperature}°C")
    else:
        st.write("No GPUs detected")

def generate_images(prompt: str, negative_prompt: str = ""):
    """Generate images with current settings."""
    # Get settings from session state
    params = GenerationParams(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=st.session_state.gen_width,
        height=st.session_state.gen_height,
        num_inference_steps=st.session_state.gen_steps,
        guidance_scale=st.session_state.gen_guidance,
        num_images_per_prompt=st.session_state.gen_num_images,
        seed=st.session_state.gen_seed if st.session_state.gen_seed >= 0 else None,
        scheduler=st.session_state.gen_scheduler
    )
    
    # Create progress containers
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Set up callbacks
    def progress_callback(step, total_steps, progress):
        progress_bar.progress(progress)
        status_text.text(f"Step {step}/{total_steps}")
    
    def status_callback(status):
        status_text.text(status)
    
    st.session_state.generation_engine.set_progress_callback(progress_callback)
    st.session_state.generation_engine.set_status_callback(status_callback)
    
    # Generate
    start_time = time.time()
    result = st.session_state.generation_engine.generate_image(params)
    
    # Clear progress
    progress_bar.empty()
    status_text.empty()
    
    # Display results
    display_generation_result(result, start_time)

def generate_video(prompt: str, negative_prompt: str = "", num_frames: int = 16, fps: int = 8, upscale: bool = False):
    """Generate video with current settings."""
    params = GenerationParams(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=st.session_state.gen_width,
        height=st.session_state.gen_height,
        num_inference_steps=st.session_state.gen_steps,
        guidance_scale=st.session_state.gen_guidance,
        seed=st.session_state.gen_seed if st.session_state.gen_seed >= 0 else None,
        scheduler=st.session_state.gen_scheduler,
        num_frames=num_frames,
        fps=fps
    )
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def progress_callback(step, total_steps, progress):
        progress_bar.progress(progress)
        status_text.text(f"Step {step}/{total_steps}")
    
    def status_callback(status):
        status_text.text(status)
    
    st.session_state.generation_engine.set_progress_callback(progress_callback)
    st.session_state.generation_engine.set_status_callback(status_callback)
    
    # Generate
    start_time = time.time()
    result = st.session_state.generation_engine.generate_video(params)
    
    # Video upscaling if requested
    if upscale and not result.error and result.images:
        try:
            status_text.text("Upscaling video...")
            # Here we would implement upscaling using our new VideoUpscaler class
            # For now, we'll just show a message that this feature is being prepared
            st.info("💡 Video upscaling feature is being prepared. This will be available in the next update.")
        except Exception as e:
            st.warning(f"Video upscaling failed: {e}")
    
    # Clear progress
    progress_bar.empty()
    status_text.empty()
    
    # Display results
    display_generation_result(result, start_time, is_video=True)

def generate_batch(prompts: List[str], negative_prompt: str = ""):
    """Generate batch of images."""
    params = GenerationParams(
        prompt="",  # Will be overridden for each prompt
        negative_prompt=negative_prompt,
        width=st.session_state.gen_width,
        height=st.session_state.gen_height,
        num_inference_steps=st.session_state.gen_steps,
        guidance_scale=st.session_state.gen_guidance,
        num_images_per_prompt=1,  # One image per prompt for batch
        scheduler=st.session_state.gen_scheduler
    )
    
    # Progress tracking
    batch_progress = st.progress(0)
    status_text = st.empty()
    current_image = st.empty()
    
    def batch_progress_callback(current, total):
        progress = current / total
        batch_progress.progress(progress)
        status_text.text(f"Processing {current}/{total} prompts...")
    
    # Generate batch
    start_time = time.time()
    results = st.session_state.generation_engine.batch_generate(
        prompts, params, batch_progress_callback
    )
    
    # Clear progress
    batch_progress.empty()
    status_text.empty()
    current_image.empty()
    
    # Display batch results
    display_batch_results(results, start_time)

def display_generation_result(result, start_time, is_video=False):
    """Display the result of a generation."""
    if result.error:
        st.error(f"Generation failed: {result.error}")
        return
    
    if not result.images:
        st.warning("No images generated")
        return
    
    st.success(f"Generated in {result.generation_time:.2f}s (Seed: {result.seed_used})")
    
    # Display images
    if len(result.images) == 1:
        st.image(result.images[0], use_container_width=True)
    else:
        cols = st.columns(min(len(result.images), 3))
        for i, img in enumerate(result.images):
            with cols[i % len(cols)]:
                st.image(img, use_container_width=True)
    
    # Save options
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("💾 Save Images"):
            saved_paths = result.save_images("outputs")
            st.success(f"Saved {len(saved_paths)} images to outputs/")
    
    if is_video and len(result.images) > 1:
        with col2:
            if st.button("🎬 Save as Video"):
                try:
                    video_path = st.session_state.generation_engine.save_result_as_video(
                        result, "outputs/generated_video.mp4"
                    )
                    st.success(f"Video saved to {video_path}")
                except Exception as e:
                    st.error(f"Failed to save video: {e}")
        
        with col3:
            if st.button("⬆️ Upscale Video"):
                try:
                    # Check if we have enough VRAM for upscaling
                    gpu_detector = st.session_state.model_manager.gpu_detector
                    best_gpu = gpu_detector.get_best_gpu()
                    
                    if best_gpu and best_gpu.memory_total >= 12000:  # 12GB+
                        video_path = st.session_state.generation_engine.upscale_video(
                            result, "outputs/upscaled_video.mp4"
                        )
                        st.success(f"Upscaled video saved to {video_path}")
                    else:
                        st.warning("⚠️ Video upscaling requires 12GB+ VRAM. Your system may not support this feature.")
                        # Still allow basic upscaling
                        video_path = st.session_state.generation_engine.upscale_video(
                            result, "outputs/upscaled_video.mp4"
                        )
                        st.success(f"Basic upscaled video saved to {video_path}")
                except Exception as e:
                    st.error(f"Failed to upscale video: {e}")
        
        with col4:
            # Video editing options
            with st.expander("✂️ Edit Video"):
                edit_operation = st.selectbox(
                    "Edit Operation",
                    ["trim", "speed", "reverse", "loop"]
                )
                
                if edit_operation == "trim":
                    start_frame = st.number_input("Start Frame", 0, len(result.images)-1, 0)
                    end_frame = st.number_input("End Frame", start_frame+1, len(result.images), len(result.images))
                    edit_params = {"start_frame": start_frame, "end_frame": end_frame}
                elif edit_operation == "speed":
                    speed_factor = st.slider("Speed Factor", 0.1, 3.0, 1.0, 0.1)
                    edit_params = {"speed_factor": speed_factor}
                elif edit_operation == "loop":
                    loops = st.number_input("Number of Loops", 1, 10, 2)
                    edit_params = {"loops": loops}
                else:
                    edit_params = {}
                
                if st.button("Apply Edit"):
                    try:
                        edited_result = st.session_state.generation_engine.edit_video(
                            result, edit_operation, **edit_params
                        )
                        st.session_state.current_edited_video = edited_result
                        st.success(f"Video edited! New length: {len(edited_result.images)} frames")
                    except Exception as e:
                        st.error(f"Failed to edit video: {e}")
        
        with col5:
            if st.button("📋 Copy Prompt"):
                st.code(result.params.prompt, language="text")
    
    # Add to history
    st.session_state.generation_history.append(result)

def display_batch_results(results, start_time):
    """Display results from batch generation."""
    successful = [r for r in results if not r.error]
    failed = [r for r in results if r.error]
    
    st.success(f"Batch completed in {time.time() - start_time:.2f}s")
    st.info(f"Successful: {len(successful)}, Failed: {len(failed)}")
    
    if failed:
        with st.expander("❌ Failed Generations"):
            for result in failed:
                st.error(f"'{result.params.prompt[:50]}...': {result.error}")
    
    # Display successful results
    if successful:
        st.subheader("✅ Generated Images")
        
        # Grid display
        cols_per_row = 3
        for i in range(0, len(successful), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, result in enumerate(successful[i:i+cols_per_row]):
                with cols[j]:
                    if result.images:
                        st.image(result.images[0], use_container_width=True)
                        st.caption(f"{result.params.prompt[:30]}...")
                        st.caption(f"Seed: {result.seed_used}")
        
        # Batch save option
        if st.button("💾 Save All Images"):
            total_saved = 0
            for result in successful:
                saved_paths = result.save_images("outputs/batch")
                total_saved += len(saved_paths)
            st.success(f"Saved {total_saved} images to outputs/batch/")

def display_history():
    """Display generation history."""
    if not st.session_state.generation_history:
        st.info("No generation history yet")
        return
    
    st.header("📈 Generation History")
    
    for i, result in enumerate(reversed(st.session_state.generation_history[-10:])):
        with st.expander(f"#{len(st.session_state.generation_history) - i}: {result.params.prompt[:50]}..."):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if result.images:
                    st.image(result.images[0], use_container_width=True)
            
            with col2:
                st.write(f"**Prompt:** {result.params.prompt}")
                st.write(f"**Time:** {result.generation_time:.2f}s")
                st.write(f"**Seed:** {result.seed_used}")
                st.write(f"**Size:** {result.params.width}x{result.params.height}")
                st.write(f"**Steps:** {result.params.num_inference_steps}")

def main():
    """Main application function."""
    initialize_session_state()
    
    display_header()
    
    # Sidebar
    display_system_info()
    model_selection_sidebar()
    
    # Main content with separate tabs for Image and Video
    tab1, tab2, tab3 = st.tabs(["✨ Generate", "📋 Batch", "📈 History"])
    
    with tab1:
        generation_interface()
    
    with tab2:
        st.header("📝 Batch Generation")
        # Check if model is loaded
        if not st.session_state.model_manager.is_model_loaded():
            st.warning("Please load a model first using the sidebar.")
        else:
            batch_generation_tab()
    
    with tab3:
        display_history()

if __name__ == "__main__":
    main()