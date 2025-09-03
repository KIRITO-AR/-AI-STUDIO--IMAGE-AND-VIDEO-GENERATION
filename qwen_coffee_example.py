#!/usr/bin/env python3
"""
Qwen-Image Clean Example

Clean implementation based on your example code.
This script demonstrates the exact approach you provided for Qwen-Image generation.
"""

from diffusers import DiffusionPipeline
import torch

def main():
    """Main function implementing your Qwen-Image example."""
    
    print("🚀 Starting Qwen-Image generation...")
    
    model_name = "Qwen/Qwen-Image"

    # Load the pipeline (exactly as in your example)
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
        print(f"✅ Using CUDA device with bfloat16")
    else:
        torch_dtype = torch.float32
        device = "cpu"
        print(f"✅ Using CPU device with float32")

    print(f"📦 Loading model: {model_name}")
    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
    pipe = pipe.to(device)
    print("✅ Model loaded successfully!")

    # Positive magic (from your example)
    positive_magic = {
        "en": ", Ultra HD, 4K, cinematic composition.",  # for english prompt
        "zh": ", 超清，4K，电影级构图."  # for chinese prompt
    }

    # Generate image (using your exact prompt)
    prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition'''

    negative_prompt = " "  # using an empty string if you do not have specific concept to remove

    # Generate with different aspect ratios (from your example)
    aspect_ratios = {
        "1:1": (1328, 1328),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1140),
        "3:4": (1140, 1472),
        "3:2": (1584, 1056),
        "2:3": (1056, 1584),
    }

    width, height = aspect_ratios["16:9"]
    print(f"🎨 Generating {width}x{height} image...")

    # Generate image (exactly as in your example)
    image = pipe(
        prompt=prompt + positive_magic["en"],
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        generator=torch.Generator(device=device).manual_seed(42)
    ).images[0]

    # Save image
    output_filename = "qwen_coffee_example.png"
    image.save(output_filename)
    print(f"💾 Image saved as: {output_filename}")
    
    print("🎉 Generation completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you have sufficient GPU memory")
        print("2. Check Hugging Face access to Qwen/Qwen-Image")
        print("3. Try: huggingface-cli login")
        print("4. Install dependencies: pip install diffusers torch transformers")
