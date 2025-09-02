"""
Launch script for AI Generation Studio.
Provides easy startup options for different interfaces.
"""

import sys
import subprocess
import argparse
from pathlib import Path

def launch_streamlit():
    """Launch the Streamlit interface."""
    streamlit_app = Path(__file__).parent / "src" / "ui" / "streamlit_app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(streamlit_app)]
    
    print("🚀 Launching Streamlit interface...")
    print("Open your browser to: http://localhost:8501")
    
    subprocess.run(cmd)

def launch_desktop():
    """Launch the desktop interface (placeholder)."""
    print("🚀 Desktop interface not yet implemented")
    print("Use 'python launch.py --streamlit' for now")

def main():
    parser = argparse.ArgumentParser(description="AI Generation Studio Launcher")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--streamlit", action="store_true", 
                      help="Launch Streamlit web interface")
    group.add_argument("--desktop", action="store_true",
                      help="Launch desktop PyQt interface")
    
    args = parser.parse_args()
    
    if args.streamlit:
        launch_streamlit()
    elif args.desktop:
        launch_desktop()
    else:
        # Default to streamlit
        print("No interface specified, launching Streamlit...")
        launch_streamlit()

if __name__ == "__main__":
    main()