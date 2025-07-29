#!/usr/bin/env python3
"""
Setup script for SmartGitMergePT with Agentic AI Tracking
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        sys.exit(1)

def check_git():
    """Check if Git is available"""
    try:
        subprocess.check_call(["git", "--version"], stdout=subprocess.DEVNULL)
        print("✅ Git detected")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Git not found. Please install Git first.")
        sys.exit(1)

def check_webcam():
    """Check if webcam is available"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Webcam detected")
            cap.release()
        else:
            print("⚠️  Webcam not detected (optional)")
    except ImportError:
        print("⚠️  OpenCV not available, webcam check skipped")
    except Exception as e:
        print(f"⚠️  Webcam check failed: {e}")

def setup_configuration():
    """Setup initial configuration"""
    print("🔧 Setting up configuration...")
    
    try:
        from src.config_manager import ConfigManager
        
        config_manager = ConfigManager(os.getcwd())
        
        # Create default configuration
        config_manager.save_config()
        
        print("✅ Configuration initialized")
        
        # Show current configuration
        config_manager.print_current_config()
        
    except ImportError as e:
        print(f"⚠️  Could not setup configuration: {e}")

def check_openai_key():
    """Check for OpenAI API key"""
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print("✅ OpenAI API key detected")
    else:
        print("⚠️  OpenAI API key not found")
        print("   Set OPENAI_API_KEY environment variable for AI validation features")
        print("   Example: export OPENAI_API_KEY='your-api-key'")

def create_demo_environment():
    """Create demo environment"""
    print("🎬 Setting up demo environment...")
    
    # Create demo directories
    demo_dirs = [
        "tracking_data",
        "tracking_data/screenshots",
        "tracking_data/webcam",
        "demo_faces"
    ]
    
    for dir_path in demo_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Demo environment created")

def run_quick_test():
    """Run a quick test to verify installation"""
    print("🧪 Running quick test...")
    
    try:
        # Test basic imports
        from src.git_utils import GitUtils
        from src.predictor import ConflictPredictor
        
        print("✅ Basic modules imported successfully")
        
        # Test agentic modules if available
        try:
            from src.agentic_tracker import AgenticTracker
            from src.ai_validator import AIValidator
            from src.config_manager import ConfigManager
            print("✅ Agentic AI modules imported successfully")
        except ImportError as e:
            print(f"⚠️  Agentic AI modules not available: {e}")
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False
    
    return True

def show_next_steps():
    """Show next steps for users"""
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE!")
    print("=" * 60)
    
    print("\n📋 Next Steps:")
    print("1. Run the demo: python src/main.py demo")
    print("2. Start tracking: python src/main.py track")
    print("3. View help: python src/main.py --help")
    print("4. Configure settings: python src/main.py config")
    
    print("\n🔧 Available Commands:")
    print("  predict    - Predict merge conflicts")
    print("  detect     - Detect actual conflicts")
    print("  resolve    - Resolve conflicts with AI")
    print("  dashboard  - Show team dashboard")
    print("  track      - Start developer tracking")
    print("  validate   - Validate session data")
    print("  estimates  - Generate project estimates")
    print("  stats      - Show developer statistics")
    print("  config     - Manage configuration")
    print("  demo       - Run comprehensive demo")
    
    print("\n📚 Documentation:")
    print("  - README.md for detailed usage")
    print("  - src/agentic_demo.py for examples")
    print("  - src/config_manager.py for configuration")
    
    print("\n⚠️  Important Notes:")
    print("  - Ensure compliance with privacy laws")
    print("  - Obtain consent before tracking")
    print("  - Set OPENAI_API_KEY for AI validation")
    print("  - Configure privacy settings as needed")
    
    print("=" * 60)

def main():
    """Main setup function"""
    print("🚀 SmartGitMergePT with Agentic AI Tracking Setup")
    print("=" * 60)
    
    # Check prerequisites
    check_python_version()
    check_git()
    check_webcam()
    check_openai_key()
    
    # Install dependencies
    install_dependencies()
    
    # Setup configuration
    setup_configuration()
    
    # Create demo environment
    create_demo_environment()
    
    # Run quick test
    if run_quick_test():
        print("✅ Setup completed successfully!")
    else:
        print("❌ Setup completed with warnings")
    
    # Show next steps
    show_next_steps()

if __name__ == "__main__":
    main() 