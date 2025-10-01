#!/usr/bin/env python3
"""
Genesis Robotics Course - Setup Script
Automated setup and verification for the course environment
"""

import os
import sys
import subprocess
import importlib.util
import platform

def check_python_version():
    """Check if Python version is compatible."""
    major, minor = sys.version_info[:2]
    print(f"Python version: {sys.version}")
    
    if major < 3 or (major == 3 and minor < 10):
        print("❌ Error: Python 3.10 or higher is required")
        print("   Please install a newer version of Python")
        return False
    
    print("✅ Python version is compatible")
    return True

def create_virtual_environment():
    """Create a virtual environment for the course."""
    venv_path = "genesis-env"
    
    if os.path.exists(venv_path):
        print("✅ Virtual environment already exists")
        return True
    
    try:
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
        print("✅ Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def get_activation_command():
    """Get the appropriate activation command based on OS."""
    if platform.system() == "Windows":
        return "genesis-env\\Scripts\\activate"
    else:
        return "source genesis-env/bin/activate"

def install_dependencies():
    """Install required Python packages."""
    try:
        print("Installing dependencies...")
        
        # Determine pip command
        if platform.system() == "Windows":
            pip_cmd = "genesis-env\\Scripts\\pip"
        else:
            pip_cmd = "genesis-env/bin/pip"
        
        # Upgrade pip first
        subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True)
        
        # Install PyTorch CPU version first
        print("Installing PyTorch (CPU version)...")
        subprocess.run([
            pip_cmd, "install", "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ], check=True)
        
        # Install other requirements
        if os.path.exists("requirements.txt"):
            subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)
        else:
            # Fallback to individual packages
            packages = ["genesis-world", "numpy", "scipy", "matplotlib"]
            for package in packages:
                subprocess.run([pip_cmd, "install", package], check=True)
        
        print("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def verify_installation():
    """Verify that Genesis is properly installed."""
    try:
        print("Verifying Genesis installation...")
        
        # Run the installation test script
        if os.path.exists("test_installation.py"):
            if platform.system() == "Windows":
                python_cmd = "genesis-env\\Scripts\\python"
            else:
                python_cmd = "genesis-env/bin/python"
            
            result = subprocess.run([python_cmd, "test_installation.py"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Genesis installation verified")
                return True
            else:
                print("❌ Genesis installation verification failed")
                print(result.stdout)
                print(result.stderr)
                return False
        else:
            print("⚠️  Installation test script not found, skipping verification")
            return True
            
    except Exception as e:
        print(f"❌ Installation verification failed: {e}")
        return False

def create_course_structure():
    """Ensure all course directories exist."""
    directories = [
        "docs",
        "lessons/lesson1",
        "lessons/lesson2", 
        "lessons/lesson3",
        "lessons/lesson4",
        "lessons/lesson5",
        "lessons/lesson6",
        "lessons/lesson7",
        "lessons/lesson8"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Course directory structure created")

def print_next_steps():
    """Print instructions for getting started."""
    activation_cmd = get_activation_command()
    
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\nTo start the course:")
    print(f"1. Activate the virtual environment:")
    print(f"   {activation_cmd}")
    print("\n2. Navigate to lesson 1:")
    print("   cd lessons/lesson1")
    print("\n3. Read the README and start with the starter code:")
    print("   python starter_code.py")
    print("\n4. Run tests to verify your solutions:")
    print("   python test_lesson1.py")
    
    print("\nDocumentation:")
    print("- Installation guide: docs/installation.md")
    print("- Genesis overview: docs/genesis-overview.md")
    print("- API reference: docs/api-reference.md")
    
    print("\nTroubleshooting:")
    print("- If you encounter issues, check docs/installation.md")
    print("- Make sure the virtual environment is activated")
    print("- Contact your instructor for help")
    
    print("=" * 60)

def main():
    """Main setup function."""
    print("=" * 60)
    print("Genesis Robotics Course - Automated Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("⚠️  Installation verification failed, but you can still try to proceed")
    
    # Create course structure
    create_course_structure()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()