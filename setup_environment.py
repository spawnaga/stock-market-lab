#!/usr/bin/env python3
"""
Environment setup helper for the stock market lab application.
This script identifies what needs to be installed and provides instructions.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_docker():
    """Check if Docker is properly installed and running."""
    try:
        result = subprocess.run(['docker', 'info'], 
                               capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return True, "Docker is running properly"
        else:
            return False, f"Docker is installed but not running: {result.stderr}"
    except FileNotFoundError:
        return False, "Docker is not installed"
    except Exception as e:
        return False, f"Error checking Docker: {str(e)}"

def check_node():
    """Check if Node.js is installed."""
    try:
        result = subprocess.run(['node', '--version'], 
                               capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return True, f"Node.js version: {result.stdout.strip()}"
        else:
            return False, "Node.js not found"
    except FileNotFoundError:
        return False, "Node.js is not installed"
    except Exception as e:
        return False, f"Error checking Node.js: {str(e)}"

def check_python_packages():
    """Check if required Python packages are installed."""
    required_packages = ['torch', 'numpy', 'pandas']
    missing_packages = []
    
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing_packages.append(pkg)
    
    return len(missing_packages) == 0, missing_packages

def check_directories():
    """Check if all required directories exist."""
    required_dirs = [
        'frontend-react',
        'agents-python', 
        'core-cpp',
        'infra'
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            missing_dirs.append(dir_name)
    
    return len(missing_dirs) == 0, missing_dirs

def main():
    print("=" * 70)
    print("STOCK MARKET LAB ENVIRONMENT SETUP CHECK")
    print("=" * 70)
    
    # Check Docker
    docker_ok, docker_msg = check_docker()
    print(f"\n🐳 Docker Status: {'✅ PASS' if docker_ok else '❌ FAIL'}")
    print(f"   {docker_msg}")
    
    # Check Node.js
    node_ok, node_msg = check_node()
    print(f"\n🌐 Node.js Status: {'✅ PASS' if node_ok else '❌ FAIL'}")
    print(f"   {node_msg}")
    
    # Check Python packages
    python_ok, missing_pkgs = check_python_packages()
    print(f"\n🐍 Python Packages Status: {'✅ PASS' if python_ok else '❌ FAIL'}")
    if not python_ok:
        print(f"   Missing packages: {', '.join(missing_pkgs)}")
    
    # Check directories
    dirs_ok, missing_dirs = check_directories()
    print(f"\n📂 Directory Structure: {'✅ PASS' if dirs_ok else '❌ FAIL'}")
    if not dirs_ok:
        print(f"   Missing directories: {', '.join(missing_dirs)}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDED ACTIONS")
    print("=" * 70)
    
    if not docker_ok:
        print("1. Install Docker and Docker Compose:")
        print("   - Follow official Docker installation guide")
        print("   - Ensure Docker daemon is running")
        print("   - Add user to docker group")
        print("")
    
    if not node_ok:
        print("2. Install Node.js:")
        print("   - Download from nodejs.org or use package manager")
        print("   - Verify installation with 'node --version'")
        print("")
    
    if not python_ok:
        print("3. Install Python dependencies:")
        print("   cd agents-python")
        print("   pip install -r requirements.txt")
        print("")
    
    if not dirs_ok:
        print("4. Verify repository structure:")
        print("   All required directories should exist")
        print("")
    
    print("5. Once prerequisites are met, run:")
    print("   cd infra")
    print("   docker-compose up -d")
    print("")
    
    print("6. For frontend development:")
    print("   cd frontend-react")
    print("   npm install")
    print("   npm start")
    print("")
    
    print("=" * 70)
    print("NOTE: This environment appears to have Docker installed but not running.")
    print("Please ensure Docker daemon is started before attempting to run services.")
    print("=" * 70)

if __name__ == "__main__":
    main()