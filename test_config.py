#!/usr/bin/env python3
"""
Test script to check configuration loading
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

print("Testing configuration loading...")
print(f"Current working directory: {Path.cwd()}")
print(f"Looking for config.yaml in: {Path.cwd() / 'config.yaml'}")

try:
    from config.settings import get_config, Config
    
    print("Attempting to create config...")
    config = Config()
    print("✅ Configuration loaded successfully!")
    print(f"Project root: {config.PROJECT_ROOT}")
    print(f"Config file: {config.CONFIG_FILE}")
    
except Exception as e:
    print(f"❌ Configuration failed: {e}")
    import traceback
    traceback.print_exc() 