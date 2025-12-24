"""
FFmpeg check and installation helper for Windows
"""

import os
import subprocess
import shutil
from typing import Tuple
from loguru import logger


def check_ffmpeg() -> Tuple[bool, str]:
    """
    Check if FFmpeg is installed and available
    
    Returns:
        Tuple of (is_available, version_or_error_message)
    """
    try:
        # Try to run ffmpeg
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            # Extract version from first line
            version_line = result.stdout.split('\n')[0]
            logger.info(f"FFmpeg found: {version_line}")
            return True, version_line
        else:
            return False, "FFmpeg found but returned error"
            
    except FileNotFoundError:
        return False, "FFmpeg not found in PATH"
    except subprocess.TimeoutExpired:
        return False, "FFmpeg check timed out"
    except Exception as e:
        return False, f"Error checking FFmpeg: {str(e)}"


def get_ffmpeg_install_instructions() -> str:
    """
    Get FFmpeg installation instructions for Windows
    
    Returns:
        str: Installation instructions
    """
    return """
FFmpeg Installation Instructions (Windows)
==========================================

Option 1: Using winget (Recommended)
------------------------------------
1. Open PowerShell as Administrator
2. Run: winget install FFmpeg

Option 2: Manual Installation
-----------------------------
1. Download FFmpeg from:
   https://github.com/BtbN/FFmpeg-Builds/releases
   (Choose: ffmpeg-master-latest-win64-gpl.zip)

2. Extract to C:\\ffmpeg

3. Add to PATH:
   - Open System Properties (Win + Pause)
   - Click "Advanced system settings"
   - Click "Environment Variables"
   - Under "System variables", find "Path"
   - Click "Edit" → "New"
   - Add: C:\\ffmpeg\\bin
   - Click OK on all dialogs

4. Restart this application

Option 3: Using Chocolatey
--------------------------
1. Install Chocolatey: https://chocolatey.org/install
2. Run: choco install ffmpeg

Verification
------------
Open a new Command Prompt and run:
> ffmpeg -version

You should see FFmpeg version information.
"""


def find_ffmpeg_path() -> str:
    """
    Try to find FFmpeg executable path
    
    Returns:
        str: Path to ffmpeg executable or empty string
    """
    # Check common locations on Windows
    common_paths = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        os.path.expanduser(r"~\ffmpeg\bin\ffmpeg.exe"),
    ]
    
    # Check if in PATH
    ffmpeg_in_path = shutil.which("ffmpeg")
    if ffmpeg_in_path:
        return ffmpeg_in_path
    
    # Check common locations
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return ""


def add_ffmpeg_to_path(ffmpeg_dir: str) -> bool:
    """
    Add FFmpeg directory to PATH for current session
    
    Args:
        ffmpeg_dir: Directory containing ffmpeg.exe
        
    Returns:
        bool: True if successful
    """
    try:
        current_path = os.environ.get("PATH", "")
        
        if ffmpeg_dir not in current_path:
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + current_path
            logger.info(f"Added to PATH: {ffmpeg_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to add to PATH: {e}")
        return False
