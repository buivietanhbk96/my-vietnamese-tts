"""
VietTTS Desktop Application
Main entry point

Usage:
    python -m app.main
    
Or run directly:
    python app/main.py
"""

import sys
import os
import io

# Force UTF-8 encoding for the entire process on Windows
# This must be done before any other imports that might use stdout/stderr
if sys.platform == 'win32':
    # Set environment variable for subprocesses
    os.environ["PYTHONUTF8"] = "1"
    
    # Re-initialize stdout and stderr with UTF-8 encoding
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from loguru import logger

# Configure logging
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/viettts_{time:YYYY-MM-DD}.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG"
)


def main():
    """Main entry point"""
    logger.info("Starting VietTTS Desktop Application")
    
    try:
        from ui.main_window import run_app
        run_app()
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("Please ensure all dependencies are installed:")
        logger.info("  pip install -r requirements.txt")
        
        # Show simple error dialog if possible
        try:
            import tkinter as tk
            from tkinter import messagebox
            
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "Import Error",
                f"Failed to import required modules:\n\n{str(e)}\n\n"
                "Please ensure all dependencies are installed:\n"
                "pip install -r requirements.txt"
            )
            root.destroy()
        except:
            pass
        
        sys.exit(1)
        
    except Exception as e:
        logger.exception(f"Application error: {e}")
        
        try:
            import tkinter as tk
            from tkinter import messagebox
            
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "Error",
                f"An unexpected error occurred:\n\n{str(e)}"
            )
            root.destroy()
        except:
            pass
        
        sys.exit(1)
    
    logger.info("Application closed")


if __name__ == "__main__":
    main()
