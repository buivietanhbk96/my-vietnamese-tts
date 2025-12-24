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
