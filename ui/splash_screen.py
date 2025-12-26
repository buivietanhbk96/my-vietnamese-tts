"""
Splash Screen for VietTTS Desktop Application
Shows loading progress during startup
"""

import customtkinter as ctk
from typing import Callable, Optional

from app.config import ThemeColors, Fonts, Config
from ui.theme import ThemedLabel, ThemedProgressBar


class SplashScreen(ctk.CTkToplevel):
    """
    Splash screen shown during application startup
    """
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        # Window setup
        self.title("")
        self.geometry("400x250")
        self.resizable(False, False)
        self.overrideredirect(True)  # Remove window decorations
        
        # Center on screen
        self.update_idletasks()
        x = (self.winfo_screenwidth() - 400) // 2
        y = (self.winfo_screenheight() - 250) // 2
        self.geometry(f"+{x}+{y}")
        
        # Configure appearance
        self.configure(fg_color=ThemeColors.BG_PRIMARY)
        
        self._create_widgets()
        
        # Keep on top
        self.lift()
        self.grab_set()
    
    def _create_widgets(self):
        """Create splash screen widgets"""
        # Logo/Title
        self.title_label = ctk.CTkLabel(
            self,
            text="🎙️ VietTTS",
            font=("Segoe UI", 32, "bold"),
            text_color=ThemeColors.TEXT_PRIMARY
        )
        self.title_label.pack(pady=(40, 5))
        
        # Subtitle
        self.subtitle_label = ctk.CTkLabel(
            self,
            text="Vietnamese Text-to-Speech",
            font=Fonts.BODY,
            text_color=ThemeColors.TEXT_SECONDARY
        )
        self.subtitle_label.pack(pady=(0, 30))
        
        # Progress bar
        self.progress_bar = ThemedProgressBar(
            self,
            width=300,
            height=6
        )
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)
        
        # Status label
        self.status_label = ctk.CTkLabel(
            self,
            text="Initializing...",
            font=Fonts.BODY_SMALL,
            text_color=ThemeColors.TEXT_MUTED
        )
        self.status_label.pack(pady=10)
        
        # Version
        self.version_label = ctk.CTkLabel(
            self,
            text="v1.0.0",
            font=Fonts.BODY_SMALL,
            text_color=ThemeColors.TEXT_MUTED
        )
        self.version_label.pack(side="bottom", pady=15)
    
    def update_progress(self, progress: float, status: str = ""):
        """
        Update progress bar and status
        
        Args:
            progress: Progress value (0.0 to 1.0)
            status: Status message
        """
        self.progress_bar.set(progress)
        if status:
            self.status_label.configure(text=status)
        self.update()
    
    def set_status(self, status: str):
        """Update status message"""
        self.status_label.configure(text=status)
        self.update()
    
    def close(self):
        """Close splash screen"""
        self.grab_release()
        self.destroy()
