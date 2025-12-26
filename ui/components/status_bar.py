"""
Status Bar Component for VietTTS
"""

import customtkinter as ctk
from typing import Optional

from app.config import ThemeColors, Fonts
from ui.theme import ThemedFrame, ThemedLabel


class StatusBar(ThemedFrame):
    """
    Status bar showing application status and info
    """
    
    STATUS_ICONS = {
        "ready": "✓",
        "loading": "⏳",
        "processing": "🔄",
        "success": "✓",
        "error": "❌",
        "warning": "⚠️"
    }
    
    STATUS_COLORS = {
        "ready": ThemeColors.SUCCESS,
        "loading": ThemeColors.WARNING,
        "processing": ThemeColors.INFO,
        "success": ThemeColors.SUCCESS,
        "error": ThemeColors.ERROR,
        "warning": ThemeColors.WARNING
    }
    
    def __init__(self, master, **kwargs):
        # Override style for status bar
        kwargs["fg_color"] = ThemeColors.BG_TERTIARY
        kwargs["corner_radius"] = 0
        kwargs["height"] = 30
        
        super().__init__(master, **kwargs)
        
        self._create_widgets()
        self._setup_layout()
    
    def _create_widgets(self):
        """Create status bar widgets"""
        # Status icon and text
        self.status_frame = ThemedFrame(self, style="transparent")
        
        self.status_icon = ThemedLabel(
            self.status_frame,
            text="✓",
            style="success"
        )
        
        self.status_label = ThemedLabel(
            self.status_frame,
            text="Ready",
            style="default"
        )
        
        # Separator
        self.separator1 = ThemedLabel(
            self,
            text="|",
            style="muted"
        )
        
        # Model info
        self.model_label = ThemedLabel(
            self,
            text="Model: viet-tts",
            style="muted"
        )
        
        # Separator
        self.separator2 = ThemedLabel(
            self,
            text="|",
            style="muted"
        )
        
        # Device info
        self.device_label = ThemedLabel(
            self,
            text="Device: CPU",
            style="muted"
        )
        
        # Right side - version
        self.version_label = ThemedLabel(
            self,
            text="v1.0.0",
            style="muted"
        )
    
    def _setup_layout(self):
        """Setup widget layout"""
        # Pack from left
        self.status_frame.pack(side="left", padx=(10, 5))
        self.status_icon.pack(side="left")
        self.status_label.pack(side="left", padx=(5, 0))
        
        self.separator1.pack(side="left", padx=10)
        self.model_label.pack(side="left")
        
        self.separator2.pack(side="left", padx=10)
        self.device_label.pack(side="left")
        
        # Pack from right
        self.version_label.pack(side="right", padx=10)
    
    def set_status(self, status: str, message: str = ""):
        """
        Set status bar status
        
        Args:
            status: Status type (ready, loading, processing, success, error, warning)
            message: Status message
        """
        icon = self.STATUS_ICONS.get(status, "")
        color = self.STATUS_COLORS.get(status, ThemeColors.TEXT_PRIMARY)
        
        self.status_icon.configure(text=icon, text_color=color)
        self.status_label.configure(text=message or status.capitalize())
    
    def set_ready(self, message: str = "Ready"):
        """Set ready status"""
        self.set_status("ready", message)
    
    def set_loading(self, message: str = "Loading..."):
        """Set loading status"""
        self.set_status("loading", message)
    
    def set_processing(self, message: str = "Processing..."):
        """Set processing status"""
        self.set_status("processing", message)
    
    def set_success(self, message: str = "Success"):
        """Set success status"""
        self.set_status("success", message)
    
    def set_error(self, message: str = "Error"):
        """Set error status"""
        self.set_status("error", message)
    
    def set_model_info(self, model_name: str):
        """Set model info display"""
        self.model_label.configure(text=f"Model: {model_name}")
    
    def set_device_info(self, device: str):
        """Set device info display"""
        self.device_label.configure(text=f"Device: {device}")
    
    def set_version(self, version: str):
        """Set version display"""
        self.version_label.configure(text=f"v{version}")
