"""
Settings Panel Component for VietTTS
"""

import customtkinter as ctk
from pathlib import Path
from typing import Callable, Optional
from tkinter import filedialog

from app.config import Config, ThemeColors, Fonts
from ui.theme import (
    ThemedFrame, ThemedLabel, ThemedButton, ThemedSlider, ThemedOptionMenu
)


class SettingsPanel(ThemedFrame):
    """
    Settings panel with speed control and output options
    """
    
    def __init__(
        self,
        master,
        on_settings_change: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(master, style="card", **kwargs)
        
        self.on_settings_change = on_settings_change
        
        self._speed = Config.DEFAULT_SPEED
        self._output_format = Config.DEFAULT_FORMAT
        self._output_dir = str(Config.OUTPUT_DIR)
        
        self._create_widgets()
        self._setup_layout()
    
    def _create_widgets(self):
        """Create panel widgets"""
        # Title
        self.title_label = ThemedLabel(
            self,
            text="⚙️ Cài đặt",
            style="subheading"
        )
        
        # Speed control
        self.speed_frame = ThemedFrame(self, style="transparent")
        
        self.speed_label = ThemedLabel(
            self.speed_frame,
            text="Tốc độ đọc:",
            style="default"
        )
        
        self.speed_slider = ThemedSlider(
            self.speed_frame,
            from_=Config.MIN_SPEED,
            to=Config.MAX_SPEED,
            number_of_steps=30,
            command=self._on_speed_change,
            width=150
        )
        self.speed_slider.set(self._speed)
        
        self.speed_value_label = ThemedLabel(
            self.speed_frame,
            text=f"{self._speed:.1f}x",
            style="default"
        )
        
        # Output format
        self.format_frame = ThemedFrame(self, style="transparent")
        
        self.format_label = ThemedLabel(
            self.format_frame,
            text="Định dạng:",
            style="default"
        )
        
        self.format_dropdown = ThemedOptionMenu(
            self.format_frame,
            values=["WAV", "MP3"],
            command=self._on_format_change,
            width=100
        )
        self.format_dropdown.set(self._output_format.upper())
        
        # Output directory
        self.output_frame = ThemedFrame(self, style="transparent")
        
        self.output_label = ThemedLabel(
            self.output_frame,
            text="Thư mục lưu:",
            style="default"
        )
        
        self.output_btn = ThemedButton(
            self.output_frame,
            text="📁 output",
            style="secondary",
            width=120,
            height=28,
            command=self._on_browse_output
        )
        
        self.output_path_label = ThemedLabel(
            self.output_frame,
            text=self._get_short_path(self._output_dir),
            style="muted"
        )
    
    def _setup_layout(self):
        """Setup widget layout"""
        self.title_label.pack(anchor="w", padx=15, pady=(15, 15))
        
        # Speed control
        self.speed_frame.pack(fill="x", padx=15, pady=5)
        self.speed_label.pack(side="left")
        self.speed_slider.pack(side="left", padx=10)
        self.speed_value_label.pack(side="left")
        
        # Output format
        self.format_frame.pack(fill="x", padx=15, pady=5)
        self.format_label.pack(side="left")
        self.format_dropdown.pack(side="left", padx=10)
        
        # Output directory
        self.output_frame.pack(fill="x", padx=15, pady=(5, 15))
        self.output_label.pack(side="left")
        self.output_btn.pack(side="left", padx=10)
        self.output_path_label.pack(side="left", padx=5)
    
    def _on_speed_change(self, value: float):
        """Handle speed slider change"""
        self._speed = round(value, 1)
        self.speed_value_label.configure(text=f"{self._speed:.1f}x")
        self._notify_change()
    
    def _on_format_change(self, value: str):
        """Handle format dropdown change"""
        self._output_format = value.lower()
        self._notify_change()
    
    def _on_browse_output(self):
        """Open folder dialog for output directory"""
        directory = filedialog.askdirectory(
            title="Chọn thư mục lưu file",
            initialdir=self._output_dir
        )
        
        if directory:
            self._output_dir = directory
            short_path = self._get_short_path(directory)
            self.output_path_label.configure(text=short_path)
            
            # Update button text
            folder_name = Path(directory).name
            if len(folder_name) > 15:
                folder_name = folder_name[:12] + "..."
            self.output_btn.configure(text=f"📁 {folder_name}")
            
            self._notify_change()
    
    def _get_short_path(self, path: str, max_len: int = 40) -> str:
        """Get shortened path for display"""
        if len(path) <= max_len:
            return path
        
        parts = Path(path).parts
        if len(parts) <= 2:
            return path
        
        # Show first and last parts with ...
        return f"{parts[0]}\\...\\{parts[-1]}"
    
    def _notify_change(self):
        """Notify settings change"""
        if self.on_settings_change:
            self.on_settings_change(self.get_settings())
    
    def get_settings(self) -> dict:
        """Get current settings"""
        return {
            "speed": self._speed,
            "output_format": self._output_format,
            "output_directory": self._output_dir
        }
    
    def get_speed(self) -> float:
        """Get current speed setting"""
        return self._speed
    
    def get_output_format(self) -> str:
        """Get current output format"""
        return self._output_format
    
    def get_output_directory(self) -> str:
        """Get current output directory"""
        return self._output_dir
    
    def set_speed(self, speed: float):
        """Set speed value"""
        self._speed = max(Config.MIN_SPEED, min(Config.MAX_SPEED, speed))
        self.speed_slider.set(self._speed)
        self.speed_value_label.configure(text=f"{self._speed:.1f}x")
    
    def set_output_format(self, fmt: str):
        """Set output format"""
        fmt = fmt.lower()
        if fmt in Config.SUPPORTED_FORMATS:
            self._output_format = fmt
            self.format_dropdown.set(fmt.upper())
    
    def set_output_directory(self, directory: str):
        """Set output directory"""
        if Path(directory).exists():
            self._output_dir = directory
            self.output_path_label.configure(text=self._get_short_path(directory))
