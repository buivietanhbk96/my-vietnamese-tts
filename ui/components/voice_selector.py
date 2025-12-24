"""
Voice Selector Panel Component for VietTTS
"""

import os
import customtkinter as ctk
from pathlib import Path
from typing import Callable, Optional, Dict
from tkinter import filedialog

from app.config import Config, ThemeColors, Fonts
from ui.theme import (
    ThemedFrame, ThemedLabel, ThemedButton, ThemedOptionMenu
)


class VoiceSelectorPanel(ThemedFrame):
    """
    Voice selection panel with dropdown and voice cloning option
    """
    
    def __init__(
        self,
        master,
        voices: Dict[str, str] = None,
        on_voice_change: Optional[Callable] = None,
        on_clone_voice: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(master, style="card", **kwargs)
        
        self.voices = voices or {}
        self.on_voice_change = on_voice_change
        self.on_clone_voice = on_clone_voice
        
        self.selected_voice = None
        self.clone_file_path = None
        self.use_clone = False
        
        self._create_widgets()
        self._setup_layout()
    
    def _create_widgets(self):
        """Create panel widgets"""
        # Title
        self.title_label = ThemedLabel(
            self,
            text="🎤 Giọng đọc",
            style="subheading"
        )
        
        # Built-in voice selection
        self.builtin_frame = ThemedFrame(self, style="transparent")
        
        self.builtin_radio = ctk.CTkRadioButton(
            self.builtin_frame,
            text="Giọng có sẵn",
            variable=ctk.StringVar(value="builtin"),
            value="builtin",
            command=self._on_mode_change,
            font=Fonts.BODY,
            text_color=ThemeColors.TEXT_PRIMARY,
            fg_color=ThemeColors.PRIMARY,
            hover_color=ThemeColors.PRIMARY_HOVER
        )
        self._mode_var = self.builtin_radio.cget("variable")
        
        # Voice dropdown
        voice_names = list(self.voices.keys()) if self.voices else ["No voices available"]
        self.voice_dropdown = ThemedOptionMenu(
            self.builtin_frame,
            values=voice_names,
            command=self._on_voice_selected,
            width=200
        )
        
        if self.voices:
            self.selected_voice = list(self.voices.keys())[0]
        
        # Clone voice section
        self.clone_frame = ThemedFrame(self, style="transparent")
        
        self.clone_radio = ctk.CTkRadioButton(
            self.clone_frame,
            text="Clone giọng",
            variable=self._mode_var,
            value="clone",
            command=self._on_mode_change,
            font=Fonts.BODY,
            text_color=ThemeColors.TEXT_PRIMARY,
            fg_color=ThemeColors.PRIMARY,
            hover_color=ThemeColors.PRIMARY_HOVER
        )
        
        # Clone file picker
        self.clone_picker_frame = ThemedFrame(self.clone_frame, style="transparent")
        
        self.clone_btn = ThemedButton(
            self.clone_picker_frame,
            text="📁 Chọn file...",
            style="secondary",
            width=120,
            height=32,
            command=self._on_browse_clone
        )
        
        self.clone_file_label = ThemedLabel(
            self.clone_picker_frame,
            text="Chưa chọn file",
            style="muted"
        )
        
        # Clone info
        self.clone_info_label = ThemedLabel(
            self.clone_frame,
            text="Hỗ trợ: MP3, WAV (3-10 giây)",
            style="muted"
        )
        
        # Preview button
        self.preview_btn = ThemedButton(
            self,
            text="▶ Preview",
            style="ghost",
            width=100,
            height=32,
            command=self._on_preview
        )
    
    def _setup_layout(self):
        """Setup widget layout"""
        self.title_label.pack(anchor="w", padx=15, pady=(15, 10))
        
        # Built-in voice section
        self.builtin_frame.pack(fill="x", padx=15, pady=5)
        self.builtin_radio.pack(side="left")
        self.voice_dropdown.pack(side="left", padx=(20, 0))
        
        # Clone voice section
        self.clone_frame.pack(fill="x", padx=15, pady=5)
        self.clone_radio.pack(anchor="w")
        
        self.clone_picker_frame.pack(fill="x", pady=(5, 0), padx=(25, 0))
        self.clone_btn.pack(side="left")
        self.clone_file_label.pack(side="left", padx=10)
        
        self.clone_info_label.pack(anchor="w", padx=(25, 0), pady=(5, 0))
        
        # Preview button
        self.preview_btn.pack(anchor="e", padx=15, pady=15)
        
        # Initial state
        self._update_ui_state()
    
    def _on_mode_change(self):
        """Handle mode change between built-in and clone"""
        self.use_clone = self._mode_var.get() == "clone"
        self._update_ui_state()
        
        if self.on_voice_change:
            self.on_voice_change(self.get_selected_voice())
    
    def _update_ui_state(self):
        """Update UI based on current mode"""
        if self.use_clone:
            self.voice_dropdown.configure(state="disabled")
            self.clone_btn.configure(state="normal")
        else:
            self.voice_dropdown.configure(state="normal")
            self.clone_btn.configure(state="normal")
    
    def _on_voice_selected(self, voice_name: str):
        """Handle voice selection from dropdown"""
        self.selected_voice = voice_name
        
        if self.on_voice_change:
            self.on_voice_change(self.get_selected_voice())
    
    def _on_browse_clone(self):
        """Open file dialog for clone voice selection"""
        filetypes = [
            ("Audio files", "*.mp3 *.wav *.m4a"),
            ("MP3 files", "*.mp3"),
            ("WAV files", "*.wav"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(
            title="Chọn file giọng mẫu",
            filetypes=filetypes,
            initialdir=str(Path.home() / "Downloads")
        )
        
        if filepath:
            self._set_clone_file(filepath)
    
    def _set_clone_file(self, filepath: str):
        """Set clone voice file"""
        self.clone_file_path = filepath
        
        # Update label
        filename = Path(filepath).name
        if len(filename) > 30:
            filename = filename[:27] + "..."
        self.clone_file_label.configure(text=filename)
        
        # Switch to clone mode
        self._mode_var.set("clone")
        self.use_clone = True
        self._update_ui_state()
        
        if self.on_clone_voice:
            self.on_clone_voice(filepath)
        
        if self.on_voice_change:
            self.on_voice_change(self.get_selected_voice())
    
    def _on_preview(self):
        """Preview selected voice"""
        voice_path = self.get_selected_voice()
        if voice_path and os.path.exists(voice_path):
            # This will be connected to audio player
            pass
    
    def get_selected_voice(self) -> Optional[str]:
        """Get currently selected voice file path"""
        if self.use_clone and self.clone_file_path:
            return self.clone_file_path
        elif self.selected_voice and self.selected_voice in self.voices:
            return self.voices[self.selected_voice]
        return None
    
    def get_voice_name(self) -> str:
        """Get name of selected voice"""
        if self.use_clone and self.clone_file_path:
            return f"Clone: {Path(self.clone_file_path).stem}"
        return self.selected_voice or "Unknown"
    
    def set_voices(self, voices: Dict[str, str]):
        """Update available voices"""
        self.voices = voices
        voice_names = list(voices.keys()) if voices else ["No voices available"]
        self.voice_dropdown.configure(values=voice_names)
        
        if voices:
            self.selected_voice = voice_names[0]
            self.voice_dropdown.set(voice_names[0])
    
    def select_voice(self, voice_name: str):
        """Select a specific voice by name"""
        if voice_name in self.voices:
            self.selected_voice = voice_name
            self.voice_dropdown.set(voice_name)
            self._mode_var.set("builtin")
            self.use_clone = False
            self._update_ui_state()
    
    def set_preview_callback(self, callback: Callable):
        """Set callback for preview button"""
        self.preview_btn.configure(command=lambda: callback(self.get_selected_voice()))
