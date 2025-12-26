"""
Audio Player Panel Component for VietTTS
"""

import os
import customtkinter as ctk
from pathlib import Path
from typing import Callable, Optional
from tkinter import filedialog
import shutil

from app.config import ThemeColors, Fonts
from app.audio_utils import AudioPlayer
from ui.theme import (
    ThemedFrame, ThemedLabel, ThemedButton, ThemedSlider, ThemedProgressBar
)


class AudioPlayerPanel(ThemedFrame):
    """
    Audio player panel with playback controls
    """
    
    def __init__(
        self,
        master,
        on_save: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(master, style="card", **kwargs)
        
        self.on_save = on_save
        self.audio_player = AudioPlayer()
        self.current_file = None
        self.duration = 0
        
        self._create_widgets()
        self._setup_layout()
        self._setup_callbacks()
    
    def _create_widgets(self):
        """Create panel widgets"""
        # Title
        self.title_label = ThemedLabel(
            self,
            text="🔊 Audio Player",
            style="subheading"
        )
        
        # No audio placeholder
        self.placeholder_label = ThemedLabel(
            self,
            text="Chưa có audio. Hãy generate để nghe kết quả.",
            style="muted"
        )
        
        # Player controls frame
        self.controls_frame = ThemedFrame(self, style="transparent")
        
        # Play/Pause button
        self.play_btn = ThemedButton(
            self.controls_frame,
            text="▶",
            style="primary",
            width=50,
            height=50,
            font=("Segoe UI", 20),
            command=self._on_play_pause
        )
        
        # Stop button
        self.stop_btn = ThemedButton(
            self.controls_frame,
            text="⏹",
            style="secondary",
            width=40,
            height=40,
            font=("Segoe UI", 16),
            command=self._on_stop
        )
        
        # Progress frame
        self.progress_frame = ThemedFrame(self.controls_frame, style="transparent")
        
        # Progress slider
        self.progress_slider = ThemedSlider(
            self.progress_frame,
            from_=0,
            to=100,
            command=self._on_seek,
            width=300
        )
        self.progress_slider.set(0)
        
        # Time labels
        self.time_frame = ThemedFrame(self.progress_frame, style="transparent")
        
        self.current_time_label = ThemedLabel(
            self.time_frame,
            text="00:00",
            style="muted"
        )
        
        self.duration_label = ThemedLabel(
            self.time_frame,
            text="00:00",
            style="muted"
        )
        
        # Volume control
        self.volume_frame = ThemedFrame(self.controls_frame, style="transparent")
        
        self.volume_icon = ThemedLabel(
            self.volume_frame,
            text="🔊",
            style="default"
        )
        
        self.volume_slider = ThemedSlider(
            self.volume_frame,
            from_=0,
            to=1,
            command=self._on_volume_change,
            width=80
        )
        self.volume_slider.set(0.8)
        
        # Action buttons
        self.actions_frame = ThemedFrame(self, style="transparent")
        
        self.save_btn = ThemedButton(
            self.actions_frame,
            text="💾 Lưu file",
            style="secondary",
            width=100,
            height=36,
            command=self._on_save
        )
        
        self.copy_btn = ThemedButton(
            self.actions_frame,
            text="📋 Copy path",
            style="ghost",
            width=100,
            height=36,
            command=self._on_copy_path
        )
        
        self.open_folder_btn = ThemedButton(
            self.actions_frame,
            text="📂 Mở folder",
            style="ghost",
            width=100,
            height=36,
            command=self._on_open_folder
        )
        
        # File info
        self.file_info_label = ThemedLabel(
            self,
            text="",
            style="muted"
        )
    
    def _setup_layout(self):
        """Setup widget layout"""
        self.title_label.pack(anchor="w", padx=15, pady=(15, 10))
        
        # Placeholder (shown when no audio)
        self.placeholder_label.pack(pady=20)
        
        # Controls (hidden initially)
        self.controls_frame.pack(fill="x", padx=15, pady=10)
        self.controls_frame.pack_forget()  # Hide initially
        
        # Playback buttons
        self.play_btn.pack(side="left", padx=(0, 10))
        self.stop_btn.pack(side="left", padx=(0, 20))
        
        # Progress
        self.progress_frame.pack(side="left", fill="x", expand=True, padx=10)
        self.progress_slider.pack(fill="x")
        
        self.time_frame.pack(fill="x", pady=(5, 0))
        self.current_time_label.pack(side="left")
        self.duration_label.pack(side="right")
        
        # Volume
        self.volume_frame.pack(side="right", padx=(20, 0))
        self.volume_icon.pack(side="left")
        self.volume_slider.pack(side="left", padx=(5, 0))
        
        # Actions
        self.actions_frame.pack(fill="x", padx=15, pady=10)
        self.actions_frame.pack_forget()  # Hide initially
        
        self.save_btn.pack(side="left", padx=(0, 10))
        self.copy_btn.pack(side="left", padx=(0, 10))
        self.open_folder_btn.pack(side="left")
        
        # File info
        self.file_info_label.pack(anchor="w", padx=15, pady=(0, 15))
    
    def _setup_callbacks(self):
        """Setup audio player callbacks"""
        self.audio_player.set_position_callback(self._update_position)
        self.audio_player.set_completion_callback(self._on_playback_complete)
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds to MM:SS"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def _update_position(self, position: float):
        """Update playback position display"""
        self.current_time_label.configure(text=self._format_time(position))
        
        if self.duration > 0:
            progress = (position / self.duration) * 100
            self.progress_slider.set(progress)
    
    def _on_playback_complete(self):
        """Handle playback completion"""
        self.play_btn.configure(text="▶")
        self.progress_slider.set(0)
        self.current_time_label.configure(text="00:00")
    
    def _on_play_pause(self):
        """Handle play/pause button click"""
        if not self.current_file:
            return
        
        if self.audio_player.is_playing and not self.audio_player.is_paused:
            self.audio_player.pause()
            self.play_btn.configure(text="▶")
        else:
            self.audio_player.play()
            self.play_btn.configure(text="⏸")
    
    def _on_stop(self):
        """Handle stop button click"""
        self.audio_player.stop()
        self.play_btn.configure(text="▶")
        self.progress_slider.set(0)
        self.current_time_label.configure(text="00:00")
    
    def _on_seek(self, value: float):
        """Handle progress slider seek"""
        if self.duration > 0:
            position = (value / 100) * self.duration
            self.audio_player.seek(position)
            self.current_time_label.configure(text=self._format_time(position))
    
    def _on_volume_change(self, value: float):
        """Handle volume slider change"""
        self.audio_player.set_volume(value)
        
        # Update icon based on volume
        if value == 0:
            self.volume_icon.configure(text="🔇")
        elif value < 0.5:
            self.volume_icon.configure(text="🔉")
        else:
            self.volume_icon.configure(text="🔊")
    
    def _on_save(self):
        """Handle save button click"""
        if not self.current_file:
            return
        
        # Get save location
        initial_name = Path(self.current_file).name
        filetypes = [
            ("WAV files", "*.wav"),
            ("MP3 files", "*.mp3"),
            ("All files", "*.*")
        ]
        
        save_path = filedialog.asksaveasfilename(
            title="Lưu audio file",
            initialfile=initial_name,
            filetypes=filetypes,
            defaultextension=".wav"
        )
        
        if save_path:
            try:
                shutil.copy2(self.current_file, save_path)
                self.file_info_label.configure(text=f"Đã lưu: {save_path}")
                
                if self.on_save:
                    self.on_save(save_path)
            except Exception as e:
                self.file_info_label.configure(text=f"Lỗi: {str(e)}")
    
    def _on_copy_path(self):
        """Copy file path to clipboard"""
        if self.current_file:
            self.clipboard_clear()
            self.clipboard_append(self.current_file)
            self.file_info_label.configure(text="Đã copy path!")
    
    def _on_open_folder(self):
        """Open containing folder"""
        if self.current_file:
            folder = Path(self.current_file).parent
            os.startfile(str(folder))
    
    def load_audio(self, filepath: str, auto_play: bool = True):
        """
        Load audio file for playback
        
        Args:
            filepath: Path to audio file
            auto_play: Whether to start playing automatically
        """
        if not os.path.exists(filepath):
            return
        
        self.current_file = filepath
        
        # Load into player
        if self.audio_player.load(filepath):
            self.duration = self.audio_player.get_duration()
            
            # Update UI
            self._show_player()
            self.duration_label.configure(text=self._format_time(self.duration))
            self.current_time_label.configure(text="00:00")
            self.progress_slider.set(0)
            
            # File info
            filename = Path(filepath).name
            self.file_info_label.configure(text=f"📄 {filename}")
            
            # Auto play
            if auto_play:
                self.audio_player.play()
                self.play_btn.configure(text="⏸")
    
    def _show_player(self):
        """Show player controls"""
        self.placeholder_label.pack_forget()
        self.controls_frame.pack(fill="x", padx=15, pady=10)
        self.actions_frame.pack(fill="x", padx=15, pady=10)
    
    def _hide_player(self):
        """Hide player controls and show placeholder"""
        self.controls_frame.pack_forget()
        self.actions_frame.pack_forget()
        self.placeholder_label.pack(pady=20)
    
    def reset(self):
        """Reset player to initial state"""
        self.audio_player.stop()
        self.current_file = None
        self.duration = 0
        self._hide_player()
        self.file_info_label.configure(text="")
    
    def cleanup(self):
        """Cleanup resources"""
        self.audio_player.cleanup()
