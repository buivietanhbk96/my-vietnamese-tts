# -*- coding: utf-8 -*-
"""
Audio Panel Pro - Panel hiển thị waveform và audio controls
VIP Pro MAX feature - Professional audio interface
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
from typing import Optional, Callable
import numpy as np
import os

# Import internal modules
try:
    from app.waveform_viewer import WaveformViewer, WaveformToolbar, WaveformStyle
    from app.audio_processor import AudioPostProcessor, ProcessingPreset
    from app.preset_manager import get_preset_manager, AudioPreset
except ImportError:
    # Fallback for development
    from waveform_viewer import WaveformViewer, WaveformToolbar, WaveformStyle
    from audio_processor import AudioPostProcessor, ProcessingPreset
    from preset_manager import get_preset_manager, AudioPreset


class AudioPanelPro(ctk.CTkFrame):
    """
    Professional Audio Panel với:
    - Waveform visualization
    - Audio processing controls
    - Preset selection
    - Playback controls
    """
    
    def __init__(
        self,
        parent,
        on_audio_processed: Optional[Callable[[np.ndarray, int], None]] = None,
        **kwargs
    ):
        super().__init__(parent, **kwargs)
        
        self.on_audio_processed = on_audio_processed
        
        # Audio state
        self._audio_data: Optional[np.ndarray] = None
        self._sample_rate: int = 22050
        self._processed_audio: Optional[np.ndarray] = None
        
        # Initialize components
        self._init_audio_processor()
        self._init_preset_manager()
        self._create_widgets()
    
    def _init_audio_processor(self):
        """Initialize audio processor"""
        self.audio_processor = AudioPostProcessor(sample_rate=22050)
    
    def _init_preset_manager(self):
        """Initialize preset manager"""
        self.preset_manager = get_preset_manager()
    
    def _create_widgets(self):
        """Create UI widgets"""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # === Header with preset selector ===
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        header_frame.grid_columnconfigure(1, weight=1)
        
        # Title
        ctk.CTkLabel(
            header_frame,
            text="🎛️ Audio Processing",
            font=("Segoe UI", 14, "bold")
        ).grid(row=0, column=0, sticky="w")
        
        # Preset selector
        preset_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        preset_frame.grid(row=0, column=2, sticky="e")
        
        ctk.CTkLabel(
            preset_frame,
            text="Preset:",
            font=("Segoe UI", 11)
        ).pack(side="left", padx=(0, 5))
        
        self.preset_var = ctk.StringVar(value="Clean Voice")
        self.preset_dropdown = ctk.CTkComboBox(
            preset_frame,
            values=self._get_preset_names(),
            variable=self.preset_var,
            width=150,
            command=self._on_preset_change
        )
        self.preset_dropdown.pack(side="left")
        
        # === Waveform area (using tk.Canvas wrapper) ===
        waveform_frame = ctk.CTkFrame(self)
        waveform_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        waveform_frame.grid_columnconfigure(0, weight=1)
        waveform_frame.grid_rowconfigure(0, weight=1)
        
        # Waveform viewer
        self.waveform_viewer = WaveformViewer(
            waveform_frame,
            width=700,
            height=120,
            style=WaveformStyle(
                background="#1a1a2e",
                waveform_color="#4cc9f0",
                selection_color="#f72585",
                playhead_color="#ffd60a",
                rms_color="#7209b7"
            )
        )
        self.waveform_viewer.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Waveform toolbar
        self.waveform_toolbar = WaveformToolbar(
            waveform_frame,
            self.waveform_viewer,
            bg="#16213e"
        )
        self.waveform_toolbar.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 5))
        
        # === Processing controls ===
        controls_frame = ctk.CTkFrame(self)
        controls_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        
        # Normalize section
        norm_frame = ctk.CTkFrame(controls_frame, fg_color="transparent")
        norm_frame.pack(side="left", padx=10, pady=5)
        
        self.normalize_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            norm_frame,
            text="Normalize",
            variable=self.normalize_var,
            width=100
        ).pack(side="left")
        
        self.norm_level_var = ctk.StringVar(value="-3")
        ctk.CTkEntry(
            norm_frame,
            textvariable=self.norm_level_var,
            width=50
        ).pack(side="left", padx=5)
        ctk.CTkLabel(norm_frame, text="dB").pack(side="left")
        
        # Noise reduction section
        noise_frame = ctk.CTkFrame(controls_frame, fg_color="transparent")
        noise_frame.pack(side="left", padx=10, pady=5)
        
        self.noise_reduction_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            noise_frame,
            text="Noise Reduction",
            variable=self.noise_reduction_var,
            width=120
        ).pack(side="left")
        
        self.noise_strength_var = ctk.DoubleVar(value=0.5)
        self.noise_slider = ctk.CTkSlider(
            noise_frame,
            from_=0.0,
            to=1.0,
            variable=self.noise_strength_var,
            width=100
        )
        self.noise_slider.pack(side="left", padx=5)
        
        # Compression section
        comp_frame = ctk.CTkFrame(controls_frame, fg_color="transparent")
        comp_frame.pack(side="left", padx=10, pady=5)
        
        self.compression_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            comp_frame,
            text="Compression",
            variable=self.compression_var,
            width=100
        ).pack(side="left")
        
        # Limiter section
        limiter_frame = ctk.CTkFrame(controls_frame, fg_color="transparent")
        limiter_frame.pack(side="left", padx=10, pady=5)
        
        self.limiter_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            limiter_frame,
            text="Limiter",
            variable=self.limiter_var,
            width=80
        ).pack(side="left")
        
        # Process button
        self.process_btn = ctk.CTkButton(
            controls_frame,
            text="⚡ Process",
            command=self._process_audio,
            width=100
        )
        self.process_btn.pack(side="right", padx=10, pady=5)
        
        # === Action buttons ===
        action_frame = ctk.CTkFrame(self, fg_color="transparent")
        action_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(5, 10))
        
        # Play/Stop
        self.play_btn = ctk.CTkButton(
            action_frame,
            text="▶️ Play",
            command=self._toggle_play,
            width=80
        )
        self.play_btn.pack(side="left", padx=5)
        
        self.stop_btn = ctk.CTkButton(
            action_frame,
            text="⏹️ Stop",
            command=self._stop_playback,
            width=80
        )
        self.stop_btn.pack(side="left", padx=5)
        
        # Export
        self.export_btn = ctk.CTkButton(
            action_frame,
            text="💾 Export",
            command=self._export_audio,
            width=80
        )
        self.export_btn.pack(side="left", padx=5)
        
        # Status
        self.status_label = ctk.CTkLabel(
            action_frame,
            text="Ready",
            font=("Segoe UI", 10)
        )
        self.status_label.pack(side="right", padx=10)
    
    def _get_preset_names(self) -> list:
        """Lấy danh sách tên presets"""
        presets = self.preset_manager.list_audio_presets()
        return [p.name for p in presets]
    
    def _on_preset_change(self, preset_name: str):
        """Xử lý khi thay đổi preset"""
        preset = self.preset_manager.load_audio_preset(preset_name)
        if preset:
            self._apply_preset(preset)
    
    def _apply_preset(self, preset: AudioPreset):
        """Apply preset settings vào UI"""
        self.normalize_var.set(preset.normalize_enabled)
        self.norm_level_var.set(str(preset.normalize_target_db))
        self.noise_reduction_var.set(preset.noise_reduction_enabled)
        self.noise_strength_var.set(preset.noise_reduction_strength)
        self.compression_var.set(preset.compression_enabled)
        self.limiter_var.set(preset.limiter_enabled)
    
    def load_audio(self, audio_data: np.ndarray, sample_rate: int = 22050):
        """Load audio data để hiển thị và xử lý"""
        self._audio_data = audio_data
        self._sample_rate = sample_rate
        self._processed_audio = None
        
        # Update waveform
        self.waveform_viewer.load_audio(audio_data, sample_rate)
        
        # Update status
        duration = len(audio_data) / sample_rate
        self.status_label.configure(
            text=f"Loaded: {duration:.2f}s @ {sample_rate}Hz"
        )
    
    def load_audio_file(self, file_path: str):
        """Load audio từ file"""
        try:
            import soundfile as sf
            audio_data, sample_rate = sf.read(file_path)
            
            # Convert stereo to mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            self.load_audio(audio_data.astype(np.float32), sample_rate)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load audio: {e}")
    
    def _process_audio(self):
        """Process audio với current settings"""
        if self._audio_data is None:
            messagebox.showwarning("Warning", "No audio loaded")
            return
        
        self.status_label.configure(text="Processing...")
        self.process_btn.configure(state="disabled")
        
        def process():
            try:
                # Get settings
                normalize = self.normalize_var.get()
                target_db = float(self.norm_level_var.get())
                noise_reduce = self.noise_reduction_var.get()
                noise_strength = self.noise_strength_var.get()
                compress = self.compression_var.get()
                limit = self.limiter_var.get()
                
                # Process
                processed = self._audio_data.copy()
                
                if noise_reduce:
                    processed = self.audio_processor.noise_reducer.reduce_spectral(
                        processed,
                        strength=noise_strength
                    )
                
                if normalize:
                    processed = self.audio_processor.normalizer.normalize_peak(
                        processed,
                        target_db=target_db
                    )
                
                if compress:
                    processed = self.audio_processor.dynamic_processor.compress(
                        processed,
                        threshold_db=-20.0,
                        ratio=4.0
                    )
                
                if limit:
                    processed = self.audio_processor.dynamic_processor.limit(
                        processed,
                        threshold_db=-1.0
                    )
                
                self._processed_audio = processed
                
                # Update UI on main thread
                self.after(0, self._on_processing_complete)
                
            except Exception as e:
                self.after(0, lambda: self._on_processing_error(str(e)))
        
        thread = threading.Thread(target=process, daemon=True)
        thread.start()
    
    def _on_processing_complete(self):
        """Callback khi xử lý xong"""
        self.process_btn.configure(state="normal")
        self.status_label.configure(text="Processing complete!")
        
        # Update waveform với processed audio
        if self._processed_audio is not None:
            self.waveform_viewer.load_audio(
                self._processed_audio,
                self._sample_rate
            )
            
            # Callback
            if self.on_audio_processed:
                self.on_audio_processed(
                    self._processed_audio,
                    self._sample_rate
                )
    
    def _on_processing_error(self, error: str):
        """Callback khi có lỗi"""
        self.process_btn.configure(state="normal")
        self.status_label.configure(text=f"Error: {error}")
        messagebox.showerror("Processing Error", error)
    
    def _toggle_play(self):
        """Toggle play/pause"""
        # Implementation depends on audio playback system
        # This is a placeholder
        self.play_btn.configure(text="⏸️ Pause")
    
    def _stop_playback(self):
        """Stop playback"""
        self.play_btn.configure(text="▶️ Play")
        self.waveform_viewer.stop_playhead_animation()
    
    def _export_audio(self):
        """Export processed audio"""
        audio_to_export = self._processed_audio or self._audio_data
        
        if audio_to_export is None:
            messagebox.showwarning("Warning", "No audio to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[
                ("WAV files", "*.wav"),
                ("MP3 files", "*.mp3"),
                ("FLAC files", "*.flac")
            ]
        )
        
        if not file_path:
            return
        
        try:
            import soundfile as sf
            sf.write(file_path, audio_to_export, self._sample_rate)
            self.status_label.configure(text=f"Exported: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))
    
    def get_processed_audio(self) -> tuple:
        """Get processed audio"""
        return self._processed_audio, self._sample_rate
    
    def clear(self):
        """Clear audio data"""
        self._audio_data = None
        self._processed_audio = None
        self.waveform_viewer.clear()
        self.status_label.configure(text="Ready")


class PresetPanel(ctk.CTkFrame):
    """
    Panel quản lý presets
    """
    
    def __init__(
        self,
        parent,
        on_preset_select: Optional[Callable[[str], None]] = None,
        **kwargs
    ):
        super().__init__(parent, **kwargs)
        
        self.on_preset_select = on_preset_select
        self.preset_manager = get_preset_manager()
        
        self._create_widgets()
        self._load_presets()
    
    def _create_widgets(self):
        """Create UI"""
        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(
            header,
            text="📁 Presets",
            font=("Segoe UI", 12, "bold")
        ).pack(side="left")
        
        # Action buttons
        ctk.CTkButton(
            header,
            text="➕",
            width=30,
            command=self._add_preset
        ).pack(side="right", padx=2)
        
        ctk.CTkButton(
            header,
            text="🔄",
            width=30,
            command=self._refresh_presets
        ).pack(side="right", padx=2)
        
        # Preset list
        self.preset_listbox = ctk.CTkScrollableFrame(self, height=200)
        self.preset_listbox.pack(fill="both", expand=True, padx=10, pady=5)
    
    def _load_presets(self):
        """Load và hiển thị presets"""
        # Clear existing
        for widget in self.preset_listbox.winfo_children():
            widget.destroy()
        
        # Load audio presets
        presets = self.preset_manager.list_audio_presets()
        
        for preset in presets:
            frame = ctk.CTkFrame(self.preset_listbox)
            frame.pack(fill="x", pady=2)
            
            # Preset name
            btn = ctk.CTkButton(
                frame,
                text=preset.name,
                anchor="w",
                fg_color="transparent",
                text_color=("gray10", "gray90"),
                hover_color=("gray70", "gray30"),
                command=lambda p=preset.name: self._select_preset(p)
            )
            btn.pack(side="left", fill="x", expand=True)
            
            # Delete button
            ctk.CTkButton(
                frame,
                text="🗑️",
                width=30,
                fg_color="transparent",
                hover_color=("gray70", "gray30"),
                command=lambda p=preset.name: self._delete_preset(p)
            ).pack(side="right")
    
    def _select_preset(self, preset_name: str):
        """Select preset"""
        if self.on_preset_select:
            self.on_preset_select(preset_name)
    
    def _add_preset(self):
        """Add new preset"""
        # Show dialog
        dialog = ctk.CTkInputDialog(
            text="Enter preset name:",
            title="New Preset"
        )
        name = dialog.get_input()
        
        if name:
            preset = AudioPreset(name=name)
            self.preset_manager.save_audio_preset(preset)
            self._load_presets()
    
    def _delete_preset(self, preset_name: str):
        """Delete preset"""
        if messagebox.askyesno("Delete Preset", f"Delete '{preset_name}'?"):
            self.preset_manager.delete_audio_preset(preset_name)
            self._load_presets()
    
    def _refresh_presets(self):
        """Refresh preset list"""
        self.preset_manager.reload_all()
        self._load_presets()


class BatchProgressPanel(ctk.CTkFrame):
    """
    Panel hiển thị progress của batch processing
    """
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create UI"""
        # Header
        ctk.CTkLabel(
            self,
            text="📊 Batch Progress",
            font=("Segoe UI", 12, "bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self, width=300)
        self.progress_bar.pack(fill="x", padx=10, pady=5)
        self.progress_bar.set(0)
        
        # Stats frame
        stats_frame = ctk.CTkFrame(self, fg_color="transparent")
        stats_frame.pack(fill="x", padx=10, pady=5)
        
        # Left stats
        left_stats = ctk.CTkFrame(stats_frame, fg_color="transparent")
        left_stats.pack(side="left")
        
        self.completed_label = ctk.CTkLabel(
            left_stats,
            text="Completed: 0/0",
            font=("Segoe UI", 10)
        )
        self.completed_label.pack(anchor="w")
        
        self.failed_label = ctk.CTkLabel(
            left_stats,
            text="Failed: 0",
            font=("Segoe UI", 10)
        )
        self.failed_label.pack(anchor="w")
        
        # Right stats
        right_stats = ctk.CTkFrame(stats_frame, fg_color="transparent")
        right_stats.pack(side="right")
        
        self.time_label = ctk.CTkLabel(
            right_stats,
            text="Time: 00:00",
            font=("Segoe UI", 10)
        )
        self.time_label.pack(anchor="e")
        
        self.eta_label = ctk.CTkLabel(
            right_stats,
            text="ETA: --:--",
            font=("Segoe UI", 10)
        )
        self.eta_label.pack(anchor="e")
        
        # Current task
        self.current_task_label = ctk.CTkLabel(
            self,
            text="Current: None",
            font=("Segoe UI", 9),
            text_color="gray"
        )
        self.current_task_label.pack(anchor="w", padx=10, pady=5)
    
    def update_progress(self, progress):
        """
        Update progress display
        
        Args:
            progress: BatchProgress object
        """
        # Update progress bar
        percent = progress.progress_percent / 100.0
        self.progress_bar.set(percent)
        
        # Update labels
        self.completed_label.configure(
            text=f"Completed: {progress.completed_tasks}/{progress.total_tasks}"
        )
        self.failed_label.configure(
            text=f"Failed: {progress.failed_tasks}"
        )
        
        # Update time
        if progress.elapsed_time:
            elapsed = int(progress.elapsed_time.total_seconds())
            minutes, seconds = divmod(elapsed, 60)
            self.time_label.configure(text=f"Time: {minutes:02d}:{seconds:02d}")
        
        # Update ETA
        if progress.remaining_time:
            remaining = int(progress.remaining_time.total_seconds())
            if remaining > 0:
                minutes, seconds = divmod(remaining, 60)
                self.eta_label.configure(text=f"ETA: {minutes:02d}:{seconds:02d}")
            else:
                self.eta_label.configure(text="ETA: Done!")
        
        # Update current task
        if progress.current_task_text:
            text = progress.current_task_text[:40]
            self.current_task_label.configure(
                text=f"Current: {progress.current_task_index}. {text}..."
            )
    
    def reset(self):
        """Reset progress display"""
        self.progress_bar.set(0)
        self.completed_label.configure(text="Completed: 0/0")
        self.failed_label.configure(text="Failed: 0")
        self.time_label.configure(text="Time: 00:00")
        self.eta_label.configure(text="ETA: --:--")
        self.current_task_label.configure(text="Current: None")
