"""
Main Window for VietTTS Desktop Application

Keyboard Shortcuts:
    Ctrl+Enter  - Generate speech
    Ctrl+S      - Save current audio
    Ctrl+O      - Open SRT file
    Ctrl+P      - Play/Pause audio
    Space       - Play/Pause audio (when audio panel focused)
    Escape      - Stop audio playback
    F5          - Refresh voices
    Ctrl+Q      - Quit application
"""

import os
import customtkinter as ctk
from pathlib import Path
from typing import Optional, Dict, Callable
from datetime import datetime
from tkinter import messagebox
from loguru import logger

from app.config import Config, ThemeColors, Fonts
from app.tts_engine import TTSEngine
from app.device_manager import get_device_manager
from app.srt_processor import SRTProcessor, SRTParser
from app.worker import TTSWorker, ResultCallback, ErrorCallback, ProgressCallback
from app.exceptions import VietTTSError

from ui.theme import (
    setup_theme, ThemedFrame, ThemedLabel, ThemedButton, ThemedTabview
)
from ui.components import (
    TextInputPanel, VoiceSelectorPanel, AudioPlayerPanel,
    SettingsPanel, SRTPanel, StatusBar
)
from ui.splash_screen import SplashScreen

from utils.settings import Settings
from utils.voice_downloader import VoiceDownloader
from utils.ffmpeg_check import check_ffmpeg, get_ffmpeg_install_instructions


class MainWindow(ctk.CTk):
    """
    Main application window for VietTTS
    """
    
    # Keyboard shortcut definitions
    SHORTCUTS: Dict[str, str] = {
        "<Control-Return>": "Generate speech",
        "<Control-s>": "Save audio",
        "<Control-o>": "Open SRT file",
        "<Control-p>": "Play/Pause",
        "<Escape>": "Stop playback",
        "<F5>": "Refresh voices",
        "<Control-q>": "Quit",
    }
    
    def __init__(self):
        super().__init__()
        
        # Setup theme
        setup_theme()
        
        # Window configuration
        self.title(Config.WINDOW_TITLE)
        self.geometry(Config.DEFAULT_WINDOW_SIZE)
        self.minsize(Config.WINDOW_MIN_WIDTH, Config.WINDOW_MIN_HEIGHT)
        self.configure(fg_color=ThemeColors.BG_PRIMARY)
        
        # Initialize state
        self.settings = Settings()
        self.tts_engine: Optional[TTSEngine] = None
        self.worker: Optional[TTSWorker] = None
        self.srt_processor: Optional[SRTProcessor] = None
        self.current_output_file: Optional[str] = None
        
        # Show splash screen
        self.withdraw()  # Hide main window initially
        self.splash = SplashScreen(self)
        
        # Start initialization
        self.after(100, self._initialize_app)
    
    def _initialize_app(self):
        """Initialize application in background"""
        try:
            # Check FFmpeg
            self.splash.update_progress(0.1, "Checking FFmpeg...")
            ffmpeg_ok, ffmpeg_msg = check_ffmpeg()
            
            if not ffmpeg_ok:
                self.splash.close()
                self.deiconify()
                self._show_ffmpeg_warning()
            
            # Check/download voice samples
            self.splash.update_progress(0.2, "Checking voice samples...")
            self._ensure_voice_samples()
            
            # Initialize TTS engine
            self.splash.update_progress(0.3, "Initializing TTS engine...")
            self._init_tts_engine()
            
            # Create UI
            self.splash.update_progress(0.8, "Creating interface...")
            self._create_ui()
            
            # Setup keyboard shortcuts
            self._setup_keyboard_shortcuts()
            
            # Load saved settings
            self.splash.update_progress(0.9, "Loading settings...")
            self._load_settings()
            
            # Complete
            self.splash.update_progress(1.0, "Ready!")
            self.after(500, self._show_main_window)
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.splash.close()
            self.deiconify()
            messagebox.showerror("Error", f"Failed to initialize: {str(e)}")
    
    def _show_main_window(self):
        """Show main window and close splash"""
        self.splash.close()
        
        # Clear splash callback to avoid "invalid command name" errors
        if self.tts_engine:
            self.tts_engine.progress_callback = None
            
        self.deiconify()
        self.lift()
    
    def _show_ffmpeg_warning(self):
        """Show FFmpeg warning dialog"""
        result = messagebox.askquestion(
            "FFmpeg Not Found",
            "FFmpeg is required for audio processing but was not found.\n\n"
            "Do you want to see installation instructions?",
            icon='warning'
        )
        
        if result == 'yes':
            # Show instructions in a dialog
            instructions = get_ffmpeg_install_instructions()
            dialog = ctk.CTkToplevel(self)
            dialog.title("FFmpeg Installation")
            dialog.geometry("600x400")
            
            text = ctk.CTkTextbox(dialog, font=Fonts.MONO)
            text.pack(fill="both", expand=True, padx=20, pady=20)
            text.insert("1.0", instructions)
            text.configure(state="disabled")
    
    def _ensure_voice_samples(self):
        """Ensure voice samples are available"""
        downloader = VoiceDownloader()
        local_voices = downloader.get_local_voices()
        
        if not local_voices:
            self.splash.set_status("Downloading voice samples...")
            
            def progress_callback(idx, total, filename, status):
                progress = 0.2 + (idx / total) * 0.1
                self.splash.update_progress(progress, f"Downloading {filename}...")
            
            downloader.download_all(progress_callback=progress_callback)
    
    def _init_tts_engine(self):
        """Initialize TTS engine with non-blocking model loading"""
        def progress_callback(status: str, progress: float):
            overall_progress = 0.3 + progress * 0.5
            self.splash.update_progress(overall_progress, status)
        
        self.tts_engine = TTSEngine(progress_callback=progress_callback)
        
        # Start worker
        self.worker = TTSWorker(self.tts_engine)
        self.worker.start()
        
        # Load model in a separate thread to keep splash screen responsive
        import threading
        load_finished = threading.Event()
        load_error = []
        
        def do_load():
            try:
                self.tts_engine.load_model()
            except Exception as e:
                load_error.append(e)
            finally:
                load_finished.set()
        
        thread = threading.Thread(target=do_load, daemon=True)
        thread.start()
        
        # Wait for load to finish while keeping UI responsive
        while not load_finished.is_set():
            self.update()
            # Wait a bit for the thread to work, but not too long to keep UI responsive
            load_finished.wait(0.05)
        
        if load_error:
            logger.error(f"Model loading failed: {load_error[0]}")
            # We don't raise here to allow the UI to open, 
            # but tts_engine handles its own error states
        
        # Initialize SRT processor
        self.srt_processor = SRTProcessor(self.tts_engine)
    
    def _setup_keyboard_shortcuts(self) -> None:
        """Setup keyboard shortcuts for the application."""
        # Generate speech: Ctrl+Enter
        self.bind("<Control-Return>", lambda e: self._on_generate())
        
        # Save audio: Ctrl+S
        self.bind("<Control-s>", lambda e: self._on_shortcut_save())
        
        # Play/Pause: Ctrl+P
        self.bind("<Control-p>", lambda e: self._on_shortcut_play_pause())
        
        # Stop playback: Escape
        self.bind("<Escape>", lambda e: self._on_shortcut_stop())
        
        # Refresh voices: F5
        self.bind("<F5>", lambda e: self._on_shortcut_refresh())
        
        # Quit: Ctrl+Q
        self.bind("<Control-q>", lambda e: self.on_closing())
        
        # Show shortcuts help: F1
        self.bind("<F1>", lambda e: self._show_shortcuts_help())
        
        logger.info("Keyboard shortcuts configured")
    
    def _on_shortcut_save(self) -> None:
        """Handle Ctrl+S shortcut to open output folder."""
        if self.settings.output_directory:
            os.startfile(self.settings.output_directory)
    
    def _on_shortcut_play_pause(self) -> None:
        """Play/Pause removed with audio panel."""
        pass
    
    def _on_shortcut_stop(self) -> None:
        """Stop playback removed with audio panel."""
        pass
    
    def _on_shortcut_refresh(self) -> None:
        """Handle F5 shortcut to refresh voices."""
        if self.tts_engine:
            self.tts_engine.reload_voices()
            voices = self.tts_engine.get_available_voices()
            if hasattr(self, 'voice_panel'):
                self.voice_panel.set_voices(voices)
            if hasattr(self, 'srt_voice_panel'):
                self.srt_voice_panel.set_voices(voices)
            self.status_bar.set_success("Voices refreshed")
    
    def _show_shortcuts_help(self) -> None:
        """Show keyboard shortcuts help dialog."""
        shortcuts_text = """Keyboard Shortcuts:

Ctrl+Enter  - Generate speech from text
Ctrl+S      - Save current audio file
Ctrl+P      - Play/Pause audio
Escape      - Stop audio playback
F5          - Refresh voice list
Ctrl+Q      - Quit application
F1          - Show this help"""
        
        messagebox.showinfo("Keyboard Shortcuts", shortcuts_text)

    def _create_ui(self):
        """Create main UI layout"""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)  # Header
        self.grid_rowconfigure(1, weight=1)  # Content
        self.grid_rowconfigure(2, weight=0)  # Status bar
        
        # Header
        self._create_header()
        
        # Main content with tabs
        self._create_content()
        
        # Status bar
        self._create_status_bar()
    
    def _create_header(self):
        """Create header section"""
        self.header_frame = ThemedFrame(self, style="transparent")
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        
        self.logo_label = ctk.CTkLabel(
            self.header_frame,
            text="🎙️ VietTTS",
            font=("Segoe UI", 24, "bold"),
            text_color=ThemeColors.TEXT_PRIMARY
        )
        self.logo_label.pack(side="left")
        
        self.tagline_label = ctk.CTkLabel(
            self.header_frame,
            text="Vietnamese Text-to-Speech",
            font=Fonts.BODY,
            text_color=ThemeColors.TEXT_SECONDARY
        )
        self.tagline_label.pack(side="left", padx=15)
    
    def _create_content(self):
        """Create main content area with tabs"""
        self.content_frame = ThemedFrame(self, style="transparent")
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)
        
        # Tab view
        self.tabview = ThemedTabview(
            self.content_frame,
            width=800,
            height=500
        )
        self.tabview.grid(row=0, column=0, sticky="nsew")
        
        # Create tabs
        self.tab_tts = self.tabview.add("💬 Text-to-Speech")
        self.tab_srt = self.tabview.add("📄 SRT Import")
        
        # TTS Tab content
        self._create_tts_tab()
        
        # SRT Tab content
        self._create_srt_tab()
    
    def _create_tts_tab(self):
        """Create TTS tab content"""
        self.tab_tts.grid_columnconfigure(0, weight=1)
        self.tab_tts.grid_columnconfigure(1, weight=0)
        self.tab_tts.grid_rowconfigure(0, weight=1)
        self.tab_tts.grid_rowconfigure(1, weight=0)
        self.tab_tts.grid_rowconfigure(2, weight=0)
        
        # Left column - Text input
        self.text_panel = TextInputPanel(
            self.tab_tts,
            on_text_change=self._on_text_change
        )
        self.text_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=5)
        
        # Right column - Voice & Settings
        self.right_frame = ThemedFrame(self.tab_tts, style="transparent")
        self.right_frame.grid(row=0, column=1, sticky="nsew", pady=5)
        
        # Get available voices
        voices = self.tts_engine.get_available_voices() if self.tts_engine else {}
        
        self.voice_panel = VoiceSelectorPanel(
            self.right_frame,
            voices=voices,
            on_voice_change=self._on_voice_change
        )
        self.voice_panel.pack(fill="x", pady=(0, 10))
        
        self.settings_panel = SettingsPanel(
            self.right_frame,
            on_settings_change=self._on_settings_change
        )
        self.settings_panel.pack(fill="x")
        
        # Generate button
        self.generate_frame = ThemedFrame(self.tab_tts, style="transparent")
        self.generate_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=15)
        
        self.generate_btn = ThemedButton(
            self.generate_frame,
            text="🎙️ Generate Speech",
            style="primary",
            width=200,
            height=50,
            font=("Segoe UI", 14, "bold"),
            command=self._on_generate
        )
        self.generate_btn.pack()
        
        # Progress bar (hidden initially)
        self.tts_progress_frame = ThemedFrame(self.tab_tts, style="transparent")
        self.tts_progress_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
        
        from ui.theme import ThemedProgressBar
        self.tts_progress = ThemedProgressBar(self.tts_progress_frame, width=600)
        self.tts_progress.pack(pady=5)
        self.tts_progress.set(0)
        self.tts_progress_frame.grid_remove()  # Hide initially
    
    def _create_srt_tab(self):
        """Create SRT import tab content"""
        self.tab_srt.grid_columnconfigure(0, weight=1)
        self.tab_srt.grid_columnconfigure(1, weight=0)
        self.tab_srt.grid_rowconfigure(0, weight=1)
        
        # SRT Panel
        self.srt_panel = SRTPanel(
            self.tab_srt,
            on_process=self._on_srt_process,
            on_cancel=self._on_srt_cancel
        )
        self.srt_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=5)
        
        # Right side - Voice & Settings for SRT
        self.srt_right_frame = ThemedFrame(self.tab_srt, style="transparent")
        self.srt_right_frame.grid(row=0, column=1, sticky="nsew", pady=5)
        
        voices = self.tts_engine.get_available_voices() if self.tts_engine else {}
        
        self.srt_voice_panel = VoiceSelectorPanel(
            self.srt_right_frame,
            voices=voices,
            on_voice_change=self._on_voice_change
        )
        self.srt_voice_panel.pack(fill="x", pady=(0, 10))
        
        self.srt_settings_panel = SettingsPanel(
            self.srt_right_frame,
            on_settings_change=self._on_settings_change
        )
        self.srt_settings_panel.pack(fill="x")
    
    def _create_status_bar(self):
        """Create status bar"""
        self.status_bar = StatusBar(self)
        self.status_bar.grid(row=2, column=0, sticky="ew")
        
        # Set initial status
        self.status_bar.set_ready()
        
        # Show actual device info from device manager
        device_manager = get_device_manager()
        device_display = device_manager.device_name
        if device_manager.is_gpu:
            device_display = f"🚀 {device_display}"
        self.status_bar.set_device_info(device_display)
    
    def _load_settings(self):
        """Load saved settings"""
        # Apply saved voice
        if self.settings.last_voice:
            self.voice_panel.select_voice(self.settings.last_voice)
            self.srt_voice_panel.select_voice(self.settings.last_voice)
        
        # Apply saved speed
        self.settings_panel.set_speed(self.settings.speed)
        self.srt_settings_panel.set_speed(self.settings.speed)
        
        # Apply saved output directory
        if self.settings.output_directory:
            self.settings_panel.set_output_directory(self.settings.output_directory)
            self.srt_settings_panel.set_output_directory(self.settings.output_directory)
    
    def _on_text_change(self, text: str):
        """Handle text input change"""
        # Enable/disable generate button based on text
        if text.strip():
            self.generate_btn.configure(state="normal")
        else:
            self.generate_btn.configure(state="disabled")
    
    def _on_voice_change(self, voice_path: Optional[str]):
        """Handle voice selection change"""
        if voice_path:
            self.settings.last_voice = self.voice_panel.get_voice_name()
    
    def _on_settings_change(self, settings: dict):
        """Handle settings change"""
        self.settings.speed = settings.get("speed", 1.0)
        self.settings.output_format = settings.get("output_format", "wav")
        self.settings.output_directory = settings.get("output_directory", str(Config.OUTPUT_DIR))
    
    def _on_generate(self):
        """Handle generate button click"""
        text = self.text_panel.get_text()
        
        if not text:
            messagebox.showwarning("Warning", "Please enter some text to synthesize.")
            return
        
        voice_path = self.voice_panel.get_selected_voice()
        
        if not voice_path:
            messagebox.showwarning("Warning", "Please select a voice.")
            return
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_format = self.settings_panel.get_output_format()
        output_dir = self.settings_panel.get_output_directory()
        output_path = Path(output_dir) / f"tts_{timestamp}.{output_format}"
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Update UI for processing
        self.generate_btn.configure(state="disabled", text="⏳ Generating...")
        self.status_bar.set_processing("Generating speech...")
        self.tts_progress_frame.grid()
        self.tts_progress.set(0)
        
        # Setup progress callback
        def progress_callback(idx, total, status):
            # idx is actually 0.0-1.0 progress for single synthesis
            self.tts_progress.set(idx)
            self.status_bar.set_processing(status)
        
        self.worker.on_progress = ProgressCallback(self, progress_callback)
        
        # Get reference text for voice cloning (if in clone mode)
        ref_text = self.voice_panel.get_ref_text()
        
        # Submit to worker
        self.worker.submit_synthesize(
            text=text,
            voice_path=voice_path,
            output_path=str(output_path),
            speed=self.settings_panel.get_speed(),
            ref_text=ref_text,  # User-provided reference text for voice cloning
            callback=ResultCallback(self, self._on_generate_complete),
            error_callback=ErrorCallback(self, self._on_generate_error)
        )
    
    def _on_generate_complete(self, output_path: str):
        """Handle generation complete"""
        self.current_output_file = output_path
        
        # Update UI
        self.generate_btn.configure(state="normal", text="🎙️ Generate Speech")
        self.status_bar.set_success(f"Generated: {Path(output_path).name}")
        self.tts_progress.set(1)
        
        # Hide progress after delay
        self.after(2000, lambda: self.tts_progress_frame.grid_remove())
        
        # Open file with default system player (prevents UI freeze)
        try:
            if os.path.exists(output_path):
                os.startfile(output_path)
        except Exception as e:
            logger.error(f"Failed to open audio: {e}")
        
        # Add to recent files
        self.settings.add_recent_file(output_path)
        
        # Save ref_text to cache for clone voices (so user doesn't need to re-enter)
        self.voice_panel.save_ref_text_to_cache()
    
    def _on_generate_error(self, error: Exception):
        """Handle generation error"""
        self.generate_btn.configure(state="normal", text="🎙️ Generate Speech")
        self.status_bar.set_error(str(error))
        self.tts_progress_frame.grid_remove()
        
        messagebox.showerror("Error", f"Generation failed: {str(error)}")
    

    
    def _on_srt_process(self, srt_file: str, start_idx: int, end_idx: int):
        """Handle SRT processing request"""
        voice_path = self.srt_voice_panel.get_selected_voice()
        
        if not voice_path:
            messagebox.showwarning("Warning", "Please select a voice.")
            return
        
        # Parse SRT
        parser = SRTParser()
        subtitles = parser.parse_file(srt_file)
        
        # Filter by range
        texts = [
            (sub.index, sub.text) for sub in subtitles
            if start_idx <= sub.index <= end_idx and sub.text.strip()
        ]
        
        if not texts:
            messagebox.showwarning("Warning", "No subtitles found in specified range.")
            return
        
        # Create output directory
        srt_name = Path(srt_file).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.srt_settings_panel.get_output_directory()) / f"{srt_name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Update UI
        self.srt_panel.start_processing()
        self.status_bar.set_processing("Processing SRT...")
        
        # Setup progress callback
        def progress_callback(idx, total, status):
            self.srt_panel.update_progress(idx, total, status)
            self.status_bar.set_processing(f"Processing {idx}/{total}...")
        
        self.worker.on_progress = ProgressCallback(self, progress_callback)
        
        # Submit to worker
        self.worker.submit_batch_synthesize(
            texts=texts,
            voice_path=voice_path,
            output_dir=str(output_dir),
            speed=self.srt_settings_panel.get_speed(),
            callback=ResultCallback(self, lambda files: self._on_srt_complete(files, output_dir)),
            error_callback=ErrorCallback(self, self._on_srt_error)
        )
    
    def _on_srt_complete(self, output_files: list, output_dir: Path):
        """Handle SRT processing complete"""
        self.srt_panel.finish_processing(
            success=True,
            message=f"Generated {len(output_files)} files in {output_dir}"
        )
        self.status_bar.set_success(f"Generated {len(output_files)} audio files")
        
        # Open output folder
        result = messagebox.askyesno(
            "Complete",
            f"Successfully generated {len(output_files)} audio files.\n\n"
            f"Output folder: {output_dir}\n\n"
            "Do you want to open the output folder?"
        )
        
        if result:
            os.startfile(str(output_dir))
    
    def _on_srt_error(self, error: Exception):
        """Handle SRT processing error"""
        self.srt_panel.finish_processing(success=False, message=str(error))
        self.status_bar.set_error(str(error))
        messagebox.showerror("Error", f"SRT processing failed: {str(error)}")
    
    def _on_srt_cancel(self):
        """Handle SRT processing cancellation"""
        if self.worker:
            self.worker.cancel_current()
        self.srt_panel.finish_processing(success=False, message="Cancelled by user")
        self.status_bar.set_ready()
    
    def on_closing(self):
        """Handle window closing"""
        # Save settings
        self.settings.save()
        
        # Cleanup
        if self.worker:
            self.worker.stop()
        
        if self.tts_engine:
            self.tts_engine.cleanup()
        
        self.destroy()


def run_app():
    """Run the application"""
    # Ensure directories exist
    Config.ensure_directories()
    
    # Create and run app
    app = MainWindow()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
