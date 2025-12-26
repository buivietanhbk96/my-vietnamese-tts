"""
SRT Processing Panel Component for VietTTS
"""

import os
import customtkinter as ctk
from pathlib import Path
from typing import Callable, Optional
from tkinter import filedialog

from app.config import Config, ThemeColors, Fonts
from app.srt_processor import SRTParser, validate_srt_file
from ui.theme import (
    ThemedFrame, ThemedLabel, ThemedButton, ThemedProgressBar, ThemedTextbox
)


class SRTPanel(ThemedFrame):
    """
    SRT file processing panel for batch TTS generation
    """
    
    def __init__(
        self,
        master,
        on_process: Optional[Callable] = None,
        on_cancel: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(master, style="card", **kwargs)
        
        self.on_process = on_process
        self.on_cancel = on_cancel
        
        self.srt_file = None
        self.subtitle_count = 0
        self.is_processing = False
        
        self._create_widgets()
        self._setup_layout()
    
    def _create_widgets(self):
        """Create panel widgets"""
        # Title
        self.title_label = ThemedLabel(
            self,
            text="📄 Import SRT",
            style="subheading"
        )
        
        # Description
        self.desc_label = ThemedLabel(
            self,
            text="Import file phụ đề SRT để tạo audio cho từng subtitle",
            style="muted"
        )
        
        # File selection
        self.file_frame = ThemedFrame(self, style="transparent")
        
        self.browse_btn = ThemedButton(
            self.file_frame,
            text="📁 Chọn file SRT...",
            style="secondary",
            width=150,
            height=36,
            command=self._on_browse
        )
        
        self.file_label = ThemedLabel(
            self.file_frame,
            text="Chưa chọn file",
            style="muted"
        )
        
        # File info
        self.info_frame = ThemedFrame(self, style="transparent")
        # Will be shown after SRT file is loaded
        
        self.subtitle_count_label = ThemedLabel(
            self.info_frame,
            text="",
            style="default"
        )
        
        # Preview
        self.preview_frame = ThemedFrame(self, style="transparent")
        # Will be shown after SRT file is loaded
        
        self.preview_label = ThemedLabel(
            self.preview_frame,
            text="Preview (5 dòng đầu):",
            style="muted"
        )
        
        self.preview_text = ThemedTextbox(
            self.preview_frame,
            height=100,
            font=Fonts.BODY_SMALL,
            state="disabled"
        )
        
        # Range selection
        self.range_frame = ThemedFrame(self, style="transparent")
        # Range selection - will be shown after SRT file is loaded
        self.range_frame = ThemedFrame(self, style="transparent")
        
        self.range_label = ThemedLabel(
            self.range_frame,
            text="Phạm vi:",
            style="default"
        )
        
        self.start_entry = ctk.CTkEntry(
            self.range_frame,
            width=80,
            placeholder_text="Từ",
            fg_color=ThemeColors.BG_TERTIARY,
            border_color=ThemeColors.BORDER,
            text_color=ThemeColors.TEXT_PRIMARY
        )
        
        self.range_to_label = ThemedLabel(
            self.range_frame,
            text="đến",
            style="default"
        )
        
        self.end_entry = ctk.CTkEntry(
            self.range_frame,
            width=80,
            placeholder_text="Cuối",
            fg_color=ThemeColors.BG_TERTIARY,
            border_color=ThemeColors.BORDER,
            text_color=ThemeColors.TEXT_PRIMARY
        )
        
        # Progress - will be shown during processing
        self.progress_frame = ThemedFrame(self, style="transparent")
        
        self.progress_bar = ThemedProgressBar(
            self.progress_frame,
            width=400
        )
        self.progress_bar.set(0)
        
        self.progress_label = ThemedLabel(
            self.progress_frame,
            text="",
            style="muted"
        )
        
        # Actions
        self.action_frame = ThemedFrame(self, style="transparent")
        
        self.process_btn = ThemedButton(
            self.action_frame,
            text="🚀 Bắt đầu xử lý",
            style="primary",
            width=150,
            height=40,
            command=self._on_process,
            state="disabled"
        )
        
        self.cancel_btn = ThemedButton(
            self.action_frame,
            text="⏹ Hủy",
            style="danger",
            width=100,
            height=40,
            command=self._on_cancel
        )
        # Will be packed during processing
        
        # Output info
        self.output_label = ThemedLabel(
            self,
            text="",
            style="muted"
        )
    
    def _setup_layout(self):
        """Setup widget layout"""
        self.title_label.pack(anchor="w", padx=15, pady=(15, 5))
        self.desc_label.pack(anchor="w", padx=15, pady=(0, 15))
        
        # File selection
        self.file_frame.pack(fill="x", padx=15, pady=5)
        self.browse_btn.pack(side="left")
        self.file_label.pack(side="left", padx=15)
        
        # Info (content setup - will be packed when SRT loaded)
        self.subtitle_count_label.pack(anchor="w")
        
        # Preview (content setup - will be packed when SRT loaded)
        self.preview_label.pack(anchor="w", pady=(0, 5))
        self.preview_text.pack(fill="x")
        
        # Range (content setup - will be packed when SRT loaded)
        self.range_label.pack(side="left")
        self.start_entry.pack(side="left", padx=(10, 5))
        self.range_to_label.pack(side="left", padx=5)
        self.end_entry.pack(side="left", padx=(5, 0))
        
        # Progress (content setup - will be packed during processing)
        self.progress_bar.pack(fill="x", pady=(0, 5))
        self.progress_label.pack(anchor="w")
        
        # Actions
        self.action_frame.pack(fill="x", padx=15, pady=15)
        self.process_btn.pack(side="left")
        # cancel_btn will be packed during processing
        
        # Output
        self.output_label.pack(anchor="w", padx=15, pady=(0, 15))
    
    def _on_browse(self):
        """Open file dialog for SRT selection"""
        filetypes = [
            ("SRT files", "*.srt"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(
            title="Chọn file SRT",
            filetypes=filetypes
        )
        
        if filepath:
            self._load_srt(filepath)
    
    def _load_srt(self, filepath: str):
        """Load and validate SRT file"""
        is_valid, message, count = validate_srt_file(filepath)
        
        if is_valid:
            self.srt_file = filepath
            self.subtitle_count = count
            
            # Update UI
            filename = Path(filepath).name
            self.file_label.configure(text=filename)
            self.subtitle_count_label.configure(
                text=f"✓ {count} subtitle được tìm thấy"
            )
            
            # Show sections
            self.info_frame.pack(fill="x", padx=15, pady=10)
            self.preview_frame.pack(fill="x", padx=15, pady=5)
            self.range_frame.pack(fill="x", padx=15, pady=10)
            
            # Load preview
            self._load_preview()
            
            # Set range defaults
            self.start_entry.delete(0, "end")
            self.start_entry.insert(0, "1")
            self.end_entry.delete(0, "end")
            self.end_entry.insert(0, str(count))
            
            # Enable process button
            self.process_btn.configure(state="normal")
            
            self.output_label.configure(
                text=f"Output: {count} files ({1}.wav - {count}.wav)"
            )
        else:
            self.file_label.configure(text=f"❌ Lỗi: {message}")
            self.process_btn.configure(state="disabled")
    
    def _load_preview(self):
        """Load preview of first 5 subtitles"""
        if not self.srt_file:
            return
        
        try:
            parser = SRTParser()
            subtitles = parser.parse_file(self.srt_file)
            
            # Get first 5
            preview_lines = []
            for sub in subtitles[:5]:
                text = sub.text[:60] + "..." if len(sub.text) > 60 else sub.text
                preview_lines.append(f"{sub.index}. {text}")
            
            if len(subtitles) > 5:
                preview_lines.append(f"... và {len(subtitles) - 5} dòng nữa")
            
            # Update preview
            self.preview_text.configure(state="normal")
            self.preview_text.delete("1.0", "end")
            self.preview_text.insert("1.0", "\n".join(preview_lines))
            self.preview_text.configure(state="disabled")
            
        except Exception as e:
            self.preview_text.configure(state="normal")
            self.preview_text.delete("1.0", "end")
            self.preview_text.insert("1.0", f"Error loading preview: {e}")
            self.preview_text.configure(state="disabled")
    
    def _on_process(self):
        """Handle process button click"""
        if not self.srt_file or self.is_processing:
            return
        
        # Get range
        try:
            start = int(self.start_entry.get() or 1)
            end_text = self.end_entry.get()
            end = int(end_text) if end_text else self.subtitle_count
        except ValueError:
            self.output_label.configure(text="❌ Phạm vi không hợp lệ")
            return
        
        if self.on_process:
            self.on_process(self.srt_file, start, end)
    
    def _on_cancel(self):
        """Handle cancel button click"""
        if self.on_cancel:
            self.on_cancel()
    
    def start_processing(self):
        """Update UI for processing state"""
        self.is_processing = True
        self.process_btn.configure(state="disabled")
        self.cancel_btn.pack(side="left", padx=10)
        self.browse_btn.configure(state="disabled")
        
        self.progress_frame.pack(fill="x", padx=15, pady=10)
        self.progress_bar.set(0)
    
    def update_progress(self, current: int, total: int, status: str = ""):
        """Update progress bar"""
        progress = current / total if total > 0 else 0
        self.progress_bar.set(progress)
        self.progress_label.configure(text=f"{status} ({current}/{total})")
    
    def finish_processing(self, success: bool = True, message: str = ""):
        """Update UI after processing complete"""
        self.is_processing = False
        self.process_btn.configure(state="normal")
        self.cancel_btn.pack_forget()
        self.browse_btn.configure(state="normal")
        
        if success:
            self.progress_bar.set(1)
            self.output_label.configure(text=f"✓ {message}")
        else:
            self.output_label.configure(text=f"❌ {message}")
    
    def get_srt_file(self) -> Optional[str]:
        """Get selected SRT file path"""
        return self.srt_file
    
    def get_range(self) -> tuple:
        """Get selected subtitle range"""
        try:
            start = int(self.start_entry.get() or 1)
            end_text = self.end_entry.get()
            end = int(end_text) if end_text else self.subtitle_count
            return (start, end)
        except ValueError:
            return (1, self.subtitle_count)
    
    def reset(self):
        """Reset panel to initial state"""
        self.srt_file = None
        self.subtitle_count = 0
        self.is_processing = False
        
        self.file_label.configure(text="Chưa chọn file")
        self.process_btn.configure(state="disabled")
        self.progress_bar.set(0)
        self.progress_label.configure(text="")
        self.output_label.configure(text="")
        
        # Hide sections
        self.info_frame.pack_forget()
        self.preview_frame.pack_forget()
        self.range_frame.pack_forget()
        self.progress_frame.pack_forget()
        self.cancel_btn.pack_forget()
