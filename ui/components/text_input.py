"""
Text Input Panel Component for VietTTS
"""

import customtkinter as ctk
from typing import Callable, Optional

from app.config import ThemeColors, Fonts
from ui.theme import (
    ThemedFrame, ThemedLabel, ThemedButton, ThemedTextbox
)


class TextInputPanel(ThemedFrame):
    """
    Text input panel with character count and clear button
    """
    
    def __init__(
        self,
        master,
        on_text_change: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(master, style="card", **kwargs)
        
        self.on_text_change = on_text_change
        
        self._create_widgets()
        self._setup_layout()
        self.bind_focus_in()
    
    def _create_widgets(self):
        """Create panel widgets"""
        # Header
        self.header_frame = ThemedFrame(self, style="transparent")
        
        self.title_label = ThemedLabel(
            self.header_frame,
            text="📝 Văn bản",
            style="subheading"
        )
        
        self.char_count_label = ThemedLabel(
            self.header_frame,
            text="0 ký tự",
            style="muted"
        )
        
        self.clear_btn = ThemedButton(
            self.header_frame,
            text="Xóa",
            style="ghost",
            width=60,
            height=28,
            command=self._on_clear
        )
        
        # Text input
        self.textbox = ThemedTextbox(
            self,
            height=150,
            font=Fonts.BODY,
            wrap="word"
        )
        
        # Bind text change event
        self.textbox.bind("<KeyRelease>", self._on_text_modified)
        
        # Placeholder
        self._placeholder_text = "Nhập văn bản tiếng Việt tại đây..."
        self._show_placeholder()
    
    def _setup_layout(self):
        """Setup widget layout"""
        # Header
        self.header_frame.pack(fill="x", padx=15, pady=(15, 10))
        self.title_label.pack(side="left")
        self.clear_btn.pack(side="right")
        self.char_count_label.pack(side="right", padx=10)
        
        # Text input
        self.textbox.pack(fill="both", expand=True, padx=15, pady=(0, 15))
    
    def _show_placeholder(self):
        """Show placeholder text"""
        self.textbox.insert("1.0", self._placeholder_text)
        self.textbox.configure(text_color=ThemeColors.TEXT_MUTED)
        self._has_placeholder = True
    
    def _hide_placeholder(self):
        """Hide placeholder text"""
        if self._has_placeholder:
            self.textbox.delete("1.0", "end")
            self.textbox.configure(text_color=ThemeColors.TEXT_PRIMARY)
            self._has_placeholder = False
    
    def _on_text_modified(self, event=None):
        """Handle text modification"""
        # Handle placeholder
        current_text = self.textbox.get("1.0", "end-1c")
        
        if self._has_placeholder and current_text != self._placeholder_text:
            self._hide_placeholder()
        
        # Update character count
        text = self.get_text()
        char_count = len(text)
        self.char_count_label.configure(text=f"{char_count} ký tự")
        
        # Callback
        if self.on_text_change:
            self.on_text_change(text)
    
    def _on_clear(self):
        """Clear text input"""
        self.textbox.delete("1.0", "end")
        self._show_placeholder()
        self._on_text_modified()
    
    def get_text(self) -> str:
        """Get current text content"""
        text = self.textbox.get("1.0", "end-1c").strip()
        
        if self._has_placeholder or text == self._placeholder_text:
            return ""
        
        return text
    
    def set_text(self, text: str):
        """Set text content"""
        self._hide_placeholder()
        self.textbox.delete("1.0", "end")
        self.textbox.insert("1.0", text)
        self._on_text_modified()
    
    def is_empty(self) -> bool:
        """Check if text input is empty"""
        return len(self.get_text()) == 0
    
    def focus_input(self):
        """Focus on text input"""
        self._hide_placeholder()
        self.textbox.focus_set()
    
    def bind_focus_in(self):
        """Setup focus bindings"""
        def on_focus_in(event):
            if self._has_placeholder:
                self._hide_placeholder()
        
        def on_focus_out(event):
            if self.is_empty():
                self._show_placeholder()
        
        self.textbox.bind("<FocusIn>", on_focus_in)
        self.textbox.bind("<FocusOut>", on_focus_out)
