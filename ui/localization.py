"""
Localization support for Vietnamese TTS PRO.

Provides multi-language UI text with support for:
- Vietnamese (vi) - Default
- English (en)

Usage:
    from ui.localization import get_text, set_language
    
    set_language("en")
    text = get_text("generate_button")  # Returns "Generate Speech"
"""

from typing import Dict, Optional
from loguru import logger


# Supported languages
SUPPORTED_LANGUAGES = {
    "vi": "Tiếng Việt",
    "en": "English",
}

# Default language
_current_language = "vi"

# Translations dictionary
_translations: Dict[str, Dict[str, str]] = {
    # ============================================
    # Main Window
    # ============================================
    "app_title": {
        "vi": "Vietnamese TTS PRO - VieNeu-TTS",
        "en": "Vietnamese TTS PRO - VieNeu-TTS",
    },
    "tagline": {
        "vi": "Chuyển văn bản thành giọng nói",
        "en": "Vietnamese Text-to-Speech",
    },
    
    # ============================================
    # Tabs
    # ============================================
    "tab_tts": {
        "vi": "💬 Văn bản",
        "en": "💬 Text-to-Speech",
    },
    "tab_srt": {
        "vi": "📄 Nhập SRT",
        "en": "📄 SRT Import",
    },
    
    # ============================================
    # Text Input Panel
    # ============================================
    "text_input_title": {
        "vi": "Nhập văn bản",
        "en": "Enter Text",
    },
    "text_input_placeholder": {
        "vi": "Nhập văn bản tiếng Việt tại đây...",
        "en": "Enter Vietnamese text here...",
    },
    "char_count": {
        "vi": "Ký tự: {count}",
        "en": "Characters: {count}",
    },
    
    # ============================================
    # Voice Selector
    # ============================================
    "voice_selector_title": {
        "vi": "Chọn giọng đọc",
        "en": "Select Voice",
    },
    "voice_mode_standard": {
        "vi": "Chuẩn",
        "en": "Standard",
    },
    "voice_mode_clone": {
        "vi": "Nhân bản",
        "en": "Clone",
    },
    "no_voices_available": {
        "vi": "Không có giọng đọc",
        "en": "No voices available",
    },
    
    # ============================================
    # Settings Panel
    # ============================================
    "settings_title": {
        "vi": "Cài đặt",
        "en": "Settings",
    },
    "speed_label": {
        "vi": "Tốc độ:",
        "en": "Speed:",
    },
    "output_format_label": {
        "vi": "Định dạng:",
        "en": "Format:",
    },
    "output_dir_label": {
        "vi": "Thư mục xuất:",
        "en": "Output folder:",
    },
    
    # ============================================
    # Buttons
    # ============================================
    "btn_generate": {
        "vi": "🎙️ Tạo giọng nói",
        "en": "🎙️ Generate Speech",
    },
    "btn_generating": {
        "vi": "⏳ Đang tạo...",
        "en": "⏳ Generating...",
    },
    "btn_play": {
        "vi": "▶ Phát",
        "en": "▶ Play",
    },
    "btn_pause": {
        "vi": "⏸ Tạm dừng",
        "en": "⏸ Pause",
    },
    "btn_stop": {
        "vi": "⏹ Dừng",
        "en": "⏹ Stop",
    },
    "btn_save": {
        "vi": "💾 Lưu file",
        "en": "💾 Save File",
    },
    "btn_copy_path": {
        "vi": "📋 Sao chép đường dẫn",
        "en": "📋 Copy Path",
    },
    "btn_open_folder": {
        "vi": "📂 Mở thư mục",
        "en": "📂 Open Folder",
    },
    "btn_browse": {
        "vi": "Duyệt...",
        "en": "Browse...",
    },
    "btn_process": {
        "vi": "Xử lý",
        "en": "Process",
    },
    "btn_cancel": {
        "vi": "Hủy",
        "en": "Cancel",
    },
    
    # ============================================
    # Audio Player
    # ============================================
    "audio_player_title": {
        "vi": "🔊 Trình phát âm thanh",
        "en": "🔊 Audio Player",
    },
    "no_audio_placeholder": {
        "vi": "Chưa có audio. Hãy tạo giọng nói để nghe kết quả.",
        "en": "No audio yet. Generate speech to hear the result.",
    },
    
    # ============================================
    # SRT Panel
    # ============================================
    "srt_title": {
        "vi": "Xử lý file phụ đề",
        "en": "Subtitle File Processing",
    },
    "srt_open_file": {
        "vi": "Mở file SRT",
        "en": "Open SRT File",
    },
    "srt_range_label": {
        "vi": "Phạm vi xử lý:",
        "en": "Processing range:",
    },
    "srt_from": {
        "vi": "Từ:",
        "en": "From:",
    },
    "srt_to": {
        "vi": "Đến:",
        "en": "To:",
    },
    
    # ============================================
    # Status Messages
    # ============================================
    "status_ready": {
        "vi": "Sẵn sàng",
        "en": "Ready",
    },
    "status_processing": {
        "vi": "Đang xử lý...",
        "en": "Processing...",
    },
    "status_success": {
        "vi": "Hoàn thành!",
        "en": "Complete!",
    },
    "status_error": {
        "vi": "Lỗi",
        "en": "Error",
    },
    "status_loading_model": {
        "vi": "Đang tải mô hình...",
        "en": "Loading model...",
    },
    "status_model_loaded": {
        "vi": "Mô hình đã tải!",
        "en": "Model loaded!",
    },
    
    # ============================================
    # Errors and Warnings
    # ============================================
    "error_empty_text": {
        "vi": "Vui lòng nhập văn bản để tạo giọng nói.",
        "en": "Please enter text to generate speech.",
    },
    "error_no_voice": {
        "vi": "Vui lòng chọn một giọng đọc.",
        "en": "Please select a voice.",
    },
    "error_generation_failed": {
        "vi": "Tạo giọng nói thất bại: {error}",
        "en": "Speech generation failed: {error}",
    },
    "warning_ffmpeg_not_found": {
        "vi": "FFmpeg không tìm thấy",
        "en": "FFmpeg Not Found",
    },
    
    # ============================================
    # Dialogs
    # ============================================
    "dialog_save_title": {
        "vi": "Lưu file âm thanh",
        "en": "Save Audio File",
    },
    "dialog_confirm_quit": {
        "vi": "Bạn có chắc muốn thoát?",
        "en": "Are you sure you want to quit?",
    },
    "dialog_success": {
        "vi": "Thành công",
        "en": "Success",
    },
    "dialog_error": {
        "vi": "Lỗi",
        "en": "Error",
    },
    "dialog_warning": {
        "vi": "Cảnh báo",
        "en": "Warning",
    },
    
    # ============================================
    # Keyboard Shortcuts Help
    # ============================================
    "shortcuts_title": {
        "vi": "Phím tắt",
        "en": "Keyboard Shortcuts",
    },
    "shortcuts_help": {
        "vi": """Phím tắt:

Ctrl+Enter  - Tạo giọng nói
Ctrl+S      - Lưu file âm thanh
Ctrl+P      - Phát/Tạm dừng
Escape      - Dừng phát
F5          - Làm mới danh sách giọng
Ctrl+Q      - Thoát ứng dụng
F1          - Hiện trợ giúp này""",
        "en": """Keyboard Shortcuts:

Ctrl+Enter  - Generate speech from text
Ctrl+S      - Save current audio file
Ctrl+P      - Play/Pause audio
Escape      - Stop audio playback
F5          - Refresh voice list
Ctrl+Q      - Quit application
F1          - Show this help""",
    },
    
    # ============================================
    # Device Info
    # ============================================
    "device_cpu": {
        "vi": "CPU",
        "en": "CPU",
    },
    "device_gpu": {
        "vi": "🚀 GPU (DirectML)",
        "en": "🚀 GPU (DirectML)",
    },
}


def get_text(key: str, **kwargs) -> str:
    """
    Get localized text for the given key.
    
    Args:
        key: Translation key
        **kwargs: Format arguments for the string
        
    Returns:
        Localized string, or key if not found
    """
    global _current_language
    
    if key not in _translations:
        logger.warning(f"Translation key not found: {key}")
        return key
    
    text_dict = _translations[key]
    
    # Try current language first
    if _current_language in text_dict:
        text = text_dict[_current_language]
    # Fallback to Vietnamese
    elif "vi" in text_dict:
        text = text_dict["vi"]
    # Fallback to English
    elif "en" in text_dict:
        text = text_dict["en"]
    else:
        return key
    
    # Format with kwargs if provided
    if kwargs:
        try:
            text = text.format(**kwargs)
        except KeyError:
            pass
    
    return text


def set_language(lang_code: str) -> bool:
    """
    Set the current UI language.
    
    Args:
        lang_code: Language code ('vi' or 'en')
        
    Returns:
        True if language was set successfully
    """
    global _current_language
    
    if lang_code not in SUPPORTED_LANGUAGES:
        logger.warning(f"Unsupported language: {lang_code}")
        return False
    
    _current_language = lang_code
    logger.info(f"Language set to: {SUPPORTED_LANGUAGES[lang_code]}")
    return True


def get_current_language() -> str:
    """Get current language code."""
    return _current_language


def get_supported_languages() -> Dict[str, str]:
    """Get dict of supported language codes and names."""
    return SUPPORTED_LANGUAGES.copy()


def t(key: str, **kwargs) -> str:
    """Shorthand alias for get_text()."""
    return get_text(key, **kwargs)
