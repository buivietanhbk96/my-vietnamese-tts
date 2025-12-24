"""
Configuration settings for VietTTS Desktop Application
Optimized for CPU inference
"""

import os
from pathlib import Path


class Config:
    """Application configuration"""
    
    # ============================================
    # Paths
    # ============================================
    APP_DIR = Path(__file__).parent.parent
    MODEL_DIR = APP_DIR / "pretrained-models"
    SAMPLES_DIR = APP_DIR / "samples"
    OUTPUT_DIR = APP_DIR / "output"
    CONFIG_FILE = APP_DIR / "settings.json"
    
    # ============================================
    # TTS Engine Settings (CPU Optimized)
    # ============================================
    DEVICE = "cpu"
    LOAD_JIT = False      # JIT slower on CPU
    LOAD_ONNX = True      # ONNX faster on CPU
    STREAM = False        # Non-stream for better quality
    
    # Threading
    TORCH_NUM_THREADS = 4  # Adjust based on CPU cores
    
    # ============================================
    # Voice Cloning Settings
    # ============================================
    MIN_VOICE_DURATION = 3.0   # seconds
    MAX_VOICE_DURATION = 5.0   # seconds
    VOICE_SAMPLE_RATE = 16000  # 16kHz for voice input
    
    # ============================================
    # Audio Output Settings
    # ============================================
    OUTPUT_SAMPLE_RATE = 22050
    DEFAULT_SPEED = 1.0
    MIN_SPEED = 0.5
    MAX_SPEED = 2.0
    DEFAULT_FORMAT = "wav"
    SUPPORTED_FORMATS = ["wav", "mp3"]
    
    # ============================================
    # SRT Processing Settings
    # ============================================
    MAX_SRT_SUBTITLES = 10000  # Maximum subtitles to process
    SRT_OUTPUT_FORMAT = "wav"
    
    # ============================================
    # UI Settings
    # ============================================
    WINDOW_TITLE = "VietTTS - Vietnamese Text to Speech"
    WINDOW_MIN_WIDTH = 800
    WINDOW_MIN_HEIGHT = 600
    DEFAULT_WINDOW_SIZE = "900x700"
    
    # ============================================
    # Voice Samples Download
    # ============================================
    VOICE_SAMPLES_REPO = "https://github.com/dangvansam/viet-tts"
    VOICE_SAMPLES_BRANCH = "main"
    VOICE_SAMPLES_PATH = "samples"
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cls.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def setup_torch_threads(cls):
        """Configure PyTorch for CPU optimization"""
        import torch
        torch.set_num_threads(cls.TORCH_NUM_THREADS)
        # Disable CUDA if available (force CPU)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


class ThemeColors:
    """Dark mode color palette - inspired by ui-ux-pro-max-skill"""
    
    # Background colors
    BG_PRIMARY = "#0F172A"      # Slate 900
    BG_SECONDARY = "#1E293B"    # Slate 800
    BG_TERTIARY = "#334155"     # Slate 700
    
    # Accent colors
    PRIMARY = "#6366F1"         # Indigo 500
    PRIMARY_HOVER = "#4F46E5"   # Indigo 600
    SECONDARY = "#8B5CF6"       # Violet 500
    
    # Text colors
    TEXT_PRIMARY = "#F8FAFC"    # Slate 50
    TEXT_SECONDARY = "#94A3B8"  # Slate 400
    TEXT_MUTED = "#64748B"      # Slate 500
    
    # Status colors
    SUCCESS = "#22C55E"         # Green 500
    WARNING = "#F59E0B"         # Amber 500
    ERROR = "#EF4444"           # Red 500
    INFO = "#3B82F6"            # Blue 500
    
    # Border colors
    BORDER = "#334155"          # Slate 700
    BORDER_FOCUS = "#6366F1"    # Indigo 500
    
    # Button colors
    BTN_PRIMARY_BG = "#6366F1"
    BTN_PRIMARY_HOVER = "#4F46E5"
    BTN_SECONDARY_BG = "#334155"
    BTN_SECONDARY_HOVER = "#475569"
    BTN_DANGER_BG = "#DC2626"
    BTN_DANGER_HOVER = "#B91C1C"
    BTN_DANGER_HOVER = "#B91C1C"


class Fonts:
    """Font configuration - using system fonts with fallbacks"""
    
    # Windows system fonts
    HEADING = ("Segoe UI", 16, "bold")
    SUBHEADING = ("Segoe UI", 14, "bold")
    BODY = ("Segoe UI", 12)
    BODY_SMALL = ("Segoe UI", 10)
    MONO = ("Consolas", 11)
    BUTTON = ("Segoe UI", 12, "bold")
