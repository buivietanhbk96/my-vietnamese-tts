"""
Configuration settings for Vietnamese TTS Desktop Application
Using VieNeu-TTS (SOTA Vietnamese TTS) with DirectML (AMD GPU) acceleration
"""

import os
from pathlib import Path


class Config:
    """Application configuration"""
    
    # ============================================
    # Paths
    # ============================================
    APP_DIR = Path(__file__).parent.parent
    MODEL_DIR = APP_DIR / "model_assets" / "vietnamese"
    SAMPLES_DIR = APP_DIR / "samples"
    OUTPUT_DIR = APP_DIR / "output"
    CONFIG_FILE = APP_DIR / "settings.json"
    
    # VieNeu-TTS paths
    VIENEU_DIR = APP_DIR / "app" / "services" / "vieneu"
    
    # Available voice models
    VOICE_MODELS = {
        "VieNeu-TTS": APP_DIR / "model_assets" / "vietnamese",
    }
    DEFAULT_MODEL = "VieNeu-TTS"
    
    # ============================================
    # TTS Engine Settings (DirectML GPU)
    # ============================================
    DEVICE = None  # None = auto-detect
    PREFER_GPU = True  # Use DirectML for AMD GPU
    
    # ONNX Runtime Settings
    ONNX_PROVIDERS = ['DmlExecutionProvider', 'CPUExecutionProvider']
    
    # Threading (for CPU operations)
    TORCH_NUM_THREADS = 4
    
    # ============================================
    # Voice Cloning Settings
    # ============================================
    MIN_VOICE_DURATION = 3.0   # seconds
    MAX_VOICE_DURATION = 30.0  # seconds (Style-Bert-VITS2 supports longer)
    VOICE_SAMPLE_RATE = 16000
    
    # ============================================
    # Audio Output Settings
    # ============================================
    OUTPUT_SAMPLE_RATE = 44100  # Style-Bert-VITS2 default
    DEFAULT_SPEED = 1.0
    MIN_SPEED = 0.5
    MAX_SPEED = 2.0
    DEFAULT_FORMAT = "wav"
    SUPPORTED_FORMATS = ["wav", "mp3"]
    
    # ============================================
    # SRT Processing Settings
    # ============================================
    MAX_SRT_SUBTITLES = 10000
    SRT_OUTPUT_FORMAT = "wav"
    
    # ============================================
    # UI Settings
    # ============================================
    WINDOW_TITLE = "Vietnamese TTS PRO - VieNeu-TTS"
    WINDOW_MIN_WIDTH = 800
    WINDOW_MIN_HEIGHT = 600
    DEFAULT_WINDOW_SIZE = "900x700"
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cls.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def setup_torch_threads(cls):
        """Configure PyTorch for optimal performance"""
        import torch
        torch.set_num_threads(cls.TORCH_NUM_THREADS)
        
        if cls.DEVICE == "cpu" or not cls.PREFER_GPU:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""


class ThemeColors:
    """Dark mode color palette"""
    
    BG_PRIMARY = "#0F172A"
    BG_SECONDARY = "#1E293B"
    BG_TERTIARY = "#334155"
    
    PRIMARY = "#6366F1"
    PRIMARY_HOVER = "#4F46E5"
    SECONDARY = "#8B5CF6"
    
    TEXT_PRIMARY = "#F8FAFC"
    TEXT_SECONDARY = "#94A3B8"
    TEXT_MUTED = "#64748B"
    
    SUCCESS = "#22C55E"
    WARNING = "#F59E0B"
    ERROR = "#EF4444"
    INFO = "#3B82F6"
    
    BORDER = "#334155"
    BORDER_FOCUS = "#6366F1"
    
    BTN_PRIMARY_BG = "#6366F1"
    BTN_PRIMARY_HOVER = "#4F46E5"
    BTN_SECONDARY_BG = "#334155"
    BTN_SECONDARY_HOVER = "#475569"
    BTN_DANGER_BG = "#DC2626"
    BTN_DANGER_HOVER = "#B91C1C"


class Fonts:
    """Font configuration"""
    
    HEADING = ("Segoe UI", 16, "bold")
    SUBHEADING = ("Segoe UI", 14, "bold")
    BODY = ("Segoe UI", 12)
    BODY_SMALL = ("Segoe UI", 10)
    MONO = ("Consolas", 11)
    BUTTON = ("Segoe UI", 12, "bold")
