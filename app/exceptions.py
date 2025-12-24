"""
Custom exceptions for VietTTS Desktop Application
"""


class VietTTSError(Exception):
    """Base exception for all VietTTS errors"""
    pass


class ModelNotFoundError(VietTTSError):
    """Raised when TTS model files are not found"""
    def __init__(self, message="TTS model not found. Please wait for automatic download."):
        self.message = message
        super().__init__(self.message)


class ModelDownloadError(VietTTSError):
    """Raised when model download fails"""
    def __init__(self, message="Failed to download TTS model. Please check your internet connection."):
        self.message = message
        super().__init__(self.message)


class VoiceFileError(VietTTSError):
    """Raised when voice file is invalid"""
    def __init__(self, message="Invalid voice file. Please use a valid audio file (mp3/wav)."):
        self.message = message
        super().__init__(self.message)


class VoiceDurationError(VietTTSError):
    """Raised when voice file duration is out of range"""
    def __init__(self, min_dur=3, max_dur=10, actual_dur=0):
        self.message = f"Voice file duration ({actual_dur:.1f}s) must be between {min_dur}-{max_dur} seconds."
        super().__init__(self.message)


class InferenceError(VietTTSError):
    """Raised when TTS inference fails"""
    def __init__(self, message="TTS inference failed. Please try again."):
        self.message = message
        super().__init__(self.message)


class FFmpegNotFoundError(VietTTSError):
    """Raised when FFmpeg is not installed"""
    def __init__(self, message="FFmpeg not found. Please install FFmpeg and add to PATH."):
        self.message = message
        super().__init__(self.message)


class SRTParseError(VietTTSError):
    """Raised when SRT file parsing fails"""
    def __init__(self, message="Failed to parse SRT file. Please check the file format."):
        self.message = message
        super().__init__(self.message)


class EmptyTextError(VietTTSError):
    """Raised when input text is empty"""
    def __init__(self, message="Text cannot be empty. Please enter some Vietnamese text."):
        self.message = message
        super().__init__(self.message)


class AudioPlaybackError(VietTTSError):
    """Raised when audio playback fails"""
    def __init__(self, message="Failed to play audio. Please check your audio device."):
        self.message = message
        super().__init__(self.message)
