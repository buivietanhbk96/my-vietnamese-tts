"""
Audio utilities for VietTTS Desktop Application
Handles audio playback, conversion, and processing
"""

import os
import io
import wave
import threading
import tempfile
from pathlib import Path
from typing import Optional, Callable
import numpy as np
from loguru import logger

# Initialize pygame mixer with error handling
import pygame

_mixer_initialized = False

def _init_mixer():
    """Initialize pygame mixer safely"""
    global _mixer_initialized
    if not _mixer_initialized:
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=1)
            _mixer_initialized = True
            logger.debug("Pygame mixer initialized")
        except pygame.error as e:
            logger.error(f"Failed to initialize pygame mixer: {e}")
            raise


class AudioPlayer:
    """
    Audio player for playback and control
    """
    
    def __init__(self):
        _init_mixer()  # Ensure mixer is initialized
        self.current_file = None
        self.is_playing = False
        self.is_paused = False
        self._lock = threading.Lock()
        self._position_callback = None
        self._completion_callback = None
        self._position_thread = None
        self._stop_position_tracking = False
        self._temp_files = []  # Track temp files for cleanup
    
    def load(self, filepath: str) -> bool:
        """
        Load audio file for playback
        
        Args:
            filepath: Path to audio file
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            with self._lock:
                self.stop()
                pygame.mixer.music.load(filepath)
                self.current_file = filepath
                logger.info(f"Audio loaded: {filepath}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            return False
    
    def load_from_array(self, audio_array: np.ndarray, sample_rate: int = 22050) -> bool:
        """
        Load audio from numpy array
        
        Args:
            audio_array: Audio waveform as numpy array
            sample_rate: Sample rate
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            # Convert to int16
            if audio_array.dtype != np.int16:
                audio_array = (audio_array * 32767).astype(np.int16)
            
            # Create temporary wav file
            temp_file = tempfile.NamedTemporaryFile(
                suffix='.wav',
                delete=False
            )
            self._temp_files.append(temp_file.name)  # Track for cleanup
            
            with wave.open(temp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_array.tobytes())
            
            return self.load(temp_file.name)
            
        except Exception as e:
            logger.error(f"Failed to load audio from array: {e}")
            return False
    
    def play(self):
        """Start or resume playback"""
        with self._lock:
            if self.is_paused:
                pygame.mixer.music.unpause()
                self.is_paused = False
            else:
                pygame.mixer.music.play()
            
            self.is_playing = True
            self._start_position_tracking()
            logger.info("Playback started")
    
    def pause(self):
        """Pause playback"""
        with self._lock:
            if self.is_playing and not self.is_paused:
                pygame.mixer.music.pause()
                self.is_paused = True
                logger.info("Playback paused")
    
    def stop(self):
        """Stop playback"""
        with self._lock:
            pygame.mixer.music.stop()
            self.is_playing = False
            self.is_paused = False
            self._stop_position_tracking = True
            logger.info("Playback stopped")
    
    def set_volume(self, volume: float):
        """
        Set playback volume
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        volume = max(0.0, min(1.0, volume))
        pygame.mixer.music.set_volume(volume)
    
    def get_volume(self) -> float:
        """Get current volume level"""
        return pygame.mixer.music.get_volume()
    
    def seek(self, position: float):
        """
        Seek to position in seconds
        
        Args:
            position: Position in seconds
        """
        try:
            pygame.mixer.music.set_pos(position)
        except Exception as e:
            logger.error(f"Seek failed: {e}")
    
    def get_position(self) -> float:
        """Get current playback position in seconds"""
        return pygame.mixer.music.get_pos() / 1000.0
    
    def get_duration(self) -> float:
        """Get total duration of loaded audio"""
        if self.current_file:
            try:
                import soundfile as sf
                info = sf.info(self.current_file)
                return info.duration
            except Exception as e:
                logger.debug(f"Failed to get duration with soundfile: {e}")
                try:
                    import librosa
                    return librosa.get_duration(path=self.current_file)
                except:
                    pass
        return 0.0
    
    def is_busy(self) -> bool:
        """Check if audio is currently playing"""
        return pygame.mixer.music.get_busy()
    
    def set_position_callback(self, callback: Callable):
        """Set callback for position updates"""
        self._position_callback = callback
    
    def set_completion_callback(self, callback: Callable):
        """Set callback for playback completion"""
        self._completion_callback = callback
    
    def _start_position_tracking(self):
        """Start thread to track playback position"""
        self._stop_position_tracking = False
        
        def track_position():
            while not self._stop_position_tracking and self.is_playing:
                if self._position_callback and self.is_busy():
                    pos = self.get_position()
                    self._position_callback(pos)
                
                # Check if playback finished
                if not self.is_busy() and self.is_playing and not self.is_paused:
                    self.is_playing = False
                    if self._completion_callback:
                        self._completion_callback()
                    break
                
                pygame.time.wait(100)  # Update every 100ms
        
        self._position_thread = threading.Thread(target=track_position, daemon=True)
        self._position_thread.start()
    
    def _cleanup_temp_files(self):
        """Remove temporary audio files"""
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    logger.debug(f"Removed temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to remove temp file {temp_file}: {e}")
        self._temp_files.clear()
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop()
        self._cleanup_temp_files()
        pygame.mixer.quit()


def convert_audio_format(
    input_path: str,
    output_path: str,
    output_format: str = "wav"
) -> bool:
    """
    Convert audio file to different format using FFmpeg
    
    Args:
        input_path: Input file path
        output_path: Output file path
        output_format: Output format (wav, mp3)
        
    Returns:
        bool: True if conversion successful
    """
    import subprocess
    
    try:
        cmd = [
            "ffmpeg", "-y",
            "-loglevel", "error",
            "-i", input_path,
            "-ar", "22050",
            "-ac", "1"
        ]
        
        if output_format == "mp3":
            cmd.extend(["-c:a", "libmp3lame", "-b:a", "128k"])
        else:
            cmd.extend(["-c:a", "pcm_s16le"])
        
        cmd.append(output_path)
        
        result = subprocess.run(cmd, capture_output=True)
        
        if result.returncode == 0:
            logger.info(f"Audio converted: {output_path}")
            return True
        else:
            logger.error(f"FFmpeg error: {result.stderr.decode()}")
            return False
            
    except Exception as e:
        logger.error(f"Audio conversion failed: {e}")
        return False


def get_audio_info(filepath: str) -> dict:
    """
    Get audio file information
    
    Args:
        filepath: Path to audio file
        
    Returns:
        dict: Audio information (duration, sample_rate, channels)
    """
    try:
        import librosa
        import soundfile as sf
        
        info = sf.info(filepath)
        duration = librosa.get_duration(path=filepath)
        
        return {
            "duration": duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "format": info.format,
            "subtype": info.subtype
        }
        
    except Exception as e:
        logger.error(f"Failed to get audio info: {e}")
        return {}


def normalize_audio(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """
    Normalize audio to target dB level
    
    Args:
        audio: Audio waveform
        target_db: Target dB level
        
    Returns:
        np.ndarray: Normalized audio
    """
    try:
        # Calculate current RMS
        rms = np.sqrt(np.mean(audio ** 2))
        
        if rms > 0:
            # Calculate target RMS
            target_rms = 10 ** (target_db / 20)
            
            # Scale audio
            audio = audio * (target_rms / rms)
            
            # Clip to prevent clipping
            audio = np.clip(audio, -1.0, 1.0)
        
        return audio
        
    except Exception as e:
        logger.error(f"Audio normalization failed: {e}")
        return audio
