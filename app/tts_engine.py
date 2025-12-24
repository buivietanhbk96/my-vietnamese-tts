"""
TTS Engine wrapper for VietTTS
Handles model loading, inference, and voice cloning
"""

import os
import time
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Generator, Callable
from loguru import logger

from app.config import Config
from app.exceptions import (
    ModelNotFoundError,
    ModelDownloadError,
    VoiceFileError,
    VoiceDurationError,
    InferenceError,
    EmptyTextError
)


class TTSEngine:
    """
    VietTTS Engine wrapper
    Provides interface for TTS synthesis and voice cloning
    """
    
    def __init__(self, progress_callback: Optional[Callable] = None):
        """
        Initialize TTS Engine
        
        Args:
            progress_callback: Optional callback function for progress updates
                              signature: callback(status: str, progress: float)
        """
        self.progress_callback = progress_callback
        self.tts = None
        self.model_loaded = False
        self.voices = {}
        
        # Setup CPU optimization
        Config.setup_torch_threads()
        Config.ensure_directories()
    
    def _update_progress(self, status: str, progress: float = 0):
        """Send progress update through callback"""
        if self.progress_callback:
            self.progress_callback(status, progress)
    
    def load_model(self) -> bool:
        """
        Load TTS model (downloads if not present)
        
        Returns:
            bool: True if model loaded successfully
        """
        try:
            self._update_progress("Checking model files...", 0.1)
            
            # Check if model exists
            required_files = [
                "config.yaml",
                "speech_embedding.onnx",
                "speech_tokenizer.onnx",
                "llm.pt",
                "flow.pt",
                "hift.pt"
            ]
            
            model_dir = Config.MODEL_DIR
            files_exist = model_dir.exists() and all(
                (model_dir / f).exists() for f in required_files
            )
            
            if not files_exist:
                self._update_progress("Downloading model from HuggingFace...", 0.2)
                logger.info("Model not found, downloading from HuggingFace...")
                self._download_model()
            
            self._update_progress("Loading TTS model...", 0.5)
            logger.info("Loading TTS model...")
            
            # Import viettts here to avoid loading at startup
            from viettts.tts import TTS
            
            self.tts = TTS(
                model_dir=str(model_dir),
                load_jit=Config.LOAD_JIT,
                load_onnx=Config.LOAD_ONNX
            )
            
            self.model_loaded = True
            self._update_progress("Model loaded successfully!", 1.0)
            logger.success("TTS model loaded successfully")
            
            # Load available voices
            self._load_voices()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_loaded = False
            raise ModelDownloadError(f"Failed to load model: {str(e)}")
    
    def _download_model(self):
        """Download model from HuggingFace"""
        try:
            from huggingface_hub import snapshot_download
            
            snapshot_download(
                repo_id="dangvansam/viet-tts",
                local_dir=str(Config.MODEL_DIR)
            )
            logger.success("Model downloaded successfully")
            
        except Exception as e:
            logger.error(f"Model download failed: {e}")
            raise ModelDownloadError(f"Download failed: {str(e)}")
    
    def _load_voices(self):
        """Load available voice samples"""
        samples_dir = Config.SAMPLES_DIR
        
        if samples_dir.exists():
            for ext in ['*.wav', '*.mp3']:
                for f in samples_dir.glob(ext):
                    voice_name = f.stem
                    self.voices[voice_name] = str(f)
        
        logger.info(f"Loaded {len(self.voices)} voice samples")
    
    def get_available_voices(self) -> dict:
        """
        Get list of available voices
        
        Returns:
            dict: {voice_name: voice_path}
        """
        return self.voices.copy()
    
    def reload_voices(self):
        """Reload voice samples from directory"""
        self.voices.clear()
        self._load_voices()
    
    def load_voice_from_file(self, filepath: str) -> torch.Tensor:
        """
        Load and process voice sample from file for cloning
        
        Args:
            filepath: Path to audio file (mp3/wav)
            
        Returns:
            torch.Tensor: Processed voice tensor (16kHz)
        """
        from viettts.utils.file_utils import load_prompt_speech_from_file
        
        if not os.path.exists(filepath):
            raise VoiceFileError(f"Voice file not found: {filepath}")
        
        # Check file extension
        ext = Path(filepath).suffix.lower()
        if ext not in ['.wav', '.mp3', '.mp4', '.m4a']:
            raise VoiceFileError(f"Unsupported audio format: {ext}")
        
        try:
            voice_tensor = load_prompt_speech_from_file(
                filepath=filepath,
                min_duration=Config.MIN_VOICE_DURATION,
                max_duration=Config.MAX_VOICE_DURATION
            )
            return voice_tensor
            
        except Exception as e:
            logger.error(f"Failed to load voice file: {e}")
            raise VoiceFileError(f"Failed to process voice file: {str(e)}")
    
    def synthesize(
        self,
        text: str,
        voice_path: str,
        speed: float = 1.0,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Synthesize speech from text
        
        Args:
            text: Vietnamese text to synthesize
            voice_path: Path to voice sample file
            speed: Speech speed (0.5-2.0)
            output_path: Optional path to save audio file
            
        Returns:
            np.ndarray: Audio waveform
        """
        if not self.model_loaded:
            raise ModelNotFoundError("Model not loaded. Call load_model() first.")
        
        if not text or not text.strip():
            raise EmptyTextError()
        
        # Validate speed
        speed = max(Config.MIN_SPEED, min(Config.MAX_SPEED, speed))
        
        try:
            # Load voice
            self._update_progress("Loading voice sample...", 0.1)
            prompt_speech_16k = self.load_voice_from_file(voice_path)
            
            # Synthesize
            self._update_progress("Generating speech...", 0.3)
            start_time = time.perf_counter()
            
            wav = self.tts.tts_to_wav(
                text=text,
                prompt_speech_16k=prompt_speech_16k,
                speed=speed
            )
            
            elapsed = time.perf_counter() - start_time
            logger.info(f"Synthesis completed in {elapsed:.2f}s")
            
            # Save to file if output path provided
            if output_path:
                self._update_progress("Saving audio file...", 0.9)
                self._save_audio(wav, output_path)
            
            self._update_progress("Done!", 1.0)
            return wav
            
        except EmptyTextError:
            raise
        except VoiceFileError:
            raise
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise InferenceError(f"Synthesis failed: {str(e)}")
    
    def _save_audio(
        self,
        wav: np.ndarray,
        output_path: str,
        sample_rate: int = None
    ):
        """Save audio array to file"""
        import soundfile as sf
        
        if sample_rate is None:
            sample_rate = Config.OUTPUT_SAMPLE_RATE
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        sf.write(output_path, wav, sample_rate)
        logger.info(f"Audio saved to: {output_path}")
    
    def synthesize_batch(
        self,
        texts: list,
        voice_path: str,
        output_dir: str,
        speed: float = 1.0,
        progress_callback: Optional[Callable] = None,
        cancel_flag: Optional[Callable] = None
    ) -> list:
        """
        Synthesize multiple texts (for SRT processing)
        
        Args:
            texts: List of (index, text) tuples to synthesize
            voice_path: Path to voice sample
            output_dir: Directory to save output files
            speed: Speech speed
            progress_callback: Callback for progress updates (idx, total, status)
            cancel_flag: Callable that returns True to cancel
            
        Returns:
            list: List of output file paths
        """
        if not self.model_loaded:
            raise ModelNotFoundError()
        
        # Load voice once
        prompt_speech_16k = self.load_voice_from_file(voice_path)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_files = []
        total = len(texts)
        
        for current_idx, item in enumerate(texts, start=1):
            # Handle both (index, text) tuples and plain text strings
            if isinstance(item, tuple):
                file_idx, text = item
            else:
                file_idx, text = current_idx, item
            
            # Check for cancellation
            if cancel_flag and cancel_flag():
                logger.info("Batch synthesis cancelled")
                break
            
            # Update progress
            if progress_callback:
                progress_callback(current_idx, total, f"Processing {current_idx}/{total}...")
            
            try:
                if text and text.strip():
                    # Synthesize
                    wav = self.tts.tts_to_wav(
                        text=text,
                        prompt_speech_16k=prompt_speech_16k,
                        speed=speed
                    )
                    
                    # Save with subtitle index name
                    output_path = output_dir / f"{file_idx}.{Config.SRT_OUTPUT_FORMAT}"
                    self._save_audio(wav, str(output_path))
                    output_files.append(str(output_path))
                else:
                    logger.warning(f"Skipping empty text at index {file_idx}")
                    
            except Exception as e:
                logger.error(f"Failed to synthesize text {file_idx}: {e}")
                continue
        
        return output_files
    
    def get_audio_duration(self, filepath: str) -> float:
        """Get duration of audio file in seconds"""
        import librosa
        
        try:
            duration = librosa.get_duration(path=filepath)
            return duration
        except Exception as e:
            logger.error(f"Failed to get audio duration: {e}")
            return 0.0
    
    def cleanup(self):
        """Cleanup resources"""
        self.tts = None
        self.model_loaded = False
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("TTS Engine cleaned up")
