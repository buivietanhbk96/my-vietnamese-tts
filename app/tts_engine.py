"""
TTS Engine wrapper for VieNeu-TTS
Handles model loading, inference, and voice cloning
Supports GPU (DirectML) via ONNX Runtime for Codec
"""

import os
import time
import json
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Generator, Callable, Union, Dict, List, Any
from loguru import logger

from app.config import Config
from app.device_manager import get_device_manager
from app.services.vieneu_engine import VieNeuEngine
from app.exceptions import (
    ModelNotFoundError,
    ModelDownloadError,
    VoiceFileError,
    InferenceError,
    EmptyTextError
)


class TTSEngine:
    """
    VieNeu-TTS Engine wrapper.
    
    Provides high-level interface for TTS synthesis and voice cloning
    with GPU acceleration support (DirectML for AMD GPUs).
    """
    
    def __init__(self, progress_callback: Optional[Callable[[str, float], None]] = None) -> None:
        self.progress_callback: Optional[Callable[[str, float], None]] = progress_callback
        self.engine: Optional[VieNeuEngine] = None
        self.model_loaded: bool = False
        self.voices: Dict[str, Dict[str, Any]] = {}  # Name -> {type, path, ref_text, codes}
        self.device_manager = get_device_manager()
        
        # Ensure directories
        Config.ensure_directories()
        
        logger.info(f"TTS Engine initialized. Device manager: {self.device_manager.device_name}")
    
    def _update_progress(
        self, 
        status: str, 
        progress: float = 0, 
        callback: Optional[Callable[[str, float], None]] = None
    ) -> None:
        """Send progress update through callback."""
        target_callback = callback or self.progress_callback
        if target_callback:
            try:
                target_callback(status, progress)
            except Exception:
                pass
    
    def load_model(self) -> bool:
        """
        Load TTS model (VieNeu-TTS).
        
        Returns:
            bool: True if model loaded successfully
            
        Raises:
            ModelNotFoundError: If model files not found
            ModelDownloadError: If model loading fails
        """
        try:
            self._update_progress("Checking model files...", 0.1)
            
            model_dir = Config.MODEL_DIR
            
            if not (model_dir / "model.safetensors").exists():
                logger.error(f"Model file not found at {model_dir}")
                raise ModelNotFoundError(f"Model not found at {model_dir}")

            self._update_progress("Initializing VieNeu-TTS Engine...", 0.3)
            logger.info(f"Loading VieNeu-TTS model from {model_dir}")
            
            self.engine = VieNeuEngine(str(model_dir))
            
            # Initialize (loads transformers backbone and ONNX codec)
            # Default to GPU if available
            success = self.engine.initialize(prefer_gpu=Config.PREFER_GPU)
            
            if not success:
                logger.error("Failed to initialize VieNeuEngine")
                self.model_loaded = False
                return False

            # Log GPU status
            if self.engine.is_using_gpu:
                logger.success("✅ GPU acceleration enabled for audio decoding")
            else:
                logger.info("Running on CPU (GPU acceleration not available)")
            
            self.model_loaded = True
            
            # Load preset voices
            self._load_voices()
            
            self._update_progress("Model loaded successfully!", 1.0)
            logger.success("VieNeu-TTS Engine loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load engine: {e}")
            self.model_loaded = False
            raise ModelDownloadError(f"Failed to load model: {str(e)}")
    
    def _load_voices(self) -> None:
        """Load available voices (presets from samples folder + cloning)."""
        self.voices.clear()
        
        # 1. Load Presets from samples folder
        samples_dir = Config.MODEL_DIR / "samples"
        if samples_dir.exists():
            # VieNeu-TTS samples usually come as .wav + .txt + .pt
            wav_files = list(samples_dir.glob("*.wav"))
            for wav_path in wav_files:
                voice_name = wav_path.stem
                txt_path = wav_path.with_suffix(".txt")
                pt_path = wav_path.with_suffix(".pt")
                
                ref_text = ""
                if txt_path.exists():
                    ref_text = txt_path.read_text(encoding="utf-8").strip()
                
                codes = []
                if pt_path.exists() and self.engine:
                    codes = self.engine.load_preencoded(str(pt_path))
                
                self.voices[voice_name] = {
                    "type": "preset",
                    "path": str(wav_path.absolute()),
                    "ref_text": ref_text,
                    "codes": codes
                }
            logger.info(f"Loaded {len(wav_files)} preset voices")
        
        # 2. Add samples from app samples dir (for cloning)
        app_samples_dir = Config.SAMPLES_DIR
        if app_samples_dir.exists():
            for ext in ['*.wav', '*.mp3']:
                for f in app_samples_dir.glob(ext):
                    voice_name = f"Clone: {f.stem}"
                    if voice_name not in self.voices:
                        self.voices[voice_name] = {
                            "type": "clone",
                            "path": str(f.absolute()),
                            "ref_text": "", # Needs to be provided by user or detected
                            "codes": []
                        }
        
        logger.info(f"Total voices available: {len(self.voices)}")
    
    def get_available_voices(self) -> dict:
        # Return just names for the UI dropdown
        return {name: data["path"] for name, data in self.voices.items()}
    
    def reload_voices(self):
        self._load_voices()

    def synthesize(
        self,
        text: str,
        voice_name: str, 
        speed: float = 1.0,
        output_path: Optional[str] = None,
        ref_text: str = "",
        progress_callback: Optional[Callable] = None
    ) -> np.ndarray:
        """
        Synthesize speech
        
        Args:
            text: Text to synthesize
            voice_name: Voice name or path for lookup
            speed: Speech speed (0.5-2.0)
            output_path: Optional path to save audio
            ref_text: User-provided reference text (for voice cloning)
            progress_callback: Progress callback function
        """
        if not self.model_loaded or not self.engine:
            raise ModelNotFoundError("Model not loaded.")
        
        if not text or not text.strip():
            raise EmptyTextError()
            
        try:
            self._update_progress("Preparing synthesis...", 0.1, progress_callback)
            
            # Find voice data
            # voice_name is the key in self.voices
            voice_data = self.voices.get(voice_name)
            if not voice_data:
                # Try finding by path if name not found (for direct calls)
                for v_name, v_data in self.voices.items():
                    if v_data["path"] == voice_name:
                        voice_data = v_data
                        break
            
            if not voice_data:
                logger.warning(f"Voice '{voice_name}' not found, using first available")
                if self.voices:
                    voice_data = list(self.voices.values())[0]
                else:
                    raise VoiceFileError("No voices available")

            ref_codes = voice_data.get("codes", [])
            ref_path = voice_data.get("path", "")
            
            # If codes are missing, encode them now
            if not ref_codes and ref_path:
                self._update_progress("Encoding reference voice...", 0.2, progress_callback)
                ref_codes = self.engine.encode_reference(ref_path)
                voice_data["codes"] = ref_codes # Cache for next time
            
            # Use user-provided ref_text if available, otherwise use voice_data ref_text
            # For cloned voices, user MUST provide ref_text for good quality
            final_ref_text = ref_text.strip() if ref_text else voice_data.get("ref_text", "")
            
            if not final_ref_text:
                logger.warning(f"Reference text missing for {voice_name}. Voice cloning quality may be poor!")
                final_ref_text = "Xin chào, đây là một đoạn văn bản mẫu."

            self._update_progress("Synthesizing (VieNeu-TTS)...", 0.4, progress_callback)
            
            # Wrap progress callback for internal engine synthesis (maps 0-1.0 to 0.4-0.85)
            def engine_progress_wrapper(status, progress):
                mapped_progress = 0.4 + (progress * 0.45)
                self._update_progress(f"Synthesizing: {status}", mapped_progress, progress_callback)
            
            # Call Engine
            audio = self.engine.synthesize(
                text=text,
                ref_codes=ref_codes,
                ref_text=final_ref_text,
                speed=speed,
                progress_callback=engine_progress_wrapper
            )
            
            # Save if needed
            if output_path:
                self._update_progress("Saving...", 0.9, progress_callback)
                self._save_audio(audio, output_path)
                
            self._update_progress("Done!", 1.0, progress_callback)
            return audio
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise InferenceError(str(e))

    def _save_audio(self, wav: np.ndarray, output_path: str):
        import soundfile as sf
        # VieNeu-TTS outputs 24kHz
        sample_rate = 24000
            
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, wav, sample_rate)

    def synthesize_batch(
        self,
        texts: list,
        voice_name: str,
        output_dir: str,
        speed: float = 1.0,
        progress_callback: Optional[Callable] = None,
        cancel_flag: Optional[Callable] = None
    ) -> list:
        """Batch synthesis for SRT"""
        if not self.model_loaded: raise ModelNotFoundError()
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        files = []
        total = len(texts)
        
        # Pre-load voice data
        voice_data = self.voices.get(voice_name)
        if not voice_data:
            voice_data = list(self.voices.values())[0]
            
        ref_codes = voice_data.get("codes", [])
        ref_text = voice_data.get("ref_text", "")
        if not ref_codes:
            ref_codes = self.engine.encode_reference(voice_data["path"])
        if not ref_text:
            ref_text = "Chào mừng bạn đến với ứng dụng chuyển đổi văn bản thành giọng nói."

        for idx, item in enumerate(texts, 1):
            if cancel_flag and cancel_flag(): break
            
            if isinstance(item, tuple): f_idx, txt = item
            else: f_idx, txt = idx, item
            
            if progress_callback:
                progress_callback(idx, total, f"Processing {idx}/{total}...")
                
            try:
                if not txt.strip(): continue
                
                audio = self.engine.synthesize(
                    text=txt,
                    ref_codes=ref_codes,
                    ref_text=ref_text,
                    speed=speed
                )
                
                out_path = output_dir / f"{f_idx}.wav"
                self._save_audio(audio, str(out_path))
                files.append(str(out_path))
                
            except Exception as e:
                logger.error(f"Error on item {f_idx}: {e}")
                continue
                
        return files

    def get_audio_duration(self, filepath: str) -> float:
        """Get duration of an audio file in seconds."""
        import librosa
        try:
            return librosa.get_duration(path=filepath)
        except Exception:
            return 0.0

    def cleanup(self) -> None:
        """Release all resources and clear GPU memory."""
        if self.engine:
            self.engine.cleanup()
        self.engine = None
        self.model_loaded = False
        self.device_manager.empty_cache()
        logger.info("TTS Engine resources cleaned up")

    def get_device_info(self) -> Dict[str, Any]:
        """Get device/hardware information."""
        return self.device_manager.get_device_info()
    
    def get_gpu_status(self) -> Dict[str, Any]:
        """
        Get detailed GPU acceleration status.
        
        Returns:
            Dict with GPU status information including:
            - codec_using_gpu: Whether codec is GPU accelerated
            - codec_provider: Active ONNX provider
            - directml_available: Whether DirectML is available
            - device_name: Name of the compute device
        """
        if self.engine:
            return self.engine.get_gpu_status()
        return {
            "codec_using_gpu": False,
            "codec_provider": "Not loaded",
            "directml_available": self.device_manager.is_directml_available,
            "device_name": self.device_manager.device_name,
        }
    
    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get memory usage information.
        
        Useful for monitoring system and GPU memory usage,
        especially on AMD RX6600 with limited VRAM.
        """
        if self.engine:
            return self.engine.get_memory_info()
        return {"error": "Engine not loaded"}
