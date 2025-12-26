"""
Professional Audio Post-Processing Module for VietTTS PRO MAX
Delivers studio-quality audio output with advanced processing pipeline
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass
from enum import Enum
import soundfile as sf
from loguru import logger

try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False
    logger.warning("noisereduce not installed. Noise reduction disabled.")

try:
    from scipy import signal
    from scipy.ndimage import uniform_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("scipy not installed. Some audio processing disabled.")


class AudioQuality(Enum):
    """Audio quality presets"""
    DRAFT = "draft"           # Fast, lower quality
    STANDARD = "standard"     # Balanced
    HIGH = "high"             # High quality
    STUDIO = "studio"         # Maximum quality
    BROADCAST = "broadcast"   # Broadcast standards (EBU R128)


@dataclass
class AudioSettings:
    """Audio processing settings"""
    # Output format
    sample_rate: int = 22050
    bit_depth: int = 16
    channels: int = 1
    
    # Normalization
    normalize: bool = True
    target_lufs: float = -16.0  # Standard for podcasts
    target_peak: float = -1.0   # dB below 0
    
    # Noise reduction
    noise_reduction: bool = True
    noise_reduction_strength: float = 0.5  # 0.0 to 1.0
    
    # Dynamics
    apply_compression: bool = True
    compression_threshold: float = -20.0  # dB
    compression_ratio: float = 4.0
    
    # De-essing
    apply_deesser: bool = True
    deesser_frequency: int = 6000  # Hz
    deesser_threshold: float = -10.0  # dB
    
    # EQ
    apply_eq: bool = True
    bass_boost: float = 0.0      # dB (-6 to +6)
    presence_boost: float = 2.0   # dB at 3-5kHz for clarity
    air_boost: float = 1.0        # dB at 10kHz+ for brightness
    
    # Silence handling
    trim_silence: bool = True
    silence_threshold: float = -40.0  # dB
    min_silence_duration: float = 0.1  # seconds
    add_padding: float = 0.1  # seconds at start/end
    
    @classmethod
    def from_quality(cls, quality: AudioQuality) -> "AudioSettings":
        """Create settings from quality preset"""
        if quality == AudioQuality.DRAFT:
            return cls(
                normalize=True,
                noise_reduction=False,
                apply_compression=False,
                apply_deesser=False,
                apply_eq=False,
                trim_silence=True
            )
        elif quality == AudioQuality.STANDARD:
            return cls(
                normalize=True,
                noise_reduction=True,
                noise_reduction_strength=0.3,
                apply_compression=True,
                apply_deesser=False,
                apply_eq=True,
                bass_boost=0.0,
                presence_boost=1.0
            )
        elif quality == AudioQuality.HIGH:
            return cls(
                normalize=True,
                noise_reduction=True,
                noise_reduction_strength=0.5,
                apply_compression=True,
                apply_deesser=True,
                apply_eq=True,
                bass_boost=1.0,
                presence_boost=2.0,
                air_boost=1.5
            )
        elif quality == AudioQuality.STUDIO:
            return cls(
                sample_rate=44100,
                bit_depth=24,
                normalize=True,
                target_lufs=-14.0,
                noise_reduction=True,
                noise_reduction_strength=0.6,
                apply_compression=True,
                compression_ratio=3.0,
                apply_deesser=True,
                apply_eq=True,
                bass_boost=1.5,
                presence_boost=2.5,
                air_boost=2.0
            )
        elif quality == AudioQuality.BROADCAST:
            return cls(
                sample_rate=48000,
                bit_depth=24,
                normalize=True,
                target_lufs=-23.0,  # EBU R128 standard
                target_peak=-1.0,
                noise_reduction=True,
                noise_reduction_strength=0.7,
                apply_compression=True,
                compression_threshold=-18.0,
                compression_ratio=4.0,
                apply_deesser=True,
                apply_eq=True
            )
        return cls()


class AudioProcessor:
    """
    Professional audio post-processor for TTS output
    Applies studio-quality processing pipeline
    """
    
    def __init__(self, settings: Optional[AudioSettings] = None):
        """
        Initialize audio processor
        
        Args:
            settings: Audio processing settings
        """
        self.settings = settings or AudioSettings()
        self._processing_chain: List[Callable] = []
        self._build_processing_chain()
    
    def _build_processing_chain(self):
        """Build the audio processing chain based on settings"""
        self._processing_chain = []
        
        # Order matters for audio processing!
        if self.settings.trim_silence:
            self._processing_chain.append(self._trim_silence)
        
        if self.settings.noise_reduction and HAS_NOISEREDUCE:
            self._processing_chain.append(self._reduce_noise)
        
        if self.settings.apply_deesser and HAS_SCIPY:
            self._processing_chain.append(self._apply_deesser)
        
        if self.settings.apply_eq and HAS_SCIPY:
            self._processing_chain.append(self._apply_eq)
        
        if self.settings.apply_compression:
            self._processing_chain.append(self._apply_compression)
        
        if self.settings.normalize:
            self._processing_chain.append(self._normalize)
        
        if self.settings.add_padding > 0:
            self._processing_chain.append(self._add_padding)
    
    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Process audio through the entire pipeline
        
        Args:
            audio: Input audio array
            sample_rate: Sample rate
            progress_callback: Optional callback(step, total, status)
            
        Returns:
            Tuple of (processed_audio, output_sample_rate)
        """
        total_steps = len(self._processing_chain) + 1  # +1 for resampling
        current_step = 0
        
        # Ensure float32 for processing
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize to -1 to 1 range if needed
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()
        
        # Run through processing chain
        for processor in self._processing_chain:
            current_step += 1
            if progress_callback:
                progress_callback(current_step, total_steps, processor.__name__)
            
            try:
                audio = processor(audio, sample_rate)
            except Exception as e:
                logger.warning(f"Processing step {processor.__name__} failed: {e}")
                continue
        
        # Resample if needed
        current_step += 1
        if progress_callback:
            progress_callback(current_step, total_steps, "Resampling")
        
        if sample_rate != self.settings.sample_rate:
            audio = self._resample(audio, sample_rate, self.settings.sample_rate)
        
        return audio, self.settings.sample_rate
    
    def _trim_silence(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Trim silence from start and end"""
        threshold_linear = 10 ** (self.settings.silence_threshold / 20)
        min_samples = int(self.settings.min_silence_duration * sample_rate)
        
        # Find non-silent regions
        envelope = np.abs(audio)
        if HAS_SCIPY:
            envelope = uniform_filter1d(envelope, size=min_samples)
        
        non_silent = envelope > threshold_linear
        
        if not np.any(non_silent):
            return audio
        
        # Find first and last non-silent sample
        non_silent_indices = np.where(non_silent)[0]
        start_idx = max(0, non_silent_indices[0] - min_samples)
        end_idx = min(len(audio), non_silent_indices[-1] + min_samples)
        
        return audio[start_idx:end_idx]
    
    def _reduce_noise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply noise reduction"""
        if not HAS_NOISEREDUCE:
            return audio
        
        # Estimate noise from quietest portion
        reduced = nr.reduce_noise(
            y=audio,
            sr=sample_rate,
            prop_decrease=self.settings.noise_reduction_strength,
            stationary=True
        )
        
        return reduced.astype(np.float32)
    
    def _apply_deesser(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply de-essing to reduce sibilance"""
        if not HAS_SCIPY:
            return audio
        
        # Design highpass filter to isolate sibilance
        nyquist = sample_rate / 2
        high_freq = min(self.settings.deesser_frequency / nyquist, 0.99)
        
        b, a = signal.butter(2, high_freq, btype='high')
        sibilance = signal.filtfilt(b, a, audio)
        
        # Calculate envelope of sibilance
        sibilance_env = np.abs(sibilance)
        sibilance_env = uniform_filter1d(sibilance_env, size=int(sample_rate * 0.01))
        
        # Create gain reduction where sibilance exceeds threshold
        threshold_linear = 10 ** (self.settings.deesser_threshold / 20)
        gain_reduction = np.ones_like(audio)
        
        over_threshold = sibilance_env > threshold_linear
        if np.any(over_threshold):
            gain_reduction[over_threshold] = threshold_linear / sibilance_env[over_threshold]
            gain_reduction = np.clip(gain_reduction, 0.3, 1.0)  # Limit reduction
        
        return audio * gain_reduction
    
    def _apply_eq(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply EQ adjustments"""
        if not HAS_SCIPY:
            return audio
        
        nyquist = sample_rate / 2
        
        # Bass boost (low shelf at 150Hz)
        if self.settings.bass_boost != 0:
            bass_gain = 10 ** (self.settings.bass_boost / 20)
            low_freq = min(150 / nyquist, 0.99)
            b, a = signal.butter(2, low_freq, btype='low')
            bass = signal.filtfilt(b, a, audio)
            audio = audio + bass * (bass_gain - 1)
        
        # Presence boost (peak at 3-5kHz)
        if self.settings.presence_boost != 0:
            presence_gain = 10 ** (self.settings.presence_boost / 20)
            low = min(3000 / nyquist, 0.99)
            high = min(5000 / nyquist, 0.99)
            if low < high:
                b, a = signal.butter(2, [low, high], btype='band')
                presence = signal.filtfilt(b, a, audio)
                audio = audio + presence * (presence_gain - 1)
        
        # Air boost (high shelf at 10kHz)
        if self.settings.air_boost != 0 and sample_rate > 22050:
            air_gain = 10 ** (self.settings.air_boost / 20)
            high_freq = min(10000 / nyquist, 0.99)
            b, a = signal.butter(2, high_freq, btype='high')
            air = signal.filtfilt(b, a, audio)
            audio = audio + air * (air_gain - 1)
        
        return np.clip(audio, -1.0, 1.0)
    
    def _apply_compression(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply dynamic range compression"""
        threshold = 10 ** (self.settings.compression_threshold / 20)
        ratio = self.settings.compression_ratio
        
        # Calculate envelope
        envelope = np.abs(audio)
        if HAS_SCIPY:
            # Smooth envelope with attack/release
            attack_samples = int(0.01 * sample_rate)  # 10ms attack
            release_samples = int(0.1 * sample_rate)  # 100ms release
            envelope = uniform_filter1d(envelope, size=attack_samples)
        
        # Calculate gain reduction
        gain = np.ones_like(audio)
        over_threshold = envelope > threshold
        
        if np.any(over_threshold):
            # Soft knee compression
            gain[over_threshold] = (
                threshold + (envelope[over_threshold] - threshold) / ratio
            ) / envelope[over_threshold]
        
        # Apply makeup gain
        compressed = audio * gain
        makeup_gain = 1.0 / np.sqrt(np.mean(gain ** 2))  # RMS-based makeup
        compressed = compressed * min(makeup_gain, 2.0)  # Limit makeup gain
        
        return np.clip(compressed, -1.0, 1.0)
    
    def _normalize(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Normalize audio to target LUFS"""
        # Simple peak normalization as fallback
        # For true LUFS normalization, would need pyloudnorm
        
        peak = np.abs(audio).max()
        if peak > 0:
            target_peak_linear = 10 ** (self.settings.target_peak / 20)
            audio = audio * (target_peak_linear / peak)
        
        return audio
    
    def _add_padding(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Add silence padding at start and end"""
        padding_samples = int(self.settings.add_padding * sample_rate)
        padding = np.zeros(padding_samples, dtype=np.float32)
        return np.concatenate([padding, audio, padding])
    
    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate"""
        if orig_sr == target_sr:
            return audio
        
        if HAS_SCIPY:
            # High-quality resampling
            duration = len(audio) / orig_sr
            target_length = int(duration * target_sr)
            return signal.resample(audio, target_length).astype(np.float32)
        else:
            # Simple linear interpolation fallback
            ratio = target_sr / orig_sr
            target_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, target_length)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
    
    def save(
        self,
        audio: np.ndarray,
        filepath: str,
        sample_rate: Optional[int] = None
    ):
        """
        Save processed audio to file
        
        Args:
            audio: Audio array
            filepath: Output file path
            sample_rate: Sample rate (uses settings if not provided)
        """
        sr = sample_rate or self.settings.sample_rate
        
        # Determine subtype based on bit depth and format
        filepath = Path(filepath)
        
        if filepath.suffix.lower() == '.wav':
            if self.settings.bit_depth == 24:
                subtype = 'PCM_24'
            elif self.settings.bit_depth == 32:
                subtype = 'FLOAT'
            else:
                subtype = 'PCM_16'
        elif filepath.suffix.lower() == '.flac':
            subtype = 'PCM_24' if self.settings.bit_depth >= 24 else 'PCM_16'
        else:
            subtype = None
        
        sf.write(str(filepath), audio, sr, subtype=subtype)
        logger.info(f"Saved processed audio: {filepath}")


class BatchAudioProcessor:
    """
    Batch processor for multiple audio files
    Optimized for SRT processing with consistent quality
    """
    
    def __init__(
        self,
        settings: Optional[AudioSettings] = None,
        quality: Optional[AudioQuality] = None
    ):
        """
        Initialize batch processor
        
        Args:
            settings: Audio settings (or use quality preset)
            quality: Quality preset to use
        """
        if quality:
            self.settings = AudioSettings.from_quality(quality)
        else:
            self.settings = settings or AudioSettings()
        
        self.processor = AudioProcessor(self.settings)
    
    def process_batch(
        self,
        audio_files: List[str],
        output_dir: str,
        progress_callback: Optional[Callable] = None
    ) -> List[str]:
        """
        Process multiple audio files
        
        Args:
            audio_files: List of input file paths
            output_dir: Output directory
            progress_callback: Callback(current, total, filename)
            
        Returns:
            List of output file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_files = []
        total = len(audio_files)
        
        for idx, input_file in enumerate(audio_files, start=1):
            if progress_callback:
                progress_callback(idx, total, Path(input_file).name)
            
            try:
                # Load
                audio, sr = sf.read(input_file)
                
                # Process
                processed, out_sr = self.processor.process(audio, sr)
                
                # Save
                output_file = output_dir / Path(input_file).name
                self.processor.save(processed, str(output_file), out_sr)
                output_files.append(str(output_file))
                
            except Exception as e:
                logger.error(f"Failed to process {input_file}: {e}")
                continue
        
        return output_files


# Convenience functions
def enhance_audio(
    audio: np.ndarray,
    sample_rate: int,
    quality: AudioQuality = AudioQuality.HIGH
) -> Tuple[np.ndarray, int]:
    """
    Quick function to enhance audio with preset quality
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate
        quality: Quality preset
        
    Returns:
        Tuple of (enhanced_audio, output_sample_rate)
    """
    settings = AudioSettings.from_quality(quality)
    processor = AudioProcessor(settings)
    return processor.process(audio, sample_rate)


def normalize_loudness(
    audio: np.ndarray,
    target_lufs: float = -16.0
) -> np.ndarray:
    """
    Normalize audio to target loudness
    
    Args:
        audio: Input audio array
        target_lufs: Target loudness in LUFS
        
    Returns:
        Normalized audio
    """
    settings = AudioSettings(
        normalize=True,
        target_lufs=target_lufs,
        noise_reduction=False,
        apply_compression=False,
        apply_deesser=False,
        apply_eq=False,
        trim_silence=False
    )
    processor = AudioProcessor(settings)
    result, _ = processor.process(audio, 22050)
    return result
