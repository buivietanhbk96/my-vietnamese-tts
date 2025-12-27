"""
VieNeu-TTS Engine for Vietnamese TTS PRO
Uses Qwen2 Backbone (transformers) and NeuCodec (ONNX DirectML)

GPU Acceleration:
- ONNX Runtime DirectML for NeuCodec decoder (AMD GPU support)
- Automatic fallback to CPU if DirectML fails
"""

import gc
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Generator, Tuple, Dict, Any, Callable

import librosa
import numpy as np
import torch
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger

# Add the vieneu subpackage to path so imports work correctly
VIENEU_PATH = Path(__file__).parent / "vieneu"
if str(VIENEU_PATH) not in sys.path:
    sys.path.append(str(VIENEU_PATH))

# Now we can import from the copied VieNeu-TTS utils
try:
    from app.services.vieneu.utils.phonemize_text import phonemize_with_dict
    from app.services.vieneu.utils.normalize_text import VietnameseTTSNormalizer
except ImportError as e:
    logger.error(f"Failed to import VieNeu-TTS utils: {e}")
    # Fallback to local import if called differently
    try:
        from utils.phonemize_text import phonemize_with_dict
        from utils.normalize_text import VietnameseTTSNormalizer
    except ImportError:
        logger.error("Total failure to import utils")
        def phonemize_with_dict(text: str, **kwargs) -> str: return text
        class VietnameseTTSNormalizer:
            def normalize(self, text: str) -> str: return text


# Cached NeuCodec encoder instance (singleton)
_neucodec_encoder: Optional[Any] = None
_neucodec_encoder_lock = None

def _get_neucodec_encoder():
    """Get or create cached NeuCodec encoder instance."""
    global _neucodec_encoder
    
    if _neucodec_encoder is None:
        try:
            from neucodec import NeuCodec
            logger.info("Loading NeuCodec encoder (cached)...")
            _neucodec_encoder = NeuCodec.from_pretrained("neuphonic/neucodec")
            _neucodec_encoder.eval()
            logger.success("NeuCodec encoder loaded and cached")
        except Exception as e:
            logger.error(f"Failed to load NeuCodec encoder: {e}")
            return None
    
    return _neucodec_encoder


class VieNeuEngine:
    """
    Engine for VieNeu-TTS synthesis with GPU acceleration.
    
    Features:
    - DirectML GPU acceleration for ONNX codec (AMD RX6600 optimized)
    - Automatic CPU fallback if GPU fails
    - Cached NeuCodec encoder for voice cloning
    - GPU memory monitoring
    """
    
    def __init__(self, model_dir: str) -> None:
        self.model_dir: Path = Path(model_dir)
        self.backbone_path: Path = self.model_dir
        self.codec_path: Path = self.model_dir / "neucodec-onnx" / "model.onnx"
        
        # Models
        self.tokenizer: Optional[AutoTokenizer] = None
        self.backbone: Optional[AutoModelForCausalLM] = None
        self.codec_session: Optional[ort.InferenceSession] = None
        
        # Configuration
        self.sample_rate: int = 24000
        self.max_context: int = 2048
        self.device: str = "cpu"
        
        # GPU status tracking
        self._using_gpu: bool = False
        self._gpu_provider: Optional[str] = None
        
        # Normalizer
        self.normalizer: VietnameseTTSNormalizer = VietnameseTTSNormalizer()
        
    def initialize(self, prefer_gpu: bool = True) -> bool:
        """
        Initialize models and sessions.
        
        Args:
            prefer_gpu: If True, attempt to use DirectML GPU acceleration
            
        Returns:
            bool: True if initialization successful
        """
        logger.info(f"Initializing VieNeuEngine from {self.model_dir}")
        
        try:
            # 1. Load Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.backbone_path))
            
            # 2. Load Backbone (Qwen2)
            # Backbone runs on CPU - transformers model is memory-intensive
            # and CPU is fast enough for the text-to-token generation
            self.backbone = AutoModelForCausalLM.from_pretrained(
                str(self.backbone_path),
                dtype=torch.float32,
                device_map="cpu"
            )
            self.backbone.eval()
            
            # 3. Load Codec (NeuCodec ONNX) with GPU support
            self._init_codec_session(prefer_gpu)
            
            logger.info("VieNeuEngine initialized successfully")
            return True
        except Exception as e:
            logger.exception(f"Failed to initialize VieNeuEngine: {e}")
            return False
    
    def _init_codec_session(self, prefer_gpu: bool = True) -> None:
        """
        Initialize ONNX codec session with GPU support.
        
        Attempts DirectML first for AMD GPU acceleration,
        falls back to CPU if DirectML fails or is unavailable.
        """
        from app.device_manager import get_device_manager
        dm = get_device_manager()
        
        # Configure session options for optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True
        
        # Set thread count for CPU operations
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 2
        
        if prefer_gpu and dm.is_directml_available:
            # Try DirectML first
            providers = dm.get_onnx_providers()
            provider_options = dm.get_onnx_provider_options()
            
            try:
                logger.info(f"Attempting DirectML for codec: {providers}")
                self.codec_session = ort.InferenceSession(
                    str(self.codec_path),
                    sess_options=sess_options,
                    providers=providers,
                    provider_options=provider_options
                )
                
                # Verify which provider is actually being used
                active_providers = self.codec_session.get_providers()
                logger.info(f"Codec session active providers: {active_providers}")
                
                if "DmlExecutionProvider" in active_providers:
                    self._using_gpu = True
                    self._gpu_provider = "DmlExecutionProvider"
                    logger.success("✅ Codec using DirectML GPU acceleration")
                else:
                    self._using_gpu = False
                    self._gpu_provider = active_providers[0] if active_providers else "CPU"
                    logger.warning(f"DirectML not active, using: {self._gpu_provider}")
                    
            except Exception as e:
                logger.warning(f"DirectML initialization failed: {e}")
                logger.info("Falling back to CPU for codec")
                self._init_cpu_codec(sess_options)
        else:
            logger.info("GPU not preferred or unavailable, using CPU for codec")
            self._init_cpu_codec(sess_options)
    
    def _init_cpu_codec(self, sess_options: ort.SessionOptions) -> None:
        """Initialize codec session with CPU only."""
        self.codec_session = ort.InferenceSession(
            str(self.codec_path),
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        self._using_gpu = False
        self._gpu_provider = "CPUExecutionProvider"
        logger.info("Codec using CPU")
    
    @property
    def is_using_gpu(self) -> bool:
        """Check if codec is using GPU acceleration."""
        return self._using_gpu
    
    def get_gpu_status(self) -> Dict[str, Any]:
        """Get GPU status information."""
        from app.device_manager import get_device_manager
        dm = get_device_manager()
        
        return {
            "codec_using_gpu": self._using_gpu,
            "codec_provider": self._gpu_provider,
            "directml_available": dm.is_directml_available,
            "device_name": dm.device_name,
        }
    
    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get memory usage information.
        Useful for monitoring AMD RX6600 VRAM usage.
        """
        import psutil
        
        info = {
            "system_memory_percent": psutil.virtual_memory().percent,
            "system_memory_available_gb": psutil.virtual_memory().available / (1024**3),
        }
        
        # Try to get GPU memory info if available
        try:
            from app.device_manager import get_device_manager
            dm = get_device_manager()
            
            if dm.is_directml_available:
                # DirectML doesn't expose memory directly, but we can track process memory
                process = psutil.Process()
                info["process_memory_mb"] = process.memory_info().rss / (1024**2)
                info["gpu_acceleration"] = True
            else:
                info["gpu_acceleration"] = False
        except Exception as e:
            logger.debug(f"Could not get GPU memory info: {e}")
            
        return info

    def get_speech_tokens(self, text: str, ref_codes: List[int], ref_text: str) -> str:
        """
        Generate speech tokens from text and reference.
        
        Args:
            text: Input text to synthesize
            ref_codes: Reference audio codes for voice cloning
            ref_text: Transcript of reference audio
            
        Returns:
            str: Generated speech token string
        """
        # 1. Phonemize
        ref_text_phones = phonemize_with_dict(ref_text)
        input_text_phones = phonemize_with_dict(text)
        
        # 2. Format Prompt
        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes])
        prompt = (
            f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text_phones} {input_text_phones}"
            f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
        )
        
        # 3. Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.backbone.device)
        input_len = inputs["input_ids"].shape[1]
        
        # 4. Generate
        speech_end_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
        with torch.no_grad():
            output_tokens = self.backbone.generate(
                **inputs,
                max_new_tokens=self.max_context - input_len,
                eos_token_id=speech_end_id,
                do_sample=True,
                temperature=0.6,        # Lower for stable output
                top_k=30,               # Reduced for quality
                top_p=0.9,              # Nucleus sampling
                repetition_penalty=1.2, # Prevent degenerate sequences
                use_cache=True,
                min_new_tokens=50,      # Ensure sufficient output length
            )
        
        # 5. Decode tokens to string
        # Skip input tokens
        new_tokens = output_tokens[0][input_len:]
        output_str = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        return output_str

    def decode_tokens(self, codes_str: str) -> np.ndarray:
        """
        Decode speech tokens string to audio waveform using ONNX Codec.
        
        Uses GPU acceleration via DirectML if available.
        
        Args:
            codes_str: String containing speech tokens like <|speech_123|>
            
        Returns:
            np.ndarray: Audio waveform at 24kHz
        """
        # Extract speech token IDs
        speech_ids = [int(num) for num in re.findall(r"<\|speech_(\d+)\|>", codes_str)]
        
        if not speech_ids:
            logger.warning("No speech tokens found in output")
            return np.zeros(0, dtype=np.float32)
        
        # Prepare ONNX input
        # NeuCodec ONNX decoder expects [1, 1, T] with int32
        codes = np.array(speech_ids, dtype=np.int32)[np.newaxis, np.newaxis, :]
        
        # Run ONNX session (GPU accelerated if DirectML is active)
        input_name = self.codec_session.get_inputs()[0].name
        recon = self.codec_session.run(None, {input_name: codes})[0]
        
        # Output is [1, 1, T]
        return recon[0, 0, :]

    def load_preencoded(self, pt_path: str) -> List[int]:
        """
        Load pre-encoded tokens from a .pt file.
        
        Args:
            pt_path: Path to .pt file containing encoded tokens
            
        Returns:
            List[int]: List of speech token IDs
        """
        try:
            tokens = torch.load(pt_path, map_location="cpu", weights_only=True)
            if isinstance(tokens, torch.Tensor):
                return tokens.flatten().tolist()
            return tokens
        except Exception as e:
            logger.error(f"Failed to load pre-encoded tokens from {pt_path}: {e}")
            return []

    def encode_reference(self, wav_path: str) -> List[int]:
        """
        Encode a reference audio file to speech tokens for voice cloning.
        
        Uses cached NeuCodec encoder instance to avoid reloading.
        
        Args:
            wav_path: Path to reference audio file (WAV/MP3)
            
        Returns:
            List[int]: List of speech token codes
        """
        try:
            # Use cached encoder instance
            codec = _get_neucodec_encoder()
            if codec is None:
                logger.error("NeuCodec encoder not available")
                return []
            
            # Load and preprocess audio at 24kHz (native sample rate for NeuCodec)
            wav, _ = librosa.load(wav_path, sr=24000, mono=True)
            wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)
            
            # Encode
            with torch.no_grad():
                ref_codes = codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
            
            return ref_codes.cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"Failed to encode reference: {e}")
            return []

    def synthesize(
        self, 
        text: str, 
        ref_codes: List[int], 
        ref_text: str, 
        speed: float = 1.0,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> np.ndarray:
        """
        Full synthesis pipeline.
        
        Args:
            text: Text to synthesize
            ref_codes: Reference audio codes for voice cloning
            ref_text: Transcript of reference audio
            speed: Speech speed multiplier (0.5-2.0)
            progress_callback: Optional callback(status, progress)
            
        Returns:
            np.ndarray: Synthesized audio waveform at 24kHz
        """
        if not text.strip():
            return np.zeros(0, dtype=np.float32)
        
        if progress_callback:
            progress_callback("Phonemizing text...", 0.1)
            
        # 1. Generate tokens
        codes_str = self.get_speech_tokens(text, ref_codes, ref_text)
        
        if progress_callback:
            progress_callback("Decoding audio (Codec)...", 0.6)
        
        # 2. Decode to audio (GPU accelerated)
        audio = self.decode_tokens(codes_str)
        
        if progress_callback:
            progress_callback("Post-processing...", 0.85)
        
        # 3. Apply speed if not 1.0
        if speed != 1.0 and len(audio) > 0:
            if progress_callback:
                progress_callback("Adjusting speed...", 0.9)
            # Fast time stretch
            audio = librosa.effects.time_stretch(audio, rate=speed)
        
        if progress_callback:
            progress_callback("Done!", 1.0)
            
        return audio
    
    def synthesize_streaming(
        self, 
        text: str, 
        ref_codes: List[int], 
        ref_text: str, 
        speed: float = 1.0,
        chunk_callback: Optional[Callable[[np.ndarray, int, int], None]] = None
    ) -> Generator[np.ndarray, None, None]:
        """
        Streaming synthesis - yields audio chunks as they're generated.
        
        This splits the text into sentences and yields audio for each sentence,
        allowing playback to begin before full synthesis completes.
        
        Args:
            text: Text to synthesize
            ref_codes: Reference audio codes for voice cloning
            ref_text: Transcript of reference audio
            speed: Speech speed multiplier (0.5-2.0)
            chunk_callback: Optional callback(audio_chunk, chunk_index, total_chunks)
            
        Yields:
            np.ndarray: Audio chunk for each sentence
        """
        if not text.strip():
            return
        
        # Split text into sentences for streaming
        sentences = self._split_into_sentences(text)
        total_chunks = len(sentences)
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            try:
                # Generate audio for this sentence
                codes_str = self.get_speech_tokens(sentence, ref_codes, ref_text)
                audio_chunk = self.decode_tokens(codes_str)
                
                # Apply speed if needed
                if speed != 1.0 and len(audio_chunk) > 0:
                    audio_chunk = librosa.effects.time_stretch(audio_chunk, rate=speed)
                
                # Callback for progress tracking
                if chunk_callback:
                    chunk_callback(audio_chunk, i, total_chunks)
                
                yield audio_chunk
                
            except Exception as e:
                logger.error(f"Failed to synthesize sentence {i}: {e}")
                continue
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences for streaming synthesis.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Vietnamese sentence delimiters
        delimiters = r'[.!?;:。！？；：]+'
        
        # Split by delimiters but keep the delimiter with the sentence
        parts = re.split(f'({delimiters})', text)
        
        sentences = []
        current = ""
        
        for part in parts:
            if re.match(delimiters, part):
                # This is a delimiter, append to current sentence
                current += part
                if current.strip():
                    sentences.append(current.strip())
                current = ""
            else:
                current = part
        
        # Add any remaining text
        if current.strip():
            sentences.append(current.strip())
        
        # If no sentences found (no delimiters), return the whole text
        if not sentences:
            sentences = [text]
        
        return sentences
    
    def cleanup(self) -> None:
        """Release resources and clear GPU memory."""
        global _neucodec_encoder
        
        # Clear models
        self.backbone = None
        self.tokenizer = None
        self.codec_session = None
        
        # Clear cached encoder
        _neucodec_encoder = None
        
        # Force garbage collection
        gc.collect()
        
        logger.info("VieNeuEngine resources cleaned up")
