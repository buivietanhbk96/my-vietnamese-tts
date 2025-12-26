"""
VieNeu-TTS Engine for Vietnamese TTS PRO
Uses Qwen2 Backbone (transformers) and NeuCodec (ONNX DirectML)
"""

import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Generator, Tuple

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
        def phonemize_with_dict(text, **kwargs): return text
        class VietnameseTTSNormalizer:
            def normalize(self, text): return text

class VieNeuEngine:
    """
    Engine for VieNeu-TTS synthesis
    """
    
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.backbone_path = self.model_dir
        self.codec_path = self.model_dir / "neucodec-onnx" / "model.onnx"
        
        # Models
        self.tokenizer = None
        self.backbone = None
        self.codec_session = None
        
        # Configuration
        self.sample_rate = 24000
        self.max_context = 2048
        self.device = "cpu"
        
        # Normalizer
        self.normalizer = VietnameseTTSNormalizer()
        
    def initialize(self, prefer_gpu: bool = True):
        """Initialize models and sessions"""
        logger.info(f"Initializing VieNeuEngine from {self.model_dir}")
        
        try:
            # 1. Load Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.backbone_path))
            
            # 2. Load Backbone (Qwen2)
            # For now, we use CPU for backbone to avoid VRAM issues and because it's fast enough
            # We can later optimize this with DirectML if needed
            self.backbone = AutoModelForCausalLM.from_pretrained(
                str(self.backbone_path),
                torch_dtype=torch.float32,
                device_map="cpu"
            )
            self.backbone.eval()
            
            # 3. Load Codec (NeuCodec ONNX)
            # NOTE: We use CPU for the codec because some ops (ConvTranspose) 
            # might have issues with DirectML on certain configurations.
            # The codec is fast enough on CPU.
            providers = ['CPUExecutionProvider']
            
            self.codec_session = ort.InferenceSession(str(self.codec_path), providers=providers)
            
            logger.info("VieNeuEngine initialized successfully")
            return True
        except Exception as e:
            logger.exception(f"Failed to initialize VieNeuEngine: {e}")
            return False

    def get_speech_tokens(self, text: str, ref_codes: List[int], ref_text: str) -> str:
        """Generate speech tokens from text and reference"""
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
                temperature=1.0,
                top_k=50,
                use_cache=True,
                min_new_tokens=20,
            )
        
        # 5. Decode tokens to string
        # Skip input tokens
        new_tokens = output_tokens[0][input_len:]
        output_str = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        return output_str

    def decode_tokens(self, codes_str: str) -> np.ndarray:
        """Decode speech tokens string to audio waveform using ONNX Codec"""
        # Extract speech token IDs
        speech_ids = [int(num) for num in re.findall(r"<\|speech_(\d+)\|>", codes_str)]
        
        if not speech_ids:
            logger.warning("No speech tokens found in output")
            return np.zeros(0, dtype=np.float32)
        
        # Prepare ONNX input
        # NeuCodec ONNX decoder expects [1, 1, T] with int32
        codes = np.array(speech_ids, dtype=np.int32)[np.newaxis, np.newaxis, :]
        
        # Run ONNX session
        # input name is likely 'codes'
        input_name = self.codec_session.get_inputs()[0].name
        recon = self.codec_session.run(None, {input_name: codes})[0]
        
        # Output is [1, 1, T]
        return recon[0, 0, :]

    def load_preencoded(self, pt_path: str) -> List[int]:
        """Load pre-encoded tokens from a .pt file"""
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
        Encode a reference audio file to speech tokens for cloning.
        Uses the torch-based NeuCodec encoder.
        """
        try:
            from neucodec import NeuCodec
            # We load a temporary instance for encoding
            codec = NeuCodec.from_pretrained("neuphonic/neucodec")
            codec.eval()
            
            wav, _ = librosa.load(wav_path, sr=16000, mono=True)
            wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)
            
            with torch.no_grad():
                ref_codes = codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
            
            return ref_codes.cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"Failed to encode reference: {e}")
            return []

    def synthesize(self, text: str, ref_codes: List[int], ref_text: str, speed: float = 1.0) -> np.ndarray:
        """Full synthesis pipeline"""
        if not text.strip():
            return np.zeros(0, dtype=np.float32)
            
        # 1. Generate tokens
        codes_str = self.get_speech_tokens(text, ref_codes, ref_text)
        
        # 2. Decode to audio
        audio = self.decode_tokens(codes_str)
        
        # 3. Apply speed if not 1.0
        if speed != 1.0 and len(audio) > 0:
             # Fast time stretch
            audio = librosa.effects.time_stretch(audio, rate=speed)
            
        return audio
