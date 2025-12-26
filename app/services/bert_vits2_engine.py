import sys
import os
import time
import logging
import torch
import numpy as np
import onnxruntime
from pathlib import Path
from typing import Optional, Dict

# Add Style-Bert-VITS2 to python path
ROOT_DIR = Path(__file__).parent.parent.parent
SBV2_DIR = ROOT_DIR / "Style-Bert-VITS2"
if str(SBV2_DIR) not in sys.path:
    sys.path.append(str(SBV2_DIR))

# Import from Style-Bert-VITS2
try:
    from style_bert_vits2.models.infer import get_text
    from style_bert_vits2.utils.utils import get_hparams_from_file
    from style_bert_vits2.constants import Languages
    from style_bert_vits2.logging import logger as sbv2_logger
    
    # Silence sbv2 logger
    sbv2_logger.setLevel(logging.WARNING)
except ImportError:
    pass # Handle in initialize if needed

logger = logging.getLogger(__name__)

class BertVITS2Engine:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.config_path = self.model_dir / "config.json"
        
        # Determine model file (ONNX preferred)
        self.onnx_path = self.model_dir / f"{self.model_dir.name}.onnx"
        self.safetensors_path = list(self.model_dir.glob("*.safetensors"))
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found at {self.config_path}")
            
        self.hparams = get_hparams_from_file(str(self.config_path))
        self.device = "cpu" # Text processing on CPU
        self.session = None
        self.style_vectors = None
        
    def initialize(self):
        """Load model and resources"""
        logger.info(f"Initializing BertVITS2Engine from {self.model_dir}")
        
        # Load ONNX model
        if self.onnx_path.exists():
            logger.info(f"Loading ONNX model: {self.onnx_path}")
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
            try:
                self.session = onnxruntime.InferenceSession(str(self.onnx_path), providers=providers)
                logger.info(f"ONNX Session initialized with providers: {self.session.get_providers()}")
            except Exception as e:
                logger.error(f"Failed to init ONNX DirectML: {e}, falling back to CPU")
                self.session = onnxruntime.InferenceSession(str(self.onnx_path), providers=['CPUExecutionProvider'])
        else:
            raise RuntimeError("ONNX model not found. Please run export_onnx.py first.")
            
        # Load style vectors
        style_vec_path = self.model_dir / "style_vectors.npy"
        if style_vec_path.exists():
            self.style_vectors = np.load(style_vec_path)
            logger.info(f"Loaded {self.style_vectors.shape[0]} style vectors")
        else:
            logger.warning("Style vectors not found")
            
    def get_style_vector(self, wav_path: str) -> np.ndarray:
        """Extract style vector from audio file"""
        from pyannote.audio import Inference, Model
        
        # Lazy load style encoder
        if not hasattr(self, 'style_encoder'):
            logger.info("Loading style encoder model...")
            self.style_encoder = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
            self.style_inference = Inference(self.style_encoder, window="whole")
            self.style_inference.to(torch.device("cpu")) # Run on CPU to avoid conflict/VRAM usage
            
        logger.info(f"Extracting style from {wav_path}")
        vec = self.style_inference(wav_path)
        return vec.reshape(1, -1)

    def synthesize(self, 
                   text: str, 
                   style_id: int = 0, 
                   speed: float = 1.0, 
                   language: str = "JP",
                   style_u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Synthesize speech from text
        style_u: Optional external style vector (from cloning)
        """
        start_time = time.time()
        
        # Determine language enum
        lang_enum = Languages.JP 
        if language.upper() == "EN":
             lang_enum = Languages.EN
        elif language.upper() == "ZH":
             lang_enum = Languages.ZH
        
        # 1. Text Processing (Get features)
        bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(
            text,
            lang_enum,
            self.hparams,
            self.device
        )
        
        # 2. Prepare ONNX Inputs
        x_tst = phones.unsqueeze(0).numpy()
        tones = tones.unsqueeze(0).numpy()
        lang_ids = lang_ids.unsqueeze(0).numpy()
        bert = bert.unsqueeze(0).numpy()
        ja_bert = ja_bert.unsqueeze(0).numpy()
        en_bert = en_bert.unsqueeze(0).numpy()
        x_tst_lengths = np.array([phones.shape[0]], dtype=np.int64)
        
        # Style vector selection
        if style_u is not None:
             style_vec = style_u
        elif self.style_vectors is not None and 0 <= style_id < self.style_vectors.shape[0]:
            style_vec = self.style_vectors[style_id:style_id+1]
        else:
            style_vec = np.zeros((1, 256), dtype=np.float32) 
            
        sid = np.array([0], dtype=np.int64) 
        
        # 3. Validating Inputs 
        inputs = {
            "x_tst": x_tst,
            "x_tst_lengths": x_tst_lengths,
            "sid": sid,
            "tones": tones,
            "language": lang_ids,
            "bert": bert,
            "ja_bert": ja_bert,
            "en_bert": en_bert,
            "style_vec": style_vec,
            "length_scale": np.array(speed, dtype=np.float32),
            "sdp_ratio": np.array(0.2, dtype=np.float32),
            "noise_scale": np.array(0.6, dtype=np.float32),
            "noise_scale_w": np.array(0.8, dtype=np.float32)
        }
        
        # Handle Non-JP-Extra / Dynamic Inputs based on session
        if self.session:
            input_names = [node.name for node in self.session.get_inputs()]
            final_inputs = {k: v for k, v in inputs.items() if k in input_names}
            
            # 4. Inference
            audio = self.session.run(None, final_inputs)[0]
            
            # 5. Post-process
            audio = audio.squeeze()
            
            logger.info(f"Synthesized in {time.time() - start_time:.3f}s")
            return audio
        else:
             raise RuntimeError("Session not initialized")

    def get_styles(self) -> Dict[int, str]:
        # TODO: Parse config or styles to return names (e.g. "Neutral", "Happy")
        return {i: f"Style {i}" for i in range(len(self.style_vectors))}
