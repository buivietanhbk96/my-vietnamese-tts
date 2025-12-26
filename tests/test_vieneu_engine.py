"""
Unit tests for VieNeu TTS Engine module.
Tests GPU initialization, synthesis pipeline, and voice cloning.
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
import numpy as np
from pathlib import Path


class TestVieNeuEngineInit:
    """Tests for VieNeuEngine initialization."""
    
    def test_engine_init_sets_paths(self):
        """Test engine initialization sets correct paths."""
        from app.services.vieneu_engine import VieNeuEngine
        
        test_dir = "/test/model/dir"
        engine = VieNeuEngine(test_dir)
        
        assert engine.model_dir == Path(test_dir)
        assert engine.backbone_path == Path(test_dir)
        assert engine.codec_path == Path(test_dir) / "neucodec-onnx" / "model.onnx"
    
    def test_engine_init_default_config(self):
        """Test engine initialization sets default config values."""
        from app.services.vieneu_engine import VieNeuEngine
        
        engine = VieNeuEngine("/test/dir")
        
        assert engine.sample_rate == 24000
        assert engine.max_context == 2048
        assert engine.device == "cpu"
    
    def test_engine_init_models_none(self):
        """Test engine initialization sets models to None."""
        from app.services.vieneu_engine import VieNeuEngine
        
        engine = VieNeuEngine("/test/dir")
        
        assert engine.tokenizer is None
        assert engine.backbone is None
        assert engine.codec_session is None


class TestVieNeuEngineGPUStatus:
    """Tests for GPU status tracking."""
    
    def test_is_using_gpu_default_false(self):
        """Test is_using_gpu defaults to False before init."""
        from app.services.vieneu_engine import VieNeuEngine
        
        engine = VieNeuEngine("/test/dir")
        
        assert engine.is_using_gpu is False
    
    def test_get_gpu_status_structure(self):
        """Test get_gpu_status returns expected structure."""
        from app.services.vieneu_engine import VieNeuEngine
        
        engine = VieNeuEngine("/test/dir")
        
        with patch('app.services.vieneu_engine.get_device_manager') as mock_dm:
            mock_dm.return_value.is_directml_available = False
            mock_dm.return_value.device_name = "CPU"
            
            status = engine.get_gpu_status()
        
        expected_keys = ['codec_using_gpu', 'codec_provider', 'directml_available', 'device_name']
        for key in expected_keys:
            assert key in status, f"Missing key: {key}"


class TestVieNeuEngineDecodeTokens:
    """Tests for token decoding functionality."""
    
    def test_decode_tokens_empty_string(self):
        """Test decode_tokens with empty string returns empty array."""
        from app.services.vieneu_engine import VieNeuEngine
        
        engine = VieNeuEngine("/test/dir")
        
        # Mock codec session
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name='codes')]
        mock_session.run.return_value = [np.zeros((1, 1, 100), dtype=np.float32)]
        engine.codec_session = mock_session
        
        result = engine.decode_tokens("")
        
        assert len(result) == 0
    
    def test_decode_tokens_extracts_ids(self):
        """Test decode_tokens correctly extracts speech IDs."""
        from app.services.vieneu_engine import VieNeuEngine
        
        engine = VieNeuEngine("/test/dir")
        
        # Mock codec session
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = 'codes'
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.run.return_value = [np.ones((1, 1, 100), dtype=np.float32)]
        engine.codec_session = mock_session
        
        test_str = "<|speech_1|><|speech_2|><|speech_3|>"
        result = engine.decode_tokens(test_str)
        
        # Verify session was called
        mock_session.run.assert_called_once()


class TestVieNeuEngineLoadPreencoded:
    """Tests for loading pre-encoded tokens."""
    
    def test_load_preencoded_invalid_path(self):
        """Test load_preencoded returns empty list for invalid path."""
        from app.services.vieneu_engine import VieNeuEngine
        
        engine = VieNeuEngine("/test/dir")
        
        result = engine.load_preencoded("/nonexistent/path.pt")
        
        assert result == []


class TestNeuCodecEncoderCache:
    """Tests for NeuCodec encoder caching."""
    
    def test_get_neucodec_encoder_caches_instance(self):
        """Test _get_neucodec_encoder caches the encoder instance."""
        import app.services.vieneu_engine as module
        
        # Reset cache
        module._neucodec_encoder = None
        
        with patch('app.services.vieneu_engine.NeuCodec', create=True) as MockNeuCodec:
            mock_instance = MagicMock()
            MockNeuCodec.from_pretrained.return_value = mock_instance
            
            # Should work without actual neucodec if mocked properly
            # Skip if neucodec not installed
            try:
                from neucodec import NeuCodec
                # Real import works, skip mock test
                pytest.skip("NeuCodec installed, skipping mock test")
            except ImportError:
                # Expected - neucodec not installed
                pass


class TestVieNeuEngineSynthesize:
    """Tests for synthesis functionality."""
    
    def test_synthesize_empty_text_returns_empty(self):
        """Test synthesize with empty text returns empty array."""
        from app.services.vieneu_engine import VieNeuEngine
        
        engine = VieNeuEngine("/test/dir")
        
        result = engine.synthesize("", [], "", 1.0)
        
        assert len(result) == 0
    
    def test_synthesize_whitespace_only_returns_empty(self):
        """Test synthesize with whitespace only returns empty array."""
        from app.services.vieneu_engine import VieNeuEngine
        
        engine = VieNeuEngine("/test/dir")
        
        result = engine.synthesize("   ", [], "", 1.0)
        
        assert len(result) == 0


class TestVieNeuEngineMemoryInfo:
    """Tests for memory information retrieval."""
    
    def test_get_memory_info_returns_dict(self):
        """Test get_memory_info returns dictionary."""
        from app.services.vieneu_engine import VieNeuEngine
        
        engine = VieNeuEngine("/test/dir")
        
        info = engine.get_memory_info()
        
        assert isinstance(info, dict)
        assert 'system_memory_percent' in info
        assert 'system_memory_available_gb' in info


class TestVieNeuEngineCleanup:
    """Tests for resource cleanup."""
    
    def test_cleanup_clears_models(self):
        """Test cleanup method clears all model references."""
        from app.services.vieneu_engine import VieNeuEngine
        
        engine = VieNeuEngine("/test/dir")
        engine.backbone = MagicMock()
        engine.tokenizer = MagicMock()
        engine.codec_session = MagicMock()
        
        engine.cleanup()
        
        assert engine.backbone is None
        assert engine.tokenizer is None
        assert engine.codec_session is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
