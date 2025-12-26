"""
Unit tests for Device Manager module.
Tests GPU detection, DirectML availability, and fallback behavior.
"""

import pytest
from unittest.mock import patch, MagicMock
import torch


class TestDeviceManager:
    """Tests for DeviceManager class."""
    
    def test_singleton_pattern(self):
        """Test that DeviceManager follows singleton pattern."""
        from app.device_manager import DeviceManager
        
        # Reset singleton for test
        DeviceManager._instance = None
        DeviceManager._initialized = False
        
        dm1 = DeviceManager()
        dm2 = DeviceManager()
        
        assert dm1 is dm2, "DeviceManager should be singleton"
    
    def test_device_property_returns_torch_device(self):
        """Test device property returns valid torch.device."""
        from app.device_manager import get_device_manager
        
        dm = get_device_manager()
        
        assert dm.device is not None
        assert isinstance(dm.device, torch.device) or hasattr(dm.device, 'type')
    
    def test_device_type_valid(self):
        """Test device_type is either 'directml' or 'cpu'."""
        from app.device_manager import get_device_manager
        
        dm = get_device_manager()
        
        assert dm.device_type in ['directml', 'cpu']
    
    def test_device_name_not_empty(self):
        """Test device_name returns non-empty string."""
        from app.device_manager import get_device_manager
        
        dm = get_device_manager()
        
        assert dm.device_name
        assert isinstance(dm.device_name, str)
        assert len(dm.device_name) > 0
    
    def test_onnx_providers_returns_list(self):
        """Test get_onnx_providers returns valid list."""
        from app.device_manager import get_device_manager
        
        dm = get_device_manager()
        providers = dm.get_onnx_providers()
        
        assert isinstance(providers, list)
        assert len(providers) > 0
        assert 'CPUExecutionProvider' in providers
    
    def test_onnx_provider_options_matches_providers(self):
        """Test provider options list matches providers list length."""
        from app.device_manager import get_device_manager
        
        dm = get_device_manager()
        providers = dm.get_onnx_providers()
        options = dm.get_onnx_provider_options()
        
        assert len(providers) == len(options)
    
    def test_get_device_info_structure(self):
        """Test get_device_info returns expected keys."""
        from app.device_manager import get_device_manager
        
        dm = get_device_manager()
        info = dm.get_device_info()
        
        expected_keys = ['device', 'device_type', 'device_name', 'is_gpu', 'directml_available']
        for key in expected_keys:
            assert key in info, f"Missing key: {key}"
    
    def test_is_gpu_matches_device_type(self):
        """Test is_gpu property matches device_type."""
        from app.device_manager import get_device_manager
        
        dm = get_device_manager()
        
        if dm.device_type == 'directml':
            assert dm.is_gpu is True
        else:
            assert dm.is_gpu is False


class TestDeviceManagerCPUFallback:
    """Tests for CPU fallback behavior."""
    
    def test_cpu_fallback_when_no_directml(self):
        """Test CPU fallback when DirectML not installed."""
        from app.device_manager import DeviceManager
        
        # Reset singleton
        DeviceManager._instance = None
        DeviceManager._initialized = False
        
        with patch.dict('sys.modules', {'torch_directml': None}):
            dm = DeviceManager()
            # Should fallback to CPU
            assert dm.device_type == 'cpu' or dm._directml_available is False
    
    def test_force_cpu_method(self):
        """Test force_cpu method switches to CPU."""
        from app.device_manager import get_device_manager
        
        dm = get_device_manager()
        dm.force_cpu()
        
        assert dm.device_type == 'cpu'
        assert dm.is_gpu is False


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    def test_get_device_manager_returns_instance(self):
        """Test get_device_manager returns DeviceManager instance."""
        from app.device_manager import get_device_manager, DeviceManager
        
        dm = get_device_manager()
        
        assert isinstance(dm, DeviceManager)
    
    def test_get_device_returns_torch_device(self):
        """Test get_device returns torch.device."""
        from app.device_manager import get_device
        
        device = get_device()
        
        assert device is not None
    
    def test_get_device_name_returns_string(self):
        """Test get_device_name returns string."""
        from app.device_manager import get_device_name
        
        name = get_device_name()
        
        assert isinstance(name, str)
    
    def test_is_gpu_available_returns_bool(self):
        """Test is_gpu_available returns boolean."""
        from app.device_manager import is_gpu_available
        
        result = is_gpu_available()
        
        assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
