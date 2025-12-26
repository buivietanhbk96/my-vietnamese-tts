"""
Pytest configuration for Vietnamese TTS PRO tests.
"""

import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root_path():
    """Get project root path."""
    return project_root


@pytest.fixture
def sample_srt_content():
    """Sample SRT content for testing."""
    return """1
00:00:01,000 --> 00:00:05,000
Xin chào các bạn.

2
00:00:06,000 --> 00:00:10,500
Đây là bài test tiếng Việt.

3
00:00:11,000 --> 00:00:15,000
Cảm ơn đã sử dụng ứng dụng."""


@pytest.fixture
def mock_device_manager():
    """Mock device manager for testing."""
    from unittest.mock import MagicMock
    
    mock = MagicMock()
    mock.device_name = "Test CPU"
    mock.device_type = "cpu"
    mock.is_gpu = False
    mock.is_directml_available = False
    mock.get_onnx_providers.return_value = ["CPUExecutionProvider"]
    mock.get_onnx_provider_options.return_value = [{}]
    mock.get_device_info.return_value = {
        "device": "cpu",
        "device_type": "cpu",
        "device_name": "Test CPU",
        "is_gpu": False,
        "directml_available": False,
    }
    
    return mock


@pytest.fixture
def reset_device_manager_singleton():
    """Reset DeviceManager singleton for isolated testing."""
    from app.device_manager import DeviceManager
    
    original_instance = DeviceManager._instance
    original_initialized = DeviceManager._initialized
    
    DeviceManager._instance = None
    DeviceManager._initialized = False
    
    yield
    
    # Restore after test
    DeviceManager._instance = original_instance
    DeviceManager._initialized = original_initialized


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow-running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if no GPU available."""
    skip_gpu = pytest.mark.skip(reason="GPU not available for testing")
    
    for item in items:
        if "gpu" in item.keywords:
            # Check if GPU is available
            try:
                import torch_directml
                if not torch_directml.is_available():
                    item.add_marker(skip_gpu)
            except ImportError:
                item.add_marker(skip_gpu)
