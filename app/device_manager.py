"""
Device Manager for VietTTS Desktop Application
Handles GPU (DirectML) and CPU device selection with automatic fallback

Supports:
- AMD GPUs via DirectML (Windows)
- CPU fallback when GPU is unavailable
"""

import os
import torch
from typing import Optional, List, Dict, Any
from loguru import logger


class DeviceManager:
    """
    Manages compute device selection for TTS processing.
    Prioritizes DirectML (AMD GPU) with CPU fallback.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern to ensure consistent device selection"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if DeviceManager._initialized:
            return
            
        self._device: Optional[torch.device] = None
        self._device_type: str = "cpu"
        self._device_name: str = "CPU"
        self._directml_available: bool = False
        self._torch_directml = None
        
        self._detect_devices()
        DeviceManager._initialized = True
    
    def _detect_devices(self):
        """Detect available compute devices"""
        logger.info("Detecting available compute devices...")
        
        # Try DirectML first (AMD GPU on Windows)
        try:
            import torch_directml
            self._torch_directml = torch_directml
            
            # Check if DirectML device is available
            if torch_directml.is_available():
                self._directml_available = True
                self._device = torch_directml.device()
                self._device_type = "directml"
                self._device_name = self._get_directml_device_name()
                logger.success(f"DirectML device detected: {self._device_name}")
            else:
                logger.warning("torch_directml installed but no compatible GPU found")
                self._setup_cpu_fallback()
                
        except ImportError:
            logger.info("torch_directml not installed, using CPU")
            self._setup_cpu_fallback()
        except Exception as e:
            logger.warning(f"DirectML initialization failed: {e}")
            self._setup_cpu_fallback()
    
    def _setup_cpu_fallback(self):
        """Configure CPU as compute device"""
        self._device = torch.device("cpu")
        self._device_type = "cpu"
        self._device_name = "CPU"
        self._directml_available = False
        logger.info("Using CPU for computation")
    
    def _get_directml_device_name(self) -> str:
        """Get the name of the DirectML device (GPU name)"""
        try:
            # Try to get GPU info from DirectML
            if self._torch_directml:
                device_count = self._torch_directml.device_count()
                if device_count > 0:
                    # torch_directml doesn't expose device name directly
                    # We'll use a generic name with device count
                    return f"AMD GPU (DirectML, {device_count} device(s))"
        except Exception as e:
            logger.debug(f"Could not get DirectML device name: {e}")
        
        return "DirectML GPU"
    
    @property
    def device(self) -> torch.device:
        """Get the selected compute device"""
        return self._device
    
    @property
    def device_type(self) -> str:
        """Get device type string: 'directml' or 'cpu'"""
        return self._device_type
    
    @property
    def device_name(self) -> str:
        """Get human-readable device name"""
        return self._device_name
    
    @property
    def is_gpu(self) -> bool:
        """Check if using GPU acceleration"""
        return self._device_type == "directml"
    
    @property
    def is_directml_available(self) -> bool:
        """Check if DirectML is available"""
        return self._directml_available
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get detailed device information"""
        info = {
            "device": str(self._device),
            "device_type": self._device_type,
            "device_name": self._device_name,
            "is_gpu": self.is_gpu,
            "directml_available": self._directml_available,
        }
        
        # Add memory info if available
        if self.is_gpu and self._torch_directml:
            try:
                info["device_count"] = self._torch_directml.device_count()
            except:
                pass
        
        return info
    
    def get_onnx_providers(self) -> List[str]:
        """
        Get ONNX Runtime execution providers in priority order.
        Returns DirectML provider if GPU is available, with CPU fallback.
        """
        providers = []
        
        if self._directml_available:
            providers.append("DmlExecutionProvider")
        
        # Always add CPU as fallback
        providers.append("CPUExecutionProvider")
        
        return providers
    
    def get_onnx_provider_options(self) -> List[Dict[str, Any]]:
        """Get ONNX Runtime provider options"""
        options = []
        
        if self._directml_available:
            # DirectML provider options
            options.append({
                "device_id": 0,  # Use first GPU
            })
        
        # CPU provider options (empty dict for default)
        options.append({})
        
        return options
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to the selected device"""
        return tensor.to(self._device)
    
    def empty_cache(self):
        """Clear GPU memory cache if applicable"""
        if self.is_gpu:
            try:
                # DirectML doesn't have explicit cache clearing like CUDA
                # but we can try to free memory by running garbage collection
                import gc
                gc.collect()
                logger.debug("Cleared memory cache")
            except Exception as e:
                logger.debug(f"Cache clearing not supported: {e}")
    
    def force_cpu(self):
        """Force CPU usage (for debugging or fallback)"""
        logger.info("Forcing CPU usage")
        self._setup_cpu_fallback()
    
    def __repr__(self) -> str:
        return f"DeviceManager(device={self._device}, type={self._device_type})"


# Global instance for easy access
_device_manager: Optional[DeviceManager] = None


def get_device_manager() -> DeviceManager:
    """Get the global DeviceManager instance"""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager


def get_device() -> torch.device:
    """Convenience function to get the current compute device"""
    return get_device_manager().device


def get_device_name() -> str:
    """Convenience function to get the current device name"""
    return get_device_manager().device_name


def is_gpu_available() -> bool:
    """Convenience function to check if GPU is available"""
    return get_device_manager().is_gpu
