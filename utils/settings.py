"""
Settings persistence for VietTTS Desktop Application
Saves and loads user preferences
"""

import json
from pathlib import Path
from typing import Any, Optional
from loguru import logger

from app.config import Config


class Settings:
    """
    Manages application settings persistence
    """
    
    DEFAULT_SETTINGS = {
        "last_voice": None,
        "speed": 1.0,
        "output_format": "wav",
        "output_directory": str(Config.OUTPUT_DIR),
        "window_geometry": None,
        "volume": 0.8,
        "theme": "dark",
        "recent_files": [],
        "max_recent_files": 10,
        "auto_play": True,
        "srt_last_directory": None,
    }
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize settings manager
        
        Args:
            config_file: Path to settings file (default: from Config)
        """
        self.config_file = config_file or Config.CONFIG_FILE
        self._settings = self.DEFAULT_SETTINGS.copy()
        self.load()
    
    def load(self) -> bool:
        """
        Load settings from file
        
        Returns:
            bool: True if loaded successfully
        """
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    saved_settings = json.load(f)
                
                # Merge with defaults (in case new settings added)
                for key, value in saved_settings.items():
                    if key in self._settings:
                        self._settings[key] = value
                
                logger.info(f"Settings loaded from: {self.config_file}")
                return True
            else:
                logger.info("No settings file found, using defaults")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            return False
    
    def save(self) -> bool:
        """
        Save settings to file
        
        Returns:
            bool: True if saved successfully
        """
        try:
            # Ensure directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._settings, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Settings saved to: {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get setting value
        
        Args:
            key: Setting key
            default: Default value if not found
            
        Returns:
            Setting value
        """
        return self._settings.get(key, default)
    
    def set(self, key: str, value: Any, auto_save: bool = True):
        """
        Set setting value
        
        Args:
            key: Setting key
            value: Setting value
            auto_save: Whether to save immediately
        """
        self._settings[key] = value
        
        if auto_save:
            self.save()
    
    def reset(self, key: Optional[str] = None):
        """
        Reset settings to defaults
        
        Args:
            key: Specific key to reset, or None for all
        """
        if key:
            if key in self.DEFAULT_SETTINGS:
                self._settings[key] = self.DEFAULT_SETTINGS[key]
        else:
            self._settings = self.DEFAULT_SETTINGS.copy()
        
        self.save()
    
    def add_recent_file(self, filepath: str):
        """Add file to recent files list"""
        recent = self._settings.get("recent_files", [])
        
        # Remove if already exists
        if filepath in recent:
            recent.remove(filepath)
        
        # Add to front
        recent.insert(0, filepath)
        
        # Limit size
        max_files = self._settings.get("max_recent_files", 10)
        self._settings["recent_files"] = recent[:max_files]
        
        self.save()
    
    def get_recent_files(self) -> list:
        """Get list of recent files"""
        return self._settings.get("recent_files", [])
    
    def clear_recent_files(self):
        """Clear recent files list"""
        self._settings["recent_files"] = []
        self.save()
    
    # Convenience properties
    @property
    def last_voice(self) -> Optional[str]:
        return self.get("last_voice")
    
    @last_voice.setter
    def last_voice(self, value: str):
        self.set("last_voice", value)
    
    @property
    def speed(self) -> float:
        return self.get("speed", 1.0)
    
    @speed.setter
    def speed(self, value: float):
        self.set("speed", value)
    
    @property
    def output_format(self) -> str:
        return self.get("output_format", "wav")
    
    @output_format.setter
    def output_format(self, value: str):
        self.set("output_format", value)
    
    @property
    def output_directory(self) -> str:
        return self.get("output_directory", str(Config.OUTPUT_DIR))
    
    @output_directory.setter
    def output_directory(self, value: str):
        self.set("output_directory", value)
    
    @property
    def volume(self) -> float:
        return self.get("volume", 0.8)
    
    @volume.setter
    def volume(self, value: float):
        self.set("volume", value)
    
    @property
    def auto_play(self) -> bool:
        return self.get("auto_play", True)
    
    @auto_play.setter
    def auto_play(self, value: bool):
        self.set("auto_play", value)
