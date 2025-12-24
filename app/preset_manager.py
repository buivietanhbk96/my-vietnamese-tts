# -*- coding: utf-8 -*-
"""
Preset System - Quản lý voice presets và settings
VIP Pro MAX feature - Lưu và load cấu hình như DAW chuyên nghiệp
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import hashlib
import shutil


class PresetCategory(Enum):
    """Danh mục preset"""
    VOICE = "voice"  # Voice settings
    AUDIO = "audio"  # Audio processing
    PROJECT = "project"  # Full project
    BATCH = "batch"  # Batch processing settings
    CUSTOM = "custom"  # User custom


@dataclass
class VoicePreset:
    """
    Preset cho voice settings
    Lưu trữ tất cả thông số để tái sử dụng voice
    """
    name: str
    description: str = ""
    
    # Voice file info
    voice_file_path: Optional[str] = None
    voice_file_hash: Optional[str] = None  # MD5 hash để verify
    
    # TTS settings
    speed: float = 1.0
    
    # Audio processing
    normalize: bool = True
    noise_reduction: bool = False
    noise_reduction_strength: float = 0.5
    
    # Advanced settings
    use_compression: bool = False
    compression_threshold: float = -20.0
    compression_ratio: float = 4.0
    
    # EQ settings
    eq_enabled: bool = False
    eq_low_gain: float = 0.0
    eq_mid_gain: float = 0.0
    eq_high_gain: float = 0.0
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)
    
    # Usage stats
    use_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Chuyển sang dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoicePreset":
        """Tạo từ dictionary"""
        # Handle missing fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class AudioPreset:
    """
    Preset cho audio processing settings
    """
    name: str
    description: str = ""
    
    # Normalization
    normalize_enabled: bool = True
    normalize_target_db: float = -3.0
    normalize_type: str = "peak"  # peak, rms, lufs
    
    # Noise reduction
    noise_reduction_enabled: bool = False
    noise_reduction_method: str = "spectral"  # spectral, rnnoise
    noise_reduction_strength: float = 0.5
    
    # Compression
    compression_enabled: bool = False
    compression_threshold: float = -20.0
    compression_ratio: float = 4.0
    compression_attack: float = 5.0
    compression_release: float = 50.0
    
    # Limiting
    limiter_enabled: bool = True
    limiter_threshold: float = -1.0
    
    # De-essing
    deesser_enabled: bool = False
    deesser_threshold: float = -30.0
    deesser_frequency: float = 6000.0
    
    # EQ
    eq_enabled: bool = False
    eq_low_freq: float = 100.0
    eq_low_gain: float = 0.0
    eq_mid_freq: float = 1000.0
    eq_mid_gain: float = 0.0
    eq_high_freq: float = 8000.0
    eq_high_gain: float = 0.0
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudioPreset":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class BatchPreset:
    """
    Preset cho batch processing
    """
    name: str
    description: str = ""
    
    # Output settings
    output_format: str = "wav"  # wav, mp3, flac
    output_sample_rate: int = 22050
    output_bit_depth: int = 16
    
    # Naming
    naming_pattern: str = "{index}"  # {index}, {text}, {timestamp}
    start_index: int = 1
    index_padding: int = 3  # 001, 002, etc.
    
    # Processing
    parallel_jobs: int = 2
    continue_on_error: bool = True
    
    # Voice preset reference
    voice_preset_name: Optional[str] = None
    audio_preset_name: Optional[str] = None
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchPreset":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class ProjectPreset:
    """
    Full project preset - bao gồm tất cả settings
    """
    name: str
    description: str = ""
    
    # Sub-presets
    voice_preset: Optional[VoicePreset] = None
    audio_preset: Optional[AudioPreset] = None
    batch_preset: Optional[BatchPreset] = None
    
    # Project-specific
    default_output_dir: str = ""
    auto_save: bool = True
    auto_save_interval: int = 300  # seconds
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "description": self.description,
            "default_output_dir": self.default_output_dir,
            "auto_save": self.auto_save,
            "auto_save_interval": self.auto_save_interval,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
        
        if self.voice_preset:
            result["voice_preset"] = self.voice_preset.to_dict()
        if self.audio_preset:
            result["audio_preset"] = self.audio_preset.to_dict()
        if self.batch_preset:
            result["batch_preset"] = self.batch_preset.to_dict()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectPreset":
        voice = None
        audio = None
        batch = None
        
        if "voice_preset" in data:
            voice = VoicePreset.from_dict(data["voice_preset"])
        if "audio_preset" in data:
            audio = AudioPreset.from_dict(data["audio_preset"])
        if "batch_preset" in data:
            batch = BatchPreset.from_dict(data["batch_preset"])
        
        return cls(
            name=data.get("name", "Unnamed"),
            description=data.get("description", ""),
            voice_preset=voice,
            audio_preset=audio,
            batch_preset=batch,
            default_output_dir=data.get("default_output_dir", ""),
            auto_save=data.get("auto_save", True),
            auto_save_interval=data.get("auto_save_interval", 300),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat())
        )


class PresetManager:
    """
    Quản lý tất cả presets
    - Load/Save presets từ file
    - Import/Export presets
    - Preset search và filtering
    """
    
    DEFAULT_PRESETS_DIR = "presets"
    VOICE_DIR = "voices"
    AUDIO_DIR = "audio"
    BATCH_DIR = "batch"
    PROJECT_DIR = "projects"
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize preset manager
        
        Args:
            base_dir: Thư mục gốc cho presets. Mặc định là ./presets
        """
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            self.base_dir = Path.cwd() / self.DEFAULT_PRESETS_DIR
        
        # Create directory structure
        self._ensure_directories()
        
        # Cache
        self._voice_presets: Dict[str, VoicePreset] = {}
        self._audio_presets: Dict[str, AudioPreset] = {}
        self._batch_presets: Dict[str, BatchPreset] = {}
        self._project_presets: Dict[str, ProjectPreset] = {}
        
        # Load existing presets
        self.reload_all()
    
    def _ensure_directories(self) -> None:
        """Tạo cấu trúc thư mục"""
        (self.base_dir / self.VOICE_DIR).mkdir(parents=True, exist_ok=True)
        (self.base_dir / self.AUDIO_DIR).mkdir(parents=True, exist_ok=True)
        (self.base_dir / self.BATCH_DIR).mkdir(parents=True, exist_ok=True)
        (self.base_dir / self.PROJECT_DIR).mkdir(parents=True, exist_ok=True)
    
    def _get_preset_path(
        self,
        name: str,
        category: PresetCategory
    ) -> Path:
        """Lấy đường dẫn file preset"""
        safe_name = self._sanitize_filename(name)
        
        if category == PresetCategory.VOICE:
            return self.base_dir / self.VOICE_DIR / f"{safe_name}.json"
        elif category == PresetCategory.AUDIO:
            return self.base_dir / self.AUDIO_DIR / f"{safe_name}.json"
        elif category == PresetCategory.BATCH:
            return self.base_dir / self.BATCH_DIR / f"{safe_name}.json"
        elif category == PresetCategory.PROJECT:
            return self.base_dir / self.PROJECT_DIR / f"{safe_name}.json"
        else:
            return self.base_dir / f"{safe_name}.json"
    
    def _sanitize_filename(self, name: str) -> str:
        """Sanitize tên file"""
        # Remove invalid characters
        invalid = '<>:"/\\|?*'
        result = name
        for char in invalid:
            result = result.replace(char, '_')
        return result[:200]  # Limit length
    
    def _compute_file_hash(self, file_path: str) -> Optional[str]:
        """Tính MD5 hash của file"""
        try:
            with open(file_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return None
    
    # =========================================================================
    # Voice Presets
    # =========================================================================
    
    def save_voice_preset(
        self,
        preset: VoicePreset,
        overwrite: bool = True
    ) -> bool:
        """
        Lưu voice preset
        
        Args:
            preset: Voice preset để lưu
            overwrite: Ghi đè nếu đã tồn tại
            
        Returns:
            True nếu thành công
        """
        path = self._get_preset_path(preset.name, PresetCategory.VOICE)
        
        if path.exists() and not overwrite:
            return False
        
        # Update timestamp
        preset.updated_at = datetime.now().isoformat()
        
        # Compute voice file hash if exists
        if preset.voice_file_path and os.path.exists(preset.voice_file_path):
            preset.voice_file_hash = self._compute_file_hash(preset.voice_file_path)
        
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(preset.to_dict(), f, ensure_ascii=False, indent=2)
            
            # Update cache
            self._voice_presets[preset.name] = preset
            return True
        except Exception as e:
            print(f"Error saving voice preset: {e}")
            return False
    
    def load_voice_preset(self, name: str) -> Optional[VoicePreset]:
        """Load voice preset theo tên"""
        # Check cache first
        if name in self._voice_presets:
            return self._voice_presets[name]
        
        path = self._get_preset_path(name, PresetCategory.VOICE)
        
        if not path.exists():
            return None
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            preset = VoicePreset.from_dict(data)
            self._voice_presets[name] = preset
            return preset
        except Exception as e:
            print(f"Error loading voice preset: {e}")
            return None
    
    def delete_voice_preset(self, name: str) -> bool:
        """Xóa voice preset"""
        path = self._get_preset_path(name, PresetCategory.VOICE)
        
        try:
            if path.exists():
                path.unlink()
            
            if name in self._voice_presets:
                del self._voice_presets[name]
            
            return True
        except Exception as e:
            print(f"Error deleting voice preset: {e}")
            return False
    
    def list_voice_presets(self) -> List[VoicePreset]:
        """Liệt kê tất cả voice presets"""
        return list(self._voice_presets.values())
    
    def search_voice_presets(
        self,
        query: str = "",
        tags: Optional[List[str]] = None
    ) -> List[VoicePreset]:
        """
        Tìm kiếm voice presets
        
        Args:
            query: Từ khóa tìm kiếm trong name/description
            tags: Filter theo tags
        """
        results = []
        query_lower = query.lower()
        
        for preset in self._voice_presets.values():
            # Match query
            if query:
                if query_lower not in preset.name.lower() and \
                   query_lower not in preset.description.lower():
                    continue
            
            # Match tags
            if tags:
                if not any(tag in preset.tags for tag in tags):
                    continue
            
            results.append(preset)
        
        # Sort by use count
        results.sort(key=lambda p: p.use_count, reverse=True)
        
        return results
    
    # =========================================================================
    # Audio Presets
    # =========================================================================
    
    def save_audio_preset(
        self,
        preset: AudioPreset,
        overwrite: bool = True
    ) -> bool:
        """Lưu audio preset"""
        path = self._get_preset_path(preset.name, PresetCategory.AUDIO)
        
        if path.exists() and not overwrite:
            return False
        
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(preset.to_dict(), f, ensure_ascii=False, indent=2)
            
            self._audio_presets[preset.name] = preset
            return True
        except Exception as e:
            print(f"Error saving audio preset: {e}")
            return False
    
    def load_audio_preset(self, name: str) -> Optional[AudioPreset]:
        """Load audio preset theo tên"""
        if name in self._audio_presets:
            return self._audio_presets[name]
        
        path = self._get_preset_path(name, PresetCategory.AUDIO)
        
        if not path.exists():
            return None
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            preset = AudioPreset.from_dict(data)
            self._audio_presets[name] = preset
            return preset
        except Exception as e:
            print(f"Error loading audio preset: {e}")
            return None
    
    def delete_audio_preset(self, name: str) -> bool:
        """Xóa audio preset"""
        path = self._get_preset_path(name, PresetCategory.AUDIO)
        
        try:
            if path.exists():
                path.unlink()
            
            if name in self._audio_presets:
                del self._audio_presets[name]
            
            return True
        except Exception as e:
            print(f"Error deleting audio preset: {e}")
            return False
    
    def list_audio_presets(self) -> List[AudioPreset]:
        """Liệt kê tất cả audio presets"""
        return list(self._audio_presets.values())
    
    # =========================================================================
    # Batch Presets
    # =========================================================================
    
    def save_batch_preset(
        self,
        preset: BatchPreset,
        overwrite: bool = True
    ) -> bool:
        """Lưu batch preset"""
        path = self._get_preset_path(preset.name, PresetCategory.BATCH)
        
        if path.exists() and not overwrite:
            return False
        
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(preset.to_dict(), f, ensure_ascii=False, indent=2)
            
            self._batch_presets[preset.name] = preset
            return True
        except Exception as e:
            print(f"Error saving batch preset: {e}")
            return False
    
    def load_batch_preset(self, name: str) -> Optional[BatchPreset]:
        """Load batch preset"""
        if name in self._batch_presets:
            return self._batch_presets[name]
        
        path = self._get_preset_path(name, PresetCategory.BATCH)
        
        if not path.exists():
            return None
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            preset = BatchPreset.from_dict(data)
            self._batch_presets[name] = preset
            return preset
        except Exception as e:
            print(f"Error loading batch preset: {e}")
            return None
    
    def delete_batch_preset(self, name: str) -> bool:
        """Xóa batch preset"""
        path = self._get_preset_path(name, PresetCategory.BATCH)
        
        try:
            if path.exists():
                path.unlink()
            
            if name in self._batch_presets:
                del self._batch_presets[name]
            
            return True
        except Exception:
            return False
    
    def list_batch_presets(self) -> List[BatchPreset]:
        """Liệt kê batch presets"""
        return list(self._batch_presets.values())
    
    # =========================================================================
    # Project Presets
    # =========================================================================
    
    def save_project_preset(
        self,
        preset: ProjectPreset,
        overwrite: bool = True
    ) -> bool:
        """Lưu project preset"""
        path = self._get_preset_path(preset.name, PresetCategory.PROJECT)
        
        if path.exists() and not overwrite:
            return False
        
        preset.updated_at = datetime.now().isoformat()
        
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(preset.to_dict(), f, ensure_ascii=False, indent=2)
            
            self._project_presets[preset.name] = preset
            return True
        except Exception as e:
            print(f"Error saving project preset: {e}")
            return False
    
    def load_project_preset(self, name: str) -> Optional[ProjectPreset]:
        """Load project preset"""
        if name in self._project_presets:
            return self._project_presets[name]
        
        path = self._get_preset_path(name, PresetCategory.PROJECT)
        
        if not path.exists():
            return None
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            preset = ProjectPreset.from_dict(data)
            self._project_presets[name] = preset
            return preset
        except Exception as e:
            print(f"Error loading project preset: {e}")
            return None
    
    def delete_project_preset(self, name: str) -> bool:
        """Xóa project preset"""
        path = self._get_preset_path(name, PresetCategory.PROJECT)
        
        try:
            if path.exists():
                path.unlink()
            
            if name in self._project_presets:
                del self._project_presets[name]
            
            return True
        except Exception:
            return False
    
    def list_project_presets(self) -> List[ProjectPreset]:
        """Liệt kê project presets"""
        return list(self._project_presets.values())
    
    # =========================================================================
    # Import/Export
    # =========================================================================
    
    def export_preset(
        self,
        name: str,
        category: PresetCategory,
        export_path: str,
        include_voice_file: bool = True
    ) -> bool:
        """
        Export preset ra file riêng
        
        Args:
            name: Tên preset
            category: Loại preset
            export_path: Đường dẫn xuất
            include_voice_file: Đính kèm voice file (cho voice preset)
        """
        # Load preset
        if category == PresetCategory.VOICE:
            preset = self.load_voice_preset(name)
        elif category == PresetCategory.AUDIO:
            preset = self.load_audio_preset(name)
        elif category == PresetCategory.BATCH:
            preset = self.load_batch_preset(name)
        elif category == PresetCategory.PROJECT:
            preset = self.load_project_preset(name)
        else:
            return False
        
        if preset is None:
            return False
        
        export_data = {
            "category": category.value,
            "preset": preset.to_dict(),
            "exported_at": datetime.now().isoformat(),
            "version": "1.0"
        }
        
        try:
            export_path = Path(export_path)
            
            # If including voice file, create a package
            if (category == PresetCategory.VOICE and include_voice_file and
                    hasattr(preset, 'voice_file_path') and preset.voice_file_path):
                
                # Create package directory
                package_dir = export_path.parent / f"{export_path.stem}_package"
                package_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy voice file
                voice_filename = Path(preset.voice_file_path).name
                voice_dest = package_dir / voice_filename
                shutil.copy2(preset.voice_file_path, voice_dest)
                
                export_data["voice_file_included"] = voice_filename
                
                # Save preset json
                with open(package_dir / "preset.json", "w", encoding="utf-8") as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
                
                # Create zip
                shutil.make_archive(str(export_path.with_suffix('')), 'zip', package_dir)
                
                # Cleanup temp directory
                shutil.rmtree(package_dir)
            else:
                # Simple export
                with open(export_path, "w", encoding="utf-8") as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"Error exporting preset: {e}")
            return False
    
    def import_preset(
        self,
        import_path: str,
        overwrite: bool = False
    ) -> Optional[Union[VoicePreset, AudioPreset, BatchPreset, ProjectPreset]]:
        """
        Import preset từ file
        
        Args:
            import_path: Đường dẫn file
            overwrite: Ghi đè nếu đã tồn tại
            
        Returns:
            Preset đã import hoặc None nếu lỗi
        """
        import_path = Path(import_path)
        
        try:
            # Check if it's a zip package
            if import_path.suffix == '.zip':
                import zipfile
                import tempfile
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    with zipfile.ZipFile(import_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    # Load preset.json
                    with open(Path(temp_dir) / "preset.json", "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    # Copy voice file if included
                    if "voice_file_included" in data:
                        voice_filename = data["voice_file_included"]
                        voice_source = Path(temp_dir) / voice_filename
                        
                        # Copy to voices directory
                        voices_dir = self.base_dir / "imported_voices"
                        voices_dir.mkdir(parents=True, exist_ok=True)
                        voice_dest = voices_dir / voice_filename
                        shutil.copy2(voice_source, voice_dest)
                        
                        # Update preset path
                        data["preset"]["voice_file_path"] = str(voice_dest)
            else:
                with open(import_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            
            # Parse and save
            category = PresetCategory(data.get("category", "voice"))
            preset_data = data["preset"]
            
            if category == PresetCategory.VOICE:
                preset = VoicePreset.from_dict(preset_data)
                self.save_voice_preset(preset, overwrite=overwrite)
            elif category == PresetCategory.AUDIO:
                preset = AudioPreset.from_dict(preset_data)
                self.save_audio_preset(preset, overwrite=overwrite)
            elif category == PresetCategory.BATCH:
                preset = BatchPreset.from_dict(preset_data)
                self.save_batch_preset(preset, overwrite=overwrite)
            elif category == PresetCategory.PROJECT:
                preset = ProjectPreset.from_dict(preset_data)
                self.save_project_preset(preset, overwrite=overwrite)
            else:
                return None
            
            return preset
        except Exception as e:
            print(f"Error importing preset: {e}")
            return None
    
    # =========================================================================
    # Utility
    # =========================================================================
    
    def reload_all(self) -> None:
        """Reload tất cả presets từ disk"""
        self._voice_presets.clear()
        self._audio_presets.clear()
        self._batch_presets.clear()
        self._project_presets.clear()
        
        # Load voice presets
        voice_dir = self.base_dir / self.VOICE_DIR
        if voice_dir.exists():
            for file in voice_dir.glob("*.json"):
                try:
                    with open(file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    preset = VoicePreset.from_dict(data)
                    self._voice_presets[preset.name] = preset
                except Exception:
                    pass
        
        # Load audio presets
        audio_dir = self.base_dir / self.AUDIO_DIR
        if audio_dir.exists():
            for file in audio_dir.glob("*.json"):
                try:
                    with open(file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    preset = AudioPreset.from_dict(data)
                    self._audio_presets[preset.name] = preset
                except Exception:
                    pass
        
        # Load batch presets
        batch_dir = self.base_dir / self.BATCH_DIR
        if batch_dir.exists():
            for file in batch_dir.glob("*.json"):
                try:
                    with open(file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    preset = BatchPreset.from_dict(data)
                    self._batch_presets[preset.name] = preset
                except Exception:
                    pass
        
        # Load project presets
        project_dir = self.base_dir / self.PROJECT_DIR
        if project_dir.exists():
            for file in project_dir.glob("*.json"):
                try:
                    with open(file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    preset = ProjectPreset.from_dict(data)
                    self._project_presets[preset.name] = preset
                except Exception:
                    pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Lấy thống kê presets"""
        return {
            "total_voice_presets": len(self._voice_presets),
            "total_audio_presets": len(self._audio_presets),
            "total_batch_presets": len(self._batch_presets),
            "total_project_presets": len(self._project_presets),
            "most_used_voice_presets": sorted(
                self._voice_presets.values(),
                key=lambda p: p.use_count,
                reverse=True
            )[:5]
        }
    
    def create_default_presets(self) -> None:
        """Tạo các preset mặc định"""
        # Default audio presets
        defaults = [
            AudioPreset(
                name="Clean Voice",
                description="Tiếng nói sạch, không xử lý nhiều",
                normalize_enabled=True,
                normalize_target_db=-3.0,
                limiter_enabled=True
            ),
            AudioPreset(
                name="Podcast",
                description="Tối ưu cho podcast và voiceover",
                normalize_enabled=True,
                normalize_target_db=-16.0,
                normalize_type="lufs",
                compression_enabled=True,
                compression_threshold=-24.0,
                compression_ratio=3.0,
                limiter_enabled=True,
                limiter_threshold=-1.0,
                deesser_enabled=True
            ),
            AudioPreset(
                name="YouTube",
                description="Tối ưu cho video YouTube",
                normalize_enabled=True,
                normalize_target_db=-14.0,
                normalize_type="lufs",
                compression_enabled=True,
                compression_threshold=-20.0,
                compression_ratio=4.0,
                eq_enabled=True,
                eq_low_gain=-2.0,
                eq_mid_gain=1.0,
                eq_high_gain=2.0
            ),
            AudioPreset(
                name="Audiobook",
                description="Tối ưu cho sách nói",
                normalize_enabled=True,
                normalize_target_db=-20.0,
                normalize_type="rms",
                noise_reduction_enabled=True,
                noise_reduction_strength=0.3,
                compression_enabled=True,
                compression_threshold=-26.0,
                compression_ratio=2.5
            ),
            AudioPreset(
                name="Broadcast",
                description="Chuẩn phát sóng broadcast",
                normalize_enabled=True,
                normalize_target_db=-23.0,
                normalize_type="lufs",
                compression_enabled=True,
                compression_threshold=-18.0,
                compression_ratio=6.0,
                limiter_enabled=True,
                limiter_threshold=-2.0,
                eq_enabled=True,
                eq_low_freq=80.0,
                eq_low_gain=-3.0
            )
        ]
        
        for preset in defaults:
            if preset.name not in self._audio_presets:
                self.save_audio_preset(preset)
        
        # Default batch presets
        batch_defaults = [
            BatchPreset(
                name="Standard SRT",
                description="Xử lý file SRT chuẩn",
                output_format="wav",
                naming_pattern="{index}",
                index_padding=4,
                parallel_jobs=2
            ),
            BatchPreset(
                name="Quick Export",
                description="Xuất nhanh với cài đặt mặc định",
                output_format="wav",
                naming_pattern="{index}_{text}",
                index_padding=3,
                parallel_jobs=4
            )
        ]
        
        for preset in batch_defaults:
            if preset.name not in self._batch_presets:
                self.save_batch_preset(preset)


# Singleton instance
_preset_manager: Optional[PresetManager] = None


def get_preset_manager(base_dir: Optional[str] = None) -> PresetManager:
    """Get singleton preset manager instance"""
    global _preset_manager
    
    if _preset_manager is None:
        _preset_manager = PresetManager(base_dir)
        _preset_manager.create_default_presets()
    
    return _preset_manager
