"""
Reference Text Cache for VietTTS Voice Cloning
Stores user-provided ref_text per audio file using hash-based filenames
"""

import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from loguru import logger

from app.config import Config


class RefTextCache:
    """
    Cache system for storing reference text per audio file.
    
    Uses MD5 hash of absolute file path as cache key to ensure
    unique storage per file regardless of filename.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize cache.
        
        Args:
            cache_dir: Directory for cache files. Defaults to app cache dir.
        """
        self.cache_dir = cache_dir or (Config.APP_DIR / "cache" / "ref_texts")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"RefTextCache initialized at {self.cache_dir}")
    
    def _get_hash(self, file_path: str) -> str:
        """Generate MD5 hash of absolute file path."""
        abs_path = str(Path(file_path).resolve())
        return hashlib.md5(abs_path.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, file_path: str) -> Path:
        """Get cache file path for given audio file."""
        file_hash = self._get_hash(file_path)
        return self.cache_dir / f"{file_hash}.json"
    
    def has(self, file_path: str) -> bool:
        """
        Check if ref_text exists in cache for given file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            bool: True if cached ref_text exists
        """
        cache_path = self._get_cache_path(file_path)
        return cache_path.exists()
    
    def get(self, file_path: str) -> Optional[str]:
        """
        Get cached ref_text for given audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            str or None: Cached ref_text if exists, None otherwise
        """
        cache_path = self._get_cache_path(file_path)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            ref_text = data.get("ref_text", "")
            logger.debug(f"Loaded cached ref_text for {Path(file_path).name}")
            return ref_text
            
        except Exception as e:
            logger.warning(f"Failed to read cache: {e}")
            return None
    
    def save(self, file_path: str, ref_text: str) -> bool:
        """
        Save ref_text to cache for given audio file.
        
        Args:
            file_path: Path to audio file
            ref_text: Reference text to cache
            
        Returns:
            bool: True if saved successfully
        """
        if not ref_text or not ref_text.strip():
            return False
        
        cache_path = self._get_cache_path(file_path)
        
        try:
            data = {
                "ref_text": ref_text.strip(),
                "file_path": str(Path(file_path).resolve()),
                "file_name": Path(file_path).name,
                "created": datetime.now().isoformat(),
                "hash": self._get_hash(file_path)
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Cached ref_text for {Path(file_path).name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            return False
    
    def delete(self, file_path: str) -> bool:
        """
        Delete cached ref_text for given audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            bool: True if deleted successfully
        """
        cache_path = self._get_cache_path(file_path)
        
        if cache_path.exists():
            try:
                cache_path.unlink()
                logger.debug(f"Deleted cache for {Path(file_path).name}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete cache: {e}")
                return False
        return False
    
    def clear_all(self) -> int:
        """
        Clear all cached ref_texts.
        
        Returns:
            int: Number of cache files deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                count += 1
            except Exception:
                pass
        
        logger.info(f"Cleared {count} cached ref_texts")
        return count
    
    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all cached entries.
        
        Returns:
            Dict mapping file names to cache data
        """
        entries = {}
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                file_name = data.get("file_name", cache_file.stem)
                entries[file_name] = data
            except Exception:
                pass
        
        return entries


# Global cache instance
_ref_text_cache: Optional[RefTextCache] = None


def get_ref_text_cache() -> RefTextCache:
    """Get global RefTextCache instance."""
    global _ref_text_cache
    if _ref_text_cache is None:
        _ref_text_cache = RefTextCache()
    return _ref_text_cache
