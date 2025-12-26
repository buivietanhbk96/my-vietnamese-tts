"""
Voice samples downloader for VietTTS
Downloads sample voices from the viet-tts repository
"""

import os
import requests
from pathlib import Path
from typing import List, Callable, Optional
from loguru import logger

from app.config import Config


class VoiceDownloader:
    """
    Downloads voice samples from viet-tts GitHub repository
    """
    
    # Known voice samples from viet-tts repo
    VOICE_SAMPLES = [
        "cdteam.wav",
        "diep-chi.wav",
        "doremon.wav",
        "jack.wav",
        "mck.wav",
        "nu-nhe-nhang.wav",
        "nu-phia-bac.wav",
        "nu-phia-nam.wav",
        "son-tung-mtp.wav",
        "nam-phia-bac.wav",
        "nam-phia-nam.wav",
        "toc-tien.wav",
    ]
    
    BASE_URL = "https://raw.githubusercontent.com/dangvansam/viet-tts/main/samples/"
    
    def __init__(self, samples_dir: Optional[Path] = None):
        """
        Initialize voice downloader
        
        Args:
            samples_dir: Directory to save voice samples
        """
        self.samples_dir = samples_dir or Config.SAMPLES_DIR
        self.samples_dir.mkdir(parents=True, exist_ok=True)
    
    def get_local_voices(self) -> List[str]:
        """
        Get list of locally available voice samples
        
        Returns:
            List of voice filenames
        """
        voices = []
        for ext in ['*.wav', '*.mp3']:
            voices.extend([f.name for f in self.samples_dir.glob(ext)])
        return voices
    
    def get_missing_voices(self) -> List[str]:
        """
        Get list of voice samples not yet downloaded
        
        Returns:
            List of missing voice filenames
        """
        local_voices = set(self.get_local_voices())
        return [v for v in self.VOICE_SAMPLES if v not in local_voices]
    
    def download_voice(
        self,
        filename: str,
        progress_callback: Optional[Callable] = None
    ) -> bool:
        """
        Download a single voice sample
        
        Args:
            filename: Voice filename to download
            progress_callback: Optional callback(bytes_downloaded, total_bytes)
            
        Returns:
            bool: True if download successful
        """
        url = self.BASE_URL + filename
        output_path = self.samples_dir / filename
        
        try:
            logger.info(f"Downloading: {filename}")
            
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback and total_size > 0:
                            progress_callback(downloaded, total_size)
            
            logger.success(f"Downloaded: {filename}")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to download {filename}: {e}")
            # Clean up partial download
            if output_path.exists():
                output_path.unlink()
            return False
        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            return False
    
    def download_all(
        self,
        progress_callback: Optional[Callable] = None,
        cancel_flag: Optional[Callable] = None
    ) -> tuple:
        """
        Download all missing voice samples
        
        Args:
            progress_callback: Callback(current_idx, total, filename, status)
            cancel_flag: Callable that returns True to cancel
            
        Returns:
            Tuple of (success_count, failed_count, failed_list)
        """
        missing = self.get_missing_voices()
        
        if not missing:
            logger.info("All voice samples already downloaded")
            return (0, 0, [])
        
        success_count = 0
        failed_count = 0
        failed_list = []
        
        total = len(missing)
        
        for idx, filename in enumerate(missing, start=1):
            # Check cancellation
            if cancel_flag and cancel_flag():
                logger.info("Download cancelled")
                break
            
            # Progress update
            if progress_callback:
                progress_callback(idx, total, filename, "Downloading...")
            
            # Download
            if self.download_voice(filename):
                success_count += 1
            else:
                failed_count += 1
                failed_list.append(filename)
        
        logger.info(f"Download complete: {success_count} success, {failed_count} failed")
        return (success_count, failed_count, failed_list)
    
    def verify_voice(self, filename: str) -> bool:
        """
        Verify that a voice file is valid
        
        Args:
            filename: Voice filename to verify
            
        Returns:
            bool: True if file is valid
        """
        filepath = self.samples_dir / filename
        
        if not filepath.exists():
            return False
        
        # Check file size (should be at least 10KB for valid audio)
        if filepath.stat().st_size < 10240:
            return False
        
        # Try to load with soundfile
        try:
            import soundfile as sf
            info = sf.info(str(filepath))
            return info.duration > 0
        except Exception:
            return False
    
    def verify_all(self) -> tuple:
        """
        Verify all downloaded voice samples
        
        Returns:
            Tuple of (valid_count, invalid_list)
        """
        local_voices = self.get_local_voices()
        invalid = []
        
        for voice in local_voices:
            if not self.verify_voice(voice):
                invalid.append(voice)
        
        valid_count = len(local_voices) - len(invalid)
        return (valid_count, invalid)
    
    def cleanup_invalid(self):
        """Remove invalid voice files"""
        _, invalid = self.verify_all()
        
        for filename in invalid:
            filepath = self.samples_dir / filename
            if filepath.exists():
                filepath.unlink()
                logger.info(f"Removed invalid: {filename}")


def download_voices_if_needed(
    progress_callback: Optional[Callable] = None
) -> bool:
    """
    Convenience function to download voices if needed
    
    Args:
        progress_callback: Progress callback
        
    Returns:
        bool: True if voices are available
    """
    downloader = VoiceDownloader()
    
    # Check if we have any voices
    local_voices = downloader.get_local_voices()
    
    if local_voices:
        logger.info(f"Found {len(local_voices)} voice samples")
        return True
    
    # Download all
    logger.info("No voice samples found, downloading...")
    success, failed, _ = downloader.download_all(progress_callback)
    
    return success > 0
