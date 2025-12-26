"""
SRT (SubRip Subtitle) processor for VietTTS
Handles parsing and batch processing of subtitle files
"""

import re
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger

from app.exceptions import SRTParseError


@dataclass
class Subtitle:
    """Represents a single subtitle entry"""
    index: int
    start_time: str
    end_time: str
    text: str
    
    @property
    def duration_ms(self) -> int:
        """Calculate duration in milliseconds"""
        start = self._time_to_ms(self.start_time)
        end = self._time_to_ms(self.end_time)
        return end - start
    
    @staticmethod
    def _time_to_ms(time_str: str) -> int:
        """Convert SRT timestamp to milliseconds"""
        # Format: 00:00:00,000 or 00:00:00.000
        time_str = time_str.replace(',', '.')
        parts = time_str.split(':')
        
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds_parts = parts[2].split('.')
        seconds = int(seconds_parts[0])
        milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
        
        return (hours * 3600 + minutes * 60 + seconds) * 1000 + milliseconds


class SRTParser:
    """
    Parser for SRT subtitle files
    """
    
    # Regex patterns
    INDEX_PATTERN = re.compile(r'^\d+$')
    TIME_PATTERN = re.compile(
        r'(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})'
    )
    
    def __init__(self):
        self.subtitles: List[Subtitle] = []
    
    def parse_file(self, filepath: str) -> List[Subtitle]:
        """
        Parse SRT file and return list of subtitles
        
        Args:
            filepath: Path to SRT file
            
        Returns:
            List[Subtitle]: Parsed subtitles
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise SRTParseError(f"File not found: {filepath}")
        
        if filepath.suffix.lower() != '.srt':
            raise SRTParseError(f"Invalid file type: {filepath.suffix}")
        
        # Try different encodings
        content = None
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            raise SRTParseError("Failed to read file with supported encodings")
        
        return self.parse_content(content)
    
    def parse_content(self, content: str) -> List[Subtitle]:
        """
        Parse SRT content string
        
        Args:
            content: SRT file content
            
        Returns:
            List[Subtitle]: Parsed subtitles
        """
        self.subtitles = []
        
        # Split by double newlines (subtitle blocks)
        blocks = re.split(r'\n\s*\n', content.strip())
        
        for block in blocks:
            if not block.strip():
                continue
            
            try:
                subtitle = self._parse_block(block)
                if subtitle:
                    self.subtitles.append(subtitle)
            except Exception as e:
                logger.warning(f"Failed to parse subtitle block: {e}")
                continue
        
        logger.info(f"Parsed {len(self.subtitles)} subtitles")
        return self.subtitles
    
    def _parse_block(self, block: str) -> Optional[Subtitle]:
        """Parse a single subtitle block"""
        lines = block.strip().split('\n')
        
        if len(lines) < 3:
            return None
        
        # Line 1: Index
        index_line = lines[0].strip()
        if not self.INDEX_PATTERN.match(index_line):
            return None
        index = int(index_line)
        
        # Line 2: Timestamp
        time_match = self.TIME_PATTERN.match(lines[1].strip())
        if not time_match:
            return None
        start_time = time_match.group(1)
        end_time = time_match.group(2)
        
        # Lines 3+: Text (may be multiple lines)
        text_lines = lines[2:]
        text = ' '.join(line.strip() for line in text_lines if line.strip())
        
        # Clean text
        text = self._clean_text(text)
        
        return Subtitle(
            index=index,
            start_time=start_time,
            end_time=end_time,
            text=text
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean subtitle text"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove formatting tags like {\\an8}
        text = re.sub(r'\{[^}]+\}', '', text)
        
        # Remove speaker labels like "- Speaker:"
        text = re.sub(r'^-\s*', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def get_texts(self) -> List[str]:
        """Get list of subtitle texts only"""
        return [sub.text for sub in self.subtitles]
    
    def get_subtitle_count(self) -> int:
        """Get number of subtitles"""
        return len(self.subtitles)
    
    def get_total_duration_ms(self) -> int:
        """Get total duration in milliseconds"""
        if not self.subtitles:
            return 0
        
        last = self.subtitles[-1]
        return Subtitle._time_to_ms(last.end_time)


class SRTProcessor:
    """
    Processor for batch TTS generation from SRT files
    """
    
    def __init__(self, tts_engine):
        """
        Initialize SRT Processor
        
        Args:
            tts_engine: TTSEngine instance
        """
        self.tts_engine = tts_engine
        self.parser = SRTParser()
        self.subtitles = []
    
    def load_srt(self, filepath: str) -> int:
        """
        Load and parse SRT file
        
        Args:
            filepath: Path to SRT file
            
        Returns:
            int: Number of subtitles loaded
        """
        self.subtitles = self.parser.parse_file(filepath)
        return len(self.subtitles)
    
    def get_preview(self, count: int = 5) -> List[Tuple[int, str]]:
        """
        Get preview of first N subtitles
        
        Args:
            count: Number of subtitles to preview
            
        Returns:
            List of (index, text) tuples
        """
        return [
            (sub.index, sub.text[:100] + '...' if len(sub.text) > 100 else sub.text)
            for sub in self.subtitles[:count]
        ]
    
    def process(
        self,
        voice_name: str,
        output_dir: str,
        speed: float = 1.0,
        start_index: int = 1,
        end_index: Optional[int] = None,
        progress_callback=None,
        cancel_flag=None
    ) -> List[str]:
        """
        Process subtitles and generate audio files
        
        Args:
            voice_name: Voice name/key for TTSEngine lookup
            output_dir: Directory for output files
            speed: Speech speed
            start_index: Start from this subtitle index
            end_index: End at this subtitle index (inclusive)
            progress_callback: Callback(current, total, status)
            cancel_flag: Callable that returns True to cancel
            
        Returns:
            List of generated file paths
        """
        if not self.subtitles:
            raise SRTParseError("No subtitles loaded. Call load_srt() first.")
        
        # Check if TTS model is loaded
        if not self.tts_engine.model_loaded:
            raise SRTParseError("TTS model not loaded. Please wait for model to load.")
        
        # Filter subtitles by index range
        subs_to_process = [
            sub for sub in self.subtitles
            if sub.index >= start_index and (end_index is None or sub.index <= end_index)
        ]
        
        if not subs_to_process:
            logger.warning("No subtitles in specified range")
            return []
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build texts list for batch processing
        texts = [(sub.index, sub.text) for sub in subs_to_process if sub.text.strip()]
        
        # Use TTSEngine's batch synthesis
        generated_files = self.tts_engine.synthesize_batch(
            texts=texts,
            voice_name=voice_name,
            output_dir=str(output_dir),
            speed=speed,
            progress_callback=progress_callback,
            cancel_flag=cancel_flag
        )
        
        logger.success(f"Generated {len(generated_files)} audio files")
        return generated_files


def validate_srt_file(filepath: str) -> Tuple[bool, str, int]:
    """
    Validate SRT file
    
    Args:
        filepath: Path to SRT file
        
    Returns:
        Tuple of (is_valid, message, subtitle_count)
    """
    try:
        parser = SRTParser()
        subtitles = parser.parse_file(filepath)
        
        if not subtitles:
            return False, "No valid subtitles found in file", 0
        
        return True, f"Valid SRT file with {len(subtitles)} subtitles", len(subtitles)
        
    except SRTParseError as e:
        return False, str(e), 0
    except Exception as e:
        return False, f"Error validating file: {str(e)}", 0
