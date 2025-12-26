"""
Unit tests for SRT Processor module.
Tests subtitle parsing and batch processing.
"""

import pytest
from pathlib import Path
import tempfile
import os


class TestSubtitle:
    """Tests for Subtitle dataclass."""
    
    def test_duration_ms_calculation(self):
        """Test duration_ms calculates correctly."""
        from app.srt_processor import Subtitle
        
        sub = Subtitle(
            index=1,
            start_time="00:00:01,000",
            end_time="00:00:05,500",
            text="Test"
        )
        
        assert sub.duration_ms == 4500  # 5.5s - 1s = 4.5s = 4500ms
    
    def test_time_to_ms_with_comma(self):
        """Test time parsing with comma separator."""
        from app.srt_processor import Subtitle
        
        ms = Subtitle._time_to_ms("01:23:45,678")
        
        # 1*3600 + 23*60 + 45 = 5025 seconds = 5025000ms + 678 = 5025678
        assert ms == 5025678
    
    def test_time_to_ms_with_dot(self):
        """Test time parsing with dot separator."""
        from app.srt_processor import Subtitle
        
        ms = Subtitle._time_to_ms("00:00:10.500")
        
        assert ms == 10500


class TestSRTParser:
    """Tests for SRTParser class."""
    
    def test_parser_init(self):
        """Test parser initialization."""
        from app.srt_processor import SRTParser
        
        parser = SRTParser()
        
        assert parser.subtitles == []
    
    def test_parse_content_single_subtitle(self):
        """Test parsing content with single subtitle."""
        from app.srt_processor import SRTParser
        
        content = """1
00:00:01,000 --> 00:00:05,000
Hello world"""
        
        parser = SRTParser()
        subtitles = parser.parse_content(content)
        
        assert len(subtitles) == 1
        assert subtitles[0].index == 1
        assert subtitles[0].text == "Hello world"
    
    def test_parse_content_multiple_subtitles(self):
        """Test parsing content with multiple subtitles."""
        from app.srt_processor import SRTParser
        
        content = """1
00:00:01,000 --> 00:00:05,000
First subtitle

2
00:00:06,000 --> 00:00:10,000
Second subtitle"""
        
        parser = SRTParser()
        subtitles = parser.parse_content(content)
        
        assert len(subtitles) == 2
        assert subtitles[0].text == "First subtitle"
        assert subtitles[1].text == "Second subtitle"
    
    def test_parse_content_multiline_text(self):
        """Test parsing subtitle with multiple text lines."""
        from app.srt_processor import SRTParser
        
        content = """1
00:00:01,000 --> 00:00:05,000
Line one
Line two"""
        
        parser = SRTParser()
        subtitles = parser.parse_content(content)
        
        assert len(subtitles) == 1
        assert subtitles[0].text == "Line one Line two"
    
    def test_parse_content_removes_html_tags(self):
        """Test parsing removes HTML tags."""
        from app.srt_processor import SRTParser
        
        content = """1
00:00:01,000 --> 00:00:05,000
<i>Italic text</i> and <b>bold</b>"""
        
        parser = SRTParser()
        subtitles = parser.parse_content(content)
        
        assert subtitles[0].text == "Italic text and bold"
    
    def test_parse_content_removes_formatting_tags(self):
        """Test parsing removes formatting tags like {\\an8}."""
        from app.srt_processor import SRTParser
        
        content = """1
00:00:01,000 --> 00:00:05,000
{\\an8}Centered text"""
        
        parser = SRTParser()
        subtitles = parser.parse_content(content)
        
        assert subtitles[0].text == "Centered text"
    
    def test_parse_file_nonexistent(self):
        """Test parse_file raises error for nonexistent file."""
        from app.srt_processor import SRTParser, SRTParseError
        
        parser = SRTParser()
        
        with pytest.raises(SRTParseError):
            parser.parse_file("/nonexistent/file.srt")
    
    def test_parse_file_wrong_extension(self):
        """Test parse_file raises error for wrong extension."""
        from app.srt_processor import SRTParser, SRTParseError
        
        parser = SRTParser()
        
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"test")
            temp_path = f.name
        
        try:
            with pytest.raises(SRTParseError):
                parser.parse_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_parse_file_valid_srt(self):
        """Test parse_file with valid SRT file."""
        from app.srt_processor import SRTParser
        
        content = """1
00:00:01,000 --> 00:00:05,000
Test subtitle"""
        
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False, mode='w', encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name
        
        try:
            parser = SRTParser()
            subtitles = parser.parse_file(temp_path)
            
            assert len(subtitles) == 1
            assert subtitles[0].text == "Test subtitle"
        finally:
            os.unlink(temp_path)
    
    def test_get_texts(self):
        """Test get_texts returns list of text only."""
        from app.srt_processor import SRTParser
        
        content = """1
00:00:01,000 --> 00:00:05,000
First

2
00:00:06,000 --> 00:00:10,000
Second"""
        
        parser = SRTParser()
        parser.parse_content(content)
        texts = parser.get_texts()
        
        assert texts == ["First", "Second"]
    
    def test_get_subtitle_count(self):
        """Test get_subtitle_count returns correct count."""
        from app.srt_processor import SRTParser
        
        content = """1
00:00:01,000 --> 00:00:05,000
Sub 1

2
00:00:06,000 --> 00:00:10,000
Sub 2

3
00:00:11,000 --> 00:00:15,000
Sub 3"""
        
        parser = SRTParser()
        parser.parse_content(content)
        
        assert parser.get_subtitle_count() == 3
    
    def test_get_total_duration_ms(self):
        """Test get_total_duration_ms returns end time of last subtitle."""
        from app.srt_processor import SRTParser
        
        content = """1
00:00:01,000 --> 00:00:05,000
First

2
00:00:06,000 --> 00:01:00,500
Last"""
        
        parser = SRTParser()
        parser.parse_content(content)
        
        assert parser.get_total_duration_ms() == 60500  # 1 minute + 0.5 seconds


class TestValidateSRTFile:
    """Tests for validate_srt_file function."""
    
    def test_validate_nonexistent_file(self):
        """Test validation of nonexistent file."""
        from app.srt_processor import validate_srt_file
        
        is_valid, message, count = validate_srt_file("/nonexistent/file.srt")
        
        assert is_valid is False
        assert count == 0
    
    def test_validate_valid_file(self):
        """Test validation of valid SRT file."""
        from app.srt_processor import validate_srt_file
        
        content = """1
00:00:01,000 --> 00:00:05,000
Valid subtitle"""
        
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False, mode='w', encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name
        
        try:
            is_valid, message, count = validate_srt_file(temp_path)
            
            assert is_valid is True
            assert count == 1
        finally:
            os.unlink(temp_path)
    
    def test_validate_empty_file(self):
        """Test validation of empty SRT file."""
        from app.srt_processor import validate_srt_file
        
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False, mode='w', encoding='utf-8') as f:
            f.write("")
            temp_path = f.name
        
        try:
            is_valid, message, count = validate_srt_file(temp_path)
            
            assert is_valid is False
            assert count == 0
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
