"""
Smart Text Processor for VietTTS PRO MAX
Intelligently processes Vietnamese text for optimal TTS output
"""

import re
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class TextEmotion(Enum):
    """Text emotion/style hints"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    EXCITED = "excited"
    CALM = "calm"
    PROFESSIONAL = "professional"
    STORYTELLING = "storytelling"


@dataclass
class ProcessedSentence:
    """A processed sentence with metadata"""
    text: str
    original: str
    emotion: TextEmotion
    pause_before: float  # seconds
    pause_after: float   # seconds
    speed_modifier: float  # 1.0 = normal
    emphasis_words: List[str]


class VietnameseTextProcessor:
    """
    Intelligent Vietnamese text processor for TTS
    Handles numbers, dates, abbreviations, and more
    """
    
    # Vietnamese number words
    DIGITS = {
        '0': 'không', '1': 'một', '2': 'hai', '3': 'ba', '4': 'bốn',
        '5': 'năm', '6': 'sáu', '7': 'bảy', '8': 'tám', '9': 'chín'
    }
    
    TEENS = {
        '10': 'mười', '11': 'mười một', '12': 'mười hai', '13': 'mười ba',
        '14': 'mười bốn', '15': 'mười lăm', '16': 'mười sáu',
        '17': 'mười bảy', '18': 'mười tám', '19': 'mười chín'
    }
    
    TENS = {
        '2': 'hai mươi', '3': 'ba mươi', '4': 'bốn mươi', '5': 'năm mươi',
        '6': 'sáu mươi', '7': 'bảy mươi', '8': 'tám mươi', '9': 'chín mươi'
    }
    
    UNITS = ['', 'nghìn', 'triệu', 'tỷ', 'nghìn tỷ', 'triệu tỷ']
    
    # Common Vietnamese abbreviations
    ABBREVIATIONS = {
        # Locations
        'tp.': 'thành phố',
        'tp.hcm': 'thành phố hồ chí minh',
        'hcm': 'hồ chí minh',
        'hn': 'hà nội',
        'đn': 'đà nẵng',
        'q.': 'quận',
        'p.': 'phường',
        'tx.': 'thị xã',
        'tt.': 'thị trấn',
        
        # Titles
        'ths.': 'thạc sĩ',
        'ts.': 'tiến sĩ',
        'pgs.': 'phó giáo sư',
        'gs.': 'giáo sư',
        'ks.': 'kỹ sư',
        'cn.': 'cử nhân',
        'bs.': 'bác sĩ',
        'ds.': 'dược sĩ',
        
        # Common
        'vnd': 'việt nam đồng',
        'usd': 'đô la mỹ',
        'eur': 'ơ rô',
        'vs': 'với',
        'etc': 'vân vân',
        'ok': 'ô kê',
        
        # Organizations
        'ubnd': 'ủy ban nhân dân',
        'hđnd': 'hội đồng nhân dân',
        'bch': 'ban chấp hành',
        'tw': 'trung ương',
        
        # Units
        'km': 'ki lô mét',
        'km/h': 'ki lô mét trên giờ',
        'm': 'mét',
        'cm': 'xen ti mét',
        'mm': 'mi li mét',
        'kg': 'ki lô gam',
        'g': 'gam',
        'mg': 'mi li gam',
        'l': 'lít',
        'ml': 'mi li lít',
        
        # Time
        'h': 'giờ',
        'ph': 'phút',
        's': 'giây',
    }
    
    # Emoji to Vietnamese text
    EMOJI_MAP = {
        '😀': 'mặt cười',
        '😃': 'mặt cười tươi',
        '😄': 'mặt cười toe toét',
        '😁': 'mặt cười nhe răng',
        '😊': 'mặt cười dịu dàng',
        '😍': 'mặt yêu thích',
        '🥰': 'mặt tình cảm',
        '😘': 'mặt hôn gió',
        '😢': 'mặt khóc',
        '😭': 'mặt khóc lớn',
        '😡': 'mặt giận dữ',
        '😠': 'mặt tức giận',
        '🤔': 'mặt suy nghĩ',
        '😱': 'mặt hoảng sợ',
        '😎': 'mặt ngầu',
        '🤗': 'mặt ôm',
        '👍': 'thích',
        '👎': 'không thích',
        '❤️': 'trái tim',
        '💔': 'trái tim vỡ',
        '🔥': 'lửa',
        '⭐': 'ngôi sao',
        '🎉': 'tiệc tùng',
        '🎊': 'lễ hội',
        '✅': 'đã xong',
        '❌': 'sai',
        '⚠️': 'cảnh báo',
        '📌': 'ghim',
        '📍': 'vị trí',
        '🔴': 'chấm đỏ',
        '🟢': 'chấm xanh',
        '🔵': 'chấm xanh dương',
    }
    
    # Emotion keywords for detection
    EMOTION_KEYWORDS = {
        TextEmotion.HAPPY: ['vui', 'hạnh phúc', 'tuyệt vời', 'xuất sắc', 'yêu', 'thích', 'tốt'],
        TextEmotion.SAD: ['buồn', 'đau', 'khóc', 'mất', 'chia tay', 'thương', 'tiếc'],
        TextEmotion.EXCITED: ['wow', 'ôi', 'tuyệt', 'quá', 'siêu', 'cực', 'khủng'],
        TextEmotion.CALM: ['bình yên', 'nhẹ nhàng', 'thư giãn', 'yên tĩnh'],
        TextEmotion.PROFESSIONAL: ['kính', 'thưa', 'trân trọng', 'báo cáo', 'thông báo'],
    }
    
    def __init__(self):
        """Initialize text processor"""
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency"""
        # Number patterns
        self.number_pattern = re.compile(r'\b(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?)\b')
        self.decimal_pattern = re.compile(r'\b(\d+)[.,](\d+)\b')
        self.percentage_pattern = re.compile(r'(\d+(?:[.,]\d+)?)\s*%')
        
        # Date patterns
        self.date_pattern = re.compile(
            r'\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b'
        )
        self.time_pattern = re.compile(
            r'\b(\d{1,2})[:\.](\d{2})(?:[:\.](\d{2}))?\b'
        )
        
        # Currency patterns
        self.currency_pattern = re.compile(
            r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?)\s*(vnđ|vnd|đ|đồng|usd|\$|eur|€)',
            re.IGNORECASE
        )
        
        # Phone pattern
        self.phone_pattern = re.compile(
            r'\b(0\d{2,3})[\s\-.]?(\d{3,4})[\s\-.]?(\d{3,4})\b'
        )
        
        # URL pattern
        self.url_pattern = re.compile(
            r'https?://[^\s<>"{}|\\^`\[\]]+'
        )
        
        # Email pattern
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
    
    def process(self, text: str) -> str:
        """
        Process text for TTS synthesis
        
        Args:
            text: Input text
            
        Returns:
            Processed text optimized for TTS
        """
        if not text:
            return ""
        
        # Store original
        original = text
        
        # Apply processing in order
        text = self._normalize_whitespace(text)
        text = self._process_urls(text)
        text = self._process_emails(text)
        text = self._process_emojis(text)
        text = self._process_abbreviations(text)
        text = self._process_currency(text)
        text = self._process_percentages(text)
        text = self._process_dates(text)
        text = self._process_times(text)
        text = self._process_phone_numbers(text)
        text = self._process_numbers(text)
        text = self._process_special_chars(text)
        text = self._normalize_punctuation(text)
        text = self._final_cleanup(text)
        
        logger.debug(f"Processed: '{original[:50]}...' -> '{text[:50]}...'")
        return text
    
    def process_sentences(self, text: str) -> List[ProcessedSentence]:
        """
        Process text into sentences with metadata
        
        Args:
            text: Input text
            
        Returns:
            List of ProcessedSentence objects
        """
        # Split into sentences
        sentences = self._split_sentences(text)
        
        result = []
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            processed_text = self.process(sentence)
            emotion = self._detect_emotion(sentence)
            
            result.append(ProcessedSentence(
                text=processed_text,
                original=sentence,
                emotion=emotion,
                pause_before=self._calculate_pause_before(sentence),
                pause_after=self._calculate_pause_after(sentence),
                speed_modifier=self._calculate_speed_modifier(sentence, emotion),
                emphasis_words=self._find_emphasis_words(sentence)
            ))
        
        return result
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace"""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _process_urls(self, text: str) -> str:
        """Process URLs"""
        def url_to_text(match):
            url = match.group(0)
            # Extract domain
            domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
            if domain_match:
                domain = domain_match.group(1)
                return f"đường dẫn {domain}"
            return "đường dẫn"
        
        return self.url_pattern.sub(url_to_text, text)
    
    def _process_emails(self, text: str) -> str:
        """Process email addresses"""
        def email_to_text(match):
            email = match.group(0)
            parts = email.split('@')
            if len(parts) == 2:
                username = ' '.join(parts[0])  # Spell out
                domain = parts[1].replace('.', ' chấm ')
                return f"{username} a còng {domain}"
            return email
        
        return self.email_pattern.sub(email_to_text, text)
    
    def _process_emojis(self, text: str) -> str:
        """Convert emojis to text"""
        for emoji, vietnamese in self.EMOJI_MAP.items():
            text = text.replace(emoji, f' {vietnamese} ')
        return text
    
    def _process_abbreviations(self, text: str) -> str:
        """Expand abbreviations"""
        # Sort by length (longer first) to avoid partial matches
        sorted_abbrevs = sorted(
            self.ABBREVIATIONS.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        
        for abbrev, expansion in sorted_abbrevs:
            # Case insensitive replacement with word boundary
            pattern = re.compile(
                r'\b' + re.escape(abbrev) + r'\b',
                re.IGNORECASE
            )
            text = pattern.sub(expansion, text)
        
        return text
    
    def _process_currency(self, text: str) -> str:
        """Process currency amounts"""
        def currency_to_text(match):
            amount = match.group(1).replace('.', '').replace(',', '.')
            currency = match.group(2).lower()
            
            # Convert number
            try:
                num = float(amount)
                num_text = self._number_to_vietnamese(num)
            except:
                num_text = amount
            
            # Currency name
            currency_names = {
                'vnđ': 'việt nam đồng',
                'vnd': 'việt nam đồng',
                'đ': 'đồng',
                'đồng': 'đồng',
                'usd': 'đô la mỹ',
                '$': 'đô la',
                'eur': 'ơ rô',
                '€': 'ơ rô'
            }
            
            currency_name = currency_names.get(currency, currency)
            return f"{num_text} {currency_name}"
        
        return self.currency_pattern.sub(currency_to_text, text)
    
    def _process_percentages(self, text: str) -> str:
        """Process percentages"""
        def percent_to_text(match):
            number = match.group(1).replace(',', '.')
            try:
                num = float(number)
                num_text = self._number_to_vietnamese(num)
            except:
                num_text = number
            return f"{num_text} phần trăm"
        
        return self.percentage_pattern.sub(percent_to_text, text)
    
    def _process_dates(self, text: str) -> str:
        """Process dates"""
        def date_to_text(match):
            day = int(match.group(1))
            month = int(match.group(2))
            year = match.group(3)
            
            day_text = self._number_to_vietnamese(day)
            month_text = self._number_to_vietnamese(month)
            
            if len(year) == 2:
                year = '20' + year if int(year) < 50 else '19' + year
            year_text = self._number_to_vietnamese(int(year))
            
            return f"ngày {day_text} tháng {month_text} năm {year_text}"
        
        return self.date_pattern.sub(date_to_text, text)
    
    def _process_times(self, text: str) -> str:
        """Process times"""
        def time_to_text(match):
            hour = int(match.group(1))
            minute = int(match.group(2))
            second = match.group(3)
            
            hour_text = self._number_to_vietnamese(hour)
            minute_text = self._number_to_vietnamese(minute)
            
            result = f"{hour_text} giờ {minute_text} phút"
            
            if second:
                second_text = self._number_to_vietnamese(int(second))
                result += f" {second_text} giây"
            
            return result
        
        return self.time_pattern.sub(time_to_text, text)
    
    def _process_phone_numbers(self, text: str) -> str:
        """Process phone numbers (read digit by digit)"""
        def phone_to_text(match):
            digits = match.group(0).replace(' ', '').replace('-', '').replace('.', '')
            digit_text = ' '.join(self.DIGITS.get(d, d) for d in digits)
            return digit_text
        
        return self.phone_pattern.sub(phone_to_text, text)
    
    def _process_numbers(self, text: str) -> str:
        """Process remaining numbers"""
        def number_to_text(match):
            num_str = match.group(1).replace('.', '').replace(',', '.')
            try:
                num = float(num_str)
                return self._number_to_vietnamese(num)
            except:
                return match.group(0)
        
        return self.number_pattern.sub(number_to_text, text)
    
    def _number_to_vietnamese(self, num: float) -> str:
        """Convert number to Vietnamese text"""
        if num == 0:
            return 'không'
        
        # Handle decimals
        if isinstance(num, float) and not num.is_integer():
            int_part = int(num)
            dec_part = str(num).split('.')[1]
            int_text = self._number_to_vietnamese(int_part) if int_part else 'không'
            dec_text = ' '.join(self.DIGITS.get(d, d) for d in dec_part)
            return f"{int_text} phẩy {dec_text}"
        
        num = int(num)
        if num < 0:
            return 'âm ' + self._number_to_vietnamese(-num)
        
        # Handle small numbers
        if num < 10:
            return self.DIGITS[str(num)]
        if num < 20:
            return self.TEENS.get(str(num), str(num))
        if num < 100:
            tens = num // 10
            ones = num % 10
            result = self.TENS[str(tens)]
            if ones == 1:
                result += ' mốt'
            elif ones == 5:
                result += ' lăm'
            elif ones > 0:
                result += ' ' + self.DIGITS[str(ones)]
            return result
        
        # Handle larger numbers
        if num < 1000:
            hundreds = num // 100
            remainder = num % 100
            result = self.DIGITS[str(hundreds)] + ' trăm'
            if remainder > 0:
                if remainder < 10:
                    result += ' lẻ ' + self.DIGITS[str(remainder)]
                else:
                    result += ' ' + self._number_to_vietnamese(remainder)
            return result
        
        # Handle thousands, millions, etc.
        parts = []
        unit_idx = 0
        
        while num > 0:
            group = num % 1000
            if group > 0:
                group_text = self._number_to_vietnamese(group)
                if unit_idx > 0:
                    group_text += ' ' + self.UNITS[unit_idx]
                parts.append(group_text)
            num //= 1000
            unit_idx += 1
        
        return ' '.join(reversed(parts))
    
    def _process_special_chars(self, text: str) -> str:
        """Process special characters"""
        replacements = {
            '&': ' và ',
            '+': ' cộng ',
            '=': ' bằng ',
            '<': ' nhỏ hơn ',
            '>': ' lớn hơn ',
            '≤': ' nhỏ hơn hoặc bằng ',
            '≥': ' lớn hơn hoặc bằng ',
            '≠': ' khác ',
            '±': ' cộng trừ ',
            '×': ' nhân ',
            '÷': ' chia ',
            '°': ' độ ',
            '°C': ' độ xê ',
            '°F': ' độ ép ',
            '™': '',
            '®': '',
            '©': '',
            '…': '...',
            '—': ', ',
            '–': ', ',
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
        }
        
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        
        return text
    
    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation for natural pauses"""
        # Multiple periods to single
        text = re.sub(r'\.{2,}', '...', text)
        
        # Ensure space after punctuation
        text = re.sub(r'([.!?,:;])([^\s\d])', r'\1 \2', text)
        
        # Remove excessive punctuation
        text = re.sub(r'([!?]){2,}', r'\1', text)
        
        return text
    
    def _final_cleanup(self, text: str) -> str:
        """Final cleanup"""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Trim
        text = text.strip()
        
        return text
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Split on sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _detect_emotion(self, text: str) -> TextEmotion:
        """Detect emotion from text"""
        text_lower = text.lower()
        
        # Count keyword matches
        scores = {emotion: 0 for emotion in TextEmotion}
        
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[emotion] += 1
        
        # Check for exclamation marks (excitement)
        if text.count('!') >= 2:
            scores[TextEmotion.EXCITED] += 2
        
        # Check for question marks (neutral/professional)
        if '?' in text:
            scores[TextEmotion.PROFESSIONAL] += 1
        
        # Return emotion with highest score, or neutral
        max_emotion = max(scores, key=scores.get)
        if scores[max_emotion] > 0:
            return max_emotion
        
        return TextEmotion.NEUTRAL
    
    def _calculate_pause_before(self, sentence: str) -> float:
        """Calculate pause before sentence"""
        # Longer pause after previous sentence for new paragraph feel
        if sentence.startswith(('Nhưng', 'Tuy nhiên', 'Mặc dù', 'Thế nhưng')):
            return 0.3
        return 0.1
    
    def _calculate_pause_after(self, sentence: str) -> float:
        """Calculate pause after sentence"""
        if sentence.endswith('...'):
            return 0.5  # Dramatic pause
        if sentence.endswith('?'):
            return 0.3  # Question pause
        if sentence.endswith('!'):
            return 0.2  # Exclamation
        return 0.2  # Normal
    
    def _calculate_speed_modifier(
        self,
        sentence: str,
        emotion: TextEmotion
    ) -> float:
        """Calculate speed modifier based on content and emotion"""
        modifier = 1.0
        
        # Emotion-based
        if emotion == TextEmotion.EXCITED:
            modifier *= 1.1  # Faster
        elif emotion == TextEmotion.SAD:
            modifier *= 0.9  # Slower
        elif emotion == TextEmotion.CALM:
            modifier *= 0.95
        
        # Content-based
        if len(sentence) > 100:
            modifier *= 1.05  # Slightly faster for long sentences
        
        return modifier
    
    def _find_emphasis_words(self, sentence: str) -> List[str]:
        """Find words that should be emphasized"""
        emphasis_markers = ['rất', 'cực kỳ', 'vô cùng', 'hoàn toàn', 'tuyệt đối']
        words = sentence.lower().split()
        
        emphasis = []
        for i, word in enumerate(words):
            if word in emphasis_markers and i + 1 < len(words):
                emphasis.append(words[i + 1])
        
        return emphasis


# Convenience function
def smart_process(text: str) -> str:
    """
    Quick function to process text for TTS
    
    Args:
        text: Input text
        
    Returns:
        Processed text
    """
    processor = VietnameseTextProcessor()
    return processor.process(text)
