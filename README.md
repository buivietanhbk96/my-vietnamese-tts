# VietTTS Desktop Application

🎙️ **Vietnamese Text-to-Speech Desktop App** với tính năng Voice Cloning

## ✨ Tính năng

- **Text-to-Speech:** Chuyển đổi văn bản tiếng Việt thành giọng nói tự nhiên
- **Voice Cloning:** Clone giọng nói từ file MP3/WAV
- **SRT Processing:** Import file SRT và generate audio cho từng subtitle
- **Built-in Voices:** Nhiều giọng đọc có sẵn
- **CPU Optimized:** Chạy tốt trên CPU, không cần GPU

---

## 🚀 PRO MAX Features (NEW!)

### 🎛️ Professional Audio Processing
- **Peak/RMS/LUFS Normalization:** Chuẩn hóa âm lượng theo các tiêu chuẩn broadcast
- **Spectral Noise Reduction:** Loại bỏ tạp âm chuyên nghiệp
- **Dynamic Compression:** Nén âm thanh với attack/release tùy chỉnh
- **Brick Wall Limiter:** Đảm bảo không clipping
- **De-esser:** Giảm âm xì
- **3-Band EQ:** Cân bằng tần số

### 📊 Waveform Visualization
- Hiển thị waveform real-time như DAW chuyên nghiệp
- Zoom in/out, selection, playhead
- RMS visualization
- Time markers

### 💾 Preset System
- Lưu và load voice/audio presets
- Built-in presets: Clean Voice, Podcast, YouTube, Audiobook, Broadcast
- Import/Export presets
- Preset search và filtering

### ⚡ Enhanced Batch Processing
- Parallel processing với nhiều workers
- Chi tiết progress tracking
- Pause/Resume/Cancel
- Auto retry failed tasks
- Export báo cáo chi tiết

### 🧠 Smart Text Processing
- Tự động đọc số tiếng Việt (123 → một trăm hai mươi ba)
- Đọc ngày tháng (15/03/2024 → ngày mười lăm tháng ba năm hai không hai tư)
- Xử lý tiền tệ (100,000đ → một trăm nghìn đồng)
- Expand abbreviations (TP.HCM → Thành phố Hồ Chí Minh)
- Đọc từ viết tắt tiếng Anh (CEO → Xi I Âu)
- Xử lý emoji thành text

---

## 🛠️ Cài đặt

### Yêu cầu
- Python 3.10 trở lên
- FFmpeg (sẽ được hướng dẫn cài đặt)
- Windows 10/11

### Bước 1: Chạy setup script
```bash
cd "D:\Source code\viet-tts"
setup_env.bat
```

### Bước 2: Cài đặt FFmpeg
1. Download FFmpeg từ: https://github.com/BtbN/FFmpeg-Builds/releases
2. Giải nén vào `C:\ffmpeg`
3. Thêm `C:\ffmpeg\bin` vào System PATH

### Bước 3: Chạy ứng dụng
```bash
venv\Scripts\activate
python -m app.main
```

## 📁 Cấu trúc thư mục

```
viet-tts/
├── app/                    # Core application
│   ├── main.py            # Entry point
│   ├── tts_engine.py      # TTS wrapper
│   ├── config.py          # Configuration
│   ├── audio_processor.py # PRO MAX: Audio processing
│   ├── text_processor.py  # PRO MAX: Smart text processing
│   ├── preset_manager.py  # PRO MAX: Preset system
│   ├── batch_processor.py # PRO MAX: Enhanced batch processing
│   ├── waveform_viewer.py # PRO MAX: Waveform visualization
│   └── ...
├── ui/                     # User interface
│   ├── main_window.py     # Main window
│   └── components/        # UI components
│       └── audio_panel_pro.py  # PRO MAX: Audio panel
├── utils/                  # Utilities
├── samples/               # Voice samples
├── output/                # Generated audio files
├── presets/               # PRO MAX: Saved presets
│   ├── voices/            # Voice presets
│   ├── audio/             # Audio processing presets
│   ├── batch/             # Batch processing presets
│   └── projects/          # Project presets
├── pretrained-models/     # TTS models (auto-download)
└── requirements.txt
```

## 🎯 Sử dụng

### Text-to-Speech cơ bản
1. Nhập văn bản tiếng Việt vào ô text
2. Chọn giọng đọc từ dropdown
3. Điều chỉnh tốc độ nếu cần
4. Click "Generate Speech"
5. Nghe và lưu file audio

### Voice Cloning
1. Click "Clone Voice"
2. Chọn file MP3/WAV (3-10 giây)
3. Preview để kiểm tra
4. Sử dụng voice mới để generate

### Import SRT
1. Click tab "SRT Import"
2. Chọn file SRT
3. Chọn voice và settings
4. Click "Generate All"
5. Output: 1.wav, 2.wav, 3.wav, ...

## ⚙️ Cấu hình

Các settings được lưu tự động:
- Voice đã chọn
- Tốc độ đọc
- Thư mục output
- Window position

---

## 🎧 Audio Processing Presets

| Preset | Target | Description |
|--------|--------|-------------|
| Clean Voice | General | Tiếng nói sạch, normalize -3dB |
| Podcast | -16 LUFS | Chuẩn podcast với compression nhẹ |
| YouTube | -14 LUFS | Tối ưu cho video với EQ boost |
| Audiobook | -20 RMS | Đều, nhẹ nhàng, noise reduction |
| Broadcast | -23 LUFS | Chuẩn phát sóng quốc tế |

---

## 🔧 API Reference (cho developers)

### Audio Processing
```python
from app.audio_processor import AudioPostProcessor, ProcessingPreset

# Khởi tạo processor
processor = AudioPostProcessor(sample_rate=22050)

# Apply preset
processed = processor.process_full(audio_data, preset=ProcessingPreset.PODCAST)

# Hoặc xử lý từng bước
normalized = processor.normalizer.normalize_lufs(audio_data, target_lufs=-16)
denoised = processor.noise_reducer.reduce_spectral(normalized, strength=0.5)
compressed = processor.dynamic_processor.compress(denoised, threshold_db=-20)
```

### Text Processing
```python
from app.text_processor import get_text_processor

processor = get_text_processor()

# Xử lý text đầy đủ
processed = processor.process("Giá 1,500,000đ ngày 15/03/2024")
# Output: "Giá một triệu năm trăm nghìn đồng ngày mười lăm tháng ba năm hai không hai tư"
```

### Preset Management
```python
from app.preset_manager import get_preset_manager, VoicePreset

pm = get_preset_manager()

# Save preset
preset = VoicePreset(name="My Voice", voice_file_path="voice.wav", speed=1.0)
pm.save_voice_preset(preset)

# Load preset
loaded = pm.load_voice_preset("My Voice")
```

### Batch Processing
```python
from app.batch_processor import BatchProcessor

processor = BatchProcessor(max_workers=4, auto_retry=True)

# Add tasks
processor.add_texts(texts=["Hello", "World"], output_dir="./output")

# Set callbacks
processor.set_callbacks(on_progress=lambda p: print(f"{p.progress_percent}%"))

# Start processing
processor.start()
```

---

## 📝 License

Apache 2.0 - Based on [viet-tts](https://github.com/dangvansam/viet-tts)

## 🙏 Credits

- [dangvansam/viet-tts](https://github.com/dangvansam/viet-tts) - Core TTS engine
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - Model architecture
- [silero-vad](https://github.com/snakers4/silero-vad) - Voice Activity Detection
