# Vietnamese TTS PRO - VieNeu-TTS

🚀 **State-of-the-Art Vietnamese Text-to-Speech** with AMD GPU acceleration via DirectML.

## ✨ Features

- **SOTA Vietnamese Voice:** Uses the advanced `VieNeu-TTS` model (Bert-VITS2 + NeuCodec) for superior naturalness.
- **Accurate Text Processing:** Integrated `VietnameseTextProcessor` for handling numbers, dates, currency, and abbreviations.
- **AMD GPU Acceleration:** Leverages ONNX Runtime DirectML for fast inference on AMD RX6600 and other AMD GPUs.
- **Instant Voice Cloning:** High-quality voice cloning from just 5-10 seconds of audio.
- **Rich Voice Library:** 10+ built-in high-quality Vietnamese voices (North and South dialects).
- **SRT Processing:** Effortlessly generate high-quality voiceovers for your subtitle files.

## 🚀 Quick Start

### System Requirements
- **OS:** Windows 10/11 (64-bit)
- **CPU:** AMD Ryzen 2700X or equivalent (multi-core recommended)
- **RAM:** 16GB minimum, 48GB recommended
- **GPU:** AMD RX6600 or compatible DirectML GPU
- **Python:** 3.10 or higher

### Prerequisites

#### 1. eSpeak-NG (Required for phonemization)

Download and install from: [eSpeak-NG Releases](https://github.com/espeak-ng/espeak-ng/releases)

**Windows Installation:**
1. Download `espeak-ng-X64.msi` (latest version)
2. Run the installer with default settings
3. Default path should be: `C:\Program Files\eSpeak NG\`

> ⚠️ **Important:** The app will still work without eSpeak but phonemization quality may be reduced.

#### 2. FFmpeg (Recommended)

Download and install from: [FFmpeg Downloads](https://ffmpeg.org/download.html)

**Windows Installation:**
1. Download `ffmpeg-release-full.7z` from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/)
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to your PATH environment variable

### Installation

```bash
# Run setup script to create environment and install dependencies
setup_env.bat

# Activate the virtual environment
env\Scripts\activate

# Launch the application
python -m app.main
```

## 📁 Project Structure

```
my-vietnamese-tts/
├── app/                    # Core logic
│   ├── main.py            # Entry point
│   ├── tts_engine.py      # Main engine wrapper
│   ├── config.py          # Configuration & paths
│   └── services/
│       ├── vieneu_engine.py   # VieNeu-TTS integration
│       └── vieneu/            # Core VieNeu models & utils
├── model_assets/           # Model files
│   └── vietnamese/        # VieNeu-TTS model & ONNX Codec
├── ui/                     # User interface components
├── samples/               # Custom voice samples for cloning
└── output/                # Generated audio recordings
```

## 🎤 Usage

### Text-to-Speech
1. Type or paste your Vietnamese text into the input field.
2. Select a voice from the **Voice Selector** panel.
3. Choose **Standard** or **Cloning** mode.
4. Click **Generate Speech** to synthsize and playback.

### Voice Cloning
1. Select the **Clone Voice** option in the selector.
2. Provide a 5-15 second WAV/MP3 sample of the target voice.
3. (Optional) Provide the reference text for the sample to improve accuracy.
4. Click **Generate** and VieNeu-TTS will clone the characteristics instantly.

### SRT to Speech
1. Navigate to the **SRT Import** tab.
2. Open your `.srt` file.
3. Select the desired voice and speed settings.
4. Click **Generate All** to process the entire file into high-quality audio chunks.

## 🔧 Technical Details

- **Backbone Model:** Qwen2-0.6B (Llama Architecture) trained for Vietnamese TTS.
- **Acoustic Model:** NeuCodec (Latent-based Codec) for high-fidelity audio reconstruction.
- **Interence Backend:** 
  - **Backbone:** PyTorch Transformers (CPU optimized).
  - **Codec:** ONNX Runtime DirectML (AMD GPU accelerated).
- **Phonemizer:** Dictionary-based matching with eSpeak-NG fallback for 100% accuracy.

## 🙏 Credits

- **pnnbao97/VieNeu-TTS:** The core SOTA Vietnamese TTS model.
- **neuphonic/neucodec:** Advanced neural audio codec.
- **Google DeepMind:** For the powerful agentic tools used in development.
