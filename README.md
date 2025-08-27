# 🎬 OpenDub

**Local AI-powered video dubbing that runs 100% on your machine**

OpenDub is a simple, privacy-focused tool for dubbing videos into multiple languages using AI. Unlike cloud services, everything runs locally - your videos never leave your machine.

## ✨ Features

- **🔒 100% Local Processing** - Complete privacy, no data leaves your machine
- **🌍 Multi-Language Support** - Dub into 15+ languages with one command
- **💾 Download & Keep** - Save dubbed videos permanently (unlike browser extensions)
- **🎯 No Limits** - No daily quotas, subscriptions, or hidden fees
- **📦 Batch Processing** - Process entire playlists or channels
- **🎛️ Flexible Models** - Choose between quality and speed
- **♻️ Smart Retries** - Automatic retry with exponential backoff for API rate limits
- **🔧 Fully Configurable** - Customize models, languages, and processing via .env

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- FFmpeg (`apt install ffmpeg` or `brew install ffmpeg`)
- At least one LLM API key (OpenAI, Anthropic, or Google)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/opendub.git
cd opendub

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install just (if not already installed)
# macOS: brew install just
# Linux: cargo install just

# Setup OpenDub
just setup

# Add your API keys to .env
nano .env  # Add at least one: OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY

# Start the web interface
just web
```

Open http://localhost:8080 in your browser!

## 📖 Usage

### Web Interface

```bash
just web        # Start on default port 8080
just web 8081   # Use custom port
```

Then:
1. Open http://localhost:8080
2. Paste a YouTube URL
3. Select target languages
4. Click "Start Dubbing"
5. Download your dubbed videos

### Command Line

```bash
# Dub a YouTube video to Portuguese
just dub https://youtube.com/watch?v=xyz -l pt

# Multiple languages
just dub https://youtube.com/watch?v=xyz -l pt -l es -l fr

# Or use the CLI directly
just cli -- https://youtube.com/watch?v=xyz -l pt

# Dub a local file
just cli -- video.mp4 -l pt -s en

# Use better quality models
just cli -- https://youtube.com/watch?v=xyz -l pt -m large
```

### Docker

```bash
# Build and run with Docker
just docker-up

# Check logs
just docker-logs

# Stop
just docker-down
```

## 📋 Common Tasks

```bash
just            # Show all available commands
just setup      # Initial setup
just web        # Start web interface
just cli        # Run CLI (add -- before arguments)
just dub        # Quick dubbing shortcut
just languages  # Show supported languages
just whisper-models  # Show available models
just check      # System diagnostics
just clean      # Clean output files
```

## ⚙️ Configuration

Create a `.env` file with your settings:

```bash
# LLM API Keys (at least one required)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...

# Model Configuration
WHISPER_MODEL=base        # Transcription model size
TTS_MODEL=auto           # auto, speecht5, or openaudio
LLM_MODEL=gemini-2.0-flash  # Specific LLM model to use

# Processing
DEVICE=auto              # auto, cuda, or cpu
GPU_ENABLED=true         # Enable GPU acceleration

# Directories
OUTPUT_DIR=output
MODEL_DIR=models
```

## 🎯 Advanced Setup

### Using OpenAudio S1-mini (Better Quality)

OpenAudio provides superior voice quality but requires downloading the model:

```bash
# Setup OpenAudio (includes model download)
just setup-openaudio

# This will:
# 1. Install extra dependencies
# 2. Download the 3.4GB model
# 3. Setup fish-speech framework

# Note: Requires HuggingFace account and accepting the model license
```

### GPU Acceleration

For faster processing with NVIDIA GPUs:

```bash
# Check GPU status
just gpu-status

# If CUDA is available, it will be used automatically
# To force CPU mode, set in .env:
echo "DEVICE=cpu" >> .env
```

### Download Different Whisper Models

```bash
# List available models
just whisper-models

# Download a specific model
just download-whisper large  # Best quality
just download-whisper tiny   # Fastest
```

## 🧪 Testing & Development

```bash
# Run a quick test
just test-youtube

# Code formatting
just format

# Linting
just lint

# Run tests
just test

# System check
just check

# Configuration check
just config
```

## 🌍 Supported Languages

- 🇵🇹 Portuguese (pt)
- 🇪🇸 Spanish (es)
- 🇫🇷 French (fr)
- 🇩🇪 German (de)
- 🇮🇹 Italian (it)
- 🇯🇵 Japanese (ja)
- 🇰🇷 Korean (ko)
- 🇨🇳 Chinese (zh)
- 🇷🇺 Russian (ru)
- 🇸🇦 Arabic (ar)
- 🇮🇳 Hindi (hi)
- 🇳🇱 Dutch (nl)
- 🇵🇱 Polish (pl)
- 🇹🇷 Turkish (tr)
- 🇸🇪 Swedish (sv)

## 🧠 Models

### Transcription (Whisper)
- `tiny` - Fastest, lowest quality (39M)
- `base` - Good balance (74M) **[Default]**
- `small` - Better quality (244M)
- `medium` - High quality (769M)
- `large` - Best quality (1550M)
- `large-v3` - Latest and best (1550M)

### Translation (LLMs)
Automatically detects available provider from .env:
- **OpenAI**: GPT-4 (configurable via `LLM_MODEL`)
- **Anthropic**: Claude 3 Sonnet (configurable via `LLM_MODEL`) 
- **Google**: Gemini 2.0 Flash **[Default]** (configurable via `LLM_MODEL`)

Configure in `.env`:
```bash
# Default models per provider
LLM_MODEL=gemini-2.0-flash  # For Google
# LLM_MODEL=gpt-4           # For OpenAI
# LLM_MODEL=claude-3-sonnet-20240229  # For Anthropic
```

### Speech Synthesis
- **SpeechT5** - Default, works everywhere
- **OpenAudio S1-mini** - Best quality (requires setup)

## 🏗️ Project Structure

```
opendub/
├── core/               # Core pipeline logic
│   ├── downloader.py   # Video download (yt-dlp)
│   ├── transcriber.py  # Whisper transcription  
│   ├── translator.py   # LLM translation (with retry logic)
│   ├── synthesizer.py  # TTS synthesis (SpeechT5/OpenAudio)
│   ├── stitcher.py     # Audio/video merge
│   └── pipeline.py     # Main orchestrator
├── modules/            # Isolated modules
│   └── synthesis/      # OpenAudio dependencies isolation
├── scripts/            # Helper scripts
│   └── start_openaudio_server.sh
├── cli.py              # Command-line interface
├── web.py              # Web interface (FastAPI)
├── justfile            # Task automation
├── pyproject.toml      # Dependencies (uv)
├── .env                # Configuration
├── Dockerfile          # Container setup
└── docker-compose.yml  # Docker orchestration
```

## 🐛 Troubleshooting

```bash
# Run diagnostics
just troubleshoot

# Common issues:

# "No LLM API keys found"
echo "OPENAI_API_KEY=sk-..." >> .env
# Or for Google (recommended):
echo "GOOGLE_API_KEY=AIza..." >> .env

# "ffmpeg: command not found"
sudo apt install ffmpeg  # Ubuntu/Debian
brew install ffmpeg      # macOS

# "CUDA out of memory"
echo "WHISPER_MODEL=tiny" >> .env
echo "DEVICE=cpu" >> .env

# Port 8080 in use
just web 8081  # Use different port

# "429 Rate limit" or "Quota exceeded"
# The tool automatically retries with exponential backoff
# For persistent issues, try:
# - Using a different LLM provider
# - Reducing batch size
# - Adding delays between requests
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] More TTS models
- [ ] Voice cloning
- [ ] Lip sync support
- [ ] GUI application
- [ ] More video platforms

## 📄 License

MIT License - Use freely for any purpose

## 🙏 Credits

Built with:
- [OpenAI Whisper](https://github.com/openai/whisper) - Transcription
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - Video download
- [FFmpeg](https://ffmpeg.org/) - Video processing
- [SpeechT5](https://huggingface.co/microsoft/speecht5_tts) - Default TTS
- [OpenAudio S1-mini](https://huggingface.co/fishaudio/openaudio-s1-mini) - Advanced TTS
- [Google Genai SDK](https://github.com/google/genai) - Google Gemini integration
- [uv](https://github.com/astral-sh/uv) - Python package management
- [just](https://github.com/casey/just) - Command runner

## ⚠️ Disclaimer

This tool is for personal use. Respect copyright laws and content creators' rights. Only dub content you have permission to modify.

---

**Made with ❤️ for parents who want to watch videos with their kids in any language**