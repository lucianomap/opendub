set dotenv-filename := '.env'
set shell := ["bash", "-uc"]

# Variables
APP_NAME           := 'opendub'
PYTHON_VERSION     := "3.13"
OUTPUT_DIR         := env_var_or_default('OUTPUT_DIR', 'output')
MODEL_DIR          := env_var_or_default('MODEL_DIR', 'models')

# Model Configuration
WHISPER_MODEL      := env_var_or_default('WHISPER_MODEL', 'base')
TTS_MODEL          := env_var_or_default('TTS_MODEL', 'auto')
LLM_MODEL          := env_var_or_default('LLM_MODEL', 'gemini-2.0-flash')
LLM_PROVIDER       := env_var_or_default('LLM_PROVIDER', 'auto')

# Processing Configuration
DEVICE             := env_var_or_default('DEVICE', 'auto')
GPU_ENABLED        := env_var_or_default('GPU_ENABLED', 'true')
CUDA_VISIBLE_DEVICES := env_var_or_default('CUDA_VISIBLE_DEVICES', '0')

# LLM API Keys (set in .env file)
OPENAI_API_KEY     := env_var_or_default('OPENAI_API_KEY', '')
ANTHROPIC_API_KEY  := env_var_or_default('ANTHROPIC_API_KEY', '')
GOOGLE_API_KEY     := env_var_or_default('GOOGLE_API_KEY', '')
HF_TOKEN           := env_var_or_default('HF_TOKEN', '')

# Aliases
alias s := setup
alias w := web
alias c := cli

# Default recipe
default:
    @just --list

# Setup development environment
setup:
    @echo "🔧 Setting up OpenDub..."
    @if ! command -v uv &> /dev/null; then \
        echo "📦 Installing uv..."; \
        curl -LsSf https://astral.sh/uv/install.sh | sh; \
        export PATH="$HOME/.cargo/bin:$PATH"; \
    fi
    @echo "🐍 Creating Python {{PYTHON_VERSION}} environment..."
    uv venv --clear --python {{PYTHON_VERSION}}
    @echo "📦 Installing dependencies..."
    uv sync
    @echo "📁 Creating directories..."
    mkdir -p {{OUTPUT_DIR}} {{MODEL_DIR}} cache logs
    @if [ ! -f .env ]; then \
        echo "📝 Creating .env file..."; \
        printf '# OpenDub Configuration\n' > .env; \
        printf '# Add at least one LLM API key\n\n' >> .env; \
        printf '# OpenAI: https://platform.openai.com/api-keys\n' >> .env; \
        printf 'OPENAI_API_KEY=\n\n' >> .env; \
        printf '# Anthropic: https://console.anthropic.com/\n' >> .env; \
        printf 'ANTHROPIC_API_KEY=\n\n' >> .env; \
        printf '# Google: https://makersuite.google.com/app/apikey\n' >> .env; \
        printf 'GOOGLE_API_KEY=\n\n' >> .env; \
        printf '# Optional: HuggingFace for model downloads\n' >> .env; \
        printf 'HF_TOKEN=\n\n' >> .env; \
        printf '# Model settings\n' >> .env; \
        printf 'WHISPER_MODEL=base\n' >> .env; \
        printf 'TTS_MODEL=auto\n' >> .env; \
        printf 'LLM_MODEL=gemini-2.0-flash\n\n' >> .env; \
        printf '# Directories\n' >> .env; \
        printf 'OUTPUT_DIR=output\n' >> .env; \
        printf 'MODEL_DIR=models\n\n' >> .env; \
        printf '# GPU settings\n' >> .env; \
        printf 'GPU_ENABLED=true\n' >> .env; \
        printf 'DEVICE=auto\n' >> .env; \
        echo "⚠️  Please edit .env and add at least one API key"; \
    fi
    @echo "📥 Downloading Whisper {{WHISPER_MODEL}} model..."
    -uv run python -c "import whisper; whisper.load_model('{{WHISPER_MODEL}}')"
    @echo "✅ Setup complete!"
    @echo "Next: Run 'just web' or 'just cli --help'"

# Update dependencies
update-deps:
    uv lock

# Install dependencies
install:
    uv sync

# Setup fish-speech for OpenAudio support
setup-fish-speech:
    @echo "🐟 Setting up fish-speech for OpenAudio support..."
    cd modules/synthesis && uv sync
    @echo "✅ Fish-speech installed successfully!"

# Check if OpenAudio is fully set up (model + fish-speech)
check-openaudio:
    @echo "Checking OpenAudio setup..."
    @echo ""
    @if [ -f "{{MODEL_DIR}}/openaudio-s1-mini/model.pth" ]; then \
        echo "✅ OpenAudio S1-mini model is downloaded"; \
    else \
        echo "❌ OpenAudio S1-mini model not found"; \
        echo "   Run 'just download-openaudio' for instructions"; \
    fi
    @if [ -d "modules/synthesis/.venv" ]; then \
        echo "✅ Fish-speech module is installed"; \
    else \
        echo "❌ Fish-speech module not installed"; \
        echo "   Run 'just setup-fish-speech' to install"; \
    fi

# Download OpenAudio S1-mini model (advanced TTS, 3.4GB)
download-openaudio:
    @echo "🎵 OpenAudio S1-mini Model Download"
    @echo "===================================="
    @echo ""
    @echo "⚠️  Prerequisites:"
    @echo "1. HuggingFace account (free): https://huggingface.co/join"
    @echo "2. Accept model license: https://huggingface.co/fishaudio/openaudio-s1-mini"
    @echo "3. HuggingFace token: https://huggingface.co/settings/tokens"
    @echo ""
    @echo "📥 Download Instructions:"
    @echo ""
    @echo "1. Login to HuggingFace:"
    @echo "   huggingface-cli login"
    @echo ""
    @echo "2. Download the model (3.4GB):"
    @echo "   huggingface-cli download fishaudio/openaudio-s1-mini \\"
    @echo "       --local-dir {{MODEL_DIR}}/openaudio-s1-mini \\"
    @echo "       --include 'model.pth' 'codec.pth' 'tokenizer.tiktoken' 'config.json' 'special_tokens.json'"
    @echo ""
    @echo "3. Once downloaded, OpenDub will automatically use OpenAudio for better voice quality!"
    @echo ""
    @echo "Note: Without OpenAudio, OpenDub uses SpeechT5 (works well but lower quality)"

# Start OpenAudio API server manually (for development)
openaudio-server PORT='8180':
    @echo "🎵 Starting OpenAudio S1-mini API server on port {{PORT}}"
    ./scripts/start_openaudio_server.sh {{PORT}}

# Stop OpenAudio API server
openaudio-stop:
    @echo "🛑 Stopping OpenAudio API server..."
    pkill -f "tools.api_server" || echo "OpenAudio server not running"

# Check OpenAudio server health
openaudio-health PORT='8180':
    @echo "🔍 Checking OpenAudio API server health on port {{PORT}}..."
    @curl -s "http://localhost:{{PORT}}/v1/health" > /dev/null 2>&1 && echo "✅ Server is healthy" || echo "❌ Server is not responding"

# Run web interface
web PORT='8080':
    @echo "🌐 Starting OpenDub web interface on http://localhost:{{PORT}}"
    uv run python web.py

# Run CLI
cli *ARGS:
    uv run python cli.py {{ARGS}}

# Quick dub shortcut
dub URL *ARGS:
    uv run python cli.py {{URL}} {{ARGS}}

# Run TTS synthesis with automatic OpenAudio server management
tts TEXT LANGUAGE='en' MODEL='auto':
    @echo "🎤 Running TTS synthesis..."
    @echo "Text: {{TEXT}}"
    @echo "Language: {{LANGUAGE}}"
    @echo "Model: {{MODEL}}"
    uv run python -c "from core.synthesizer import SpeechSynthesizer; import soundfile as sf; synth = SpeechSynthesizer('{{MODEL}}'); audio = synth._synthesize_segment('{{TEXT}}', '{{LANGUAGE}}'); sf.write('tts_output.wav', audio, 16000); print('✅ Saved to tts_output.wav')"

# Test with sample YouTube video
test-youtube:
    @echo "🧪 Testing with sample video..."
    uv run python cli.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" -l es -l fr --keep-intermediates

# Docker commands
docker-build:
    @echo "🐳 Building Docker image..."
    docker build -t opendub:latest .

docker-up:
    @echo "🐳 Starting OpenDub with Docker..."
    docker-compose up -d
    @echo "✅ OpenDub running at http://localhost:8080"

docker-down:
    @echo "🛑 Stopping OpenDub..."
    docker-compose down

docker-logs:
    docker-compose logs -f

# Download Whisper model
download-whisper MODEL='base':
    @echo "📥 Downloading Whisper {{MODEL}} model..."
    uv run python -c "import whisper; whisper.load_model('{{MODEL}}')"

# Show available Whisper models
whisper-models:
    @echo "=== Available Whisper Models ==="
    @echo "tiny     - 39M parameters  (fastest, lowest quality)"
    @echo "base     - 74M parameters  (good balance) [DEFAULT]"
    @echo "small    - 244M parameters (better quality)"
    @echo "medium   - 769M parameters (high quality)"
    @echo "large    - 1550M parameters (best quality)"
    @echo "large-v3 - 1550M parameters (latest version)"

# Show supported languages
languages:
    @echo "=== Supported Languages ==="
    @echo "pt - Portuguese    es - Spanish      fr - French"
    @echo "de - German        it - Italian      ja - Japanese"
    @echo "ko - Korean        zh - Chinese      ru - Russian"
    @echo "ar - Arabic        hi - Hindi        nl - Dutch"
    @echo "pl - Polish        tr - Turkish      sv - Swedish"

# Clean output files
clean:
    rm -rf {{OUTPUT_DIR}}/*
    rm -rf logs/*
    rm -rf cache/*
    rm -f *.mp4 *.wav *.json

# Clean everything including models
clean-all: clean
    rm -rf {{MODEL_DIR}}/*
    rm -rf .venv uv.lock

# Code quality tools
lint:
    uv run ruff check .

format:
    uv run black .
    uv run ruff check . --fix

quality: lint format

# Run tests
test:
    uv run pytest tests/ -v || echo "No tests yet"

# GPU monitoring
gpu-status:
    @if command -v nvidia-smi &> /dev/null; then \
        nvidia-smi; \
    else \
        echo "No NVIDIA GPU detected or nvidia-smi not installed"; \
    fi

# System check
check:
    @echo "=== System Check ==="
    @echo ""
    @echo -n "Python: "
    @uv run python --version || echo "Not found"
    @echo -n "FFmpeg: "
    @ffmpeg -version 2>/dev/null | head -n1 || echo "Not found - install with: apt install ffmpeg"
    @echo -n "yt-dlp: "
    @yt-dlp --version 2>/dev/null || echo "Not found - install with: pip install yt-dlp"
    @echo -n "CUDA: "
    @if command -v nvidia-smi &> /dev/null; then \
        echo "Available"; \
    else \
        echo "Not available (CPU mode)"; \
    fi
    @echo ""
    @echo "API Keys:"
    @[ -n "{{OPENAI_API_KEY}}" ] && echo "  OpenAI: Set" || echo "  OpenAI: Not set"
    @[ -n "{{ANTHROPIC_API_KEY}}" ] && echo "  Anthropic: Set" || echo "  Anthropic: Not set"  
    @[ -n "{{GOOGLE_API_KEY}}" ] && echo "  Google: Set" || echo "  Google: Not set"
    @[ -n "{{HF_TOKEN}}" ] && echo "  HuggingFace: Set" || echo "  HuggingFace: Not set"

# Show configuration
show-config:
    @echo "=== OpenDub Configuration ==="
    @echo "OUTPUT_DIR: {{OUTPUT_DIR}}"
    @echo "MODEL_DIR: {{MODEL_DIR}}"
    @echo "WHISPER_MODEL: {{WHISPER_MODEL}}"
    @echo "TTS_MODEL: {{TTS_MODEL}}"
    @echo "LLM_MODEL: {{LLM_MODEL}}"
    @echo "DEVICE: {{DEVICE}}"
    @echo "GPU_ENABLED: {{GPU_ENABLED}}"
    @echo "OPENAI_API_KEY: $(if [ -n '{{OPENAI_API_KEY}}' ]; then echo 'Set'; else echo 'Not set'; fi)"
    @echo "ANTHROPIC_API_KEY: $(if [ -n '{{ANTHROPIC_API_KEY}}' ]; then echo 'Set'; else echo 'Not set'; fi)"
    @echo "GOOGLE_API_KEY: $(if [ -n '{{GOOGLE_API_KEY}}' ]; then echo 'Set'; else echo 'Not set'; fi)"

# Help for common issues  
troubleshoot:
    @echo "=== Troubleshooting Guide ==="
    @echo ""
    @echo "Issue: 'No LLM API keys found'"
    @echo "  Solution: Edit .env and add at least one API key"
    @echo ""
    @echo "Issue: 'ffmpeg: command not found'"
    @echo "  Solution: Install FFmpeg"
    @echo "    Ubuntu/Debian: sudo apt install ffmpeg"
    @echo "    macOS: brew install ffmpeg"
    @echo ""
    @echo "Issue: 'CUDA out of memory'"
    @echo "  Solution: Use smaller models or CPU"
    @echo "    Set WHISPER_MODEL=tiny in .env"
    @echo "    Set DEVICE=cpu in .env"
    @echo ""
    @echo "Issue: Port 8080 already in use"
    @echo "  Solution: Use different port"
    @echo "    just web 8081"
