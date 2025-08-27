#!/bin/bash
# Start the OpenAudio S1-mini API server using fish-speech

# Accept port as first argument, default to 8180
PORT="${1:-8180}"

echo "Starting OpenAudio S1-mini API server..."
echo "This will run the fish-speech API server on port $PORT"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Get the project root directory (parent of scripts directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT/modules/synthesis"

# Start the API server with the OpenAudio S1-mini model
# Note: llama-checkpoint-path should point to the directory containing model.pth and config.json
# Performance optimizations:
# --half: Use FP16 precision for faster inference
# --compile: Compile model with torch.compile for better performance (requires PyTorch 2.0+)
uv run python -m tools.api_server \
    --listen "0.0.0.0:$PORT" \
    --llama-checkpoint-path "$PROJECT_ROOT/models/openaudio-s1-mini" \
    --decoder-checkpoint-path "$PROJECT_ROOT/models/openaudio-s1-mini/codec.pth" \
    --decoder-config-name modded_dac_vq \
    --half \
    --compile