"""Synthesize speech using OpenAudio S1-mini."""

import os
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import structlog

logger = structlog.get_logger(__name__)


class SpeechSynthesizer:
    """Synthesize speech from translated text."""

    def __init__(self, model: str = "auto", device: str | None = None):
        """Initialize TTS synthesizer.

        Args:
            model: TTS model to use (openaudio or auto)
            device: Device to use (cuda, cpu, or None for auto)
        """
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_type = model
        self.server_process = None
        self.api_running = False

        # Get OpenAudio API server port from environment or use default
        self.api_port = int(os.environ.get("OPENAUDIO_PORT", "8180"))

        # Check for OpenAudio model
        self.openaudio_available = self._check_openaudio()

        if model == "auto":
            if self.openaudio_available:
                self.model_type = "openaudio"
                logger.info("Using OpenAudio S1-mini for synthesis")
            else:
                logger.warning("OpenAudio not available, will generate silence")
                self.model_type = None

        # Initialize OpenAudio if available
        if self.model_type == "openaudio" and self.openaudio_available:
            from core.openaudio_synthesizer import OpenAudioSynthesizer

            self.synthesizer = OpenAudioSynthesizer(device=self.device)
        else:
            self.synthesizer = None

    def _check_openaudio(self) -> bool:
        """Check if OpenAudio model and API are available."""
        model_path = Path("models/openaudio-s1-mini/model.pth")
        codec_path = Path("models/openaudio-s1-mini/codec.pth")

        # Check if model files exist
        if not (model_path.exists() and codec_path.exists()):
            return False

        # Check if API server is running
        import requests

        try:
            response = requests.get(f"http://localhost:{self.api_port}/v1/health", timeout=1)
            if response.status_code == 200:
                logger.info(f"OpenAudio API server already running on port {self.api_port}")
                self.api_running = True
                return True
        except Exception:
            pass

        # Find a free port if configured port is taken
        def find_free_port(start_port: int = 8180, max_attempts: int = 10) -> int:
            """Find a free port starting from start_port."""
            for port in range(start_port, start_port + max_attempts):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    try:
                        s.bind(("", port))
                        return port
                    except OSError:
                        continue
            raise RuntimeError(
                f"Could not find free port in range {start_port}-{start_port + max_attempts - 1}"
            )

        try:
            self.api_port = find_free_port(self.api_port)
            logger.info(f"Using port {self.api_port} for OpenAudio API server")
        except RuntimeError as e:
            logger.warning(f"Could not find free port: {e}")
            return False

        # Try to start the API server
        logger.info(f"Starting OpenAudio API server on port {self.api_port}...")
        try:
            # Start server in background
            env = os.environ.copy()
            env["OPENAUDIO_PORT"] = str(self.api_port)

            self.server_process = subprocess.Popen(
                ["bash", "scripts/start_openaudio_server.sh", str(self.api_port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path(__file__).parent.parent,
                env=env,
            )

            # Wait for server to be ready (max 120 seconds)
            max_wait = 120
            for i in range(max_wait):
                # Check if process is still alive
                if self.server_process.poll() is not None:
                    # Process died, check stderr for error
                    stderr = (
                        self.server_process.stderr.read() if self.server_process.stderr else b""
                    )
                    logger.warning(
                        f"OpenAudio server process died: {stderr.decode('utf-8', errors='ignore')}"
                    )
                    return False

                try:
                    response = requests.get(
                        f"http://localhost:{self.api_port}/v1/health", timeout=1
                    )
                    if response.status_code == 200:
                        logger.info(
                            f"OpenAudio API server started successfully on port {self.api_port}"
                        )
                        self.api_running = True
                        return True
                except Exception:
                    pass

                # Show progress every 5 seconds
                if i > 0 and i % 5 == 0:
                    logger.debug(f"Waiting for OpenAudio API server... ({i}/{max_wait}s)")

                time.sleep(1)

            logger.warning("OpenAudio API server failed to start in time")
            return False

        except Exception as e:
            logger.warning(f"Failed to start OpenAudio API server: {e}")
            return False

    def synthesize(self, translation: dict, output_path: Path, language: str) -> Path:
        """Synthesize speech from translation.

        Args:
            translation: Translation data with segments
            output_path: Directory to save audio files
            language: Target language code

        Returns:
            Path to synthesized audio file
        """
        logger.info(f"Synthesizing speech for {language}")

        if self.synthesizer:
            # Use OpenAudio synthesizer
            audio_segments = self.synthesizer.synthesize_segments(translation["segments"])

            # Combine segments with proper timing
            combined_audio, sample_rate = self.synthesizer.combine_segments(audio_segments)

            # Save audio file
            audio_path = output_path / f"audio_{language}.wav"
            sf.write(audio_path, combined_audio, sample_rate)

            logger.info(f"Saved synthesized audio to {audio_path.name}")
            return audio_path
        else:
            # Generate silence
            logger.warning("No synthesizer available, generating silence")

            # Calculate total duration
            duration = translation["duration"] if "duration" in translation else 10.0
            sample_rate = 16000

            # Generate silence
            silence = np.zeros(int(duration * sample_rate))

            # Save audio file
            audio_path = output_path / f"audio_{language}.wav"
            sf.write(audio_path, silence, sample_rate)

            logger.info(f"Saved silence audio to {audio_path.name}")
            return audio_path

    def cleanup(self):
        """Clean up resources including stopping the API server if started."""
        if hasattr(self, "synthesizer") and self.synthesizer:
            self.synthesizer.cleanup()

        if hasattr(self, "server_process") and self.server_process and self.api_running:
            try:
                logger.info("Stopping OpenAudio API server...")
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except Exception:
                try:
                    self.server_process.kill()
                except Exception:
                    pass
            self.api_running = False
            self.server_process = None

    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass
