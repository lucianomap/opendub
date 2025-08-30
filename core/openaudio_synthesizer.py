"""Text-to-speech synthesis using OpenAudio S1-mini model via fish-speech."""

import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import structlog
import torch

logger = structlog.get_logger(__name__)

try:
    import ormsgpack
except ImportError:
    ormsgpack = None
    logger.warning("ormsgpack not installed, reference audio may not work properly")


@dataclass
class AudioSegment:
    """Audio segment with timing information."""

    audio: np.ndarray
    sample_rate: int
    start: float
    end: float
    text: str


class OpenAudioSynthesizer:
    """TTS synthesizer using OpenAudio S1-mini model via fish-speech."""

    def __init__(self, language: str = "pt", device: str = "cuda", varied_narrators: bool = False):
        self.language = language
        self.device = device if torch.cuda.is_available() else "cpu"
        self.varied_narrators = varied_narrators

        # Model path - relative to the module file
        module_dir = Path(__file__).parent.parent  # Goes up to project root
        self.model_dir = module_dir / "models" / "openaudio-s1-mini"

        # API server settings - configurable via environment variables
        self.api_host = os.getenv("OPENAUDIO_API_HOST", "localhost")
        self.api_port = int(os.getenv("OPENAUDIO_API_PORT", "8180"))
        self.api_url = f"http://{self.api_host}:{self.api_port}"

        # Check if API server is running
        self.api_available = self._check_api_server()
        self.server_process = None  # Track server process for restart
        self.restart_attempts = 0
        self.max_restart_attempts = 3

        if not self.api_available:
            logger.warning(f"OpenAudio API server not running at {self.api_url}")
            logger.warning("Start it with: just openaudio-server")

        # Sample rate for OpenAudio models
        self.sample_rate = 44100  # Fish-speech typically uses 44.1kHz

        # Reference audio for voice consistency
        self.reference_audio = None
        self.reference_text = None

        # Self-referencing: Use first segment's voice as reference for consistency
        # This ensures same voice throughout without needing pre-uploaded models
        self.self_reference_audio = None
        self.self_reference_text = None

        if varied_narrators:
            logger.info("Random voices mode: Each segment will have a unique voice")
        else:
            logger.info("Self-referencing mode: Will use first segment's voice for consistency")

    def _check_api_server(self) -> bool:
        """Check if the fish-speech API server is running."""
        try:
            response = requests.get(f"{self.api_url}/v1/health", timeout=1)
            return response.status_code == 200
        except Exception:
            return False

    def _restart_api_server(self) -> bool:
        """Attempt to restart the API server."""

        if self.restart_attempts >= self.max_restart_attempts:
            logger.error(f"Max restart attempts ({self.max_restart_attempts}) reached")
            return False

        self.restart_attempts += 1
        logger.info(
            f"Attempting to restart API server (attempt {self.restart_attempts}/{self.max_restart_attempts})"
        )

        # Kill existing server if running
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except Exception:
                try:
                    self.server_process.kill()
                except Exception:
                    pass
            self.server_process = None

        # Also try to kill any orphaned processes on the port
        try:
            subprocess.run(["fuser", "-k", f"{self.api_port}/tcp"], capture_output=True)
            time.sleep(2)  # Give time for port to be released
        except Exception:
            pass

        # Start new server
        try:
            env = os.environ.copy()
            env["OPENAUDIO_PORT"] = str(self.api_port)

            self.server_process = subprocess.Popen(
                ["bash", "scripts/start_openaudio_server.sh", str(self.api_port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path(__file__).parent.parent,
                env=env,
            )

            # Wait for server to be ready (max 60 seconds)
            for i in range(60):
                if self.server_process.poll() is not None:
                    # Process died
                    stderr = (
                        self.server_process.stderr.read() if self.server_process.stderr else b""
                    )
                    logger.error(f"Server process died: {stderr.decode('utf-8', errors='ignore')}")
                    return False

                if self._check_api_server():
                    logger.info(f"API server restarted successfully on port {self.api_port}")
                    self.api_available = True
                    return True

                if i % 10 == 0 and i > 0:
                    logger.debug(f"Waiting for API server restart... ({i}/60s)")

                time.sleep(1)

            logger.error("API server failed to restart in time")
            return False

        except Exception as e:
            logger.error(f"Failed to restart API server: {e}")
            return False

    def synthesize_segments(self, segments: list[dict]) -> list[AudioSegment]:
        """Synthesize audio for all segments."""
        audio_segments = []
        total_segments = len(segments)

        # Process segments
        start_time = time.time()

        for idx, segment in enumerate(segments):
            segment_num = idx + 1

            # Calculate ETA
            if idx > 0:
                elapsed = time.time() - start_time
                avg_time_per_segment = elapsed / idx
                remaining_segments = total_segments - idx
                eta_seconds = avg_time_per_segment * remaining_segments
                eta_str = f", ETA: {int(eta_seconds//60)}m {int(eta_seconds%60)}s"
            else:
                eta_str = ""

            logger.info(f"Synthesizing segment {segment_num}/{total_segments}{eta_str}")

            try:
                if self.api_available:
                    # Use API to generate speech
                    audio_array = self._synthesize_with_api(
                        segment["text"], is_first_segment=(idx == 0)
                    )
                    if audio_array is not None:
                        # For self-referencing: save first segment as reference (unless varied_narrators is enabled)
                        if (
                            idx == 0
                            and self.self_reference_audio is None
                            and not self.varied_narrators
                        ):
                            self._save_as_self_reference(audio_array, segment["text"])
                    else:
                        # API failed, use fallback
                        audio_array = self._generate_fallback_audio(segment)
                else:
                    # API not available, use fallback
                    audio_array = self._generate_fallback_audio(segment)

                audio_segments.append(
                    AudioSegment(
                        audio=audio_array,
                        sample_rate=self.sample_rate,
                        start=segment["start"],
                        end=segment["end"],
                        text=segment["text"],
                    )
                )

            except Exception as e:
                logger.error(f"Failed to synthesize segment {segment_num}: {e}")
                # Create silent segment as fallback
                duration = segment["end"] - segment["start"]
                num_samples = int(duration * self.sample_rate)
                silent_audio = np.zeros(num_samples)

                audio_segments.append(
                    AudioSegment(
                        audio=silent_audio,
                        sample_rate=self.sample_rate,
                        start=segment["start"],
                        end=segment["end"],
                        text=segment["text"],
                    )
                )

        return audio_segments

    def _save_as_self_reference(self, audio_array: np.ndarray, text: str):
        """Save first segment's audio as reference for subsequent segments."""
        try:
            # Extract first 10 seconds (or less) as reference
            max_samples = min(len(audio_array), self.sample_rate * 10)
            reference_snippet = audio_array[:max_samples]

            # Save to temporary file
            ref_path = tempfile.mktemp(suffix=".wav")
            sf.write(ref_path, reference_snippet, self.sample_rate)

            # Read as binary for API
            with open(ref_path, "rb") as f:
                self.self_reference_audio = f.read()

            self.self_reference_text = text[:100] if len(text) > 100 else text
            os.unlink(ref_path)

            logger.info("Saved first segment as self-reference for voice consistency")
        except Exception as e:
            logger.warning(f"Failed to save self-reference: {e}")

    def _synthesize_with_api(
        self, text: str, is_first_segment: bool = False, retry_count: int = 3
    ) -> np.ndarray | None:
        """Synthesize audio using the fish-speech API with retry logic."""
        last_error = None

        for attempt in range(retry_count):
            try:
                # Check API health before attempting
                if not self.api_available and not self._check_api_server():
                    logger.warning("API server not available, attempting restart...")
                    if not self._restart_api_server():
                        logger.error("Failed to restart API server")
                        return None
                    # Give server a moment to stabilize after restart
                    time.sleep(2)

                # Don't include language tag in the text - it gets spoken!
                formatted_text = text

                # Prepare request based on voice mode
                if (
                    self.self_reference_audio
                    and not is_first_segment
                    and ormsgpack
                    and not self.varied_narrators
                ):
                    # Self-referencing mode: Use first segment's voice for consistency
                    payload = {
                        "text": formatted_text,
                        "format": "wav",
                        "normalize": False,
                        "latency": "normal",
                        "references": [
                            {
                                "audio": self.self_reference_audio,
                                "text": self.self_reference_text,
                            }
                        ],
                    }

                    response = requests.post(
                        f"{self.api_url}/v1/tts",
                        data=ormsgpack.packb(payload),
                        headers={
                            "content-type": "application/msgpack",
                            "model": "s1-mini",
                        },
                        timeout=30,  # Reduced timeout to fail faster
                    )
                else:
                    # First segment or fallback
                    logger.info(
                        "Generating first segment to establish reference voice"
                        if is_first_segment
                        else "No reference available"
                    )
                    payload = {"text": formatted_text, "format": "wav"}
                    response = requests.post(
                        f"{self.api_url}/v1/tts",
                        json=payload,
                        headers={"model": "s1-mini"},
                        timeout=30,  # Reduced timeout to fail faster
                    )

                if response.status_code == 200:
                    # Save response content to temporary file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                        tmp_file.write(response.content)
                        tmp_path = tmp_file.name

                    try:
                        # Load the generated audio
                        waveform, sample_rate = sf.read(tmp_path)

                        # Convert to expected sample rate if needed
                        if sample_rate != self.sample_rate:
                            from scipy import signal

                            waveform = signal.resample(
                                waveform, int(len(waveform) * self.sample_rate / sample_rate)
                            )

                        # Reset restart attempts on success
                        if self.restart_attempts > 0:
                            logger.info("API server recovered, resetting restart counter")
                            self.restart_attempts = 0

                        return waveform
                    finally:
                        # Clean up temp file
                        os.unlink(tmp_path)
                else:
                    logger.error(f"API request failed: {response.status_code}")
                    if response.text:
                        logger.error(f"Response: {response.text}")
                    last_error = f"HTTP {response.status_code}"

            except requests.exceptions.Timeout as e:
                logger.warning(f"API timeout on attempt {attempt + 1}/{retry_count}: {e}")
                last_error = e
                self.api_available = False  # Mark as unavailable to trigger restart

            except requests.exceptions.ConnectionError as e:
                logger.warning(f"API connection error on attempt {attempt + 1}/{retry_count}: {e}")
                last_error = e
                self.api_available = False  # Mark as unavailable to trigger restart

            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}/{retry_count}: {e}")
                last_error = e

            # If not the last attempt, wait before retrying
            if attempt < retry_count - 1:
                wait_time = min(2**attempt, 10)  # Exponential backoff, max 10s
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)

        logger.error(f"Failed to synthesize after {retry_count} attempts. Last error: {last_error}")
        return None

    def _generate_fallback_audio(self, segment: dict) -> np.ndarray:
        """Generate fallback audio (sine wave) when API is not available."""
        logger.debug(f"Using fallback for: {segment['text'][:50]}...")
        duration = segment["end"] - segment["start"]
        num_samples = int(duration * self.sample_rate)

        # Create a simple sine wave as placeholder
        t = np.linspace(0, duration, num_samples)
        frequency = 440  # A4 note
        audio_array = np.sin(2 * np.pi * frequency * t) * 0.1
        return audio_array

    def combine_segments(
        self, audio_segments: list[AudioSegment], output_sample_rate: int = 16000
    ) -> tuple[np.ndarray, int]:
        """Combine audio segments into a single track with proper timing."""
        if not audio_segments:
            return np.zeros(1), output_sample_rate

        # Calculate total duration
        total_duration = audio_segments[-1].end if audio_segments else 0
        total_samples = int(total_duration * output_sample_rate)

        # Create output array
        combined = np.zeros(total_samples)

        # Place each segment at the correct position
        for segment in audio_segments:
            # Resample if needed
            if segment.sample_rate != output_sample_rate:
                from scipy import signal

                audio = signal.resample(
                    segment.audio,
                    int(len(segment.audio) * output_sample_rate / segment.sample_rate),
                )
            else:
                audio = segment.audio

            # Calculate position
            start_sample = int(segment.start * output_sample_rate)
            end_sample = start_sample + len(audio)

            # Ensure we don't exceed bounds
            if end_sample > total_samples:
                audio = audio[: total_samples - start_sample]
                end_sample = total_samples

            # Place the audio
            if start_sample < total_samples:
                combined[start_sample:end_sample] = audio

        return combined, output_sample_rate

    def cleanup(self):
        """Clean up resources."""
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Clean up server process if we started it
        if self.server_process:
            try:
                logger.info("Stopping OpenAudio API server...")
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except Exception:
                try:
                    self.server_process.kill()
                except Exception:
                    pass
            self.server_process = None
