"""Text-to-speech synthesis using OpenAudio S1-mini model via fish-speech."""

import os
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

    def __init__(self, language: str = "pt", device: str = "cuda"):
        self.language = language
        self.device = device if torch.cuda.is_available() else "cpu"

        # Model path - relative to the module file
        module_dir = Path(__file__).parent.parent  # Goes up to project root
        self.model_dir = module_dir / "models" / "openaudio-s1-mini"

        # API server settings - configurable via environment variables
        self.api_host = os.getenv("OPENAUDIO_API_HOST", "localhost")
        self.api_port = int(os.getenv("OPENAUDIO_API_PORT", "8180"))
        self.api_url = f"http://{self.api_host}:{self.api_port}"

        # Check if API server is running
        self.api_available = self._check_api_server()

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

        logger.info("Self-referencing mode: Will use first segment's voice for consistency")

    def _check_api_server(self) -> bool:
        """Check if the fish-speech API server is running."""
        try:
            response = requests.get(f"{self.api_url}/v1/health", timeout=1)
            return response.status_code == 200
        except Exception:
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
                        # For self-referencing: save first segment as reference
                        if idx == 0 and self.self_reference_audio is None:
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

    def _synthesize_with_api(self, text: str, is_first_segment: bool = False) -> np.ndarray | None:
        """Synthesize audio using the fish-speech API."""
        try:
            # Don't include language tag in the text - it gets spoken!
            formatted_text = text

            # Prepare request based on voice mode
            if self.self_reference_audio and not is_first_segment and ormsgpack:
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
                    timeout=60,
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
                    timeout=60,
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

                    return waveform
                finally:
                    # Clean up temp file
                    os.unlink(tmp_path)
            else:
                logger.error(f"API request failed: {response.status_code}")
                if response.text:
                    logger.error(f"Response: {response.text}")
                return None

        except Exception as e:
            logger.error(f"Failed to synthesize with API: {e}")
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
