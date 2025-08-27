"""Transcribe audio using OpenAI Whisper."""

import json
import logging
from pathlib import Path

import structlog
import torch
import whisper

# Disable numba debug logging
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("numba.core.byteflow").setLevel(logging.WARNING)
logging.getLogger("numba.core.ssa").setLevel(logging.WARNING)
logging.getLogger("numba.core.interpreter").setLevel(logging.WARNING)

logger = structlog.get_logger(__name__)


class WhisperTranscriber:
    """Transcribe audio using OpenAI Whisper."""

    def __init__(self, model_size: str = "base", device: str | None = None):
        """Initialize Whisper model.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3)
            device: Device to use (cuda, cpu, or None for auto)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_size = model_size

        logger.info(f"Loading Whisper {model_size} model on {device}")
        self.model = whisper.load_model(model_size, device=device)

    def transcribe(self, audio_path: Path, language: str | None = None) -> dict:
        """Transcribe audio file.

        Args:
            audio_path: Path to audio file
            language: Optional language code (e.g., 'en', 'pt')

        Returns:
            Transcription result with segments and metadata
        """
        logger.info(f"Transcribing {audio_path.name}")

        # Transcribe with Whisper
        result = self.model.transcribe(
            str(audio_path), language=language, word_timestamps=True, verbose=False
        )

        # Format segments for easier processing
        segments = []
        for seg in result["segments"]:
            segment = {
                "id": seg["id"],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
                "words": seg.get("words", []),
            }
            segments.append(segment)

        transcription = {
            "text": result["text"],
            "segments": segments,
            "language": result.get("language", language),
            "duration": segments[-1]["end"] if segments else 0,
        }

        logger.info(f"Transcribed {len(segments)} segments in {transcription['language']}")

        return transcription

    def save_transcription(self, transcription: dict, output_path: Path) -> Path:
        """Save transcription to JSON file.

        Args:
            transcription: Transcription data
            output_path: Path to save JSON file

        Returns:
            Path to saved file
        """
        json_path = output_path / "transcription.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(transcription, f, ensure_ascii=False, indent=2)

        # Also save as SRT for compatibility
        srt_path = output_path / "transcription.srt"
        self._save_srt(transcription["segments"], srt_path)

        logger.info(f"Saved transcription to {json_path.name}")
        return json_path

    def _save_srt(self, segments: list, srt_path: Path) -> None:
        """Save segments as SRT subtitle file."""
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                start = self._format_timestamp(seg["start"])
                end = self._format_timestamp(seg["end"])
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{seg['text']}\n\n")

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds to SRT timestamp (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
