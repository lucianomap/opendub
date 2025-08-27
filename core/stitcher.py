"""Stitch dubbed audio with original video."""

import subprocess
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


class VideoStitcher:
    """Combine dubbed audio with original video."""

    def stitch(self, video_path: Path, audio_path: Path, output_path: Path, language: str) -> Path:
        """Replace video audio with dubbed version.

        Args:
            video_path: Path to original video
            audio_path: Path to dubbed audio
            output_path: Directory to save output video
            language: Language code for output filename

        Returns:
            Path to dubbed video file
        """
        output_file = output_path / f"dubbed_{language}.mp4"

        # FFmpeg command to replace audio
        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),  # Original video
            "-i",
            str(audio_path),  # New audio
            "-c:v",
            "copy",  # Copy video codec
            "-c:a",
            "aac",  # AAC audio codec
            "-b:a",
            "192k",  # Audio bitrate
            "-map",
            "0:v:0",  # Use video from first input
            "-map",
            "1:a:0",  # Use audio from second input
            "-shortest",  # Match shortest stream
            "-y",  # Overwrite output
            str(output_file),
        ]

        logger.info(f"Stitching dubbed audio for {language}")

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Check file was created
            if not output_file.exists():
                raise RuntimeError("Output file was not created")

            # Get file size for logging
            size_mb = output_file.stat().st_size / (1024 * 1024)
            logger.info(f"Created dubbed video: {output_file.name} ({size_mb:.1f} MB)")

            return output_file

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed: {e.stderr}")
            raise RuntimeError(f"Video stitching failed: {e.stderr}") from e
