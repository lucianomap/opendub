"""Download videos and extract audio using yt-dlp."""

import json
import subprocess
import tempfile
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


class VideoDownloader:
    """Download videos from various platforms using yt-dlp."""

    def __init__(self, output_dir: Path | None = None):
        """Initialize downloader with output directory."""
        self.output_dir = output_dir or Path(tempfile.gettempdir()) / "opendub"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_video(self, url: str, job_id: str) -> tuple[Path, dict]:
        """Download video and extract metadata.

        Args:
            url: Video URL to download
            job_id: Unique job identifier

        Returns:
            Tuple of (video_path, metadata)
        """
        output_path = self.output_dir / job_id
        output_path.mkdir(parents=True, exist_ok=True)

        video_file = output_path / "video.mp4"

        # Download video with yt-dlp
        cmd = [
            "yt-dlp",
            "-f",
            "best[ext=mp4]/best",
            "-o",
            str(video_file),
            "--no-playlist",
            "--quiet",
            "--print-json",
            url,
        ]

        logger.info(f"Downloading video from {url}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            metadata = json.loads(result.stdout.strip().split("\n")[-1])

            # Extract useful metadata
            video_info = {
                "title": metadata.get("title", "Unknown"),
                "duration": metadata.get("duration", 0),
                "uploader": metadata.get("uploader", "Unknown"),
                "view_count": metadata.get("view_count", 0),
                "upload_date": metadata.get("upload_date", ""),
                "description": metadata.get("description", "")[:500],  # Truncate
            }

            logger.info(f"Downloaded video: {video_info['title']} ({video_info['duration']}s)")

            return video_file, video_info

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download video: {e.stderr}")
            raise RuntimeError(f"Video download failed: {e.stderr}") from e
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse metadata: {e}")
            # Still return the video if download succeeded
            if video_file.exists():
                return video_file, {"title": "Unknown", "duration": 0}
            raise

    def extract_audio(self, video_path: Path) -> Path:
        """Extract audio from video file.

        Args:
            video_path: Path to video file

        Returns:
            Path to extracted audio file
        """
        audio_path = video_path.with_suffix(".wav")

        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vn",  # No video
            "-acodec",
            "pcm_s16le",  # PCM 16-bit
            "-ar",
            "16000",  # 16kHz sample rate for Whisper
            "-ac",
            "1",  # Mono
            "-y",  # Overwrite output
            str(audio_path),
        ]

        logger.info(f"Extracting audio from {video_path.name}")

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            logger.info(f"Audio extracted to {audio_path.name}")
            return audio_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract audio: {e.stderr}")
            raise RuntimeError(f"Audio extraction failed: {e.stderr}") from e
