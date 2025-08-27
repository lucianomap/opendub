"""Main dubbing pipeline orchestrator."""

import json
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

import structlog

from .downloader import VideoDownloader
from .transcriber import WhisperTranscriber
from .translator import LLMTranslator
from .synthesizer import SpeechSynthesizer
from .stitcher import VideoStitcher

logger = structlog.get_logger(__name__)


class DubbingPipeline:
    """Orchestrate the complete dubbing pipeline."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        whisper_model: str = "base",
        tts_model: str = "auto",
        llm_provider: Optional[str] = None,
        device: Optional[str] = None
    ):
        """Initialize pipeline with all components.
        
        Args:
            output_dir: Directory for output files
            whisper_model: Whisper model size
            tts_model: TTS model to use
            llm_provider: LLM provider for translation
            device: Device to use (cuda/cpu)
        """
        self.output_dir = output_dir or Path(tempfile.gettempdir()) / "opendub"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.downloader = VideoDownloader(self.output_dir)
        self.transcriber = WhisperTranscriber(whisper_model, device)
        self.translator = LLMTranslator(llm_provider)
        self.synthesizer = SpeechSynthesizer(tts_model, device)
        self.stitcher = VideoStitcher()

    def dub_video(
        self,
        url: str,
        target_languages: List[str],
        source_language: Optional[str] = None,
        keep_intermediates: bool = False
    ) -> dict:
        """Dub a video into multiple languages.
        
        Args:
            url: Video URL or local file path
            target_languages: List of target language codes
            source_language: Optional source language code
            keep_intermediates: Keep intermediate files
            
        Returns:
            Dictionary with job results and file paths
        """
        # Generate job ID
        job_id = str(uuid.uuid4())[:8]
        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting dubbing job {job_id}")
        
        results = {
            "job_id": job_id,
            "source_url": url,
            "output_dir": str(job_dir),
            "videos": {},
            "metadata": {}
        }
        
        try:
            # Step 1: Download video
            logger.info("Step 1/5: Downloading video")
            if url.startswith(("http://", "https://")):
                video_path, metadata = self.downloader.download_video(url, job_id)
                results["metadata"] = metadata
            else:
                # Local file
                video_path = Path(url)
                if not video_path.exists():
                    raise FileNotFoundError(f"Video file not found: {url}")
                results["metadata"] = {"title": video_path.stem, "duration": 0}
            
            # Step 2: Extract audio
            logger.info("Step 2/5: Extracting audio")
            audio_path = self.downloader.extract_audio(video_path)
            
            # Step 3: Transcribe
            logger.info("Step 3/5: Transcribing audio")
            transcription = self.transcriber.transcribe(audio_path, source_language)
            
            if keep_intermediates:
                self.transcriber.save_transcription(transcription, job_dir)
            
            detected_language = transcription.get("language", "unknown")
            logger.info(f"Detected language: {detected_language}")
            results["source_language"] = detected_language
            
            # Step 4 & 5: Translate and synthesize for each target language
            for lang in target_languages:
                logger.info(f"Step 4/5: Translating to {lang}")
                
                # Skip if same as source language
                if lang == detected_language:
                    logger.info(f"Skipping {lang} (same as source)")
                    results["videos"][lang] = str(video_path)
                    continue
                
                # Translate
                translation = self.translator.translate(transcription, lang)
                
                if keep_intermediates:
                    self.translator.save_translation(translation, job_dir, lang)
                
                # Synthesize
                logger.info(f"Step 5/5: Synthesizing speech for {lang}")
                audio_path = self.synthesizer.synthesize(translation, job_dir, lang)
                
                # Stitch with video
                dubbed_video = self.stitcher.stitch(video_path, audio_path, job_dir, lang)
                
                results["videos"][lang] = str(dubbed_video)
                logger.info(f"Completed dubbing for {lang}")
            
            # Clean up intermediates if requested
            if not keep_intermediates:
                self._cleanup_intermediates(job_dir, results["videos"])
            
            logger.info(f"Job {job_id} completed successfully")
            
            # Save results
            results_file = job_dir / "results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            return results
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            results["error"] = str(e)
            results["status"] = "failed"
            return results

    def _cleanup_intermediates(self, job_dir: Path, keep_files: dict):
        """Remove intermediate files, keeping only final videos."""
        keep_paths = {Path(p) for p in keep_files.values()}
        
        for file_path in job_dir.iterdir():
            if file_path.is_file() and file_path not in keep_paths:
                if file_path.suffix not in ['.mp4', '.json']:
                    try:
                        file_path.unlink()
                    except Exception as e:
                        logger.debug(f"Could not remove {file_path}: {e}")