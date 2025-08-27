#!/usr/bin/env python3
"""OpenDub CLI - Local video dubbing with AI."""

import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import click
import structlog

# Disable numba debug logging globally
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("numba.core").setLevel(logging.WARNING)

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from core.pipeline import DubbingPipeline

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer(colors=True)
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)


@click.command()
@click.argument('url')  # YouTube URL or local video file path
@click.option(
    '-l', '--lang',
    multiple=True,
    required=True,
    help='Target language codes (e.g., pt, es, fr). Can specify multiple.'
)
@click.option(
    '-s', '--source-lang',
    default=None,
    help='Source language code (auto-detect if not specified)'
)
@click.option(
    '-o', '--output',
    type=click.Path(path_type=Path),
    default=None,
    help='Output directory (default: ./output)'
)
@click.option(
    '-m', '--whisper-model',
    type=click.Choice(['tiny', 'base', 'small', 'medium', 'large', 'large-v3']),
    default='base',
    help='Whisper model size (default: base)'
)
@click.option(
    '-t', '--tts-model',
    type=click.Choice(['openaudio', 'speecht5', 'auto']),
    default='auto',
    help='TTS model to use (default: auto)'
)
@click.option(
    '-p', '--llm-provider',
    type=click.Choice(['openai', 'anthropic', 'google', 'auto']),
    default='auto',
    help='LLM provider for translation (default: auto-detect)'
)
@click.option(
    '-d', '--device',
    type=click.Choice(['cuda', 'cpu', 'auto']),
    default='auto',
    help='Device to use for processing (default: auto)'
)
@click.option(
    '-k', '--keep-intermediates',
    is_flag=True,
    help='Keep intermediate files (transcripts, translations)'
)
@click.option(
    '-q', '--quiet',
    is_flag=True,
    help='Suppress output except errors'
)
@click.option(
    '-v', '--verbose',
    is_flag=True,
    help='Verbose output'
)
def main(
    url: str,
    lang: tuple,
    source_lang: Optional[str],
    output: Optional[Path],
    whisper_model: str,
    tts_model: str,
    llm_provider: str,
    device: str,
    keep_intermediates: bool,
    quiet: bool,
    verbose: bool
):
    """
    OpenDub - Dub videos locally with AI.
    
    Examples:
    
    \b
    # Dub a YouTube video to Portuguese
    opendub https://youtube.com/watch?v=xyz -l pt
    
    \b
    # Dub to multiple languages
    opendub https://youtube.com/watch?v=xyz -l pt -l es -l fr
    
    \b
    # Dub a local file
    opendub video.mp4 -l pt -s en
    
    \b
    # Use specific models
    opendub https://youtube.com/watch?v=xyz -l pt -m large -t openaudio
    
    \b
    # Process entire playlist (URLs from file)
    cat urls.txt | xargs -I {} opendub {} -l pt
    """
    # Set up logging level
    import logging
    if quiet:
        log_level = logging.ERROR
    elif verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    
    # Configure standard logging first
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s' if verbose else '%(message)s'
    )
    
    logger = structlog.get_logger()
    
    # Convert device option
    if device == 'auto':
        device = None
    
    # Convert provider option
    if llm_provider == 'auto':
        llm_provider = None
    
    # Set output directory
    if output is None:
        output = Path.cwd() / "output"
    
    # Print banner
    if not quiet:
        click.echo(click.style("\n🎬 OpenDub - Local Video Dubbing with AI", fg="cyan", bold=True))
        click.echo(click.style("=" * 50, fg="cyan"))
        click.echo(f"URL: {url}")
        click.echo(f"Languages: {', '.join(lang)}")
        click.echo(f"Output: {output}")
        click.echo(click.style("=" * 50 + "\n", fg="cyan"))
    
    try:
        # Initialize pipeline
        if not quiet:
            click.echo("📦 Initializing pipeline...")
            if tts_model == "openaudio" or tts_model == "auto":
                click.echo("  ⚠️  Note: OpenAudio initialization may take up to 2 minutes on first run")
        
        pipeline = DubbingPipeline(
            output_dir=output,
            whisper_model=whisper_model,
            tts_model=tts_model,
            llm_provider=llm_provider,
            device=device
        )
        
        if not quiet:
            click.echo("✅ Pipeline initialized\n")
        
        # Process video
        if not quiet:
            click.echo("🎥 Processing video...")
        
        results = pipeline.dub_video(
            url=url,
            target_languages=list(lang),
            source_language=source_lang,
            keep_intermediates=keep_intermediates
        )
        
        # Check for errors
        if results.get("status") == "failed":
            click.echo(click.style(f"\n❌ Dubbing failed: {results.get('error')}", fg="red"), err=True)
            sys.exit(1)
        
        # Print results
        if not quiet:
            click.echo(click.style("\n✅ Dubbing completed!", fg="green", bold=True))
            click.echo(click.style("-" * 50, fg="green"))
            
            if results.get("metadata"):
                click.echo(f"Title: {results['metadata'].get('title', 'Unknown')}")
                duration = results['metadata'].get('duration', 0)
                if duration:
                    click.echo(f"Duration: {duration // 60}:{duration % 60:02d}")
            
            click.echo(f"Source language: {results.get('source_language', 'unknown')}")
            click.echo(f"\nDubbed videos:")
            
            for lang_code, video_path in results.get("videos", {}).items():
                size_mb = Path(video_path).stat().st_size / (1024 * 1024)
                click.echo(f"  • {lang_code}: {video_path} ({size_mb:.1f} MB)")
            
            click.echo(click.style("-" * 50, fg="green"))
            click.echo(f"Job ID: {results['job_id']}")
            click.echo(f"Output directory: {results['output_dir']}")
        
        # Output JSON for scripting
        if verbose:
            results_json = json.dumps(results, indent=2)
            click.echo(f"\nFull results (JSON):\n{results_json}")
        
    except KeyboardInterrupt:
        click.echo(click.style("\n⚠️  Dubbing cancelled by user", fg="yellow"), err=True)
        sys.exit(130)
    except Exception as e:
        click.echo(click.style(f"\n❌ Error: {e}", fg="red"), err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()