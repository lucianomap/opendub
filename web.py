#!/usr/bin/env python3
"""OpenDub Web - Minimal web interface for video dubbing."""

import asyncio
import json
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import structlog
import uvicorn
from dotenv import load_dotenv

# Disable numba debug logging globally
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("numba.core").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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

logger = structlog.get_logger(__name__)

# Store pipeline as global (will be initialized in lifespan)
pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    # Startup
    global pipeline
    logger.info("Initializing pipeline...")
    pipeline = DubbingPipeline()
    logger.info("Pipeline initialized")
    yield
    # Shutdown
    logger.info("Shutting down...")
    # Cleanup if needed

# Create FastAPI app with lifespan
app = FastAPI(
    title="OpenDub",
    description="Local video dubbing with AI",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store job status in memory (for simplicity)
jobs = {}


class DubRequest(BaseModel):
    """Request model for dubbing."""
    url: str
    languages: List[str]
    source_language: Optional[str] = None
    whisper_model: str = "base"
    tts_model: str = "auto"


class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: Optional[str] = None
    results: Optional[dict] = None
    error: Optional[str] = None


# Startup event handler removed - now using lifespan context manager


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main web interface."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    
    # Return embedded HTML if static file doesn't exist
    return HTMLResponse(content=DEFAULT_HTML)


@app.post("/api/dub", response_model=JobStatus)
async def start_dubbing(request: DubRequest, background_tasks: BackgroundTasks):
    """Start a dubbing job."""
    job_id = str(uuid.uuid4())[:8]
    
    # Initialize job status
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": "Job queued",
        "results": None,
        "error": None
    }
    
    # Start dubbing in background
    background_tasks.add_task(
        process_dubbing,
        job_id,
        request.url,
        request.languages,
        request.source_language,
        request.whisper_model,
        request.tts_model
    )
    
    logger.info(f"Started job {job_id}")
    return JobStatus(**jobs[job_id])


async def process_dubbing(
    job_id: str,
    url: str,
    languages: List[str],
    source_language: Optional[str],
    whisper_model: str,
    tts_model: str
):
    """Process dubbing job in background."""
    try:
        # Update status
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = "Starting dubbing process..."
        
        # Run dubbing (in thread pool to avoid blocking)
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            pipeline.dub_video,
            url,
            languages,
            source_language,
            False  # Don't keep intermediates
        )
        
        # Update job with results
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = "Dubbing completed"
        jobs[job_id]["results"] = results
        
        logger.info(f"Job {job_id} completed")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.get("/api/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """Get job status."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatus(**jobs[job_id])


@app.get("/api/download/{job_id}/{language}")
async def download_video(job_id: str, language: str):
    """Download dubbed video."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    videos = job["results"].get("videos", {})
    if language not in videos:
        raise HTTPException(status_code=404, detail="Language not found")
    
    video_path = Path(videos[language])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"dubbed_{language}.mp4"
    )


@app.get("/api/supported-languages")
async def supported_languages():
    """Get list of supported languages."""
    return {
        "languages": [
            {"code": "pt", "name": "Portuguese"},
            {"code": "es", "name": "Spanish"},
            {"code": "fr", "name": "French"},
            {"code": "de", "name": "German"},
            {"code": "it", "name": "Italian"},
            {"code": "ja", "name": "Japanese"},
            {"code": "ko", "name": "Korean"},
            {"code": "zh", "name": "Chinese"},
            {"code": "ru", "name": "Russian"},
            {"code": "ar", "name": "Arabic"},
            {"code": "hi", "name": "Hindi"},
            {"code": "nl", "name": "Dutch"},
            {"code": "pl", "name": "Polish"},
            {"code": "tr", "name": "Turkish"},
            {"code": "sv", "name": "Swedish"},
        ]
    }


# Default HTML if static file is missing
DEFAULT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpenDub - Local Video Dubbing</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #333; min-height: 100vh; display: flex; justify-content: center; align-items: center; }
        .container { background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); max-width: 600px; width: 90%; }
        h1 { color: #667eea; margin-bottom: 0.5rem; font-size: 2rem; }
        .subtitle { color: #666; margin-bottom: 2rem; }
        .form-group { margin-bottom: 1.5rem; }
        label { display: block; margin-bottom: 0.5rem; font-weight: 500; }
        input, select { width: 100%; padding: 0.75rem; border: 2px solid #e1e8ed; border-radius: 8px; font-size: 1rem; transition: border-color 0.3s; }
        input:focus, select:focus { outline: none; border-color: #667eea; }
        .checkbox-group { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 0.5rem; }
        .checkbox-item { display: flex; align-items: center; }
        .checkbox-item input { width: auto; margin-right: 0.5rem; }
        button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 0.75rem 2rem; border: none; border-radius: 8px; font-size: 1rem; font-weight: 600; cursor: pointer; transition: transform 0.2s; width: 100%; }
        button:hover { transform: translateY(-2px); }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .status { margin-top: 2rem; padding: 1rem; border-radius: 8px; background: #f7f9fb; }
        .status.processing { border-left: 4px solid #667eea; }
        .status.completed { border-left: 4px solid #48bb78; }
        .status.failed { border-left: 4px solid #f56565; }
        .downloads { margin-top: 1rem; }
        .download-link { display: inline-block; margin: 0.5rem 0.5rem 0 0; padding: 0.5rem 1rem; background: #667eea; color: white; text-decoration: none; border-radius: 6px; }
        .spinner { border: 3px solid #f3f3f3; border-top: 3px solid #667eea; border-radius: 50%; width: 30px; height: 30px; animation: spin 1s linear infinite; display: inline-block; margin-right: 0.5rem; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 OpenDub</h1>
        <p class="subtitle">Local AI-powered video dubbing</p>
        
        <form id="dubForm">
            <div class="form-group">
                <label for="url">Video URL</label>
                <input type="url" id="url" placeholder="https://youtube.com/watch?v=..." required>
            </div>
            
            <div class="form-group">
                <label>Target Languages</label>
                <div class="checkbox-group" id="languages">
                    <div class="checkbox-item">
                        <input type="checkbox" id="pt" value="pt">
                        <label for="pt">Portuguese</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="es" value="es">
                        <label for="es">Spanish</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="fr" value="fr">
                        <label for="fr">French</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="de" value="de">
                        <label for="de">German</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="it" value="it">
                        <label for="it">Italian</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="ja" value="ja">
                        <label for="ja">Japanese</label>
                    </div>
                </div>
            </div>
            
            <button type="submit" id="submitBtn">Start Dubbing</button>
        </form>
        
        <div id="status"></div>
    </div>
    
    <script>
        const form = document.getElementById('dubForm');
        const statusDiv = document.getElementById('status');
        const submitBtn = document.getElementById('submitBtn');
        
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const url = document.getElementById('url').value;
            const checkboxes = document.querySelectorAll('#languages input:checked');
            const languages = Array.from(checkboxes).map(cb => cb.value);
            
            if (languages.length === 0) {
                alert('Please select at least one language');
                return;
            }
            
            submitBtn.disabled = true;
            statusDiv.innerHTML = '<div class="status processing"><div class="spinner"></div>Starting dubbing process...</div>';
            
            try {
                const response = await fetch('/api/dub', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url, languages })
                });
                
                const job = await response.json();
                pollStatus(job.job_id);
                
            } catch (error) {
                statusDiv.innerHTML = `<div class="status failed">Error: ${error.message}</div>`;
                submitBtn.disabled = false;
            }
        });
        
        async function pollStatus(jobId) {
            const interval = setInterval(async () => {
                try {
                    const response = await fetch(`/api/status/${jobId}`);
                    const status = await response.json();
                    
                    if (status.status === 'completed') {
                        clearInterval(interval);
                        showResults(status);
                        submitBtn.disabled = false;
                    } else if (status.status === 'failed') {
                        clearInterval(interval);
                        statusDiv.innerHTML = `<div class="status failed">Error: ${status.error}</div>`;
                        submitBtn.disabled = false;
                    } else {
                        statusDiv.innerHTML = `<div class="status processing"><div class="spinner"></div>${status.progress || 'Processing...'}</div>`;
                    }
                } catch (error) {
                    clearInterval(interval);
                    statusDiv.innerHTML = `<div class="status failed">Error checking status</div>`;
                    submitBtn.disabled = false;
                }
            }, 2000);
        }
        
        function showResults(status) {
            const videos = status.results.videos;
            const downloads = Object.entries(videos).map(([lang, path]) => 
                `<a href="/api/download/${status.job_id}/${lang}" class="download-link" download>Download ${lang.toUpperCase()}</a>`
            ).join('');
            
            statusDiv.innerHTML = `
                <div class="status completed">
                    ✅ Dubbing completed!
                    <div class="downloads">${downloads}</div>
                </div>
            `;
        }
    </script>
</body>
</html>
"""


def main():
    """Run the web server."""
    uvicorn.run(
        "web:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    )


if __name__ == "__main__":
    main()