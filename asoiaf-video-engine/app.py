#!/usr/bin/env python3
"""
ASOIAF Video Engine — Desktop App (NiceGUI)

Launch with: python app.py
Opens a browser-based GUI at http://localhost:8080
"""

import logging
import socket
import sys
from pathlib import Path

from nicegui import ui, app

from config import Config
from modules.ffmpeg_helper import ensure_ffmpeg
from pipeline_runner import PipelineProgress

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")

# Shared state
config = Config()
progress = PipelineProgress()

# Ensure FFmpeg is available at startup
try:
    ensure_ffmpeg()
except Exception as e:
    logger.warning("FFmpeg setup issue: %s", e)

# Serve local image files so the UI can display them
app.add_static_files("/assets", str(Path("assets").resolve()))
app.add_static_files("/input", str(Path("input").resolve()))
app.add_static_files("/output", str(Path("output").resolve()))
# Serve library images
library_path = config.image_search.local_library_path
if library_path.exists():
    app.add_static_files("/library-images", str(library_path.resolve()))


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
body {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}
</style>
"""


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@ui.page("/")
def page_dashboard():
    _page_wrapper()
    from ui.dashboard import create
    create(config)


@ui.page("/new")
def page_new_video():
    _page_wrapper()
    from ui.new_video import create
    create(config, progress)


@ui.page("/progress")
def page_progress():
    _page_wrapper()
    from ui.progress import create
    create(config, progress)


@ui.page("/shopping-list")
def page_shopping_list():
    _page_wrapper()
    from ui.shopping_list import create
    create(config, progress)


@ui.page("/library")
def page_library():
    _page_wrapper()
    from ui.image_library import create
    create(config)


@ui.page("/settings")
def page_settings():
    _page_wrapper()
    from ui.settings import create
    create(config)


@ui.page("/preview")
def page_preview():
    _page_wrapper()
    from nicegui import app as _app
    path = ui.context.client.request.query_params.get("path", "")
    with ui.column().classes("w-full max-w-4xl mx-auto p-6 gap-4"):
        with ui.row().classes("w-full items-center gap-4"):
            ui.button(icon="arrow_back", on_click=lambda: ui.navigate.to("/")).props("flat")
            ui.label("Video Preview").classes("text-2xl font-bold")

        if path and Path(path).exists():
            ui.video(path).classes("w-full rounded-lg")
            ui.label(Path(path).name).classes("text-gray-500")
        else:
            ui.label("Video file not found.").classes("text-red-500")


def _page_wrapper():
    """Common wrapper for all pages: dark mode toggle, nav."""
    ui.add_head_html(CUSTOM_CSS)
    ui.colors(primary="#DC2626")  # Red theme matching FICTOPIA brand


# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------

app.on_exception(lambda e: logger.error("Unhandled exception: %s", e))


@app.exception_handler(500)
async def handle_500(request, exc):
    """Show a friendly error page instead of a raw 500."""
    logger.error("Server error: %s", exc)
    return {"detail": str(exc)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def find_free_port(start: int = 8080, max_attempts: int = 20) -> int:
    """Find an available port starting from `start`."""
    for port in range(start, start + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start  # fallback


if __name__ in {"__main__", "__mp_main__"}:
    port = find_free_port(8080)
    if port != 8080:
        logger.info("Port 8080 in use, using port %d", port)

    ui.run(
        title="ASOIAF Video Engine",
        port=port,
        show=True,
        reload=False,
        storage_secret="asoiaf-video-engine-2024",
    )
