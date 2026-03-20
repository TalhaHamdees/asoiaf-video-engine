"""
FFmpeg helper — provides a bundled ffmpeg binary via imageio-ffmpeg,
so the user doesn't need a system-wide FFmpeg install.
"""

import os
import logging

logger = logging.getLogger(__name__)


def get_ffmpeg_path() -> str:
    """Return the path to the bundled ffmpeg binary and ensure it's on PATH."""
    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        logger.warning("imageio-ffmpeg not installed, falling back to system ffmpeg")
        ffmpeg_path = "ffmpeg"

    # Also set env vars so moviepy / pydub / subprocess pick it up
    os.environ["FFMPEG_BINARY"] = ffmpeg_path
    ffmpeg_dir = os.path.dirname(ffmpeg_path)
    if ffmpeg_dir and ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

    return ffmpeg_path


def ensure_ffmpeg():
    """Call once at startup to configure ffmpeg for the whole process."""
    path = get_ffmpeg_path()
    logger.info("Using ffmpeg: %s", path)
    return path
