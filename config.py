"""
Configuration for the ASOIAF Video Engine.
Copy this file, fill in your API keys, and update paths as needed.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ElevenLabsConfig:
    api_key: str = "YOUR_ELEVENLABS_API_KEY"
    voice_id: str = "YOUR_VOICE_ID"  # e.g. "pNInz6obpgDQGcFmaJgB" (Adam)
    model_id: str = "eleven_multilingual_v2"
    stability: float = 0.5
    similarity_boost: float = 0.75
    style: float = 0.0
    speed: float = 1.0


@dataclass
class ImageSearchConfig:
    # Google Custom Search (optional — leave empty to skip web search)
    google_api_key: str = ""
    google_cx: str = ""  # Custom Search Engine ID
    # Bing Image Search (alternative)
    bing_api_key: str = ""
    # Local library path
    local_library_path: Path = Path("assets/images")
    # How many web images to fetch per segment as fallback
    web_results_per_query: int = 3
    # Anthropic API key for generating search queries from script
    anthropic_api_key: str = ""


@dataclass
class AudioConfig:
    silence_thresh_dbfs: int = -40  # dBFS threshold for silence detection
    min_silence_ms: int = 400       # silences shorter than this are kept
    keep_silence_ms: int = 150      # pad kept around non-silent chunks
    normalize_target_dbfs: float = -20.0


@dataclass
class CaptionConfig:
    font_path: str = "assets/fonts/Montserrat-Bold.ttf"
    font_size: int = 60
    font_color: str = "white"
    stroke_color: str = "black"
    stroke_width: int = 3
    highlight_color: str = "#FFD700"  # gold highlight for active word
    words_per_group: int = 3          # words per caption group
    position_y_ratio: float = 0.75    # vertical position (0=top, 1=bottom)
    # Style: "srt" for basic subtitles, "highlight" for word-by-word
    style: str = "srt"


@dataclass
class VideoConfig:
    width: int = 1080
    height: int = 1920
    fps: int = 30
    # Ken Burns settings
    ken_burns_zoom_range: tuple = (1.0, 1.15)  # min/max zoom
    ken_burns_pan_pixels: int = 80              # max pan offset in px
    # Transitions
    crossfade_duration: float = 0.3  # seconds between image clips
    # Output
    output_codec: str = "libx264"
    output_bitrate: str = "8M"
    audio_bitrate: str = "192k"


@dataclass
class WhisperConfig:
    model_size: str = "base"  # tiny, base, small, medium, large-v3
    language: str = "en"
    device: str = "cpu"  # "cuda" if GPU available


@dataclass
class Config:
    elevenlabs: ElevenLabsConfig = field(default_factory=ElevenLabsConfig)
    image_search: ImageSearchConfig = field(default_factory=ImageSearchConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    caption: CaptionConfig = field(default_factory=CaptionConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    temp_dir: Path = Path("temp")
    output_dir: Path = Path("output")
