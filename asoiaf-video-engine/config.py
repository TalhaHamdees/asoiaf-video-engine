"""
Configuration for the ASOIAF Video Engine.
Fill in your API keys and adjust settings as needed.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ElevenLabsConfig:
    api_key: str = "8285089754:2f6179734538506f51624f6c4d59726d552b633869773d3d"
    voice_id: str = "G17SuINrv2H9FC6nvetn"
    template_uuid: str = "089d3a91-1902-42fa-b06c-7c5098bd8a52"  # "Christopher" template
    base_url: str = "https://voiceapi.csv666.ru"
    model_id: str = "eleven_multilingual_v2"
    stability: float = 0.5
    similarity_boost: float = 0.75
    style: float = 0.0
    speed: float = 1.0
    poll_interval: float = 3.0  # seconds between status checks
    poll_timeout: float = 300.0  # max wait time in seconds


@dataclass
class ImageSearchConfig:
    # Anthropic API key for entity extraction + auto-tagging
    anthropic_api_key: str = ""
    # Library paths
    local_library_path: Path = Path("assets/images")
    # Matching
    match_threshold: float = 5.0   # min score to accept a library match
    # Image timing
    target_interval_sec: float = 3.5  # new image every ~3.5 seconds
    min_interval_sec: float = 2.0
    max_interval_sec: float = 5.0


@dataclass
class AudioConfig:
    silence_thresh_dbfs: int = -40
    min_silence_ms: int = 400
    keep_silence_ms: int = 150
    normalize_target_dbfs: float = -20.0


@dataclass
class CaptionConfig:
    font_path: str = "assets/fonts/Poppins-Bold.ttf"
    font_size: int = 70
    font_color: str = "#FFFFFF"
    stroke_color: str = "#000000"
    stroke_width: int = 5
    highlight_color: str = "#FF4444"  # Red highlight for current word
    words_per_group: int = 2
    position_y_ratio: float = 0.78
    style: str = "highlight"  # "srt" or "highlight"
    uppercase: bool = True


@dataclass
class VideoConfig:
    width: int = 1080
    height: int = 1920
    fps: int = 30
    ken_burns_zoom_range: tuple = (1.0, 1.15)
    ken_burns_pan_pixels: int = 80
    crossfade_duration: float = 0.3
    output_codec: str = "libx264"
    output_bitrate: str = "8M"
    audio_bitrate: str = "192k"


@dataclass
class WhisperConfig:
    model_size: str = "base"
    language: str = "en"
    device: str = "cpu"


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
    input_dir: Path = Path("input/images")
    shopping_list_path: Path = Path("output/shopping_list.txt")
    state_file: Path = Path("temp/pipeline_state.json")
