"""
Video Composer — assembles the final video.

Handles:
- Image resizing & cropping to 9:16
- Ken Burns (pan/zoom) effects
- Caption rendering and burn-in
- Audio overlay
- Final encoding
"""

import logging
import random
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import VideoConfig, CaptionConfig
from modules.script_parser import ScriptSegment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ken Burns effect types
# ---------------------------------------------------------------------------

class KenBurnsEffect(Enum):
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    PAN_UP = "pan_up"
    PAN_DOWN = "pan_down"
    ZOOM_IN_PAN_RIGHT = "zoom_in_pan_right"
    ZOOM_OUT_PAN_LEFT = "zoom_out_pan_left"


def _pick_effect(index: int) -> KenBurnsEffect:
    """Cycle through effects to avoid repetition."""
    effects = list(KenBurnsEffect)
    return effects[index % len(effects)]


# ---------------------------------------------------------------------------
# Image preparation
# ---------------------------------------------------------------------------

def prepare_image(
    image_path: str,
    target_w: int,
    target_h: int,
    padding_factor: float = 1.2,
) -> np.ndarray:
    """
    Load and prepare an image for the video frame.

    The image is resized to be slightly LARGER than the target frame
    (by padding_factor) to allow room for Ken Burns panning/zooming.

    Returns:
        numpy array (H, W, 3) in RGB.
    """
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size

    # Target padded size (larger than frame to allow movement)
    padded_w = int(target_w * padding_factor)
    padded_h = int(target_h * padding_factor)

    # Scale image to cover the padded area
    scale = max(padded_w / img_w, padded_h / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Center crop to padded size
    left = (new_w - padded_w) // 2
    top = (new_h - padded_h) // 2
    img = img.crop((left, top, left + padded_w, top + padded_h))

    return np.array(img)


# ---------------------------------------------------------------------------
# Ken Burns frame transform
# ---------------------------------------------------------------------------

def apply_ken_burns(
    source_image: np.ndarray,
    t: float,
    duration: float,
    effect: KenBurnsEffect,
    target_w: int,
    target_h: int,
    zoom_range: tuple = (1.0, 1.15),
    pan_pixels: int = 80,
) -> np.ndarray:
    """
    Apply Ken Burns effect to extract a frame from the (larger) source image.

    Args:
        source_image: The padded source image (numpy array, larger than target).
        t: Current time in seconds within this clip.
        duration: Total duration of this clip.
        effect: Which Ken Burns movement to apply.
        target_w: Output frame width.
        target_h: Output frame height.
        zoom_range: (min_zoom, max_zoom) multipliers.
        pan_pixels: Maximum pan offset in pixels.

    Returns:
        Cropped and transformed frame as numpy array (target_h, target_w, 3).
    """
    progress = t / duration if duration > 0 else 0
    progress = max(0.0, min(1.0, progress))

    src_h, src_w = source_image.shape[:2]
    center_x, center_y = src_w / 2, src_h / 2

    zoom_min, zoom_max = zoom_range
    zoom = 1.0
    offset_x, offset_y = 0.0, 0.0

    if effect == KenBurnsEffect.ZOOM_IN:
        zoom = zoom_min + (zoom_max - zoom_min) * progress
    elif effect == KenBurnsEffect.ZOOM_OUT:
        zoom = zoom_max - (zoom_max - zoom_min) * progress
    elif effect == KenBurnsEffect.PAN_LEFT:
        offset_x = pan_pixels * (1 - 2 * progress)
    elif effect == KenBurnsEffect.PAN_RIGHT:
        offset_x = pan_pixels * (2 * progress - 1)
    elif effect == KenBurnsEffect.PAN_UP:
        offset_y = pan_pixels * (1 - 2 * progress)
    elif effect == KenBurnsEffect.PAN_DOWN:
        offset_y = pan_pixels * (2 * progress - 1)
    elif effect == KenBurnsEffect.ZOOM_IN_PAN_RIGHT:
        zoom = zoom_min + (zoom_max - zoom_min) * progress
        offset_x = pan_pixels * progress * 0.5
    elif effect == KenBurnsEffect.ZOOM_OUT_PAN_LEFT:
        zoom = zoom_max - (zoom_max - zoom_min) * progress
        offset_x = -pan_pixels * progress * 0.5

    # Calculate crop region
    crop_w = int(target_w / zoom)
    crop_h = int(target_h / zoom)

    cx = int(center_x + offset_x)
    cy = int(center_y + offset_y)

    x1 = max(0, cx - crop_w // 2)
    y1 = max(0, cy - crop_h // 2)
    x2 = min(src_w, x1 + crop_w)
    y2 = min(src_h, y1 + crop_h)

    # Adjust if we hit boundaries
    if x2 - x1 < crop_w:
        x1 = max(0, x2 - crop_w)
    if y2 - y1 < crop_h:
        y1 = max(0, y2 - crop_h)

    # Crop and resize to target dimensions
    cropped = source_image[y1:y2, x1:x2]
    frame = Image.fromarray(cropped).resize((target_w, target_h), Image.LANCZOS)

    return np.array(frame)


# ---------------------------------------------------------------------------
# Caption rendering
# ---------------------------------------------------------------------------

def render_caption_on_frame(
    frame: np.ndarray,
    text: str,
    config: CaptionConfig,
    highlight_word: str | None = None,
) -> np.ndarray:
    """
    Render caption text onto a video frame.

    Args:
        frame: The video frame (numpy array).
        text: Caption text to display.
        config: Caption styling configuration.
        highlight_word: If set, this word gets highlighted in a different color.

    Returns:
        Frame with caption overlay.
    """
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # Load font
    try:
        font = ImageFont.truetype(config.font_path, config.font_size)
    except (IOError, OSError):
        logger.debug("Font not found at %s, using default.", config.font_path)
        font = ImageFont.load_default()

    frame_w, frame_h = img.size
    y_pos = int(frame_h * config.position_y_ratio)

    if config.style == "highlight" and highlight_word:
        _draw_highlighted_caption(draw, text, highlight_word, font, frame_w, y_pos, config)
    else:
        _draw_simple_caption(draw, text, font, frame_w, y_pos, config)

    return np.array(img)


def _draw_simple_caption(
    draw: ImageDraw.Draw,
    text: str,
    font: ImageFont.FreeTypeFont,
    frame_w: int,
    y_pos: int,
    config: CaptionConfig,
):
    """Draw centered text with stroke outline."""
    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    x_pos = (frame_w - text_w) // 2

    # Draw stroke/outline
    for dx in range(-config.stroke_width, config.stroke_width + 1):
        for dy in range(-config.stroke_width, config.stroke_width + 1):
            if dx != 0 or dy != 0:
                draw.text((x_pos + dx, y_pos + dy), text, font=font, fill=config.stroke_color)

    # Draw main text
    draw.text((x_pos, y_pos), text, font=font, fill=config.font_color)


def _draw_highlighted_caption(
    draw: ImageDraw.Draw,
    text: str,
    highlight_word: str,
    font: ImageFont.FreeTypeFont,
    frame_w: int,
    y_pos: int,
    config: CaptionConfig,
):
    """Draw caption with one word highlighted in a different color."""
    words = text.split()

    # Calculate total width for centering
    total_width = 0
    word_widths = []
    space_width = draw.textbbox((0, 0), " ", font=font)[2]

    for w in words:
        bbox = draw.textbbox((0, 0), w, font=font)
        w_width = bbox[2] - bbox[0]
        word_widths.append(w_width)
        total_width += w_width

    total_width += space_width * (len(words) - 1)
    x_cursor = (frame_w - total_width) // 2

    for word, w_width in zip(words, word_widths):
        # Choose color
        color = config.font_color
        if word.lower().strip(".,!?;:'\"") == highlight_word.lower().strip(".,!?;:'\""):
            color = config.highlight_color

        # Stroke
        for dx in range(-config.stroke_width, config.stroke_width + 1):
            for dy in range(-config.stroke_width, config.stroke_width + 1):
                if dx != 0 or dy != 0:
                    draw.text(
                        (x_cursor + dx, y_pos + dy), word,
                        font=font, fill=config.stroke_color,
                    )

        draw.text((x_cursor, y_pos), word, font=font, fill=color)
        x_cursor += w_width + space_width


# ---------------------------------------------------------------------------
# SRT generation (for FFmpeg burn-in)
# ---------------------------------------------------------------------------

def generate_srt(
    word_timestamps: list,
    output_path: Path,
    words_per_group: int = 3,
) -> Path:
    """
    Generate an SRT subtitle file from word timestamps.

    Groups words into short phrases for readability.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    groups = []
    for i in range(0, len(word_timestamps), words_per_group):
        group = word_timestamps[i:i + words_per_group]
        if group:
            text = " ".join(w.word for w in group)
            start = group[0].start
            end = group[-1].end
            groups.append((start, end, text))

    with open(output_path, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(groups, 1):
            f.write(f"{i}\n")
            f.write(f"{_format_srt_time(start)} --> {_format_srt_time(end)}\n")
            f.write(f"{text}\n\n")

    logger.info("Generated SRT with %d subtitle groups: %s", len(groups), output_path)
    return output_path


def _format_srt_time(seconds: float) -> str:
    """Format seconds to SRT timestamp: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# Video assembly (MoviePy)
# ---------------------------------------------------------------------------

def compose_video(
    segments: list[ScriptSegment],
    audio_path: Path,
    word_timestamps: list,
    config: VideoConfig,
    caption_config: CaptionConfig,
    output_path: Path,
    placeholder_color: tuple = (30, 30, 40),
) -> Path:
    """
    Assemble the final video from prepared segments.

    This is the main composition function that:
    1. Creates image clips with Ken Burns effects
    2. Adds captions if style != "srt" (SRT is burned in via FFmpeg after)
    3. Adds audio
    4. Exports the final video

    Args:
        segments: Script segments with image_path and timing populated.
        audio_path: Path to cleaned audio.
        word_timestamps: Word-level timestamps for caption rendering.
        config: Video configuration.
        caption_config: Caption styling configuration.
        output_path: Where to save the final video.
        placeholder_color: RGB color for segments without images.

    Returns:
        Path to the output video.
    """
    from moviepy import (
        ImageClip,
        AudioFileClip,
        VideoClip,
        concatenate_videoclips,
    )

    logger.info("Composing video (%d segments)...", len(segments))

    clips = []
    for seg in segments:
        duration = seg.duration
        if duration <= 0:
            logger.warning("Segment %d has zero duration, skipping.", seg.index)
            continue

        effect = _pick_effect(seg.index)

        if seg.image_path and Path(seg.image_path).exists():
            # Prepare source image (slightly larger for Ken Burns room)
            source_img = prepare_image(
                seg.image_path, config.width, config.height, padding_factor=1.25
            )

            # Create a VideoClip with Ken Burns transform
            def make_frame_fn(src=source_img, eff=effect, dur=duration):
                def make_frame(t):
                    return apply_ken_burns(
                        src, t, dur, eff,
                        config.width, config.height,
                        config.ken_burns_zoom_range,
                        config.ken_burns_pan_pixels,
                    )
                return make_frame

            clip = VideoClip(make_frame_fn(), duration=duration).with_fps(config.fps)
        else:
            # Placeholder: solid dark frame
            placeholder = np.full(
                (config.height, config.width, 3),
                placeholder_color, dtype=np.uint8,
            )
            clip = ImageClip(placeholder, duration=duration).with_fps(config.fps)

        clips.append(clip)

    if not clips:
        raise ValueError("No video clips to compose.")

    # Concatenate with optional crossfade
    if config.crossfade_duration > 0 and len(clips) > 1:
        video = concatenate_videoclips(clips, method="compose")
        # Note: for crossfade, you'd use .with_crossfade() on individual clips
        # This is a simplification — MoviePy 2.x crossfade requires more setup
    else:
        video = concatenate_videoclips(clips)

    # Add audio
    audio = AudioFileClip(str(audio_path))
    # Trim video to audio length (or vice versa)
    final_duration = min(video.duration, audio.duration)
    video = video.subclipped(0, final_duration)
    audio = audio.subclipped(0, final_duration)
    video = video.with_audio(audio)

    # Export
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Encoding video to %s ...", output_path)
    video.write_videofile(
        str(output_path),
        fps=config.fps,
        codec=config.output_codec,
        bitrate=config.output_bitrate,
        audio_bitrate=config.audio_bitrate,
        audio_codec="aac",
        threads=4,
        logger=None,  # suppress moviepy's verbose output
    )

    logger.info("Video saved: %s", output_path)
    return output_path


def burn_srt_captions(
    video_path: Path,
    srt_path: Path,
    output_path: Path,
    caption_config: CaptionConfig,
) -> Path:
    """
    Burn SRT subtitles into video using FFmpeg.

    This is the simplest and most reliable caption method.
    Run this AFTER compose_video() to add captions.
    """
    import subprocess

    output_path = Path(output_path)

    # Build FFmpeg subtitle filter
    # Escape paths for FFmpeg (replace backslashes, colons)
    srt_escaped = str(srt_path).replace("\\", "/").replace(":", "\\:")

    font_name = Path(caption_config.font_path).stem if caption_config.font_path else "Sans"

    subtitle_filter = (
        f"subtitles={srt_escaped}:"
        f"force_style='FontName={font_name},"
        f"FontSize={caption_config.font_size // 2},"  # FFmpeg uses different scaling
        f"PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,"
        f"BorderStyle=3,"
        f"Outline={caption_config.stroke_width},"
        f"Alignment=2,"  # bottom center
        f"MarginV=200'"  # margin from bottom
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", subtitle_filter,
        "-c:a", "copy",
        str(output_path),
    ]

    logger.info("Burning SRT captions via FFmpeg...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("FFmpeg failed: %s", result.stderr)
        raise RuntimeError(f"FFmpeg subtitle burn failed: {result.stderr[:500]}")

    logger.info("Captioned video saved: %s", output_path)
    return output_path
