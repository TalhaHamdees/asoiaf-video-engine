"""
Video Composer — assembles final video with Ken Burns effects and captions.
"""

import logging
import subprocess
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import VideoConfig, CaptionConfig
from modules.script_parser import ScriptSegment

logger = logging.getLogger(__name__)


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
    effects = list(KenBurnsEffect)
    return effects[index % len(effects)]


def prepare_image(image_path: str, target_w: int, target_h: int, padding: float = 1.25) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    pw, ph = int(target_w * padding), int(target_h * padding)
    scale = max(pw / img.width, ph / img.height)
    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    left = (img.width - pw) // 2
    top = (img.height - ph) // 2
    img = img.crop((left, top, left + pw, top + ph))
    return np.array(img)


def apply_ken_burns(
    src: np.ndarray, t: float, duration: float, effect: KenBurnsEffect,
    tw: int, th: int, zoom_range=(1.0, 1.15), pan_px: int = 80,
) -> np.ndarray:
    progress = max(0.0, min(1.0, t / duration if duration > 0 else 0))
    sh, sw = src.shape[:2]
    cx, cy = sw / 2, sh / 2
    z_min, z_max = zoom_range
    zoom, ox, oy = 1.0, 0.0, 0.0

    if effect == KenBurnsEffect.ZOOM_IN:
        zoom = z_min + (z_max - z_min) * progress
    elif effect == KenBurnsEffect.ZOOM_OUT:
        zoom = z_max - (z_max - z_min) * progress
    elif effect == KenBurnsEffect.PAN_LEFT:
        ox = pan_px * (1 - 2 * progress)
    elif effect == KenBurnsEffect.PAN_RIGHT:
        ox = pan_px * (2 * progress - 1)
    elif effect == KenBurnsEffect.PAN_UP:
        oy = pan_px * (1 - 2 * progress)
    elif effect == KenBurnsEffect.PAN_DOWN:
        oy = pan_px * (2 * progress - 1)
    elif effect == KenBurnsEffect.ZOOM_IN_PAN_RIGHT:
        zoom = z_min + (z_max - z_min) * progress
        ox = pan_px * progress * 0.5
    elif effect == KenBurnsEffect.ZOOM_OUT_PAN_LEFT:
        zoom = z_max - (z_max - z_min) * progress
        ox = -pan_px * progress * 0.5

    cw, ch = int(tw / zoom), int(th / zoom)
    x1 = max(0, int(cx + ox) - cw // 2)
    y1 = max(0, int(cy + oy) - ch // 2)
    x2, y2 = min(sw, x1 + cw), min(sh, y1 + ch)
    if x2 - x1 < cw: x1 = max(0, x2 - cw)
    if y2 - y1 < ch: y1 = max(0, y2 - ch)

    cropped = src[y1:y2, x1:x2]
    return np.array(Image.fromarray(cropped).resize((tw, th), Image.LANCZOS))


def generate_srt(word_timestamps: list, output_path: Path, words_per_group: int = 3) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    groups = []
    for i in range(0, len(word_timestamps), words_per_group):
        grp = word_timestamps[i:i + words_per_group]
        if grp:
            groups.append((grp[0].start, grp[-1].end, " ".join(w.word for w in grp)))

    with open(output_path, "w", encoding="utf-8") as f:
        for i, (s, e, txt) in enumerate(groups, 1):
            f.write(f"{i}\n{_srt_time(s)} --> {_srt_time(e)}\n{txt}\n\n")
    logger.info("Generated SRT: %d groups → %s", len(groups), output_path)
    return output_path


def _srt_time(sec: float) -> str:
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    ms = int((s % 1) * 1000)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"


def _build_word_groups(word_timestamps: list, words_per_group: int) -> list[dict]:
    """
    Pre-build word groups for highlight captions.

    Each group: {
        "start": float, "end": float,
        "words": [{"word": str, "start": float, "end": float}, ...]
    }
    """
    groups = []
    for i in range(0, len(word_timestamps), words_per_group):
        grp = word_timestamps[i:i + words_per_group]
        if grp:
            groups.append({
                "start": grp[0].start,
                "end": grp[-1].end,
                "words": [{"word": w.word, "start": w.start, "end": w.end} for w in grp],
            })
    return groups


def _render_highlight_caption(
    frame: np.ndarray, t: float,
    word_groups: list[dict], caption_config: CaptionConfig, font: ImageFont.FreeTypeFont,
) -> np.ndarray:
    """Render karaoke-style highlight captions onto a frame at time t."""
    # Find active word group
    active_group = None
    for grp in word_groups:
        if grp["start"] <= t <= grp["end"]:
            active_group = grp
            break

    if active_group is None:
        return frame

    words = active_group["words"]
    uppercase = getattr(caption_config, "uppercase", True)

    # Determine which word is currently highlighted
    highlight_idx = 0
    for i, w in enumerate(words):
        if t >= w["start"]:
            highlight_idx = i

    h, w_frame = frame.shape[:2]
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # Build display strings
    display_words = []
    for w in words:
        text = w["word"].upper() if uppercase else w["word"]
        # Clean punctuation attached to word for cleaner look
        display_words.append(text)

    # Measure total width
    space_width = font.getlength(" ")
    word_widths = [font.getlength(dw) for dw in display_words]
    total_width = sum(word_widths) + space_width * (len(display_words) - 1)

    # Position: centered horizontally, at position_y_ratio vertically
    y_pos = int(h * caption_config.position_y_ratio)
    x_start = (w_frame - total_width) / 2

    stroke_w = caption_config.stroke_width
    font_color = caption_config.font_color
    highlight_color = caption_config.highlight_color
    stroke_color = caption_config.stroke_color

    # Draw each word
    x = x_start
    for i, dw in enumerate(display_words):
        color = highlight_color if i == highlight_idx else font_color
        # Draw text with stroke
        draw.text(
            (x, y_pos), dw, font=font, fill=color,
            stroke_width=stroke_w, stroke_fill=stroke_color,
        )
        x += word_widths[i] + space_width

    return np.array(img)


def compose_video(
    segments: list[ScriptSegment], audio_path: Path, word_timestamps: list,
    config: VideoConfig, caption_config: CaptionConfig, output_path: Path,
) -> Path:
    from moviepy import ImageClip, AudioFileClip, VideoClip, concatenate_videoclips

    logger.info("Composing video (%d segments)...", len(segments))

    # Pre-build caption data if using highlight style
    use_highlight = caption_config.style == "highlight" and word_timestamps
    word_groups = []
    font = None

    if use_highlight:
        word_groups = _build_word_groups(word_timestamps, caption_config.words_per_group)
        font_path = Path(caption_config.font_path)
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), caption_config.font_size)
        else:
            logger.warning("Font %s not found, using default.", font_path)
            font = ImageFont.load_default()
        logger.info("Highlight captions: %d word groups, font=%s", len(word_groups), font_path.name)

    # Track cumulative time offset for each segment
    time_offsets = []
    cumulative = 0.0
    for seg in segments:
        time_offsets.append((cumulative, seg.start_time))
        cumulative += seg.duration

    clips = []
    for seg_idx, seg in enumerate(segments):
        dur = seg.duration
        if dur <= 0:
            continue
        effect = _pick_effect(seg.index)
        clip_time_offset, seg_audio_start = time_offsets[seg_idx]

        if seg.image_path and Path(seg.image_path).exists():
            src = prepare_image(seg.image_path, config.width, config.height)
        else:
            src = None

        def make_frame_fn(
            s=src, e=effect, d=dur, audio_start=seg_audio_start,
            wg=word_groups, cc=caption_config, f=font, hl=use_highlight,
        ):
            def make_frame(t):
                if s is not None:
                    frame = apply_ken_burns(
                        s, t, d, e, config.width, config.height,
                        config.ken_burns_zoom_range, config.ken_burns_pan_pixels,
                    )
                else:
                    frame = np.full((config.height, config.width, 3), (30, 30, 40), dtype=np.uint8)

                if hl and f is not None:
                    # Map clip-local time to absolute audio time
                    abs_time = audio_start + t
                    frame = _render_highlight_caption(frame, abs_time, wg, cc, f)

                return frame
            return make_frame

        clip = VideoClip(make_frame_fn(), duration=dur).with_fps(config.fps)
        clips.append(clip)

    if not clips:
        raise ValueError("No clips to compose.")

    video = concatenate_videoclips(clips)
    audio = AudioFileClip(str(audio_path))
    final_dur = min(video.duration, audio.duration)
    video = video.subclipped(0, final_dur).with_audio(audio.subclipped(0, final_dur))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Encoding -> %s ...", output_path)
    video.write_videofile(
        str(output_path), fps=config.fps, codec=config.output_codec,
        bitrate=config.output_bitrate, audio_bitrate=config.audio_bitrate,
        audio_codec="aac", threads=4, logger=None,
    )
    logger.info("Video saved: %s", output_path)
    return output_path


def burn_srt_captions(video_path: Path, srt_path: Path, output_path: Path, config: CaptionConfig) -> Path:
    # FFmpeg subtitles filter needs forward slashes and escaped colons.
    # Try relative path first (avoids Windows drive letter issues),
    # fall back to absolute with proper escaping.
    try:
        srt_rel = Path(srt_path).relative_to(Path.cwd())
        srt_str = str(srt_rel).replace("\\", "/")
    except ValueError:
        srt_str = str(srt_path).replace("\\", "/")
    srt_escaped = srt_str.replace(":", "\\:")

    font_path = Path(config.font_path)
    if font_path.exists():
        # Use FontFile for reliable font loading
        try:
            font_rel = font_path.relative_to(Path.cwd())
            font_file = str(font_rel).replace("\\", "/")
        except ValueError:
            font_file = str(font_path).replace("\\", "/")
        font_file_escaped = font_file.replace(":", "\\:")
        font_style = f"Fontfile={font_file_escaped}"
    else:
        font_style = f"FontName={font_path.stem}"

    vf = (
        f"subtitles={srt_escaped}:force_style='{font_style},"
        f"FontSize={config.font_size // 2},PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,BorderStyle=3,Outline={config.stroke_width},"
        f"Alignment=2,MarginV=200'"
    )
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vf", vf, "-c:a", "copy", str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Extract actual error from stderr (skip ffmpeg banner)
        stderr_lines = result.stderr.strip().split("\n")
        error_lines = [l for l in stderr_lines if "error" in l.lower() or "unable" in l.lower() or "invalid" in l.lower()]
        error_msg = "\n".join(error_lines) if error_lines else result.stderr[-500:]
        raise RuntimeError(f"FFmpeg subtitle burn failed: {error_msg}")
    logger.info("Captioned video: %s", output_path)
    return Path(output_path)
