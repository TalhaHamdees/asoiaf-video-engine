"""
Video Composer — assembles final video with Ken Burns effects,
karaoke-style highlight captions, motion blur transitions, and watermark.
"""

import logging
import subprocess
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from config import VideoConfig, CaptionConfig, WatermarkConfig
from modules.script_parser import ScriptSegment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ken Burns
# ---------------------------------------------------------------------------

class KenBurnsEffect(Enum):
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    ZOOM_IN_PAN_RIGHT = "zoom_in_pan_right"
    ZOOM_IN_PAN_LEFT = "zoom_in_pan_left"
    ZOOM_OUT_PAN_RIGHT = "zoom_out_pan_right"
    ZOOM_OUT_PAN_LEFT = "zoom_out_pan_left"


def _pick_effect(index: int) -> KenBurnsEffect:
    # Favor zoom-in variants (matches reference style)
    effects = [
        KenBurnsEffect.ZOOM_IN,
        KenBurnsEffect.ZOOM_IN_PAN_RIGHT,
        KenBurnsEffect.ZOOM_IN,
        KenBurnsEffect.ZOOM_IN_PAN_LEFT,
        KenBurnsEffect.ZOOM_OUT,
        KenBurnsEffect.ZOOM_IN,
        KenBurnsEffect.ZOOM_OUT_PAN_RIGHT,
        KenBurnsEffect.ZOOM_IN,
    ]
    return effects[index % len(effects)]


def prepare_image(image_path: str, target_w: int, target_h: int, padding: float = 1.35) -> np.ndarray:
    """Load and crop image to target size with padding for Ken Burns room."""
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
    tw: int, th: int, zoom_range=(1.0, 1.25), pan_px: int = 50,
) -> np.ndarray:
    progress = max(0.0, min(1.0, t / duration if duration > 0 else 0))
    # Ease-in-out for smoother motion
    progress = progress * progress * (3 - 2 * progress)

    sh, sw = src.shape[:2]
    cx, cy = sw / 2, sh / 2
    z_min, z_max = zoom_range
    zoom, ox, oy = 1.0, 0.0, 0.0

    if effect == KenBurnsEffect.ZOOM_IN:
        zoom = z_min + (z_max - z_min) * progress
    elif effect == KenBurnsEffect.ZOOM_OUT:
        zoom = z_max - (z_max - z_min) * progress
    elif effect == KenBurnsEffect.ZOOM_IN_PAN_RIGHT:
        zoom = z_min + (z_max - z_min) * progress
        ox = pan_px * progress
    elif effect == KenBurnsEffect.ZOOM_IN_PAN_LEFT:
        zoom = z_min + (z_max - z_min) * progress
        ox = -pan_px * progress
    elif effect == KenBurnsEffect.ZOOM_OUT_PAN_RIGHT:
        zoom = z_max - (z_max - z_min) * progress
        ox = pan_px * progress
    elif effect == KenBurnsEffect.ZOOM_OUT_PAN_LEFT:
        zoom = z_max - (z_max - z_min) * progress
        ox = -pan_px * progress

    cw, ch = int(tw / zoom), int(th / zoom)
    x1 = max(0, int(cx + ox) - cw // 2)
    y1 = max(0, int(cy + oy) - ch // 2)
    x2, y2 = min(sw, x1 + cw), min(sh, y1 + ch)
    if x2 - x1 < cw: x1 = max(0, x2 - cw)
    if y2 - y1 < ch: y1 = max(0, y2 - ch)

    cropped = src[y1:y2, x1:x2]
    return np.array(Image.fromarray(cropped).resize((tw, th), Image.LANCZOS))


# ---------------------------------------------------------------------------
# Motion blur transition
# ---------------------------------------------------------------------------

def apply_motion_blur(frame: np.ndarray, strength: float) -> np.ndarray:
    """Apply horizontal motion blur for whip-cut transition effect."""
    if strength <= 0:
        return frame
    kernel_size = max(3, int(strength * 80))
    if kernel_size % 2 == 0:
        kernel_size += 1
    img = Image.fromarray(frame)
    img = img.filter(ImageFilter.BoxBlur(kernel_size))
    # Also apply directional stretch by squeezing vertically slightly
    return np.array(img)


def create_whip_blur_frame(
    prev_frame: np.ndarray, next_frame: np.ndarray,
    progress: float, tw: int, th: int,
) -> np.ndarray:
    """Create a single whip-pan transition frame blending two images with motion blur."""
    # Heavy horizontal motion blur
    blur_strength = 1.0 - abs(progress - 0.5) * 2  # Peak at 0.5
    kernel_w = max(3, int(blur_strength * 120))

    if progress < 0.5:
        base = prev_frame
    else:
        base = next_frame

    img = Image.fromarray(base)
    # Horizontal-only motion blur via resize trick
    small = img.resize((max(1, tw // (kernel_w + 1)), th), Image.BILINEAR)
    blurred = small.resize((tw, th), Image.BILINEAR)
    return np.array(blurred)


# ---------------------------------------------------------------------------
# Captions (karaoke highlight)
# ---------------------------------------------------------------------------

def _build_word_groups(word_timestamps: list, words_per_group: int) -> list[dict]:
    """Pre-build word groups for highlight captions."""
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
        display_words.append(text)

    # Measure total width
    space_width = font.getlength(" ")
    word_widths = [font.getlength(dw) for dw in display_words]
    total_width = sum(word_widths) + space_width * (len(display_words) - 1)

    # Get font metrics for vertical centering
    bbox = font.getbbox("Ay")
    text_height = bbox[3] - bbox[1]

    # Position: centered horizontally, at position_y_ratio vertically
    y_pos = int(h * caption_config.position_y_ratio) - text_height // 2
    x_start = (w_frame - total_width) / 2

    stroke_w = caption_config.stroke_width
    font_color = caption_config.font_color
    highlight_color = caption_config.highlight_color
    stroke_color = caption_config.stroke_color

    # Draw each word
    x = x_start
    for i, dw in enumerate(display_words):
        color = highlight_color if i == highlight_idx else font_color
        draw.text(
            (x, y_pos), dw, font=font, fill=color,
            stroke_width=stroke_w, stroke_fill=stroke_color,
        )
        x += word_widths[i] + space_width

    return np.array(img)


# ---------------------------------------------------------------------------
# Watermark
# ---------------------------------------------------------------------------

def _render_watermark(
    frame: np.ndarray, watermark_config: WatermarkConfig,
    font: ImageFont.FreeTypeFont | None,
) -> np.ndarray:
    """Render persistent channel watermark onto frame."""
    if not watermark_config.enabled or font is None:
        return frame

    h, w = frame.shape[:2]
    img = Image.fromarray(frame)

    # Create transparent overlay for opacity
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    text = watermark_config.channel_name
    bbox = font.getbbox(text)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (w - text_w) // 2
    y = int(h * watermark_config.position_y_ratio) - text_h // 2

    # Parse color
    base_r, base_g, base_b = _hex_to_rgb(watermark_config.base_color)
    alpha = watermark_config.opacity

    # Draw with slight stroke for readability
    draw.text(
        (x, y), text, font=font,
        fill=(base_r, base_g, base_b, alpha),
        stroke_width=2,
        stroke_fill=(0, 0, 0, alpha // 2),
    )

    # Composite
    img_rgba = img.convert("RGBA")
    composited = Image.alpha_composite(img_rgba, overlay)
    return np.array(composited.convert("RGB"))


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


# ---------------------------------------------------------------------------
# SRT (fallback)
# ---------------------------------------------------------------------------

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
    logger.info("Generated SRT: %d groups -> %s", len(groups), output_path)
    return output_path


def _srt_time(sec: float) -> str:
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    ms = int((s % 1) * 1000)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"


# ---------------------------------------------------------------------------
# Main composition
# ---------------------------------------------------------------------------

def compose_video(
    segments: list[ScriptSegment], audio_path: Path, word_timestamps: list,
    config: VideoConfig, caption_config: CaptionConfig, output_path: Path,
    watermark_config: WatermarkConfig | None = None,
) -> Path:
    from moviepy import ImageClip, AudioFileClip, VideoClip, concatenate_videoclips

    logger.info("Composing video (%d segments, %dfps)...", len(segments), config.fps)

    # Pre-build caption data
    use_highlight = caption_config.style == "highlight" and word_timestamps
    word_groups = []
    caption_font = None

    if use_highlight:
        word_groups = _build_word_groups(word_timestamps, caption_config.words_per_group)
        font_path = Path(caption_config.font_path)
        if font_path.exists():
            caption_font = ImageFont.truetype(str(font_path), caption_config.font_size)
        else:
            logger.warning("Font %s not found, using default.", font_path)
            caption_font = ImageFont.load_default()
        logger.info("Highlight captions: %d word groups, font=%s", len(word_groups), font_path.name)

    # Watermark font
    wm_font = None
    if watermark_config and watermark_config.enabled:
        wm_font_path = Path(watermark_config.font_path)
        if wm_font_path.exists():
            wm_font = ImageFont.truetype(str(wm_font_path), watermark_config.font_size)
        else:
            logger.warning("Watermark font %s not found.", wm_font_path)

    # Prepare segment images and data
    seg_data = []
    for seg in segments:
        if seg.duration <= 0:
            continue
        effect = _pick_effect(seg.index)
        if seg.image_path and Path(seg.image_path).exists():
            src = prepare_image(seg.image_path, config.width, config.height)
        else:
            src = None
        seg_data.append((seg, src, effect))

    if not seg_data:
        raise ValueError("No segments to compose.")

    # Transition duration in seconds
    blur_frames = getattr(config, "transition_blur_frames", 4)
    transition_sec = blur_frames / config.fps

    clips = []
    for seg_idx, (seg, src, effect) in enumerate(seg_data):
        dur = seg.duration
        audio_start = seg.start_time

        # Get prev/next frames for transitions
        prev_src = seg_data[seg_idx - 1][1] if seg_idx > 0 else None
        next_src = seg_data[seg_idx + 1][1] if seg_idx < len(seg_data) - 1 else None

        def make_frame_fn(
            s=src, e=effect, d=dur, a_start=audio_start,
            wg=word_groups, cc=caption_config, cf=caption_font,
            hl=use_highlight, wmc=watermark_config, wmf=wm_font,
            p_src=prev_src, trans_dur=transition_sec,
            vid_w=config.width, vid_h=config.height,
            zoom=config.ken_burns_zoom_range, pan=config.ken_burns_pan_pixels,
        ):
            def make_frame(t):
                # Check if we're in the transition-in zone (first few frames)
                if t < trans_dur and p_src is not None:
                    progress = t / trans_dur
                    # Get end frame of previous segment
                    prev_frame = apply_ken_burns(
                        p_src, 1.0, 1.0, KenBurnsEffect.ZOOM_IN,
                        vid_w, vid_h, zoom, pan,
                    )
                    if s is not None:
                        cur_frame = apply_ken_burns(s, 0.0, d, e, vid_w, vid_h, zoom, pan)
                    else:
                        cur_frame = np.full((vid_h, vid_w, 3), (30, 30, 40), dtype=np.uint8)
                    frame = create_whip_blur_frame(prev_frame, cur_frame, progress, vid_w, vid_h)
                else:
                    if s is not None:
                        frame = apply_ken_burns(s, t, d, e, vid_w, vid_h, zoom, pan)
                    else:
                        frame = np.full((vid_h, vid_w, 3), (30, 30, 40), dtype=np.uint8)

                # Captions
                if hl and cf is not None:
                    abs_time = a_start + t
                    frame = _render_highlight_caption(frame, abs_time, wg, cc, cf)

                # Watermark
                if wmc and wmf is not None:
                    frame = _render_watermark(frame, wmc, wmf)

                return frame
            return make_frame

        clip = VideoClip(make_frame_fn(), duration=dur).with_fps(config.fps)
        clips.append(clip)

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
    try:
        srt_rel = Path(srt_path).relative_to(Path.cwd())
        srt_str = str(srt_rel).replace("\\", "/")
    except ValueError:
        srt_str = str(srt_path).replace("\\", "/")
    srt_escaped = srt_str.replace(":", "\\:")

    font_path = Path(config.font_path)
    if font_path.exists():
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
        stderr_lines = result.stderr.strip().split("\n")
        error_lines = [l for l in stderr_lines if "error" in l.lower() or "unable" in l.lower() or "invalid" in l.lower()]
        error_msg = "\n".join(error_lines) if error_lines else result.stderr[-500:]
        raise RuntimeError(f"FFmpeg subtitle burn failed: {error_msg}")
    logger.info("Captioned video: %s", output_path)
    return Path(output_path)
