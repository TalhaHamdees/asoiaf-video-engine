"""
Video Composer — assembles final video with Ken Burns effects and captions.
"""

import logging
import subprocess
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image

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


def compose_video(
    segments: list[ScriptSegment], audio_path: Path, word_timestamps: list,
    config: VideoConfig, caption_config: CaptionConfig, output_path: Path,
) -> Path:
    from moviepy import ImageClip, AudioFileClip, VideoClip, concatenate_videoclips

    logger.info("Composing video (%d segments)...", len(segments))
    clips = []

    for seg in segments:
        dur = seg.duration
        if dur <= 0:
            continue
        effect = _pick_effect(seg.index)

        if seg.image_path and Path(seg.image_path).exists():
            src = prepare_image(seg.image_path, config.width, config.height)

            def make_frame_fn(s=src, e=effect, d=dur):
                def make_frame(t):
                    return apply_ken_burns(
                        s, t, d, e, config.width, config.height,
                        config.ken_burns_zoom_range, config.ken_burns_pan_pixels,
                    )
                return make_frame

            clip = VideoClip(make_frame_fn(), duration=dur).with_fps(config.fps)
        else:
            placeholder = np.full((config.height, config.width, 3), (30, 30, 40), dtype=np.uint8)
            clip = ImageClip(placeholder, duration=dur).with_fps(config.fps)

        clips.append(clip)

    if not clips:
        raise ValueError("No clips to compose.")

    video = concatenate_videoclips(clips)
    audio = AudioFileClip(str(audio_path))
    final_dur = min(video.duration, audio.duration)
    video = video.subclipped(0, final_dur).with_audio(audio.subclipped(0, final_dur))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Encoding → %s ...", output_path)
    video.write_videofile(
        str(output_path), fps=config.fps, codec=config.output_codec,
        bitrate=config.output_bitrate, audio_bitrate=config.audio_bitrate,
        audio_codec="aac", threads=4, logger=None,
    )
    logger.info("Video saved: %s", output_path)
    return output_path


def burn_srt_captions(video_path: Path, srt_path: Path, output_path: Path, config: CaptionConfig) -> Path:
    srt_escaped = str(srt_path).replace("\\", "/").replace(":", "\\:")
    font = Path(config.font_path).stem if config.font_path else "Sans"
    vf = (
        f"subtitles={srt_escaped}:force_style='FontName={font},"
        f"FontSize={config.font_size // 2},PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,BorderStyle=3,Outline={config.stroke_width},"
        f"Alignment=2,MarginV=200'"
    )
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vf", vf, "-c:a", "copy", str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr[:500]}")
    logger.info("Captioned video: %s", output_path)
    return Path(output_path)
