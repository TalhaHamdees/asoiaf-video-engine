"""
Run the pipeline with a pre-recorded voiceover and a folder of in-order media
(images and/or video clips).

Skips TTS. Whisper-transcribes the existing audio for word-level timing, splits
the timeline into N slots (N = number of media) paced for short-form (2-3s per
visual, cuts snapped to word gaps), maps media 1..N to slots in order, then
composes the video. Each visual is cropped subject-aware (faces -> saliency ->
center) so the character/subject stays in focus instead of being center-cropped
out of frame, and Ken Burns motion is aimed at the subject.

Usage:
    python run_existing.py --audio input/existing_audio/strongest_hand.mp3 \
                           --images input/images --title strongest_hand

--images may contain images (.jpg/.png/.webp) and clips (.mp4/.mov/...).
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

from config import Config
from modules.ffmpeg_helper import ensure_ffmpeg
from modules.script_parser import ScriptSegment
from modules.transcriber import transcribe_with_timestamps, WordTimestamp
from modules.audio_processor import normalize_audio, get_audio_duration
from modules.video_composer import compose_video, generate_srt, burn_srt_captions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}

# Short-form pacing: every visual stays on screen 2-3s (max 3s).
MAX_SLOT_SEC = 3.0
TARGET_SLOT_SEC = 2.5
SNAP_TOLERANCE_SEC = 0.35  # how far a cut may move to land on a word gap


def collect_ordered_media(media_dir: Path) -> list[Path]:
    """Collect images and video clips in numeric/alphabetical order."""
    extensions = IMAGE_EXTS | VIDEO_EXTS
    files = [p for p in media_dir.iterdir() if p.suffix.lower() in extensions]

    def sort_key(p: Path):
        stem = p.stem
        try:
            return (0, int(stem))
        except ValueError:
            return (1, stem.lower())

    return sorted(files, key=sort_key)


def _snap_to_word_gap(t: float, word_starts: list[float], lo: float) -> float:
    """Snap a cut time to the nearest word start within tolerance, staying > lo."""
    candidates = [s for s in word_starts if abs(s - t) <= SNAP_TOLERANCE_SEC and s > lo + 0.2]
    return min(candidates, key=lambda s: abs(s - t)) if candidates else t


def _slot_text(words: list[WordTimestamp], start: float, end: float) -> str:
    grp = [w for w in words if start <= ((w.start + w.end) / 2) < end]
    return " ".join(w.word for w in grp).strip() or "..."


def build_schedule(
    words: list[WordTimestamp],
    media: list[Path],
    total_duration: float,
) -> list[ScriptSegment]:
    """
    Map media 1:1 to equally-spaced, word-gap-snapped slots, then enforce the
    2-3s pacing: any slot longer than MAX_SLOT_SEC is split into N equal
    sub-shots of the SAME media (N = ceil(dur / TARGET_SLOT_SEC)), so no visual
    ever stays on screen longer than ~3s. Sub-shots get an increasing
    `variation` so the composer can punch in (wide -> tight on the subject)
    instead of holding a frozen frame.
    """
    n = len(media)
    if n <= 0:
        raise ValueError("need at least one media item")

    word_starts = sorted(w.start for w in words)

    # Base 1:1 boundaries, interior cuts snapped to word gaps.
    bounds = [i * total_duration / n for i in range(n + 1)]
    bounds[0], bounds[-1] = 0.0, total_duration
    for i in range(1, n):
        bounds[i] = _snap_to_word_gap(bounds[i], word_starts, bounds[i - 1])

    segments: list[ScriptSegment] = []
    idx = 0
    for i in range(n):
        s_start, s_end = bounds[i], bounds[i + 1]
        dur = s_end - s_start
        media_path = str(media[i])

        if dur <= MAX_SLOT_SEC + 0.05:
            segments.append(ScriptSegment(
                index=idx, text=_slot_text(words, s_start, s_end),
                start_time=s_start, end_time=s_end, image_path=media_path,
            ))
            idx += 1
            continue

        # Split this overlong slot into equal <=3s sub-shots of the same media.
        k = max(2, -(-int(dur * 100) // int(TARGET_SLOT_SEC * 100)))  # ceil(dur/target)
        sub = [s_start + j * dur / k for j in range(k + 1)]
        sub[-1] = s_end
        for j in range(1, k):
            sub[j] = _snap_to_word_gap(sub[j], word_starts, sub[j - 1])
        for j in range(k):
            segments.append(ScriptSegment(
                index=idx, text=_slot_text(words, sub[j], sub[j + 1]),
                start_time=sub[j], end_time=sub[j + 1], image_path=media_path,
                variation=j,
            ))
            idx += 1
    return segments


def main():
    parser = argparse.ArgumentParser(description="Run pipeline with existing voiceover + images.")
    parser.add_argument("--audio", required=True, type=Path, help="Path to existing voiceover (mp3/wav).")
    parser.add_argument("--images", required=True, type=Path, help="Folder containing ordered images.")
    parser.add_argument("--title", default="video", help="Output filename (no extension).")
    parser.add_argument("--no-normalize", action="store_true", help="Skip loudness normalization (keeps original mp3 levels).")
    args = parser.parse_args()

    ensure_ffmpeg()
    config = Config()
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    audio_in = args.audio.resolve()
    if not audio_in.exists():
        logger.error("Audio not found: %s", audio_in)
        sys.exit(1)

    media = collect_ordered_media(args.images.resolve())
    if not media:
        logger.error("No media (images/clips) found in %s", args.images)
        sys.exit(1)
    n_vid = sum(1 for p in media if p.suffix.lower() in VIDEO_EXTS)
    logger.info("Found %d media items (%d images, %d clips).", len(media), len(media) - n_vid, n_vid)
    for p in media:
        kind = "clip" if p.suffix.lower() in VIDEO_EXTS else "img"
        logger.info("  [%s] %s", kind, p.name)

    work_audio = config.temp_dir / "voiceover_clean.wav"
    if work_audio.exists():
        work_audio.unlink()
    if args.no_normalize:
        from pydub import AudioSegment
        AudioSegment.from_file(str(audio_in)).export(str(work_audio), format="wav")
    else:
        logger.info("Normalizing audio to %.1f dBFS...", config.audio.normalize_target_dbfs)
        work_audio = normalize_audio(audio_in, config.audio.normalize_target_dbfs, work_audio)
    duration = get_audio_duration(work_audio)
    logger.info("Audio duration: %.2fs", duration)

    logger.info("Transcribing with Whisper (model=%s)...", config.whisper.model_size)
    word_timestamps = transcribe_with_timestamps(work_audio, config.whisper)
    logger.info("Got %d word timestamps.", len(word_timestamps))

    # Pacing sanity check: warn if the media count won't hold 2-3s per visual.
    avg_slot = duration / len(media)
    ideal_count = round(duration / TARGET_SLOT_SEC)
    if avg_slot > MAX_SLOT_SEC:
        logger.warning(
            "Only %d media for %.1fs VO -> avg %.2fs/visual exceeds the %.1fs max. "
            "Provide ~%d media for 2-3s pacing.",
            len(media), duration, avg_slot, MAX_SLOT_SEC, ideal_count)
    elif avg_slot < 1.5:
        logger.warning(
            "%d media for %.1fs VO -> avg %.2fs/visual is very fast. "
            "~%d media would give a calmer 2-3s pace.",
            len(media), duration, avg_slot, ideal_count)
    else:
        logger.info("Pacing: %d visuals over %.1fs -> avg %.2fs each.",
                    len(media), duration, avg_slot)

    segments = build_schedule(word_timestamps, media, duration)
    over = 0
    for s in segments:
        if s.duration > MAX_SLOT_SEC + 0.05:
            over += 1
        kind = "clip" if Path(s.image_path).suffix.lower() in VIDEO_EXTS else "img"
        tag = f"{kind}#{s.variation}" if s.variation else kind
        logger.info("  [%02d] %5.2f-%5.2fs (%.2fs) %s=%s  text=%s",
                    s.index + 1, s.start_time, s.end_time, s.duration,
                    tag, Path(s.image_path).name if s.image_path else "-",
                    s.text[:50])
    logger.info("Scheduled %d visual slots from %d media (all <=%.1fs each).",
                len(segments), len(media), MAX_SLOT_SEC)
    if over:
        logger.warning("%d slot(s) still exceed %.1fs (word-gap snapping pushed them over).",
                        over, MAX_SLOT_SEC)

    raw_video = config.temp_dir / "video_no_captions.mp4"
    logger.info("Composing video...")
    compose_video(
        segments, work_audio, word_timestamps,
        config.video, config.caption, raw_video,
        watermark_config=config.watermark,
    )

    final_output = config.output_dir / f"{args.title}.mp4"
    if config.caption.style == "srt":
        srt_path = config.temp_dir / "captions.srt"
        generate_srt(word_timestamps, srt_path, config.caption.words_per_group)
        burn_srt_captions(raw_video, srt_path, final_output, config.caption)
    else:
        shutil.copy2(raw_video, final_output)

    logger.info("=" * 60)
    logger.info("DONE: %s", final_output)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
