#!/usr/bin/env python3
"""
ASOIAF Short-Form Video Engine — Main Orchestrator

Usage:
    python main.py --script "path/to/script.txt" --output "output/my_video.mp4"
    python main.py --script-text "Aegon Targaryen conquered Westeros..."

The pipeline:
    Script → Parse → Fetch Images → TTS → Remove Silence → 
    Transcribe → Align → Compose Video → Burn Captions → Done
"""

import argparse
import logging
import sys
from pathlib import Path

from config import Config
from modules.script_parser import parse_script, merge_short_segments, get_full_script_text
from modules.tts_generator import generate_voiceover, generate_voiceover_with_timestamps
from modules.audio_processor import (
    remove_silence,
    normalize_audio,
    get_audio_duration,
    create_silence_mapping,
    remap_timestamp,
)
from modules.image_fetcher import fetch_images_for_segments
from modules.transcriber import transcribe_with_timestamps, align_segments_to_audio, WordTimestamp
from modules.video_composer import compose_video, generate_srt, burn_srt_captions


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def run_pipeline(
    script_text: str,
    config: Config,
    output_filename: str = "video.mp4",
) -> Path:
    """
    Run the full video generation pipeline.

    Args:
        script_text: The narration script.
        config: Full configuration object.
        output_filename: Name of the output video file.

    Returns:
        Path to the final video.
    """
    temp = config.temp_dir
    temp.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Parse script into segments ──────────────────────────
    logger.info("═" * 60)
    logger.info("STEP 1: Parsing script...")
    logger.info("═" * 60)

    segments = parse_script(script_text, mode="sentence")
    segments = merge_short_segments(segments, min_words=5)
    logger.info("Parsed %d segments.", len(segments))

    for seg in segments:
        logger.debug("  [%d] %s", seg.index, seg.text[:60])

    # ── Step 2: Generate voiceover ──────────────────────────────────
    logger.info("═" * 60)
    logger.info("STEP 2: Generating voiceover (ElevenLabs)...")
    logger.info("═" * 60)

    full_script = get_full_script_text(segments)
    raw_audio_path = temp / "voiceover_raw.mp3"

    # Try to get timestamps directly from ElevenLabs
    elevenlabs_timestamps = []
    try:
        raw_audio_path, elevenlabs_timestamps = generate_voiceover_with_timestamps(
            full_script, config.elevenlabs, raw_audio_path
        )
        logger.info("Got %d word timestamps from ElevenLabs.", len(elevenlabs_timestamps))
    except Exception as e:
        logger.warning("Timestamp endpoint failed (%s), using basic TTS.", e)
        raw_audio_path = generate_voiceover(full_script, config.elevenlabs, raw_audio_path)

    # ── Step 3: Remove silence & normalize ──────────────────────────
    logger.info("═" * 60)
    logger.info("STEP 3: Removing silence...")
    logger.info("═" * 60)

    clean_audio_path = temp / "voiceover_clean.wav"
    clean_audio_path = remove_silence(raw_audio_path, config.audio, clean_audio_path)
    clean_audio_path = normalize_audio(clean_audio_path, config.audio.normalize_target_dbfs)

    audio_duration = get_audio_duration(clean_audio_path)
    logger.info("Clean audio duration: %.1fs", audio_duration)

    # ── Step 3b: Remap timestamps after silence removal ─────────────
    # If we got timestamps from ElevenLabs, they reference the ORIGINAL audio.
    # After silence removal, we need to adjust them.
    if elevenlabs_timestamps:
        logger.info("Remapping ElevenLabs timestamps to cleaned audio...")
        silence_map = create_silence_mapping(raw_audio_path, clean_audio_path, config.audio)
        remapped_timestamps = []
        for wt in elevenlabs_timestamps:
            new_start = remap_timestamp(wt.start, silence_map)
            new_end = remap_timestamp(wt.end, silence_map)
            if new_start is not None and new_end is not None:
                remapped_timestamps.append(WordTimestamp(
                    word=wt.word,
                    start=new_start,
                    end=new_end,
                ))
        word_timestamps = remapped_timestamps
        logger.info("Remapped %d timestamps.", len(word_timestamps))
    else:
        word_timestamps = []

    # ── Step 3c: Whisper fallback ───────────────────────────────────
    if not word_timestamps:
        logger.info("Running Whisper on cleaned audio for timestamps...")
        word_timestamps = transcribe_with_timestamps(clean_audio_path, config.whisper)

    # ── Step 4: Align segments to audio timestamps ──────────────────
    logger.info("═" * 60)
    logger.info("STEP 4: Aligning segments to audio...")
    logger.info("═" * 60)

    segments = align_segments_to_audio(segments, word_timestamps)

    for seg in segments:
        logger.info(
            "  [%d] %.1fs-%.1fs (%.1fs): %s",
            seg.index, seg.start_time, seg.end_time, seg.duration, seg.text[:50],
        )

    # ── Step 5: Fetch images ────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("STEP 5: Fetching images...")
    logger.info("═" * 60)

    image_download_dir = temp / "images"
    segments = fetch_images_for_segments(segments, config.image_search, image_download_dir)

    images_found = sum(1 for s in segments if s.image_path)
    logger.info("Images: %d/%d segments have images.", images_found, len(segments))

    # ── Step 6: Compose video with Ken Burns ────────────────────────
    logger.info("═" * 60)
    logger.info("STEP 6: Composing video...")
    logger.info("═" * 60)

    raw_video_path = temp / "video_no_captions.mp4"
    raw_video_path = compose_video(
        segments=segments,
        audio_path=clean_audio_path,
        word_timestamps=word_timestamps,
        config=config.video,
        caption_config=config.caption,
        output_path=raw_video_path,
    )

    # ── Step 7: Add captions ────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("STEP 7: Adding captions...")
    logger.info("═" * 60)

    final_output = config.output_dir / output_filename

    if config.caption.style == "srt":
        # Generate SRT and burn via FFmpeg
        srt_path = temp / "captions.srt"
        generate_srt(word_timestamps, srt_path, config.caption.words_per_group)
        final_output = burn_srt_captions(
            raw_video_path, srt_path, final_output, config.caption
        )
    else:
        # Captions were rendered directly on frames in compose_video
        # Just copy the video to output
        import shutil
        shutil.copy2(raw_video_path, final_output)

    # ── Done ────────────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("✓ PIPELINE COMPLETE")
    logger.info("  Output: %s", final_output)
    logger.info("  Duration: %.1fs", audio_duration)
    logger.info("  Segments: %d", len(segments))
    logger.info("═" * 60)

    return final_output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ASOIAF Short-Form Video Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --script script.txt
    python main.py --script-text "The Doom of Valyria destroyed an empire..."
    python main.py --script script.txt --output my_video.mp4 --caption-style srt
        """,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--script", type=str, help="Path to script text file")
    input_group.add_argument("--script-text", type=str, help="Script text directly")

    parser.add_argument("--output", type=str, default="video.mp4", help="Output filename")
    parser.add_argument("--caption-style", choices=["srt", "highlight"], default="srt")
    parser.add_argument("--voice-id", type=str, help="ElevenLabs voice ID override")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load script
    if args.script:
        script_path = Path(args.script)
        if not script_path.exists():
            logger.error("Script file not found: %s", script_path)
            sys.exit(1)
        script_text = script_path.read_text(encoding="utf-8")
    else:
        script_text = args.script_text

    # Build config
    config = Config()
    config.caption.style = args.caption_style

    if args.voice_id:
        config.elevenlabs.voice_id = args.voice_id

    # Validate API keys
    if config.elevenlabs.api_key == "YOUR_ELEVENLABS_API_KEY":
        logger.error(
            "Please set your ElevenLabs API key in config.py "
            "or via ELEVENLABS_API_KEY environment variable."
        )
        sys.exit(1)

    # Run the pipeline
    try:
        output = run_pipeline(script_text, config, args.output)
        print(f"\n✅ Video generated: {output}")
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
