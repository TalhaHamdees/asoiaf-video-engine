#!/usr/bin/env python3
"""
ASOIAF Short-Form Video Engine — Main Orchestrator

Two-phase workflow:
    Phase 1 (prepare):   Script → TTS → Silence removal → Timestamps →
                          Check library → Output shopping list for missing images
    Phase 2 (continue):  Pick up user images → Generate video → Auto-tag → Grow library

Usage:
    # Phase 1: Generate voiceover and shopping list
    python main.py prepare --script script.txt --title "The Doom of Valyria"

    # (You download images listed in output/shopping_list.txt → input/images/)

    # Phase 2: Build the video
    python main.py continue

    # If the library covers everything, Phase 1 skips straight to Phase 2:
    python main.py prepare --script script.txt --title "Jon Snow" --auto
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from config import Config
from modules.script_parser import parse_script, resegment_by_time, get_full_script_text
from modules.tts_generator import generate_voiceover, generate_voiceover_with_timestamps
from modules.audio_processor import (
    remove_silence, normalize_audio, get_audio_duration,
    create_silence_mapping, remap_timestamp,
)
from modules.transcriber import (
    transcribe_with_timestamps, align_segments_to_audio,
    WordTimestamp,
)
from modules.image_manager import (
    process_images_for_segments, generate_shopping_list,
    ingest_user_images_to_library,
)
from modules.video_composer import compose_video, generate_srt, burn_srt_captions


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def _safe_print(msg: str):
    """Print with fallback for Windows consoles that can't handle Unicode."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


def save_state(config: Config, state: dict):
    """Save pipeline state between phases."""
    config.state_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert ScriptSegments to serializable dicts
    serializable = {}
    for key, value in state.items():
        if key == "segments":
            serializable[key] = [
                {
                    "index": s.index, "text": s.text,
                    "image_path": s.image_path, "search_query": s.search_query,
                    "start_time": s.start_time, "end_time": s.end_time,
                    "manual_image": s.manual_image,
                }
                for s in value
            ]
        elif key == "word_timestamps":
            serializable[key] = [
                {"word": w.word, "start": w.start, "end": w.end}
                for w in value
            ]
        elif key == "assignments":
            serializable[key] = [
                {
                    "segment_index": a.segment_index, "image_path": a.image_path,
                    "score": a.score, "source": a.source,
                    "search_query": a.search_query, "description": a.description,
                }
                for a in value
            ]
        elif key == "entities_list":
            serializable[key] = value  # already dicts
        else:
            serializable[key] = str(value) if isinstance(value, Path) else value

    with open(config.state_file, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Pipeline state saved: %s", config.state_file)


def load_state(config: Config) -> dict:
    """Load pipeline state from a previous prepare phase."""
    from modules.script_parser import ScriptSegment
    from modules.image_manager import ImageAssignment

    if not config.state_file.exists():
        logger.error("No pipeline state found. Run 'prepare' first.")
        sys.exit(1)

    with open(config.state_file, "r") as f:
        raw = json.load(f)

    state = {}
    state["segments"] = [
        ScriptSegment(
            index=s["index"], text=s["text"],
            image_path=s.get("image_path"), search_query=s.get("search_query"),
            start_time=s.get("start_time", 0), end_time=s.get("end_time", 0),
            manual_image=s.get("manual_image"),
        )
        for s in raw.get("segments", [])
    ]
    state["word_timestamps"] = [
        WordTimestamp(w["word"], w["start"], w["end"])
        for w in raw.get("word_timestamps", [])
    ]
    state["assignments"] = [
        ImageAssignment(
            segment_index=a["segment_index"], image_path=a.get("image_path"),
            score=a.get("score", 0), source=a.get("source", "needed"),
            search_query=a.get("search_query", ""),
            description=a.get("description", ""),
        )
        for a in raw.get("assignments", [])
    ]
    state["entities_list"] = raw.get("entities_list", [])
    state["clean_audio_path"] = Path(raw["clean_audio_path"])
    state["video_name"] = raw.get("video_name", "video")

    logger.info("Loaded pipeline state: %d segments, %d timestamps.",
                len(state["segments"]), len(state["word_timestamps"]))
    return state


# ---------------------------------------------------------------------------
# Phase 1: Prepare (TTS + audio + timestamps + shopping list)
# ---------------------------------------------------------------------------

def phase_prepare(script_text: str, config: Config, video_name: str = "video", auto_continue: bool = False):
    """
    Phase 1: Generate audio, extract timestamps, check library, produce shopping list.
    """
    temp = config.temp_dir
    temp.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Parse script ──
    logger.info("=" * 60)
    logger.info("PHASE 1: PREPARE")
    logger.info("=" * 60)

    segments = parse_script(script_text, mode="sentence")
    logger.info("Parsed %d segments.", len(segments))

    # ── Generate voiceover ──
    logger.info("Generating voiceover (ElevenLabs)...")
    full_script = get_full_script_text(segments)
    raw_audio = temp / "voiceover_raw.mp3"

    elevenlabs_ts = []
    try:
        raw_audio, elevenlabs_ts = generate_voiceover_with_timestamps(
            full_script, config.elevenlabs, raw_audio
        )
        logger.info("Got %d word timestamps from ElevenLabs.", len(elevenlabs_ts))
    except Exception as e:
        logger.warning("Timestamp endpoint failed (%s), using basic TTS.", e)
        raw_audio = generate_voiceover(full_script, config.elevenlabs, raw_audio)

    # ── Remove silence & normalize ──
    logger.info("Removing silence...")
    clean_audio = temp / "voiceover_clean.wav"
    clean_audio = remove_silence(raw_audio, config.audio, clean_audio)
    clean_audio = normalize_audio(clean_audio, config.audio.normalize_target_dbfs)
    audio_dur = get_audio_duration(clean_audio)
    logger.info("Clean audio: %.1fs", audio_dur)

    # ── Remap timestamps ──
    if elevenlabs_ts:
        smap = create_silence_mapping(raw_audio, clean_audio, config.audio)
        word_timestamps = []
        for wt in elevenlabs_ts:
            ns = remap_timestamp(wt.start, smap)
            ne = remap_timestamp(wt.end, smap)
            if ns is not None and ne is not None:
                word_timestamps.append(WordTimestamp(wt.word, ns, ne))
    else:
        word_timestamps = []

    if not word_timestamps:
        logger.info("Running Whisper for timestamps...")
        word_timestamps = transcribe_with_timestamps(clean_audio, config.whisper)

    # ── Align segments to audio ──
    logger.info("Aligning segments to audio...")
    segments = align_segments_to_audio(segments, word_timestamps)

    # ── Re-segment for 3-4 second intervals ──
    logger.info("Re-segmenting for %.1fs image intervals...", config.image_search.target_interval_sec)
    segments = resegment_by_time(
        segments,
        target_interval=config.image_search.target_interval_sec,
        min_interval=config.image_search.min_interval_sec,
        max_interval=config.image_search.max_interval_sec,
    )
    logger.info("Final segment count: %d", len(segments))

    for s in segments:
        logger.info("  [%02d] %.1fs–%.1fs (%.1fs): %s",
                     s.index + 1, s.start_time, s.end_time, s.duration, s.text[:50])

    # ── Check library + generate shopping list ──
    logger.info("Checking image library...")
    assignments, entities_list, all_covered = process_images_for_segments(
        segments, config.image_search, video_name, config.input_dir,
        config.image_search.match_threshold,
    )

    matched = sum(1 for a in assignments if a.source == "library")
    needed = sum(1 for a in assignments if a.source != "library")
    logger.info("Library coverage: %d/%d (need %d more)", matched, len(segments), needed)

    # Save state for Phase 2
    save_state(config, {
        "segments": segments,
        "word_timestamps": word_timestamps,
        "assignments": assignments,
        "entities_list": entities_list,
        "clean_audio_path": clean_audio,
        "video_name": video_name,
    })

    if all_covered:
        _safe_print("\n[OK] All segments covered from library! No images needed.")
        if auto_continue:
            print("Auto-continuing to video generation...\n")
            phase_continue(config)
        else:
            print("Run `python main.py continue` to generate the video.\n")
    else:
        # Generate shopping list
        generate_shopping_list(
            segments, entities_list, assignments,
            config.shopping_list_path, video_name,
        )
        _safe_print(f"\n[INFO] Shopping list saved: {config.shopping_list_path}")
        _safe_print(f"   Download images -> {config.input_dir}/")
        _safe_print(f"   Then run: python main.py continue\n")


# ---------------------------------------------------------------------------
# Phase 2: Continue (pick up images → video → ingest to library)
# ---------------------------------------------------------------------------

def phase_continue(config: Config):
    """
    Phase 2: Pick up user-provided images, generate video, auto-tag new images.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: GENERATE VIDEO")
    logger.info("=" * 60)

    state = load_state(config)
    segments = state["segments"]
    word_timestamps = state["word_timestamps"]
    assignments = state["assignments"]
    entities_list = state["entities_list"]
    clean_audio = state["clean_audio_path"]
    video_name = state["video_name"]

    # ── Pick up user images ──
    input_dir = config.input_dir
    extensions = {".jpg", ".jpeg", ".png", ".webp"}

    user_images_found = 0
    for a in assignments:
        if a.image_path is not None:
            continue  # already covered

        seg_num = a.segment_index + 1
        for ext in extensions:
            candidate = input_dir / f"{seg_num:02d}{ext}"
            if candidate.exists():
                a.image_path = str(candidate)
                a.source = "user_input"
                user_images_found += 1
                logger.info("  Seg %02d: Picked up %s", seg_num, candidate.name)
                break

    # Check coverage
    still_missing = [a for a in assignments if a.image_path is None]
    if still_missing:
        _safe_print(f"\n[WARN] {len(still_missing)} segments still missing images:")
        for a in still_missing:
            seg = segments[a.segment_index]
            _safe_print(f"   [X] Seg {a.segment_index + 1:02d}: \"{seg.text[:60]}...\"")
            _safe_print(f"      Expected: {config.input_dir}/{a.segment_index + 1:02d}.jpg")
        _safe_print(f"\nPlease add the missing images and run `python main.py continue` again.")
        # Save updated state
        save_state(config, {
            "segments": segments, "word_timestamps": word_timestamps,
            "assignments": assignments, "entities_list": entities_list,
            "clean_audio_path": clean_audio, "video_name": video_name,
        })
        return

    # ── Apply images to segments ──
    for a in assignments:
        segments[a.segment_index].image_path = a.image_path

    # ── Compose video ──
    logger.info("Composing video...")
    raw_video = config.temp_dir / "video_no_captions.mp4"
    raw_video = compose_video(
        segments, clean_audio, word_timestamps,
        config.video, config.caption, raw_video,
    )

    # ── Add captions ──
    logger.info("Adding captions...")
    final_output = config.output_dir / f"{video_name}.mp4"

    if config.caption.style == "srt":
        srt_path = config.temp_dir / "captions.srt"
        generate_srt(word_timestamps, srt_path, config.caption.words_per_group)
        final_output = burn_srt_captions(raw_video, srt_path, final_output, config.caption)
    else:
        shutil.copy2(raw_video, final_output)

    # ── Ingest new images to library ──
    logger.info("Ingesting new images to library...")
    ingest_user_images_to_library(assignments, entities_list, segments, config.image_search)

    # ── Clean up input folder ──
    for f in input_dir.glob("*"):
        if f.suffix.lower() in extensions:
            f.unlink()
    logger.info("Cleared input folder.")

    # ── Done ──
    logger.info("=" * 60)
    logger.info("✅ VIDEO COMPLETE")
    logger.info("   Output: %s", final_output)
    logger.info("   Duration: %.1fs", get_audio_duration(clean_audio))
    logger.info("   Segments: %d", len(segments))
    logger.info("=" * 60)

    _safe_print(f"\n[OK] Video generated: {final_output}")
    _safe_print(f"   Library now has images from {sum(1 for a in assignments if a.source == 'user_input')} new + existing entries.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ASOIAF Short-Form Video Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline phase")

    # Prepare phase
    prep = subparsers.add_parser("prepare", help="Phase 1: Generate audio + shopping list")
    prep_input = prep.add_mutually_exclusive_group(required=True)
    prep_input.add_argument("--script", type=str, help="Path to script text file")
    prep_input.add_argument("--script-text", type=str, help="Script text directly")
    prep.add_argument("--title", type=str, default="video", help="Video title/name")
    prep.add_argument("--auto", action="store_true",
                       help="Auto-continue to video if library covers all segments")
    prep.add_argument("--voice-id", type=str, help="ElevenLabs voice ID override")

    # Continue phase
    cont = subparsers.add_parser("continue", help="Phase 2: Build video from images")

    # One-shot (for fully covered library)
    oneshot = subparsers.add_parser("run", help="Full run (prepare + continue)")
    oneshot_input = oneshot.add_mutually_exclusive_group(required=True)
    oneshot_input.add_argument("--script", type=str, help="Path to script text file")
    oneshot_input.add_argument("--script-text", type=str, help="Script text directly")
    oneshot.add_argument("--title", type=str, default="video", help="Video title/name")
    oneshot.add_argument("--voice-id", type=str, help="ElevenLabs voice ID override")

    parser.add_argument("--debug", action="store_true", help="Debug logging")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    config = Config()

    if hasattr(args, 'voice_id') and args.voice_id:
        config.elevenlabs.voice_id = args.voice_id

    if config.elevenlabs.api_key == "YOUR_ELEVENLABS_API_KEY":
        logger.error("Set your ElevenLabs API key in config.py.")
        sys.exit(1)

    if args.command in ("prepare", "run"):
        if args.script:
            script_path = Path(args.script)
            if not script_path.exists():
                logger.error("Script not found: %s", script_path)
                sys.exit(1)
            script_text = script_path.read_text(encoding="utf-8")
        else:
            script_text = args.script_text

        auto = args.command == "run" or getattr(args, 'auto', False)
        phase_prepare(script_text, config, args.title, auto_continue=auto)

        if args.command == "run":
            phase_continue(config)

    elif args.command == "continue":
        phase_continue(config)


if __name__ == "__main__":
    main()
