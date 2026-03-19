#!/usr/bin/env python3
"""
ASOIAF Video Engine — Stage-by-Stage Pipeline Tests

Run individual steps:
    python tests/test_pipeline_stages.py 1      # Whisper transcription
    python tests/test_pipeline_stages.py 2      # Script parsing + alignment
    python tests/test_pipeline_stages.py 3      # Time-based re-segmentation
    python tests/test_pipeline_stages.py 4      # Entity extraction (naive)
    python tests/test_pipeline_stages.py 5      # Image library + shopping list
    python tests/test_pipeline_stages.py 6      # Full Phase 1 (prepare)
    python tests/test_pipeline_stages.py 7      # Video composition (placeholders)
    python tests/test_pipeline_stages.py 8      # SRT + caption burning
    python tests/test_pipeline_stages.py 9      # Full Phase 2 (continue)
    python tests/test_pipeline_stages.py all    # Run steps 1–5 (unit tests)
"""

import io
import json
import logging
import shutil
import sys
import os
from pathlib import Path

# Fix Windows console encoding for Unicode output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from config import Config, WhisperConfig, AudioConfig, VideoConfig, CaptionConfig, ImageSearchConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TEMP = PROJECT_ROOT / "temp"
OUTPUT = PROJECT_ROOT / "output"
CLEAN_AUDIO = TEMP / "test_vo_clean.wav"
TEST_SCRIPT = TEMP / "test_script.txt"

PASS = "[PASS]"
FAIL = "[FAIL]"


def check(condition: bool, msg: str):
    tag = PASS if condition else FAIL
    print(f"  {tag} {msg}")
    if not condition:
        raise AssertionError(msg)


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Whisper Transcription
# ═══════════════════════════════════════════════════════════════════════════

def step1_whisper():
    print("\n" + "=" * 60)
    print("STEP 1: Whisper Transcription (word timestamps)")
    print("=" * 60)

    check(CLEAN_AUDIO.exists(), f"Clean audio exists: {CLEAN_AUDIO}")

    from modules.transcriber import transcribe_with_timestamps, WordTimestamp

    config = WhisperConfig(model_size="base", language="en", device="cpu")
    words = transcribe_with_timestamps(CLEAN_AUDIO, config)

    check(len(words) > 0, f"Got {len(words)} word timestamps")
    check(all(isinstance(w, WordTimestamp) for w in words), "All items are WordTimestamp")
    check(all(w.end >= w.start for w in words), "All end >= start")
    check(words[0].start >= 0, f"First word starts at {words[0].start:.3f}s")
    check(words[-1].end > 0, f"Last word ends at {words[-1].end:.3f}s")

    # Check for reasonable coverage
    from modules.audio_processor import get_audio_duration
    audio_dur = get_audio_duration(CLEAN_AUDIO)
    coverage = words[-1].end / audio_dur
    check(coverage > 0.8, f"Timestamps cover {coverage:.0%} of audio ({audio_dur:.1f}s)")

    print(f"\n  First 5 words:")
    for w in words[:5]:
        print(f"    {w.start:.2f}–{w.end:.2f}: '{w.word}'")
    print(f"  Last 5 words:")
    for w in words[-5:]:
        print(f"    {w.start:.2f}–{w.end:.2f}: '{w.word}'")

    # Save for downstream steps
    _save_words(words)
    print(f"\n  Saved {len(words)} timestamps to temp/test_words.json")
    return words


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Script Parsing + Segment Alignment
# ═══════════════════════════════════════════════════════════════════════════

def step2_parse_and_align():
    print("\n" + "=" * 60)
    print("STEP 2: Script Parsing + Segment Alignment")
    print("=" * 60)

    check(TEST_SCRIPT.exists(), f"Test script exists: {TEST_SCRIPT}")

    from modules.script_parser import parse_script, ScriptSegment
    from modules.transcriber import align_segments_to_audio

    script_text = TEST_SCRIPT.read_text(encoding="utf-8")
    segments = parse_script(script_text, mode="sentence")

    check(len(segments) > 0, f"Parsed {len(segments)} segments")
    check(all(isinstance(s, ScriptSegment) for s in segments), "All items are ScriptSegment")
    check(all(s.text.strip() for s in segments), "No empty segments")

    print(f"\n  Segments before alignment:")
    for s in segments:
        print(f"    [{s.index:02d}] {s.text[:60]}...")

    # Load word timestamps from step 1
    words = _load_words()
    check(len(words) > 0, f"Loaded {len(words)} word timestamps from step 1")

    segments = align_segments_to_audio(segments, words)

    check(all(s.end_time > 0 for s in segments), "All segments have end_time > 0")
    check(all(s.end_time > s.start_time for s in segments), "All segments have end > start")

    # Check no major gaps
    for i in range(1, len(segments)):
        gap = segments[i].start_time - segments[i - 1].end_time
        check(abs(gap) < 2.0, f"Gap between seg {i-1} and {i}: {gap:.2f}s (< 2s)")

    print(f"\n  Segments after alignment:")
    for s in segments:
        print(f"    [{s.index:02d}] {s.start_time:.2f}–{s.end_time:.2f} ({s.duration:.1f}s): {s.text[:50]}...")

    _save_segments(segments)
    print(f"\n  Saved {len(segments)} aligned segments to temp/test_segments.json")
    return segments


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Time-based Re-segmentation
# ═══════════════════════════════════════════════════════════════════════════

def step3_resegment():
    print("\n" + "=" * 60)
    print("STEP 3: Time-based Re-segmentation")
    print("=" * 60)

    from modules.script_parser import resegment_by_time, ScriptSegment

    segments = _load_segments()
    check(len(segments) > 0, f"Loaded {len(segments)} segments from step 2")

    original_count = len(segments)
    from modules.audio_processor import get_audio_duration
    audio_dur = get_audio_duration(CLEAN_AUDIO)

    resegmented = resegment_by_time(
        segments,
        target_interval=3.5,
        min_interval=2.0,
        max_interval=5.0,
    )

    check(len(resegmented) > 0, f"Re-segmented: {original_count} -> {len(resegmented)} segments")

    for s in resegmented:
        check(
            s.duration >= 1.5,  # slightly under 2.0 tolerance for edge cases
            f"Seg {s.index:02d} duration {s.duration:.2f}s >= 1.5s"
        )
        check(
            s.duration <= 5.5,  # slight tolerance
            f"Seg {s.index:02d} duration {s.duration:.2f}s <= 5.5s"
        )

    # Check total coverage (gaps from silence between sentences are expected)
    total_covered = sum(s.duration for s in resegmented)
    check(
        abs(total_covered - audio_dur) < 3.0,
        f"Total coverage {total_covered:.1f}s ~= audio {audio_dur:.1f}s (within 3s)"
    )

    print(f"\n  Re-segmented timeline:")
    for s in resegmented:
        print(f"    [{s.index:02d}] {s.start_time:.2f}–{s.end_time:.2f} ({s.duration:.1f}s): {s.text[:50]}...")

    _save_segments(resegmented, "test_resegmented.json")
    print(f"\n  Saved {len(resegmented)} re-segmented to temp/test_resegmented.json")
    return resegmented


# ═══════════════════════════════════════════════════════════════════════════
# Step 4: Entity Extraction (naive mode)
# ═══════════════════════════════════════════════════════════════════════════

def step4_entities():
    print("\n" + "=" * 60)
    print("STEP 4: Entity Extraction (naive mode)")
    print("=" * 60)

    from modules.image_manager import _naive_entity_extraction

    segments = _load_segments("test_resegmented.json")
    check(len(segments) > 0, f"Loaded {len(segments)} segments")

    entities_list = []
    for seg in segments:
        entities = _naive_entity_extraction(seg.text)
        entities_list.append(entities)

        check("characters" in entities, f"Seg {seg.index}: has 'characters' key")
        check("search_query" in entities, f"Seg {seg.index}: has 'search_query' key")
        check("mood" in entities, f"Seg {seg.index}: has 'mood' key")
        check(isinstance(entities["characters"], list), f"Seg {seg.index}: characters is list")
        check(len(entities["search_query"]) > 0, f"Seg {seg.index}: search_query not empty")

    print(f"\n  Extracted entities:")
    for i, (seg, ent) in enumerate(zip(segments, entities_list)):
        chars = ", ".join(ent["characters"]) or "(none)"
        print(f"    [{i:02d}] chars=[{chars}] query=\"{ent['search_query'][:50]}\"")

    # Save for step 5
    _save_json("test_entities.json", entities_list)
    print(f"\n  Saved entities for {len(entities_list)} segments to temp/test_entities.json")
    return entities_list


# ═══════════════════════════════════════════════════════════════════════════
# Step 5: Image Library + Shopping List
# ═══════════════════════════════════════════════════════════════════════════

def step5_shopping_list():
    print("\n" + "=" * 60)
    print("STEP 5: Image Library + Shopping List")
    print("=" * 60)

    from modules.image_manager import process_images_for_segments, generate_shopping_list, ImageAssignment

    segments = _load_segments("test_resegmented.json")
    check(len(segments) > 0, f"Loaded {len(segments)} segments")

    config = ImageSearchConfig(
        anthropic_api_key="",  # naive mode
        local_library_path=PROJECT_ROOT / "assets" / "images",
    )

    # Ensure input dir exists
    input_dir = PROJECT_ROOT / "input" / "images"
    input_dir.mkdir(parents=True, exist_ok=True)

    assignments, entities_list, all_covered = process_images_for_segments(
        segments, config, "test_video", input_dir,
        config.match_threshold,
    )

    check(len(assignments) == len(segments), f"Got {len(assignments)} assignments for {len(segments)} segments")
    check(all(isinstance(a, ImageAssignment) for a in assignments), "All items are ImageAssignment")

    needed = sum(1 for a in assignments if a.source == "needed")
    matched = sum(1 for a in assignments if a.source == "library")
    print(f"\n  Results: {matched} from library, {needed} needed")
    check(needed + matched == len(segments), f"All {len(segments)} segments accounted for ({matched} matched, {needed} needed)")

    # Generate shopping list
    output_path = OUTPUT / "test_shopping_list.txt"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sl_path = generate_shopping_list(
        segments, entities_list, assignments,
        output_path, "Test Video",
    )

    check(sl_path.exists(), f"Shopping list created: {sl_path}")
    content = sl_path.read_text(encoding="utf-8")
    check("NEED FROM YOU" in content, "Shopping list has 'NEED FROM YOU' section")
    check(len(content) > 100, f"Shopping list has content ({len(content)} chars)")

    print(f"\n  Shopping list preview (first 10 lines):")
    for line in content.split("\n")[:10]:
        try:
            print(f"    {line}")
        except UnicodeEncodeError:
            print(f"    {line.encode('ascii', errors='replace').decode('ascii')}")

    return assignments, entities_list


# ═══════════════════════════════════════════════════════════════════════════
# Step 6: Full Phase 1 (prepare command)
# ═══════════════════════════════════════════════════════════════════════════

def step6_phase_prepare():
    print("\n" + "=" * 60)
    print("STEP 6: Full Phase 1 (prepare command)")
    print("=" * 60)
    print("  NOTE: This calls the real TTS API — skipping if no API key set.")
    print("  We test Phase 1 logic by simulating it with existing audio.\n")

    from modules.script_parser import parse_script, resegment_by_time, get_full_script_text, ScriptSegment
    from modules.transcriber import transcribe_with_timestamps, align_segments_to_audio, WordTimestamp
    from modules.audio_processor import get_audio_duration
    from modules.image_manager import process_images_for_segments, generate_shopping_list

    config = Config()

    # Use our existing test audio + script (bypass TTS)
    script_text = TEST_SCRIPT.read_text(encoding="utf-8")
    clean_audio = CLEAN_AUDIO

    # Parse
    segments = parse_script(script_text, mode="sentence")
    check(len(segments) > 0, f"Parsed {len(segments)} segments")

    # Transcribe
    words = _load_words()  # reuse cached whisper results
    if not words:
        words = transcribe_with_timestamps(clean_audio, config.whisper)
    check(len(words) > 0, f"Got {len(words)} word timestamps")

    # Align
    segments = align_segments_to_audio(segments, words)
    check(all(s.end_time > 0 for s in segments), "All segments aligned")

    # Re-segment
    segments = resegment_by_time(
        segments,
        target_interval=config.image_search.target_interval_sec,
        min_interval=config.image_search.min_interval_sec,
        max_interval=config.image_search.max_interval_sec,
    )
    check(len(segments) > 0, f"Re-segmented to {len(segments)} segments")

    # Image processing
    input_dir = config.input_dir
    input_dir.mkdir(parents=True, exist_ok=True)

    assignments, entities_list, all_covered = process_images_for_segments(
        segments, config.image_search, "test", input_dir,
        config.image_search.match_threshold,
    )

    # Save state (same as phase_prepare does)
    from main import save_state
    save_state(config, {
        "segments": segments,
        "word_timestamps": words,
        "assignments": assignments,
        "entities_list": entities_list,
        "clean_audio_path": clean_audio,
        "video_name": "test",
    })

    check(config.state_file.exists(), f"State file created: {config.state_file}")

    # Generate shopping list
    generate_shopping_list(
        segments, entities_list, assignments,
        config.shopping_list_path, "test",
    )
    check(config.shopping_list_path.exists(), f"Shopping list created: {config.shopping_list_path}")

    print(f"\n  Phase 1 simulation complete.")
    print(f"  State file: {config.state_file}")
    print(f"  Shopping list: {config.shopping_list_path}")
    return segments, words, assignments, entities_list


# ═══════════════════════════════════════════════════════════════════════════
# Step 7: Video Composition (placeholder images)
# ═══════════════════════════════════════════════════════════════════════════

def step7_compose_video():
    print("\n" + "=" * 60)
    print("STEP 7: Video Composition (placeholder images)")
    print("=" * 60)

    import subprocess

    # Check ffmpeg
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    check(result.returncode == 0, "FFmpeg is available")

    from modules.video_composer import compose_video
    from modules.audio_processor import get_audio_duration

    segments = _load_segments("test_resegmented.json")
    words = _load_words()
    config = Config()

    check(len(segments) > 0, f"Loaded {len(segments)} segments")
    check(len(words) > 0, f"Loaded {len(words)} word timestamps")
    check(CLEAN_AUDIO.exists(), "Clean audio exists")

    # All segments have no image_path → will use dark placeholders
    for s in segments:
        s.image_path = None

    output_path = TEMP / "test_video_placeholders.mp4"

    video_path = compose_video(
        segments, CLEAN_AUDIO, words,
        config.video, config.caption, output_path,
    )

    check(video_path.exists(), f"Video file created: {video_path}")
    size_mb = video_path.stat().st_size / (1024 * 1024)
    check(size_mb > 0.1, f"Video file has content ({size_mb:.1f} MB)")

    # Verify resolution with ffprobe
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,duration",
         "-of", "json", str(video_path)],
        capture_output=True, text=True,
    )
    if probe.returncode == 0:
        info = json.loads(probe.stdout)
        stream = info.get("streams", [{}])[0]
        w = int(stream.get("width", 0))
        h = int(stream.get("height", 0))
        check(w == config.video.width, f"Width is {w} (expected {config.video.width})")
        check(h == config.video.height, f"Height is {h} (expected {config.video.height})")
        vid_dur = float(stream.get("duration", 0))
        audio_dur = get_audio_duration(CLEAN_AUDIO)
        check(
            abs(vid_dur - audio_dur) < 2.0,
            f"Video duration {vid_dur:.1f}s ~= audio {audio_dur:.1f}s"
        )

    print(f"\n  Video: {video_path} ({size_mb:.1f} MB)")
    return video_path


# ═══════════════════════════════════════════════════════════════════════════
# Step 8: SRT Caption Generation + Burning
# ═══════════════════════════════════════════════════════════════════════════

def step8_captions():
    print("\n" + "=" * 60)
    print("STEP 8: SRT Caption Generation + Burning")
    print("=" * 60)

    from modules.video_composer import generate_srt, burn_srt_captions

    words = _load_words()
    config = Config()

    check(len(words) > 0, f"Loaded {len(words)} word timestamps")

    # Check font
    font_path = Path(config.caption.font_path)
    if not font_path.exists():
        print(f"  WARNING: Font not found at {font_path}")
        print(f"  Downloading Montserrat-Bold.ttf...")
        _download_montserrat(font_path)

    check(font_path.exists(), f"Font file exists: {font_path}")

    # Generate SRT
    srt_path = TEMP / "test_captions.srt"
    generate_srt(words, srt_path, config.caption.words_per_group)
    check(srt_path.exists(), f"SRT file created: {srt_path}")

    srt_content = srt_path.read_text(encoding="utf-8")
    lines = [l for l in srt_content.strip().split("\n") if l.strip()]
    check(len(lines) > 5, f"SRT has {len(lines)} non-empty lines")
    check("-->" in srt_content, "SRT contains time markers")

    print(f"\n  SRT preview (first 5 entries):")
    for line in srt_content.split("\n")[:15]:
        print(f"    {line}")

    # Burn captions onto video
    video_in = TEMP / "test_video_placeholders.mp4"
    if not video_in.exists():
        print(f"\n  WARNING: No video from step 7 at {video_in}")
        print(f"  Running step 7 first...")
        video_in = step7_compose_video()

    video_out = TEMP / "test_video_captioned.mp4"
    result = burn_srt_captions(video_in, srt_path, video_out, config.caption)

    check(result.exists(), f"Captioned video created: {result}")
    size_mb = result.stat().st_size / (1024 * 1024)
    check(size_mb > 0.1, f"Captioned video has content ({size_mb:.1f} MB)")

    print(f"\n  Captioned video: {result} ({size_mb:.1f} MB)")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Step 9: Full Phase 2 (continue command) with dummy test images
# ═══════════════════════════════════════════════════════════════════════════

def step9_phase_continue():
    print("\n" + "=" * 60)
    print("STEP 9: Full Phase 2 (continue command)")
    print("=" * 60)

    from PIL import Image
    import numpy as np

    config = Config()

    # Ensure state file exists (from step 6)
    if not config.state_file.exists():
        print("  State file missing — running step 6 first...")
        step6_phase_prepare()

    check(config.state_file.exists(), f"State file exists: {config.state_file}")

    # Load state to know how many segments need images
    from main import load_state
    state = load_state(config)
    needed = [a for a in state["assignments"] if a.image_path is None]
    print(f"  Need to create {len(needed)} dummy images")

    # Create dark gradient test images for each needed segment
    input_dir = config.input_dir
    input_dir.mkdir(parents=True, exist_ok=True)

    for a in needed:
        seg_num = a.segment_index + 1
        img_path = input_dir / f"{seg_num:02d}.jpg"
        # Create a dark gradient image with segment number
        img = Image.new("RGB", (1080, 1920), (20, 20, 30))
        # Add some color variation per segment
        arr = np.array(img)
        hue_shift = (seg_num * 30) % 180
        arr[:, :, 0] = np.clip(arr[:, :, 0] + hue_shift // 3, 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] + hue_shift // 2, 0, 255)
        Image.fromarray(arr).save(str(img_path), quality=85)
        print(f"    Created dummy: {img_path.name}")

    # Now run phase_continue
    from main import phase_continue
    phase_continue(config)

    # Verify output
    final_video = config.output_dir / "test.mp4"
    check(final_video.exists(), f"Final video created: {final_video}")

    size_mb = final_video.stat().st_size / (1024 * 1024)
    check(size_mb > 0.1, f"Final video has content ({size_mb:.1f} MB)")

    print(f"\n  Final video: {final_video} ({size_mb:.1f} MB)")
    print(f"  Phase 2 complete!")
    return final_video


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _save_words(words):
    data = [{"word": w.word, "start": w.start, "end": w.end} for w in words]
    _save_json("test_words.json", data)


def _load_words():
    from modules.transcriber import WordTimestamp
    data = _load_json("test_words.json")
    if not data:
        return []
    return [WordTimestamp(w["word"], w["start"], w["end"]) for w in data]


def _save_segments(segments, filename="test_segments.json"):
    data = [
        {
            "index": s.index, "text": s.text,
            "start_time": s.start_time, "end_time": s.end_time,
            "image_path": s.image_path, "search_query": s.search_query,
            "manual_image": s.manual_image,
        }
        for s in segments
    ]
    _save_json(filename, data)


def _load_segments(filename="test_segments.json"):
    from modules.script_parser import ScriptSegment
    data = _load_json(filename)
    if not data:
        return []
    return [
        ScriptSegment(
            index=s["index"], text=s["text"],
            start_time=s.get("start_time", 0), end_time=s.get("end_time", 0),
            image_path=s.get("image_path"), search_query=s.get("search_query"),
            manual_image=s.get("manual_image"),
        )
        for s in data
    ]


def _save_json(filename, data):
    path = TEMP / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_json(filename):
    path = TEMP / filename
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def _download_montserrat(dest_path: Path):
    """Download Montserrat-Bold.ttf from Google Fonts."""
    import urllib.request
    url = "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Bold.ttf"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, str(dest_path))
        print(f"  Downloaded: {dest_path}")
    except Exception as e:
        print(f"  ERROR downloading font: {e}")
        print(f"  Please manually place Montserrat-Bold.ttf at {dest_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

STEPS = {
    "1": ("Whisper Transcription", step1_whisper),
    "2": ("Script Parsing + Alignment", step2_parse_and_align),
    "3": ("Time-based Re-segmentation", step3_resegment),
    "4": ("Entity Extraction (naive)", step4_entities),
    "5": ("Image Library + Shopping List", step5_shopping_list),
    "6": ("Full Phase 1 (prepare)", step6_phase_prepare),
    "7": ("Video Composition (placeholders)", step7_compose_video),
    "8": ("SRT + Caption Burning", step8_captions),
    "9": ("Full Phase 2 (continue)", step9_phase_continue),
}


def main():
    if len(sys.argv) < 2:
        print("Usage: python tests/test_pipeline_stages.py <step_number|all>")
        print("\nAvailable steps:")
        for k, (name, _) in STEPS.items():
            print(f"  {k}: {name}")
        print("  all: Run steps 1–5 (unit tests)")
        sys.exit(1)

    target = sys.argv[1].lower()

    if target == "all":
        steps_to_run = ["1", "2", "3", "4", "5"]
    elif target == "full":
        steps_to_run = list(STEPS.keys())
    elif target in STEPS:
        steps_to_run = [target]
    else:
        print(f"Unknown step: {target}")
        sys.exit(1)

    passed = 0
    failed = 0

    for step_key in steps_to_run:
        name, fn = STEPS[step_key]
        try:
            fn()
            passed += 1
            print(f"\n  >>> Step {step_key} PASSED <<<\n")
        except Exception as e:
            failed += 1
            print(f"\n  >>> Step {step_key} FAILED: {e} <<<\n")
            import traceback
            traceback.print_exc()
            if target != "all" and target != "full":
                sys.exit(1)

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
