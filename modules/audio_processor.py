"""
Audio Processor — silence removal and normalization.

Takes raw TTS output and produces clean, tight audio
suitable for short-form video pacing.
"""

import logging
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from config import AudioConfig

logger = logging.getLogger(__name__)


def remove_silence(
    audio_path: Path,
    config: AudioConfig,
    output_path: Path | None = None,
) -> Path:
    """
    Remove long silences from audio while preserving natural pacing.

    Args:
        audio_path: Path to the input audio file.
        config: Audio processing configuration.
        output_path: Where to save (defaults to {name}_clean.wav).

    Returns:
        Path to the cleaned audio file.
    """
    audio_path = Path(audio_path)
    if output_path is None:
        output_path = audio_path.with_stem(audio_path.stem + "_clean").with_suffix(".wav")
    output_path = Path(output_path)

    logger.info("Loading audio: %s", audio_path)
    audio = AudioSegment.from_file(str(audio_path))
    original_duration = len(audio) / 1000.0

    # Detect non-silent chunks
    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len=config.min_silence_ms,
        silence_thresh=config.silence_thresh_dbfs,
        seek_step=10,
    )

    if not nonsilent_ranges:
        logger.warning("No non-silent segments found — returning original audio.")
        audio.export(str(output_path), format="wav")
        return output_path

    # Rebuild audio with padding around each non-silent chunk
    chunks = []
    for start_ms, end_ms in nonsilent_ranges:
        # Add small padding to keep it natural
        padded_start = max(0, start_ms - config.keep_silence_ms)
        padded_end = min(len(audio), end_ms + config.keep_silence_ms)
        chunks.append(audio[padded_start:padded_end])

    cleaned = chunks[0]
    for chunk in chunks[1:]:
        cleaned += chunk

    cleaned_duration = len(cleaned) / 1000.0
    removed = original_duration - cleaned_duration

    logger.info(
        "Silence removal: %.1fs → %.1fs (removed %.1fs / %.0f%%)",
        original_duration, cleaned_duration, removed,
        (removed / original_duration) * 100 if original_duration > 0 else 0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.export(str(output_path), format="wav")
    return output_path


def normalize_audio(
    audio_path: Path,
    target_dbfs: float = -20.0,
    output_path: Path | None = None,
) -> Path:
    """
    Normalize audio volume to a target dBFS level.

    Args:
        audio_path: Path to input audio.
        target_dbfs: Target loudness in dBFS.
        output_path: Where to save (defaults to overwrite).

    Returns:
        Path to the normalized audio.
    """
    audio_path = Path(audio_path)
    if output_path is None:
        output_path = audio_path

    audio = AudioSegment.from_file(str(audio_path))
    change_in_dbfs = target_dbfs - audio.dBFS
    normalized = audio.apply_gain(change_in_dbfs)

    logger.info("Normalized audio: %.1f dBFS → %.1f dBFS", audio.dBFS, normalized.dBFS)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    normalized.export(str(output_path), format="wav")
    return output_path


def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds."""
    audio = AudioSegment.from_file(str(audio_path))
    return len(audio) / 1000.0


def create_silence_mapping(
    original_path: Path,
    cleaned_path: Path,
    config: AudioConfig,
) -> list[tuple[float, float, float]]:
    """
    Create a mapping between original audio timestamps and cleaned audio timestamps.
    
    This is critical: when we remove silence, the word timestamps from ElevenLabs
    (which reference the ORIGINAL audio) become wrong. This function produces a 
    mapping so we can adjust timestamps to match the cleaned audio.

    Returns:
        List of (original_start, original_end, cleaned_start) tuples
        for each non-silent chunk.
    """
    original = AudioSegment.from_file(str(original_path))

    nonsilent_ranges = detect_nonsilent(
        original,
        min_silence_len=config.min_silence_ms,
        silence_thresh=config.silence_thresh_dbfs,
        seek_step=10,
    )

    mapping = []
    cleaned_cursor = 0.0  # current position in cleaned audio (seconds)

    for start_ms, end_ms in nonsilent_ranges:
        padded_start = max(0, start_ms - config.keep_silence_ms)
        padded_end = min(len(original), end_ms + config.keep_silence_ms)
        chunk_duration = (padded_end - padded_start) / 1000.0

        mapping.append((
            padded_start / 1000.0,   # original start (seconds)
            padded_end / 1000.0,     # original end (seconds)
            cleaned_cursor,          # where this chunk starts in cleaned audio
        ))
        cleaned_cursor += chunk_duration

    return mapping


def remap_timestamp(
    original_time: float,
    silence_mapping: list[tuple[float, float, float]],
) -> float | None:
    """
    Convert a timestamp from original audio space to cleaned audio space.

    Args:
        original_time: Timestamp in the original (pre-silence-removal) audio.
        silence_mapping: Output from create_silence_mapping().

    Returns:
        Adjusted timestamp in cleaned audio, or None if the timestamp
        falls in a removed silence gap.
    """
    for orig_start, orig_end, cleaned_start in silence_mapping:
        if orig_start <= original_time <= orig_end:
            offset = original_time - orig_start
            return cleaned_start + offset

    # Timestamp falls in a silence gap — find nearest chunk
    # Return the start of the next chunk
    for orig_start, orig_end, cleaned_start in silence_mapping:
        if original_time < orig_start:
            return cleaned_start

    # Past all chunks — return end of last chunk
    if silence_mapping:
        last_orig_start, last_orig_end, last_cleaned_start = silence_mapping[-1]
        return last_cleaned_start + (last_orig_end - last_orig_start)

    return 0.0
