"""
Transcriber — Whisper-based word-level timestamp extraction.

Used as a fallback when ElevenLabs doesn't return timestamps,
or to align cleaned audio after silence removal.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from config import WhisperConfig

logger = logging.getLogger(__name__)


@dataclass
class WordTimestamp:
    """A single word with its timing in the audio."""
    word: str
    start: float  # seconds
    end: float    # seconds


def transcribe_with_timestamps(
    audio_path: Path,
    config: WhisperConfig,
) -> list[WordTimestamp]:
    """
    Transcribe audio and extract word-level timestamps using faster-whisper.

    Args:
        audio_path: Path to the audio file.
        config: Whisper configuration.

    Returns:
        List of WordTimestamp objects.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.warning("faster-whisper not installed. Trying openai-whisper...")
        return _transcribe_with_openai_whisper(audio_path, config)

    logger.info("Transcribing with faster-whisper (model=%s)...", config.model_size)

    model = WhisperModel(
        config.model_size,
        device=config.device,
        compute_type="int8" if config.device == "cpu" else "float16",
    )

    segments, info = model.transcribe(
        str(audio_path),
        language=config.language,
        word_timestamps=True,
    )

    words = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                words.append(WordTimestamp(
                    word=w.word.strip(),
                    start=w.start,
                    end=w.end,
                ))

    logger.info("Transcribed %d words (duration: %.1fs).", len(words), info.duration)
    return words


def _transcribe_with_openai_whisper(
    audio_path: Path,
    config: WhisperConfig,
) -> list[WordTimestamp]:
    """Fallback: use openai-whisper package."""
    import whisper

    logger.info("Transcribing with openai-whisper (model=%s)...", config.model_size)

    model = whisper.load_model(config.model_size, device=config.device)
    result = model.transcribe(
        str(audio_path),
        language=config.language,
        word_timestamps=True,
    )

    words = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            words.append(WordTimestamp(
                word=w["word"].strip(),
                start=w["start"],
                end=w["end"],
            ))

    logger.info("Transcribed %d words.", len(words))
    return words


def align_segments_to_audio(
    segments: list,  # list[ScriptSegment]
    word_timestamps: list[WordTimestamp],
) -> list:
    """
    Map script segments to time ranges using word timestamps.

    Strategy: for each segment, find the best matching span of words
    in the transcript using sequential text matching.
    """
    if not word_timestamps:
        logger.warning("No word timestamps — falling back to equal distribution.")
        return segments

    # Build a running word index
    transcript_words = [w.word.lower().strip(".,!?;:'\"") for w in word_timestamps]
    word_idx = 0  # cursor through transcript

    for seg in segments:
        seg_words = seg.text.lower().split()
        seg_words_clean = [w.strip(".,!?;:'\"") for w in seg_words]

        if not seg_words_clean:
            continue

        # Find the first word of this segment in the transcript
        best_start_idx = _find_best_match_start(
            seg_words_clean, transcript_words, word_idx
        )

        if best_start_idx is not None:
            end_idx = min(best_start_idx + len(seg_words_clean) - 1, len(word_timestamps) - 1)
            seg.start_time = word_timestamps[best_start_idx].start
            seg.end_time = word_timestamps[end_idx].end
            word_idx = end_idx + 1  # advance cursor
        else:
            # Couldn't find match — estimate based on position
            logger.debug("Could not align segment %d: '%s...'", seg.index, seg.text[:40])

    # Fill in any segments that didn't get aligned
    _fill_gaps(segments, word_timestamps)

    return segments


def _find_best_match_start(
    seg_words: list[str],
    transcript_words: list[str],
    start_from: int,
) -> int | None:
    """Find where seg_words best matches in transcript_words, starting from start_from."""
    if not seg_words:
        return None

    first_word = seg_words[0]
    # Search forward from cursor
    for i in range(start_from, len(transcript_words)):
        if transcript_words[i] == first_word:
            # Check if subsequent words also match (allow some fuzziness)
            match_count = 0
            for j, sw in enumerate(seg_words[:5]):  # check first 5 words
                if i + j < len(transcript_words) and transcript_words[i + j] == sw:
                    match_count += 1
            if match_count >= min(3, len(seg_words)):
                return i

    return None


def _fill_gaps(segments: list, word_timestamps: list[WordTimestamp]):
    """Fill in timestamps for segments that couldn't be aligned."""
    if not word_timestamps:
        return

    total_duration = word_timestamps[-1].end if word_timestamps else 0
    aligned = [(i, s) for i, s in enumerate(segments) if s.end_time > 0]

    if not aligned:
        # No segments aligned — distribute equally
        dur_per_seg = total_duration / len(segments) if segments else 0
        for i, seg in enumerate(segments):
            seg.start_time = i * dur_per_seg
            seg.end_time = (i + 1) * dur_per_seg
        return

    # Fill gaps between aligned segments
    for i, seg in enumerate(segments):
        if seg.end_time > 0:
            continue  # already aligned

        # Find surrounding aligned segments
        prev_end = 0.0
        next_start = total_duration
        for j in range(i - 1, -1, -1):
            if segments[j].end_time > 0:
                prev_end = segments[j].end_time
                break
        for j in range(i + 1, len(segments)):
            if segments[j].start_time > 0:
                next_start = segments[j].start_time
                break

        seg.start_time = prev_end
        seg.end_time = next_start
