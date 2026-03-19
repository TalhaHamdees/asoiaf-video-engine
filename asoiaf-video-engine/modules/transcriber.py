"""
Transcriber — Whisper word-level timestamps + segment alignment.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from config import WhisperConfig

logger = logging.getLogger(__name__)


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float


def transcribe_with_timestamps(audio_path: Path, config: WhisperConfig) -> list[WordTimestamp]:
    try:
        from faster_whisper import WhisperModel
        logger.info("Transcribing with faster-whisper (model=%s)...", config.model_size)
        model = WhisperModel(config.model_size, device=config.device,
                             compute_type="int8" if config.device == "cpu" else "float16")
        segments, info = model.transcribe(str(audio_path), language=config.language, word_timestamps=True)
        words = []
        for seg in segments:
            if seg.words:
                for w in seg.words:
                    words.append(WordTimestamp(w.word.strip(), w.start, w.end))
        logger.info("Transcribed %d words (%.1fs).", len(words), info.duration)
        return words
    except ImportError:
        pass

    import whisper
    logger.info("Transcribing with openai-whisper (model=%s)...", config.model_size)
    model = whisper.load_model(config.model_size, device=config.device)
    result = model.transcribe(str(audio_path), language=config.language, word_timestamps=True)
    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            words.append(WordTimestamp(w["word"].strip(), w["start"], w["end"]))
    logger.info("Transcribed %d words.", len(words))
    return words


def align_segments_to_audio(segments: list, word_timestamps: list[WordTimestamp]) -> list:
    """Map script segments to time ranges using word timestamps."""
    if not word_timestamps:
        return segments

    transcript_words = [w.word.lower().strip(".,!?;:'\"") for w in word_timestamps]
    word_idx = 0

    for seg in segments:
        seg_words = [w.strip(".,!?;:'\"") for w in seg.text.lower().split()]
        if not seg_words:
            continue

        best_start = _find_match(seg_words, transcript_words, word_idx)
        if best_start is not None:
            end_idx = min(best_start + len(seg_words) - 1, len(word_timestamps) - 1)
            seg.start_time = word_timestamps[best_start].start
            seg.end_time = word_timestamps[end_idx].end
            word_idx = end_idx + 1

    _fill_gaps(segments, word_timestamps)
    return segments


def _find_match(seg_words, transcript_words, start_from):
    if not seg_words:
        return None
    first = seg_words[0]
    for i in range(start_from, len(transcript_words)):
        if transcript_words[i] == first:
            matches = sum(
                1 for j, sw in enumerate(seg_words[:5])
                if i + j < len(transcript_words) and transcript_words[i + j] == sw
            )
            if matches >= min(3, len(seg_words)):
                return i
    return None


def _fill_gaps(segments, word_timestamps):
    if not word_timestamps:
        return
    total = word_timestamps[-1].end
    aligned = [(i, s) for i, s in enumerate(segments) if s.end_time > 0]
    if not aligned:
        dur = total / len(segments) if segments else 0
        for i, s in enumerate(segments):
            s.start_time = i * dur
            s.end_time = (i + 1) * dur
        return
    for i, seg in enumerate(segments):
        if seg.end_time > 0:
            continue
        prev_end = 0.0
        next_start = total
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
