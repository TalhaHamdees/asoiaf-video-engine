"""
Audio Processor — silence removal, normalization, timestamp remapping.
"""

import logging
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from config import AudioConfig
from modules.tts_generator import WordTimestamp

logger = logging.getLogger(__name__)


def remove_silence(audio_path: Path, config: AudioConfig, output_path: Path) -> Path:
    audio = AudioSegment.from_file(str(audio_path))
    original_dur = len(audio) / 1000.0

    nonsilent = detect_nonsilent(
        audio, min_silence_len=config.min_silence_ms,
        silence_thresh=config.silence_thresh_dbfs, seek_step=10,
    )

    if not nonsilent:
        audio.export(str(output_path), format="wav")
        return Path(output_path)

    chunks = []
    for start_ms, end_ms in nonsilent:
        ps = max(0, start_ms - config.keep_silence_ms)
        pe = min(len(audio), end_ms + config.keep_silence_ms)
        chunks.append(audio[ps:pe])

    cleaned = chunks[0]
    for c in chunks[1:]:
        cleaned += c

    cleaned_dur = len(cleaned) / 1000.0
    logger.info("Silence removal: %.1fs → %.1fs (removed %.1fs)", original_dur, cleaned_dur, original_dur - cleaned_dur)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cleaned.export(str(output_path), format="wav")
    return Path(output_path)


def normalize_audio(audio_path: Path, target_dbfs: float = -20.0, output_path: Path = None) -> Path:
    if output_path is None:
        output_path = audio_path
    audio = AudioSegment.from_file(str(audio_path))
    change = target_dbfs - audio.dBFS
    normalized = audio.apply_gain(change)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    normalized.export(str(output_path), format="wav")
    return Path(output_path)


def get_audio_duration(audio_path: Path) -> float:
    return len(AudioSegment.from_file(str(audio_path))) / 1000.0


def create_silence_mapping(original_path: Path, cleaned_path: Path, config: AudioConfig):
    original = AudioSegment.from_file(str(original_path))
    nonsilent = detect_nonsilent(
        original, min_silence_len=config.min_silence_ms,
        silence_thresh=config.silence_thresh_dbfs, seek_step=10,
    )
    mapping = []
    cursor = 0.0
    for start_ms, end_ms in nonsilent:
        ps = max(0, start_ms - config.keep_silence_ms)
        pe = min(len(original), end_ms + config.keep_silence_ms)
        dur = (pe - ps) / 1000.0
        mapping.append((ps / 1000.0, pe / 1000.0, cursor))
        cursor += dur
    return mapping


def remap_timestamp(t: float, mapping: list) -> float | None:
    for orig_s, orig_e, clean_s in mapping:
        if orig_s <= t <= orig_e:
            return clean_s + (t - orig_s)
    for orig_s, orig_e, clean_s in mapping:
        if t < orig_s:
            return clean_s
    if mapping:
        last_s, last_e, last_cs = mapping[-1]
        return last_cs + (last_e - last_s)
    return 0.0
