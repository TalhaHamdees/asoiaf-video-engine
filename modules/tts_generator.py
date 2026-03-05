"""
TTS Generator — ElevenLabs integration.

Generates voiceover audio from script text and retrieves
word-level timestamps for caption alignment.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import requests

from config import ElevenLabsConfig

logger = logging.getLogger(__name__)

ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"


@dataclass
class WordTimestamp:
    """A single word with its timing in the audio."""
    word: str
    start: float  # seconds
    end: float    # seconds


def generate_voiceover(
    text: str,
    config: ElevenLabsConfig,
    output_path: Path,
) -> Path:
    """
    Generate TTS audio using ElevenLabs.

    Args:
        text: The script text to convert to speech.
        config: ElevenLabs configuration.
        output_path: Where to save the audio file.

    Returns:
        Path to the generated audio file.
    """
    url = f"{ELEVENLABS_BASE_URL}/text-to-speech/{config.voice_id}"

    headers = {
        "xi-api-key": config.api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }

    payload = {
        "text": text,
        "model_id": config.model_id,
        "voice_settings": {
            "stability": config.stability,
            "similarity_boost": config.similarity_boost,
            "style": config.style,
            "use_speaker_boost": True,
        },
    }

    logger.info("Generating voiceover via ElevenLabs (%d chars)...", len(text))
    response = requests.post(url, json=payload, headers=headers, timeout=120)
    response.raise_for_status()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)

    logger.info("Voiceover saved to %s (%.1f KB)", output_path, len(response.content) / 1024)
    return output_path


def generate_voiceover_with_timestamps(
    text: str,
    config: ElevenLabsConfig,
    output_audio_path: Path,
) -> tuple[Path, list[WordTimestamp]]:
    """
    Generate TTS audio AND get word-level timestamps.

    Uses the ElevenLabs streaming endpoint with timestamp alignment.

    Returns:
        Tuple of (audio_path, list of WordTimestamps).
    """
    url = f"{ELEVENLABS_BASE_URL}/text-to-speech/{config.voice_id}/with-timestamps"

    headers = {
        "xi-api-key": config.api_key,
        "Content-Type": "application/json",
    }

    payload = {
        "text": text,
        "model_id": config.model_id,
        "voice_settings": {
            "stability": config.stability,
            "similarity_boost": config.similarity_boost,
            "style": config.style,
            "use_speaker_boost": True,
        },
    }

    logger.info("Generating voiceover with timestamps (%d chars)...", len(text))
    response = requests.post(url, json=payload, headers=headers, timeout=120)
    response.raise_for_status()

    data = response.json()

    # Save audio (base64 encoded in response)
    import base64
    audio_bytes = base64.b64decode(data["audio_base64"])
    output_audio_path = Path(output_audio_path)
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    output_audio_path.write_bytes(audio_bytes)

    # Parse word timestamps
    timestamps = []
    alignment = data.get("alignment", {})
    characters = alignment.get("characters", [])
    char_start_times = alignment.get("character_start_times_seconds", [])
    char_end_times = alignment.get("character_end_times_seconds", [])

    if characters and char_start_times and char_end_times:
        timestamps = _chars_to_words(characters, char_start_times, char_end_times)
        logger.info("Extracted %d word timestamps from ElevenLabs.", len(timestamps))
    else:
        logger.warning("No alignment data returned — will fall back to Whisper.")

    return output_audio_path, timestamps


def _chars_to_words(
    characters: list[str],
    start_times: list[float],
    end_times: list[float],
) -> list[WordTimestamp]:
    """Convert character-level timestamps to word-level timestamps."""
    words = []
    current_word = ""
    word_start = None

    for char, start, end in zip(characters, start_times, end_times):
        if char == " ":
            if current_word:
                words.append(WordTimestamp(
                    word=current_word,
                    start=word_start,
                    end=end,
                ))
                current_word = ""
                word_start = None
        else:
            if word_start is None:
                word_start = start
            current_word += char

    # Don't forget the last word
    if current_word and word_start is not None:
        words.append(WordTimestamp(
            word=current_word,
            start=word_start,
            end=end_times[-1] if end_times else word_start + 0.1,
        ))

    return words
