"""
TTS Generator — ElevenLabs integration.
Generates voiceover audio and retrieves word-level timestamps.
"""

import base64
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import requests

from config import ElevenLabsConfig

logger = logging.getLogger(__name__)

ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float


def generate_voiceover(
    text: str, config: ElevenLabsConfig, output_path: Path,
) -> Path:
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

    logger.info("Generating voiceover (%d chars)...", len(text))
    response = requests.post(url, json=payload, headers=headers, timeout=120)
    response.raise_for_status()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)
    logger.info("Voiceover saved: %s (%.1f KB)", output_path, len(response.content) / 1024)
    return output_path


def generate_voiceover_with_timestamps(
    text: str, config: ElevenLabsConfig, output_audio_path: Path,
) -> tuple[Path, list[WordTimestamp]]:
    url = f"{ELEVENLABS_BASE_URL}/text-to-speech/{config.voice_id}/with-timestamps"
    headers = {"xi-api-key": config.api_key, "Content-Type": "application/json"}
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

    audio_bytes = base64.b64decode(data["audio_base64"])
    output_audio_path = Path(output_audio_path)
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    output_audio_path.write_bytes(audio_bytes)

    timestamps = []
    alignment = data.get("alignment", {})
    characters = alignment.get("characters", [])
    starts = alignment.get("character_start_times_seconds", [])
    ends = alignment.get("character_end_times_seconds", [])

    if characters and starts and ends:
        timestamps = _chars_to_words(characters, starts, ends)
        logger.info("Got %d word timestamps from ElevenLabs.", len(timestamps))

    return output_audio_path, timestamps


def _chars_to_words(chars, starts, ends) -> list[WordTimestamp]:
    words = []
    current_word = ""
    word_start = None

    for char, start, end in zip(chars, starts, ends):
        if char == " ":
            if current_word:
                words.append(WordTimestamp(current_word, word_start, end))
                current_word = ""
                word_start = None
        else:
            if word_start is None:
                word_start = start
            current_word += char

    if current_word and word_start is not None:
        words.append(WordTimestamp(current_word, word_start, ends[-1]))

    return words
