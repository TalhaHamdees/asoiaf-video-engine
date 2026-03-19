"""
TTS Generator — Voicer API integration (ElevenLabs-compatible proxy).

Uses an async task-based workflow:
    1. POST /tasks → get task_id
    2. Poll GET /tasks/{task_id}/status until "ending"
    3. GET /tasks/{task_id}/result → download MP3

Word-level timestamps are handled separately by Whisper (transcriber module)
since this API doesn't support the with-timestamps endpoint.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from config import ElevenLabsConfig

logger = logging.getLogger(__name__)


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float


def generate_voiceover(
    text: str, config: ElevenLabsConfig, output_path: Path,
) -> Path:
    """
    Generate voiceover via the Voicer async TTS API.

    Submits a task, polls until complete, downloads the MP3 result.
    """
    base = config.base_url.rstrip("/")
    headers = {"X-API-Key": config.api_key, "Content-Type": "application/json"}

    # Step 1: Submit TTS task
    # Prefer saved template_uuid (pre-configured on the server) over inline settings
    if config.template_uuid:
        payload = {
            "text": text,
            "template_uuid": config.template_uuid,
        }
    else:
        payload = {
            "text": text,
            "template": {
                "voice_id": config.voice_id,
                "public_owner_id": "",
                "model_id": config.model_id,
                "voice_settings": {
                    "stability": config.stability,
                    "similarity_boost": config.similarity_boost,
                    "use_speaker_boost": True,
                    "speed": config.speed,
                },
            },
        }

    logger.info("Submitting TTS task (%d chars)...", len(text))
    resp = requests.post(f"{base}/tasks", json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    task_data = resp.json()
    task_id = task_data["task_id"]
    logger.info("Task created: %s", task_id)

    # Step 2: Poll for completion
    start_time = time.time()
    terminal_statuses = {"ending", "error", "error_handled"}

    while True:
        elapsed = time.time() - start_time
        if elapsed > config.poll_timeout:
            raise TimeoutError(
                f"TTS task {task_id} did not complete within {config.poll_timeout}s"
            )

        time.sleep(config.poll_interval)

        status_resp = requests.get(
            f"{base}/tasks/{task_id}/status", headers=headers, timeout=15
        )
        status_resp.raise_for_status()
        status_data = status_resp.json()
        status = status_data.get("status", "")
        logger.info(
            "  Task %s: %s (%.0fs elapsed)",
            task_id, status_data.get("status_label", status), elapsed,
        )

        if status in ("error", "error_handled"):
            raise RuntimeError(f"TTS task failed: {status_data}")

        if status == "ending":
            break

    # Step 3: Download result
    logger.info("Downloading audio...")
    result_resp = requests.get(
        f"{base}/tasks/{task_id}/result", headers=headers, timeout=60
    )
    result_resp.raise_for_status()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(result_resp.content)
    logger.info(
        "Voiceover saved: %s (%.1f KB)", output_path, len(result_resp.content) / 1024
    )
    return output_path


def generate_voiceover_with_timestamps(
    text: str, config: ElevenLabsConfig, output_audio_path: Path,
) -> tuple[Path, list[WordTimestamp]]:
    """
    Generate voiceover. This API does not support word timestamps,
    so we return an empty list — the pipeline will fall back to Whisper.
    """
    output_path = generate_voiceover(text, config, output_audio_path)
    return output_path, []
