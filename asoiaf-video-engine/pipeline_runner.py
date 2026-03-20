"""
Pipeline Runner — wraps pipeline.py with progress tracking for the GUI.

Intercepts log messages from the pipeline to track stage transitions
and progress, without modifying the pipeline internals.
"""

import logging
import threading
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable


class PipelineStage(Enum):
    IDLE = "idle"
    PARSING = "parsing"
    TTS = "tts"
    SILENCE_REMOVAL = "silence_removal"
    TIMESTAMPS = "timestamps"
    ALIGNMENT = "alignment"
    RESEGMENT = "resegment"
    LIBRARY_CHECK = "library_check"
    SHOPPING_LIST = "shopping_list"
    PHASE1_DONE = "phase1_done"
    PICKUP_IMAGES = "pickup_images"
    COMPOSING = "composing"
    CAPTIONS = "captions"
    INGESTING = "ingesting"
    CLEANUP = "cleanup"
    DONE = "done"
    ERROR = "error"


# Map log message substrings to stages
_STAGE_TRIGGERS = {
    "PHASE 1: PREPARE": PipelineStage.PARSING,
    "Parsed": PipelineStage.PARSING,
    "Generating voiceover": PipelineStage.TTS,
    "Removing silence": PipelineStage.SILENCE_REMOVAL,
    "Running Whisper": PipelineStage.TIMESTAMPS,
    "word timestamps from ElevenLabs": PipelineStage.TIMESTAMPS,
    "Aligning segments": PipelineStage.ALIGNMENT,
    "Re-segmenting": PipelineStage.RESEGMENT,
    "Checking image library": PipelineStage.LIBRARY_CHECK,
    "Shopping list saved": PipelineStage.SHOPPING_LIST,
    "All segments covered": PipelineStage.PHASE1_DONE,
    "PHASE 2: GENERATE VIDEO": PipelineStage.PICKUP_IMAGES,
    "Composing video": PipelineStage.COMPOSING,
    "Adding captions": PipelineStage.CAPTIONS,
    "Ingesting new images": PipelineStage.INGESTING,
    "Cleared input folder": PipelineStage.CLEANUP,
    "VIDEO COMPLETE": PipelineStage.DONE,
}

STAGE_LABELS = {
    PipelineStage.IDLE: "Ready",
    PipelineStage.PARSING: "Parsing script",
    PipelineStage.TTS: "Generating voiceover",
    PipelineStage.SILENCE_REMOVAL: "Removing silence",
    PipelineStage.TIMESTAMPS: "Extracting timestamps",
    PipelineStage.ALIGNMENT: "Aligning segments",
    PipelineStage.RESEGMENT: "Re-segmenting",
    PipelineStage.LIBRARY_CHECK: "Checking image library",
    PipelineStage.SHOPPING_LIST: "Shopping list ready",
    PipelineStage.PHASE1_DONE: "Phase 1 complete",
    PipelineStage.PICKUP_IMAGES: "Picking up images",
    PipelineStage.COMPOSING: "Composing video",
    PipelineStage.CAPTIONS: "Adding captions",
    PipelineStage.INGESTING: "Ingesting to library",
    PipelineStage.CLEANUP: "Cleaning up",
    PipelineStage.DONE: "Done!",
    PipelineStage.ERROR: "Error",
}

# Stages in Phase 1 order, for stepper display
PHASE1_STAGES = [
    PipelineStage.PARSING, PipelineStage.TTS, PipelineStage.SILENCE_REMOVAL,
    PipelineStage.TIMESTAMPS, PipelineStage.ALIGNMENT, PipelineStage.RESEGMENT,
    PipelineStage.LIBRARY_CHECK,
]

PHASE2_STAGES = [
    PipelineStage.PICKUP_IMAGES, PipelineStage.COMPOSING,
    PipelineStage.CAPTIONS, PipelineStage.INGESTING, PipelineStage.DONE,
]


@dataclass
class PipelineProgress:
    stage: PipelineStage = PipelineStage.IDLE
    log_lines: list = field(default_factory=list)
    error: Optional[str] = None
    result: Optional[dict] = None
    running: bool = False

    def reset(self):
        self.stage = PipelineStage.IDLE
        self.log_lines.clear()
        self.error = None
        self.result = None
        self.running = False


class PipelineLogHandler(logging.Handler):
    """Intercepts pipeline log messages to update progress state."""

    def __init__(self, progress: PipelineProgress):
        super().__init__()
        self.progress = progress

    def emit(self, record):
        msg = self.format(record)
        self.progress.log_lines.append(msg)
        # Keep max 500 lines
        if len(self.progress.log_lines) > 500:
            self.progress.log_lines = self.progress.log_lines[-300:]

        # Check for stage transitions
        log_msg = record.getMessage()
        for trigger, stage in _STAGE_TRIGGERS.items():
            if trigger in log_msg:
                self.progress.stage = stage
                break


def run_phase1(
    script_text: str,
    config,
    video_name: str,
    progress: PipelineProgress,
    on_done: Optional[Callable] = None,
):
    """Run Phase 1 in a background thread with progress tracking."""
    from pipeline import phase_prepare

    progress.reset()
    progress.running = True

    handler = PipelineLogHandler(progress)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"))

    # Attach handler to pipeline and module loggers
    loggers = [
        logging.getLogger("pipeline"),
        logging.getLogger("modules"),
        logging.getLogger("tts"),
        logging.getLogger("audio"),
        logging.getLogger("image_manager"),
        logging.getLogger("video_composer"),
        logging.getLogger("transcriber"),
    ]
    for lg in loggers:
        lg.addHandler(handler)

    def _run():
        try:
            result = phase_prepare(script_text, config, video_name, auto_continue=False)
            progress.result = result
            if result["all_covered"]:
                progress.stage = PipelineStage.PHASE1_DONE
            else:
                progress.stage = PipelineStage.SHOPPING_LIST
        except Exception as e:
            progress.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            progress.stage = PipelineStage.ERROR
        finally:
            progress.running = False
            for lg in loggers:
                lg.removeHandler(handler)
            if on_done:
                on_done()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


def run_phase2(
    config,
    progress: PipelineProgress,
    on_done: Optional[Callable] = None,
):
    """Run Phase 2 in a background thread with progress tracking."""
    from pipeline import phase_continue

    progress.reset()
    progress.running = True

    handler = PipelineLogHandler(progress)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"))

    loggers = [
        logging.getLogger("pipeline"),
        logging.getLogger("modules"),
        logging.getLogger("video_composer"),
        logging.getLogger("image_manager"),
    ]
    for lg in loggers:
        lg.addHandler(handler)

    def _run():
        try:
            output_path = phase_continue(config)
            progress.result = {"output_path": str(output_path)}
            progress.stage = PipelineStage.DONE
        except Exception as e:
            progress.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            progress.stage = PipelineStage.ERROR
        finally:
            progress.running = False
            for lg in loggers:
                lg.removeHandler(handler)
            if on_done:
                on_done()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread
