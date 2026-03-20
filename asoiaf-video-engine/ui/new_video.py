"""New Video screen — script input, title, voice config, start Phase 1."""

from pathlib import Path
from nicegui import ui, events

from config import Config
from pipeline_runner import PipelineProgress, run_phase1


def create(config: Config, progress: PipelineProgress):
    """Build the new video page."""

    script_text = {"value": ""}
    title_input = {"value": "video"}

    with ui.column().classes("w-full max-w-4xl mx-auto p-6 gap-4"):
        # Header
        with ui.row().classes("w-full items-center gap-4"):
            ui.button(icon="arrow_back", on_click=lambda: ui.navigate.to("/")).props("flat")
            ui.label("New Video").classes("text-2xl font-bold")

        ui.separator()

        # Title
        title = ui.input("Video Title", value="video").classes("w-full").props("outlined")
        title.on_value_change(lambda e: title_input.update({"value": e.value}))

        # Script input
        ui.label("Script").classes("text-lg font-semibold")
        ui.label("Paste your script or upload a .txt file.").classes("text-gray-500 text-sm")

        textarea = ui.textarea("Script text").classes("w-full").props(
            "outlined rows=12 placeholder='Paste your ASOIAF script here...'"
        )
        textarea.on_value_change(lambda e: script_text.update({"value": e.value}))

        # File upload
        async def handle_upload(e: events.UploadEventArguments):
            content = e.content.read().decode("utf-8")
            textarea.set_value(content)
            script_text["value"] = content
            ui.notify(f"Loaded {e.name}", type="positive")

        ui.upload(
            label="Or upload a .txt file",
            auto_upload=True,
            on_upload=handle_upload,
        ).props("accept=.txt flat bordered").classes("w-full max-w-sm")

        # Voice override (optional)
        with ui.expansion("Voice Settings", icon="record_voice_over").classes("w-full"):
            voice_id = ui.input(
                "Voice ID (leave blank for default)",
                value="",
            ).classes("w-full").props("outlined dense")

        ui.separator()

        # Start button
        def start_phase1():
            text = script_text["value"].strip()
            if not text:
                ui.notify("Please enter a script first.", type="warning")
                return
            if not config.elevenlabs.api_key:
                ui.notify("ElevenLabs API key not set. Check Settings.", type="negative")
                return

            vid = voice_id.value.strip()
            if vid:
                config.elevenlabs.voice_id = vid

            run_phase1(text, config, title.value or "video", progress)
            ui.navigate.to("/progress?phase=1")

        ui.button(
            "Start Phase 1",
            icon="rocket_launch",
            on_click=start_phase1,
        ).classes("bg-red-500 text-white self-end").props("size=lg")
