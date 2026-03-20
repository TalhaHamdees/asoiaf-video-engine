"""Settings screen — API keys, Drive auth, video/caption settings."""

import os
from pathlib import Path
from nicegui import ui

from config import Config


def create(config: Config):
    """Build the settings page."""

    env_path = Path(".env")

    # Load current .env values
    env_vars = _load_env(env_path)

    with ui.column().classes("w-full max-w-3xl mx-auto p-6 gap-4"):
        with ui.row().classes("w-full items-center gap-4"):
            ui.button(icon="arrow_back", on_click=lambda: ui.navigate.to("/")).props("flat")
            ui.label("Settings").classes("text-2xl font-bold")

        ui.separator()

        # API Keys section
        with ui.card().classes("w-full"):
            ui.label("API Keys").classes("text-xl font-semibold mb-2")
            ui.label("These are saved to your .env file (never committed to git).").classes(
                "text-sm text-gray-500 mb-4"
            )

            elevenlabs_key = ui.input(
                "ElevenLabs API Key",
                value=env_vars.get("ELEVENLABS_API_KEY", ""),
                password=True,
                password_toggle_button=True,
            ).classes("w-full").props("outlined")

            elevenlabs_voice = ui.input(
                "ElevenLabs Voice ID",
                value=env_vars.get("ELEVENLABS_VOICE_ID", config.elevenlabs.voice_id),
            ).classes("w-full").props("outlined")

            elevenlabs_url = ui.input(
                "ElevenLabs Base URL",
                value=env_vars.get("ELEVENLABS_BASE_URL", config.elevenlabs.base_url),
            ).classes("w-full").props("outlined")

            anthropic_key = ui.input(
                "Anthropic API Key",
                value=env_vars.get("ANTHROPIC_API_KEY", ""),
                password=True,
                password_toggle_button=True,
            ).classes("w-full").props("outlined")

        # Google Drive section
        with ui.card().classes("w-full"):
            ui.label("Google Drive (Optional)").classes("text-xl font-semibold mb-2")
            ui.label("For syncing the shared image library.").classes(
                "text-sm text-gray-500 mb-4"
            )

            gdrive_folder = ui.input(
                "Google Drive Folder ID",
                value=env_vars.get("GDRIVE_FOLDER_ID", ""),
            ).classes("w-full").props("outlined")

            creds_path = Path("credentials.json")
            if creds_path.exists():
                ui.label("Google Drive credentials found.").classes("text-green-600 text-sm")
            else:
                ui.label("No credentials.json found. Upload or run Drive auth to set up.").classes(
                    "text-orange-600 text-sm"
                )

        # Video settings
        with ui.card().classes("w-full"):
            ui.label("Video Settings").classes("text-xl font-semibold mb-2")

            with ui.row().classes("w-full gap-4"):
                ui.number("Width", value=config.video.width, min=480, max=3840, step=1).classes(
                    "flex-1"
                ).props("outlined dense readonly")
                ui.number("Height", value=config.video.height, min=480, max=3840, step=1).classes(
                    "flex-1"
                ).props("outlined dense readonly")
                ui.number("FPS", value=config.video.fps, min=24, max=120, step=1).classes(
                    "flex-1"
                ).props("outlined dense readonly")

            ui.label("Video dimensions are configured in config.py").classes(
                "text-xs text-gray-400"
            )

        # Caption settings preview
        with ui.card().classes("w-full"):
            ui.label("Caption Settings").classes("text-xl font-semibold mb-2")
            with ui.row().classes("gap-4 flex-wrap"):
                ui.label(f"Style: {config.caption.style}").classes("text-sm")
                ui.label(f"Font size: {config.caption.font_size}").classes("text-sm")
                ui.label(f"Words per group: {config.caption.words_per_group}").classes("text-sm")
                ui.label(f"Highlight: {config.caption.highlight_color}").classes("text-sm")
            ui.label("Caption settings are configured in config.py").classes(
                "text-xs text-gray-400"
            )

        ui.separator()

        # Save button
        def save_settings():
            new_env = {
                "ELEVENLABS_API_KEY": elevenlabs_key.value.strip(),
                "ELEVENLABS_VOICE_ID": elevenlabs_voice.value.strip(),
                "ELEVENLABS_BASE_URL": elevenlabs_url.value.strip(),
                "ANTHROPIC_API_KEY": anthropic_key.value.strip(),
            }
            gd = gdrive_folder.value.strip()
            if gd:
                new_env["GDRIVE_FOLDER_ID"] = gd

            _save_env(env_path, new_env)

            # Also update running config
            os.environ.update(new_env)
            config.elevenlabs.api_key = new_env["ELEVENLABS_API_KEY"]
            config.elevenlabs.voice_id = new_env["ELEVENLABS_VOICE_ID"]
            config.elevenlabs.base_url = new_env["ELEVENLABS_BASE_URL"]
            config.image_search.anthropic_api_key = new_env["ANTHROPIC_API_KEY"]

            ui.notify("Settings saved!", type="positive")

        ui.button("Save Settings", icon="save", on_click=save_settings).classes(
            "bg-green-600 text-white self-end"
        ).props("size=lg")

        # Whisper model management
        with ui.card().classes("w-full"):
            ui.label("Whisper Model").classes("text-xl font-semibold mb-2")
            ui.label("Used for word-level timestamps when ElevenLabs timestamps are unavailable.").classes(
                "text-sm text-gray-500 mb-2"
            )
            whisper_status = ui.label("").classes("text-sm")

            # Check if model is cached
            try:
                from faster_whisper.utils import download_model
                whisper_status.set_text("Checking model cache...")
            except ImportError:
                whisper_status.set_text("faster-whisper not installed.")

            async def download_whisper():
                whisper_status.set_text("Downloading Whisper 'base' model...")
                try:
                    import asyncio
                    await asyncio.to_thread(
                        lambda: __import__("faster_whisper").WhisperModel("base", device="cpu")
                    )
                    whisper_status.set_text("Whisper 'base' model ready!")
                    ui.notify("Whisper model downloaded.", type="positive")
                except Exception as e:
                    whisper_status.set_text(f"Download failed: {e}")
                    ui.notify(f"Whisper download failed: {e}", type="negative")

            ui.button("Download / Verify Model", icon="download", on_click=download_whisper).props(
                "flat"
            )


def _load_env(path: Path) -> dict:
    """Parse a .env file into a dict."""
    env = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip().strip("'\"")
    return env


def _save_env(path: Path, values: dict):
    """Write values to a .env file, preserving comments and unknown keys."""
    existing_lines = []
    if path.exists():
        existing_lines = path.read_text(encoding="utf-8").splitlines()

    written_keys = set()
    new_lines = []

    for line in existing_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in values:
                new_lines.append(f"{key}={values[key]}")
                written_keys.add(key)
                continue
        new_lines.append(line)

    # Add any new keys not already in the file
    for key, value in values.items():
        if key not in written_keys and value:
            new_lines.append(f"{key}={value}")

    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
