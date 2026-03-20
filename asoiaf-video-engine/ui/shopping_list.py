"""Shopping List screen — shows needed images with upload zones per segment."""

import shutil
from pathlib import Path
from nicegui import ui, events

from config import Config
from pipeline import load_state, save_state, has_state
from pipeline_runner import PipelineProgress, run_phase2


def create(config: Config, progress: PipelineProgress):
    """Build the shopping list page."""

    with ui.column().classes("w-full max-w-5xl mx-auto p-6 gap-4"):
        with ui.row().classes("w-full items-center gap-4"):
            ui.button(icon="arrow_back", on_click=lambda: ui.navigate.to("/")).props("flat")
            ui.label("Shopping List").classes("text-2xl font-bold")

        if not has_state(config):
            ui.label("No active project. Run Phase 1 first.").classes("text-gray-500")
            return

        try:
            state = load_state(config)
        except Exception as e:
            ui.label(f"Error loading state: {e}").classes("text-red-500")
            return

        segments = state["segments"]
        assignments = state["assignments"]

        # Show the text shopping list if it exists
        if config.shopping_list_path.exists():
            with ui.expansion("Shopping List (text)", icon="list").classes("w-full"):
                content = config.shopping_list_path.read_text(encoding="utf-8")
                ui.code(content).classes("w-full text-xs")

        ui.separator()

        # Stats
        covered = sum(1 for a in assignments if a.image_path is not None)
        needed = len(assignments) - covered
        ui.label(f"Coverage: {covered}/{len(assignments)} segments").classes("text-lg font-medium")
        if needed > 0:
            ui.label(f"{needed} images still needed").classes("text-orange-600")
        else:
            ui.label("All images covered!").classes("text-green-600 font-bold")

        ui.separator()

        # Per-segment cards
        input_dir = config.input_dir
        input_dir.mkdir(parents=True, exist_ok=True)

        segment_cards = {}

        for a in assignments:
            seg = segments[a.segment_index]
            seg_num = a.segment_index + 1
            has_image = a.image_path is not None

            with ui.card().classes(
                f"w-full {'bg-green-50 border-green-200' if has_image else 'bg-orange-50 border-orange-200'}"
            ):
                with ui.row().classes("w-full items-start gap-4"):
                    # Segment info
                    with ui.column().classes("flex-1 gap-1"):
                        with ui.row().classes("items-center gap-2"):
                            ui.badge(f"{seg_num:02d}", color="primary")
                            ui.label(f"{seg.start_time:.1f}s - {seg.end_time:.1f}s").classes(
                                "text-sm text-gray-500"
                            )
                            if has_image:
                                ui.badge("COVERED", color="green").props("outline")
                            else:
                                ui.badge("NEEDED", color="orange").props("outline")

                        ui.label(seg.text).classes("text-sm text-gray-700")

                        if a.search_query:
                            ui.label(f"Search: {a.search_query}").classes(
                                "text-xs text-gray-400 italic"
                            )
                        if a.description:
                            ui.label(f"Description: {a.description}").classes(
                                "text-xs text-gray-400 italic"
                            )

                    # Image preview / upload zone
                    with ui.column().classes("w-48 items-center gap-2"):
                        if has_image and Path(a.image_path).exists():
                            ui.image(a.image_path).classes("w-full rounded")
                        else:
                            # Upload zone for this segment
                            status_label = ui.label("Drop image here").classes(
                                "text-sm text-gray-400"
                            )

                            async def handle_upload(
                                e: events.UploadEventArguments,
                                seg_idx=a.segment_index,
                                label=status_label,
                            ):
                                ext = Path(e.name).suffix or ".jpg"
                                dest = input_dir / f"{seg_idx + 1:02d}{ext}"
                                with open(dest, "wb") as f:
                                    f.write(e.content.read())
                                label.set_text(f"Uploaded: {dest.name}")
                                ui.notify(f"Image saved for segment {seg_idx + 1}", type="positive")

                            ui.upload(
                                auto_upload=True,
                                on_upload=handle_upload,
                            ).props("accept='image/*' flat bordered dense").classes("w-full")

        ui.separator()

        # Action buttons
        with ui.row().classes("w-full justify-end gap-4"):
            def start_phase2():
                run_phase2(config, progress)
                ui.navigate.to("/progress?phase=2")

            ui.button(
                "Start Phase 2 (Generate Video)",
                icon="movie",
                on_click=start_phase2,
            ).classes("bg-green-600 text-white").props("size=lg")
