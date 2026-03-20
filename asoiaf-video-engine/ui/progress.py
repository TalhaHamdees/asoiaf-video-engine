"""Progress screen — stage stepper, progress indicator, live log output."""

from nicegui import ui, app

from pipeline_runner import (
    PipelineProgress, PipelineStage, STAGE_LABELS,
    PHASE1_STAGES, PHASE2_STAGES, run_phase2,
)
from config import Config


def create(config: Config, progress: PipelineProgress):
    """Build the progress page."""

    phase = app.storage.general.get("current_phase", 1)
    # Also check URL query param
    try:
        from starlette.requests import Request
        # NiceGUI passes query params via app.storage
    except Exception:
        pass

    with ui.column().classes("w-full max-w-4xl mx-auto p-6 gap-4"):
        with ui.row().classes("w-full items-center gap-4"):
            ui.button(icon="arrow_back", on_click=lambda: ui.navigate.to("/")).props("flat")
            header = ui.label("Pipeline Progress").classes("text-2xl font-bold")

        ui.separator()

        # Stage stepper
        stages_display = PHASE1_STAGES if not _is_phase2(progress) else PHASE2_STAGES
        stage_labels_container = ui.row().classes("w-full justify-between gap-1 flex-wrap")
        stage_chips = {}

        with stage_labels_container:
            for stage in stages_display:
                chip = ui.chip(
                    STAGE_LABELS[stage],
                    icon=_stage_icon(stage),
                ).props("outline").classes("text-xs")
                stage_chips[stage] = chip

        # Status line
        status_label = ui.label("Waiting...").classes("text-lg font-medium text-gray-700")
        spinner = ui.spinner("dots", size="lg").classes("self-center")

        # Error display
        error_card = ui.card().classes("w-full bg-red-50 border-red-200 hidden")
        with error_card:
            ui.label("Error").classes("text-red-700 font-bold")
            error_text = ui.label("").classes("text-red-600 text-sm whitespace-pre-wrap font-mono")

        # Action buttons (shown when done)
        action_row = ui.row().classes("w-full gap-4 hidden")
        with action_row:
            shopping_btn = ui.button(
                "View Shopping List", icon="shopping_cart",
                on_click=lambda: ui.navigate.to("/shopping-list"),
            ).classes("bg-orange-500 text-white")
            continue_btn = ui.button(
                "Generate Video", icon="movie",
                on_click=lambda: _start_phase2(config, progress),
            ).classes("bg-green-600 text-white")
            done_btn = ui.button(
                "View Output", icon="folder_open",
                on_click=lambda: ui.navigate.to("/"),
            ).classes("bg-blue-500 text-white hidden")

        # Live log viewer
        with ui.expansion("Live Logs", icon="terminal").classes("w-full"):
            log_area = ui.log(max_lines=200).classes("w-full h-64")

        # Timer to poll progress
        last_log_count = {"value": 0}

        def update_progress():
            # Update stage chips
            current_stages = PHASE2_STAGES if _is_phase2(progress) else PHASE1_STAGES
            current_idx = -1
            for i, stage in enumerate(current_stages):
                if stage == progress.stage:
                    current_idx = i
                    break

            for i, stage in enumerate(current_stages):
                chip = stage_chips.get(stage)
                if chip is None:
                    continue
                if i < current_idx:
                    chip.props("color=green")
                elif i == current_idx:
                    chip.props("color=primary")
                else:
                    chip.props(remove="color")

            # Update status
            label = STAGE_LABELS.get(progress.stage, "Working...")
            status_label.set_text(label)

            # Push new log lines
            new_lines = progress.log_lines[last_log_count["value"]:]
            for line in new_lines:
                log_area.push(line)
            last_log_count["value"] = len(progress.log_lines)

            # Handle completion
            if not progress.running and progress.stage != PipelineStage.IDLE:
                spinner.set_visibility(False)

                if progress.stage == PipelineStage.ERROR:
                    error_card.classes(remove="hidden")
                    error_text.set_text(progress.error or "Unknown error")
                elif progress.stage == PipelineStage.DONE:
                    action_row.classes(remove="hidden")
                    shopping_btn.set_visibility(False)
                    continue_btn.set_visibility(False)
                    done_btn.classes(remove="hidden")
                elif progress.stage in (PipelineStage.SHOPPING_LIST, PipelineStage.PHASE1_DONE):
                    action_row.classes(remove="hidden")
                    if progress.result and progress.result.get("all_covered"):
                        shopping_btn.set_visibility(False)
                    else:
                        continue_btn.set_visibility(False)

            elif progress.running:
                spinner.set_visibility(True)

        ui.timer(0.5, update_progress)


def _is_phase2(progress: PipelineProgress) -> bool:
    return progress.stage in (
        PipelineStage.PICKUP_IMAGES, PipelineStage.COMPOSING,
        PipelineStage.CAPTIONS, PipelineStage.INGESTING,
        PipelineStage.CLEANUP, PipelineStage.DONE,
    )


def _start_phase2(config: Config, progress: PipelineProgress):
    run_phase2(config, progress)
    ui.navigate.to("/progress?phase=2")


def _stage_icon(stage: PipelineStage) -> str:
    icons = {
        PipelineStage.PARSING: "description",
        PipelineStage.TTS: "record_voice_over",
        PipelineStage.SILENCE_REMOVAL: "volume_off",
        PipelineStage.TIMESTAMPS: "timer",
        PipelineStage.ALIGNMENT: "align_horizontal_left",
        PipelineStage.RESEGMENT: "view_timeline",
        PipelineStage.LIBRARY_CHECK: "photo_library",
        PipelineStage.PICKUP_IMAGES: "add_photo_alternate",
        PipelineStage.COMPOSING: "movie_creation",
        PipelineStage.CAPTIONS: "subtitles",
        PipelineStage.INGESTING: "cloud_upload",
        PipelineStage.DONE: "check_circle",
    }
    return icons.get(stage, "circle")
