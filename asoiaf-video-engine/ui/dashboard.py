"""Dashboard — home screen showing project status and navigation."""

from pathlib import Path
from nicegui import ui

from config import Config
from pipeline import has_state, load_state


def create(config: Config):
    """Build the dashboard page."""

    with ui.column().classes("w-full max-w-4xl mx-auto p-6 gap-6"):
        # Header
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("ASOIAF Video Engine").classes("text-3xl font-bold text-gray-800")
            ui.label("FICTOPIA").classes("text-lg font-semibold text-red-500")

        ui.separator()

        # Project status card
        with ui.card().classes("w-full"):
            ui.label("Project Status").classes("text-xl font-semibold mb-2")

            state_exists = has_state(config)
            if state_exists:
                try:
                    state = load_state(config)
                    seg_count = len(state["segments"])
                    video_name = state["video_name"]
                    assignments = state["assignments"]
                    covered = sum(1 for a in assignments if a.image_path is not None)
                    needed = seg_count - covered

                    with ui.row().classes("gap-4"):
                        with ui.column().classes("gap-1"):
                            ui.label(f"Video: {video_name}").classes("font-medium")
                            ui.label(f"Segments: {seg_count}").classes("text-gray-600")
                            ui.label(f"Images: {covered}/{seg_count} covered").classes("text-gray-600")

                    if needed > 0:
                        ui.label(f"{needed} images still needed — check the Shopping List").classes(
                            "text-orange-600 font-medium mt-2"
                        )
                    else:
                        ui.label("All images ready — you can generate the video!").classes(
                            "text-green-600 font-medium mt-2"
                        )
                except Exception:
                    ui.label("Saved state found but could not be loaded.").classes("text-orange-600")
            else:
                ui.label("No active project. Start a new video below.").classes("text-gray-500")

        # Action buttons
        with ui.row().classes("w-full gap-4"):
            ui.button("New Video", icon="add", on_click=lambda: ui.navigate.to("/new")).classes(
                "bg-red-500 text-white"
            )

            if state_exists:
                ui.button("Shopping List", icon="shopping_cart",
                          on_click=lambda: ui.navigate.to("/shopping-list")).classes(
                    "bg-orange-500 text-white"
                )
                ui.button("Generate Video", icon="movie",
                          on_click=lambda: ui.navigate.to("/progress?phase=2")).classes(
                    "bg-green-600 text-white"
                )

            ui.button("Image Library", icon="photo_library",
                      on_click=lambda: ui.navigate.to("/library")).classes(
                "bg-blue-500 text-white"
            )
            ui.button("Settings", icon="settings",
                      on_click=lambda: ui.navigate.to("/settings")).classes(
                "bg-gray-500 text-white"
            )

        # Recent outputs
        output_dir = config.output_dir
        if output_dir.exists():
            videos = sorted(output_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
            if videos:
                with ui.card().classes("w-full"):
                    ui.label("Recent Outputs").classes("text-xl font-semibold mb-2")
                    for v in videos[:5]:
                        size_mb = v.stat().st_size / (1024 * 1024)
                        with ui.row().classes("items-center gap-2"):
                            ui.icon("video_file").classes("text-gray-400")
                            ui.label(v.name).classes("font-medium")
                            ui.label(f"{size_mb:.1f} MB").classes("text-gray-500 text-sm")
                            ui.button("Preview", icon="play_arrow",
                                      on_click=lambda p=v: ui.navigate.to(f"/preview?path={p}")).props(
                                "flat dense size=sm"
                            )
