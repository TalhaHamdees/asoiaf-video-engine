"""Image Library screen — grid view of library images with search and upload."""

import json
from pathlib import Path
from nicegui import ui, events

from config import Config


def create(config: Config):
    """Build the image library page."""

    library_path = config.image_search.local_library_path

    with ui.column().classes("w-full max-w-6xl mx-auto p-6 gap-4"):
        with ui.row().classes("w-full items-center gap-4"):
            ui.button(icon="arrow_back", on_click=lambda: ui.navigate.to("/")).props("flat")
            ui.label("Image Library").classes("text-2xl font-bold")

        if not library_path.exists():
            ui.label(f"Library directory not found: {library_path}").classes("text-gray-500")
            library_path.mkdir(parents=True, exist_ok=True)
            ui.label("Created empty library directory.").classes("text-gray-400 text-sm")

        # Stats
        all_images = _get_library_images(library_path)
        tags_index = _load_tags_index(library_path)
        all_tags = set()
        for tags in tags_index.values():
            all_tags.update(tags)

        ui.label(f"{len(all_images)} images in library").classes("text-lg text-gray-600")

        # Search / filter
        with ui.row().classes("w-full gap-4 items-end"):
            search_input = ui.input("Search by filename or tag", placeholder="e.g. dragon, winterfell").classes(
                "flex-1"
            ).props("outlined dense clearable")

            category_select = ui.select(
                ["All"] + sorted(all_tags),
                value="All",
                label="Filter by tag",
            ).classes("w-48").props("outlined dense")

        ui.separator()

        # Image grid container
        grid_container = ui.element("div").classes(
            "w-full grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4"
        )

        def refresh_grid():
            grid_container.clear()
            query = (search_input.value or "").lower().strip()
            tag_filter = category_select.value

            with grid_container:
                count = 0
                for img_path in all_images:
                    name_lower = img_path.stem.lower()
                    img_tags = tags_index.get(img_path.name, [])

                    # Apply filters
                    if query and query not in name_lower and not any(query in t.lower() for t in img_tags):
                        continue
                    if tag_filter != "All" and tag_filter not in img_tags:
                        continue

                    with ui.card().classes("p-1"):
                        ui.image(str(img_path)).classes("w-full h-32 object-cover rounded")
                        ui.label(img_path.stem).classes("text-xs text-gray-600 truncate")
                        if img_tags:
                            with ui.row().classes("flex-wrap gap-1"):
                                for tag in img_tags[:3]:
                                    ui.badge(tag).props("outline dense").classes("text-xs")

                    count += 1
                    if count >= 100:
                        ui.label("Showing first 100 results...").classes("text-gray-400 col-span-full")
                        break

                if count == 0:
                    ui.label("No images found.").classes("text-gray-400 col-span-full")

        refresh_grid()
        search_input.on("keydown.enter", lambda: refresh_grid())
        category_select.on_value_change(lambda: refresh_grid())

        ui.separator()

        # Upload new images
        with ui.card().classes("w-full"):
            ui.label("Upload New Images").classes("text-lg font-semibold mb-2")

            async def handle_upload(e: events.UploadEventArguments):
                dest = library_path / e.name
                with open(dest, "wb") as f:
                    f.write(e.content.read())
                ui.notify(f"Added {e.name} to library", type="positive")

            ui.upload(
                label="Drop images here",
                auto_upload=True,
                on_upload=handle_upload,
                multiple=True,
            ).props("accept='image/*' flat bordered").classes("w-full")

            ui.button("Refresh", icon="refresh", on_click=lambda: ui.navigate.to("/library")).props(
                "flat"
            )

        # Google Drive sync
        from modules.drive_sync import DriveSync
        drive = DriveSync(library_path)
        if drive.is_configured:
            with ui.card().classes("w-full"):
                ui.label("Google Drive Sync").classes("text-lg font-semibold mb-2")
                sync_status = ui.label("").classes("text-sm text-gray-500")

                async def do_sync():
                    sync_status.set_text("Syncing...")
                    try:
                        result = await ui.run_cpu_bound(drive.sync_bidirectional)
                        down = len(result["downloaded"])
                        up = len(result["uploaded"])
                        sync_status.set_text(f"Done: {down} downloaded, {up} uploaded")
                        ui.notify(f"Sync complete: {down} down, {up} up", type="positive")
                    except Exception as e:
                        sync_status.set_text(f"Error: {e}")
                        ui.notify(f"Sync failed: {e}", type="negative")

                ui.button("Sync with Drive", icon="sync", on_click=do_sync).classes(
                    "bg-blue-500 text-white"
                )


def _get_library_images(library_path: Path) -> list:
    """Get all image files in the library, sorted by modification time."""
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = []
    if library_path.exists():
        for f in library_path.rglob("*"):
            if f.suffix.lower() in extensions and f.is_file():
                images.append(f)
    images.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return images


def _load_tags_index(library_path: Path) -> dict:
    """Load the tags index if it exists (generated by image_manager auto-tagger)."""
    tags_file = library_path / "tags_index.json"
    if tags_file.exists():
        try:
            with open(tags_file) as f:
                return json.load(f)
        except Exception:
            pass
    return {}
