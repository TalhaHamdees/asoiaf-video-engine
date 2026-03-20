"""
Google Drive Sync — bidirectional sync for the shared image library.

Uses PyDrive2 for OAuth + file operations.
Credentials are stored locally in credentials.json (gitignored).
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DriveSync:
    """Sync local image library with a Google Drive folder."""

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, local_path: Path, folder_id: Optional[str] = None):
        self.local_path = Path(local_path)
        self.folder_id = folder_id or os.environ.get("GDRIVE_FOLDER_ID", "")
        self._drive = None

    @property
    def is_configured(self) -> bool:
        return bool(self.folder_id)

    def _get_drive(self):
        """Lazy-init the PyDrive2 GoogleDrive client."""
        if self._drive is not None:
            return self._drive

        try:
            from pydrive2.auth import GoogleAuth
            from pydrive2.drive import GoogleDrive
        except ImportError:
            raise RuntimeError("pydrive2 not installed. Run: pip install pydrive2")

        gauth = GoogleAuth()

        # Try to load saved credentials
        creds_path = Path("credentials.json")
        if creds_path.exists():
            gauth.LoadCredentialsFile(str(creds_path))

        if gauth.credentials is None:
            # First-time auth: opens browser
            gauth.LocalWebserverAuth()
        elif gauth.access_token_expired:
            gauth.Refresh()
        else:
            gauth.Authorize()

        gauth.SaveCredentialsFile(str(creds_path))
        self._drive = GoogleDrive(gauth)
        return self._drive

    def _list_remote_files(self) -> dict:
        """List files in the Drive folder. Returns {filename: file_obj}."""
        drive = self._get_drive()
        query = f"'{self.folder_id}' in parents and trashed=false"
        file_list = drive.ListFile({"q": query}).GetList()
        return {f["title"]: f for f in file_list if self._is_image(f["title"])}

    def _list_local_files(self) -> dict:
        """List local image files. Returns {filename: Path}."""
        files = {}
        if self.local_path.exists():
            for f in self.local_path.iterdir():
                if f.is_file() and self._is_image(f.name):
                    files[f.name] = f
        return files

    def _is_image(self, filename: str) -> bool:
        return Path(filename).suffix.lower() in self.IMAGE_EXTENSIONS

    def sync_down(self) -> dict:
        """Download new files from Drive that don't exist locally.

        Returns: {"downloaded": [filenames], "skipped": int}
        """
        if not self.is_configured:
            raise ValueError("Google Drive folder ID not configured")

        self.local_path.mkdir(parents=True, exist_ok=True)
        remote_files = self._list_remote_files()
        local_files = self._list_local_files()

        downloaded = []
        skipped = 0

        for name, remote_file in remote_files.items():
            if name in local_files:
                skipped += 1
                continue

            dest = self.local_path / name
            logger.info("Downloading: %s", name)
            remote_file.GetContentFile(str(dest))
            downloaded.append(name)

        logger.info("Drive sync down: %d downloaded, %d skipped", len(downloaded), skipped)
        return {"downloaded": downloaded, "skipped": skipped}

    def sync_up(self) -> dict:
        """Upload new local files that don't exist on Drive.

        Returns: {"uploaded": [filenames], "skipped": int}
        """
        if not self.is_configured:
            raise ValueError("Google Drive folder ID not configured")

        drive = self._get_drive()
        remote_files = self._list_remote_files()
        local_files = self._list_local_files()

        uploaded = []
        skipped = 0

        for name, local_path in local_files.items():
            if name in remote_files:
                skipped += 1
                continue

            logger.info("Uploading: %s", name)
            f = drive.CreateFile({
                "title": name,
                "parents": [{"id": self.folder_id}],
            })
            f.SetContentFile(str(local_path))
            f.Upload()
            uploaded.append(name)

        logger.info("Drive sync up: %d uploaded, %d skipped", len(uploaded), skipped)
        return {"uploaded": uploaded, "skipped": skipped}

    def sync_bidirectional(self) -> dict:
        """Download new remote files, then upload new local files.

        Returns combined results.
        """
        down = self.sync_down()
        up = self.sync_up()
        return {
            "downloaded": down["downloaded"],
            "uploaded": up["uploaded"],
            "skipped_down": down["skipped"],
            "skipped_up": up["skipped"],
        }
