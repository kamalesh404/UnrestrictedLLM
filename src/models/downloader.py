"""Model downloader — download models from HuggingFace with progress tracking."""

import os
import hashlib
import logging
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger("unrestrictedllm.downloader")


@dataclass
class DownloadProgress:
    filename: str
    bytes_downloaded: int
    total_bytes: int
    percent: float
    speed_bytes_per_sec: float = 0

    @property
    def status_line(self) -> str:
        mb_done = self.bytes_downloaded / (1024 * 1024)
        mb_total = self.total_bytes / (1024 * 1024)
        speed = self.speed_bytes_per_sec / (1024 * 1024)
        return f"{self.filename}: {mb_done:.1f}/{mb_total:.1f} MB ({self.percent:.1f}%) @ {speed:.1f} MB/s"


class ModelDownloader:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._progress_callback: Optional[Callable[[DownloadProgress], None]] = None

    def set_progress_callback(self, callback: Callable[[DownloadProgress], None]):
        self._progress_callback = callback

    def download_model(self, repo_id: str, filename: str, local_dir: Optional[Path] = None) -> Path:
        try:
            from huggingface_hub import hf_hub_download, get_hf_token
        except ImportError:
            raise ImportError("huggingface_hub is required: pip install huggingface_hub")

        dest = local_dir or self.cache_dir
        dest.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading {filename} from {repo_id}...")
        path = hf_hub_download(
            repo_id=repo_id, filename=filename, local_dir=str(dest),
            token=get_hf_token(), resume_download=True,
        )
        logger.info(f"Downloaded to: {path}")
        return Path(path)

    def download_with_progress(self, repo_id: str, filename: str, local_dir: Optional[Path] = None) -> Path:
        import time
        import requests
        from huggingface_hub import get_hf_token

        dest = local_dir or self.cache_dir
        dest.mkdir(parents=True, exist_ok=True)
        output_path = dest / filename

        if output_path.exists():
            logger.info(f"Model already cached: {output_path}")
            return output_path

        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        headers = {}
        token = get_hf_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        resp = requests.head(url, headers=headers, allow_redirects=True, timeout=10)
        total = int(resp.headers.get("content-length", 0))

        resp = requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=30)
        resp.raise_for_status()

        downloaded = 0
        start = time.time()
        temp_path = output_path.with_suffix(".tmp")

        with open(temp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                elapsed = time.time() - start
                speed = downloaded / elapsed if elapsed > 0 else 0
                percent = (downloaded / total * 100) if total > 0 else 0
                if self._progress_callback:
                    self._progress_callback(DownloadProgress(
                        filename=filename, bytes_downloaded=downloaded,
                        total_bytes=total, percent=percent, speed_bytes_per_sec=speed,
                    ))

        temp_path.rename(output_path)
        logger.info(f"Downloaded: {output_path} ({downloaded / (1024*1024):.1f} MB)")
        return output_path

    def file_hash(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def is_downloaded(self, filename: str) -> bool:
        return (self.cache_dir / filename).exists()
