"""Seed pack registry: fetch index from GitHub, resolve download URLs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_URL = "https://raw.githubusercontent.com/smart-memory/seed-registry/main/index.json"


@dataclass
class PackInfo:
    """Registry entry for a seed pack."""

    name: str
    version: str
    url: str
    size_mb: float = 0.0
    domain: str = ""
    description: str = ""

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> PackInfo:
        return cls(
            name=name,
            version=data.get("version", ""),
            url=data.get("url", ""),
            size_mb=data.get("size_mb", 0.0),
            domain=data.get("domain", ""),
            description=data.get("description", ""),
        )


class PackRegistry:
    """Fetches and queries the seed pack registry."""

    def __init__(self, registry_url: str = DEFAULT_REGISTRY_URL):
        self._url = registry_url
        self._packs: dict[str, PackInfo] = {}

    def fetch(self) -> list[PackInfo]:
        """Fetch the registry index. Returns list of available packs."""
        try:
            import httpx

            response = httpx.get(self._url, timeout=10.0)
            if response.status_code != 200:
                logger.debug("Pack registry returned %d", response.status_code)
                return []
            data = response.json()
        except Exception:
            logger.debug("Pack registry not reachable")
            return []

        packs_data = data.get("packs", {})
        self._packs = {name: PackInfo.from_dict(name, info) for name, info in packs_data.items()}
        return list(self._packs.values())

    def get(self, name: str) -> PackInfo | None:
        """Get info for a specific pack by name."""
        return self._packs.get(name)

    def download(self, pack_info: PackInfo, target_dir: str) -> str:
        """Download and extract a seed pack.

        Returns:
            Path to the extracted pack directory.
        """
        import tempfile
        from pathlib import Path

        import httpx

        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        pack_dir = target / pack_info.name

        if pack_dir.exists():
            logger.info("Pack directory already exists: %s", pack_dir)
            return str(pack_dir)

        # Download tarball
        response = httpx.get(pack_info.url, timeout=60.0, follow_redirects=True)
        response.raise_for_status()

        # Write to temp file and extract
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        import tarfile

        with tarfile.open(tmp_path, "r:gz") as tar:
            tar.extractall(path=str(target), filter="data")

        Path(tmp_path).unlink(missing_ok=True)
        logger.info("Downloaded pack %s to %s", pack_info.name, pack_dir)
        return str(pack_dir)
