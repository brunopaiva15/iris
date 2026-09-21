"""The image sources. Every fetcher returns `ImageCandidate` objects: an
image, its origin page, its author and its licence as the source declares it.
The licence filter downstream is what decides."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ImageCandidate:
    source: str
    source_id: str
    observation_id: str
    original_url: str
    image_url: str
    author: str
    license_raw: str        # as the source gives it (URL or code)
    publisher: str = ''
    dataset_key: str = ''
    extra: dict | None = None
