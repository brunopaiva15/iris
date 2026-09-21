"""The manifest: one JSON line per image, which never loses its provenance.

`dataset/manifest.jsonl` is the single source of truth about what the set
holds. The image folders rebuild from it; the reverse is not true. The
ATTRIBUTIONS.md and attributions.csv files are derived from it.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

STATUS_KEPT = 'kept'
STATUS_DUPLICATE = 'duplicate'
STATUS_REJECTED = 'rejected'
STATUS_REVIEW = 'review'


@dataclass
class ImageRecord:
    species: str
    internal_plant_id: str
    source: str                 # gbif | inaturalist | wikimedia | plantnet300k
    source_id: str              # identifier at the source (occurrence#media)
    original_url: str           # the page the image comes from
    image_url: str              # the downloaded file
    author: str
    license: str                # "CC BY 4.0"
    license_url: str
    downloaded_at: str          # ISO 8601, UTC
    checksum: str               # sha256 of the stored bytes
    path: str = ''              # path relative to dataset/
    observation_id: str = ''    # groups the photographs of one observation
    publisher: str = ''
    dataset_key: str = ''
    width: int = 0
    height: int = 0
    phash: str = ''             # 16 hexadecimal characters
    status: str = STATUS_KEPT
    reason: str = ''            # why rejected / to review / a duplicate of what
    duplicate_of: str = ''      # checksum of the original when status == duplicate
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> 'ImageRecord':
        data = json.loads(line)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


class Manifest:
    """The manifest in memory, backed by a JSONL file appended as it goes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.records: list[ImageRecord] = []
        self._by_checksum: dict[str, ImageRecord] = {}
        self._by_source_id: dict[tuple[str, str], ImageRecord] = {}
        if self.path.exists():
            with open(self.path, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self._index(ImageRecord.from_json(line))

    def _index(self, r: ImageRecord) -> None:
        self.records.append(r)
        if r.checksum:
            self._by_checksum.setdefault(r.checksum, r)
        self._by_source_id[(r.source, r.source_id)] = r

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[ImageRecord]:
        return iter(self.records)

    def has_source(self, source: str, source_id: str) -> bool:
        return (source, source_id) in self._by_source_id

    def by_checksum(self, checksum: str) -> ImageRecord | None:
        return self._by_checksum.get(checksum)

    def append(self, record: ImageRecord) -> None:
        """Append and write immediately: an interruption loses nothing."""
        self._index(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(record.to_json() + '\n')

    def rewrite(self) -> None:
        """Rewrite the whole file after a pass that changes statuses."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.jsonl.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            for r in self.records:
                f.write(r.to_json() + '\n')
        tmp.replace(self.path)

    def kept(self) -> list[ImageRecord]:
        return [r for r in self.records if r.status == STATUS_KEPT]

    def stats(self) -> dict:
        """Counts per species and per status, and per licence for the kept ones."""
        per_species: dict[str, dict[str, int]] = {}
        licenses: dict[str, int] = {}
        sources: dict[str, int] = {}
        for r in self.records:
            s = per_species.setdefault(r.species, {})
            s[r.status] = s.get(r.status, 0) + 1
            if r.status == STATUS_KEPT:
                licenses[r.license] = licenses.get(r.license, 0) + 1
                sources[r.source] = sources.get(r.source, 0) + 1
        return {
            'images': len(self.records),
            'kept': sum(1 for r in self.records if r.status == STATUS_KEPT),
            'species': {k: dict(sorted(v.items())) for k, v in sorted(per_species.items())},
            'licenses': dict(sorted(licenses.items())),
            'sources': dict(sorted(sources.items())),
        }


def write_attributions(records: Iterable[ImageRecord], md_path: str | Path, csv_path: str | Path) -> int:
    """Generate ATTRIBUTIONS.md and attributions.csv for the kept images.

    Every line names the author, the licence and the source: that is what CC BY
    requires, and what must travel with the model. Copy these files off the
    training machine before it is returned — they are the only record of where
    a given set of images came from, and the collection does not rebuild
    identically.
    """
    kept = sorted((r for r in records if r.status == STATUS_KEPT), key=lambda r: (r.species, r.author, r.source_id))
    md_path, csv_path = Path(md_path), Path(csv_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['species', 'author', 'license', 'license_url', 'source', 'original_url', 'image_url', 'checksum'])
        for r in kept:
            w.writerow([r.species, r.author, r.license, r.license_url, r.source, r.original_url, r.image_url, r.checksum])
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# Attributions\n\n')
        f.write('Images used to train the recognition model. Each one is listed here with its '
                "author, its licence and its source, as the licence requires.\n\n")
        current = None
        for r in kept:
            if r.species != current:
                current = r.species
                f.write(f'\n## {current}\n\n')
            f.write(f'- {r.author or "author not recorded"} — [{r.license}]({r.license_url}) — [{r.source}]({r.original_url})\n')
    return len(kept)
