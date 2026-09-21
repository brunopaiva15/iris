#!/usr/bin/env python3
"""Merge collections run in parallel over disjoint shards.

    python3 merge_shards.py --out dataset shard0 shard1 shard2
    python3 build_dataset.py --out dataset --skip-fetch --plants plants.csv

Collecting 1,558 species in a row occupies one core and leaves the network
waiting. Cut into disjoint shards, the same work fits on the machine's four
cores. The shards do not overlap — a species is in one only — so merging is a
move of folders and a concatenation of manifests.

Deduplication, splitting and statistics are *not* redone here: they bear on
the whole set and `build_dataset.py --skip-fetch` takes care of them, on the
merged set. That matters for the cross-species duplicates, which no shard can
see on its own.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Written by the finalisation of each shard; redone over the whole set.
PER_RUN = {'manifest.jsonl', 'splits.csv', 'stats.json', 'ATTRIBUTIONS.md', 'attributions.csv'}
CACHES = ('species.json', 'species_inat.json')


def move_into(source: Path, target: Path) -> int:
    """Move `source` onto `target`, merging what exists on both sides.
    Returns the number of files moved.

    Two folders carry the same name across shards, for two different reasons.
    The status folders (`_rejected`, `_duplicates`, `_review`) exist in each
    one. And two catalogue names can give the same species folder: `Citrus ×
    sinensis` appeared there twice, and the sharding sent the two copies to
    different shards. In both cases, overwriting the destination would lose
    images; so the tree is walked down instead.

    When two files carry the same name, they are the same image: the name is
    the start of its SHA-256. Overwriting is then a no-op.
    """
    if source.is_dir() and target.is_dir():
        moved = 0
        for child in sorted(source.iterdir()):
            moved += move_into(child, target / child.name)
        source.rmdir()
        return moved
    target.parent.mkdir(parents=True, exist_ok=True)
    source.replace(target)
    return 1


def merge_caches(shards: list[Path], out: Path) -> dict[str, int]:
    """Name resolutions cost hours of network. Each shard resolved its own;
    the merged set keeps them all."""
    sizes = {}
    for name in CACHES:
        merged: dict = {}
        for shard in shards:
            path = shard / name
            if path.exists():
                merged.update(json.loads(path.read_text()))
        (out / name).write_text(json.dumps(merged, ensure_ascii=False, indent=1))
        sizes[name] = len(merged)
    return sizes


def merge(shards: list[Path], out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    lines, moved = 0, 0
    with open(out / 'manifest.jsonl', 'a', encoding='utf-8') as manifest:
        for shard in shards:
            source = shard / 'manifest.jsonl'
            if not source.exists():
                raise SystemExit(f'{shard}: no manifest, the shard produced nothing')
            for line in source.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    manifest.write(line + '\n')
                    lines += 1
            for entry in sorted(shard.iterdir()):
                if entry.name in PER_RUN or entry.name in CACHES:
                    continue
                moved += move_into(entry, out / entry.name)
    return {'lines': lines, 'folders': moved}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('shards', nargs='+', help='folders of the shards to merge')
    ap.add_argument('--out', default='dataset', help='folder of the merged set')
    args = ap.parse_args()

    shards = [Path(s) for s in args.shards]
    out = Path(args.out)
    for shard in shards:
        if not shard.is_dir():
            raise SystemExit(f'{shard}: folder not found')
        if shard.resolve() == out.resolve():
            raise SystemExit('a shard cannot be the merged set')

    counts = merge(shards, out)
    caches = merge_caches(shards, out)
    print(f'{counts["lines"]} manifest lines, {counts["folders"]} folders moved to {out}')
    print('  ' + ', '.join(f'{k}: {v} names' for k, v in caches.items()))
    print(f'\nNext, over the whole set:\n'
          f'  python3 build_dataset.py --out {out} --plants plants.csv --skip-fetch')
    return 0


if __name__ == '__main__':
    sys.exit(main())
