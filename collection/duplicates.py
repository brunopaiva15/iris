#!/usr/bin/env python3
"""The model classes that are the same plant under two names.

    python3 duplicates.py --labels ./labels.txt

*Sansevieria trifasciata* and *Dracaena trifasciata* are one plant: the
snake plant changed genus in 2017. Both names are in the catalogue, so the
model has **two classes** for it. That costs three times over:

- the images are **shared** between the two classes, and each trains on half
  of what it should;
- the confusion is **unwinnable** — no photograph can settle it, since there
  is nothing to settle. Yet it shows in the confusion breakdown as a defect of
  the model;
- and the announced class count is wrong by as much.

The first reflex — looking for identifiers sharing their epithet within one
family — is not enough: on one catalogue it returns 34 groups of which **4**
are real duplicates. *Populus alba* and *Salix alba* share their epithet and
are not the same plant; `officinalis`, `vulgaris` and `japonica` are
all-purpose epithets.

Hence the choice of asking GBIF everything, with no heuristic: every catalogue
name is resolved to its **accepted taxon**, and two names landing on the same
key are the same plant. It is already the arbiter of the collection
(`fetchers/gbif.py`) and of `synonyms.txt`; there is no reason to take another
one here.

**Its limit, to know before concluding.** GBIF follows its own taxonomic
backbone and it lags on some recent transfers: *Schefflera arboricola* and
*Heptapleurum arboricola* are two accepted taxa there, so this tool does not
report them, although the recent literature holds them to be one plant. The
tool returns the **certain** duplicates, not every duplicate.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def group(keys: dict[str, int | None]) -> list[list[str]]:
    """The identifiers landing on the same accepted taxon.

    A missing key (`None`) means "unresolved": those group with nobody, or
    every unknown would become one and the same duplicate.
    """
    per_key: dict[int, list[str]] = defaultdict(list)
    for internal, key in keys.items():
        if key is not None:
            per_key[key].append(internal)
    return sorted((sorted(v) for v in per_key.values() if len(v) > 1), key=lambda g: g[0])


def images_per_class(dataset: Path) -> dict[str, int]:
    """How many images each class holds — to say what the split really costs,
    rather than announcing it in principle."""
    splits = dataset / 'splits.csv'
    if not splits.exists():
        return {}
    count: dict[str, int] = defaultdict(int)
    with splits.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            count[row['internal_plant_id']] += 1
    return dict(count)


def accepted_key(client, name: str) -> int | None:
    """The accepted taxon of a name, or `None` if GBIF does not resolve it.

    `acceptedUsageKey` is only filled in for a synonym; for an accepted name it
    is `usageKey` that counts. Taking one or the other at random would file
    every synonym apart from its accepted name, which is exactly what this is
    meant to spot.
    """
    d = client._get('/species/match', name=name, kingdom='Plantae')
    key = d.get('acceptedUsageKey') or d.get('usageKey')
    return int(key) if key else None


def _report(groups: list[list[str]], names: dict[str, str], images: dict[str, int], total: int) -> None:
    if not groups:
        print('no duplicate: every class is a distinct plant.')
        return
    lost = sum(len(g) - 1 for g in groups)
    print(f'**{len(groups)} plants counted twice** — {lost} classes too many out of {total}.\n')
    for g in groups:
        print('  ' + ' = '.join(names.get(i, i) for i in g))
        if images:
            parts = ' + '.join(f'{images.get(i, 0)}' for i in g)
            total_images = sum(images.get(i, 0) for i in g)
            print(f'    {parts} = {total_images} images, separated today')
    print('\nTo merge in `plants.csv`: keep the accepted name, make the other a')
    print('synonym. The images rejoin, the confusion disappears — it was not an')
    print('error of the model, there was nothing to settle.')


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--labels', default='../training/out/labels.txt')
    ap.add_argument('--plants', default='plants.csv')
    ap.add_argument('--dataset', default='dataset', help='to say how many images each split separates')
    ap.add_argument('--cache', default='.cache/duplicates.json', help='keys already resolved; a resume asks for nothing again')
    ap.add_argument('--pause', type=float, default=0.15)
    ap.add_argument('--limit', type=int)
    args = ap.parse_args(argv)

    labels = [l.strip() for l in Path(args.labels).read_text(encoding='utf-8').splitlines() if l.strip()]
    if args.limit:
        labels = labels[:args.limit]
    plants = Path(args.plants)
    names = ({r['internal_id']: r['scientific_name'] for r in csv.DictReader(plants.open(encoding='utf-8'))}
             if plants.exists() else {})

    cache = Path(args.cache)
    keys: dict[str, int | None] = json.loads(cache.read_text()) if cache.exists() else {}
    missing = [i for i in labels if i not in keys]
    if missing:
        from plant_dataset.fetchers.gbif import GbifClient   # network: not at module import
        client = GbifClient(pause=args.pause)
        print(f'{len(missing)} names to resolve at GBIF…', file=sys.stderr, flush=True)
        for n, internal in enumerate(missing, 1):
            try:
                keys[internal] = accepted_key(client, names.get(internal, internal.replace('-', ' ')))
            except Exception as e:      # a source falling over must not pass for "unresolved"
                print(f'  {internal}: FAILED ({type(e).__name__}: {e})', file=sys.stderr)
            if n % 100 == 0:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(keys))
                print(f'  {n}/{len(missing)}', file=sys.stderr, flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(keys))

    unresolved = [i for i in labels if keys.get(i) is None]
    if unresolved:
        print(f'{len(unresolved)} unresolved names, ignored: {", ".join(unresolved[:8])}'
              f'{" …" if len(unresolved) > 8 else ""}\n')
    _report(group({i: keys.get(i) for i in labels}), names,
            images_per_class(Path(args.dataset)), len(labels))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
