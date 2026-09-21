#!/usr/bin/env python3
"""How many images a species has, before deciding to collect it.

    python3 availability.py --species-file candidates.txt --threshold 25

Growing the catalogue costs top-1: 78 classes returned 63.8 %, 542 returned
44.2 %. That is not a fatality — one version added 551 species without losing
anything — but it is **if the added species have no images**. A class below 25
training images is dropped by `--min-train`; between 25 and 50, it enters the
model without learning anything there and dilutes the measurement.

Hence this tool, to be run **before** a collection: it downloads no image, it
asks GBIF how many photographed occurrences exist per species. A few hours of
queries against a twelve-hour collection.

**It is a lower bound, and that must be known.** GBIF can only filter CC0 and
CC BY in the query (`licenses.py`): share-alike, which the project accepts,
only shows on the medium. And iNaturalist directly, which brings the
cultivated plants, is not queried at all. A species announced at 40 here will
often have more; a species announced at 3 will never have 25.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plant_dataset.fetchers.gbif import GBIF_LICENSE_CODES  # noqa: E402

#: The `train.py --min-train` threshold: below it, the class is dropped.
THRESHOLD = 25


def verdict(counts: dict[str, int], threshold: int = THRESHOLD) -> dict:
    """Sort the candidate species by what can be expected of them.

    Four populations, because they call for four decisions:

    - **unknown**: GBIF does not know the name. `synonyms.txt` answers that,
      not a collection run.
    - **empty**: the name resolves but there is no photographed occurrence. The
      collection will cross them for nothing.
    - **thin**: below the threshold. They would enter the catalogue and leave
      the model — the worst of both worlds, a collection cost with no class at
      the end.
    - **solid**: enough to learn from. It is the only number that counts when
      announcing "the model grows to N species".
    """
    unknown = sorted(s for s, n in counts.items() if n < 0)
    empty = sorted(s for s, n in counts.items() if n == 0)
    thin = sorted((s for s, n in counts.items() if 0 < n < threshold), key=lambda s: -counts[s])
    solid = sorted((s for s, n in counts.items() if n >= threshold), key=lambda s: -counts[s])
    values = [counts[s] for s in solid]
    return {
        'requested': len(counts),
        'unknown': unknown,
        'empty': empty,
        'thin': thin,
        'solid': solid,
        'median': statistics.median(values) if values else 0,
        'counts': counts,
    }


def available(client, name: str, codes: list[str] = GBIF_LICENSE_CODES) -> int:
    """Photographed, freely reusable occurrences, without reading anything.

    `limit=0`: GBIF returns the count and no result. One request per licence,
    because the parameter accepts a single value.

    Returns -1 when the name does not resolve — to be distinguished from zero,
    which means "known, but never photographed".
    """
    taxon = client.match(name)
    if taxon is None or not taxon.key:
        return -1
    total = 0
    for code in codes:
        d = client._get('/occurrence/search', taxonKey=taxon.key, mediaType='StillImage',
                        basisOfRecord='HUMAN_OBSERVATION', license=code, limit=0)
        total += int(d.get('count', 0))
    return total


def _report(r: dict, threshold: int, detail: int) -> None:
    n = r['requested']
    print(f'\n{n} species queried:')
    print(f"  {len(r['solid']):5d} above {threshold} occurrences  ← the real classes")
    print(f"  {len(r['thin']):5d} between 1 and {threshold - 1}  ← collected for nothing, dropped from the model")
    print(f"  {len(r['empty']):5d} known to GBIF, never photographed")
    print(f"  {len(r['unknown']):5d} name unresolved — `synonyms.txt` answers that")
    if r['solid']:
        print(f"\nmedian of the solid ones: {r['median']:.0f} occurrences")
        print(f"→ a catalogue of {n} names would give about **{len(r['solid'])} classes**")
    if r['thin'][:detail]:
        print('\nthe closest to the threshold, worth watching:')
        for s in r['thin'][:detail]:
            print(f'  {s:38s} {r["counts"][s]:4d}')


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--species-file', required=True, help='one scientific name per line')
    ap.add_argument('--threshold', type=int, default=THRESHOLD, help="the training run's `--min-train`")
    ap.add_argument('--limit', type=int, help='measure only the first N')
    ap.add_argument('--pause', type=float, default=0.25, help='GBIF pace, in seconds')
    ap.add_argument('--detail', type=int, default=15)
    ap.add_argument('--csv', help='write the count per species')
    args = ap.parse_args(argv)

    from plant_dataset.fetchers.gbif import GbifClient   # network: not at module import

    species = [l.strip() for l in Path(args.species_file).read_text(encoding='utf-8').splitlines() if l.strip()]
    if args.limit:
        species = species[:args.limit]
    client = GbifClient(pause=args.pause)

    counts: dict[str, int] = {}
    for i, one in enumerate(species, 1):
        try:
            counts[one] = available(client, one)
        except Exception as e:      # a source falling over must never read as a zero
            print(f'  [{i}/{len(species)}] {one}: FAILED ({type(e).__name__}: {e})', file=sys.stderr)
            continue
        if i % 25 == 0 or i == len(species):
            print(f'  [{i}/{len(species)}]', flush=True)

    r = verdict(counts, args.threshold)
    _report(r, args.threshold, args.detail)
    if args.csv:
        import csv as _csv
        with open(args.csv, 'w', newline='', encoding='utf-8') as f:
            w = _csv.writer(f)
            w.writerow(['species', 'occurrences'])
            for s in sorted(counts, key=lambda x: -counts[x]):
                w.writerow([s, counts[s]])
        print(f'\ncount written to {args.csv}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
