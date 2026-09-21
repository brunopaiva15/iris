#!/usr/bin/env python3
"""What Wikimedia Commons would add, species by species, before collecting.

    python3 commons_gain.py --limit 30
    python3 commons_gain.py --species-file phase1_species.txt --dataset dataset

The connector exists: 97 % usable licences against 18 % on GBIF, and above all
plants photographed **in people's homes**. It has never served a full
collection, and what it brings must be measured *before* it goes into the
recipe.

It is the same principle that, elsewhere, saved writing a Smithsonian
connector for seven photographs after ten minutes of measurement: **a source
is not worth its size but its overlap with the catalogue.**

The tool downloads no image. It counts, per species, the files the connector
would keep — photographs, accepted licence — and sets them against what the
set already holds. A species with 200 images that Commons offers 12 more does
not justify a pass; a species with 30 images that it offers 150 justifies one
on its own.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def summarise(offered: dict[str, int], already: dict[str, int] | None = None,
              thin_threshold: int = 100) -> dict:
    """The report, from the counts — no network, hence testable.

    `offered`: species → files Commons would keep.
    `already`: species → images already in the set, when it is there.

    Three populations are worth distinguishing, because they call for three
    different decisions: the ones Commons ignores (nothing to do), the ones it
    barely knows (they do not justify the pass on their own), and the ones it
    could more than double — those are what we are looking for.
    """
    already = already or {}
    unknown = sorted(s for s, n in offered.items() if n == 0)
    known = {s: n for s, n in offered.items() if n > 0}
    values = sorted(known.values())
    doubled = sorted((s for s, n in known.items() if already.get(s) and n >= already[s]),
                     key=lambda s: -known[s])
    return {
        'species': len(offered),
        'unknown': unknown,
        'known': len(known),
        'total': sum(values),
        'median': statistics.median(values) if values else 0,
        'thin': sorted(s for s, n in known.items() if n < thin_threshold),
        'doubled': doubled,
        'offered': offered,
        'already': already,
    }


def images_in_set(dataset: Path) -> dict[str, int]:
    """What the set already holds, per scientific name."""
    splits = dataset / 'splits.csv'
    if not splits.exists():
        return {}
    count: Counter = Counter()
    with splits.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            count[row['species']] += 1
    return dict(count)


def _report(r: dict, detail: int) -> None:
    print(f"\n{r['species']} species queried, {r['known']} known to Commons")
    if r['unknown']:
        print(f"  {len(r['unknown'])} without a category: {', '.join(r['unknown'][:6])}"
              f"{' …' if len(r['unknown']) > 6 else ''}")
    if not r['known']:
        return
    print(f"  {r['total']} usable photographs in total, median {r['median']:.0f} per species")
    print(f"  {len(r['thin'])} species under 100 photographs — they do not justify the pass on their own")
    if r['already']:
        print(f"  **{len(r['doubled'])} species Commons could at least double**")

    print(f"\nthe {detail} species that would gain the most:")
    for species in sorted(r['offered'], key=lambda s: -r['offered'][s])[:detail]:
        how_many = r['offered'][species]
        if not how_many:
            continue
        before = r['already'].get(species)
        context = f"  (the set has {before} → x{(before + how_many) / before:.1f})" if before else ''
        print(f'  {species:34s} {how_many:4d} photographs{context}')


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--species-file', default='phase1_species.txt',
                    help='one scientific name per line; by default the indoor plants')
    ap.add_argument('--dataset', default='dataset', help='to compare with what the set already has')
    ap.add_argument('--limit', type=int, help='measure only the first N, for an order of magnitude')
    ap.add_argument('--max-files', type=int, default=300, help='cap per species, as at collection time')
    ap.add_argument('--pause', type=float, default=1.0,
                    help='Commons pace; below one second the API answers 429')
    ap.add_argument('--no-sa', action='store_true', help='refuse CC BY-SA (accepted by default)')
    ap.add_argument('--detail', type=int, default=15)
    ap.add_argument('--csv', help='write the count per species')
    args = ap.parse_args(argv)

    from plant_dataset.fetchers.wikimedia import CommonsClient  # network: not at module import

    species = [l.strip() for l in Path(args.species_file).read_text(encoding='utf-8').splitlines() if l.strip()]
    if args.limit:
        species = species[:args.limit]
    client = CommonsClient(pause=args.pause)

    offered: dict[str, int] = {}
    for i, one in enumerate(species, 1):
        try:
            offered[one] = sum(1 for _ in client.image_candidates(
                one, max_files=args.max_files, allow_share_alike=not args.no_sa))
        except Exception as e:      # a source falling over must never read as a zero
            print(f'  [{i}/{len(species)}] {one}: FAILED ({type(e).__name__}: {e})', file=sys.stderr)
            continue
        print(f'  [{i}/{len(species)}] {one:34s} {offered[one]:4d}', flush=True)

    report = summarise(offered, images_in_set(Path(args.dataset)))
    _report(report, args.detail)
    if args.csv:
        with open(args.csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['species', 'commons', 'already_in_the_set'])
            for s in sorted(offered, key=lambda x: -offered[x]):
                w.writerow([s, offered[s], report['already'].get(s, '')])
        print(f'\ncount written to {args.csv}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
