#!/usr/bin/env python3
"""The species to add to the catalogue, ranked by what people actually grow.

    python3 candidates.py --how-many 2000 --out candidates-v8.txt

The rule is **select, do not draw**: a random draw from the extended catalogue
would give only 40 % usable classes, and the confusion breakdown adds that the
model's weakest species are already trees and wild plants — 62 of the 63
listed. Growing that population would cost ten points to whoever photographs
their living room.

What remains is to know what "cultivated" means to a database.

**GBIF is not the one that answers.** It has the right field —
`degreeOfEstablishment=cultivated` — and nobody fills it in: 288 occurrences
out of the 76 million photographed plants, four parts per million. Measured,
not assumed.

**iNaturalist is.** Its "captive/cultivated" flag is set by the observers
themselves and massively used, and the API returns the ranking directly:
`/observations/species_counts` with `captive=true` gives 58,000 species sorted
by number of observations. At the top come hibiscus, oleander, Japanese maple,
crape myrtle, aloe — that is, the aisles of a garden centre, not a field flora.

That ranking says **what we want to collect**. It does not say what we *can*:
iNaturalist's images are not all freely licensed. `availability.py` then
checks, species by species, that there is enough to train a class.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plant_dataset.taxonomy import internal_id, normalize_scientific_name  # noqa: E402

#: The ranks that make a class. A genus or a family does not make one, and a
#: subspecies would be confused with the species in the same image.
RANKS = ('species', 'hybrid')

#: The cap of one API page.
PAGE = 500


def keys_of(internal: str) -> set[str]:
    """The spellings under which the same plant can present itself.

    iNaturalist writes sometimes "Citrus × limon", sometimes "Citrus limon",
    and the two give different identifiers — `citrus-x-limon` and
    `citrus-limon`. Letting both through means collecting the same plant twice
    and manufacturing the very defect that duplicates.py exists to find: two
    classes for one species, the images shared, an unwinnable confusion.
    """
    parts = internal.split('-')
    without_x = '-'.join(p for p in parts if p != 'x')
    genus, _, epithet = without_x.partition('-')
    with_x = f'{genus}-x-{epithet}' if epithet else without_x
    return {internal, without_x, with_x}


def keep(results: list[dict], known: set[str], ranks: tuple[str, ...] = RANKS) -> list[tuple[str, int]]:
    """The genuinely new candidates, in the order they come.

    `results` is the shape iNaturalist returns: `{'count': n, 'taxon':
    {'name': ..., 'rank': ...}}`. `known` holds the internal identifiers
    already in the catalogue — the same key as everywhere else.
    """
    seen: set[str] = set()
    out: list[tuple[str, int]] = []
    for r in results:
        taxon = r.get('taxon') or {}
        if taxon.get('rank') not in ranks:
            continue
        name = normalize_scientific_name(taxon.get('name') or '')
        if not name:
            continue
        variants = keys_of(internal_id(name))
        if variants & known or variants & seen:
            continue
        seen |= variants
        out.append((name, int(r.get('count', 0))))
    return out


def register(existing: list, candidates: list[tuple[str, int]]) -> list:
    """The `plants.csv` rows to add for these candidates.

    This is the piece that was missing between selection and collection:
    `build_dataset.py --only-file` only **filters** the plants already present
    in `plants.csv`. A candidate absent from the catalogue is not collected —
    it is ignored in silence, which only shows after the collection, in a
    shorter count than expected.

    The rows created carry only what the name gives: internal identifier,
    genus, epithet. The family, the GBIF key and the Wikidata identifier come
    afterwards from `enrich_plants.py --gbif --wikidata`, and the common names
    from the application's catalogue.
    """
    from plant_dataset.taxonomy import PlantEntry
    known = {e.internal_id for e in existing}
    additions = []
    for name, _ in candidates:
        entry = PlantEntry.from_name(name)
        if entry.internal_id in known:
            continue
        known.add(entry.internal_id)
        additions.append(entry)
    return additions


def already_in_catalogue(plants: Path, labels: Path | None = None) -> set[str]:
    """What we already have: the collection catalogue, and the shipped classes.

    Both, because they do not coincide — a species can be in the catalogue and
    have been dropped from the model for lack of images. Proposing that one
    again would be right: it is a collection to redo, not a species to
    discover. But it must not occupy a slot meant for novelty.
    """
    import csv
    known: set[str] = set()
    if plants.exists():
        known |= {r['internal_id'] for r in csv.DictReader(plants.open(encoding='utf-8'))}
    if labels and labels.exists():
        known |= {l.strip() for l in labels.read_text(encoding='utf-8').splitlines() if l.strip()}
    return known


def _pages(client, how_many: int, extra: dict) -> list[dict]:
    """Enough pages for `how_many` candidates, never more."""
    results: list[dict] = []
    page = 1
    while len(results) < how_many:
        d = client._get('/observations/species_counts', per_page=PAGE, page=page, **extra)
        batch = d.get('results') or []
        if not batch:
            break
        results.extend(batch)
        print(f'  page {page}: {len(results)} taxa', file=sys.stderr, flush=True)
        page += 1
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--how-many', type=int, default=2000, help='new candidates to return')
    ap.add_argument('--plants', default='plants.csv')
    ap.add_argument('--labels', default='../training/out/labels.txt')
    ap.add_argument('--place', type=int, help='restrict to an iNaturalist place (place_id), e.g. Europe')
    ap.add_argument('--pause', type=float, default=1.0, help='iNaturalist pace; below it, 429')
    ap.add_argument('--out', help='one name per line, ready for availability.py')
    ap.add_argument('--csv', help='with the number of cultivated observations')
    ap.add_argument('--register', action='store_true',
                    help='add the candidates to plants.csv — without which the collection ignores them')
    args = ap.parse_args(argv)

    from plant_dataset.fetchers.inaturalist import InatClient   # network: not at module import

    known = already_in_catalogue(Path(args.plants), Path(args.labels))
    print(f'{len(known)} plants already known to the catalogue or the model\n', file=sys.stderr)

    extra = {'iconic_taxa': 'Plantae', 'captive': 'true', 'photos': 'true'}
    if args.place:
        extra['place_id'] = args.place
    # Ask more broadly than `--how-many`: a good share of the taxa returned are
    # already in the catalogue, or of a rank that does not make a class.
    raw = _pages(InatClient(pause=args.pause), args.how_many * 2, extra)
    candidates = keep(raw, known)[:args.how_many]

    print(f'\n{len(raw)} taxa walked through, {len(candidates)} new candidates')
    if candidates:
        print(f"cultivated observations: from {candidates[0][1]} for the first "
              f"to {candidates[-1][1]} for the last\n")
        for name, n in candidates[:15]:
            print(f'  {n:7d}  {name}')
        if len(candidates) > 15:
            print(f'  … and {len(candidates) - 15} more')

    if args.out:
        Path(args.out).write_text('\n'.join(n for n, _ in candidates) + '\n', encoding='utf-8')
        print(f'\n{len(candidates)} names written to {args.out}')
        print(f'→ check what can really be collected:\n'
              f'   python3 availability.py --species-file {args.out} --csv availability.csv')
    if args.csv:
        import csv as _csv
        with open(args.csv, 'w', newline='', encoding='utf-8') as f:
            w = _csv.writer(f)
            w.writerow(['species', 'cultivated_observations'])
            w.writerows(candidates)

    if args.register:
        from plant_dataset.taxonomy import load_plants, save_plants
        existing = load_plants(args.plants)
        additions = register(existing, candidates)
        save_plants(args.plants, existing + additions)
        print(f'\n{len(additions)} rows added to {args.plants} '
              f'({len(existing)} → {len(existing) + len(additions)})')
        print('→ fill in family, GBIF key and Wikidata:\n'
              '   python3 enrich_plants.py --gbif --wikidata')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
