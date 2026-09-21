#!/usr/bin/env python3
"""The known cultivars of each catalogue species, from Wikidata.

    python3 cultivars.py --out cultivars.csv

**This is not a training set, and that is the point.** A cultivar is a plant
horticulture has selected: *Monstera deliciosa* "Thai Constellation" is a
variegated *Monstera deliciosa*, not another species. People own many of them,
and they want the name.

But they cannot be made into classes, for a measured reason:

- **iNaturalist does not model cultivars.** Among the 2,000 most observed
  cultivated taxa: 1,935 species, 65 hybrids, zero cultivars. A photograph of
  "Thai Constellation" is recorded there as *Monstera deliciosa*;
- **Commons has the shelves, and they are empty.** The hierarchy exists —
  `Monstera deliciosa (cultivars)`, with one subcategory per named cultivar —
  but it holds **1** file for the Monstera, **3** for *Epipremnum aureum*
  "Marble Queen", **6** for "N'Joy". The entry threshold of a class is 25
  images;
- and hundreds of visually near-identical classes would recreate at scale the
  duplicate-class defect: the images of one plant shared between two labels, a
  confusion no photograph can settle.

The cultivar is therefore a **second axis**, not one more class: the model
answers the species, and the application offers the known cultivars of that
species. This file is that list — a catalogue, not pixels.

Wikidata gives it: a cultivar there is an instance of `Q4886`, attached to its
parent taxon by `P171`.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

WIKIDATA = 'https://query.wikidata.org/sparql'
UA = 'IrisPlantDataset/0.1 (github.com/brunopaiva15/iris; cultivar catalogue)'

#: Wikidata writes the cultivar epithet with every apostrophe Unicode has to
#: offer: "Hosta ʽFortunei’", "Hosta 'Gold Standard'". The horticultural rule
#: knows only one, the straight apostrophe.
_APOSTROPHES = re.compile('[‘’ʻʼʽ′`´]')


#: The botanical ranks: what follows them is a variety or a subspecies, not a
#: cultivar.
_RANK = re.compile(r'^(var|subsp|ssp|f|forma|subvar|cv)\b\.?', re.I)


def epithet(name: str, parent: str) -> str | None:
    """The cultivar epithet alone: "Ficus elastica ʽRobusta’" → "Robusta".

    Two spellings, and the second calls for caution. The horticultural code
    puts the epithet between single quotes, and that is clear-cut. When
    Wikidata omits them, guessing is all that is left — and `Q4886` also
    carries taxa that are not cultivars: *Hosta decorata* is a species, "Agave
    americana var. medio-picta alba" a variety. Letting them through would fill
    the cultivar list with things that are not cultivars.

    Hence the horticultural code's rule as a safeguard: **a cultivar name takes
    a capital**, a species or variety epithet never does.
    """
    clean = _APOSTROPHES.sub("'", name or '').strip()
    m = re.search(r"'([^']+)'", clean)
    if m:
        return m.group(1).strip() or None
    if not parent or not clean.lower().startswith(parent.lower()):
        return None
    rest = clean[len(parent):].strip()
    if not rest or _RANK.match(rest) or not rest[0].isupper():
        return None
    return rest


def group(rows: list[tuple[str, str]]) -> dict[str, list[str]]:
    """species → its cultivars, deduplicated and sorted.

    `rows` is a list of (parent name, cultivar name).
    """
    out: dict[str, set] = defaultdict(set)
    for parent, name in rows:
        e = epithet(name, parent)
        if e:
            out[parent].add(e)
    return {k: sorted(v) for k, v in sorted(out.items())}


def _query(names: list[str], timeout: float) -> list[tuple[str, str]]:
    import requests
    values = ' '.join('"%s"' % n.replace('"', '') for n in names)
    query = (
        'SELECT ?parentName ?cultivarLabel WHERE { '
        f'VALUES ?parentName {{ {values} }} '
        '?parent wdt:P225 ?parentName . '
        '?cultivar wdt:P31 wd:Q4886 ; wdt:P171 ?parent . '
        'SERVICE wikibase:label { bd:serviceParam wikibase:language "fr,en". } }'
    )
    r = requests.get(WIKIDATA, params={'query': query, 'format': 'json'},
                     headers={'User-Agent': UA}, timeout=timeout)
    r.raise_for_status()
    return [(b['parentName']['value'], b['cultivarLabel']['value'])
            for b in r.json()['results']['bindings']]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plants', default='plants.csv')
    ap.add_argument('--out', default='cultivars.csv')
    ap.add_argument('--batch', type=int, default=120, help='species per SPARQL query')
    ap.add_argument('--timeout', type=float, default=120)
    ap.add_argument('--limit', type=int, help='process only the first N species')
    args = ap.parse_args(argv)

    species = [r['scientific_name'] for r in csv.DictReader(Path(args.plants).open(encoding='utf-8'))]
    if args.limit:
        species = species[:args.limit]

    rows: list[tuple[str, str]] = []
    for start in range(0, len(species), args.batch):
        batch = species[start:start + args.batch]
        try:
            rows += _query(batch, args.timeout)
        except Exception as e:      # a query falling over must not pass for "no cultivar"
            print(f'  [{start}/{len(species)}] FAILED ({type(e).__name__}: {e})', file=sys.stderr)
            continue
        print(f'  [{min(start + args.batch, len(species))}/{len(species)}] {len(rows)} cultivars',
              file=sys.stderr, flush=True)

    per_species = group(rows)
    total = sum(len(v) for v in per_species.values())
    print(f'\n{total} cultivars over {len(per_species)} species '
          f'({len(per_species) / len(species):.0%} of the catalogue has at least one)')
    for name, listing in list(per_species.items())[:10]:
        print(f'  {name:30s} {", ".join(listing[:6])}{" …" if len(listing) > 6 else ""}')

    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['species', 'cultivars'])
        for name, listing in per_species.items():
            w.writerow([name, '|'.join(listing)])
    print(f'\nwritten to {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
