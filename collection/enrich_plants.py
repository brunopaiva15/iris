#!/usr/bin/env python3
"""Complete `plants.csv` with the external identifiers.

- GBIF: the key of the accepted taxon, and the family according to the GBIF
  backbone; the names GBIF does not recognise clearly are listed for review.
- Wikidata: the Q identifier of the item whose scientific name (P225) is ours,
  in a single SPARQL query.

    python3 enrich_plants.py [--gbif] [--wikidata]     (both by default)
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.parse

import requests

from plant_dataset.fetchers.gbif import GbifClient
from plant_dataset.taxonomy import load_plants, save_plants

PLANTS = 'plants.csv'
WIKIDATA = 'https://query.wikidata.org/sparql'


def enrich_gbif(entries, plants_path=None, every: int = 200) -> list[str]:
    """Resolve what is not resolved yet — and **save along the way**.

    A resume asks for nothing again: the entries already resolved are skipped.
    They still have to have been written: over 4,220 new species the pass takes
    an hour and a half, and saving only at the end lost the whole hour to a
    network outage or a Ctrl-C.
    """
    client = GbifClient()
    doubtful = []
    resolved = 0
    for i, e in enumerate(entries, 1):
        if e.gbif_key:
            continue
        m = next((m for m in (client.match(n) for n in [e.scientific_name, *e.synonyms]) if m is not None and m.usable), None)
        if m is None:
            doubtful.append(f'{e.scientific_name} → no clear species, synonyms included')
            continue
        e.gbif_key = m.accepted_key or m.key
        if not e.family and m.family:
            e.family = m.family
        resolved += 1
        if resolved % every == 0 and plants_path:
            save_plants(plants_path, entries)
        if i % 25 == 0:
            print(f'  gbif {i}/{len(entries)}', file=sys.stderr, flush=True)
    return doubtful


def enrich_wikidata(entries) -> None:
    todo = [e for e in entries if not e.wikidata_id]
    for start in range(0, len(todo), 150):
        chunk = todo[start:start + 150]
        values = ' '.join('"%s"' % e.scientific_name.replace('"', '') for e in chunk)
        query = f'SELECT ?name ?item WHERE {{ VALUES ?name {{ {values} }} ?item wdt:P225 ?name . }}'
        r = requests.get(WIKIDATA, params={'query': query, 'format': 'json'},
                         headers={'User-Agent': 'IrisPlantDataset/0.1 (github.com/brunopaiva15/iris)'}, timeout=120)
        r.raise_for_status()
        found = {}
        for b in r.json()['results']['bindings']:
            found.setdefault(b['name']['value'], b['item']['value'].rsplit('/', 1)[-1])
        for e in chunk:
            e.wikidata_id = found.get(e.scientific_name)
        time.sleep(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--gbif', action='store_true')
    ap.add_argument('--wikidata', action='store_true')
    ap.add_argument('--plants', default=PLANTS)
    args = ap.parse_args()
    do_all = not (args.gbif or args.wikidata)
    entries = load_plants(args.plants)
    if args.gbif or do_all:
        doubtful = enrich_gbif(entries, args.plants)
        save_plants(args.plants, entries)
        print(f'gbif: {sum(1 for e in entries if e.gbif_key)}/{len(entries)} resolved')
        for d in doubtful:
            print('  to review:', d)
    if args.wikidata or do_all:
        enrich_wikidata(entries)
        save_plants(args.plants, entries)
        print(f'wikidata: {sum(1 for e in entries if e.wikidata_id)}/{len(entries)} found')
    return 0


if __name__ == '__main__':
    sys.exit(main())
