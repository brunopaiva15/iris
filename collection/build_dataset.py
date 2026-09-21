#!/usr/bin/env python3
"""Build the image set, species by species.

    python3 build_dataset.py --plants plants.csv --target-per-species 200
    python3 build_dataset.py --plants plants.csv --only "Monstera deliciosa,Ficus elastica" --target-per-species 20

For every plant: name resolution at GBIF, collection of images under an
accepted licence, download, checking, recording in the manifest. Then, over
the whole set: deduplication, splitting, statistics, attributions.
Re-runnable: what is already in the manifest is not downloaded again.

Output:
    dataset/
      manifest.jsonl          one line per image, whatever its status
      splits.csv              train / val / test of the kept images
      stats.json              counts per species, licence, source
      ATTRIBUTIONS.md         authors and licences, to ship with the model
      attributions.csv
      species.json            what GBIF answered for each name
      <Genus_epithet>/        the kept images
      _rejected/  _duplicates/  _review/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from plant_dataset.fetchers.gbif import GbifClient
from plant_dataset.fetchers.inaturalist import InatClient
from plant_dataset.fetchers.wikimedia import CommonsClient
from plant_dataset.images import ImageRejected, download, prepare, store
from plant_dataset.licenses import GBIF_LICENSE_CODES, GBIF_LICENSE_CODES_WITH_SA, parse_license
from plant_dataset.manifest import (STATUS_DUPLICATE, STATUS_KEPT, STATUS_REJECTED, STATUS_REVIEW, ImageRecord, Manifest,
                                    now_iso, write_attributions)
from plant_dataset.dedup import flag_cross_species, mark_duplicates
from plant_dataset.splits import write_splits
from plant_dataset.taxonomy import PlantEntry, load_plants, species_slug

STATUS_DIR = {STATUS_REJECTED: '_rejected', STATUS_DUPLICATE: '_duplicates', STATUS_REVIEW: '_review'}


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _first_usable(match, plant: PlantEntry):
    """The canonical name first, then the catalogue's synonyms, until an
    answer at species rank. "Sorbus aria" only gives GBIF a family (the name
    is ambiguous); "Aria edulis", its accepted name, gives the species. That is
    what the synonyms in `plants.csv` are for."""
    for name in [plant.scientific_name, *plant.synonyms]:
        m = match(name)
        if m is not None and getattr(m, 'usable', True):
            return m
    return None


def resolve(client: GbifClient, plant: PlantEntry, cache: dict) -> dict | None:
    """The GBIF taxon of a plant, memoised in species.json.

    Failures are not memoised: a name that gave nothing yesterday can give
    something today, because the source changed or — more often — because our
    way of searching improved. A cache of failures makes the fixes invisible;
    one request per unresolved name per run is a negligible price.
    """
    if cache.get(plant.scientific_name) is not None:
        return cache[plant.scientific_name]
    m = _first_usable(client.match, plant)
    entry = None if m is None else {
        'key': m.accepted_key or m.key, 'matched_key': m.key, 'canonical': m.canonical_name, 'scientific': m.scientific_name,
        'rank': m.rank, 'status': m.status, 'match': m.match_type, 'confidence': m.confidence, 'family': m.family, 'usable': m.usable,
    }
    cache[plant.scientific_name] = entry
    return entry


def resolve_inat(client: InatClient, plant: PlantEntry, cache: dict) -> dict | None:
    """The iNaturalist taxon of a plant, memoised in species_inat.json.

    As for GBIF, failures are not memoised — that is precisely what had hidden
    the support for synonyms: the `null` values written before the fix
    prevented any retry.
    """
    if cache.get(plant.scientific_name) is not None:
        return cache[plant.scientific_name]
    t = _first_usable(client.match, plant)
    entry = None if t is None else {'id': t.id, 'name': t.name, 'rank': t.rank, 'observations': t.observations,
                                    'matched_as': t.matched_as, 'synonym': t.is_synonym}
    cache[plant.scientific_name] = entry
    return entry


def collect_species(candidates, http: requests.Session, plant: PlantEntry, manifest: Manifest, out: Path,
                    target: int, workers: int = 6) -> dict:
    """Collect one species from an iterator of candidates, whatever the
    source. Downloads run in small parallel batches (the network is the
    bottleneck); writing the manifest stays sequential, in the order of the
    candidates."""
    kept_before = sum(1 for r in manifest if r.species == plant.scientific_name and r.status == STATUS_KEPT)
    state = {'kept': kept_before, 'tried': 0, 'rejected': 0}
    slug = species_slug(plant.scientific_name)
    # The same iNaturalist photograph arrives through GBIF *and* through the
    # direct API: its photo identifier recognises it before any download.
    known_photos = {r.extra.get('photo_id') for r in manifest if r.extra.get('photo_id')}

    def fetch(cand):
        try:
            return cand, prepare(download(cand.image_url, http)), None
        except (ImageRejected, requests.RequestException) as e:
            return cand, None, e

    def handle(cand, prepared, error) -> None:
        state['tried'] += 1
        lic = parse_license(cand.license_raw)
        base = dict(
            species=plant.scientific_name, internal_plant_id=plant.internal_id, source=cand.source, source_id=cand.source_id,
            original_url=cand.original_url, image_url=cand.image_url, author=cand.author, license=lic.code if lic else '',
            license_url=lic.url if lic else '', downloaded_at=now_iso(), observation_id=cand.observation_id,
            publisher=cand.publisher, dataset_key=cand.dataset_key, extra=cand.extra or {},
        )
        if error is not None:
            state['rejected'] += 1
            manifest.append(ImageRecord(checksum='', status=STATUS_REJECTED, reason=str(error)[:200], **base))
            return
        existing = manifest.by_checksum(prepared.sha256)
        rel = f'{slug}/{prepared.sha256[:16]}.jpg'
        record = ImageRecord(checksum=prepared.sha256, path=rel, width=prepared.width, height=prepared.height, phash=prepared.phash, **base)
        if existing is not None:
            record.status = STATUS_DUPLICATE
            record.duplicate_of = existing.checksum
            record.reason = 'exact duplicate'
            record.path = f'{STATUS_DIR[STATUS_DUPLICATE]}/{rel}'
        else:
            state['kept'] += 1
        store(prepared, out / record.path)
        manifest.append(record)

    batch: list = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        def flush() -> None:
            for cand, prepared, error in pool.map(fetch, batch):
                if state['kept'] >= target:
                    break
                handle(cand, prepared, error)
            batch.clear()

        for cand in candidates:
            if state['kept'] >= target:
                break
            if manifest.has_source(cand.source, cand.source_id):
                continue
            photo_id = (cand.extra or {}).get('photo_id')
            if photo_id and photo_id in known_photos:
                continue
            if photo_id:
                known_photos.add(photo_id)
            batch.append(cand)
            if len(batch) >= max(1, workers) * 2:
                flush()
        if batch and state['kept'] < target:
            flush()
    return {'kept': state['kept'], 'new': state['kept'] - kept_before, 'tried': state['tried'], 'rejected': state['rejected']}


def relocate(manifest: Manifest, out: Path) -> None:
    """After the deduplication pass, every file goes to the folder of its
    status; the manifest follows."""
    for r in manifest:
        if not r.path or not r.checksum:
            continue
        wanted = r.path
        base = r.path.split('/', 1)[1] if r.path.split('/', 1)[0] in STATUS_DIR.values() else r.path
        if r.status == STATUS_KEPT:
            wanted = base
        else:
            wanted = f'{STATUS_DIR[r.status]}/{base}'
        if wanted != r.path:
            src, dst = out / r.path, out / wanted
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                src.replace(dst)
            r.path = wanted


def collect_one(i: int, plant: PlantEntry, client: GbifClient, inat, http: requests.Session, manifest: Manifest,
                out: Path, species_cache: dict, inat_cache: dict, species_path: Path, inat_path: Path,
                license_codes: list[str], args, t0: float, total: int, commons=None) -> None:
    """Collect one species, from name resolution to images.

    Four passes, in this order:

    0. **Cultivated plants of one region** (iNaturalist, `captive=true` and
       `place_id`), for the species of `--captive-file`, up to `--place-share`
       of the target **on top of** it. "Cultivated" alone was not enough:
       iNaturalist's cultivated yuccas are Californian garden trees, and the
       model still did not recognise a potted yucca. In Europe, the same
       species observed is a houseplant.
    1. **Cultivated plants** (iNaturalist, `captive=true`), for the species of
       the `--captive-file` list only, up to `--captive-share` of the target.
       An earlier model did not recognise a living-room yucca because its 120
       images of Yucca gigantea, all from GBIF, were wild trees in the scrub:
       GBIF only receives observations of wild plants, and the collector only
       asked iNaturalist as a complement, when GBIF was not enough. For indoor
       plants — the ones users photograph — the cultivated share is reserved
       before anything else.
    2. **GBIF**, up to the target.
    3. **iNaturalist, every observation**, if images are still missing.
    """
    taxon = resolve(client, plant, species_cache)
    species_path.write_text(json.dumps(species_cache, ensure_ascii=False, indent=1))
    if taxon is None or not taxon['usable']:
        log(f'[{i}/{total}] {plant.scientific_name}: name unresolved at GBIF ({taxon and taxon["match"]}), to review')
        return
    target = args.target_per_species
    sources: list[str] = []
    res = {'kept': 0, 'new': 0, 'tried': 0, 'rejected': 0}

    def run(candidates, goal: int) -> int:
        nonlocal res
        r = collect_species(candidates, http, plant, manifest, out, goal, workers=args.workers)
        res = {'kept': r['kept'], 'new': res['new'] + r['new'],
               'tried': res['tried'] + r['tried'], 'rejected': res['rejected'] + r['rejected']}
        return r['new']

    def inat_taxon():
        t = resolve_inat(inat, plant, inat_cache)
        inat_path.write_text(json.dumps(inat_cache, ensure_ascii=False, indent=1))
        return t

    taxon_inat = None
    if inat is not None and plant.scientific_name.lower() in args.captive_set:
        taxon_inat = inat_taxon()
        if taxon_inat is not None and args.captive_place is not None:
            quota = round(args.place_share * target)
            kept = [r for r in manifest if r.species == plant.scientific_name and r.status == STATUS_KEPT]
            have = sum(1 for r in kept if (r.extra or {}).get('place_id') == args.captive_place)
            if have < quota:
                # On top of the target: these images add to what is there,
                # they do not replace the wild plants.
                added = run(inat.image_candidates(taxon_inat['id'], max_observations=args.max_candidates,
                                                  allow_share_alike=args.allow_sa, captive=True, place_id=args.captive_place),
                            len(kept) + (quota - have))
                sources.append(f'inat cultivated, region {added}')
        if taxon_inat is not None:
            quota = round(args.captive_share * target)
            kept = [r for r in manifest if r.species == plant.scientific_name and r.status == STATUS_KEPT]
            have = sum(1 for r in kept if (r.extra or {}).get('captive'))
            if have < quota:
                # The target passed is a total; it is expressed from what is
                # already there so as to get exactly `quota - have` more
                # cultivated images, even on a resume.
                added = run(inat.image_candidates(taxon_inat['id'], max_observations=args.max_candidates,
                                                  allow_share_alike=args.allow_sa, captive=True),
                            len(kept) + (quota - have))
                sources.append(f'inat cultivated {added}')

    added = run(client.image_candidates(taxon['key'], license_codes, max_occurrences=args.max_candidates,
                                        allow_share_alike=args.allow_sa), target)
    sources.append(f'gbif {added}')

    if inat is not None and res['kept'] < target:
        if taxon_inat is None:
            taxon_inat = inat_taxon()
        if taxon_inat is not None:
            added = run(inat.image_candidates(taxon_inat['id'], max_observations=args.max_candidates,
                                              allow_share_alike=args.allow_sa), target)
            sources.append(f'inat {added}')

    if commons is not None and res['kept'] < target:
        # Commons last: it is a complement. GBIF and iNaturalist describe
        # observations, with one observation per split group; Commons gives
        # isolated files, often of cultivated plants, and that is precisely
        # what the model lacks on indoor photographs.
        added = run(commons.image_candidates(plant.scientific_name, max_files=args.max_candidates,
                                             allow_share_alike=args.allow_sa), target)
        sources.append(f'commons {added}')

    log(f'[{i}/{total}] {plant.scientific_name}: {res["kept"]} kept (+{res["new"]}, {" + ".join(sources)}), '
        f'{res["rejected"]} rejected out of {res["tried"]} tried, {time.time() - t0:.0f} s')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plants', default='plants.csv')
    ap.add_argument('--out', default='dataset')
    ap.add_argument('--target-per-species', type=int, default=200)
    ap.add_argument('--only', help='scientific names, comma separated')
    ap.add_argument('--limit-species', type=int, help='process only the first N plants')
    ap.add_argument('--allow-sa', action='store_true', help='also accept CC BY-SA (refused by default)')
    ap.add_argument('--max-candidates', type=int, default=1500, help='GBIF occurrences walked through at most, per licence')
    ap.add_argument('--skip-fetch', action='store_true', help='download nothing: deduplicate, split, count')
    ap.add_argument('--repair-splits', action=argparse.BooleanOptionalAction, default=True,
                    help='give a validation group, then a test one, to the species that have none; '
                         '--no-repair-splits returns the earlier split, to reproduce a previous model')
    ap.add_argument('--workers', type=int, default=6, help='parallel downloads')
    ap.add_argument('--only-file', help='file with one scientific name per line (like --only)')
    ap.add_argument('--no-inaturalist', action='store_true', help='do not complement with the iNaturalist API')
    ap.add_argument('--wikimedia', action='store_true',
                    help='complement with Wikimedia Commons: cultivated plants, photographed in homes')
    ap.add_argument('--commons-pause', type=float, default=1.0,
                    help='pause between Commons requests, in seconds; below 1 s the API answers 429')
    ap.add_argument('--inat-pause', type=float, default=1.0, help='pause between iNaturalist requests, in seconds')
    ap.add_argument('--gbif-pause', type=float, default=0.25, help='pause between GBIF requests, in seconds; raise it when several collections run in parallel')
    ap.add_argument('--captive-file', help='species (one per line) for which to reserve a share of cultivated plants')
    ap.add_argument('--captive-share', type=float, default=0.5, help='share of the target reserved for cultivated plants (iNaturalist captive=true)')
    ap.add_argument('--captive-place', type=int, help='iNaturalist place identifier (97391 = Europe): cultivated plants of that region, collected first')
    ap.add_argument('--place-share', type=float, default=0.25, help='share of the target added as cultivated plants from that region, on top of the target')
    args = ap.parse_args()
    args.captive_set = ({line.strip().lower() for line in Path(args.captive_file).read_text().splitlines() if line.strip()}
                        if args.captive_file else set())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plants = load_plants(args.plants)
    wanted = {n.strip().lower() for n in (args.only or '').split(',') if n.strip()}
    if args.only_file:
        wanted |= {line.strip().lower() for line in Path(args.only_file).read_text().splitlines() if line.strip()}
    if wanted:
        plants = [p for p in plants if p.scientific_name.lower() in wanted]
    if args.limit_species:
        plants = plants[:args.limit_species]
    if not plants:
        log('no plant to process')
        return 1

    manifest = Manifest(out / 'manifest.jsonl')
    species_path = out / 'species.json'
    species_cache = json.loads(species_path.read_text()) if species_path.exists() else {}
    license_codes = GBIF_LICENSE_CODES_WITH_SA if args.allow_sa else GBIF_LICENSE_CODES

    inat_path = out / 'species_inat.json'
    inat_cache = json.loads(inat_path.read_text()) if inat_path.exists() else {}

    if not args.skip_fetch:
        client = GbifClient(pause=args.gbif_pause)
        inat = None if args.no_inaturalist else InatClient(pause=args.inat_pause)
        commons = CommonsClient(pause=args.commons_pause) if args.wikimedia else None
        http = requests.Session()
        http.headers['User-Agent'] = client.session.headers['User-Agent']
        for i, plant in enumerate(plants, 1):
            t0 = time.time()
            try:
                collect_one(i, plant, client, inat, http, manifest, out, species_cache, inat_cache,
                            species_path, inat_path, license_codes, args, t0, len(plants), commons)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                # The network falls over, a source returns an absurd answer:
                # that is this species' problem, not the other three hundred's.
                # Note it and carry on; the resume will catch up on these.
                log(f'[{i}/{len(plants)}] {plant.scientific_name}: FAILED ({type(e).__name__}: {e}), species skipped')

    dups = mark_duplicates(manifest.records)
    flagged = flag_cross_species(manifest.records)
    relocate(manifest, out)
    manifest.rewrite()
    counts = write_splits(manifest.records, out / 'splits.csv', repair=args.repair_splits)
    stats = manifest.stats()
    stats['duplicates'] = dups
    stats['cross_species_review'] = flagged
    stats['splits'] = counts
    (out / 'stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    n = write_attributions(manifest.records, out / 'ATTRIBUTIONS.md', out / 'attributions.csv')

    log('')
    log(f'{stats["kept"]} images kept out of {stats["images"]}; exact duplicates {dups["exact"]}, near {dups["near"]}; '
        f'{flagged} in review (doubtful label); {n} attributions')
    for species, st in stats['species'].items():
        sp = counts.get(species, {})
        log(f'  {species:32s} kept {st.get(STATUS_KEPT, 0):4d}  rejected {st.get(STATUS_REJECTED, 0):3d}  '
            f'duplicates {st.get(STATUS_DUPLICATE, 0):3d}  review {st.get(STATUS_REVIEW, 0):3d}  '
            f'train/val/test {sp.get("train", 0)}/{sp.get("val", 0)}/{sp.get("test", 0)}')
    log(f'licences: {stats["licenses"]}')
    # `train.py` drops from the model any class without validation images. A
    # species remaining here has a single observation group: it needs
    # photographs, not a different split.
    without_val = [s for s, c in counts.items() if not c.get('val')]
    if without_val:
        log(f'{len(without_val)} species without validation, hence absent from the model: '
            f'{", ".join(sorted(without_val)[:8])}{" …" if len(without_val) > 8 else ""}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
