#!/usr/bin/env python3
"""What PlantNet-300K would bring to our model, measured before writing a
connector.

    python3 plantnet300k.py --labels ../training/out/labels.txt

The set was once set aside on an intuition: its flora is European and wild,
ours is indoors. The intuition deserved a number, and the number says
something else.

Two metadata files are enough, 66 MB in total, without downloading the 306,000
images or the 32 GB of the full set. They are published separately by Pl@ntNet,
precisely for this.

What the tool returns, and why each line counts:

- **the species overlap**, which bounds everything: an image only serves if the
  species is one of our classes;
- **the licences**, filtered by the same rule as the collection;
- **the distribution by organ**, which is the real question. `habit` is the
  whole plant; `flower` and `leaf` are close-ups. A model fed close-ups of
  European flowers will not recognise a zamioculcas in a living room any
  better — that is the mistake an earlier version already paid for once.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plant_dataset.taxonomy import internal_id, normalize_scientific_name  # noqa: E402

#: The metadata alone, without the images. The address is given by the
#: PlantNet-300K repository itself, Zenodo not allowing them to be downloaded
#: separately.
SEAFILE = 'https://seafile.plantnet.org/d/bed81bc15e8944969cf6/files/?p=%2F{}&dl=1'
NAMES = 'plantnet300K_species_id_2_name.json'
METADATA = 'plantnet300K_metadata.json'

#: The project's licence rule, in PlantNet's vocabulary.
ACCEPTED_LICENCES = {'cc0', 'cc-by', 'cc-by-sa'}

#: The whole plant. The only organ whose framing resembles what a user of the
#: application photographs.
WHOLE_PLANT = 'habit'


def mapping(species_id_2_name: dict[str, str], labels: set[str]) -> dict[str, str]:
    """PlantNet species identifier → our internal identifier.

    Their names carry their author ("Lactuca virosa L."); the same
    normalisation as the collection is applied, or nothing would ever meet.
    Several of their identifiers can land on a single species of ours —
    synonyms or subspecies — and that is fine: their images would rejoin.
    """
    out = {}
    for sid, name in species_id_2_name.items():
        canonical = normalize_scientific_name(name)
        if canonical and internal_id(canonical) in labels:
            out[sid] = internal_id(canonical)
    return out


def count(metadata: dict[str, dict], mappings: dict[str, str],
          licences: set[str] = ACCEPTED_LICENCES) -> dict:
    """What the shared species offer: images, licences, organs.

    The images whose licence is refused are counted apart rather than ignored:
    knowing whether a set is 100 % or 18 % usable changes the decision, and
    that is what ruled out another source.
    """
    per_species, per_organ, per_licence, whole = Counter(), Counter(), Counter(), Counter()
    for image in metadata.values():
        internal = mappings.get(image.get('species_id'))
        if internal is None:
            continue
        per_licence[image.get('license', '?')] += 1
        if image.get('license') not in licences:
            continue
        per_species[internal] += 1
        per_organ[image.get('organ', '?')] += 1
        if image.get('organ') == WHOLE_PLANT:
            whole[internal] += 1
    return {
        'species': len(per_species),
        'images': sum(per_species.values()),
        'images_all_licences': sum(per_licence.values()),
        'per_species': per_species,
        'per_organ': per_organ,
        'per_licence': per_licence,
        'whole': whole,
    }


def download(name: str, folder: Path) -> Path:
    target = folder / name
    if target.exists():
        return target
    folder.mkdir(parents=True, exist_ok=True)
    print(f'  downloading {name}…', file=sys.stderr, flush=True)
    urllib.request.urlretrieve(SEAFILE.format(name), target)
    return target


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--labels', default='../training/out/labels.txt',
                    help='the classes of the shipped model, one internal identifier per line')
    ap.add_argument('--cache', default='.cache/plantnet300k', help='where to keep the two metadata files')
    ap.add_argument('--plants', default='plants.csv', help='to name the species in the report')
    ap.add_argument('--top', type=int, default=15, help='how many species to detail')
    args = ap.parse_args(argv)

    labels = {l.strip() for l in Path(args.labels).read_text().splitlines() if l.strip()}
    cache = Path(args.cache)
    names = json.loads(download(NAMES, cache).read_text())
    mapped = mapping(names, labels)

    canonical = {normalize_scientific_name(n) for n in names.values()}
    canonical.discard('')
    print(f'PlantNet-300K: {len(names)} identifiers, {len(canonical)} species')
    print(f'our model    : {len(labels)} classes')
    shared = set(mapped.values())
    print(f'\noverlap: {len(shared)} species — {len(shared) / len(labels):.1%} of our classes\n')
    if not shared:
        return 0

    stats = count(json.loads(download(METADATA, cache).read_text()), mapped)
    usable = stats['images']
    total = stats['images_all_licences']
    print(f"images on those species: {total}, of which {usable} under an accepted licence ({usable / total:.0%})")
    print('  licences:', dict(stats['per_licence'].most_common(5)))
    print('  organs  :', dict(stats['per_organ'].most_common(8)))
    whole = sum(stats['whole'].values())
    print(f"  whole plant ({WHOLE_PLANT}): {whole} images, {whole / usable:.1%} of the total")

    per = stats['per_species']
    readable = {}
    plants = Path(args.plants)
    if plants.exists():
        import csv
        readable = {r['internal_id']: r['scientific_name'] for r in csv.DictReader(plants.open())}
    print(f'\nthe {args.top} species that would gain the most:')
    for i, n in per.most_common(args.top):
        print(f'  {readable.get(i, i):36s} {n:6d} images  of which {stats["whole"][i]:4d} whole')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
