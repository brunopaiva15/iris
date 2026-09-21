#!/usr/bin/env python3
"""What the model returns to someone photographing a houseplant.

    python3 indoor.py --coverage                         # without TensorFlow
    python3 indoor.py --dataset ../collection/dataset \\
        --model ./out --sample 4000

The published top-1 is an average over more than a thousand species, most of
them wild, European, and photographed by nobody in a living room. The
application serves first the names of `phase1_species.txt`: the plants people
buy in a garden centre and put on a shelf.

Two numbers answer, and both are needed:

- **coverage**: how many of these plants the model can so much as name. A
  species absent from the catalogue is a certain failure for the user, and it
  is *invisible* in any accuracy measurement — one does not get a
  non-existent class wrong, it is simply never offered. This part needs
  neither TensorFlow nor the image set;
- **accuracy on what is covered**: the top-1 on the test images of those
  species only, measured on the shipped `.tflite`.

And, for accuracy, two readings that answer each other:

- **whole catalogue**: every output of the shipped model stays open (their
  number is in `model.json`). This is what the user lives with;
- **restricted catalogue**: the outputs are masked down to the indoor plants
  alone. This is what a model that had learned only those would return.

The gap between the two is the **price of breadth**: what the species nobody
will ever photograph cost the person photographing their living room.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'collection'))

#: The application's acceptance threshold.
THRESHOLD = 0.70


def _id(name: str) -> str:
    """Scientific name → internal id, without depending on the collector.

    `plant_dataset.taxonomy.internal_id` is authoritative and is used when it
    can be imported; this fallback copy is enough for the names of a species
    list, which are already clean. Both must return the same key, or coverage
    would be measured against a different catalogue from the model's.
    """
    try:
        from plant_dataset.taxonomy import internal_id
        return internal_id(name)
    except Exception:
        clean = ' '.join(name.replace('×', 'x').replace('.', ' ').split())
        return clean.lower().replace(' ', '-')


def alias_from(rows) -> dict[str, str]:
    """id of a synonym → id kept by the catalogue.

    `plants.csv` carries a `synonyms` column: it is what attaches
    *Streptocarpus ionanthus* to `saintpaulia-ionantha`. Without it, a species
    present under its other name would be counted as absent — the exact
    mistake that once cost a collection run.
    """
    alias: dict[str, str] = {}
    for r in rows:
        kept = r['internal_id']
        alias.setdefault(kept, kept)
        for s in (r.get('synonyms') or '').replace(';', '|').replace(',', '|').split('|'):
            if s.strip():
                alias.setdefault(_id(s.strip()), kept)
        alias.setdefault(_id(r['scientific_name']), kept)
    return alias


def resolve(names: list[str], labels: set[str], alias: dict[str, str] | None = None) -> dict:
    """Sort the target names by whether the model can name them.

    Three ways of being a class, and all three are needed, or plants the model
    knows get announced as missing:

    - **directly**: the identifier is a label;
    - **by synonym**: `plants.csv` attaches the old name to the new one;
    - **through the hybrid**: the catalogue writes `citrus-x-limon` where the
      list writes "Citrus limon". The × is taxonomic information, not another
      plant; dropping it would make the lemon tree look like a missing
      species.

    Two names can designate the same plant — *Calathea orbifolia* and
    *Goeppertia orbifolia* are both in the list. They are counted once,
    including when neither is a class: otherwise coverage would be measured
    against an inflated denominator.
    """
    alias = alias or {}
    canonical: dict[str, str] = {}      # target name → the plant it designates
    per_name: dict[str, str] = {}       # target name → the model's class
    synonyms: dict[str, str] = {}
    hybrids: dict[str, str] = {}
    for name in names:
        i = _id(name)
        j = alias.get(i, i)
        genus, _, epithet = i.partition('-')
        h = f'{genus}-x-{epithet}' if epithet else ''
        if i in labels:
            per_name[name] = i
        elif j in labels:
            per_name[name] = synonyms[name] = j
        elif h in labels:
            per_name[name] = hybrids[name] = h
        canonical[name] = per_name.get(name, j)
    count = Counter(canonical.values())
    classes = sorted(set(per_name.values()))
    distinct = len(count)
    return {
        'requested': len(names),
        'distinct': distinct,
        'classes': classes,
        'coverage': round(len(classes) / distinct, 4) if distinct else 0.0,
        'per_name': per_name,
        'synonyms': synonyms,
        'hybrids': hybrids,
        'missing': [n for n in names if n not in per_name],
        'duplicates': sorted(i for i, n in count.items() if n > 1),
    }


def cost_of_breadth(whole: dict, restricted: dict) -> dict | None:
    """The gap between the two readings, in points of top-1.

    Positive: the other species cost something to the person photographing
    their living room. Zero or negative: breadth is free, and the catalogue
    can be widened without taking anything back.
    """
    if whole.get('top1') is None or restricted.get('top1') is None:
        return None
    return {
        'top1': round(restricted['top1'] - whole['top1'], 4),
        'top3': round((restricted.get('top3') or 0) - (whole.get('top3') or 0), 4),
        'accepted': round((restricted.get('accepted_rate') or 0) - (whole.get('accepted_rate') or 0), 4),
    }


def _read_names(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def _print_coverage(r: dict, readable: dict[str, str], detail: int) -> None:
    print(f"{r['requested']} target names, {r['distinct']} distinct plants, "
          f"{len(r['classes'])} the model can name — **{r['coverage']:.0%} coverage**\n")
    for name, i in r['synonyms'].items():
        print(f'  {name} is in the catalogue as {i}')
    for name, i in r['hybrids'].items():
        print(f'  {name} is in the catalogue as {i} — the × the list leaves out')
    if r['duplicates']:
        names = ', '.join(readable.get(i, i) for i in r['duplicates'])
        print(f"  {len(r['duplicates'])} plants named twice in the list: {names}")
    if r['missing']:
        print(f"\n**{len(r['missing'])} indoor names the model cannot return.**")
        print('For those, no accuracy can be measured: the class does not exist, it will')
        print('never be offered, and the answer is wrong for certain.')
        for name in r['missing'][:detail]:
            print(f'  {name}')
        if len(r['missing']) > detail:
            print(f"  … and {len(r['missing']) - detail} more")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='./out', help='the shipped model, the one running in the app')
    ap.add_argument('--species-file', default='../collection/phase1_species.txt')
    ap.add_argument('--plants', default='../collection/plants.csv', help='for the synonyms')
    ap.add_argument('--dataset', default='../collection/dataset')
    ap.add_argument('--coverage', action='store_true',
                    help='measure coverage only: neither TensorFlow nor images')
    ap.add_argument('--sample', type=int, default=4000, help='test images drawn at random, 0 = all of them')
    ap.add_argument('--seed', type=int, default=20260909)
    ap.add_argument('--detail', type=int, default=20)
    args = ap.parse_args(argv)

    model_dir = Path(args.model)
    labels = {l.strip() for l in (model_dir / 'labels.txt').read_text(encoding='utf-8').splitlines() if l.strip()}
    names = _read_names(Path(args.species_file))
    plants = Path(args.plants)
    rows = list(csv.DictReader(plants.open(encoding='utf-8'))) if plants.exists() else []
    readable = {r['internal_id']: r['scientific_name'] for r in rows}
    r = resolve(names, labels, alias_from(rows))
    print(f"model: {len(labels)} classes\n")
    _print_coverage(r, readable, args.detail)
    if args.coverage:
        return 0

    # Imported here: the coverage above must be readable on any machine,
    # including one without TensorFlow or the image set.
    try:
        from compare_models import load_model, read_test, score
    except ImportError as e:
        print(f"\nwhat follows needs TensorFlow ({e}): `pip install -r requirements.txt`,"
              "\nor `--coverage` to stop at what precedes.", file=sys.stderr)
        return 1
    dataset = Path(args.dataset)
    if not (dataset / 'splits.csv').exists():
        print(f'\n{dataset}/splits.csv is missing: what follows needs the image set.', file=sys.stderr)
        return 1

    indoor = set(r['classes'])
    model = load_model(model_dir)
    everything = read_test(dataset)
    inside = [(p, t) for p, t, _ in everything if t in indoor]
    cultivated = [(p, t) for p, t, cap in everything if t in indoor and cap]
    rng = random.Random(args.seed)

    def draw(items, n):
        return rng.sample(items, n) if n and len(items) > n else items

    species_seen = len({t for _, t in inside})
    print(f"\n{len(inside)} test images of these plants, {species_seen} species represented")
    if not inside:
        print('none: the test set holds no image of these species.')
        return 0

    batch = draw(inside, args.sample)
    print(f'\n— houseplants ({len(batch)} images)')
    whole = score(batch, model, None)
    # Renormalised: a model that had learned only these classes would spread
    # among them the mass this one gives to all the others. The top-1 does not
    # move — masking preserves the order — only the threshold columns are
    # concerned, and without this one would read a falling autonomy where it
    # actually rises.
    restricted = score(batch, model, indoor, renormalise=True)
    for title, res in (('whole catalogue     ', whole), ('restricted catalogue', restricted)):
        print(f"   {title}: top1 {res['top1']}  top3 {res['top3']}  "
              f"threshold {THRESHOLD} → {res['accepted_rate']} accepted, precision {res['precision_when_accepted']}")
    gap = cost_of_breadth(whole, restricted)
    if gap:
        print(f"   → the price of breadth: {gap['top1']:+.4f} of top-1, {gap['accepted']:+.4f} of autonomy")

    if cultivated:
        batch = draw(cultivated, max(1, args.sample // 2))
        res = score(batch, model, None)
        print(f"\n— the same ones, photographed in a pot ({len(batch)} images) — closest to the application")
        print(f"   top1 {res['top1']}  top3 {res['top3']}  "
              f"threshold {THRESHOLD} → {res['accepted_rate']} accepted, precision {res['precision_when_accepted']}")

    rest = draw([(p, t) for p, t, _ in everything if t not in indoor], args.sample)
    if rest:
        res = score(rest, model, None)
        print(f'\n— for comparison, the whole rest of the catalogue ({len(rest)} images)')
        print(f"   top1 {res['top1']}  top3 {res['top3']}")
        print('   (measured on the shipped `.tflite`, not on the Keras network: that is what\n'
              '    the user runs, and `model.json` had never checked it)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
