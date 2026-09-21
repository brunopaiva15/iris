#!/usr/bin/env python3
"""A frozen evaluation set, so that two models can be compared at all.

    python3 benchmark.py --dataset ../collection/dataset \\
        --off-topic ~/plant-data/off-topic --out benchmark.csv

The numbers in two `model.json` files do not compare. Every training run has
produced its own test split so far, which means a model that "gained three
points" may simply have been handed an easier test. Anything meant to beat the
shipped model on its own ground needs rules that stop moving.

What this script writes is therefore a **manifest**, not images: a CSV saying
which photograph belongs to which slice and what its ground truth is. It is
reproducible down to the seed, and recomputes identically as long as the image
set does not change.

Five slices, each answering a question the others do not ask:

| slice | what it measures |
|---|---|
| `indoor` | the species of the indoor mask, on photographs of cultivated plants |
| `outdoor` | the species the outdoor mask exposes and the indoor one does not |
| `multi` | observations with two or more photographs, where merging is measured |
| `ood_plant` | real plants outside both masks: the most common hard case |
| `ood_other` | what is not a plant at all, if anything was supplied |

The `ood_other` slice stays empty until `--off-topic` points at populated
folders. That is deliberate: a benchmark claiming to measure refusal without a
single photograph of a cat would mislead more than it helps.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


def read_mask(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    return {l.strip() for l in path.read_text(encoding='utf-8').splitlines() if l.strip()}


def read_test(dataset: Path) -> list[dict]:
    """The test images, with their species, their group and their domain."""
    rows = []
    with open(dataset / 'splits.csv', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['split'] != 'test':
                continue
            path = dataset / r['path']
            if not path.exists():
                continue
            rows.append({
                'path': str(path),
                'truth': r['internal_plant_id'],
                'group': r.get('group', ''),
                'captive': '1' if r.get('captive') == '1' else '0',
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='../collection/dataset')
    ap.add_argument('--indoor', default='../collection/mask_indoor.txt')
    ap.add_argument('--outdoor', default='../collection/mask_outdoor.txt')
    ap.add_argument('--off-topic', help='folder of off-topic photographs, one subfolder per kind')
    ap.add_argument('--out', default='benchmark.csv')
    ap.add_argument('--per-slice', type=int, default=2000,
                    help='at most this many images per slice; 0 = all of them')
    ap.add_argument('--seed', type=int, default=20260919)
    args = ap.parse_args()

    dataset = Path(args.dataset).expanduser()
    indoor = read_mask(Path(args.indoor).expanduser())
    outdoor = read_mask(Path(args.outdoor).expanduser())
    if not indoor:
        raise SystemExit(f'indoor mask missing or empty: {args.indoor}')

    test = read_test(dataset)
    if not test:
        raise SystemExit(f'no test image in {dataset}')

    # Groups first: an observation with several photographs must not have them
    # spread across slices, or the merging slice would be measured on images
    # the other slices already gave away.
    by_group = defaultdict(list)
    for row in test:
        by_group[row['group']].append(row)
    multi = [row for g, batch in by_group.items() if g and len(batch) >= 2 for row in batch]
    seen_in_multi = {row['path'] for row in multi}

    slices: dict[str, list[dict]] = {
        'indoor': [r for r in test if r['truth'] in indoor
                   and r['captive'] == '1' and r['path'] not in seen_in_multi],
        'outdoor': [r for r in test if r['truth'] in outdoor and r['truth'] not in indoor
                    and r['path'] not in seen_in_multi],
        'multi': multi,
        'ood_plant': [r for r in test if r['truth'] not in indoor and r['truth'] not in outdoor
                      and r['path'] not in seen_in_multi],
        'ood_other': [],
    }

    if args.off_topic:
        root = Path(args.off_topic).expanduser()
        for p in sorted(root.rglob('*')):
            if p.suffix.lower() in EXTENSIONS:
                slices['ood_other'].append({
                    'path': str(p), 'truth': '',
                    'group': p.parent.name, 'captive': '0',
                })

    rng = random.Random(args.seed)
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['slice', 'path', 'truth', 'group', 'captive'])
        for name, batch in slices.items():
            kept = batch
            if args.per_slice and len(batch) > args.per_slice:
                # Draw by group, not by image: splitting an observation in two
                # would measure the merge on a missing photograph.
                groups = sorted({r['group'] or r['path'] for r in batch})
                rng.shuffle(groups)
                chosen, n = set(), 0
                for g in groups:
                    if n >= args.per_slice:
                        break
                    chosen.add(g)
                    n += sum(1 for r in batch if (r['group'] or r['path']) == g)
                kept = [r for r in batch if (r['group'] or r['path']) in chosen]
            species = len({r['truth'] for r in kept if r['truth']})
            print(f'{name:12s} {len(kept):6d} images  {species:5d} species')
            for r in kept:
                w.writerow([name, r['path'], r['truth'], r['group'], r['captive']])
    print(f'\n{args.out} written — seed {args.seed}, recomputes identically')
    if not slices['ood_other']:
        print('ood_other is empty: point --off-topic at populated folders')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
