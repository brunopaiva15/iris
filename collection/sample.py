#!/usr/bin/env python3
"""A small dataset cut to measure a machine, not a model.

    python3 sample.py --dataset ~/plant-data/dataset \\
        --out ~/plant-data/dataset-sample

Moving tens of gigabytes only to discover that a machine is no faster is a day
lost. This script takes a few hundred megabytes, enough to measure a
throughput: same images, same pipeline, same recipe — only the number of
classes changes.

**And the number of classes falsifies almost nothing.** At 320 px the
MobileNetV3Large backbone costs about 0.45 GFLOP per image; the head,
`960 x classes`, costs 0.01 at 5,376 classes. Less than 3 % of the
computation. A throughput measured on 120 classes therefore describes the
machine to within a few percent — far below the difference one is looking for.

What is copied respects the `train.py` thresholds (`--min-train`,
`--min-val`): the classes kept are those with enough to pass them, otherwise
the model would drop them and the sample would be smaller than announced.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path


def read(dataset: Path) -> dict[str, dict[str, list[dict]]]:
    """{internal_id: {split: [rows]}}, in the order of the file."""
    per_class: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    with open(dataset / 'splits.csv', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            per_class[row['internal_plant_id']][row['split']].append(row)
    return per_class


def choose(per_class: dict, classes: int, train: int, val: int, test: int) -> list[str]:
    """The best-fed classes, so that none falls under a threshold.

    The largest are taken rather than a random draw: the sample is not there to
    measure an accuracy, only a throughput, and a class dropped by
    `--min-train` would be copying work for nothing.
    """
    eligible = [
        (len(s['train']), c) for c, s in per_class.items()
        if len(s['train']) >= train and len(s['val']) >= val and len(s['test']) >= test
    ]
    eligible.sort(key=lambda e: (-e[0], e[1]))
    return [c for _, c in eligible[:classes]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--classes', type=int, default=120)
    ap.add_argument('--train', type=int, default=60)
    ap.add_argument('--val', type=int, default=5)
    ap.add_argument('--test', type=int, default=5)
    args = ap.parse_args()

    dataset = Path(args.dataset).expanduser()
    out = Path(args.out).expanduser()
    if not (dataset / 'splits.csv').exists():
        raise SystemExit(f'splits.csv not found in {dataset}')

    per_class = read(dataset)
    kept = choose(per_class, args.classes, args.train, args.val, args.test)
    if not kept:
        raise SystemExit('no class meets the requested thresholds; lower --train / --val')
    if len(kept) < args.classes:
        print(f'only {len(kept)} classes meet the thresholds, instead of {args.classes}')

    out.mkdir(parents=True, exist_ok=True)
    quotas = {'train': args.train, 'val': args.val, 'test': args.test}
    rows, species, total_bytes, missing = [], {}, 0, 0
    for one_class in kept:
        for split, quota in quotas.items():
            for row in per_class[one_class][split][:quota]:
                source = dataset / row['path']
                if not source.exists():
                    missing += 1
                    continue
                target = out / row['path']
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                total_bytes += target.stat().st_size
                rows.append(row)
                species[row['internal_plant_id']] = row['species']

    with open(out / 'splits.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['path', 'species', 'internal_plant_id', 'split', 'group', 'captive'])
        for r in rows:
            w.writerow([r['path'], r['species'], r['internal_plant_id'],
                        r['split'], r.get('group', ''), r.get('captive', '0')])

    # `train.py` reads the species names from the manifest, at export time. A
    # minimal manifest is enough: the sample is not there to ship a model.
    with open(out / 'manifest.jsonl', 'w', encoding='utf-8') as f:
        for internal, name in sorted(species.items()):
            f.write(json.dumps({'internal_plant_id': internal, 'species': name},
                               ensure_ascii=False) + '\n')

    if missing:
        print(f'{missing} image(s) announced by splits.csv and absent from disk, ignored')
    print(f'{len(kept)} classes, {len(rows)} images, {total_bytes / 1e6:.0f} MB → {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
