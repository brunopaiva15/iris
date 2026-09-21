#!/usr/bin/env python3
"""What the model answers when it is not shown a plant.

    python3 off_topic.py ~/off-topic-photos --model ./out

A classifier whose outputs are all plants has no way of saying "this is not a
plant": it spreads its mass among the species it knows, whatever it is shown.
In front of a cat it answers a plant — the only question is with how much
confidence.

The only guard today is the floor at 0.10, the margin being inactive at the
0.70 threshold. This measurement says whether it is enough, and it decides
between three pieces of work of very different sizes:

| what is observed          | what it means                                   |
|---------------------------|-------------------------------------------------|
| nearly everything < 0.10  | the floor is working, an "other" class is work  |
|                           | for nothing                                     |
| much between 0.10-0.70    | a "plausible" list on a photograph of a cat:    |
|                           | awkward, not serious — a message would do       |
| scores above 0.70         | the model **asserts** a species in front of     |
|                           | anything. Only then is an "other" class earned  |

Thirty photographs are enough: animals, furniture, faces, walls, plates.
Mixing in **plants outside the catalogue** is worth it — that is the most
common case in reality, and the hardest. Since the model started masking down
to 336 indoor species out of 1,569, it has become much more common still: the
species outside the mask do not disappear from the world.

`--per-folder` sorts the photographs by subfolder, if there are any: a
photograph of a cat and a ficus outside the catalogue do not tell the same
story, and mixing them into one average erases both.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from compare_models import load_model, predict
from identify import species_names

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
FLOOR, THRESHOLD = 0.10, 0.70


def photos(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob('*') if p.suffix.lower() in EXTENSIONS)


def band(score: float) -> str:
    if score < FLOOR:
        return 'under the floor'
    if score < THRESHOLD:
        return 'plausible'
    return 'ASSERTED'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folder', help='a folder of off-topic photographs, or one photograph')
    ap.add_argument('--model', default='./out')
    ap.add_argument('--per-folder', action='store_true',
                    help='break the report down by subfolder rather than one heap')
    ap.add_argument('--detail', type=int, default=10,
                    help='how many photographs to list, most asserted first')
    args = ap.parse_args()

    folder = Path(args.folder).expanduser()
    files = photos(folder)
    if not files:
        raise SystemExit(f'no image in {folder}')

    model = load_model(Path(args.model))
    names = species_names(Path(args.model))
    print(f'{len(files)} photographs through {model["name"]} '
          f'(version {model["version"]}, {len(model["labels"])} classes)\n')

    rows = []
    for f in files:
        probs = predict(model, str(f))
        order = np.argsort(-probs)[:2]
        top, second = float(probs[order[0]]), float(probs[order[1]])
        label = model['labels'][order[0]]
        rows.append({
            'photo': f,
            'group': f.parent.name if f.parent != folder else '(root)',
            'score': top,
            'margin': top - second,
            'species': names.get(label, label),
        })

    def table(title: str, batch: list[dict]) -> None:
        count = defaultdict(int)
        for r in batch:
            count[band(r['score'])] += 1
        scores = sorted(r['score'] for r in batch)
        median = scores[len(scores) // 2]
        print(f'— {title} ({len(batch)} photographs)')
        for name in ('under the floor', 'plausible', 'ASSERTED'):
            n = count[name]
            print(f'   {name:18s} {n:3d}  {n / len(batch):5.1%}')
        print(f'   median score       {median:.4f}   maximum {max(scores):.4f}\n')

    if args.per_folder:
        groups = defaultdict(list)
        for r in rows:
            groups[r['group']].append(r)
        for name in sorted(groups):
            table(name, groups[name])
    table('all', rows)

    # The most asserted first: those are what decide, not the average.
    worst = sorted(rows, key=lambda r: -r['score'])[:args.detail]
    print(f'the {len(worst)} most asserted')
    for r in worst:
        accepted = 'accepted' if r['score'] >= THRESHOLD and r['margin'] >= 0.25 else ''
        print(f'   {r["score"]:.4f}  margin {r["margin"]:.4f}  {r["species"]:34s} '
              f'{r["photo"].name}  {accepted}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
