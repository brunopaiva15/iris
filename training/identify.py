#!/usr/bin/env python3
"""Run a photograph through one or more exported models and show their opinions.

    python3 identify.py photo.jpg --model ./out --model ./shipped

Replaying a real photograph, taken by someone in their living room, says
things no average over the test set says: what the model answers when the
plant is crooked, the background cluttered, the light indoors. That is how a
yucca taken for maize was found.

The decision shown is the application's acceptance rule: the model answers
alone only if its confidence passes the threshold and the gap to the second
candidate is wide enough.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compare_models import load_model, predict

THRESHOLD, MIN_MARGIN = 0.70, 0.25


def species_names(folder: Path) -> dict:
    meta = json.loads((folder / 'model.json').read_text())
    return meta.get('species', {})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('photos', nargs='+')
    ap.add_argument('--model', action='append', required=True, help='folder of an exported model; repeatable')
    ap.add_argument('--top', type=int, default=5)
    args = ap.parse_args()

    models = [(load_model(Path(m)), species_names(Path(m))) for m in args.model]
    for photo in args.photos:
        print(f'\n=== {photo} ===')
        for model, names in models:
            probs = predict(model, photo)
            order = np.argsort(-probs)[:max(args.top, 2)]
            best, second = probs[order[0]], probs[order[1]]
            alone = best >= THRESHOLD and (best - second) >= MIN_MARGIN
            print(f"\n  v{model['version']} ({len(model['labels'])} classes) — "
                  f"{'answers alone' if alone else 'hesitates, the app would ask a remote service'}"
                  f"  [confidence {best:.3f}, gap {best - second:.3f}]")
            for rank, i in enumerate(order[:args.top], 1):
                internal = model['labels'][i]
                print(f'    {rank}. {names.get(internal, internal):38s} {probs[i]:6.1%}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
