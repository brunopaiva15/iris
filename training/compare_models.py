#!/usr/bin/env python3
"""Compare two exported models on the same images, on equal terms.

    python3 compare_models.py --dataset ../collection/dataset \\
        --a ./shipped --b ./candidate --sample 6000

The `model.json` numbers of two versions do not compare: they were measured on
different test sets, each drawn from its own collection. A model that gains
three points may simply have been handed an easier test.

Here both models see **the same images** — those of the current set's test
split — and only on the species **both** of them know. That is the only
measurement saying whether replacing the shipped model makes the user gain or
lose on the plants they already had.

Each ground is read twice, and the two readings answer different questions:

- **masked**: the classes the other model does not have are removed from the
  outputs. This is model quality on equal terms — the wider one is not
  punished for knowing more. But it is not what the user receives;
- **full outputs**: each model answers with its whole catalogue. A photograph
  of a *Monstera* can now be taken for one of the thousands of species the old
  model ignored, and that risk is real. This is the reading that decides
  whether to ship.

A model can win the first and lose the second: that is precisely what breadth
costs, and it is better known before the shipped file is replaced.

Three grounds:

- **common classes**: the shared ground;
- **cultivated plants**: the potted photographs, that is what the application
  really sees;
- **coverage**: what the new model can name and the old one could not,
  measured on its own images.

Each model's preprocessing is read from its own `model.json`: two versions may
have been trained at different sizes, and applying them the wrong way round
would cost several points.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np


def _tf():
    """TensorFlow, imported only when inferring.

    The counting (`tally`) needs numpy alone: keeping it importable without
    TensorFlow allows testing it on a machine that has none, and that is where
    mask mistakes hide. `indoor.py --coverage` already defers its import of
    this module for the same reason.
    """
    import tensorflow as tf
    return tf


def load_model(folder: Path):
    """An exported model: its interpreter, its labels, its preprocessing
    recipe."""
    tf = _tf()
    meta = json.loads((folder / 'model.json').read_text())
    labels = (folder / 'labels.txt').read_text().split()
    interpreter = tf.lite.Interpreter(model_path=str(folder / 'plants.tflite'))
    interpreter.allocate_tensors()
    return {
        'name': folder.name,
        'version': meta.get('version', '?'),
        'labels': labels,
        'index': {c: i for i, c in enumerate(labels)},
        'input_size': int(meta.get('input_size', 224)),
        'load_size': int(meta.get('load_size', 256)),
        'interpreter': interpreter,
        'in': interpreter.get_input_details()[0],
        'out': interpreter.get_output_details()[0],
    }


def prepare(path: str, load_size: int, input_size: int) -> np.ndarray:
    """The centred square reduced to `load_size`, then cropped to
    `input_size` — exactly the recipe written in `model.json` and applied by
    the application."""
    tf = _tf()
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    side = tf.reduce_min(tf.shape(image)[:2])
    image = tf.image.resize_with_crop_or_pad(image, side, side)
    image = tf.image.resize(image, [load_size, load_size])
    offset = (load_size - input_size) // 2
    image = tf.image.crop_to_bounding_box(image, offset, offset, input_size, input_size)
    return tf.cast(image, tf.float32).numpy()[None, ...]


def predict(model, path: str) -> np.ndarray:
    x = prepare(path, model['load_size'], model['input_size'])
    if model['in']['dtype'] != np.float32:
        x = x.astype(model['in']['dtype'])
    model['interpreter'].set_tensor(model['in']['index'], x)
    model['interpreter'].invoke()
    return model['interpreter'].get_tensor(model['out']['index'])[0]


def predict_rows(rows, model) -> list[tuple[str, np.ndarray]]:
    """The model's outputs on these images, computed once.

    Separate from the counting because one pass serves several readings:
    masked and not are two additions over the same probabilities. The cost is
    in the inference, not in the counting — and running six thousand images
    again to change a mask would be paying twice.
    """
    return [(truth, predict(model, path)) for path, truth in rows if truth in model['index']]


def tally(predictions, model, restrict: set[str] | None, renormalise: bool = False,
          threshold: float = 0.70) -> dict:
    """Top-1, top-3 and acceptance rate at the application's threshold.

    `renormalise` does **not** change the top-1: masking preserves the order
    among the classes that remain. It only changes the threshold columns, and
    it answers two different questions:

    - **without** (the default): "this model, judged on the classes the other
      one also knows". The confidence stays the one the application would
      read, so the 0.70 threshold keeps its meaning;
    - **with**: "what a model that had learned only those classes would
      return". Its final layer would spread the mass among them; without
      renormalising, the measured autonomy is artificially low, since the
      probability that went to the masked classes comes back to nobody.

    It is this second reading that describes an application restricting its
    outputs to its catalogue: it masks, therefore it renormalises before
    showing a confidence. `threshold` is then swept to find the one that
    returns the previous version's autonomy without falling below its
    precision.
    """
    seen = hit1 = hit3 = accepted = accepted_ok = 0
    for truth, probs in predictions:
        if restrict is not None:
            # Equal terms: the classes the other model does not have are
            # masked. Without this, the wider one is punished for knowing more.
            mask = np.zeros_like(probs)
            for c in restrict:
                i = model['index'].get(c)
                if i is not None:
                    mask[i] = 1.0
            probs = probs * mask
            if renormalise:
                mass = float(probs.sum())
                if mass > 0:
                    probs = probs / mass
        order = np.argsort(-probs)[:3]
        top = [model['labels'][i] for i in order]
        seen += 1
        hit1 += top[0] == truth
        hit3 += truth in top
        if probs[order[0]] >= threshold:
            accepted += 1
            accepted_ok += top[0] == truth
    return {
        'images': seen,
        'top1': round(hit1 / seen, 4) if seen else None,
        'top3': round(hit3 / seen, 4) if seen else None,
        'accepted_rate': round(accepted / seen, 4) if seen else None,
        'precision_when_accepted': round(accepted_ok / accepted, 4) if accepted else None,
    }


def score(rows, model, restrict: set[str] | None, renormalise: bool = False,
          threshold: float = 0.70) -> dict:
    """Inference then counting, for whoever has only one reading to do."""
    return tally(predict_rows(rows, model), model, restrict, renormalise, threshold)


def read_rows(dataset: Path, split: str) -> list[tuple[str, str, bool]]:
    """Path, species, `captive` flag — for one split."""
    rows = []
    with open(dataset / 'splits.csv', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['split'] != split:
                continue
            path = dataset / row['path']
            if path.exists():
                rows.append((str(path), row['internal_plant_id'], row.get('captive') == '1'))
    return rows


def read_test(dataset: Path) -> list[tuple[str, str, bool]]:
    return read_rows(dataset, 'test')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='../collection/dataset')
    ap.add_argument('--a', required=True, help='reference model (the shipped one)')
    ap.add_argument('--b', required=True, help='candidate model')
    ap.add_argument('--sample', type=int, default=6000, help='test images drawn at random, 0 = all of them')
    ap.add_argument('--seed', type=int, default=20260905)
    ap.add_argument('--restricted', action='store_true',
                    help='add the "masked and renormalised" reading: what an application that '
                         'restricts its outputs to its own catalogue would return, swept over '
                         'several thresholds')
    ap.add_argument('--thresholds', default='0.5,0.6,0.7,0.8',
                    help='thresholds swept by --restricted')
    args = ap.parse_args()

    a, b = load_model(Path(args.a)), load_model(Path(args.b))
    rows = read_test(Path(args.dataset))
    shared = set(a['index']) & set(b['index'])
    only_b = set(b['index']) - set(a['index'])
    print(f"A = v{a['version']} ({len(a['labels'])} classes)   B = v{b['version']} ({len(b['labels'])} classes)")
    print(f'{len(shared)} common classes, {len(only_b)} known to B only, {len(set(a["index"]) - set(b["index"]))} to A only\n')

    common = [(p, t) for p, t, _ in rows if t in shared]
    captive = [(p, t) for p, t, c in rows if t in shared and c]
    new = [(p, t) for p, t, _ in rows if t in only_b]
    rng = random.Random(args.seed)
    thresholds = [float(x) for x in args.thresholds.split(',') if x.strip()]

    def take(items, n):
        return rng.sample(items, n) if n and len(items) > n else items

    def line(model, r):
        print(f"   v{model['version']}: top1 {r['top1']}  top3 {r['top3']}  "
              f"threshold 0.70 → {r['accepted_rate']} accepted, precision {r['precision_when_accepted']}")

    for title, subset in (
        ('common classes', take(common, args.sample)),
        ('cultivated plants, common classes', take(captive, args.sample // 3)),
    ):
        # One inference pass per model, several readings over it.
        outputs = [(model, predict_rows(subset, model)) for model in (a, b)]
        print(f'— {title} ({len(subset)} images) — equal terms, masked outputs')
        for model, pred in outputs:
            line(model, tally(pred, model, shared))
        print()
        print(f'— {title} ({len(subset)} images) — full outputs, what the application returns')
        for model, pred in outputs:
            line(model, tally(pred, model, None))
        print()
        if args.restricted:
            # Masking removes mass; without giving it back, the threshold is
            # harsher than it looks and the measured autonomy is wrong.
            print(f'— {title} ({len(subset)} images) — masked and renormalised, '
                  'an application restricted to its catalogue')
            for model, pred in outputs:
                for threshold in thresholds:
                    r = tally(pred, model, shared, renormalise=True, threshold=threshold)
                    print(f"   v{model['version']} top1 {r['top1']}  threshold {threshold:.2f} → "
                          f"{r['accepted_rate']} accepted, precision {r['precision_when_accepted']}")
            print()

    if new:
        subset = take(new, args.sample // 3)
        r = score(subset, b, None)
        species = len({t for _, t in subset})
        print(f'— coverage gained: {species} species v{a["version"]} does not know ({len(subset)} images)')
        print(f"   v{b['version']}: top1 {r['top1']}  top3 {r['top3']}")
        print(f'   (v{a["version"]} is necessarily at 0 there: those species are not among its outputs)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
