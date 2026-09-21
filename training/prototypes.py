#!/usr/bin/env python3
"""Does the embedding separate cultivars, or was it trained to erase them?

    python3 prototypes.py --harvest --species "Acer palmatum,Hosta,Rosa"
    python3 prototypes.py --measure --folder .cache/cultivars

A cultivar cannot be a class: the sources do not have enough images of one.
But it does not need to be. The architecture considered has two storeys — the
classifier answers the species, then a **prototype in the embedding space**
proposes the cultivar, from ten to thirty photographs instead of two hundred.
Conditioned on the species, the subproblem is tiny: four or eight cultivars,
not five thousand classes.

Everything rests on one assumption, and **it is doubtful for a precise
reason**: every photograph of "Thai Constellation" in the set is labelled
*Monstera deliciosa*. Fine-tuning — a hundred unfrozen layers — therefore
explicitly pushes the embedding to **make the variegated cultivar and the
ordinary plant converge** to the same point. That is its job. Looking for
cultivars in this representation means looking for them in the one
representation trained to confuse them.

Hence this measurement, before anything else is written. It compares:

- the similarity **between two photographs of one cultivar**;
- and the one **between two cultivars of the same species** — the only
  comparison that matters, separating two species being already solved.

If the gap is nil, the embedding really has erased what is being looked for,
and one has to start from a frozen ImageNet network or from an earlier layer
(variegation is a colour signal, which the low layers keep better) before
going any further.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'collection'))


def normalise(v: np.ndarray) -> np.ndarray:
    """Vectors of norm 1: cosine similarity becomes a dot product."""
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def prototypes(vectors: np.ndarray, labels: list[str]) -> dict[str, np.ndarray]:
    """One mean vector per cultivar — all an embedded database would have to
    carry, and what allows a cultivar to be added without retraining
    anything."""
    v = normalise(np.asarray(vectors, dtype=np.float32))
    per: dict[str, list] = defaultdict(list)
    for i, e in enumerate(labels):
        per[e].append(v[i])
    return {e: normalise(np.mean(x, axis=0, keepdims=True))[0] for e, x in per.items()}


def separability(vectors: np.ndarray, labels: list[str]) -> dict:
    """Are two photographs of one cultivar closer than two cultivars of the
    same species?

    Also returns the accuracy of a prototype **leaving one photograph out**:
    computing the prototype over every photograph then classifying those same
    photographs would give a flattering and false figure.
    """
    v = normalise(np.asarray(vectors, dtype=np.float32))
    n = len(labels)
    intra, inter = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (intra if labels[i] == labels[j] else inter).append(float(v[i] @ v[j]))

    right = classifiable = 0
    for i in range(n):
        others = [k for k in range(n) if k != i]
        protos = prototypes(v[others], [labels[k] for k in others])
        if labels[i] not in protos or len(protos) < 2:
            continue      # without another example of the cultivar, nothing to find
        classifiable += 1
        right += max(protos, key=lambda e: float(v[i] @ protos[e])) == labels[i]

    cultivars = sorted(set(labels))
    return {
        'photos': n,
        'cultivars': len(cultivars),
        'intra': round(float(np.mean(intra)), 4) if intra else None,
        'inter': round(float(np.mean(inter)), 4) if inter else None,
        'gap': round(float(np.mean(intra) - np.mean(inter)), 4) if intra and inter else None,
        'prototype_accuracy': round(right / classifiable, 4) if classifiable else None,
        'chance': round(1 / len(cultivars), 4) if cultivars else None,
        'classifiable': classifiable,
    }


def control(vectors: np.ndarray, labels: list[str], shuffles: int = 200,
            seed: int = 20260910) -> dict:
    """A permutation test: does the observed gap stand out from what chance
    produces on those same photographs?

    **Without this control the experiment would conclude wrongly.** Ten
    photographs in a 960-dimensional space nearly always separate: on pure
    noise, a leave-one-out prototype reaches 0.9 accuracy. That is not a
    property of cultivars, it is a property of small samples in high
    dimension.

    And comparing the observed value to the **mean** of the shuffles is not
    enough: one realisation exceeds a mean half the time. What is needed is
    the share of shuffles doing as well — a p-value. Below 0.05, the
    separation is not an accident of the draw.
    """
    real = separability(vectors, labels)
    rng = np.random.default_rng(seed)
    gaps, accuracies = [], []
    for _ in range(shuffles):
        shuffled = list(labels)
        rng.shuffle(shuffled)
        r = separability(vectors, shuffled)
        if r['gap'] is not None:
            gaps.append(r['gap'])
        if r['prototype_accuracy'] is not None:
            accuracies.append(r['prototype_accuracy'])

    def p_value(observed, draws):
        if observed is None or not draws:
            return None
        # +1 on the numerator and the denominator: the observed value is
        # itself a permutation, and a p-value of zero does not exist.
        return round((1 + sum(1 for t in draws if t >= observed)) / (1 + len(draws)), 4)

    return {
        'mean_gap': round(float(np.mean(gaps)), 4) if gaps else None,
        'mean_accuracy': round(float(np.mean(accuracies)), 4) if accuracies else None,
        'p_gap': p_value(real['gap'], gaps),
        'p_prototype': p_value(real['prototype_accuracy'], accuracies),
        'shuffles': shuffles,
    }


def _harvest(species: list[str], folder: Path, per_cultivar: int, pause: float) -> None:
    """The cultivar photographs Commons really has, sorted by species.

    Commons does not name its categories after the cultivar but after the
    species: `Acer palmatum (cultivars)`, with one subcategory per named
    cultivar. Searching for `Acer palmatum 'Bloodgood'` returns zero — the
    method mistake that nearly led to concluding too early.
    """
    import requests
    from plant_dataset.fetchers.wikimedia import CommonsClient
    client = CommonsClient(pause=pause)
    for one in species:
        root = f'{one} (cultivars)'
        subs = [c['title'].split(':', 1)[-1] for c in client._members(root, 'subcat', 100)]
        print(f'{one}: {len(subs)} named cultivars', flush=True)
        for cat in subs:
            name = cat.split("'")[1] if "'" in cat else cat.replace(one, '').strip()
            target = folder / one.replace(' ', '-') / (name or cat).replace(' ', '-')
            target.mkdir(parents=True, exist_ok=True)
            kept = 0
            for cand in client.image_candidates(cat, max_files=per_cultivar, allow_share_alike=True):
                path = target / f'{kept:03d}.jpg'
                try:
                    r = requests.get(cand.image_url, timeout=60,
                                     headers={'User-Agent': 'IrisPlantDataset/0.1 (github.com/brunopaiva15/iris)'})
                    r.raise_for_status()
                    path.write_bytes(r.content)
                    kept += 1
                except Exception as e:
                    print(f'    {cand.image_url}: FAILED ({type(e).__name__})', file=sys.stderr)
            print(f'   {name:28s} {kept} photographs', flush=True)


def _read(folder: Path) -> list[tuple[Path, str, str]]:
    """(path, species, cultivar) for everything harvested."""
    out = []
    for species in sorted(p for p in folder.iterdir() if p.is_dir()):
        for cultivar in sorted(p for p in species.iterdir() if p.is_dir()):
            for image in sorted(cultivar.glob('*.jpg')):
                out.append((image, species.name, cultivar.name))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--harvest', action='store_true', help='download cultivar photographs from Commons')
    ap.add_argument('--measure', action='store_true', help='embed and measure separability')
    ap.add_argument('--species', default='Acer palmatum,Hosta,Rosa,Camellia japonica,Hydrangea macrophylla',
                    help='the genera rich in cultivars, where Commons has enough to measure')
    ap.add_argument('--folder', default='.cache/cultivars')
    ap.add_argument('--per-cultivar', type=int, default=40)
    ap.add_argument('--pause', type=float, default=1.0)
    ap.add_argument('--backbone', default='large')
    ap.add_argument('--weights', help='fine-tuned weights, to compare against the frozen ImageNet network')
    ap.add_argument('--shuffles', type=int, default=200,
                    help='permutations of the control; without it, ten photographs in 960\n'
                         'dimensions always separate')
    ap.add_argument('--minimum', type=int, default=3, help='minimum photographs for a cultivar to count')
    args = ap.parse_args(argv)

    folder = Path(args.folder)
    if args.harvest:
        _harvest([e.strip() for e in args.species.split(',') if e.strip()],
                 folder, args.per_cultivar, args.pause)
    if not args.measure:
        return 0

    rows = _read(folder)
    if not rows:
        print(f'{folder} is empty: run --harvest first', file=sys.stderr)
        return 1

    # Imported here: the pure functions above are testable without TensorFlow.
    import tensorflow as tf
    from train import IMAGE_SIZE, frozen_backbone

    model = frozen_backbone(args.backbone)
    if args.weights:
        model.load_weights(args.weights, skip_mismatch=True, by_name=True)
        print(f'weights loaded from {args.weights}')

    vectors = []
    for path, _, _ in rows:
        img = tf.io.decode_jpeg(tf.io.read_file(str(path)), channels=3)
        side = tf.reduce_min(tf.shape(img)[:2])
        img = tf.image.resize_with_crop_or_pad(img, side, side)
        img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
        vectors.append(model(tf.cast(img, tf.float32)[None, ...], training=False).numpy()[0])
    vectors = np.stack(vectors)

    print(f'\n{len(rows)} photographs embedded in {vectors.shape[1]} dimensions\n')
    for species in sorted({e for _, e, _ in rows}):
        idx = [i for i, (_, e, _) in enumerate(rows) if e == species]
        labels = [rows[i][2] for i in idx]
        enough = {c for c in set(labels) if labels.count(c) >= args.minimum}
        idx = [i for i in idx if rows[i][2] in enough]
        if len({rows[i][2] for i in idx}) < 2:
            print(f'{species}: fewer than two cultivars with {args.minimum} photographs — not measurable')
            continue
        kept_labels = [rows[i][2] for i in idx]
        r = separability(vectors[idx], kept_labels)
        t = control(vectors[idx], kept_labels, args.shuffles)
        print(f"{species} — {r['photos']} photographs, {r['cultivars']} cultivars")
        print(f"   same cultivar      : {r['intra']}")
        print(f"   different cultivars: {r['inter']}")
        print(f"   gap: {r['gap']}   (shuffled: {t['mean_gap']}, p = {t['p_gap']})")
        print(f"   prototype: {r['prototype_accuracy']}   (shuffled: {t['mean_accuracy']}, "
              f"p = {t['p_prototype']}, chance {r['chance']})")
        significant = (t['p_prototype'] is not None and t['p_prototype'] < 0.05)
        verdict = ('the cultivar is in the embedding' if significant
                   else 'NOTHING THERE: chance does as well on these same photographs')
        print(f'   → {verdict}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
