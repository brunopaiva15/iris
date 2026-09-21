#!/usr/bin/env python3
"""What answering at genus level would be worth, when the species hesitates.

    python3 genus.py --model ./out --dataset ../collection/dataset
    python3 genus.py --model ./out --dataset ../collection/dataset --thresholds 0.6,0.7,0.8

The confusion breakdown only gives a **floor**: 64.3 % at genus level against
59.0 % at species level, counting only the cases where the *first* answer
already fell in the right genus. It ignores those where the genus mass was
right but **spread** over five species — that is, precisely the ones of
interest.

Here the distributions are kept. The mass of a genus is the sum of the
probabilities of its species: five *Picea* at 0.15 weigh 0.75, and "a spruce,
species uncertain" becomes a **true** answer where five names are not one.

The question is not academic, it is a product question: **on the photographs
the application does not accept today — the ones that go off to a remote
service on a quota — how many would the genus save, and at what price in
correctness?**

A genus with a single species in the catalogue changes nothing: its mass *is*
the species score. The gain comes from well-stocked genera, and therefore
grows with the catalogue — unlike top-1.

`--top N` answers the implementation question: the application only keeps
**five** candidates. Would summing over those five be enough? If so, the genus
can be computed in the cascade without touching the model; otherwise it has to
be computed on the whole vector, where it is still available, and carried
upwards. The difference between the two columns is the price of simplicity.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def genera_of(labels: list[str], names: dict[str, str] | None = None) -> list[str]:
    """The genus of each class, in the model's order."""
    from confusions import genus
    return [genus(c, names) for c in labels]


def genus_table(genera: list[str]) -> tuple[list[str], np.ndarray]:
    """The distinct genera, and the sparse matrix summing the species into
    them: `probs @ M` returns the mass of each genus, in one multiplication.
    """
    distinct = sorted(set(genera))
    rank = {g: i for i, g in enumerate(distinct)}
    M = np.zeros((len(genera), len(distinct)), dtype=np.float32)
    M[np.arange(len(genera)), [rank[g] for g in genera]] = 1.0
    return distinct, M


def truncate(P: np.ndarray, k: int) -> np.ndarray:
    """Keep only the `k` best scores of each row, the rest at zero.

    This is what the cascade sees: it sorts, cuts at five, and the mass of a
    genus spread over twenty species at 0.04 escapes it entirely.
    """
    if k <= 0 or k >= P.shape[1]:
        return P
    kept = np.argpartition(-P, k - 1, axis=1)[:, :k]
    out = np.zeros_like(P)
    rows = np.arange(len(P))[:, None]
    out[rows, kept] = P[rows, kept]
    return out


def answer(probs: np.ndarray, masses: np.ndarray, threshold: float) -> tuple[str, int, float]:
    """What the application would return for one photograph: a species if it
    passes the threshold, otherwise a genus if it does, otherwise nothing.

    The order matters. A species name is more useful than a genus name, so the
    species keeps priority; the genus only serves where the species was giving
    up — a net gain, never a replacement.
    """
    s = int(np.argmax(probs))
    if probs[s] >= threshold:
        return 'species', s, float(probs[s])
    g = int(np.argmax(masses))
    if masses[g] >= threshold:
        return 'genus', g, float(masses[g])
    return 'nothing', s, float(probs[s])


def measure(P: np.ndarray, truth: np.ndarray, M: np.ndarray, genus_of: np.ndarray,
            threshold: float, top: int = 0) -> dict:
    """The decisions over every photograph, counted by kind of answer.

    `top` limits the genus mass to the best classes, as a cascade holding only
    the candidates would. The species answer does not change: the best score
    remains the best.
    """
    masses = truncate(P, top) @ M
    true_genus = genus_of[truth]
    count = {'species': 0, 'species_right': 0, 'genus': 0, 'genus_right': 0, 'nothing': 0}
    for i in range(len(P)):
        what, which, _ = answer(P[i], masses[i], threshold)
        if what == 'species':
            count['species'] += 1
            count['species_right'] += int(which == truth[i])
        elif what == 'genus':
            count['genus'] += 1
            count['genus_right'] += int(which == true_genus[i])
        else:
            count['nothing'] += 1
    n = len(P)
    accepted = count['species'] + count['genus']
    right = count['species_right'] + count['genus_right']
    return {
        'images': n,
        'species_rate': count['species'] / n,
        'species_precision': count['species_right'] / count['species'] if count['species'] else None,
        'genus_rate': count['genus'] / n,
        'genus_precision': count['genus_right'] / count['genus'] if count['genus'] else None,
        'autonomy': accepted / n,
        'precision': right / accepted if accepted else None,
    }


def percent(x) -> str:
    return '—' if x is None else f'{x:.1%}'


def main() -> int:
    from compare_models import load_model, predict_rows, read_test

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='./out')
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--plants', default='../collection/plants.csv',
                    help='to read the genus from the scientific name rather than from the identifier')
    ap.add_argument('--thresholds', default='0.6,0.65,0.7,0.75,0.8')
    ap.add_argument('--top', type=int, default=0,
                    help='sum only the N best classes, as a cascade keeping five would; '
                         '0 = the whole vector')
    ap.add_argument('--sample', type=int, default=6000)
    ap.add_argument('--seed', type=int, default=20260905)
    args = ap.parse_args()

    model = load_model(Path(args.model))
    labels = model['labels']

    names = None
    plants = Path(args.plants)
    if plants.exists():
        import csv
        with open(plants, newline='', encoding='utf-8') as f:
            names = {r['internal_id']: r['scientific_name'] for r in csv.DictReader(f)}

    genera = genera_of(labels, names)
    distinct, M = genus_table(genera)
    rank = {g: i for i, g in enumerate(distinct)}
    genus_of = np.array([rank[g] for g in genera])

    sizes = np.bincount(genus_of, minlength=len(distinct))
    alone = int((sizes == 1).sum())
    print(f'{len(labels)} classes, {len(distinct)} genera — {alone} hold a single species '
          f'({alone / len(distinct):.0%}), the richest holds {int(sizes.max())}')

    rows = [(p, t) for p, t, _ in read_test(Path(args.dataset)) if t in model['index']]
    captive = {p for p, _, c in read_test(Path(args.dataset)) if c}
    rng = random.Random(args.seed)
    if args.sample and len(rows) > args.sample:
        rows = rng.sample(rows, args.sample)
    potted = np.array([p in captive for p, _ in rows])

    print(f'\ninference on {len(rows)} test images…', flush=True)
    pred = predict_rows(rows, model)
    P = np.stack([p for _, p in pred])
    truth = np.array([model['index'][t] for t, _ in pred])
    potted = potted[:len(P)]

    share = (sizes[genus_of[truth]] > 1).mean()
    masses = P @ M
    print(f'{share:.0%} of the test images are of a species whose genus holds several — '
          'the only ones where the genus adds anything')
    print(f'\ntop-1 at species level {(np.argmax(P, axis=1) == truth).mean():.4f}   '
          f'at genus level {(np.argmax(masses, axis=1) == genus_of[truth]).mean():.4f}   '
          '(the confusion breakdown only gave a floor)')

    for name, mask in (('all photographs', np.ones(len(P), bool)), ('potted photographs', potted)):
        if not mask.any():
            continue
        print(f'\n— {name} ({int(mask.sum())} images)')
        print(f"{'thresh':>6} {'species':>8} {'right':>8}   {'+ genus':>8} {'right':>8}   "
              f"{'autonomy':>10} {'precision':>9}")
        for s in (float(x) for x in args.thresholds.split(',')):
            r = measure(P[mask], truth[mask], M, genus_of, s, args.top)
            print(f'{s:>6.2f} {percent(r["species_rate"]):>8} {percent(r["species_precision"]):>8}   '
                  f'{percent(r["genus_rate"]):>8} {percent(r["genus_precision"]):>8}   '
                  f'{percent(r["autonomy"]):>10} {percent(r["precision"]):>9}')
    if args.top:
        print(f'\nMasses summed over the {args.top} best classes only.')
    print('\nThe species keeps priority: the genus answers only where it was giving up.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
