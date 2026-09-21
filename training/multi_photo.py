#!/usr/bin/env python3
"""What the model gains when the user takes two or three photographs.

    python3 multi_photo.py --dataset ../collection/dataset --model ./out

The application's cascade already receives a list of files and uses only one.
Asking for a second photograph is the main lever of the remote services, and
it costs no training at all: the outputs only have to be averaged. What
remains is to know what it really returns — hence this measurement, before a
single line is written in the application.

The test set lends itself to this without collecting anything: `splits.csv`
carries a `group` column where all the photographs of one observation are
gathered, and the split never separates a group. Two photographs of the same
group are two photographs of the same plant — the exact scenario.

**What the measurement overestimates, and must be kept in mind:** the
photographs of one observation were taken in the same session, under the same
light, often in the same place. A user photographing a leaf and then the whole
plant brings more diversity, but also more difference in framing. The
near-duplicates having already been removed at collection time (perceptual
hash, distance <= 6), the remaining photographs are visually distinct; the
figure is still an optimistic ceiling, not a promise.

Two ways of combining are compared: the mean of the probabilities, and the
geometric mean (the mean of the logarithms). The first forgives a bad
photograph, the second demands that both agree.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict

import numpy as np

from compare_models import load_model, predict

THRESHOLD = 0.70  # the application's acceptance threshold


def test_groups(dataset, min_photos: int):
    """The test observations with enough photographs, with their species."""
    groups = defaultdict(list)
    meta = {}
    with open(dataset / 'splits.csv', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['split'] != 'test':
                continue
            path = dataset / row['path']
            if path.exists():
                groups[row['group']].append(str(path))
                meta[row['group']] = (row['internal_plant_id'], row.get('captive') == '1')
    return {g: (paths, *meta[g]) for g, paths in groups.items() if len(paths) >= min_photos}


def combine(probs: list[np.ndarray], geometric: bool) -> np.ndarray:
    """Merge the opinions of several photographs into one."""
    stack = np.stack(probs)
    if geometric:
        # The mean of the logarithms: a species one of the photographs judges
        # very improbable does not come back up, even if the other is sure.
        return np.exp(np.mean(np.log(np.clip(stack, 1e-9, None)), axis=0))
    return stack.mean(axis=0)


def like_app(probs: list[np.ndarray], top_k: int, cut: float = 0.01) -> np.ndarray:
    """Exactly what the application can compute, and nothing more.

    The on-device model only returns the `top_k` best candidates above `cut`:
    the cascade never sees the whole vector. A species absent from a list is
    not worth zero there — it is worth *at most* the smallest score returned,
    and at most `cut`. It is given that bound, which penalises it without
    annulling it.

    The result is rescaled to the mass the input lists covered on average —
    not to 1. Averaging flattens the distribution and makes the first
    candidate's confidence fall, hence the acceptance rate, even as the
    ranking improves; so it must be straightened. But bringing it back to 1
    would manufacture confidence: two photographs returning a single candidate
    each, at 0.30 then 0.90, would come out at 1.00. With the mean mass they
    come out at 0.60. Multiplying by a constant changes no ordering.
    """
    if len(probs) == 1:
        # A single photograph: the cascade returns the model's scores as they
        # are, without merging or renormalising. The measurement must do
        # exactly the same, or it calibrates a threshold for a computation the
        # application never runs.
        return probs[0]
    kept = []
    for p in probs:
        order = np.argsort(-p)[:top_k]
        order = [i for i in order if p[i] >= cut]
        floor = min(float(p[order[-1]]), cut) if order else cut
        kept.append(({int(i): float(p[i]) for i in order}, floor))
    union = sorted({i for d, _ in kept for i in d})
    if not union:
        return np.zeros_like(probs[0])
    merged = np.zeros_like(probs[0])
    for i in union:
        merged[i] = np.exp(np.mean([np.log(max(d.get(i, floor), 1e-9)) for d, floor in kept]))
    covered = float(np.mean([sum(d.values()) for d, _ in kept]))
    total = merged.sum()
    return np.clip(merged * (covered / total), 0.0, 1.0) if total > 0 else merged


def measure(cached, labels, k: int, geometric: bool, only_captive: bool | None = None,
            app_top_k: int | None = None, threshold: float = THRESHOLD,
            min_margin: float = 0.0) -> dict:
    seen = hit1 = hit3 = accepted = accepted_ok = 0
    for probs, truth, captive in cached:
        if only_captive is not None and captive != only_captive:
            continue
        if app_top_k is not None:
            merged = like_app(probs[:k], app_top_k)
        else:
            merged = combine(probs[:k], geometric) if k > 1 else probs[0]
        order = np.argsort(-merged)[:3]
        top = [labels[i] for i in order]
        seen += 1
        hit1 += top[0] == truth
        hit3 += truth in top
        second = merged[order[1]] if len(order) > 1 else 0.0
        if merged[order[0]] >= threshold and merged[order[0]] - second >= min_margin:
            accepted += 1
            accepted_ok += top[0] == truth
    if not seen:
        return {}
    return {
        'observations': seen,
        'top1': round(hit1 / seen, 4),
        'top3': round(hit3 / seen, 4),
        'accepted_rate': round(accepted / seen, 4),
        'precision_when_accepted': round(accepted_ok / accepted, 4) if accepted else None,
    }


def show(title: str, rows: list[tuple[str, dict]]) -> None:
    print(f'— {title}')
    base = rows[0][1]
    for label, r in rows:
        if not r:
            continue
        delta = f"  ({r['top1'] - base['top1']:+.4f})" if r is not base else ''
        print(f"   {label:24s} top1 {r['top1']}  top3 {r['top3']}  "
              f"threshold {THRESHOLD} → {r['accepted_rate']} accepted, "
              f"precision {r['precision_when_accepted']}{delta}")
    print()


def sweep(cached, labels, photos: int, only_captive: bool | None) -> None:
    """The table used to tune the acceptance rule.

    The shipped threshold (0.70) was calibrated on an earlier version, on one
    photograph. A wider model spreads its confidence over more candidates and
    the same threshold makes it too cautious. Merging several photographs
    moves the distribution again. A threshold does not transpose from one
    model to another: it is measured again.
    """
    who = 'cultivated plants' if only_captive else 'all species'
    print(f"— thresholds, {who} (the app's photographs are the left-hand ones)\n")
    print(f"   {'thresh':>6s} {'margin':>6s} | " + ' | '.join(f'{k} photo{"s" if k > 1 else " "}' for k in range(1, photos + 1)))
    print('   ' + '-' * (16 + 22 * photos))
    for threshold in (0.40, 0.50, 0.55, 0.60, 0.70, 0.80):
        for min_margin in (0.0, 0.25):
            cells = []
            for k in range(1, photos + 1):
                r = measure(cached, labels, k, False, only_captive=only_captive,
                            app_top_k=5, threshold=threshold, min_margin=min_margin)
                cells.append(f"{r['accepted_rate']:5.0%} at {r['precision_when_accepted'] or 0:5.1%}"
                             if r and r['accepted_rate'] else '        —       ')
            print(f'   {threshold:6.2f} {min_margin:6.2f} | ' + ' | '.join(cells))
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='../collection/dataset')
    ap.add_argument('--model', default='./out')
    ap.add_argument('--photos', type=int, default=3, help='how many photographs to combine at most')
    ap.add_argument('--sample', type=int, default=2500, help='observations drawn at random')
    ap.add_argument('--seed', type=int, default=20260905)
    args = ap.parse_args()

    from pathlib import Path
    model = load_model(Path(args.model))
    groups = test_groups(Path(args.dataset), args.photos)
    keys = sorted(groups)
    rng = random.Random(args.seed)
    if args.sample and len(keys) > args.sample:
        keys = rng.sample(keys, args.sample)
    print(f"model v{model['version']} — {len(keys)} observations of at least {args.photos} photographs\n")

    cached = []
    for n, key in enumerate(keys, 1):
        paths, truth, captive = groups[key]
        if truth not in model['index']:
            continue
        cached.append(([predict(model, p) for p in paths[:args.photos]], truth, captive))
        if n % 250 == 0:
            print(f'  {n}/{len(keys)} observations', flush=True)
    print()

    for geometric in (False, True):
        name = 'geometric mean' if geometric else 'mean of the probabilities'
        show(f'all species — {name}',
             [(f'{k} photo{"s" if k > 1 else ""}', measure(cached, model['labels'], k, geometric))
              for k in range(1, args.photos + 1)])
    for top_k in (5, 40):
        show(f'all species — as the app does (lists of {top_k}, renormalised)',
             [(f'{k} photo{"s" if k > 1 else ""}', measure(cached, model['labels'], k, False, app_top_k=top_k))
              for k in range(1, args.photos + 1)])
        show(f'cultivated plants — as the app does (lists of {top_k}, renormalised)',
             [(f'{k} photo{"s" if k > 1 else ""}',
               measure(cached, model['labels'], k, False, only_captive=True, app_top_k=top_k))
              for k in range(1, args.photos + 1)])
    show('cultivated plants — mean of the probabilities',
         [(f'{k} photo{"s" if k > 1 else ""}', measure(cached, model['labels'], k, False, only_captive=True))
          for k in range(1, args.photos + 1)])
    sweep(cached, model['labels'], args.photos, only_captive=True)
    sweep(cached, model['labels'], args.photos, only_captive=None)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
