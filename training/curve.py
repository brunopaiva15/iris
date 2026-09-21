#!/usr/bin/env python3
"""What every exposed species costs the others.

    python3 curve.py --model ./wide --dataset ../collection/dataset \\
        --core ./shipped/labels.txt \\
        --order ../collection/candidates_v8.txt \\
        --sizes 1444,1800,2200,3000,4000,5259

Two ends of the measurement are known: the wide model returns 0.6543 on the
core species when it exposes only them, and 0.5528 when it exposes 5,259. Ten
points, but the **shape** in between is unknown — the knee may be at 1,600 or
at 4,000, and it is the knee that says how many species one can afford.

**One inference pass is enough for the whole curve.** Masking a softmax down
to the kept classes then renormalising gives exactly what a model resliced to
those classes would return (`reslice.py`): `exp(zi) / sum kept`. So the 5,259
probabilities are computed once per image, and each set size is then only an
addition. What used to cost an hour per point costs a few seconds.

Two columns, and both are needed:

- **the cost** — top-1 on the images of the core species, the same ones at
  every size. This is what the current user loses when the set is widened;
- **the gain** — top-1 on the images of the added species, which the previous
  set could not name at all.

And two tables: all the photographs, then the photographs of **cultivated**
plants only (`captive`). The cost of breadth was measured at 10.2 points in
general and **11.1 on potted plants** — the application's real domain. The
knee may be earlier there than elsewhere, and that is where it must be read.

The sets are **nested**: every size contains the previous one. Without that,
two points of the curve would not compare.

**`--by-theft`: add by cost, not by priority.** The first curve showed that
the cost of breadth is not a function of *how many* species are added but of
*which*: a species that never comes ahead of the right answer on a core image
is free, a species that resembles one costs on every photograph. So for each
candidate, the number of core images it would flip from right to wrong if it
were exposed is counted — its **theft cost** — and candidates are added by
increasing cost, the cultivation priority only breaking ties.

The selection runs on the **validation** split and the measurement on the
**test** split: choosing and measuring on the same images would flatter the
result. One extra size is added to the curve as a matter of course: the core
plus every species that steals nothing in validation.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'collection'))


def subsets(core: list[str], order: list[str], everything: list[str],
            sizes: list[int]) -> dict[int, list[str]]:
    """One exposed set per requested size, nested from smallest to largest.

    The core first — the species already served, never removed — then the next
    ones in the priority order given, then, if the size still demands it, the
    rest in the model's order. A size smaller than the core is raised to the
    core: the point is to measure what is added, not to amputate what works.
    """
    known = set(everything)
    base = [c for c in everything if c in set(core) & known]
    seen = set(base)
    rest = [c for c in order if c in known and c not in seen]
    seen.update(rest)
    rest += [c for c in everything if c not in seen]

    out = {}
    for n in sorted(set(sizes)):
        take = max(n - len(base), 0)
        out[max(n, len(base))] = base + rest[:take]
    return out


def theft_costs(P: np.ndarray, truth: np.ndarray, core: np.ndarray,
                candidates: np.ndarray) -> tuple[np.ndarray, int]:
    """For each candidate, the number of core images it would flip from right
    to wrong if it were exposed alongside the core.

    Only the images the core alone gets right can count: an image already
    wrong cannot become more wrong, and a candidate is never the truth of a
    core image. On a right image, the core's best score *is* the truth's; the
    candidate steals if it exceeds it.

    `P`: one row per image, one column per class of the model. `truth`,
    `core`, `candidates`: column indices. Also returns the number of right
    images, so the costs can be read as proportions.
    """
    core_argmax = core[np.argmax(P[:, core], axis=1)]
    right = core_argmax == truth
    threshold = P[np.arange(len(P)), truth][right]
    return (P[right][:, candidates] > threshold[:, None]).sum(axis=0), int(right.sum())


def order_by_theft(costs: dict[str, int], priority: list[str]) -> list[str]:
    """The candidates by increasing cost; at equal cost, the cultivation
    priority, then the identifier so that two runs look alike."""
    rank = {c: i for i, c in enumerate(priority)}
    return sorted(costs, key=lambda c: (costs[c], rank.get(c, len(rank)), c))


def read_order(path: Path | None) -> list[str]:
    """The internal ids, in the priority order of the file.

    `candidates_v8.txt` holds scientific names, sorted by what people grow:
    that is the order in which species should be added, not the model's
    alphabetical order.
    """
    if path is None:
        return []
    from plant_dataset.taxonomy import internal_id
    names = [l.strip() for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
    return [internal_id(n) for n in names]


def main() -> int:
    from compare_models import load_model, predict_rows, read_rows, read_test, tally

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, help='the wide model, the one that learned everything')
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--core', default='./shipped/labels.txt',
                    help='the species already served, never removed')
    ap.add_argument('--order', help='priority order of the additions (scientific names)')
    ap.add_argument('--sizes', default='1444,1800,2200,2800,3600,5259')
    ap.add_argument('--sample', type=int, default=6000,
                    help='core test images; the other species get half as many')
    ap.add_argument('--seed', type=int, default=20260905)
    ap.add_argument('--threshold', type=float, default=0.70)
    ap.add_argument('--by-theft', action='store_true',
                    help='order the additions by theft cost measured on validation, '
                         'the cultivation priority only breaking ties')
    ap.add_argument('--val-sample', type=int, default=12000,
                    help='core validation images used to measure the costs; 0 = all of them')
    args = ap.parse_args()

    model = load_model(Path(args.model))
    everything = model['labels']
    core = Path(args.core).read_text(encoding='utf-8').split()
    base = set(core) & set(everything)
    priority = read_order(Path(args.order) if args.order else None)
    sizes = [int(x) for x in args.sizes.split(',')]

    if args.by_theft:
        # A separate generator: the test draw must remain the first call on
        # `Random(args.seed)`, or the safeguard below falls.
        val = [(p, t) for p, t, _ in read_rows(Path(args.dataset), 'val') if t in base]
        if args.val_sample and len(val) > args.val_sample:
            val = random.Random(args.seed + 1).sample(val, args.val_sample)
        print(f'theft costs: {len(val)} core images in validation…', flush=True)
        pv = predict_rows(val, model)
        P = np.stack([p for _, p in pv])
        truth = np.array([model['index'][t] for t, _ in pv])
        core_idx = np.array(sorted(model['index'][c] for c in base))
        cand_ids = [c for c in everything if c not in base]
        thefts, right = theft_costs(P, truth, core_idx,
                                    np.array([model['index'][c] for c in cand_ids]))
        costs = dict(zip(cand_ids, (int(v) for v in thefts)))
        priority = order_by_theft(costs, priority)
        bands = [(0, 'steal nothing'), (2, 'steal <= 2'), (5, '<= 5'), (20, '<= 20')]
        print(f'  {right} images right with the core alone; out of {len(costs)} candidates, '
              + ', '.join(f'{sum(1 for v in costs.values() if v <= n)} {word}' for n, word in bands)
              + f', {sum(1 for v in costs.values() if v > 20)} more than 20')
        free = sum(1 for v in costs.values() if v == 0)
        sizes.append(len(base) + free)
        print(f'  → size added to the curve: {len(base) + free} '
              f'(the core plus the {free} free ones)\n')

    sets = subsets(core, priority, everything, sizes)
    rows = [(p, t, c) for p, t, c in read_test(Path(args.dataset)) if t in model['index']]

    # The core draw reproduces the one in `compare_models.py` — same seed,
    # same filter, same first call — hence **the same images**. The core-only
    # line must then land on the published figure to the fourth decimal: a
    # free safeguard, and the only way to know the rest of the curve is
    # readable.
    rng = random.Random(args.seed)
    in_core = [r for r in rows if r[1] in base]
    elsewhere = [r for r in rows if r[1] not in base]
    if args.sample:
        if len(in_core) > args.sample:
            in_core = rng.sample(in_core, args.sample)
        if len(elsewhere) > args.sample // 2:
            elsewhere = rng.sample(elsewhere, args.sample // 2)
    rows = in_core + elsewhere
    print(f"{len(everything)} classes learned, {len(base)} in the core\n"
          f"{len(in_core)} core images, {len(elsewhere)} from the other species\n")

    print('inference, once and for all…', flush=True)
    pred = predict_rows([(p, t) for p, t, _ in rows], model)
    # `predict_rows` only filters out truths unknown to the model, already
    # removed above: the flags stay aligned with the outputs.
    assert len(pred) == len(rows)
    potted = [c for _, _, c in rows]

    def cell(value) -> str:
        return '—' if value is None else f'{value:.4f}'

    def table(title: str, outputs: list) -> None:
        core_pred = [(t, p) for t, p in outputs if t in base]
        print(f"\n— {title}: {len(core_pred)} core images, "
              f"{len(outputs) - len(core_pred)} from the other species")
        print(f"{'exposed':>9}  {'core: top1':>12} {'top3':>7} {'autonomy':>10} {'precision':>9}"
              f"   {'added: top1':>16} {'images':>7}")
        for n in sorted(sets):
            kept = set(sets[n])
            added_pred = [(t, p) for t, p in outputs if t in kept and t not in base]
            c = tally(core_pred, model, kept, renormalise=True, threshold=args.threshold)
            a = (tally(added_pred, model, kept, renormalise=True, threshold=args.threshold)
                 if added_pred else {})
            print(f'{n:>9}  {cell(c["top1"]):>12} {cell(c["top3"]):>7} '
                  f'{cell(c["accepted_rate"]):>10} {cell(c["precision_when_accepted"]):>9}   '
                  f'{cell(a.get("top1")):>16} {a.get("images", 0):>7}')

    table('all photographs', pred)
    table("photographs of cultivated plants — the application's domain",
          [tp for tp, c in zip(pred, potted) if c])
    print(f'\ncore: {len(base)} species, the same ones on every line. '
          'The core-only line of the first table must land on the published figure.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
