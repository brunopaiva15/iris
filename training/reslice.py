#!/usr/bin/env python3
"""Re-export a trained model keeping only some of its classes.

    python3 reslice.py --weights .cache/ckpt/fine.weights.h5 \\
        --labels ./out/labels.txt --keep ./shipped/labels.txt \\
        --dataset ../collection/dataset --out ./resliced --version 8

And for a union model, which keeps the classes of several places and records
which belong to which — `--keep` is then unnecessary, the union of the masks
makes the list:

    python3 reslice.py --weights … --labels … --dataset … --out … \\
        --mask indoor=../collection/mask_indoor.txt \\
        --mask outdoor=../collection/mask_outdoor.txt --version 9

**This is not retraining.** The learned weights are reused, only the columns
of the wanted species are kept in the last layer, and the model is exported
again. The computation is identical to what an application would do by masking
the outputs and renormalising the confidence: the softmax of a truncated layer
is `exp(zi) / sum_kept exp(zj)`, which is exactly what
`compare_models.py --restricted` measures.

Why do it here rather than in the application: measured on the same 6,000
images, the wide model scores 0.6543 on the previous version's classes and
0.5528 with all 5,259 of its outputs. The new species are not merely
recognised badly — they steal the answers of the old ones. The model does not
need relearning, it needs bounding, and bounding it here avoids freezing a
list of species into the application's code.

The shipped file shrinks accordingly: the head is 960 x classes x 2 bytes.

**Context masks are not a second cut.** A union model keeps the union of the
masks in its head and writes into `model.json` which classes belong to which
place; the application renormalises over the classes of the place at inference
time, which yields exactly what this model resliced on that mask would yield —
same formula, same result. Two files would cost two backbones for two last
layers.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def keep(all_classes: list[str], wanted: list[str]) -> list[str]:
    """The classes to keep, in the order of the trained model.

    Order matters: `labels.txt` and the columns of the head must match line
    for line. A requested class the model never learned is dropped — there is
    nothing to take from it, and saying so beats shipping a label with no
    output behind it.
    """
    requested = set(wanted)
    return [c for c in all_classes if c in requested]


def read_masks(arguments: list[str]) -> dict[str, list[str]]:
    """The context masks passed as `--mask name=file`.

    A mask is the list of the classes of one place, one per line, in the same
    vocabulary as `labels.txt`. Classes the model has not learned disappear by
    themselves at export time: `export_tflite` only writes the ones that are
    in the head.
    """
    masks: dict[str, list[str]] = {}
    for raw in arguments:
        name, _, path = raw.partition('=')
        if not name or not path:
            raise SystemExit(f'--mask expects "name=file", got "{raw}"')
        ids = Path(path).expanduser().read_text(encoding='utf-8').split()
        if not ids:
            raise SystemExit(f'empty mask: {path}')
        masks[name] = ids
    return masks


def cut(weights: list, classes: list[str], kept: list[str]) -> list:
    """The weights of the resliced model.

    `get_weights()` returns the arrays flat, in layer order; the head being
    last, its kernel and bias close the list. They alone are cut — by column,
    in the order of `kept`, so that column *i* of the head answers to line *i*
    of `labels.txt`. Everything else, that is the backbone, is copied
    untouched.
    """
    index = {c: i for i, c in enumerate(classes)}
    columns = [index[c] for c in kept]
    kernel, bias = weights[-2], weights[-1]
    return weights[:-2] + [kernel[:, columns], bias[columns]]


def reslice(model, classes: list[str], kept: list[str], dropout: float, backbone: str):
    """An identical model, down to its last layer."""
    import train
    smaller = train.build_model(len(kept), dropout, backbone)
    smaller.set_weights(cut(model.get_weights(), classes, kept))
    return smaller


def main() -> int:
    import train

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', required=True, help='fine.weights.h5 of the trained model')
    ap.add_argument('--labels', required=True, help='labels.txt of the trained model')
    ap.add_argument('--keep',
                    help='file listing the classes to keep, one per line — the labels.txt '
                         'of the shipped model does nicely; defaults to the union of --mask')
    ap.add_argument('--mask', action='append', default=[], metavar='NAME=FILE',
                    help='one context mask, "indoor=mask_indoor.txt"; '
                         'repeatable, written into model.json')
    ap.add_argument('--dataset', required=True, help='to re-evaluate and to name the species')
    ap.add_argument('--out', required=True)
    ap.add_argument('--version', default='8')
    ap.add_argument('--backbone', default='large', choices=list(train.BACKBONES))
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--input-size', type=int, default=320)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--ram-budget', type=float, default=5.0)
    args = ap.parse_args()

    train.set_input_size(args.input_size)
    classes = Path(args.labels).read_text(encoding='utf-8').split()
    masks = read_masks(args.mask)
    if args.keep:
        wanted = Path(args.keep).read_text(encoding='utf-8').split()
    elif masks:
        # With no explicit list, keep the union of the masks: that is exactly
        # what a union model must expose, and saying it twice would invite the
        # two lists to drift apart.
        wanted = sorted({c for ids in masks.values() for c in ids})
    else:
        raise SystemExit('either --keep or at least one --mask is required')
    kept = keep(classes, wanted)
    lost = sorted(set(wanted) - set(kept))
    print(f'{len(classes)} classes learned, {len(wanted)} requested, {len(kept)} kept')
    if lost:
        print(f'{len(lost)} requested that the model has not learned, ignored:')
        for c in lost:
            print(f'   {c}')
    if not kept:
        raise SystemExit('no class in common: wrong file?')

    print('loading the learned weights…', flush=True)
    model = train.build_model(len(classes), args.dropout, args.backbone)
    model.load_weights(args.weights)
    smaller = reslice(model, classes, kept, args.dropout, args.backbone)

    dataset = Path(args.dataset)
    rows, captive = train.read_splits(dataset)
    test_ds, _, test_paths = train.make_dataset(rows['test'], kept, args.batch, training=False,
                                                ram_budget_gb=args.ram_budget, preload=False)
    print(f'evaluating on {len(test_paths)} test images…', flush=True)
    metrics = train.evaluate(smaller, test_ds, kept,
                             captive_mask=[p in captive for p in test_paths])
    metrics['version'] = args.version

    meta = train.export_tflite(smaller, Path(args.out), kept,
                               train.species_names(dataset), metrics, masks=masks)
    print(f"\n{args.out}/plants.tflite — {meta['bytes'] / 1e6:.1f} MB, {len(kept)} classes")
    for name, ids in meta.get('masks', {}).items():
        print(f"  mask {name}: {len(ids)} classes")
    print(f"  top1 {metrics['top1']}  top3 {metrics['top3']}  macro_f1 {metrics['macro_f1']}")
    for e in meta['threshold_curve']:
        if e['min_margin'] == 0.25 and e['threshold'] in (0.6, 0.7, 0.8):
            print(f"  threshold {e['threshold']} margin 0.25 → {e['accepted_rate']:.1%} accepted, "
                  f"precision {e['precision_when_accepted']:.4f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
