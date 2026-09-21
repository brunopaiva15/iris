#!/usr/bin/env python3
"""Identify a plant from one or more photographs with Iris 9.

    python3 identify.py photo.jpg
    python3 identify.py leaf.jpg whole.jpg --context indoor
    python3 identify.py photo.jpg --model ./local-copy   # instead of the Hub

The model is downloaded from the Hugging Face Hub on first use and cached.
Everything after that runs offline.

What this script demonstrates, beyond a bare inference call:

- the exact preprocessing recipe the model was trained with, which is part of
  the model and costs several points of top-1 when it is approximated;
- the context mask, which renormalises the softmax over the species that can
  plausibly live where the photograph was taken;
- the acceptance rule, which decides whether an answer is good enough to be
  shown as "probably this" rather than as a list of guesses;
- averaging several photographs of the same plant, which is the cheapest
  accuracy there is.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

REPO_ID = 'brunopaiva15/iris'
FILES = ('plants.tflite', 'labels.txt', 'model.json')

# The app's acceptance rule. These three numbers belong to this version of the
# model: `model.json` carries the full threshold curve, and a threshold read
# from one model does not describe another.
ACCEPT_THRESHOLD = 0.70
MIN_MARGIN = 0.25
FLOOR = 0.10


def load_model(source: str | None):
    """Return (interpreter, labels, meta). `source` is a local directory, or
    None to fetch the model from the Hugging Face Hub."""
    if source:
        folder = Path(source)
        paths = {name: folder / name for name in FILES}
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise SystemExit('missing file(s): ' + ', '.join(missing))
    else:
        from huggingface_hub import hf_hub_download
        paths = {name: Path(hf_hub_download(REPO_ID, name)) for name in FILES}

    interpreter = open_interpreter(paths['plants.tflite'])
    interpreter.allocate_tensors()

    labels = paths['labels.txt'].read_text(encoding='utf-8').split()
    meta = json.loads(paths['model.json'].read_text(encoding='utf-8'))
    if len(labels) != meta['classes']:
        raise SystemExit(f"labels.txt has {len(labels)} lines, model.json declares {meta['classes']}")
    return interpreter, labels, meta


def open_interpreter(weights: Path):
    """The first TFLite runtime available, in order of preference.

    LiteRT is the current name of the runtime; `tflite_runtime` is the older
    standalone package, and full TensorFlow works as a last resort. A phone
    app uses none of these - it links the native library directly.
    """
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter(model_path=str(weights))
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter(model_path=str(weights))
    except ImportError:
        pass
    import tensorflow as tf
    return tf.lite.Interpreter(model_path=str(weights))


def prepare(path: Path, size: int, load_size: int, source_size: int) -> np.ndarray:
    """Decode and frame a photograph exactly as training did.

    Centre square, area-average down to `source_size`, bilinear to
    `load_size`, centre crop to `size`. The two-step reduction matters: the
    training images were stored at `source_size`, and resizing a 4,000 px
    phone photograph straight to 366 px produces aliasing the model has never
    seen.

    Values are returned in 0-255. The rescaling layer lives inside the graph;
    dividing by 255 here would halve the accuracy in silence.
    """
    image = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    if image.width > source_size > load_size:
        image = image.resize((source_size, source_size), Image.BOX)
    image = image.resize((load_size, load_size), Image.BILINEAR)
    offset = (load_size - size) // 2
    image = image.crop((offset, offset, offset + size, offset + size))
    return np.asarray(image, dtype=np.float32)[np.newaxis]


def infer(interpreter, batch: np.ndarray) -> np.ndarray:
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    interpreter.set_tensor(inp['index'], batch)
    interpreter.invoke()
    return interpreter.get_tensor(out['index'])[0]


def apply_mask(scores: np.ndarray, labels: list[str], meta: dict, context: str) -> np.ndarray:
    """Keep the species of one context and renormalise over them.

    Dropping columns from the last layer and re-applying softmax gives
    `exp(zi) / sum_kept exp(zj)`; dividing the kept probabilities by their
    mass computes the same thing. The mask is therefore not an approximation
    of a model trained on those classes alone - it equals one.
    """
    ids = (meta.get('masks') or {}).get(context)
    if not ids:
        return scores
    keep = np.zeros(len(labels), dtype=bool)
    index = {label: i for i, label in enumerate(labels)}
    for internal_id in ids:
        if internal_id in index:
            keep[index[internal_id]] = True
    mass = float(scores[keep].sum())
    if mass < 1e-6:          # nothing of this context in the photograph
        return scores
    masked = np.zeros_like(scores)
    masked[keep] = scores[keep] / mass
    return masked


def verdict(scores: np.ndarray) -> str:
    """What the app would do with this answer."""
    order = np.argsort(scores)[::-1]
    best = scores[order[0]]
    second = scores[order[1]] if len(order) > 1 else 0.0
    if best < FLOOR:
        return 'no candidate — nothing here looks like a plant this model knows'
    if best >= ACCEPT_THRESHOLD and best - second >= MIN_MARGIN:
        return 'accepted — good enough to show as the answer'
    return 'uncertain — show the list, or ask for a second photograph'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('photos', nargs='+', type=Path,
                        help='one to three photographs of the SAME plant')
    parser.add_argument('--context', choices=('indoor', 'outdoor', 'unknown'), default='unknown',
                        help='where the plant lives; masks and renormalises the output')
    parser.add_argument('--model', help='local directory holding the three model files')
    parser.add_argument('--top', type=int, default=5, help='how many candidates to print')
    args = parser.parse_args()

    interpreter, labels, meta = load_model(args.model)
    size, load_size = meta['input_size'], meta['load_size']
    source_size = meta.get('source_size', load_size)

    # Several photographs of one plant: average the probability vectors. On an
    # earlier version this was worth +13.7 points of top-1 for two photographs
    # and +22.4 for three, for no extra computation worth mentioning.
    scores = np.zeros(len(labels), dtype=np.float64)
    for photo in args.photos:
        scores += infer(interpreter, prepare(photo, size, load_size, source_size))
    scores /= len(args.photos)

    if args.context != 'unknown':
        scores = apply_mask(scores, labels, meta, args.context)

    names = meta.get('species', {})
    print(f"Iris {meta['version']} — {meta['classes']} species, "
          f"{len(args.photos)} photograph(s), context: {args.context}\n")
    for rank, i in enumerate(np.argsort(scores)[::-1][:args.top], start=1):
        print(f'{rank}. {names.get(labels[i], labels[i]):40s} {scores[i]:.4f}')
    print(f'\n{verdict(scores)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
