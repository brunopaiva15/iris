#!/usr/bin/env python3
"""Publish a trained Iris model to the Hugging Face Hub.

    python3 scripts/push_to_hub.py --model ../plant/assets/model
    python3 scripts/push_to_hub.py --model ../plant/assets/model --dry-run
    python3 scripts/push_to_hub.py --model ../plant/assets/model --private

What it uploads: the three files an export produces - `plants.tflite`,
`labels.txt`, `model.json` - plus `model-card/README.md`, which becomes the
repository's front page. Nothing else; the training code lives on GitHub.

Before uploading it checks that the weights match the SHA-256 that
`model.json` claims, that `labels.txt` has as many lines as `model.json`
declares classes, and that the card's version matches the model's. A model
card that disagrees with its own weights is the one failure mode nobody
notices for months.

Authentication: `hf auth login`, or a `HF_TOKEN` environment variable holding
a token with write access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

FILES = ('plants.tflite', 'labels.txt', 'model.json')
DEFAULT_REPO = 'brunopaiva15/iris'


def check(model_dir: Path, card: Path) -> dict:
    """Refuse to publish a model that contradicts itself."""
    missing = [name for name in FILES if not (model_dir / name).exists()]
    if missing:
        raise SystemExit(f"{model_dir}: missing {', '.join(missing)}")
    if not card.exists():
        raise SystemExit(f'{card}: no model card to publish')

    meta = json.loads((model_dir / 'model.json').read_text(encoding='utf-8'))
    blob = (model_dir / 'plants.tflite').read_bytes()

    digest = hashlib.sha256(blob).hexdigest()
    if digest != meta['sha256']:
        raise SystemExit(f"plants.tflite is not the file model.json describes\n"
                         f"  model.json: {meta['sha256']}\n  on disk:    {digest}")
    if len(blob) != meta['bytes']:
        raise SystemExit(f"plants.tflite is {len(blob)} bytes, model.json says {meta['bytes']}")

    labels = (model_dir / 'labels.txt').read_text(encoding='utf-8').split()
    if len(labels) != meta['classes']:
        raise SystemExit(f"labels.txt has {len(labels)} lines, model.json declares {meta['classes']}")

    text = card.read_text(encoding='utf-8')
    if not re.search(rf"(?m)^# Iris {re.escape(str(meta['version']))}\b", text):
        raise SystemExit(f"the card's title is not '# Iris {meta['version']}' - "
                         f'update the card, or publish the matching model')
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True, type=Path,
                        help='directory holding plants.tflite, labels.txt and model.json')
    parser.add_argument('--repo', default=DEFAULT_REPO, help=f'Hub repo id (default {DEFAULT_REPO})')
    parser.add_argument('--card', type=Path, default=Path(__file__).resolve().parent.parent / 'model-card' / 'README.md')
    parser.add_argument('--private', action='store_true', help='create the repo private')
    parser.add_argument('--tag', help='also tag this upload, e.g. v9')
    parser.add_argument('--message', help='commit message (default: "Iris <version>")')
    parser.add_argument('--dry-run', action='store_true', help='check everything, upload nothing')
    args = parser.parse_args()

    meta = check(args.model, args.card)
    version = meta['version']
    print(f"Iris {version}: {meta['classes']} classes, {meta['bytes']:,} bytes, "
          f"top-1 {meta['metrics']['top1']}, sha256 ok")

    if args.dry_run:
        print(f'dry run - would upload {", ".join(FILES)} and the card to {args.repo}')
        return 0

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo, repo_type='model', private=args.private, exist_ok=True)

    with __import__('tempfile').TemporaryDirectory() as tmp:
        staging = Path(tmp)
        for name in FILES:
            (staging / name).write_bytes((args.model / name).read_bytes())
        (staging / 'README.md').write_text(args.card.read_text(encoding='utf-8'), encoding='utf-8')
        api.upload_folder(repo_id=args.repo, folder_path=str(staging), repo_type='model',
                          commit_message=args.message or f'Iris {version}')

    if args.tag:
        api.create_tag(args.repo, tag=args.tag, repo_type='model', exist_ok=True)

    print(f'https://huggingface.co/{args.repo}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
