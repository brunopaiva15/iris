"""The sample used to measure a machine.

What it must never do: produce a set from which `train.py` would drop classes.
A sample announced at 120 classes that teaches 80 would measure something other
than what is believed, and the mistake would be invisible — the throughput
would stay plausible.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sample import choose, main, read

MIN_TRAIN, MIN_VAL = 25, 3


def fake_set(tmp_path: Path, species: int = 5, big: int = 3) -> Path:
    """A dummy set: `big` well-fed species, the rest thin."""
    src = tmp_path / 'source'
    rows = []
    for n in range(species):
        cid = f'species-{n}'
        quotas = {'train': 80 if n < big else 10, 'val': 6, 'test': 6}
        for split, q in quotas.items():
            for i in range(q):
                rel = f'images/{cid}/{split}-{i}.jpg'
                p = src / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b'\xff\xd8' + bytes(400))
                rows.append([rel, f'Genus specie{n}', cid, split, f'g{n}-{i}', '1' if i % 2 else '0'])
    # A row announcing an image absent from disk: splits.csv holds some, and
    # `train.py` already ignores them (`read_splits` tests `path.exists()`).
    rows.append(['images/species-0/ghost.jpg', 'Genus specie0', 'species-0', 'train', 'gX', '0'])
    with open(src / 'splits.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['path', 'species', 'internal_plant_id', 'split', 'group', 'captive'])
        w.writerows(rows)
    return src


def build(tmp_path: Path, **kw) -> Path:
    src = fake_set(tmp_path, **{k: v for k, v in kw.items() if k in ('species', 'big')})
    out = tmp_path / 'sample'
    sys.argv = ['sample.py', '--dataset', str(src), '--out', str(out),
                '--classes', str(kw.get('classes', 4)), '--train', str(kw.get('train', 60)),
                '--val', '5', '--test', '5']
    assert main() == 0
    return out


def as_read(out: Path) -> dict[str, list[tuple[str, str]]]:
    """What `read_splits` of `train.py` would see, to the letter."""
    rows = {'train': [], 'val': [], 'test': []}
    with open(out / 'splits.csv', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if (out / r['path']).exists():
                rows[r['split']].append((r['path'], r['internal_plant_id']))
    return rows


def test_every_kept_class_survives_the_thresholds_of_train_py(tmp_path):
    rows = as_read(build(tmp_path))
    train, val = Counter(p for _, p in rows['train']), Counter(p for _, p in rows['val'])
    usable = [c for c in train if train[c] >= MIN_TRAIN and val[c] >= MIN_VAL]
    assert usable, 'no usable class'
    assert len(usable) == len(train), 'a copied class would be dropped by train.py'


def test_a_class_too_thin_is_left_out_rather_than_copied_short(tmp_path):
    """Copying a ten-image class only to see it dropped afterwards is disk
    spent for nothing."""
    rows = as_read(build(tmp_path, species=5, big=3, classes=5))
    classes = {p for _, p in rows['train']}
    assert classes == {'species-0', 'species-1', 'species-2'}


def test_a_line_without_its_file_is_skipped(tmp_path):
    out = build(tmp_path)
    assert not (out / 'images/species-0/ghost.jpg').exists()
    with open(out / 'splits.csv', newline='', encoding='utf-8') as f:
        assert all(r['path'] != 'images/species-0/ghost.jpg' for r in csv.DictReader(f))


def test_the_manifest_names_every_kept_class(tmp_path):
    """`train.py` reads the species names there at export time; a missing
    identifier would come out as it is in `model.json`."""
    out = build(tmp_path)
    names = {json.loads(l)['internal_plant_id'] for l in open(out / 'manifest.jsonl', encoding='utf-8')}
    assert names == {p for _, p in as_read(out)['train']}


def test_quotas_cap_each_split(tmp_path):
    rows = as_read(build(tmp_path, train=30))
    train = Counter(p for _, p in rows['train'])
    assert set(train.values()) == {30}
    assert set(Counter(p for _, p in rows['val']).values()) == {5}


def test_the_choice_is_deterministic(tmp_path):
    """Two machines must take the same sample, or they are not measuring the
    same thing."""
    src = fake_set(tmp_path)
    per_class = read(src)
    assert choose(per_class, 2, 60, 5, 5) == choose(per_class, 2, 60, 5, 5)


def test_impossible_thresholds_say_so(tmp_path):
    src = fake_set(tmp_path)
    sys.argv = ['sample.py', '--dataset', str(src), '--out', str(tmp_path / 'x'),
                '--train', '5000']
    with pytest.raises(SystemExit):
        main()
