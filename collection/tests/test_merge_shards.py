"""Merging collections run in parallel.

A shard only sees its own species. The merge must return a set identical to
that of a single-run collection: every manifest end to end, every image
folder, and the name resolutions of both sides — without leaving behind the
files the finalisation will redo (`splits.csv`, `stats.json`, the
attributions).
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from merge_shards import merge, merge_caches, move_into  # noqa: E402


def make_shard(root: Path, species: dict[str, list[str]], caches: dict | None = None) -> Path:
    root.mkdir(parents=True)
    with open(root / 'manifest.jsonl', 'w', encoding='utf-8') as f:
        for name, files in species.items():
            for checksum in files:
                f.write(json.dumps({'species': name, 'checksum': checksum,
                                    'path': f'{name}/{checksum}.jpg'}) + '\n')
    for name, files in species.items():
        (root / name).mkdir()
        for checksum in files:
            (root / name / f'{checksum}.jpg').write_bytes(b'jpeg')
    # What the shard's finalisation wrote, and what will be redone.
    (root / 'splits.csv').write_text('path,species\n')
    (root / 'stats.json').write_text('{}')
    (root / 'ATTRIBUTIONS.md').write_text('#\n')
    for name, content in (caches or {}).items():
        (root / name).write_text(json.dumps(content))
    return root


def test_the_merge_gathers_manifests_and_images(tmp_path):
    a = make_shard(tmp_path / 'a', {'Monstera_deliciosa': ['aa', 'bb']})
    b = make_shard(tmp_path / 'b', {'Aloe_vera': ['cc']})
    out = tmp_path / 'dataset'

    counts = merge([a, b], out)

    assert counts['lines'] == 3
    lines = [json.loads(l) for l in (out / 'manifest.jsonl').read_text().splitlines()]
    assert {r['checksum'] for r in lines} == {'aa', 'bb', 'cc'}
    assert sorted(p.name for p in out.rglob('*.jpg')) == ['aa.jpg', 'bb.jpg', 'cc.jpg']


def test_the_merge_does_not_copy_what_will_be_redone(tmp_path):
    a = make_shard(tmp_path / 'a', {'Monstera_deliciosa': ['aa']})
    out = tmp_path / 'dataset'

    merge([a], out)

    for name in ('splits.csv', 'stats.json', 'ATTRIBUTIONS.md'):
        assert not (out / name).exists(), f'{name} comes from a shard, it must be redone over the whole set'


def test_the_status_folders_rejoin(tmp_path):
    """`_rejected` exists in every shard: the second must not overwrite the
    first, nor make the merge fail."""
    a = make_shard(tmp_path / 'a', {'Monstera_deliciosa': ['aa'], '_rejected': ['x']})
    b = make_shard(tmp_path / 'b', {'Aloe_vera': ['cc'], '_rejected': ['y']})
    out = tmp_path / 'dataset'

    merge([a, b], out)

    assert sorted(p.name for p in (out / '_rejected').iterdir()) == ['x.jpg', 'y.jpg']


def test_the_name_resolutions_of_both_shards_are_kept(tmp_path):
    a = make_shard(tmp_path / 'a', {'Monstera_deliciosa': ['aa']},
                   {'species.json': {'Monstera deliciosa': {'key': 1}}})
    b = make_shard(tmp_path / 'b', {'Aloe_vera': ['cc']},
                   {'species.json': {'Aloe vera': {'key': 2}}, 'species_inat.json': {'Aloe vera': {'id': 9}}})
    out = tmp_path / 'dataset'
    out.mkdir()

    sizes = merge_caches([a, b], out)

    assert sizes['species.json'] == 2
    assert json.loads((out / 'species.json').read_text())['Aloe vera']['key'] == 2
    assert json.loads((out / 'species_inat.json').read_text())['Aloe vera']['id'] == 9


def test_a_shard_without_a_manifest_stops_the_merge(tmp_path):
    """A shard interrupted before writing anything would produce a silently
    incomplete set. Better to stop."""
    a = make_shard(tmp_path / 'a', {'Monstera_deliciosa': ['aa']})
    empty = tmp_path / 'empty'
    empty.mkdir()

    with pytest.raises(SystemExit):
        merge([a, empty], tmp_path / 'dataset')


def test_a_species_present_in_two_shards_does_not_lose_its_images(tmp_path):
    """Two catalogue names can give the same folder — `Citrus × sinensis`
    appeared there twice — and the sharding sends them to different shards.
    Overwriting the folder would lose half the images."""
    a = make_shard(tmp_path / 'a', {'Citrus_x_sinensis': ['aa', 'bb']})
    b = make_shard(tmp_path / 'b', {'Citrus_x_sinensis': ['cc']})
    out = tmp_path / 'dataset'

    merge([a, b], out)

    assert sorted(p.name for p in (out / 'Citrus_x_sinensis').iterdir()) == ['aa.jpg', 'bb.jpg', 'cc.jpg']


def test_two_files_with_the_same_name_are_the_same_image(tmp_path):
    """A file's name is the start of its hash: two shards that downloaded the
    same image write the same name. Overwriting is a no-op, and the merge must
    not complain about it."""
    a = make_shard(tmp_path / 'a', {'Citrus_x_sinensis': ['aa']})
    b = make_shard(tmp_path / 'b', {'Citrus_x_sinensis': ['aa']})
    out = tmp_path / 'dataset'

    merge([a, b], out)

    assert [p.name for p in (out / 'Citrus_x_sinensis').iterdir()] == ['aa.jpg']


def test_the_status_subfolders_merge_in_depth(tmp_path):
    """`_rejected/<species>/<image>`: the collision is two levels deep."""
    a = tmp_path / 'a'
    make_shard(a, {'Monstera_deliciosa': ['aa']})
    (a / '_rejected' / 'Rosa_x_hybrida').mkdir(parents=True)
    (a / '_rejected' / 'Rosa_x_hybrida' / 'x.jpg').write_bytes(b'jpeg')
    b = tmp_path / 'b'
    make_shard(b, {'Aloe_vera': ['cc']})
    (b / '_rejected' / 'Rosa_x_hybrida').mkdir(parents=True)
    (b / '_rejected' / 'Rosa_x_hybrida' / 'y.jpg').write_bytes(b'jpeg')
    out = tmp_path / 'dataset'

    merge([a, b], out)

    assert sorted(p.name for p in (out / '_rejected' / 'Rosa_x_hybrida').iterdir()) == ['x.jpg', 'y.jpg']


def test_move_into_counts_files_not_folders(tmp_path):
    source, target = tmp_path / 's', tmp_path / 't'
    (source / 'sub').mkdir(parents=True)
    (source / 'sub' / 'a.jpg').write_bytes(b'1')
    (source / 'sub' / 'b.jpg').write_bytes(b'2')
    target.mkdir()
    (target / 'sub').mkdir()

    assert move_into(source, target) == 2
    assert not source.exists()
