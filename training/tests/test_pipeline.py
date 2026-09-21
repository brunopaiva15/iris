"""Shuffling the image list, before it reaches tf.data.

The set is written sorted by species. Without a global shuffle, a buffer of a
few thousand elements covers only a handful of species, every batch becomes
nearly single-species, and the model learns to guess among ten classes.
"""
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import SHUFFLE_SEED, read_splits  # noqa: E402


def species_sorted_pairs(species: int = 500, per_species: int = 100):
    """Like splits.csv: all the images of one species follow each other."""
    return [(f'{s:03d}/{i}.jpg', s) for s in range(species) for i in range(per_species)]


def shuffle_like_training(pairs):
    pairs = list(pairs)
    random.Random(SHUFFLE_SEED).shuffle(pairs)
    return pairs


def distinct_species_in_window(pairs, window: int) -> int:
    return len({label for _, label in pairs[:window]})


def test_unshuffled_order_would_starve_a_shuffle_buffer():
    """The observation that motivated the fix: without shuffling, a buffer of
    4,096 elements sees only a few dozen species out of 500."""
    assert distinct_species_in_window(species_sorted_pairs(), 4096) <= 45


def test_shuffling_spreads_species_across_the_buffer():
    pairs = shuffle_like_training(species_sorted_pairs())
    assert distinct_species_in_window(pairs, 4096) > 450


def test_a_batch_is_no_longer_almost_single_species():
    pairs = shuffle_like_training(species_sorted_pairs())
    batch = pairs[:32]
    counts = Counter(label for _, label in batch)
    assert len(counts) >= 28, f'{len(counts)} species in a batch of 32'
    assert counts.most_common(1)[0][1] <= 3


def test_shuffle_keeps_every_image_exactly_once():
    original = species_sorted_pairs(50, 20)
    shuffled = shuffle_like_training(original)
    assert sorted(shuffled) == sorted(original)
    assert len(shuffled) == len(original)


def test_shuffle_is_reproducible():
    a = shuffle_like_training(species_sorted_pairs(50, 20))
    b = shuffle_like_training(species_sorted_pairs(50, 20))
    assert a == b, 'two runs must start from the same order'


def test_read_splits_returns_the_cultivated_photos(tmp_path):
    (tmp_path / 'Yucca_gigantea').mkdir()
    for name in ('a.jpg', 'b.jpg', 'c.jpg'):
        (tmp_path / 'Yucca_gigantea' / name).write_bytes(b'x')
    (tmp_path / 'splits.csv').write_text(
        'path,species,internal_plant_id,split,group,captive\n'
        'Yucca_gigantea/a.jpg,Yucca gigantea,yucca-gigantea,test,g1,1\n'
        'Yucca_gigantea/b.jpg,Yucca gigantea,yucca-gigantea,test,g2,0\n'
        'Yucca_gigantea/c.jpg,Yucca gigantea,yucca-gigantea,train,g3,1\n'
        'Yucca_gigantea/absent.jpg,Yucca gigantea,yucca-gigantea,test,g4,1\n'
    )
    rows, captive = read_splits(tmp_path)
    assert [p.endswith('a.jpg') or p.endswith('b.jpg') for p, _ in rows['test']] == [True, True]
    assert {p.rsplit('/', 1)[1] for p in captive} == {'a.jpg', 'c.jpg'}, 'absent from disk: ignored'
