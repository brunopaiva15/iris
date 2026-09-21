"""The activation cache of the frozen network.

During the head phase the network does not move: its outputs are the same at
every epoch. Computing them once and reading them back is what makes the head
phase bearable without a GPU. Two things must hold: the cache is read back
when nothing has changed, and it is rebuilt as soon as the set or the classes
change — a stale cache would train the head on images other than the announced
ones.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import encoded, fit_head  # noqa: E402


def encoder(calls, n=64, dim=16, die_after=None):
    """A dummy encoder: writes vectors into the memmap file, as the real one
    does, and can stop halfway to exercise the resume."""
    def run(x_path, y_path, mark, done):
        calls.append(done)
        if x_path is None:
            rng = np.random.default_rng(0)
            return rng.random((n, dim), dtype=np.float32).astype(np.float16), np.arange(n), None
        x = np.lib.format.open_memmap(x_path, mode='r+' if done else 'w+',
                                      dtype=np.float16, shape=(n, dim))
        y = np.lib.format.open_memmap(y_path, mode='r+' if done else 'w+',
                                      dtype=np.int32, shape=(n,))
        stop = n if die_after is None else min(n, done + die_after)
        for i in range(done, stop):
            x[i] = np.float16(i)
            y[i] = i
        x.flush()
        y.flush()
        mark(stop)
        if stop < n:
            raise KeyboardInterrupt('the machine disappeared')
        return x, y, None
    return run


def test_the_cache_avoids_the_second_encoding(tmp_path):
    calls = []
    signature = {'backbone': 'large', 'images': 64}
    x1, y1 = encoded(tmp_path, 'train', signature, encoder(calls))
    x2, y2 = encoded(tmp_path, 'train', signature, encoder(calls))
    assert len(calls) == 1, 'the second call re-encoded instead of reading back'
    assert np.array_equal(np.asarray(x1), np.asarray(x2))
    assert np.array_equal(np.asarray(y1), np.asarray(y2))


def test_an_interrupted_encoding_resumes_where_it_stopped(tmp_path):
    """The case that cost an hour: the machine disappears mid-pass. What is
    written must stay acquired."""
    signature = {'backbone': 'large', 'images': 64}
    calls = []
    with pytest.raises(KeyboardInterrupt):
        encoded(tmp_path, 'train', signature, encoder(calls, die_after=40))
    assert json.loads((tmp_path / 'train.json').read_text())['done'] == 40

    resumed = []
    x, y = encoded(tmp_path, 'train', signature, encoder(resumed))
    assert resumed == [40], f'the resume restarted at {resumed}, not at 40'
    assert np.array_equal(np.asarray(y), np.arange(64)), 'vectors are missing or shuffled'


def test_the_cache_is_rebuilt_when_the_set_changes(tmp_path):
    calls = []
    encoded(tmp_path, 'train', {'backbone': 'large', 'images': 64}, encoder(calls))
    encoded(tmp_path, 'train', {'backbone': 'large', 'images': 65}, encoder(calls, n=65))
    assert calls == [0, 0], 'a stale cache was read back, or wrongly resumed'


def test_without_a_cache_folder_everything_is_encoded_every_time(tmp_path):
    calls = []
    encoded(None, 'train', {'images': 64}, encoder(calls))
    encoded(None, 'train', {'images': 64}, encoder(calls))
    assert len(calls) == 2


def test_the_head_weights_have_the_shape_of_the_full_model():
    """`fit_head` trains a head apart; its weights are put back as they are
    into the full model. The shapes must match exactly."""
    rng = np.random.default_rng(0)
    x, y = rng.random((80, 32), dtype=np.float32).astype(np.float16), rng.integers(0, 5, 80)
    xv, yv = rng.random((20, 32), dtype=np.float32).astype(np.float16), rng.integers(0, 5, 20)
    kernel, bias = fit_head(x, y, (xv, yv), 5, 0.3, 1, {i: 1.0 for i in range(5)}, 4)
    assert kernel.shape == (32, 5)
    assert bias.shape == (5,)
