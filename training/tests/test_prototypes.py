"""The cultivar storey stands or falls on one question: does the embedding
separate two cultivars of the same species?

The doubt is well founded. Every photograph of "Thai Constellation" in the set
is labelled *Monstera deliciosa*: fine-tuning pushes the embedding to make the
variegated cultivar and the ordinary plant converge. These tests check that the
measurement would tell the truth either way — that it sees the separation when
it exists, and the void when there is nothing.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prototypes import control, normalise, prototypes, separability  # noqa: E402


def _cluster(centre, n, noise=0.02, seed=0):
    r = np.random.default_rng(seed)
    return np.asarray(centre, dtype=np.float32) + r.normal(0, noise, (n, len(centre))).astype(np.float32)


def test_well_separated_cultivars_give_a_clear_gap():
    v = np.vstack([_cluster([1, 0, 0], 5, seed=1), _cluster([0, 1, 0], 5, seed=2)])
    r = separability(v, ['Thai'] * 5 + ['Albo'] * 5)
    assert r['gap'] > 0.9
    assert r['prototype_accuracy'] == 1.0


def test_on_noise_the_permutation_test_does_not_claim_victory():
    # The real null: labels drawn **independently** of the vectors. (Cutting a
    # noise sample into two contiguous halves is not a null: one half always
    # has a slightly different mean from the other, and the test rightly
    # detects it.)
    ps = []
    for seed in range(5):
        r = np.random.default_rng(100 + seed)
        v = _cluster([1, 0, 0], 12, noise=0.3, seed=seed)
        labels = list(r.choice(['Thai', 'Albo'], size=12))
        ps.append(control(v, labels, shuffles=150)['p_prototype'])
    assert sorted(ps)[len(ps) // 2] > 0.05, f'without structure the p-value must not be small: {ps}'


def test_a_real_separation_goes_down_to_the_floor():
    v = np.vstack([_cluster([1, 0, 0], 6, seed=1), _cluster([0, 1, 0], 6, seed=2)])
    labels = ['Thai'] * 6 + ['Albo'] * 6
    t = control(v, labels, shuffles=200)
    assert t['p_prototype'] <= 0.01
    assert t['mean_accuracy'] < 0.8, 'the shuffle must be clearly worse'


def test_the_accuracy_really_leaves_one_photograph_out():
    # With a single example per cultivar, a prototype computed over every
    # photograph would find them all; leaving this one out, it has nothing
    # left to attach it to.
    v = np.vstack([_cluster([1, 0, 0], 1, seed=4), _cluster([0, 1, 0], 1, seed=5)])
    r = separability(v, ['Thai', 'Albo'])
    assert r['classifiable'] == 0 and r['prototype_accuracy'] is None


def test_the_prototype_is_the_normalised_mean():
    p = prototypes(np.asarray([[2.0, 0, 0], [0, 2.0, 0]]), ['a', 'a'])
    assert set(p) == {'a'}
    assert abs(np.linalg.norm(p['a']) - 1.0) < 1e-5
    assert abs(p['a'][0] - p['a'][1]) < 1e-5, 'both directions weigh the same'


def test_one_cultivar_per_prototype_and_not_one_more():
    p = prototypes(np.eye(3, dtype=np.float32), ['Thai', 'Albo', 'Thai'])
    assert sorted(p) == ['Albo', 'Thai']


def test_normalisation_does_not_divide_by_zero():
    assert np.all(np.isfinite(normalise(np.zeros((2, 4), dtype=np.float32))))
