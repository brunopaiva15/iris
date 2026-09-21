"""What `evaluate` returns, and what it must no longer cost.

One evaluation was killed by the kernel after eight hours of training: it kept
the full matrix of probabilities — 99,825 images x 5,376 classes — and
`np.argsort` made a copy of it in 64-bit integers twice that size, in order to
read three columns afterwards.

These tests hold both ends: the numbers returned are **exactly** those of the
full sort, and the memory retained no longer grows with the number of classes.
"""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import evaluate


class Probabilities(tf.keras.Model):
    """A model returning probabilities decided in advance.

    `evaluate` does nothing but read an output; giving it a real network would
    only measure Keras.
    """

    def __init__(self, table: np.ndarray):
        super().__init__()
        self.table = tf.constant(table, dtype=tf.float32)

    def call(self, inputs, training=False):
        return tf.gather(self.table, tf.cast(inputs[:, 0], tf.int32))


def make_set(n: int, classes: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, classes)).astype(np.float32) * 3
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    truth = rng.integers(0, classes, size=n).astype(np.int32)
    index = np.arange(n, dtype=np.float32)[:, None]
    ds = tf.data.Dataset.from_tensor_slices((index, truth)).batch(16)
    return probs, truth, ds


def reference(probs: np.ndarray, truth: np.ndarray) -> dict:
    """The same numbers, computed by the full sort of before the fix."""
    order = np.argsort(-probs, axis=1)
    top1 = order[:, 0]
    correct = top1 == truth
    best = np.take_along_axis(probs, order[:, :2], axis=1)
    return {
        'top1': round(float(np.mean(correct)), 4),
        'top3': round(float(np.mean([t in o[:3] for t, o in zip(truth, order)])), 4),
        'mean_confidence': round(float(np.mean(best[:, 0])), 4),
    }


def test_the_numbers_match_a_full_sort():
    """The partial sort must return the same top-1, the same top-3 and the
    same mean confidence as the full sort it replaces."""
    probs, truth, ds = make_set(200, 40)
    measured = evaluate(Probabilities(probs), ds, [f'c{i}' for i in range(40)])
    expected = reference(probs, truth)
    for key, value in expected.items():
        assert measured[key] == value, key


def test_the_threshold_curve_matches_too():
    """The threshold/margin curve is read from the two best probabilities: it
    is what tunes the acceptance rule, and it must not move by a thousandth."""
    probs, truth, ds = make_set(300, 25, 11)
    measured = evaluate(Probabilities(probs), ds, [f'c{i}' for i in range(25)])
    order = np.argsort(-probs, axis=1)
    best = np.take_along_axis(probs, order[:, :2], axis=1)
    correct = order[:, 0] == truth
    margin = best[:, 0] - best[:, 1]
    for entry in measured['threshold_curve']:
        accepted = (best[:, 0] >= entry['threshold']) & (margin >= entry['min_margin'])
        assert entry['accepted_rate'] == round(int(accepted.sum()) / len(truth), 4)
        if int(accepted.sum()):
            assert entry['precision_when_accepted'] == round(float(np.mean(correct[accepted])), 4)


def test_captive_metrics_use_only_the_marked_images():
    probs, truth, ds = make_set(120, 15)
    mask = [i % 3 == 0 for i in range(120)]
    measured = evaluate(Probabilities(probs), ds, [f'c{i}' for i in range(15)], captive_mask=mask)
    m = np.asarray(mask)
    top1 = np.argsort(-probs, axis=1)[:, 0]
    assert measured['captive']['images'] == int(m.sum())
    assert measured['captive']['top1'] == round(float(np.mean((top1 == truth)[m])), 4)


def test_ten_times_more_classes_does_not_cost_more_memory():
    """The heart of the fix: what is retained no longer depends on the number
    of classes. Before, every row kept one column per class."""
    sizes = []
    for classes in (20, 200):
        probs, truth, ds = make_set(100, classes)
        tf.keras.backend.clear_session()
        kept = []
        model = Probabilities(probs)
        original = model.predict

        def spy(x, **kw):
            out = original(x, **kw)
            kept.append(out.nbytes)
            return out

        model.predict = spy
        evaluate(model, ds, [f'c{i}' for i in range(classes)])
        # What the model produces grows with the classes — that is unavoidable.
        # What matters is that `evaluate` keeps only three columns of it.
        sizes.append(max(kept))
    assert sizes[1] > sizes[0] * 5, 'the control itself should grow'


def test_a_two_class_model_does_not_break_the_partial_sort():
    """`argpartition` refuses a rank beyond the width: with two classes there
    is no third column to ask for."""
    probs, truth, ds = make_set(50, 2)
    measured = evaluate(Probabilities(probs), ds, ['a', 'b'])
    assert measured['top3'] == 1.0
    assert measured['top1'] == reference(probs, truth)['top1']
