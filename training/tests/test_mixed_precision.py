"""The float16 → float32 switch before the export.

Mixed precision doubles throughput on a card with tensor cores, but the TFLite
converter cannot convert a float16 graph: it asks for "flex ops" the
application does not embed. So the network is rebuilt in float32 and the
learned weights are put back into it.

This is a weight transfer between two distinct objects: were it to shift by
one layer, the exported model would stay plausible — same size, same
well-formed outputs — and be wrong. Hence these tests.
"""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import IMAGE_SIZE, build_model  # noqa: E402


def teardown_function():
    tf.keras.mixed_precision.set_global_policy('float32')


def test_the_head_stays_in_float32_even_under_mixed_precision():
    """A float16 softmax overflows as soon as the logits pass ~11, and those
    probabilities are what the application uses as a threshold."""
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    model = build_model(5, 0.3, 'small')
    assert model.get_layer('species').dtype_policy.name == 'float32'
    assert model.outputs[0].dtype == tf.float32


def test_the_network_rebuilt_in_float32_returns_the_same_outputs():
    """The heart of the switch: same weights, same answers."""
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    mixed = build_model(5, 0.0, 'small')
    weights = mixed.get_weights()

    tf.keras.mixed_precision.set_global_policy('float32')
    plain = build_model(5, 0.0, 'small')
    plain.set_weights(weights)

    image = tf.random.stateless_uniform([2, IMAGE_SIZE, IMAGE_SIZE, 3], seed=[3, 7], maxval=255)
    gap = np.abs(mixed(image, training=False).numpy() - plain(image, training=False).numpy()).max()
    # The intermediate computation differs (float16 against float32); the
    # probabilities must coincide to float16 precision.
    assert gap < 2e-2, f'the two networks do not answer the same thing (gap {gap})'


def test_the_transferred_weights_really_are_the_same():
    """A one-layer shift would pass the output test on an untrained network;
    here the tensors are compared one by one."""
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    mixed = build_model(5, 0.0, 'small')
    weights = mixed.get_weights()

    tf.keras.mixed_precision.set_global_policy('float32')
    plain = build_model(5, 0.0, 'small')
    plain.set_weights(weights)

    for before, after in zip(weights, plain.get_weights()):
        assert before.shape == after.shape
        assert np.array_equal(before, after)
    assert all(w.dtype == np.float32 for w in plain.get_weights()), 'the exported weights must be float32'
