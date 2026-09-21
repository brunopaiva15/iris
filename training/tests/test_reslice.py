"""Reslicing a trained model without retraining it.

Two things can go wrong in silence and ship a model that answers beside the
point: the order of the kept classes, and the cutting of the head's columns. A
column off by one would give a model that works — it would simply return the
wrong name, without ever crashing.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reslice import cut, keep, read_masks


def test_kept_classes_follow_the_trained_order_not_the_requested_one():
    """`labels.txt` and the head's columns are read line for line, so the kept
    classes must follow the model's order, not the file's."""
    assert keep(['abies', 'betula', 'cedrus'], ['cedrus', 'abies']) == ['abies', 'cedrus']


def test_a_requested_class_the_model_never_learned_is_dropped():
    """A species the previous model named and this one never learned has no
    column behind it; keeping the label would ship a name with no output."""
    assert keep(['abies', 'betula'], ['abies', 'hoya-kerrii']) == ['abies']


def test_asking_for_everything_changes_nothing():
    everything = ['a', 'b', 'c']
    assert keep(everything, everything) == everything


def test_the_head_is_cut_by_columns_in_the_kept_order():
    kernel = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])          # 2 features x 3 classes
    bias = np.array([10.0, 20.0, 30.0])
    weights = [np.zeros((2, 2)), kernel, bias]    # one backbone array in front
    out = cut(weights, ['a', 'b', 'c'], ['a', 'c'])
    assert np.array_equal(out[-2], np.array([[1.0, 3.0], [4.0, 6.0]]))
    assert np.array_equal(out[-1], np.array([10.0, 30.0]))


def test_everything_before_the_head_is_copied_untouched():
    """The backbone is not concerned: same vision, fewer names."""
    backbone = [np.arange(6).reshape(2, 3), np.array([7.0, 8.0])]
    weights = backbone + [np.ones((2, 4)), np.zeros(4)]
    out = cut(weights, list('abcd'), ['b', 'd'])
    assert len(out) == len(weights)
    for before, after in zip(backbone, out[:-2]):
        assert before is after


def test_cutting_preserves_the_ratio_between_two_kept_logits():
    """The softmax of a truncated head is exp(zi) / sum over the kept ones, so
    the ranking between two kept species never moves — which is what makes
    reslicing equivalent to a renormalised mask."""
    kernel = np.array([[2.0, 9.0, 1.0]])
    bias = np.array([0.5, 0.0, 0.25])
    features = np.array([[3.0]])
    full = features @ kernel + bias
    smaller = cut([kernel, bias], ['a', 'b', 'c'], ['a', 'c'])
    resliced = features @ smaller[-2] + smaller[-1]
    assert np.allclose(resliced, full[:, [0, 2]])


def test_a_mask_is_read_as_name_equals_file(tmp_path):
    path = tmp_path / 'indoor.txt'
    path.write_text('monstera-deliciosa\nficus-lyrata\n', encoding='utf-8')
    assert read_masks([f'indoor={path}']) == {
        'indoor': ['monstera-deliciosa', 'ficus-lyrata'],
    }


def test_a_mask_without_a_name_is_refused(tmp_path):
    """"--mask file.txt" does not say which place it describes, and a mask
    without a place has nothing to write into `model.json`."""
    with pytest.raises(SystemExit):
        read_masks([str(tmp_path / 'indoor.txt')])


def test_two_masks_keep_their_own_lists(tmp_path):
    (tmp_path / 'a.txt').write_text('a\nb\n', encoding='utf-8')
    (tmp_path / 'b.txt').write_text('b\nc\n', encoding='utf-8')
    masks = read_masks([f'indoor={tmp_path / "a.txt"}', f'outdoor={tmp_path / "b.txt"}'])
    # A species can belong to both: these are contexts, not two exclusive
    # taxonomies.
    assert masks == {'indoor': ['a', 'b'], 'outdoor': ['b', 'c']}


def test_the_union_of_two_masks_is_what_a_union_model_exposes(tmp_path):
    """Without `--keep`, the union of the masks makes the list: saying it
    twice would invite the two lists to drift apart."""
    (tmp_path / 'a.txt').write_text('a\nb\n', encoding='utf-8')
    (tmp_path / 'b.txt').write_text('b\nc\n', encoding='utf-8')
    masks = read_masks([f'indoor={tmp_path / "a.txt"}', f'outdoor={tmp_path / "b.txt"}'])
    union = sorted({c for ids in masks.values() for c in ids})
    assert keep(['a', 'b', 'c', 'd'], union) == ['a', 'b', 'c']
