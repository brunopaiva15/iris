"""The counting side of `compare_models`, without TensorFlow.

Inference is separate from counting (`predict_rows` / `tally`), and the
counting carries the subtlety: masking the classes the other model does not
know, or not, answers a different question, and one of the two numbers
flatters the wider model.
"""
import numpy as np
import pytest

from compare_models import tally


def model_of(labels: list[str], version: str = '8') -> dict:
    return {'labels': labels, 'index': {c: i for i, c in enumerate(labels)}, 'version': version}


def output(model: dict, scores: dict[str, float]) -> np.ndarray:
    p = np.zeros(len(model['labels']), dtype=np.float32)
    for c, v in scores.items():
        p[model['index'][c]] = v
    return p


def test_masking_does_not_change_the_ranking_of_what_remains():
    """The docstring claims it and everything else depends on it: masking does
    not reorder the classes that remain, it only removes some."""
    m = model_of(['monstera', 'ficus', 'unknown'])
    pred = [('monstera', output(m, {'monstera': 0.5, 'ficus': 0.3, 'unknown': 0.2}))]
    assert tally(pred, m, {'monstera', 'ficus'})['top1'] == 1.0
    assert tally(pred, m, None)['top1'] == 1.0


def test_a_new_class_can_steal_the_answer_when_nothing_is_masked():
    """The heart of the matter: a species the old model ignored can take first
    place. The mask removes it; the user does not."""
    m = model_of(['monstera', 'ficus', 'new-one'])
    pred = [('monstera', output(m, {'monstera': 0.35, 'ficus': 0.05, 'new-one': 0.6}))]
    assert tally(pred, m, {'monstera', 'ficus'})['top1'] == 1.0
    assert tally(pred, m, None)['top1'] == 0.0


def test_top3_counts_a_truth_ranked_third():
    m = model_of(['a', 'b', 'c', 'd'])
    pred = [('d', output(m, {'a': 0.4, 'b': 0.3, 'd': 0.2, 'c': 0.1}))]
    r = tally(pred, m, None)
    assert r['top1'] == 0.0 and r['top3'] == 1.0


def test_acceptance_uses_the_app_threshold_of_070():
    m = model_of(['a', 'b'])
    above = [('a', output(m, {'a': 0.71, 'b': 0.29}))]
    below = [('a', output(m, {'a': 0.69, 'b': 0.31}))]
    assert tally(above, m, None)['accepted_rate'] == 1.0
    assert tally(below, m, None)['accepted_rate'] == 0.0
    assert tally(below, m, None)['precision_when_accepted'] is None


def test_precision_when_accepted_counts_only_accepted_answers():
    m = model_of(['a', 'b'])
    pred = [('a', output(m, {'a': 0.9, 'b': 0.1})),      # accepted, right
            ('b', output(m, {'a': 0.8, 'b': 0.2})),      # accepted, wrong
            ('a', output(m, {'a': 0.6, 'b': 0.4}))]      # below the threshold, ignored
    r = tally(pred, m, None)
    assert r['accepted_rate'] == round(2 / 3, 4)
    assert r['precision_when_accepted'] == 0.5


def test_renormalising_raises_autonomy_without_touching_top1():
    """Without renormalising, the mass that went to the masked classes comes
    back to nobody and the measured autonomy is artificially low."""
    m = model_of(['a', 'b', 'masked'])
    pred = [('a', output(m, {'a': 0.5, 'b': 0.1, 'masked': 0.4}))]
    raw = tally(pred, m, {'a', 'b'})
    renormalised = tally(pred, m, {'a', 'b'}, renormalise=True)
    assert raw['top1'] == renormalised['top1'] == 1.0
    assert raw['accepted_rate'] == 0.0             # 0.5 stays under 0.70
    assert renormalised['accepted_rate'] == 1.0    # 0.5 / 0.6 = 0.83


def test_images_of_unknown_species_are_not_counted():
    """`predict_rows` drops them; `tally` sees only what it is given."""
    m = model_of(['a', 'b'])
    assert tally([], m, None)['images'] == 0
    assert tally([], m, None)['top1'] is None


@pytest.mark.parametrize('restrict', [None, {'a', 'b'}])
def test_tally_does_not_modify_the_probabilities_it_is_given(restrict):
    """Two readings run over the same inference pass: the first must not spoil
    the arrays the second is going to read again."""
    m = model_of(['a', 'b', 'c'])
    probs = output(m, {'a': 0.5, 'b': 0.2, 'c': 0.3})
    before = probs.copy()
    tally([('a', probs)], m, restrict, renormalise=True)
    assert np.array_equal(probs, before)


def test_the_threshold_is_a_parameter_not_a_constant():
    """Choosing the acceptance threshold means sweeping it: an answer at 0.62
    is refused at 0.70 and accepted at 0.60, and that is the whole point of
    the tuning."""
    m = model_of(['a', 'b'])
    pred = [('a', output(m, {'a': 0.62, 'b': 0.38}))]
    assert tally(pred, m, None, threshold=0.70)['accepted_rate'] == 0.0
    assert tally(pred, m, None, threshold=0.60)['accepted_rate'] == 1.0
    # The threshold does not touch the ranking.
    assert tally(pred, m, None, threshold=0.90)['top1'] == 1.0


def test_restricting_and_renormalising_is_what_a_narrowed_app_would_render():
    """The configuration in question: mask down to the catalogue's species,
    give back the removed mass, then apply the threshold. Without
    renormalising, the same answer looks less certain than it is."""
    m = model_of(['known', 'also-known', 'outside-catalogue'])
    pred = [('known', output(m, {'known': 0.45, 'also-known': 0.15, 'outside-catalogue': 0.4}))]
    catalogue = {'known', 'also-known'}
    assert tally(pred, m, catalogue, threshold=0.70)['accepted_rate'] == 0.0
    renormalised = tally(pred, m, catalogue, renormalise=True, threshold=0.70)
    assert renormalised['accepted_rate'] == 1.0          # 0.45 / 0.60 = 0.75
    assert renormalised['precision_when_accepted'] == 1.0
