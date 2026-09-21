"""Answering at genus level when the species hesitates.

Two things would go wrong in silence: a genus mass summed incorrectly — the
number would be false without anything crashing — and a genus taking the place
of a species the model could name, which would be a loss dressed up as a gain.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genus import answer, genera_of, genus_table, measure

LABELS = ['picea-abies', 'picea-glauca', 'picea-pungens', 'monstera-deliciosa']


def output(scores: dict[str, float]) -> np.ndarray:
    return np.array([scores.get(c, 0.0) for c in LABELS], dtype=np.float32)


def test_the_genus_of_a_class_comes_from_its_scientific_name_when_known():
    """The identifier is usually enough, but not for a hybrid:
    `citrus-x-limon` would give the genus "citrus" from the identifier and
    "citrus" from the name — the name has the last word."""
    assert genera_of(LABELS) == ['picea', 'picea', 'picea', 'monstera']
    assert genera_of(['citrus-x-limon'], {'citrus-x-limon': 'Citrus × limon'}) == ['citrus']


def test_the_mass_of_a_genus_is_the_sum_of_its_species():
    """The heart of the idea: five Picea at 0.15 weigh 0.75. Summing into the
    wrong column would return a false number without anything crashing."""
    distinct, M = genus_table(genera_of(LABELS))
    assert distinct == ['monstera', 'picea']
    masses = output({'picea-abies': 0.3, 'picea-glauca': 0.25, 'picea-pungens': 0.2, 'monstera-deliciosa': 0.25}) @ M
    assert np.allclose(masses, [0.25, 0.75])


def test_a_species_that_passes_the_threshold_answers_alone():
    """The species keeps priority: a species name beats a genus, even when the
    genus is more certain still."""
    distinct, M = genus_table(genera_of(LABELS))
    probs = output({'picea-abies': 0.75, 'picea-glauca': 0.2, 'monstera-deliciosa': 0.05})
    what, which, score = answer(probs, probs @ M, 0.70)
    assert what == 'species' and LABELS[which] == 'picea-abies' and score == 0.75


def test_the_genus_answers_only_where_the_species_gave_up():
    distinct, M = genus_table(genera_of(LABELS))
    probs = output({'picea-abies': 0.3, 'picea-glauca': 0.25, 'picea-pungens': 0.2, 'monstera-deliciosa': 0.25})
    what, which, score = answer(probs, probs @ M, 0.70)
    assert what == 'genus' and distinct[which] == 'picea'
    assert score == 0.75


def test_neither_passes_and_nothing_is_answered():
    """The remote fallback keeps its reason to exist: the genus does not save
    everything."""
    distinct, M = genus_table(genera_of(LABELS))
    probs = output({'monstera-deliciosa': 0.35, 'picea-abies': 0.25, 'picea-glauca': 0.2, 'picea-pungens': 0.2})
    assert answer(probs, probs @ M, 0.70)[0] == 'nothing'


def test_a_lone_species_in_its_genus_gains_nothing():
    """Its mass *is* its score: it cannot pass at genus level what it failed to
    pass at species level."""
    distinct, M = genus_table(genera_of(LABELS))
    probs = output({'monstera-deliciosa': 0.6, 'picea-abies': 0.4})
    assert answer(probs, probs @ M, 0.70)[0] == 'nothing'


def test_the_tally_never_counts_a_genus_answer_as_a_species_one():
    distinct, M = genus_table(genera_of(LABELS))
    P = np.stack([
        output({'picea-abies': 0.9, 'picea-glauca': 0.1}),                                  # species, right
        output({'picea-abies': 0.3, 'picea-glauca': 0.25, 'picea-pungens': 0.25, 'monstera-deliciosa': 0.2}),  # genus, right
        output({'monstera-deliciosa': 0.4, 'picea-abies': 0.35, 'picea-glauca': 0.25}),     # nothing
    ])
    genus_of = np.array([1, 1, 1, 0])
    r = measure(P, np.array([0, 2, 3]), M, genus_of, 0.70)
    assert (r['species_rate'], r['species_precision']) == (1 / 3, 1.0)
    assert (r['genus_rate'], r['genus_precision']) == (1 / 3, 1.0)
    assert r['autonomy'] == 2 / 3 and r['precision'] == 1.0


def test_a_wrong_genus_answer_counts_against_precision():
    distinct, M = genus_table(genera_of(LABELS))
    probs = output({'picea-abies': 0.3, 'picea-glauca': 0.25, 'picea-pungens': 0.25, 'monstera-deliciosa': 0.2})
    r = measure(np.stack([probs]), np.array([3]), M, np.array([1, 1, 1, 0]), 0.70)
    assert r['genus_rate'] == 1.0 and r['genus_precision'] == 0.0


def test_truncating_to_the_top_k_is_what_the_cascade_sees():
    """The application sorts and cuts at five: a genus spread over twenty
    species at 0.04 escapes it entirely, and that is what `--top` quantifies."""
    from genus import truncate
    P = np.array([[0.4, 0.3, 0.2, 0.1]], dtype=np.float32)
    assert np.allclose(truncate(P, 2), [[0.4, 0.3, 0.0, 0.0]])
    assert np.allclose(truncate(P, 0), P)
    assert np.allclose(truncate(P, 9), P)


def test_a_genus_spread_too_thin_is_lost_to_a_truncated_view():
    distinct, M = genus_table(genera_of(LABELS))
    probs = output({'picea-abies': 0.3, 'picea-glauca': 0.25, 'picea-pungens': 0.25, 'monstera-deliciosa': 0.2})
    genus_of = np.array([1, 1, 1, 0])
    whole = measure(np.stack([probs]), np.array([0]), M, genus_of, 0.70)
    cut = measure(np.stack([probs]), np.array([0]), M, genus_of, 0.70, top=2)
    assert whole['genus_rate'] == 1.0        # 0.80: the genus answers
    assert cut['genus_rate'] == 0.0          # 0.55 over two classes: lost
