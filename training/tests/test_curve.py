"""The exposed sets of the curve.

Two points of the curve only compare if they are **nested**: a larger size
must contain exactly the previous one, plus extra species. Otherwise the
measured gap mixes what is added with what is removed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from curve import subsets


EVERYTHING = ['a', 'b', 'c', 'd', 'e']


def test_the_core_is_always_there_whatever_the_size():
    sets = subsets(['a', 'b'], ['e', 'd'], EVERYTHING, [2, 3, 5])
    for size, kept in sets.items():
        assert {'a', 'b'} <= set(kept), size


def test_sets_are_nested_so_two_points_compare():
    sets = subsets(['a'], ['e', 'd', 'c'], EVERYTHING, [1, 2, 3, 5])
    by_size = [set(sets[n]) for n in sorted(sets)]
    for smaller, larger in zip(by_size, by_size[1:]):
        assert smaller < larger


def test_additions_follow_the_priority_order_not_the_model_order():
    """`candidates_v8.txt` is sorted by what people grow: that order decides
    the next species added, not the alphabet."""
    sets = subsets(['a'], ['e', 'd'], EVERYTHING, [2, 3])
    assert set(sets[2]) == {'a', 'e'}
    assert set(sets[3]) == {'a', 'e', 'd'}


def test_species_absent_from_the_priority_file_still_fill_the_tail():
    """The order need not cover everything: the rest follows, or a large size
    would be impossible to reach."""
    sets = subsets(['a'], ['e'], EVERYTHING, [5])
    assert set(sets[5]) == set(EVERYTHING)


def test_a_size_below_the_core_is_raised_to_it():
    """What is added is measured, not what is amputated: asking for less than
    the core returns the core, under its real name."""
    sets = subsets(['a', 'b', 'c'], [], EVERYTHING, [1])
    assert list(sets) == [3]
    assert set(sets[3]) == {'a', 'b', 'c'}


def test_a_core_species_the_model_never_learned_is_ignored():
    """A species with no column behind it cannot be part of an exposed set."""
    sets = subsets(['a', 'unknown'], ['e'], EVERYTHING, [2])
    assert set(sets[2]) == {'a', 'e'}


def test_the_priority_order_never_duplicates_the_core():
    sets = subsets(['a', 'b'], ['b', 'a', 'e'], EVERYTHING, [3])
    assert sorted(sets[3]) == ['a', 'b', 'e']
    assert len(sets[3]) == len(set(sets[3]))


# ── the theft cost ────────────────────────────────────────────────────────

import numpy as np

from curve import order_by_theft, theft_costs


def test_a_candidate_steals_only_images_the_core_gets_right():
    """Three classes: 0 and 1 in the core, 2 the candidate. The candidate
    steals B, where it exceeds the truth; not A, where it stays behind; and C
    does not count, the core was already wrong there."""
    P = np.array([[0.6, 0.1, 0.3],      # A: truth 0, right, candidate behind
                  [0.2, 0.35, 0.45],    # B: truth 1, right, candidate ahead → theft
                  [0.2, 0.4, 0.4]])     # C: truth 0, the core answers 1 → already wrong
    thefts, right = theft_costs(P, np.array([0, 1, 0]), np.array([0, 1]), np.array([2]))
    assert list(thefts) == [1]
    assert right == 2


def test_stealing_requires_strictly_more_than_the_truth():
    P = np.array([[0.5, 0.0, 0.5]])
    thefts, _ = theft_costs(P, np.array([0]), np.array([0, 1]), np.array([2]))
    assert list(thefts) == [0]


def test_costs_are_counted_per_candidate():
    P = np.array([[0.4, 0.1, 0.5, 0.0],
                  [0.1, 0.4, 0.5, 0.45]])
    thefts, _ = theft_costs(P, np.array([0, 1]), np.array([0, 1]), np.array([2, 3]))
    assert list(thefts) == [2, 1]


def test_free_candidates_come_first_then_priority_breaks_ties():
    """Two free ones and one expensive: the free ones first, in cultivation
    order — and the expensive one last even if it is the most grown."""
    costs = {'expensive': 3, 'b': 0, 'c': 0}
    assert order_by_theft(costs, ['expensive', 'c', 'b']) == ['c', 'b', 'expensive']


def test_a_candidate_absent_from_the_priority_file_goes_after_its_peers():
    costs = {'known': 0, 'unknown': 0}
    assert order_by_theft(costs, ['known']) == ['known', 'unknown']
