"""Sorting the candidate species before committing to a collection run.

Growing the catalogue costs top-1 when the added species have no images: they
enter the catalogue, leave the model through `--min-train`, and only the cost
remains. This sorting separates the four possible fates, because they call for
four different decisions.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from availability import THRESHOLD, verdict  # noqa: E402


def test_the_four_populations_are_separated():
    r = verdict({'Solid sp': 400, 'Thin sp': 10, 'Empty sp': 0, 'Unknown sp': -1}, threshold=25)
    assert r['solid'] == ['Solid sp']
    assert r['thin'] == ['Thin sp']
    assert r['empty'] == ['Empty sp']
    assert r['unknown'] == ['Unknown sp']


def test_an_unresolved_name_is_not_a_species_without_photographs():
    """-1 means "GBIF does not know this name": `synonyms.txt` answers that,
    not a collection run. Zero means "known, never photographed", and there it
    is the collection that would run for nothing."""
    r = verdict({'Citrus limon': -1, 'Rara avis': 0})
    assert r['unknown'] == ['Citrus limon'] and r['empty'] == ['Rara avis']
    assert r['thin'] == [] and r['solid'] == []


def test_the_threshold_is_the_training_one():
    """A species exactly at the threshold is kept: `--min-train` compares with
    greater-or-equal."""
    r = verdict({'Right on sp': THRESHOLD, 'Just below sp': THRESHOLD - 1})
    assert r['solid'] == ['Right on sp']
    assert r['thin'] == ['Just below sp']


def test_only_the_solid_ones_count_in_the_median():
    """Otherwise a tail of empty species would drag the median to zero and
    condemn an otherwise healthy catalogue."""
    r = verdict({'A a': 100, 'B b': 300, 'C c': 0, 'D d': 0, 'E e': -1})
    assert r['median'] == 200
    assert r['requested'] == 5


def test_the_thin_ones_come_out_closest_to_the_threshold_first():
    """That is where one more collection pass can tip the balance."""
    r = verdict({'Far sp': 2, 'Close sp': 24, 'Middle sp': 12}, threshold=25)
    assert r['thin'] == ['Close sp', 'Middle sp', 'Far sp']


def test_an_empty_catalogue_does_not_break():
    r = verdict({})
    assert r['requested'] == 0 and r['median'] == 0 and r['solid'] == []
