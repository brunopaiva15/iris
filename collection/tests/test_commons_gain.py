"""The report that decides whether Commons is worth a collection pass.

Measuring elsewhere showed what not measuring costs: Smithsonian Gardens
offered 4,884 CC0 photographs of cultivated plants, exactly the right visual
domain, and seven of them concerned our species. A source is not worth its
size but its overlap.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from commons_gain import summarise  # noqa: E402


def test_a_species_without_a_category_is_reported_apart():
    """Zero is not "few": it is "Commons does not know this plant"."""
    r = summarise({'Monstera deliciosa': 40, 'Rara avis': 0})
    assert r['unknown'] == ['Rara avis']
    assert r['known'] == 1


def test_the_thin_ones_are_counted_as_such():
    r = summarise({'A a': 12, 'B b': 250}, thin_threshold=100)
    assert r['thin'] == ['A a']
    assert r['total'] == 262


def test_a_species_commons_could_double_stands_out():
    """That is the population being looked for: few images here, many there."""
    r = summarise({'Hoya kerrii': 150, 'Monstera deliciosa': 20},
                  already={'Hoya kerrii': 40, 'Monstera deliciosa': 400})
    assert r['doubled'] == ['Hoya kerrii']


def test_without_the_set_no_species_is_said_to_be_doubled():
    """Not knowing what we already have, we do not pretend to know what we
    gain."""
    r = summarise({'Hoya kerrii': 150})
    assert r['doubled'] == []


def test_the_median_ignores_the_unknown_species():
    """Otherwise a handful of absences would drag the median to zero and
    condemn an otherwise rich source."""
    r = summarise({'A a': 100, 'B b': 200, 'C c': 0, 'D d': 0})
    assert r['median'] == 150
    assert r['species'] == 4 and r['known'] == 2


def test_an_empty_report_does_not_break():
    r = summarise({})
    assert r['median'] == 0 and r['total'] == 0 and r['known'] == 0
